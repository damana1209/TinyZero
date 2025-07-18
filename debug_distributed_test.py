#!/usr/bin/env python3
"""
Distributed-specific debugging for CUDA illegal memory access
Run with: torchrun --nproc_per_node=2 debug_distributed_test.py
"""
import os
import sys
import torch
import torch.distributed as dist
import logging
from debug_memory_checks import validate_tensor_bounds, safe_all_reduce, MemoryDebugContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed environment"""
    try:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        logger.info(f"Rank {dist.get_rank()}/{dist.get_world_size()} initialized on GPU {local_rank}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize distributed: {e}")
        return False

def test_distributed_allreduce():
    """Test all_reduce operations with various tensor sizes"""
    logger.info("=== Testing Distributed All-Reduce ===")
    
    # Test the exact size mentioned in error log
    test_sizes = [1, 100, 1000, 98304, 1000000]
    
    for size in test_sizes:
        try:
            with MemoryDebugContext(f"allreduce_size_{size}"):
                tensor = torch.randn(size, device='cuda', dtype=torch.float32)
                original_sum = tensor.sum().item()
                
                # Test with validation
                safe_all_reduce(tensor, name=f"test_tensor_{size}")
                
                # Verify result makes sense
                expected_sum = original_sum * dist.get_world_size()
                actual_sum = tensor.sum().item()
                
                if abs(actual_sum - expected_sum) < 1e-6:
                    logger.info(f"✓ All-reduce passed for size {size}")
                else:
                    logger.error(f"✗ All-reduce result incorrect for size {size}: {actual_sum} vs {expected_sum}")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ All-reduce failed for size {size}: {e}")
            return False
    
    return True

def test_rmpad_distributed():
    """Test RMPAD operations in distributed context"""
    logger.info("=== Testing RMPAD in Distributed Context ===")
    
    try:
        import importlib
        flash_attn_module = importlib.import_module('flash_attn.bert_padding')
        unpad_input = flash_attn_module.unpad_input
        pad_input = flash_attn_module.pad_input
    except ImportError:
        logger.info("flash_attn not available, skipping RMPAD distributed tests")
        return True
    
    try:
        with MemoryDebugContext("rmpad_distributed"):
            batch_size, seq_len = 8, 512  # Larger sizes
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
            attention_mask = torch.ones_like(input_ids)
            
            # Add random padding patterns (more realistic)
            for i in range(batch_size):
                pad_start = torch.randint(seq_len//2, seq_len, (1,)).item()
                attention_mask[i, pad_start:] = 0
            
            # Test unpadding
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            
            # Test distributed operation on unpadded data
            if input_ids_rmpad.numel() > 0:
                test_tensor = torch.randn(input_ids_rmpad.shape[0], device='cuda')
                safe_all_reduce(test_tensor, name="rmpad_tensor")
            
            # Test padding back
            reconstructed = pad_input(input_ids_rmpad, indices, batch_size, seq_len)
            
            logger.info("✓ RMPAD distributed operations passed")
            return True
            
    except Exception as e:
        logger.error(f"✗ RMPAD distributed operations failed: {e}")
        return False

def test_ulysses_distributed():
    """Test Ulysses sequence parallelism in distributed context"""
    logger.info("=== Testing Ulysses in Distributed Context ===")
    
    try:
        import importlib
        ulysses_module = importlib.import_module('verl.utils.ulysses')
        slice_input_tensor = ulysses_module.slice_input_tensor
        all_to_all_tensor = ulysses_module.all_to_all_tensor
    except ImportError:
        logger.info("verl.utils.ulysses not available, skipping Ulysses tests")
        return True
    
    try:
        # Initialize Ulysses sequence parallel
        world_size = dist.get_world_size()
        if world_size < 2:
            logger.info("Need at least 2 GPUs for Ulysses test, skipping")
            return True
            
        with MemoryDebugContext("ulysses_distributed"):
            # Test basic tensor slicing (simpler test)
            batch_size, seq_len, hidden_size = 4, 128, 512
            tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
            
            # Test slice operation
            sliced = slice_input_tensor(tensor, dim=1, padding=True)
            
            logger.info("✓ Ulysses distributed operations passed")
            return True
            
    except Exception as e:
        logger.error(f"✗ Ulysses distributed operations failed: {e}")
        return False

def test_memory_intensive_operations():
    """Test memory-intensive operations that might trigger issues"""
    logger.info("=== Testing Memory-Intensive Operations ===")
    
    try:
        with MemoryDebugContext("memory_intensive"):
            # Test large tensor operations
            large_tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000, device='cuda')
                large_tensors.append(tensor)
            
            # Test distributed operations on large tensors
            for i, tensor in enumerate(large_tensors):
                safe_all_reduce(tensor, name=f"large_tensor_{i}")
            
            # Test memory allocation/deallocation patterns
            for _ in range(10):
                temp_tensor = torch.randn(5000, 5000, device='cuda')
                _ = temp_tensor.sum()
                del temp_tensor
                torch.cuda.empty_cache()
            
            logger.info("✓ Memory-intensive operations passed")
            return True
            
    except Exception as e:
        logger.error(f"✗ Memory-intensive operations failed: {e}")
        return False

def test_gradient_synchronization():
    """Test gradient synchronization patterns"""
    logger.info("=== Testing Gradient Synchronization ===")
    
    try:
        with MemoryDebugContext("grad_sync"):
            # Create a simple model
            model = torch.nn.Linear(1000, 1000).cuda()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            # Forward/backward pass
            input_data = torch.randn(32, 1000, device='cuda')
            target = torch.randn(32, 1000, device='cuda')
            
            optimizer.zero_grad()
            output = model(input_data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            
            # Manually synchronize gradients (simulating FSDP behavior)
            for param in model.parameters():
                if param.grad is not None:
                    safe_all_reduce(param.grad, name="gradient")
            
            optimizer.step()
            
            logger.info("✓ Gradient synchronization passed")
            return True
            
    except Exception as e:
        logger.error(f"✗ Gradient synchronization failed: {e}")
        return False

def main():
    """Run distributed debugging tests"""
    logger.info("Starting distributed debugging tests...")
    
    # Enable CUDA debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    if not setup_distributed():
        sys.exit(1)
    
    tests = [
        test_distributed_allreduce,
        test_rmpad_distributed,
        test_memory_intensive_operations,
        test_gradient_synchronization,
        test_ulysses_distributed,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            failed += 1
    
    # Synchronize results across ranks
    results_tensor = torch.tensor([passed, failed], device='cuda', dtype=torch.int64)
    dist.all_reduce(results_tensor)
    
    if dist.get_rank() == 0:
        total_passed = results_tensor[0].item()
        total_failed = results_tensor[1].item()
        logger.info(f"=== Final Results: {total_passed} passed, {total_failed} failed ===")
        
        if total_failed > 0:
            logger.error("Some distributed tests failed!")
        else:
            logger.info("All distributed tests passed!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main() 