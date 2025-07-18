#!/usr/bin/env python3
"""
Systematic debugging script for CUDA illegal memory access
"""
import os
import sys
import torch
import torch.distributed as dist
import logging
from debug_memory_checks import validate_tensor_bounds, safe_all_reduce, validate_rmpad_tensors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_operations():
    """Test basic tensor operations that might cause issues"""
    logger.info("=== Testing Basic Operations ===")
    
    # Test 1: Basic tensor creation and access
    try:
        x = torch.randn(1000, 1000, device='cuda')
        y = x.sum()
        logger.info("✓ Basic tensor operations passed")
    except Exception as e:
        logger.error(f"✗ Basic tensor operations failed: {e}")
        return False
    
    # Test 2: Memory boundary access
    try:
        large_tensor = torch.randn(10000, 10000, device='cuda')
        _ = large_tensor[9999, 9999]  # Access last element
        logger.info("✓ Memory boundary access passed")
    except Exception as e:
        logger.error(f"✗ Memory boundary access failed: {e}")
        return False
    
    return True

def test_rmpad_operations():
    """Test remove padding operations"""
    logger.info("=== Testing RMPAD Operations ===")
    
    try:
        import importlib
        flash_attn_module = importlib.import_module('flash_attn.bert_padding')
        unpad_input = flash_attn_module.unpad_input
        pad_input = flash_attn_module.pad_input
    except ImportError:
        logger.info("flash_attn not available, skipping RMPAD tests")
        return True
    
    try:
        
        batch_size, seq_len = 4, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device='cuda')
        attention_mask = torch.ones_like(input_ids)
        
        # Add some padding
        attention_mask[:, -20:] = 0
        
        # Test unpadding
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        
        if not validate_rmpad_tensors(input_ids, attention_mask, indices, "test_rmpad"):
            return False
        
        # Test padding back
        reconstructed = pad_input(input_ids_rmpad, indices, batch_size, seq_len)
        
        logger.info("✓ RMPAD operations passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ RMPAD operations failed: {e}")
        return False

def test_distributed_operations():
    """Test distributed operations"""
    logger.info("=== Testing Distributed Operations ===")
    
    if not dist.is_initialized():
        logger.info("Distributed not initialized, skipping")
        return True
    
    try:
        # Test all_reduce with various tensor sizes
        test_sizes = [1, 100, 1000, 98304]  # 98304 was in the error log
        
        for size in test_sizes:
            tensor = torch.randn(size, device='cuda')
            original_sum = tensor.sum().item()
            
            safe_all_reduce(tensor, name=f"test_tensor_{size}")
            
            logger.info(f"✓ All_reduce test passed for size {size}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Distributed operations failed: {e}")
        return False

def test_memory_buffers():
    """Test custom memory buffer operations"""
    logger.info("=== Testing Memory Buffers ===")
    
    try:
        from verl.utils.memory_buffer import MemoryBuffer
        
        # Test memory buffer creation and access
        buffer = MemoryBuffer(numel=1000, numel_padded=1024, dtype=torch.float32)
        
        # Test getting tensors from buffer
        tensor1 = buffer.get(shape=torch.Size([10, 10]), start_index=0)
        tensor2 = buffer.get(shape=torch.Size([20, 20]), start_index=100)
        
        # Test boundary conditions
        try:
            # This should fail safely
            bad_tensor = buffer.get(shape=torch.Size([100, 100]), start_index=900)
            logger.error("✗ Memory buffer boundary check failed")
            return False
        except AssertionError:
            logger.info("✓ Memory buffer boundary check passed")
        
        logger.info("✓ Memory buffer operations passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory buffer operations failed: {e}")
        return False

def test_ulysses_operations():
    """Test Ulysses sequence parallelism operations"""
    logger.info("=== Testing Ulysses Operations ===")
    
    try:
        from verl.utils.ulysses import slice_input_tensor, _pad_tensor, _unpad_tensor
        
        # Test tensor slicing
        tensor = torch.randn(4, 128, 512, device='cuda')
        sliced = slice_input_tensor(tensor, dim=1, padding=True)
        
        # Test padding operations
        padded = _pad_tensor(tensor, dim=1, padding_size=8)
        unpadded = _unpad_tensor(padded, dim=1, padding_size=8)
        
        logger.info("✓ Ulysses operations passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Ulysses operations failed: {e}")
        return False

def main():
    """Run systematic debugging tests"""
    logger.info("Starting systematic debugging tests...")
    
    # Enable CUDA debugging
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    tests = [
        test_basic_operations,
        test_memory_buffers,
        test_rmpad_operations,
        test_distributed_operations,
        test_ulysses_operations,
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
    
    logger.info(f"=== Results: {passed} passed, {failed} failed ===")
    
    if failed > 0:
        logger.error("Some tests failed. Check the logs above for details.")
        sys.exit(1)
    else:
        logger.info("All tests passed!")

if __name__ == "__main__":
    main() 