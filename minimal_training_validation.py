#!/usr/bin/env python3
"""
Minimal validation for training - add this to your training script
"""
import torch
import torch.distributed as dist
import logging

logger = logging.getLogger(__name__)

def validate_tensor_safe(tensor, name="tensor"):
    """Lightweight tensor validation"""
    try:
        if not isinstance(tensor, torch.Tensor):
            return True
        # Quick validation - just access memory
        _ = tensor.sum()
        return True
    except Exception as e:
        logger.error(f"Tensor validation failed for {name}: {e}")
        return False

def safe_all_reduce_minimal(tensor, op=dist.ReduceOp.SUM, group=None, name="tensor"):
    """Minimal wrapper for all_reduce with basic validation"""
    if not validate_tensor_safe(tensor, name):
        raise RuntimeError(f"Tensor {name} failed validation before all_reduce")
    
    try:
        dist.all_reduce(tensor, op=op, group=group)
        return tensor
    except Exception as e:
        logger.error(f"All_reduce failed for {name}: {e}")
        raise

def monkey_patch_all_reduce():
    """Replace torch.distributed.all_reduce with validated version"""
    original_all_reduce = dist.all_reduce
    
    def validated_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if async_op:
            # Don't validate async operations
            return original_all_reduce(tensor, op=op, group=group, async_op=async_op)
        else:
            return safe_all_reduce_minimal(tensor, op=op, group=group, name="auto_validated")
    
    dist.all_reduce = validated_all_reduce
    logger.info("Applied minimal validation to torch.distributed.all_reduce")

def enable_cuda_debugging():
    """Enable CUDA debugging flags"""
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    logger.info("Enabled CUDA debugging flags")

def setup_memory_debugging():
    """Setup all memory debugging - call this at start of training"""
    enable_cuda_debugging()
    monkey_patch_all_reduce()
    logger.info("Memory debugging setup complete")

# Example usage:
# At the top of your training script, add:
# from minimal_training_validation import setup_memory_debugging
# setup_memory_debugging() 