#!/usr/bin/env python3
"""
Memory validation utilities for debugging CUDA illegal memory access
"""
import torch
import logging

logger = logging.getLogger(__name__)

def validate_tensor_bounds(tensor, name="tensor"):
    """Check if tensor has valid memory layout and values"""
    try:
        if not isinstance(tensor, torch.Tensor):
            return True
            
        # Check for NaN/Inf
        if torch.isnan(tensor).any():
            logger.error(f"{name} contains NaN values")
            return False
            
        if torch.isinf(tensor).any():
            logger.error(f"{name} contains Inf values")
            return False
            
        # Force memory access to check bounds
        _ = tensor.sum()
        
        # Check memory layout
        if not tensor.is_contiguous():
            logger.warning(f"{name} is not contiguous")
            
        return True
    except Exception as e:
        logger.error(f"Memory validation failed for {name}: {e}")
        return False

def safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, name="tensor"):
    """Wrapper for all_reduce with validation"""
    if not validate_tensor_bounds(tensor, name):
        raise RuntimeError(f"Tensor {name} failed validation before all_reduce")
    
    # Synchronize before operation
    torch.cuda.synchronize()
    
    try:
        torch.distributed.all_reduce(tensor, op=op, group=group)
        torch.cuda.synchronize()
        
        if not validate_tensor_bounds(tensor, f"{name}_after_allreduce"):
            raise RuntimeError(f"Tensor {name} corrupted after all_reduce")
            
    except Exception as e:
        logger.error(f"All_reduce failed for {name}: {e}")
        raise

def validate_rmpad_tensors(input_ids, attention_mask, indices, name="rmpad"):
    """Validate remove padding operations"""
    try:
        # Check input shapes
        batch_size, seq_len = input_ids.shape
        assert attention_mask.shape == input_ids.shape
        
        # Check indices bounds
        max_index = batch_size * seq_len - 1
        if indices.max() > max_index:
            logger.error(f"{name}: indices out of bounds. Max: {indices.max()}, Expected: {max_index}")
            return False
            
        # Check total non-zero elements
        total_nnz = attention_mask.sum().item()
        if len(indices) != total_nnz:
            logger.error(f"{name}: indices length mismatch. Got: {len(indices)}, Expected: {total_nnz}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"RMPAD validation failed for {name}: {e}")
        return False

def debug_tensor_info(tensor, name="tensor"):
    """Print detailed tensor information"""
    if not isinstance(tensor, torch.Tensor):
        logger.info(f"{name}: Not a tensor")
        return
        
    logger.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    logger.info(f"{name}: contiguous={tensor.is_contiguous()}, requires_grad={tensor.requires_grad}")
    
    try:
        logger.info(f"{name}: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")
        logger.info(f"{name}: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
    except:
        logger.warning(f"{name}: Could not compute statistics")

# Context manager for safer operations
class MemoryDebugContext:
    def __init__(self, operation_name):
        self.operation_name = operation_name
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_memory = torch.cuda.memory_allocated()
        logger.info(f"Starting {self.operation_name}, memory: {self.start_memory / 1e9:.2f}GB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated()
        logger.info(f"Finished {self.operation_name}, memory: {end_memory / 1e9:.2f}GB, delta: {(end_memory - self.start_memory) / 1e9:.2f}GB")
        
        if exc_type is not None:
            logger.error(f"Exception in {self.operation_name}: {exc_val}") 