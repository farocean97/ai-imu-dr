#!/usr/bin/env python3
import argparse
import os
import sys

try:
    import torch
    from train_torch_filter import TORCHIEKF
except ImportError:
    print("PyTorch is not installed. Please install it using:")
    print("pip install torch")
    sys.exit(1)

def test_gpu_support():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        
        # Create a simple model and move it to GPU
        model = TORCHIEKF()
        model.to(torch.device('cuda'))
        print("Model moved to:", next(model.parameters()).device)
        
        # Create some test tensors
        cpu_tensor = torch.randn(3, 3).double()
        gpu_tensor = cpu_tensor.to(torch.device('cuda'))
        
        print("CPU tensor device:", cpu_tensor.device)
        print("GPU tensor device:", gpu_tensor.device)
        
        # Test computation
        result = model.normalize_rot(gpu_tensor)
        print("Computation result device:", result.device)
    else:
        print("CUDA is not available. Running on CPU only.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPU support for AI-IMU Dead-Reckoning")
    args = parser.parse_args()
    
    test_gpu_support()