#!/usr/bin/env python3
"""
Benchmark script for Multi-Person ST-GCN model
Compares PyTorch vs TensorRT performance and accuracy
"""

import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from pathlib import Path

# Import ST-GCN model (original non-modified version for PyTorch)
import sys
sys.path.append('/home/wisevision/projects/shoplifting-deploy/wisevision-shoplifting-jetson')
from net.st_gcn_twostream import Model as TwoStreamModel

# Configuration
PYTORCH_MODEL_PATH = "/home/wisevision/projects/stgcn-tensorrt-optimization/models/stgcn_unified.pt"
TRT_ENGINE_PATH = "/home/wisevision/projects/stgcn-tensorrt-optimization/tensorrt_engines/stgcn_M2_unified_fp16.engine"

# Model parameters for M=2 (dual-person)
IN_CHANNELS = 3
NUM_CLASS = 2
TEMPORAL_FRAMES = 150
VERTICES = 33
MAX_PERSONS = 2
LAYOUT = 'mediapipe'

# Benchmark settings
NUM_WARMUP = 10
NUM_ITERATIONS = 100


class TensorRTInference:
    """TensorRT inference wrapper"""
    
    def __init__(self, engine_path):
        """Initialize TensorRT engine"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        print(f"Loading TensorRT engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        
        # Get input/output information
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)
        
        print(f"  Input: {self.input_name} -> {self.input_shape}")
        print(f"  Output: {self.output_name} -> {self.output_shape}")
        
        # Allocate buffers
        self.input_nbytes = trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        self.output_nbytes = trt.volume(self.output_shape) * np.dtype(np.float32).itemsize
        
        self.d_input = cuda.mem_alloc(self.input_nbytes)
        self.d_output = cuda.mem_alloc(self.output_nbytes)
        
        self.stream = cuda.Stream()
        
        print("TensorRT engine loaded successfully!\n")
    
    def infer(self, input_data):
        """Run inference"""
        # Ensure input is float32 and contiguous
        input_data = np.ascontiguousarray(input_data.astype(np.float32))
        
        # Copy input to device
        cuda.memcpy_htod_async(self.d_input, input_data, self.stream)
        
        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        
        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy output to host
        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        
        # Synchronize
        self.stream.synchronize()
        
        return output
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'd_input'):
            self.d_input.free()
        if hasattr(self, 'd_output'):
            self.d_output.free()


def load_pytorch_model(model_path):
    """Load PyTorch ST-GCN model"""
    print(f"Loading PyTorch model: {model_path}")
    
    # Create model
    model = TwoStreamModel(
        in_channels=IN_CHANNELS,
        num_class=NUM_CLASS,
        graph_args={'layout': LAYOUT, 'strategy': 'spatial'},
        edge_importance_weighting=True,
        dropout=0.5
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cuda')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.cuda()
    model.eval()
    
    print("PyTorch model loaded successfully!\n")
    return model


def generate_random_input(batch_size=1):
    """Generate random input data for testing"""
    # Shape: (N, C, T, V, M)
    shape = (batch_size, IN_CHANNELS, TEMPORAL_FRAMES, VERTICES, MAX_PERSONS)
    data = np.random.randn(*shape).astype(np.float32)
    return data


def benchmark_pytorch(model, input_data, num_warmup=NUM_WARMUP, num_iterations=NUM_ITERATIONS):
    """Benchmark PyTorch model"""
    print("="*60)
    print("PYTORCH BENCHMARK")
    print("="*60)
    
    input_tensor = torch.from_numpy(input_data).cuda()
    
    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.time()
            output = model(input_tensor)
            # Apply softmax to match TensorRT output
            output = torch.nn.functional.softmax(output, dim=1)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
    
    times = np.array(times)
    output_np = output.cpu().numpy()
    
    print(f"\nResults:")
    print(f"  Mean time: {times.mean()*1000:.2f} ms")
    print(f"  Std time:  {times.std()*1000:.2f} ms")
    print(f"  Min time:  {times.min()*1000:.2f} ms")
    print(f"  Max time:  {times.max()*1000:.2f} ms")
    print(f"  FPS:       {1.0/times.mean():.2f}")
    print(f"  Output shape: {output_np.shape}")
    print(f"  Output (softmax): {output_np[0]}")
    print()
    
    return output_np, times


def benchmark_tensorrt(trt_model, input_data, num_warmup=NUM_WARMUP, num_iterations=NUM_ITERATIONS):
    """Benchmark TensorRT model"""
    print("="*60)
    print("TENSORRT BENCHMARK")
    print("="*60)
    
    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        _ = trt_model.infer(input_data)
    
    # Benchmark
    print(f"Running benchmark ({num_iterations} iterations)...")
    times = []
    
    for _ in range(num_iterations):
        start = time.time()
        output = trt_model.infer(input_data)
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    print(f"\nResults:")
    print(f"  Mean time: {times.mean()*1000:.2f} ms")
    print(f"  Std time:  {times.std()*1000:.2f} ms")
    print(f"  Min time:  {times.min()*1000:.2f} ms")
    print(f"  Max time:  {times.max()*1000:.2f} ms")
    print(f"  FPS:       {1.0/times.mean():.2f}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (softmax): {output[0]}")
    print()
    
    return output, times


def compare_accuracy(pytorch_output, trt_output):
    """Compare accuracy between PyTorch and TensorRT outputs"""
    print("="*60)
    print("ACCURACY COMPARISON")
    print("="*60)
    
    # Calculate differences
    abs_diff = np.abs(pytorch_output - trt_output)
    rel_diff = abs_diff / (np.abs(pytorch_output) + 1e-8)
    
    print(f"PyTorch output: {pytorch_output[0]}")
    print(f"TensorRT output: {trt_output[0]}")
    print(f"\nAbsolute difference: {abs_diff[0]}")
    print(f"Relative difference: {rel_diff[0]}")
    print(f"\nMax absolute difference: {abs_diff.max():.6f}")
    print(f"Mean absolute difference: {abs_diff.mean():.6f}")
    print(f"Max relative difference: {rel_diff.max():.6f}")
    print(f"Mean relative difference: {rel_diff.mean():.6f}")
    
    # Check if predictions match
    pytorch_pred = pytorch_output.argmax(axis=1)
    trt_pred = trt_output.argmax(axis=1)
    
    print(f"\nPyTorch prediction: Class {pytorch_pred[0]} (confidence: {pytorch_output[0, pytorch_pred[0]]:.4f})")
    print(f"TensorRT prediction: Class {trt_pred[0]} (confidence: {trt_output[0, trt_pred[0]]:.4f})")
    print(f"Predictions match: {pytorch_pred[0] == trt_pred[0]}")
    print()


def main():
    """Main benchmark function"""
    print("\n" + "="*60)
    print("ST-GCN DUAL-PERSON (M=2) BENCHMARK")
    print("="*60)
    print(f"Model Configuration:")
    print(f"  Input channels: {IN_CHANNELS}")
    print(f"  Output classes: {NUM_CLASS}")
    print(f"  Temporal frames: {TEMPORAL_FRAMES}")
    print(f"  Vertices: {VERTICES}")
    print(f"  Max persons: {MAX_PERSONS}")
    print(f"  Layout: {LAYOUT}")
    print()
    
    # Check if files exist
    if not Path(PYTORCH_MODEL_PATH).exists():
        print(f"ERROR: PyTorch model not found: {PYTORCH_MODEL_PATH}")
        return
    
    if not Path(TRT_ENGINE_PATH).exists():
        print(f"ERROR: TensorRT engine not found: {TRT_ENGINE_PATH}")
        return
    
    # Load models
    pytorch_model = load_pytorch_model(PYTORCH_MODEL_PATH)
    trt_model = TensorRTInference(TRT_ENGINE_PATH)
    
    # Generate test input
    print("Generating random test input...")
    input_data = generate_random_input(batch_size=1)
    print(f"Input shape: {input_data.shape}\n")
    
    # Benchmark PyTorch
    pytorch_output, pytorch_times = benchmark_pytorch(pytorch_model, input_data)
    
    # Benchmark TensorRT
    trt_output, trt_times = benchmark_tensorrt(trt_model, input_data)
    
    # Compare accuracy
    compare_accuracy(pytorch_output, trt_output)
    
    # Summary
    print("="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    pytorch_fps = 1.0 / pytorch_times.mean()
    trt_fps = 1.0 / trt_times.mean()
    speedup = pytorch_times.mean() / trt_times.mean()
    
    print(f"PyTorch:")
    print(f"  Average latency: {pytorch_times.mean()*1000:.2f} ms")
    print(f"  FPS: {pytorch_fps:.2f}")
    print()
    print(f"TensorRT:")
    print(f"  Average latency: {trt_times.mean()*1000:.2f} ms")
    print(f"  FPS: {trt_fps:.2f}")
    print()
    print(f"Speedup: {speedup:.2f}x")
    print(f"FPS improvement: {((trt_fps - pytorch_fps) / pytorch_fps * 100):.1f}%")
    print("="*60)
    print()


if __name__ == "__main__":
    main()
