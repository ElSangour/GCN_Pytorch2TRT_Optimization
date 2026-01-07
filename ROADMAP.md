# ST-GCN TensorRT Export Roadmap
## Complete Guide for Jetson ORIN NX Deployment

 
**Training Configuration:** 640x480 frames, SEQ_LEN=150, 25fps, MediaPipe (33 keypoints)  
**Models:** Two-Stream ST-GCN (single-person: M=1, multi-person: M=5)

---

## Table of Contents

1. [Prerequisites & Understanding](#1-prerequisites--understanding)
2. [CUDA Concepts Explained](#2-cuda-concepts-explained)
3. [Model Architecture Review](#3-model-architecture-review)
4. [Export Pipeline: PyTorch → ONNX → TensorRT](#4-export-pipeline-pytorch--onnx--tensorrt)
5. [TensorRT Integration into Pipeline](#5-tensorrt-integration-into-pipeline)
6. [Validation & Benchmarking](#6-validation--benchmarking)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Prerequisites & Understanding

### 1.1 Critical Training Details (MUST MATCH)

You should know your tained models' exact parameters:
In my case, I have two models:
```python
# Single-Person Model
Input Shape: (batch_size, 3, 150, 33, 1)
- batch_size: 1 (real-time inference)
- 3: Coordinate channels (x, y, visibility)
- 150: Temporal sequence length (SEQ_LEN) 
- 33: MediaPipe skeleton keypoints (V)
- 1: Single person (M)

# Multi-Person Model
Input Shape: (batch_size, 3, 150, 33, 5)
- Same as above, but M=5 (max 5 persons)

# Frame Resolution: 640x480
# Frame Rate: 25fps
# Keypoint Format: MediaPipe normalized coordinates [0, 1]
```

### 1.2 Two-Stream Architecture

Your models use **Two-Stream ST-GCN**:
- **Origin Stream**: Processes raw skeleton sequences
- **Motion Stream**: Processes temporal differences (motion vectors)
- **Final Output**: `origin_stream(x) + motion_stream(motion)`

```python
# Motion stream computation (in forward):
m = torch.cat((
    torch.zeros(N, C, 1, V, M),  # First frame: zero motion
    x[:, :, 1:] - x[:, :, :-1]   # Frame differences
), dim=2)  # Concatenate along temporal dimension
```

### 1.3 Software Requirements

```bash
# On Jetson ORIN NX
- JetPack 5.x (CUDA 11.4+)
- PyTorch 2.0+ (Jetson-optimized build)
- TensorRT 8.5+
- ONNX Runtime (optional, for validation)
- Python 3.8+

# Verify installations
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

---

## 2. CUDA Concepts Explained

### 2.1 CUDA Context

**What is it?**
- CUDA context = GPU state container (memory, kernels, streams)
- Each process/thread needs its own context to use GPU
- Multiple contexts can exist but switching is expensive

A **CUDA context** is an **isolated GPU environment**.

- Owns:
  - GPU memory
  - loaded kernels / modules
  - CUDA streams
- Contexts **cannot see or interact with each other**
- All contexts **share the same GPU hardware**

Execution behavior:
- Contexts may run **sequentially**, **interleaved**, or **partially in parallel**
- Parallel execution between contexts is **not guaranteed**
- Switching between contexts is **expensive**


**Main role:** isolation (not parallelism)

**In my pipeline:**
- PyTorch creates primary CUDA context automatically on first GPU call
- TensorRT reuses PyTorch's context (critical for Jetson servers) 
- In my case, the per-camera threads share GPU but have separate CUDA streams!

```python
# PyTorch creates context implicitly:
device = torch.device("cuda")
x = torch.randn(1, 3, 150, 33, 1).to(device)  # Context created here

# TensorRT must reuse this context:
# (PyTorch-first architecture in camera_pipeline2.py)
```
---
### 2.2 CUDA Streams

**What is it?**
- CUDA stream = queue of GPU operations (kernels, memory transfers)
- Operations in same stream execute sequentially
- Operations in different streams can execute concurrently

A **CUDA stream** is a **queue of GPU operations**.

- Operations in the **same stream run sequentially**
- Operations in **different streams (same context)** may run in parallel
- Stream switching is **cheap**

**Main role:** ordering and concurrency

---


**Why it matters:**
- **Parallel execution**: Multiple cameras can infer simultaneously
- **Non-blocking**: One camera's inference doesn't block others
- **Performance**: Better GPU utilization

**In my pipeline:**

```python
# Each camera has its own stream:
self.pytorch_stream = torch.cuda.Stream()  # Per-camera stream

# ST-GCN inference on this stream:
with torch.cuda.stream(self.pytorch_stream):
    output = model(input_data)  # Non-blocking for other cameras

# Explicit sync when needed:
self.pytorch_stream.synchronize()  # Wait for this camera's ops
```

**Visual Example:**

```
Camera 1 Stream: [YOLO] → [MediaPipe] → [ST-GCN] → [done]
Camera 2 Stream:           [YOLO] → [MediaPipe] → [ST-GCN]
Camera 3 Stream:                         [YOLO] → [MediaPipe] → [ST-GCN]

GPU Timeline:
Time →  [Cam1:YOLO][Cam2:YOLO][Cam1:STGCN][Cam3:YOLO][Cam2:STGCN][Cam3:STGCN]
        └─ Parallel execution ─────────────────────────┘
```

### 2.3 Context vs Stream Summary

| Concept | Purpose | Scope | Your Usage |
|---------|---------|-------|------------|
| **CUDA Context** | GPU state container | Per-process/thread | Shared (PyTorch creates, TensorRT reuses) |
| **CUDA Stream** | Operation queue | Per-thread/operation | Per-camera (parallel inference) |
| **GPU Memory** | Shared across contexts | Global | All cameras share GPU RAM |

**Best Practice for Jetson servers:**
1. PyTorch creates context FIRST (happens automatically)
2. TensorRT reuses it (no manual context management)
3. Each camera thread uses its own stream for parallelism


### 2.4 Relationship Between Contexts and Streams

- Each **context owns its own streams**
- Streams **do not cross context boundaries**
- Context switching → expensive  
- Stream switching → cheap

### 2.5 Clear Answers to Common Questions

#### Do contexts run on different threads?
- Contexts are **associated with host threads** for work submission
- They do **not execute on CPU threads**

#### Do contexts run in parallel?
- **Sometimes**, depending on hardware and scheduling
- Not guaranteed

#### Do streams run sequentially?
- **Yes**, within a stream
- **No**, across streams
---x
## 3. Model Architecture Review

### 3.1 ST-GCN Input/Output Flow

```
Input: (N, C, T, V, M) = (1, 3, 150, 33, 1)
  ↓
Normalize & Reshape: (N*M, C, T, V) = (1, 3, 150, 33)
  ↓
[ST-GCN Layer 1] → (1, 64, 150, 33)
[ST-GCN Layer 2] → (1, 64, 150, 33)
[ST-GCN Layer 3] → (1, 64, 150, 33)
[ST-GCN Layer 4] → (1, 64, 150, 33)
[ST-GCN Layer 5] → (1, 128, 75, 33)   ← stride=2 (T halves)
[ST-GCN Layer 6] → (1, 128, 75, 33)
[ST-GCN Layer 7] → (1, 128, 75, 33)
[ST-GCN Layer 8] → (1, 256, 38, 33)   ← stride=2 (T halves again)
[ST-GCN Layer 9] → (1, 256, 38, 33)
[ST-GCN Layer 10] → (1, 256, 38, 33)
  ↓
Global Pooling: (1, 256, 1, 1)
  ↓
FCN (Conv1x1): (1, 2, 1, 1)
  ↓
Output: (1, 2)  ← [logit_normal, logit_shoplifting]
```

### 3.2 Two-Stream Details

```python
# Forward pass:
def forward(self, x):  # x: (1, 3, 150, 33, 1)
    # Origin stream: raw skeleton
    origin_out = self.origin_stream(x)  # (1, 2)
    
    # Motion stream: temporal differences
    motion = torch.cat((
        torch.zeros(1, 3, 1, 33, 1),  # First frame
        x[:, :, 1:] - x[:, :, :-1]    # Differences
    ), dim=2)  # (1, 3, 150, 33, 1)
    motion_out = self.motion_stream(motion)  # (1, 2)
    
    # Fusion: element-wise addition
    return origin_out + motion_out  # (1, 2)
```

### 3.3 Key Export Considerations

**1. Static Shapes:**
- ONNX/TensorRT requires fixed dimensions (no dynamic axes for T, V, M)
- Your training uses fixed: T=150, V=33, M=1 or 5

**2. Graph Adjacency Matrix:**
- MediaPipe layout: 33 keypoints, predefined spatial connections
- Stored as buffer (not parameter) in model: `self.register_buffer('A', A)`

**3. Edge Importance:**
- Learnable weights per layer: `self.edge_importance`
- Applied as: `A * importance` in each ST-GCN layer

**4. Temporal Pooling:**
- Dynamic pooling: `F.avg_pool2d(x, kernel_size=(current_T, current_V))`
- **Issue**: ONNX needs static kernel size
- **Solution**: Use `AdaptiveAvgPool2d(1)` which converts to GlobalAveragePool

---

## 4. Export Pipeline: PyTorch → ONNX → TensorRT

### 4.1 Step 1: Correct ONNX Export (FIXED Dimensions)

Create: `GCN_Pytorch2TRT_Optimization/export_stgcn_onnx_correct.py`

```python
#!/usr/bin/env python3
"""
Correct ONNX export matching training dimensions:
- T=150 (not 300!)
- M=1 for single, M=5 for multi
- Two-Stream architecture preserved
"""

import torch
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from net.st_gcn_twostream import Model as STGCN_TwoStream

def export_single_person_model():
    """Export single-person ST-GCN model to ONNX."""
    print("=" * 60)
    print("EXPORTING SINGLE-PERSON MODEL")
    print("=" * 60)
    
    # Initialize model (EXACT architecture from training)
    model = STGCN_TwoStream(
        in_channels=3,
        num_class=2,
        graph_args={'layout': 'mediapipe', 'strategy': 'spatial'},
        edge_importance_weighting=True
    )
    
    # Load trained weights
    model_path = '../wisevision-shoplifting-jetson/work_dir/recognition/shoplifting_mg1/ST_GCN_TWO_STREAM/default/epoch60_model.pt'
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # CORRECT input shape: (N, C, T, V, M) = (1, 3, 150, 33, 1)
    C, T, V, M = 3, 150, 33, 1  
    dummy_input = torch.randn(1, C, T, V, M)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Model type: {type(model).__name__}")
    
    # Export to ONNX
    output_path = '../models/stgcn_single_correct.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=12,  # Required for einsum
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        dynamic_axes=None  # Static shapes only
    )
    
    print(f"[SUCCESS] Exported to {output_path}")
    
    # Verify output shape
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape} (expected: (1, 2))")
    
    return output_path

def export_multi_person_model():
    """Export multi-person ST-GCN model to ONNX."""
    print("\n" + "=" * 60)
    print("EXPORTING MULTI-PERSON MODEL")
    print("=" * 60)
    
    model = STGCN_TwoStream(
        in_channels=3,
        num_class=2,
        graph_args={'layout': 'mediapipe', 'strategy': 'spatial'},
        edge_importance_weighting=True
    )
    
    model_path = '../wisevision-shoplifting-jetson/work_dir/recognition/shoplifting_mg_yolo_media/ST_GCN_TWO_STREAM/default/epoch50_model.pt'
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # CORRECT input shape: (1, 3, 150, 33, 5)
    C, T, V, M = 3, 150, 33, 5  # ⚠️ M=5 for multi-person!
    dummy_input = torch.randn(1, C, T, V, M)
    
    print(f"Input shape: {dummy_input.shape}")
    
    output_path = '../models/stgcn_multi_correct.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=12,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        dynamic_axes=None
    )
    
    print(f"✅ Exported to {output_path}")
    
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape} (expected: (1, 2))")
    
    return output_path

if __name__ == "__main__":
    export_single_person_model()
    export_multi_person_model()
    print("\n" + "=" * 60)
    print("✅ ONNX Export Complete!")
    print("=" * 60)
```

### 4.2 Step 2: Validate ONNX Models

Create: `GCN_Pytorch2TRT_Optimization/validate_onnx.py`

```python
#!/usr/bin/env python3
"""Validate ONNX models match PyTorch outputs."""

import torch
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from net.st_gcn_twostream import Model as STGCN_TwoStream

def validate_model(onnx_path, pytorch_model_path, input_shape, model_type):
    """Validate ONNX model against PyTorch."""
    print(f"\n{'='*60}")
    print(f"Validating {model_type} Model")
    print(f"{'='*60}")
    
    # Load PyTorch model
    model = STGCN_TwoStream(
        in_channels=3, num_class=2,
        graph_args={'layout': 'mediapipe', 'strategy': 'spatial'},
        edge_importance_weighting=True
    )
    state_dict = torch.load(pytorch_model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create test input
    test_input = torch.randn(input_shape)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = model(test_input).numpy()
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    onnx_input = {session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = session.run(None, onnx_input)[0]
    
    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    
    if max_diff < 1e-4:
        print("✅ ONNX model matches PyTorch!")
        return True
    else:
        print("⚠️ Significant differences detected")
        return False

if __name__ == "__main__":
    # Validate single-person
    validate_model(
        '../models/stgcn_single_correct.onnx',
        '../wisevision-shoplifting-jetson/work_dir/recognition/shoplifting_mg1/ST_GCN_TWO_STREAM/default/epoch60_model.pt',
        (1, 3, 150, 33, 1),
        "Single-Person"
    )
    
    # Validate multi-person
    validate_model(
        '../models/stgcn_multi_correct.onnx',
        '../wisevision-shoplifting-jetson/work_dir/recognition/shoplifting_mg_yolo_media/ST_GCN_TWO_STREAM/default/epoch50_model.pt',
        (1, 3, 150, 33, 5),
        "Multi-Person"
    )
```

### 4.3 Step 3: Convert ONNX to TensorRT Engine

Create: `GCN_Pytorch2TRT_Optimization/onnx_to_tensorrt.py`

```python
#!/usr/bin/env python3
"""
Convert ONNX models to TensorRT engines on Jetson.
Supports FP16 and INT8 quantization.
"""

import tensorrt as trt
import numpy as np
import os
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, precision='fp16', workspace_size=4):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
        workspace_size: Workspace size in GB (default: 4GB)
    """
    print(f"\n{'='*60}")
    print(f"Building TensorRT Engine")
    print(f"{'='*60}")
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"Precision: {precision.upper()}")
    print(f"Workspace: {workspace_size}GB")
    
    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX model
    print("\n[1/5] Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("❌ Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    print("✅ ONNX parsed successfully")
    
    # Configure builder
    print(f"\n[2/5] Configuring builder...")
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size * (1 << 30)  # GB to bytes
    
    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✅ FP16 enabled")
        else:
            print("⚠️ FP16 not supported, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # TODO: Add calibration dataset for INT8
            print("⚠️ INT8 enabled (calibration needed)")
        else:
            print("⚠️ INT8 not supported, using FP32")
    
    # Build engine
    print(f"\n[3/5] Building TensorRT engine (this may take a while)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        print("❌ Engine build failed")
        return False
    
    # Save engine
    print(f"\n[4/5] Saving engine to {engine_path}...")
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    
    # Print engine info
    print(f"\n[5/5] Engine Information:")
    print(f"   Input bindings: {engine.num_bindings // 2}")
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        print(f"   [{i}] {name}: {shape} ({dtype})")
    
    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    print(f"   Engine size: {engine_size_mb:.2f} MB")
    
    print(f"\n✅ TensorRT engine built successfully!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT')
    parser.add_argument('--onnx', type=str, required=True, help='ONNX model path')
    parser.add_argument('--engine', type=str, required=True, help='Output engine path')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'],
                       default='fp16', help='Precision mode')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size (GB)')
    
    args = parser.parse_args()
    
    build_engine(args.onnx, args.engine, args.precision, args.workspace)
```

**Usage:**

```bash
# Convert single-person model
python3 onnx_to_tensorrt.py \
    --onnx ../models/stgcn_single_correct.onnx \
    --engine ../models/stgcn_single_fp16.engine \
    --precision fp16 \
    --workspace 4

# Convert multi-person model
python3 onnx_to_tensorrt.py \
    --onnx ../models/stgcn_multi_correct.onnx \
    --engine ../models/stgcn_multi_fp16.engine \
    --precision fp16 \
    --workspace 4
```

---

## 5. TensorRT Integration into Pipeline

### 5.1 Create TensorRT Inference Wrapper

Create: `real_time_detectionV2/processing/stgcn_tensorrt_inference.py`

```python
"""
TensorRT inference wrapper for ST-GCN models.
Replaces PyTorch ST-GCN inference with TensorRT engines.
"""

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torch
import logging

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class STGCNTensorRTEngine:
    """
    TensorRT engine wrapper for ST-GCN inference.
    Handles engine loading, memory allocation, and inference.
    """
    
    def __init__(self, engine_path, input_shape, output_shape):
        """
        Initialize TensorRT engine.
        
        Args:
            engine_path: Path to .engine file
            input_shape: Input tensor shape (C, T, V, M)
            output_shape: Output tensor shape (num_classes,)
        """
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # Load engine
        self.engine = self._load_engine()
        self.context = self.engine.create_execution_context()
        
        # Allocate GPU memory
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        logging.info(f"✅ TensorRT engine loaded: {engine_path}")
    
    def _load_engine(self):
        """Load TensorRT engine from file."""
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            return runtime.deserialize_cuda_engine(f.read())
    
    def _allocate_buffers(self):
        """Allocate GPU memory for inputs and outputs."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def infer(self, input_data):
        """
        Run inference on input data.
        
        Args:
            input_data: numpy array of shape (C, T, V, M) or (1, C, T, V, M)
        
        Returns:
            numpy array of shape (num_classes,)
        """
        # Ensure correct shape: (1, C, T, V, M)
        if input_data.ndim == 4:
            input_data = input_data[np.newaxis, ...]
        
        # Copy input to GPU
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )
        
        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # Copy output from GPU
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()
        
        # Reshape output
        output = self.outputs[0]['host'].reshape(self.output_shape)
        return output
    
    def __del__(self):
        """Cleanup GPU memory."""
        if hasattr(self, 'inputs'):
            for inp in self.inputs:
                if 'device' in inp:
                    inp['device'].free()
        if hasattr(self, 'outputs'):
            for out in self.outputs:
                if 'device' in out:
                    out['device'].free()


def load_stgcn_tensorrt_models(config):
    """
    Load TensorRT engines for single and multi-person models.
    
    Args:
        config: Configuration dictionary with engine paths
    
    Returns:
        tuple: (model_single, model_multi)
    """
    model_single = STGCNTensorRTEngine(
        engine_path=config['TRT_ENGINE_PATH_SINGLE'],
        input_shape=(3, 150, 33, 1),
        output_shape=(2,)
    )
    
    model_multi = STGCNTensorRTEngine(
        engine_path=config['TRT_ENGINE_PATH_MULTI'],
        input_shape=(3, 150, 33, 5),
        output_shape=(2,)
    )
    
    return model_single, model_multi
```

### 5.2 Update Prediction Function

Modify: `real_time_detectionV2/processing/prediction.py`

```python
# Add TensorRT support to predict_action function

def predict_action(buffer, model, C, V, M, SEQ_LEN, device, 
                   pytorch_stream=None, use_tensorrt=False):
    """
    Predict action using ST-GCN model (PyTorch or TensorRT).
    
    Args:
        use_tensorrt: If True, model is TensorRT engine, else PyTorch
    """
    buf_list = list(buffer)
    if len(buf_list) < SEQ_LEN:
        padded = buf_list + [np.zeros((C, V, M))] * (SEQ_LEN - len(buf_list))
    else:
        padded = buf_list[-SEQ_LEN:]
    
    # Stack to (C, T, V, M)
    data = np.stack(padded, axis=1)
    
    if use_tensorrt:
        # TensorRT inference (no batch dimension prepending needed)
        # Input shape: (C, T, V, M) = (3, 150, 33, M)
        output = model.infer(data)  # Returns (2,)
        probs = softmax(output)
        pred_class = np.argmax(probs)
        confidence = float(probs[pred_class])
        prob_list = probs.tolist()
    else:
        # PyTorch inference (existing code)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
        # ... rest of existing PyTorch code ...
    
    return pred_class, confidence, prob_list
```

---

## 6. Validation & Benchmarking

### 6.1 Performance Comparison Script

Create: `GCN_Pytorch2TRT_Optimization/benchmark_tensorrt.py`

```python
#!/usr/bin/env python3
"""
Benchmark TensorRT vs PyTorch ST-GCN inference.
"""

import time
import numpy as np
import torch
from processing.stgcn_tensorrt_inference import STGCNTensorRTEngine

def benchmark_pytorch(model, input_data, num_iterations=100, warmup=10):
    """Benchmark PyTorch model."""
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device))
            torch.cuda.synchronize()
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device))
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)

def benchmark_tensorrt(engine, input_data, num_iterations=100, warmup=10):
    """Benchmark TensorRT engine."""
    # Warmup
    for _ in range(warmup):
        _ = engine.infer(input_data)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = engine.infer(input_data)
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)

if __name__ == "__main__":
    # Load models and run benchmarks
    # ... implementation ...
```

---

## 7. Troubleshooting

### 7.1 Common Issues

**Issue: "Input shape mismatch"**
- **Cause**: Training used T=150, export used T=300
- **Fix**: Use correct dimensions in export script

**Issue: "CUDA context errors"**
- **Cause**: TensorRT trying to create its own context
- **Fix**: Ensure PyTorch creates context first (PyTorch-first architecture)

**Issue: "Engine build fails"**
- **Cause**: Insufficient workspace memory
- **Fix**: Increase `--workspace` parameter or reduce batch size

**Issue: "Accuracy degradation"**
- **Cause**: FP16 precision loss
- **Fix**: Use FP32 or add calibration dataset for INT8

### 7.2 Debug Checklist

- [ ] ONNX export uses T=150 (not 300)
- [ ] Input shapes match training: (1, 3, 150, 33, M)
- [ ] TensorRT engine builds without errors
- [ ] Inference output shape is (1, 2) or (2,)
- [ ] PyTorch and TensorRT outputs match within tolerance
- [ ] CUDA streams properly synchronized
- [ ] No memory leaks (check GPU memory usage)

---

## Next Steps

1. **Run export scripts** with correct dimensions
2. **Validate ONNX models** match PyTorch
3. **Build TensorRT engines** on Jetson
4. **Integrate into pipeline** with TensorRT wrapper
5. **Benchmark performance** (should see 2-4x speedup)
6. **Test end-to-end** with real camera feeds

---

**Questions or Issues?** Check the troubleshooting section or review CUDA concepts in Section 2.
