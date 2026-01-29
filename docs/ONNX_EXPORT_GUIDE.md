# ST-GCN ONNX Export Guide

## Overview

This document describes the modifications made to the ST-GCN (Spatial-Temporal Graph Convolutional Network) architecture to enable successful ONNX export and TensorRT optimization. The changes ensure compatibility with ONNX's computational graph requirements and TensorRT's inference engine.

---

## Table of Contents

1. [Architecture Modifications](#architecture-modifications)
2. [Export Configuration](#export-configuration)
3. [Export Commands](#export-commands)
4. [Verification](#verification)

---

## Architecture Modifications

### 1. Inplace Operations → Non-Inplace Operations

**Files Modified:** `net/st_gcn_trt.py`

**Changes Made:**
```python
# Before (Original)
nn.ReLU(inplace=True)
nn.Dropout(dropout, inplace=True)

# After (ONNX-Compatible)
nn.ReLU(inplace=False)
nn.Dropout(dropout, inplace=False)
```


**Reason:**
- **ONNX Requirement:** Inplace operations modify tensors directly in memory, which breaks ONNX's computational graph tracking

---

### 2. Adaptive Average Pooling for Dynamic Shapes

**Files Modified:** `net/st_gcn_trt.py`

**Original Implementation (st_gcn.py):**
```python
# Lines 82-85 in st_gcn.py
# Pool over the current temporal and spatial dimensions (after downsampling)
t_dim = x.size(2)
v_dim = x.size(3)
x = F.avg_pool2d(x, kernel_size=(t_dim, v_dim))
x = x.view(N, M, -1, 1, 1).mean(dim=1)
```

**New Implementation (st_gcn_trt.py):**
```python
# Line 85 in st_gcn_trt.py
# Reduces (N*M, 256, T_final, V) -> (N*M, 256, 1, 1)
x = F.adaptive_avg_pool2d(x, (1, 1))

# Multi-person Fusion
# Decouple N and M: (N, M, 256)
x = x.view(N, M, 256)

# Average features across people (dim=1)
x = torch.mean(x, dim=1)

# prediction
x = x.view(N, 256, 1, 1)
```

**Reason:**
- **ONNX Compatibility:** `adaptive_avg_pool2d` with fixed output size (1, 1) is better supported in ONNX/TensorRT than dynamic kernel size computation
- **Deterministic Behavior:** Original code calculated pool size dynamically from tensor dimensions, which can cause issues during ONNX graph tracing
- **Simplification:** Output size (1, 1) achieves global average pooling without needing to query tensor dimensions
- **Performance:** TensorRT can optimize fixed-size pooling operations more effectively
- **Multi-person Handling:** Clearer separation of multi-person averaging logic for better readability


---

### 3. Motion Stream Computation with F.pad

**Files Modified:** `net/st_gcn_twostream_trt.py`

**Original Implementation (st_gcn_twostream.py):**
```python
# Lines 24-26 in st_gcn_twostream.py
m = torch.cat((torch.zeros(N, C, 1, V, M, device=x.device, dtype=x.dtype),
                x[:, :, 1:] - x[:, :, :-1]), 2)
```

**New Implementation (st_gcn_twostream_trt.py):**
```python
# Lines 24-25 in st_gcn_twostream_trt.py
m_diff = x[:, :, 1:] - x[:, :, :-1]
m = F.pad(m_diff, (0, 0, 0, 0, 1, 0), "constant", 0)
```

**Explanation:**
- **Frame Differencing:** Computes motion between consecutive frames: `x[:, :, 1:] - x[:, :, :-1]`
  - Input: 150 frames → Output: 149 difference frames
- **Padding to Maintain Size:** `F.pad(m_diff, (0, 0, 0, 0, 1, 0), "constant", 0)`
  - Pad parameters: `(left, right, top, bottom, front, back)` for 5D tensor
  - `(0, 0, 0, 0, 1, 0)` adds 1 zero frame at the beginning of the temporal dimension
  - Restores tensor size from 149 to 150 frames

**Reason:**
- **ONNX Operator Support:** `F.pad` is a standard ONNX operator with excellent TensorRT support
- **Memory Efficiency:** `F.pad` is more efficient than `torch.cat` with zero tensor creation
- **Deterministic Tracing:** Avoids dynamic tensor allocation (`torch.zeros`) during ONNX export
- **Explicit Semantics:** Clearer that we're padding motion differences rather than concatenating
- **Constant Folding:** TensorRT can optimize padding operations more effectively


**Why Prepend Zero Frame:**
- Motion at frame 0 is undefined (no previous frame)
- Prepending zeros maintains temporal alignment: frame i in motion stream corresponds to frame i in origin stream
- Preserves sequence length at 150 frames for both streams

---

### 4. Einsum-Based Graph Convolution (TensorRT Optimized)

**Files Modified:** `net/st_gcn_trt.py`, `net/st_gcn_twostream_trt.py`

**Implementation:**
```python
# Einsum operation for graph convolution (net/utils/tgcn.py)
x = torch.einsum('nkctv,kvw->nctw', (x, A))
```

**Why Einsum:**
- **TensorRT Performance:** Modern TensorRT (10.3+) with ONNX opset 18 efficiently optimizes einsum operations
- **Original Design:** The original ST-GCN implementation used einsum - it was already optimal
- **Proven Results:** 
  - M=1: 8.86ms latency (112 FPS)
  - M=5: 54.84ms latency (18 FPS)
  - 20 Einsum operators preserved in ONNX graph
  - 0 MatMul operators (avoided slow decomposition)

**Critical Finding:**
- Initial MatMul-based decomposition caused catastrophic 410ms latency
- Reverting to original einsum implementation restored optimal performance
- TensorRT can directly optimize einsum when exported with ONNX opset 18

---

### 5. Softmax Layer Addition

**Files Modified:** `stgcn_to_onnx.py`, `stgcn_to_onnx_einsum.py`

**Implementation:**
```python
class ModelWithSoftmax(nn.Module):
    """Wrapper to add softmax layer for ONNX export"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
    
    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=1)
```

**Reason:**
- **Training vs Inference:** During training, PyTorch's `CrossEntropyLoss` includes softmax internally
- **Deployment Need:** Inference requires explicit probabilities for classification decisions
- **GPU Acceleration:** Softmax runs on GPU within the model, avoiding CPU post-processing
- **Simplicity:** Eliminates need for manual softmax in deployment code
- **TensorRT Optimization:** TensorRT can fuse softmax with preceding layers for better performance

**Output Format:**
- **Without Softmax:** Raw logits (e.g., `[-92.37, 93.28]`)
- **With Softmax:** Probabilities that sum to 1.0 (e.g., `[0.0, 1.0]`)

---

## Export Configuration

### Model Variants

#### Variant 1: Single Person (M=1)
```python
ExportConfig(
    M=1,          # Single person
    T=150,        # 150 temporal frames
    C=3,          # 3 channels (x, y, confidence)
    V=33,         # 33 joints (MediaPipe skeleton)
    num_class=2,  # Binary classification
    layout='mediapipe'
)
```

**Input Shape:** `(N, 3, 150, 33, 1)`
- N: Batch size (fixed at 1 for TensorRT optimization)
- 3: X coordinate, Y coordinate, confidence score
- 150: Sequence length (temporal dimension)
- 33: Number of skeleton joints
- 1: Single person

**Output Shape:** `(N, 2)`
- N: Batch size
- 2: Class probabilities [class_0_prob, class_1_prob]

---

#### Variant 2: Multi-Person (M=5)
```python
ExportConfig(
    M=5,          # Up to 5 people
    T=150,        # 150 temporal frames
    C=3,          # 3 channels (x, y, confidence)
    V=33,         # 33 joints (MediaPipe skeleton)
    num_class=2,  # Binary classification
    layout='mediapipe'
)
```

**Input Shape:** `(N, 3, 150, 33, 5)`
- Same as Variant 1, but M=5 for multi-person scenarios

**Output Shape:** `(N, 2)`
- Multi-person features are aggregated internally via average pooling
- Single prediction per batch (not per person)


---

## Export Commands

### Recommended: TensorRT-Optimized Export (Einsum-Based)

**This is the recommended method for TensorRT deployment** - uses the optimized `stgcn_to_onnx_einsum.py` script with einsum preservation.

**Single Person (M=1):**
```bash
python3 stgcn_to_onnx_einsum.py \
    --checkpoint models/epoch60_model.pt \
    --output exported_models/stgcn_M1_einsum.onnx \
    --M 1
```

**Dual Person (M=2):**
```bash
python3 stgcn_to_onnx_einsum.py \
    --checkpoint models/stgcn_unified.pt \
    --output exported_models/stgcn_M2_einsum.onnx \
    --M 2
```

**Multi-Person (M=5):**
```bash
python3 stgcn_to_onnx_einsum.py \
    --checkpoint models/epoch50_model.pt \
    --output exported_models/stgcn_M5_einsum.onnx \
    --M 5
```

**Parameters:**
- `--checkpoint`: Path to trained PyTorch checkpoint (.pt file or directory)
- `--output`: Output ONNX file path
- `--M`: Number of persons (1, 2, 3, 4, or 5)
- `--opset`: ONNX opset version (default: 18, minimum: 12 for einsum)

**Output:**
- ONNX model file (e.g., `stgcn_M1_einsum.onnx`)
- External data file (e.g., `stgcn_M1_einsum.onnx.data`) - ~24MB weights
- Includes softmax layer for probability output

**Performance (TensorRT FP16 on Jetson Orin NX):**
- M=1: 8.86ms latency (112 FPS)
- M=5: 54.84ms latency (18 FPS)
- 20 Einsum operators optimized by TensorRT

---

### Alternative: Legacy Export Method

**Use the original `stgcn_to_onnx.py` for verification or compatibility testing:**

**Variant 1 (M=1):**
```bash
python3 stgcn_to_onnx.py \
    --variant variant1 \
    --checkpoint models/single_model.pt \
    --output-dir ./exported_models/variant1
```

**Variant 2 (M=5):**
```bash
python3 stgcn_to_onnx.py \
    --variant variant2 \
    --checkpoint models/multi_model.pt \
    --output-dir ./exported_models/variant2
```

**Output Files:**
- `stgcn_two_stream_M{1|5}_T150_C3_V33_softmax.onnx` - ONNX model with softmax
- `stgcn_two_stream_M{1|5}_T150_C3_V33_softmax_config.txt` - Configuration metadata

---

### Export Without Softmax (Raw Logits)

**For advanced use cases requiring custom post-processing:**

```bash
python3 stgcn_to_onnx.py \
    --variant variant1 \
    --checkpoint models/single_model.pt \
    --output-dir ./exported_models/variant1 \
    --no-softmax
```

**Use Cases:**
- Custom probability calibration
- Ensemble models requiring logit averaging
- Integration with frameworks that expect raw scores

---

### Custom Configuration

```bash
python3 stgcn_to_onnx.py \
    --variant variant1 \
    --checkpoint path/to/your/model.pt \
    --output-dir ./custom_output \
    --num-class 2 \
    --layout mediapipe
```

**Parameters:**
- `--variant`: Model variant (variant1 or variant2)
- `--checkpoint`: Path to trained PyTorch checkpoint (.pt file)
- `--output-dir`: Directory for exported ONNX models
- `--num-class`: Number of action classes (default: 2)
- `--layout`: Skeleton layout (mediapipe/openpose/ntu-rgb+d)
- `--no-softmax`: Export without softmax layer (optional flag)

---

## Verification

### ONNX Model Validation

The export script automatically performs:

1. **ONNX Structure Check:**
   ```
   ✓ ONNX model is valid
   ```
   - Validates graph structure
   - Checks operator compatibility
   - Verifies tensor dimensions

2. **Input/Output Inspection:**
   ```
   Inputs:
     input: [1, 3, 150, 33, M]
   Outputs:
     output: [1, 2]
   ```

3. **ONNX Runtime Test:**
   ```
   ✓ ONNX Runtime inference successful
     Output shape: (1, 2)
     Output sample: [0.4969551 0.5030449]
     Sum of outputs: 1.000000 (probabilities ✓)
   ```
   - Executes forward pass with random input
   - Validates output shape
   - Confirms probability distribution (sum ≈ 1.0)

### Manual Verification

```python
import onnx
import onnxruntime as ort
import numpy as np

# Load ONNX model
model_path = "exported_models/variant1/stgcn_two_stream_M1_T150_C3_V33_softmax.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

# Create inference session
session = ort.InferenceSession(model_path)

# Prepare input (random test data)
input_data = np.random.randn(1, 3, 150, 33, 1).astype(np.float32)

# Run inference
outputs = session.run(None, {'input': input_data})
probabilities = outputs[0]

print(f"Probabilities: {probabilities}")
print(f"Predicted class: {np.argmax(probabilities)}")
print(f"Confidence: {np.max(probabilities):.2%}")
```


---



## File Summary

### Modified Files

1. **`net/st_gcn_trt.py`**
   - Changed: All `inplace=True` → `inplace=False`
   - Lines: 167, 176, 195
   - Purpose: ONNX graph compatibility

2. **`net/st_gcn_trt.py`**
   - Changed: `F.avg_pool2d(x, kernel_size=(t_dim, v_dim))` → `F.adaptive_avg_pool2d(x, (1, 1))`
   - Line: 85
   - Purpose: Fixed-size pooling for ONNX deterministic graph tracing

3. **`net/st_gcn_twostream_trt.py`**
   - Changed: `torch.cat` with `torch.zeros` → `F.pad` for motion stream
   - Lines: 24-25
   - Purpose: Replace dynamic tensor creation with efficient padding operator

4. **`net/utils/tgcn.py`**
   - Uses: `torch.einsum` for graph convolution (original implementation)
   - Purpose: TensorRT-optimized graph operations with opset 18

5. **`stgcn_to_onnx_einsum.py`** (New - Recommended)
   - Added: Streamlined export script with einsum preservation
   - Purpose: Production TensorRT-optimized ONNX export
   - Features: Supports M=1-5, opset 18, fixed batch size, softmax included

6. **`stgcn_to_onnx.py`**
   - Added: `ModelWithSoftmax` wrapper class
   - Added: Softmax integration in export pipeline
   - Added: CLI flag `--no-softmax` for flexibility
   - Purpose: Export automation and probability output


---

**Document Version:** 2.0  
**Last Updated:** January 28, 2026  
**Branch:** `feature/onnx-export-modifications`  
**Key Update:** Reverted to original einsum implementation for optimal TensorRT performance
