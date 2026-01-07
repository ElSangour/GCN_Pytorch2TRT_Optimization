# ST-GCN TensorRT Export - Quick Start Guide

**Platform:** Jetson ORIN NX

This guide provides the minimum required steps to export trained ST-GCN models to TensorRT and integrate them into the real-time pipeline. For architectural details, CUDA concepts, and design rationale, see `ROADMAP.md`.

## Scope

- **Platform:** Jetson ORIN NX
- **Models:**
  - Single-person ST-GCN (M=1)
  - Multi-person ST-GCN (M=5)
- **Input sequence length:** T=150 (must match training)
- **Output:** TensorRT engines for FP16 inference

## Prerequisites

Ensure the following are available:

- **Trained PyTorch checkpoints:**
  - `epoch60_model.pt` (single-person)
  - `epoch50_model.pt` (multi-person)
- **JetPack 5.x**
- **Python environment with:**
  - `torch`
  - `onnx`
  - `tensorrt`
  - `onnxruntime` (for validation)

## Installation Steps

### Step 1 — Export ONNX Models

From the optimization directory:

```bash
cd GCN_Pytorch2TRT_Optimization
python3 export_stgcn_onnx_correct.py
```

**Expected outputs:**

```
models/stgcn_single_correct.onnx
models/stgcn_multi_correct.onnx
```

### Step 2 — Validate ONNX Outputs

Verify numerical consistency between PyTorch and ONNX:

```bash
python3 validate_onnx.py
```

**Validation criteria:**
- ONNX model loads successfully
- Output shape is `(1, 2)`
- Maximum numerical difference is within tolerance

### Step 3 — Build TensorRT Engines (Jetson)

**Single-person model (FP16):**

```bash
python3 onnx_to_tensorrt.py \
  --onnx models/stgcn_single_correct.onnx \
  --engine models/stgcn_single_fp16.engine \
  --precision fp16 \
  --workspace 4
```

**Multi-person model (FP16):**

```bash
python3 onnx_to_tensorrt.py \
  --onnx models/stgcn_multi_correct.onnx \
  --engine models/stgcn_multi_fp16.engine \
  --precision fp16 \
  --workspace 4
```

**Note:** Engine build time is typically 5–15 minutes per model on first run.

### Step 4 — Configure Runtime Paths

Update `real_time_detectionV2/configV2.py`:

```python
TRT_ENGINE_PATH_SINGLE = "GCN_Pytorch2TRT_Optimization/models/stgcn_single_fp16.engine"
TRT_ENGINE_PATH_MULTI  = "GCN_Pytorch2TRT_Optimization/models/stgcn_multi_fp16.engine"
```

### Step 5 — Enable TensorRT Inference

Replace PyTorch ST-GCN loading with TensorRT loading:

- Use `processing/stgcn_tensorrt_inference.py`
- Enable TensorRT path in the prediction logic
- Ensure TensorRT inference is selected via configuration or flag

## Expected Performance

Typical improvements compared to PyTorch FP32:

| Model | PyTorch | TensorRT FP16 |
|-------|---------|---------------|
| Single-person | ~150 ms | ~40–60 ms |
| Multi-person | ~200 ms | ~60–80 ms |

**End-to-end FPS:** Typically improves from ~5–6 FPS to ~15–20 FPS per camera.

## Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|--------------|------------|
| Input shape error | Sequence length mismatch | Re-export with T=150 |
| Engine build failure | Insufficient memory | Reduce workspace or free GPU memory |
| Accuracy drop | FP16 precision | Test FP32 engine first |

## Next Actions

1. Run end-to-end tests with live camera input
2. Benchmark TensorRT vs PyTorch inference
3. Optionally evaluate INT8 quantization

## Additional Documentation

For detailed explanations, internal mechanics, and troubleshooting, refer to `ROADMAP.md`.
