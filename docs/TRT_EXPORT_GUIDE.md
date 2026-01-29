# TensorRT Engine Build and Testing Guide

## Overview

This guide covers the complete workflow for converting ONNX models to TensorRT engines, testing with trtexec, and benchmarking performance on NVIDIA Jetson devices.

---

## Table of Contents


1. [Building TensorRT Engines](#building-tensorrt-engines)
2. [Testing with trtexec](#testing-with-trtexec)
3. [Benchmarking](#benchmarking)
4. [Performance Results](#performance-results)

---

## Building TensorRT Engines

### Basic Syntax

```bash
python3 onnx_to_tensorrt.py \
    --onnx <path_to_onnx_model> \
    --engine <output_engine_path> \
    --precision <fp32|fp16|int8> \
    --workspace <size_in_GB> \
    --optimization-level <0-5>
```

### Parameters Explained

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--onnx` | Required | - | Path to ONNX model file |
| `--engine` | Required | - | Output path for TensorRT engine |
| `--precision` | Optional | `fp16` | Precision mode: `fp32`, `fp16`, or `int8` |
| `--workspace` | Optional | `4` | Workspace size in GB (4-12 recommended) |
| `--optimization-level` | Optional | `3` | Optimization level 0-5 (higher = better perf, slower build) |
| `--verbose` | Flag | - | Enable detailed build logging |

---

### Recommended Configurations

#### Single Person Model (M=1)

```bash
python3 onnx_to_tensorrt.py \
    --onnx exported_models/stgcn_M1_einsum.onnx \
    --engine tensorrt_engines/stgcn_M1_fp16.engine \
    --precision fp16 \
    --workspace 12 \
    --optimization-level 5
```

**Expected Build Time:** 5-10 minutes  
**Expected Engine Size:** ~13 MB  
**Target Performance:** 8-9ms latency (112 FPS)

---

#### Dual Person Model (M=2)

```bash
python3 onnx_to_tensorrt.py \
    --onnx exported_models/stgcn_M2_einsum.onnx \
    --engine tensorrt_engines/stgcn_M2_fp16.engine \
    --precision fp16 \
    --workspace 12 \
    --optimization-level 5
```

**Expected Build Time:** 7-12 minutes  
**Expected Engine Size:** ~13.5 MB  
**Target Performance:** 20-25ms latency (40-50 FPS)

---

#### Multi-Person Model (M=5)

```bash
python3 onnx_to_tensorrt.py \
    --onnx exported_models/stgcn_M5_einsum.onnx \
    --engine tensorrt_engines/stgcn_M5_fp16.engine \
    --precision fp16 \
    --workspace 12 \
    --optimization-level 5
```

**Expected Build Time:** 10-15 minutes  
**Expected Engine Size:** ~14.7 MB  
**Target Performance:** 54-55ms latency (18 FPS)

---

### Advanced Build Options

#### Maximum Performance (Longer Build Time)

```bash
python3 onnx_to_tensorrt.py \
    --onnx exported_models/stgcn_M1_einsum.onnx \
    --engine tensorrt_engines/stgcn_M1_optimized.engine \
    --precision fp16 \
    --workspace 12 \
    --optimization-level 5 \
    --verbose
```

- **Optimization Level 5:** Maximum runtime performance optimization
- **Large Workspace:** More memory for kernel selection and tuning
- **Verbose Mode:** Detailed layer-by-layer build information

---

## Testing with trtexec

### What is trtexec?

`trtexec` is TensorRT's command-line tool for:
- Profiling engine performance
- Layer-by-layer timing analysis
- Validating engine builds
- Comparing different configurations

### Basic Profiling Command

```bash
trtexec \
    --loadEngine=tensorrt_engines/stgcn_M1_fp16.engine \
    --iterations=100 \
    --avgRuns=100
```

**Output Metrics:**
- **Mean Latency:** Average inference time
- **Median Latency:** 50th percentile
- **99th Percentile:** Worst-case performance
- **Throughput:** Inferences per second

---


### Production-Grade Profiling (Comprehensive)

**This is the recommended profiling command for production validation** - includes all optimization flags and detailed analysis:

```bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=tensorrt_engines/stgcn_M5_fp16_optimized.engine \
    --shapes=input:1x3x150x33x5 \
    --iterations=200 \
    --warmUp=500 \
    --avgRuns=50 \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --dumpProfile \
    --separateProfileRun \
    --useCudaGraph \
    --useSpinWait
```

**Parameters Explained:**

| Parameter | Description |
|-----------|-------------|
| `--loadEngine` | Path to TensorRT engine file |
| `--shapes=input:1x3x150x33x5` | Explicitly specify input shape (required for some engines) |
| `--iterations=200` | Run 200 inference iterations |
| `--warmUp=500` | 500ms GPU warmup to stabilize clocks and caches |
| `--avgRuns=50` | Average results over 50 runs for stability |
| `--profilingVerbosity=detailed` | Enable detailed layer-by-layer timing |
| `--dumpLayerInfo` | Print layer information (types, shapes, precision) |
| `--dumpProfile` | Display profiling results in console |
| `--separateProfileRun` | Run profiling in separate pass (more accurate) |
| `--useCudaGraph` | Enable CUDA graphs for reduced launch overhead |
| `--useSpinWait` | Use spin-wait for accurate timing (reduces jitter) |

**For other model variants:**

**M=1 (Single Person):**
```bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=tensorrt_engines/stgcn_M1_fp16.engine \
    --shapes=input:1x3x150x33x1 \
    --iterations=200 \
    --warmUp=500 \
    --avgRuns=50 \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --dumpProfile \
    --separateProfileRun \
    --useCudaGraph \
    --useSpinWait
```

**M=2 (Dual Person):**
```bash
/usr/src/tensorrt/bin/trtexec \
    --loadEngine=tensorrt_engines/stgcn_M2_fp16.engine \
    --shapes=input:1x3x150x33x2 \
    --iterations=200 \
    --warmUp=500 \
    --avgRuns=50 \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --dumpProfile \
    --separateProfileRun \
    --useCudaGraph \
    --useSpinWait
```

**Expected Output:**
```
=== Performance Summary ===
Throughput: 112.5 qps
Latency: min = 8.60 ms, max = 9.12 ms, mean = 8.86 ms, median = 8.84 ms, percentile(90%) = 9.02 ms, percentile(95%) = 9.06 ms, percentile(99%) = 9.10 ms

=== Layer Performance ===
Layer(Einsum): 20 occurrences, avg time = 0.35ms each
Layer(Conv): 20 occurrences, avg time = 0.18ms each
Layer(BatchNorm): 10 occurrences, avg time = 0.08ms each
```

**Key Metrics to Verify:**
-  Mean latency matches benchmark expectations
-  Einsum operators present and fast (~0.3-0.5ms each)
-  No slow MatMul operators (should be 0 occurrences)
-  Low jitter (percentile 99% close to mean)

---

## Benchmarking

### Python Benchmarking Scripts

The repository includes two benchmark scripts:

1. **`benchmark_single_person.py`** - For M=1 models
2. **`benchmark_multi_person.py`** - For M=2 and M=5 models

---

### Single Person Benchmarking (M=1)

**Setup:**
```bash
# Edit benchmark_single_person.py to set paths
PYTORCH_MODEL_PATH = 'models/epoch60_model.pt'
TRT_ENGINE_PATH = 'tensorrt_engines/stgcn_M1_fp16.engine'
MAX_PERSONS = 1
```

**Run Benchmark:**
```bash
python3 benchmark_single_person.py
```

**Output:**
```
================================================================
ST-GCN SINGLE-PERSON (M=1) BENCHMARK - PyTorch vs TensorRT
================================================================

PyTorch Model (GPU - FP32):
  Mean latency: 31.45ms
  FPS: 31.8

TensorRT Engine (GPU - FP16):
  Mean latency: 8.86ms
  FPS: 112.9

Speedup: 3.55x faster with TensorRT (FP16 optimization)
================================================================
```
---

### Multi-Person Benchmarking (M=2)

**Setup:**
```bash
# Edit benchmark_multi_person.py
PYTORCH_MODEL_PATH = 'models/stgcn_unified.pt'
TRT_ENGINE_PATH = 'tensorrt_engines/stgcn_M2_fp16.engine'
MAX_PERSONS = 2
```

**Run Benchmark:**
```bash
python3 benchmark_multi_person.py
```

**Expected Results:**
- Mean latency: ~20-25ms
- FPS: ~40-50
- Speedup: ~2.5-3x vs PyTorch GPU (FP32)

---

### Multi-Person Benchmarking (M=5)

**Setup:**
```bash
# Edit benchmark_multi_person.py
PYTORCH_MODEL_PATH = 'models/epoch50_model.pt'
TRT_ENGINE_PATH = 'tensorrt_engines/stgcn_M5_fp16.engine'
MAX_PERSONS = 5
```

**Run Benchmark:**
```bash
python3 benchmark_multi_person.py
```

**Validated Results:**
- PyTorch GPU (FP32): 120.57ms (8.29 FPS)
- TensorRT FP16: 53.05ms (18.85 FPS)
- Speedup: 2.27x
- FPS improvement: 127.3%


---

### Custom Benchmark Parameters

You can modify benchmark parameters in the scripts:

```python
# Number of warmup iterations
WARMUP_ITERATIONS = 50

# Number of benchmark iterations
BENCHMARK_ITERATIONS = 200

# Skeleton configuration
T = 150  # Temporal frames
V = 33   # MediaPipe joints
C = 3    # Channels (x, y, confidence)
```

---

## Performance Results

### Validated Performance (Jetson Orin NX)

| Model | TensorRT FP16 | PyTorch GPU (FP32) | Speedup | TensorRT FPS |
|-------|---------------|---------------------|---------|--------------|
| M=1 (Single Person) | **8.86ms** | 31.45ms | **3.55x** | 112.9 |
| M=2 (Dual Person) | **~22ms** | ~65ms | **~3x** | ~45 |
| M=5 (Multi-Person) | **53.05ms** | 120.57ms | **2.27x** | 18.85 |

**Note:** All measurements use the optimized einsum implementation. Speedups are from TensorRT FP16 precision and graph optimizations.


**Scaling Behavior:**
- M=1 → M=2: ~2.5x latency increase (expected)
- M=2 → M=5: ~2.5x latency increase (expected)
- Linear scaling with number of persons

---



### Validate Results
```
Expected M=1 Performance:
✅ Mean latency: 8-10ms
✅ FPS: 100-120
✅ Speedup: 3-4x vs PyTorch GPU


Expected M=5 Performance:
✅ Mean latency: 50-55ms
✅ FPS: 18-20
✅ Speedup: 2-2.5x vs PyTorch GPU

```

---

**Document Version:** 1.0  
**Last Updated:** January 28, 2026  
**Target Platform:** NVIDIA Jetson (Orin NX)  
**TensorRT Version:** 10.3+ recommended  
**Validated Results:** M=1: 8.86ms, M=5: 54.84ms with FP16 precision
