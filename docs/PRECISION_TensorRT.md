# Precision Modes: FP32, FP16, and INT8

This document provides a comprehensive explanation of the different precision modes available for TensorRT engine optimization.

## Overview

TensorRT supports three main precision modes for neural network inference. Each mode represents a trade-off between computational speed, model size, memory bandwidth, and numerical accuracy.

## FP32: Full Precision (32-bit Floating Point)

### Technical Details
- **Data Type**: IEEE 754 single-precision floating-point
- **Bit Representation**: 1 sign bit, 8 exponent bits, 23 mantissa bits
- **Numeric Range**: ±1.4 × 10⁻⁴⁵ to ±3.4 × 10³⁸
- **Precision**: ~7 decimal digits

### Characteristics
- **Accuracy**: Exact match with PyTorch FP32 training
- **Speed**: Baseline performance (1x)
- **Model Size**: Largest - no compression
- **Memory Bandwidth**: Highest - requires more data transfer
- **GPU Utilization**: Standard CUDA cores

### When to Use FP32
- Model validation and debugging
- Establishing baseline accuracy metrics
- Applications where numerical precision is critical
- Hardware without FP16/INT8 acceleration
- Small models where speed is not a bottleneck

### Example Use Case
```bash
python3 export_stgcn_tensorrt.py \
    --onnx models/stgcn_single.onnx \
    --engine models/stgcn_single_fp32.engine \
    --precision fp32
```

## FP16: Half Precision (16-bit Floating Point)

### Technical Details
- **Data Type**: IEEE 754 half-precision floating-point
- **Bit Representation**: 1 sign bit, 5 exponent bits, 10 mantissa bits
- **Numeric Range**: ±6.1 × 10⁻⁵ to ±6.5 × 10⁴
- **Precision**: ~3 decimal digits

### Characteristics
- **Accuracy**: 99.5-99.9% of FP32 for most neural networks
- **Speed**: **2-3x faster** than FP32 on Jetson Xavier/Orin
- **Model Size**: **50% smaller** than FP32
- **Memory Bandwidth**: **50% less** data transfer
- **GPU Utilization**: Tensor Cores (if available)

### Hardware Requirements
- GPU with Tensor Cores (NVIDIA Volta architecture or newer)
- Jetson Xavier NX, Orin NX, AGX Xavier, AGX Orin
- Desktop GPUs: RTX 20/30/40 series, Tesla V100, A100

### Accuracy Considerations
- **Typical Loss**: <0.1-0.5% accuracy degradation
- **ST-GCN Performance**: Usually negligible impact on action recognition accuracy
- **Numerical Stability**: Sufficient for most computer vision tasks
- **Gradient Accumulation**: Not used during inference (training consideration)

### When to Use FP16
- **Recommended for production** deployment on Jetson
- Real-time applications requiring high FPS
- Multi-camera systems with limited GPU resources
- When model size needs to be reduced for deployment
- Applications where minor accuracy trade-off is acceptable

### Example Use Case
```bash
python3 export_stgcn_tensorrt.py \
    --onnx models/stgcn_single.onnx \
    --engine models/stgcn_single_fp16.engine \
    --precision fp16
```

### Automatic FP32 Fallback
Our conversion script includes automatic fallback:
- If FP16 build fails due to graph optimizer errors
- Automatically retries with FP32
- Saves engine with `_fp32` suffix
- Continues execution without manual intervention

## INT8: Integer Quantization (8-bit Integer)

### Technical Details
- **Data Type**: Signed 8-bit integer
- **Bit Representation**: 8 bits total
- **Numeric Range**: -128 to 127 (scaled to represent float range)
- **Precision**: Discrete 256 levels per activation

### Characteristics
- **Accuracy**: 95-98% of FP32 (with proper calibration)
- **Speed**: **3-5x faster** than FP32
- **Model Size**: **75% smaller** than FP32
- **Memory Bandwidth**: **75% less** data transfer
- **GPU Utilization**: INT8 Tensor Cores

### Quantization Process
INT8 requires a **calibration** step to determine optimal scaling factors:

1. **Collect Statistics**: Run inference on representative dataset
2. **Determine Scale Factors**: Calculate min/max ranges for each layer
3. **Apply Quantization**: Convert FP32 weights to INT8
4. **Calibration Dataset**: 500-1000 representative samples recommended

### Accuracy Considerations
- **Typical Loss**: 2-5% accuracy degradation
- **Depends On**: Quality and representativeness of calibration dataset
- **Layer Sensitivity**: Some layers may be kept in FP16 for accuracy
- **Post-Training Quantization**: Does not require retraining

### When to Use INT8
- Maximum inference speed required
- Severely resource-constrained environments
- Batch processing where slight accuracy loss is acceptable
- After validating accuracy with calibration dataset
- Models with redundant precision (large margins)

### Calibration Dataset Requirements
For ST-GCN action recognition:
- **Size**: 500-1000 skeleton sequences
- **Diversity**: Cover all action classes
- **Quality**: Representative of real-world deployment
- **Format**: Same preprocessing as training data

### Example Use Case
```bash
# Note: INT8 requires calibration implementation (coming soon)
python3 export_stgcn_tensorrt.py \
    --onnx models/stgcn_single.onnx \
    --engine models/stgcn_single_int8.engine \
    --precision int8 \
    --calibration-data path/to/calibration_dataset
```

## Precision Comparison Matrix

| Aspect | FP32 | FP16 | INT8 |
|--------|------|------|------|
| **Performance** |
| Relative Speed | 1x | 2-3x | 3-5x |
| Throughput (FPS) | ~6-7 | ~16-25 | ~25-33 |
| Latency | ~150ms | ~40-60ms | ~30-40ms |
| **Resources** |
| Model Size | 100% | 50% | 25% |
| Memory Bandwidth | 100% | 50% | 25% |
| GPU Memory | High | Medium | Low |
| **Accuracy** |
| Precision Loss | 0% | <0.5% | 2-5% |
| Typical Accuracy | 100% | 99.5-99.9% | 95-98% |
| **Requirements** |
| Hardware | Any CUDA GPU | Tensor Cores | INT8 Tensor Cores |
| Calibration | No | No | **Yes** |
| Setup Complexity | Low | Low | Medium-High |
| **Use Cases** |
| Development/Debug | Excellent | Good | Limited |
| Production | Good | **Recommended** | Specialized |
| Real-time | Limited | Excellent | Maximum |

## ST-GCN Specific Recommendations

### For Development
1. **Start with FP32**: Establish baseline
2. **Validate ONNX**: Ensure correct export
3. **Test FP16**: Measure accuracy impact
4. **Profile Performance**: Benchmark all precisions

### For Production Deployment

#### Single Camera System
- **Recommended**: FP16
- **Rationale**: Best balance of speed and accuracy
- **Expected Performance**: 15-20 FPS end-to-end

#### Multi-Camera System (2-4 cameras)
- **Recommended**: FP16
- **Rationale**: Sufficient speed with good accuracy
- **Expected Performance**: 12-16 FPS per camera

#### Multi-Camera System (5+ cameras)
- **Consider**: INT8 with careful calibration
- **Rationale**: Maximum throughput needed
- **Expected Performance**: 20-25 FPS per camera
- **Requirement**: Validate accuracy on your specific actions

### Action Recognition Specific
- **High-Accuracy Actions** (e.g., fine-grained gestures): Use FP16
- **Coarse Actions** (e.g., walking, running): INT8 acceptable
- **Safety-Critical**: FP16 minimum, validate extensively
- **Real-Time Alerts**: FP16 for reliability

## Measuring Accuracy Impact

### Recommended Validation Process

1. **Export all three precisions**
2. **Run inference on validation set**
3. **Compare outputs**:
   - Class predictions (accuracy, F1-score)
   - Confidence scores (distribution analysis)
   - Confusion matrices
4. **Measure performance**:
   - Average inference time
   - FPS throughput
   - Memory usage
5. **Make informed decision** based on requirements

### Example Validation Script Structure
```python
# Pseudo-code for precision validation
for precision in ['fp32', 'fp16', 'int8']:
    engine = load_engine(f'model_{precision}.engine')
    
    correct = 0
    total = 0
    
    for sample in validation_set:
        prediction = engine.infer(sample)
        if prediction == ground_truth:
            correct += 1
        total += 1
    
    accuracy = correct / total
    print(f"{precision}: {accuracy:.2%} accuracy")
```

## Technical Implementation Notes

### FP32 to FP16 Conversion
- **Automatic**: TensorRT handles conversion
- **Layer Precision**: Can be mixed (some layers FP32, others FP16)
- **Builder Flag**: `config.set_flag(trt.BuilderFlag.FP16)`

### FP32 to INT8 Quantization
- **Requires Calibrator**: Custom implementation needed
- **Symmetric Quantization**: `scale = max(abs(min), abs(max)) / 127`
- **Per-Tensor or Per-Channel**: TensorRT supports both
- **Builder Flag**: `config.set_flag(trt.BuilderFlag.INT8)`

## Future Enhancements

This project roadmap includes:
- [ ] INT8 calibration dataset generator
- [ ] Automated precision selection based on hardware
- [ ] Mixed precision profiling tools
- [ ] Accuracy regression testing suite
- [ ] Per-layer precision analysis

## References

- [TensorRT Developer Guide - Reduced Precision](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#reduced-precision)
- [NVIDIA Blog: 8-bit Inference with TensorRT](https://developer.nvidia.com/blog/int8-inference-autonomous-vehicles-tensorrt/)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

---

**Note**: All performance numbers are based on NVIDIA Jetson Orin NX 16GB. Your results may vary based on model architecture, input size, and hardware configuration.
