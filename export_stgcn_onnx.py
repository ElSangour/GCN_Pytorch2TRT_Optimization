#!/usr/bin/env python3
"""
Correct ONNX export matching training dimensions:
- T=150 (not 300!) - CRITICAL FIX
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
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = '../models/epoch60_model.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # CORRECT input shape: (N, C, T, V, M) = (1, 3, 150, 33, 1)
    # ⚠️ CRITICAL: T=150 (SEQ_LEN from training), NOT 300!
    C, T, V, M = 3, 150, 33, 1
    dummy_input = torch.randn(1, C, T, V, M)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Model type: {type(model).__name__}")
    
    # Verify forward pass works
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Test output shape: {test_output.shape} (expected: (1, 2))")
        if test_output.shape != (1, 2):
            print(f"⚠️ Warning: Unexpected output shape!")
    
    # Export to ONNX
    output_path = '../models/stgcn_single_correct.onnx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nExporting to: {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=12,  # Required for einsum support
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        dynamic_axes=None  # Static shapes only for TensorRT
    )
    
    print(f"✅ Exported to {output_path}")
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
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = '../models/epoch50_model.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None
    
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # CORRECT input shape: (1, 3, 150, 33, 5)
    # ⚠️ CRITICAL: T=150, M=5 for multi-person!
    C, T, V, M = 3, 150, 33, 5
    dummy_input = torch.randn(1, C, T, V, M)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Verify forward pass
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Test output shape: {test_output.shape} (expected: (1, 2))")
        if test_output.shape != (1, 2):
            print(f"⚠️ Warning: Unexpected output shape!")
    
    output_path = '../models/stgcn_multi_correct.onnx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nExporting to: {output_path}")
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
        dynamic_axes=None  # Static shapes
    )
    
    print(f"✅ Exported to {output_path}")
    return output_path

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ST-GCN ONNX Export (CORRECTED DIMENSIONS)")
    print("=" * 60)
    print("⚠️  CRITICAL: Using T=150 (SEQ_LEN) not T=300!")
    print("=" * 60 + "\n")
    
    single_path = export_single_person_model()
    multi_path = export_multi_person_model()
    
    print("\n" + "=" * 60)
    if single_path and multi_path:
        print("✅ ONNX Export Complete!")
        print(f"   Single-person: {single_path}")
        print(f"   Multi-person: {multi_path}")
        print("\nNext steps:")
        print("1. Validate ONNX models: python validate_onnx.py")
        print("2. Convert to TensorRT: python onnx_to_tensorrt.py")
    else:
        print("❌ Export failed - check model paths")
    print("=" * 60)
