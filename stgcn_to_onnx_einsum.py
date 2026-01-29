#!/usr/bin/env python3
"""
Quick re-export script with einsum for TensorRT optimization
Uses opset 18 
"""

import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the einsum-based model
from net.st_gcn_twostream_trt import Model as TwoStreamModel

def export_model_einsum(checkpoint_path, output_path, M, opset_version=18):
    """
    Export ST-GCN model to ONNX using einsum (faster for TensorRT)
    
    Args:
        checkpoint_path: Path to PyTorch model checkpoint
        output_path: Path for output ONNX file
        M: Number of persons (1 or 5)
        opset_version: ONNX opset version (12+ for einsum support, default: 18)
    """
    print(f"\n{'='*60}")
    print(f"EXPORTING WITH EINSUM (OPSET {opset_version})")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Max persons (M): {M}")
    print(f"ONNX opset: {opset_version}")
    print()
    
    # Model configuration
    in_channels = 3
    num_class = 2
    layout = 'mediapipe'
    
    # Create model with einsum-based graph convolution
    model = TwoStreamModel(
        in_channels=in_channels,
        num_class=num_class,
        graph_args={'layout': layout, 'strategy': 'spatial'},
        edge_importance_weighting=True,
        dropout=0.5
    )
    
    # Load weights
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Create wrapper with softmax
    class ModelWithSoftmax(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
        def forward(self, x):
            logits = self.base_model(x)
            return torch.nn.functional.softmax(logits, dim=1)
    
    wrapped_model = ModelWithSoftmax(model)
    wrapped_model.eval()
    
    # Create dummy input
    T = 150
    V = 33
    C = 3
    N = 1
    dummy_input = torch.randn(N, C, T, V, M)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Test forward pass
    with torch.no_grad():
        output = wrapped_model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Output (softmax): {output[0].numpy()}")
        print(f"Sum of probabilities: {output[0].sum().item():.6f}")
    
    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None,  # Fixed batch size for TensorRT optimization
        verbose=False
    )
    
    print(f"\n SUCCESS: ONNX model exported to {output_path}")
    print(f"   Using einsum with opset {opset_version} (TensorRT-optimized)")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Re-export ST-GCN with einsum for TensorRT')
    parser.add_argument('--checkpoint', required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', required=True, help='Output ONNX file path')
    parser.add_argument('--M', type=int, required=True, choices=[1, 2, 3, 4, 5], help='Number of persons (1, 2, 3, 4, or 5)')
    parser.add_argument('--opset', type=int, default=18, help='ONNX opset version (default: 12, minimum for einsum)')
    
    args = parser.parse_args()
    
    # Validate opset version
    if args.opset < 12:
        print(f"  WARNING: Opset {args.opset} does not support Einsum. Using opset 12 instead.")
        args.opset = 12
    
    export_model_einsum(args.checkpoint, args.output, args.M, args.opset)
    
