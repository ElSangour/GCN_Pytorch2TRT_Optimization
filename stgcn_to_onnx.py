import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import numpy as np
import argparse
import os
from pathlib import Path


class ModelWithSoftmax(nn.Module):
    """Wrapper to add softmax layer for ONNX export"""
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
    
    def forward(self, x):
        logits = self.model(x)
        return F.softmax(logits, dim=1)


class ExportConfig:
    """Configuration for model export"""
    def __init__(self, M, T, C, V, num_class, layout='mediapipe'):
        self.M = M  # Number of people
        self.T = T  # Temporal frames
        self.C = C  # Input channels (x, y, confidence)
        self.V = V  # Number of joints
        self.num_class = num_class
        self.layout = layout
        
    def __str__(self):
        return f"M{self.M}_T{self.T}_C{self.C}_V{self.V}"


def export_to_onnx(model, config, output_path, batch_size=1, opset_version=11, include_softmax=True):
    """
    Export ST-GCN model to ONNX format
    
    Args:
        model: PyTorch model
        config: ExportConfig object
        output_path: Path to save ONNX model
        batch_size: Batch size for export
        opset_version: ONNX opset version
        include_softmax: Whether to include softmax in the exported model (default: True)
    """
    model.eval()
    
    # Wrap model with softmax if requested
    if include_softmax:
        model = ModelWithSoftmax(model)
        print("✓ Adding softmax layer to exported model (output will be probabilities)")
    
    # Create dummy input with exact dimensions
    dummy_input = torch.randn(batch_size, config.C, config.T, config.V, config.M)
    
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
    
    print(f"Exporting model with input shape: {dummy_input.shape}")
    print(f"Expected output shape: ({batch_size}, {config.num_class})")
    
    # Test forward pass first
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"Test forward pass successful. Output shape: {output.shape}")
        except Exception as e:
            print(f"ERROR: Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            print("\nDEBUG INFO:")
            print(f"  Model type: {type(model).__name__}")
            print(f"  Input shape: {dummy_input.shape}")
            return False
    
    # Export to ONNX
    input_names = ['input']
    output_names = ['output']
    
    # For fixed shapes (recommended for TensorRT optimization)
    dynamic_axes = None
    
    # Alternative: If you want dynamic batch size
    # dynamic_axes = {
    #     'input': {0: 'batch_size'},
    #     'output': {0: 'batch_size'}
    # }
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        print(f"✓ ONNX model exported successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"ERROR during ONNX export: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_onnx(onnx_path, config, batch_size=1):
    """Verify ONNX model correctness"""
    import onnx
    import onnxruntime as ort
    
    print(f"\nVerifying ONNX model: {onnx_path}")
    
    # Load and check model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"ERROR: ONNX model validation failed: {e}")
        return False
    
    # Print model info
    print("\nModel Information:")
    print("Inputs:")
    for inp in onnx_model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")
    
    print("Outputs:")
    for out in onnx_model.graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")
    
    # Test inference with ONNX Runtime
    try:
        ort_session = ort.InferenceSession(onnx_path)
        dummy_input = np.random.randn(batch_size, config.C, config.T, 
                                      config.V, config.M).astype(np.float32)
        
        outputs = ort_session.run(None, {'input': dummy_input})
        print(f"\n✓ ONNX Runtime inference successful")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Output sample (first 5 values): {outputs[0][0][:5]}")
        
        # Check if output looks like probabilities (sums to ~1.0)
        output_sum = np.sum(outputs[0][0])
        print(f"  Sum of outputs: {output_sum:.6f} {'(probabilities ✓)' if abs(output_sum - 1.0) < 0.01 else '(logits)'}")
        
        return True
    except Exception as e:
        print(f"ERROR: ONNX Runtime inference failed: {e}")
        return False


def load_trained_model(checkpoint_path, model_class, config, two_stream=False):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        model_class: Model class to instantiate
        config: ExportConfig object
        two_stream: Whether this is a two-stream model
    """
    print(f"Loading model from: {checkpoint_path}")
    
    # Model arguments
    model_args = {
        'in_channels': config.C,
        'num_class': config.num_class,
        'graph_args': {
            'layout': config.layout,
            'strategy': 'spatial',
            'max_hop': 1,
            'dilation': 1
        },
        'edge_importance_weighting': True,
        'dropout': 0.5
    }
    
    # Initialize model
    model = model_class(**model_args)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"WARNING: Issue loading state dict: {e}")
        print("Trying to load with strict=False...")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model


def export_variant(checkpoint_path, output_dir, config, include_softmax=True):
    """
    Export a specific model variant
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        output_dir: Directory to save exported models
        config: ExportConfig object
        include_softmax: Whether to include softmax in exported model (default: True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Exporting TWO-STREAM model")
    print(f"Configuration: {config}")
    print(f"{'='*60}")
    
    # Import two-stream model class
    from net.st_gcn_twostream_trt import Model
    
    # Load model
    model = load_trained_model(checkpoint_path, Model, config, two_stream=True)
    
    # Output filename
    model_name = f"stgcn_two_stream_{config}"
    if include_softmax:
        model_name += "_softmax"
    onnx_path = output_dir / f"{model_name}.onnx"
    
    # Export to ONNX
    success = export_to_onnx(model, config, str(onnx_path), batch_size=1, include_softmax=include_softmax)
    
    if success:
        # Verify ONNX
        verify_onnx(str(onnx_path), config, batch_size=1)
        
        # Save configuration for later reference
        config_path = output_dir / f"{model_name}_config.txt"
        with open(config_path, 'w') as f:
            f.write(f"Model Type: two_stream\n")
            f.write(f"Input Shape: (N, {config.C}, {config.T}, {config.V}, {config.M})\n")
            f.write(f"Output Shape: (N, {config.num_class})\n")
            f.write(f"Output Type: {'probabilities (with softmax)' if include_softmax else 'logits (raw scores)'}\n")
            f.write(f"Layout: {config.layout}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
        
        print(f"\n✓ Export complete! Files saved in: {output_dir}")
        print(f"  - ONNX model: {onnx_path}")
        print(f"  - Config: {config_path}")
        
        return str(onnx_path)
    else:
        print(f"\n✗ Export failed!")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export ST-GCN Two-Stream Models to ONNX')
    
    # Variant selection
    parser.add_argument('--variant', type=str, required=True,
                       choices=['variant1', 'variant2'],
                       help='Which variant to export (variant1: M=1, variant2: M=5)')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    
    parser.add_argument('--output-dir', type=str, default='./exported_models',
                       help='Output directory for exported models')
    
    parser.add_argument('--num-class', type=int, default=2,
                       help='Number of action classes (default: 2)')
    
    parser.add_argument('--layout', type=str, default='mediapipe',
                       choices=['openpose', 'ntu-rgb+d', 'mediapipe'],
                       help='Skeleton layout (mediapipe for 33 joints, openpose for 18, ntu-rgb+d for 25)')
    
    parser.add_argument('--no-softmax', action='store_true',
                       help='Export without softmax layer (output raw logits instead of probabilities)')
    
    args = parser.parse_args()
    
    # Define configurations for your two variants
    if args.variant == 'variant1':
        config = ExportConfig(M=1, T=150, C=3, V=33, 
                            num_class=args.num_class, 
                            layout=args.layout)
    else:  # variant2
        config = ExportConfig(M=5, T=150, C=3, V=33, 
                            num_class=args.num_class,
                            layout=args.layout)
    
    # Export
    onnx_path = export_variant(
        args.checkpoint,
        args.output_dir,
        config,
        include_softmax=not args.no_softmax
    )



if __name__ == '__main__':
    # Example usage without argparse:
    # Uncomment and modify as needed
    
    # # Variant 1: M=1, T=150, C=3, V=33, num_class=2 (Single person, two-stream)
    # config1 = ExportConfig(M=1, T=150, C=3, V=33, num_class=2, layout='mediapipe')
    # export_variant(
    #     checkpoint_path='models/single_model.pt',
    #     output_dir='./exported_models/variant1',
    #     config=config1
    # )
    
    # # Variant 2: M=5, T=150, C=3, V=33, num_class=2 (Multi person, two-stream)
    # config2 = ExportConfig(M=5, T=150, C=3, V=33, num_class=2, layout='mediapipe')
    # export_variant(
    #     checkpoint_path='models/multi_model.pt',
    #     output_dir='./exported_models/variant2',
    #     config=config2
    # )
    
    main()