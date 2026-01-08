#!/usr/bin/env python3
"""
ST-GCN ONNX Export Script (Interactive)

Exports trained ST-GCN models to ONNX format with customizable parameters.
Supports interactive CLI prompts and command-line arguments.
"""

import torch
import sys
import os
import argparse

# Get script directory and models directory
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models')

# Add project root to path
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from net.st_gcn_twostream import Model as STGCN_TwoStream

def list_available_models(search_dir):
    """
    List available .pt model files in the specified directory.
    
    Args:
        search_dir: Directory to search for .pt files
    
    Returns:
        List of full paths to .pt model files
    """
    models = []
    if os.path.exists(search_dir) and os.path.isdir(search_dir):
        for file in os.listdir(search_dir):
            if file.endswith('.pt'):
                full_path = os.path.join(search_dir, file)
                if os.path.isfile(full_path):
                    models.append(full_path)
    return sorted(models)

def get_user_input(prompt, default=None, input_type=str, choices=None):
    """Get user input with default value and validation."""
    if default is not None:
        if choices:
            prompt_text = f"{prompt} [{choices}] (default: {default}): "
        else:
            prompt_text = f"{prompt} (default: {default}): "
    else:
        prompt_text = f"{prompt}: "
    
    while True:
        user_input = input(prompt_text).strip()
        
        if not user_input and default is not None:
            # Convert default to the requested type
            try:
                return input_type(default)
            except (ValueError, TypeError):
                # If conversion fails, return default as-is (for strings, paths, etc.)
                return default
        
        if not user_input and default is None:
            print("This field is required. Please enter a value.")
            continue
        
        if choices and user_input not in choices:
            print(f"Invalid choice. Please select from: {', '.join(choices)}")
            continue
        
        try:
            return input_type(user_input)
        except ValueError:
            print(f"Invalid input. Please enter a valid {input_type.__name__}.")
            continue

def export_model(
    model_type='single',
    model_path=None,
    output_path=None,
    C=3,
    T=150,
    V=33,
    M=None,
    opset_version=11,
    models_dir=models_dir,
    project_root=project_root
):
    """
    Export ST-GCN model to ONNX with customizable parameters.
    
    Args:
        model_type: 'single' or 'multi'
        model_path: Path to .pt model file
        output_path: Path to save .onnx file
        C: Number of channels (default: 3)
        T: Temporal sequence length (default: 150)
        V: Number of vertices/keypoints (default: 33)
        M: Number of persons (default: 1 for single, 5 for multi)
        opset_version: ONNX opset version (default: 11, recommended for TensorRT compatibility. Opset 12 may cause graph optimizer errors in TensorRT)
        models_dir: Directory containing models
        project_root: Project root directory
    """
    print("=" * 60)
    print(f"EXPORTING {model_type.upper()}-PERSON MODEL")
    print("=" * 60)
    
    # Set default M based on model type if not provided
    if M is None:
        M = 1 if model_type == 'single' else 5
    
    # Initialize model (EXACT architecture from training)
    model = STGCN_TwoStream(
        in_channels=C,
        num_class=2,
        graph_args={'layout': 'mediapipe', 'strategy': 'spatial'},
        edge_importance_weighting=True
    )
    
    # Validate model path
    if model_path is None:
        print("ERROR: Model path is required")
        return None
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Create dummy input with specified dimensions
    dummy_input = torch.randn(1, C, T, V, M)
    
    print(f"\nInput dimensions:")
    print(f"  C (channels): {C}")
    print(f"  T (temporal): {T}")
    print(f"  V (vertices): {V}")
    print(f"  M (persons): {M}")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Model type: {type(model).__name__}")
    
    # Verify forward pass works
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"\nTest output shape: {test_output.shape} (expected: (1, 2))")
        if test_output.shape != (1, 2):
            print(f"WARNING: Unexpected output shape")
    
    # Determine output path if not provided
    if output_path is None:
        # Generate output filename from input model name
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        output_filename = f'{model_basename}.onnx'
        output_path = os.path.join(models_dir, output_filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nExporting to: {output_path}")
    print(f"ONNX opset version: {opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        export_params=True,
        dynamic_axes=None  # Static shapes only for TensorRT
    )
    
    print(f"SUCCESS: Exported to {output_path}")
    return output_path

def interactive_mode():
    """Run interactive mode with CLI prompts."""
    print("\n" + "=" * 60)
    print("ST-GCN ONNX Export (Interactive Mode)")
    print("=" * 60)
    print("Press Enter to use default values shown in brackets\n")
    
    # Ask for models directory
    default_models_dir = os.path.join(script_dir, 'models')
    models_dir_input = get_user_input(
        "Models directory path (where .pt files are located)",
        default=default_models_dir
    )
    
    # Expand user home directory if needed
    models_dir_input = os.path.expanduser(models_dir_input)
    
    if not os.path.exists(models_dir_input):
        print(f"ERROR: Directory not found: {models_dir_input}")
        return
    
    if not os.path.isdir(models_dir_input):
        print(f"ERROR: Not a directory: {models_dir_input}")
        return
    
    # List available models
    available_models = list_available_models(models_dir_input)
    num_models = len(available_models)
    
    if num_models == 0:
        print(f"\nNo .pt model files found in: {models_dir_input}")
        print("Please check the directory path and try again.")
        return
    
    print(f"\nFound {num_models} model file(s):")
    for i, model_path in enumerate(available_models, 1):
        model_name = os.path.basename(model_path)
        print(f"  [{i}] {model_name}")
        print(f"      {model_path}")
    
    # Let user select which models to export
    print("\nWhich model(s) would you like to export?")
    print("  Enter model numbers separated by commas (e.g., 1,2,3)")
    print("  Or enter 'all' to export all models")
    
    selection_input = get_user_input(
        "Selection",
        default="all"
    ).strip().lower()
    
    # Parse selection
    if selection_input == 'all':
        selected_indices = list(range(1, num_models + 1))
    else:
        try:
            selected_indices = [int(x.strip()) for x in selection_input.split(',')]
            # Validate indices
            invalid = [idx for idx in selected_indices if idx < 1 or idx > num_models]
            if invalid:
                print(f"ERROR: Invalid model numbers: {invalid}")
                return
        except ValueError:
            print("ERROR: Invalid input. Please enter numbers separated by commas or 'all'.")
            return
    
    # Export selected models
    results = []
    
    for idx in selected_indices:
        model_path = available_models[idx - 1]
        model_name = os.path.basename(model_path)
        model_basename = os.path.splitext(model_name)[0]
        
        print("\n" + "=" * 60)
        print(f"CONFIGURING MODEL: {model_name}")
        print("=" * 60)
        
        # Ask for model type (single or multi-person)
        print("\nWhat type of model is this?")
        model_type_choice = get_user_input(
            "  [1] Single-person (M=1)\n  [2] Multi-person (M=5)\n  Enter choice",
            default="1",
            choices=["1", "2"]
        )
        
        model_type = 'single' if model_type_choice == "1" else 'multi'
        default_M = 1 if model_type == 'single' else 5
        
        # Get output filename
        default_output_name = f"{model_basename}.onnx"
        output_filename = get_user_input(
            "Output filename (.onnx)",
            default=default_output_name
        )
        
        # Ensure .onnx extension
        if not output_filename.endswith('.onnx'):
            output_filename += '.onnx'
        
        output_path = os.path.join(models_dir_input, output_filename)
        
        # Get model dimensions
        print("\nModel dimensions:")
        C = get_user_input("  C (channels)", default="3", input_type=int)
        T = get_user_input("  T (temporal sequence length)", default="150", input_type=int)
        V = get_user_input("  V (vertices/keypoints)", default="33", input_type=int)
        M = get_user_input("  M (number of persons)", default=str(default_M), input_type=int)
        opset_version = get_user_input("  ONNX opset version (11 recommended for TensorRT)", default="11", input_type=int)
        
        # Export the model
        print("\n")
        result_path = export_model(
            model_type=model_type,
            model_path=model_path,
            output_path=output_path,
            C=C,
            T=T,
            V=V,
            M=M,
            opset_version=opset_version,
            models_dir=models_dir_input,
            project_root=project_root
        )
        
        results.append({
            'model': model_name,
            'path': result_path,
            'success': result_path is not None
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r['success'])
    
    for result in results:
        if result['success']:
            print(f"SUCCESS: {result['model']} -> {result['path']}")
        else:
            print(f"ERROR: {result['model']} export failed")
    
    if success_count == len(results):
        print(f"\nAll {success_count} model(s) exported successfully!")
        print("\nNext steps:")
        print("1. Validate ONNX models: python validate_onnx.py")
        print("2. Convert to TensorRT: python export_stgcn_tensorrt.py")
    else:
        print(f"\n{success_count}/{len(results)} model(s) exported successfully.")
    
    print("=" * 60)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Export ST-GCN models to ONNX format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python export_stgcn_onnx.py
  
  # Export single-person model with custom parameters
  python export_stgcn_onnx.py --type single --T 150 --M 1 --model models/epoch60_model.pt
  
  # Export multi-person model with custom parameters
  python export_stgcn_onnx.py --type multi --T 150 --M 5 --model models/epoch50_model.pt
        """
    )
    
    parser.add_argument('--type', choices=['single', 'multi', 'both'], default=None,
                        help='Model type to export (default: interactive mode)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to .pt model file')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output .onnx file')
    parser.add_argument('--C', type=int, default=3,
                        help='Number of channels (default: 3)')
    parser.add_argument('--T', type=int, default=150,
                        help='Temporal sequence length (default: 150)')
    parser.add_argument('--V', type=int, default=33,
                        help='Number of vertices/keypoints (default: 33)')
    parser.add_argument('--M', type=int, default=None,
                        help='Number of persons (default: 1 for single, 5 for multi)')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version (default: 11, recommended for TensorRT)')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # If no arguments provided, run interactive mode
    if args.type is None:
        interactive_mode()
    else:
        # Command-line mode
        print("\n" + "=" * 60)
        print("ST-GCN ONNX Export (Command-Line Mode)")
        print("=" * 60)
        
        results = {}
        
        if args.type in ['single', 'both']:
            M = args.M if args.M is not None else 1
            results['single'] = export_model(
                model_type='single',
                model_path=args.model,
                output_path=args.output if args.type == 'single' else None,
                C=args.C,
                T=args.T,
                V=args.V,
                M=M,
                opset_version=args.opset
            )
        
        if args.type in ['multi', 'both']:
            M = args.M if args.M is not None else 5
            output_path = args.output if args.type == 'multi' else None
            if args.type == 'both' and args.output:
                # For 'both', modify output path for multi-person
                base, ext = os.path.splitext(args.output)
                output_path = f"{base}_multi{ext}"
            
            results['multi'] = export_model(
                model_type='multi',
                model_path=args.model,
                output_path=output_path,
                C=args.C,
                T=args.T,
                V=args.V,
                M=M,
                opset_version=args.opset
            )
        
        # Summary
        print("\n" + "=" * 60)
        print("EXPORT SUMMARY")
        print("=" * 60)
        
        for model_type, path in results.items():
            if path:
                print(f"SUCCESS: {model_type}-person model exported to {path}")
            else:
                print(f"ERROR: {model_type}-person model export failed")
        
        print("=" * 60)
