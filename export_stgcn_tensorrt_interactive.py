#!/usr/bin/env python3
"""
Interactive ST-GCN TensorRT Export Script

Allows user to choose:
- Model type (single-person M=1 or multi-person M=5)
- Temporal sequence length (T)
- Precision (FP32, FP16, INT8)
- Workspace size
- Other TensorRT optimization options
"""

import tensorrt as trt
import os
import argparse
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_user_choice(prompt, choices, default=None):
    """Get user choice from a list of options."""
    while True:
        print(f"\n{prompt}")
        for i, choice in enumerate(choices, 1):
            marker = " [DEFAULT]" if choice == default else ""
            print(f"  [{i}] {choice}{marker}")
        
        user_input = input(f"\nEnter choice (1-{len(choices)}): ").strip()
        
        if not user_input and default:
            return default
        
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            else:
                print(f"ERROR: Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("ERROR: Please enter a valid number")

def get_integer_input(prompt, default=None, min_val=None, max_val=None):
    """Get integer input from user with validation."""
    while True:
        default_str = f" (default: {default})" if default else ""
        user_input = input(f"{prompt}{default_str}: ").strip()
        
        if not user_input and default is not None:
            return default
        
        try:
            value = int(user_input)
            if min_val is not None and value < min_val:
                print(f"ERROR: Value must be at least {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"ERROR: Value must be at most {max_val}")
                continue
            return value
        except ValueError:
            print("ERROR: Please enter a valid integer")

def get_boolean_input(prompt, default=False):
    """Get yes/no input from user."""
    default_str = "Y/n" if default else "y/N"
    while True:
        user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
        
        if not user_input:
            return default
        
        if user_input in ['y', 'yes']:
            return True
        elif user_input in ['n', 'no']:
            return False
        else:
            print("ERROR: Please enter 'y' or 'n'")

def build_engine(onnx_path, engine_path, precision='fp16', workspace_size=4, 
                 min_timing_iterations=1, avg_timing_iterations=8, 
                 max_aux_streams=None, use_dla=False, dla_core=0):
    """
    Build TensorRT engine from ONNX model with customizable options.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
        workspace_size: Workspace size in GB
        min_timing_iterations: Minimum timing iterations for kernel selection
        avg_timing_iterations: Average timing iterations for kernel selection
        max_aux_streams: Maximum auxiliary streams (None = auto)
        use_dla: Use Deep Learning Accelerator (Jetson only)
        dla_core: DLA core to use (0 or 1)
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Building TensorRT Engine")
    print(f"{'='*60}")
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"Precision: {precision.upper()}")
    print(f"Workspace: {workspace_size}GB")
    print(f"Timing iterations: min={min_timing_iterations}, avg={avg_timing_iterations}")
    if use_dla:
        print(f"DLA: Enabled (core {dla_core})")
    print(f"{'='*60}")
    
    # Verify ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}")
        return False
    
    # Get absolute paths and directory
    onnx_path = os.path.abspath(onnx_path)
    onnx_dir = os.path.dirname(onnx_path)
    onnx_filename = os.path.basename(onnx_path)
    
    # Change to ONNX directory so TensorRT can find .data files
    original_dir = os.getcwd()
    os.chdir(onnx_dir)
    
    try:
        # Create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX model
        print("\n[1/6] Parsing ONNX model...")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                return False
        print("SUCCESS: ONNX parsed successfully")
        
        # Configure builder
        print(f"\n[2/6] Configuring builder...")
        config = builder.create_builder_config()
        
        # Set workspace size (TensorRT 8.5+ uses set_memory_pool_limit, older versions use max_workspace_size)
        workspace_bytes = workspace_size * (1 << 30)  # GB to bytes
        try:
            # TensorRT 8.5+ API
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
            print(f"Workspace set to {workspace_size}GB (TensorRT 8.5+ API)")
        except (AttributeError, TypeError):
            # Fallback for older TensorRT versions (< 8.5)
            try:
                config.max_workspace_size = workspace_bytes
                print(f"Workspace set to {workspace_size}GB (TensorRT < 8.5 API)")
            except AttributeError:
                print(f"WARNING: Could not set workspace size, using default")
        
        # Set precision
        print(f"\n[3/6] Setting precision to {precision.upper()}...")
        if precision == 'fp16':
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("SUCCESS: FP16 enabled")
            else:
                print("WARNING: FP16 not supported on this platform, using FP32")
        elif precision == 'int8':
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                print("WARNING: INT8 enabled (calibration dataset required for accuracy)")
            else:
                print("WARNING: INT8 not supported on this platform, using FP32")
        
        # Set timing iterations for kernel selection
        print(f"\n[4/6] Configuring timing iterations...")
        try:
            config.min_timing_iterations = min_timing_iterations
            config.avg_timing_iterations = avg_timing_iterations
            print(f"Timing: min={min_timing_iterations}, avg={avg_timing_iterations}")
        except AttributeError:
            print("WARNING: Timing iterations not supported in this TensorRT version")
        
        # Set auxiliary streams
        if max_aux_streams is not None:
            try:
                config.max_aux_streams = max_aux_streams
                print(f"Auxiliary streams: {max_aux_streams}")
            except AttributeError:
                print("WARNING: Auxiliary streams not supported in this TensorRT version")
        
        # Configure DLA (Deep Learning Accelerator) for Jetson
        if use_dla:
            print(f"\n[5/6] Configuring DLA...")
            try:
                config.default_device_type = trt.DeviceType.DLA
                config.DLA_core = dla_core
                config.set_flag(trt.BuilderFlag.GPU_FALLBACK)  # Fallback to GPU if DLA fails
                print(f"DLA enabled on core {dla_core} with GPU fallback")
            except AttributeError:
                print("WARNING: DLA not available on this platform")
        
        # Build engine
        print(f"\n[6/6] Building TensorRT engine (this may take 5-15 minutes)...")
        try:
            engine = builder.build_engine(network, config)
        except Exception as e:
            print(f"ERROR: Engine build failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        if engine is None:
            print("ERROR: Engine build returned None")
            return False
        
        # Save engine (use absolute path for output)
        if not os.path.isabs(engine_path):
            engine_path = os.path.join(original_dir, engine_path)
        engine_path = os.path.abspath(engine_path)
        
        print(f"\nSaving engine to {engine_path}...")
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        # Print engine info
        print(f"\n{'='*60}")
        print("Engine Information:")
        print(f"{'='*60}")
        print(f"  Input bindings: {engine.num_bindings // 2}")
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = engine.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            print(f"  [{i}] {name}: {shape} ({dtype})")
        
        engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"  Engine size: {engine_size_mb:.2f} MB")
        print(f"  Precision: {precision.upper()}")
        print(f"{'='*60}")
        
        print(f"\nSUCCESS: TensorRT engine built successfully!")
        return True
        
    finally:
        # Restore original directory
        os.chdir(original_dir)

def interactive_mode():
    """Interactive mode - prompts user for all parameters."""
    print("="*60)
    print("ST-GCN TensorRT Export - Interactive Mode")
    print("="*60)
    
    # Model type selection
    model_type = get_user_choice(
        "Select model type:",
        ["Single-person (M=1)", "Multi-person (M=5)"],
        default="Single-person (M=1)"
    )
    M = 1 if "Single" in model_type else 5
    
    # Temporal sequence length
    print(f"\nTemporal sequence length (T):")
    print("  This should match your training SEQ_LEN parameter")
    T = get_integer_input("  Enter T", default=150, min_val=1, max_val=1000)
    
    # Determine ONNX file path
    onnx_filename = f"stgcn_single_correct.onnx" if M == 1 else f"stgcn_multi_correct.onnx"
    onnx_path = f"models/{onnx_filename}"
    
    if not os.path.exists(onnx_path):
        print(f"\nWARNING: ONNX file not found: {onnx_path}")
        onnx_path = input("Enter full path to ONNX file: ").strip()
        if not os.path.exists(onnx_path):
            print(f"ERROR: File not found: {onnx_path}")
            return False
    
    # Precision selection
    precision = get_user_choice(
        "Select precision:",
        ["fp32", "fp16", "int8"],
        default="fp16"
    )
    
    # Workspace size
    print(f"\nWorkspace size (GB):")
    print("  Higher values allow better optimization but require more RAM")
    print("  Recommended: 6-8GB for 12GB RAM Jetson")
    workspace_size = get_integer_input("  Enter workspace size", default=8, min_val=1, max_val=16)
    
    # Timing iterations
    print(f"\nTiming iterations for kernel selection:")
    print("  Higher values = better optimization but slower build time")
    min_timing = get_integer_input("  Minimum timing iterations", default=1, min_val=1, max_val=100)
    avg_timing = get_integer_input("  Average timing iterations", default=8, min_val=1, max_val=100)
    
    # Auxiliary streams
    use_aux_streams = get_boolean_input("  Use auxiliary streams for parallel execution?", default=False)
    max_aux_streams = None
    if use_aux_streams:
        max_aux_streams = get_integer_input("  Maximum auxiliary streams", default=2, min_val=1, max_val=8)
    
    # DLA configuration (Jetson specific)
    use_dla = get_boolean_input("  Use DLA (Deep Learning Accelerator)?", default=False)
    dla_core = 0
    if use_dla:
        dla_core = get_user_choice("  Select DLA core:", ["0", "1"], default="0")
        dla_core = int(dla_core)
    
    # Engine output path
    precision_suffix = precision.upper()
    engine_filename = f"stgcn_{'single' if M == 1 else 'multi'}_{precision_suffix}.engine"
    engine_path = f"models/{engine_filename}"
    
    print(f"\n{'='*60}")
    print("Configuration Summary:")
    print(f"{'='*60}")
    print(f"  Model type: {'Single-person' if M == 1 else 'Multi-person'} (M={M})")
    print(f"  Temporal length: T={T}")
    print(f"  ONNX file: {onnx_path}")
    print(f"  Precision: {precision.upper()}")
    print(f"  Workspace: {workspace_size}GB")
    print(f"  Timing: min={min_timing}, avg={avg_timing}")
    if max_aux_streams:
        print(f"  Auxiliary streams: {max_aux_streams}")
    if use_dla:
        print(f"  DLA: Enabled (core {dla_core})")
    print(f"  Output engine: {engine_path}")
    print(f"{'='*60}")
    
    confirm = get_boolean_input("\nProceed with engine build?", default=True)
    if not confirm:
        print("Build cancelled by user")
        return False
    
    # Build engine
    return build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=precision,
        workspace_size=workspace_size,
        min_timing_iterations=min_timing,
        avg_timing_iterations=avg_timing,
        max_aux_streams=max_aux_streams,
        use_dla=use_dla,
        dla_core=dla_core
    )

def command_line_mode():
    """Command-line mode with all options."""
    parser = argparse.ArgumentParser(
        description='Convert ONNX models to TensorRT engines with full customization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 export_stgcn_tensorrt_interactive.py

  # Command-line: Single-person FP16
  python3 export_stgcn_tensorrt_interactive.py \\
    --onnx models/stgcn_single_correct.onnx \\
    --engine models/stgcn_single_fp16.engine \\
    --precision fp16 \\
    --workspace 8

  # Command-line: Multi-person INT8 with DLA
  python3 export_stgcn_tensorrt_interactive.py \\
    --onnx models/stgcn_multi_correct.onnx \\
    --engine models/stgcn_multi_int8.engine \\
    --precision int8 \\
    --workspace 6 \\
    --use-dla \\
    --dla-core 0
        """
    )
    parser.add_argument('--onnx', type=str, help='Path to ONNX model file')
    parser.add_argument('--engine', type=str, help='Output path for TensorRT engine file')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'],
                       default='fp16', help='Precision mode (default: fp16)')
    parser.add_argument('--workspace', type=int, default=4,
                       help='Workspace size in GB (default: 4)')
    parser.add_argument('--min-timing', type=int, default=1,
                       help='Minimum timing iterations (default: 1)')
    parser.add_argument('--avg-timing', type=int, default=8,
                       help='Average timing iterations (default: 8)')
    parser.add_argument('--max-aux-streams', type=int, default=None,
                       help='Maximum auxiliary streams (default: None/auto)')
    parser.add_argument('--use-dla', action='store_true',
                       help='Use Deep Learning Accelerator (Jetson only)')
    parser.add_argument('--dla-core', type=int, default=0, choices=[0, 1],
                       help='DLA core to use (default: 0)')
    
    args = parser.parse_args()
    
    # If required args missing, use interactive mode
    if not args.onnx or not args.engine:
        return interactive_mode()
    
    return build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace_size=args.workspace,
        min_timing_iterations=args.min_timing,
        avg_timing_iterations=args.avg_timing,
        max_aux_streams=args.max_aux_streams,
        use_dla=args.use_dla,
        dla_core=args.dla_core
    )

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments - use interactive mode
        success = interactive_mode()
    else:
        # Arguments provided - use command-line mode
        success = command_line_mode()
    
    if not success:
        exit(1)

