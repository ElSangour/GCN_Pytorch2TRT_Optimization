#!/usr/bin/env python3
"""
Convert ONNX models to TensorRT engines on Jetson.
Supports FP32, FP16, and INT8 quantization.
"""

import tensorrt as trt
import os
import argparse

# Logger will be configured based on verbosity level
TRT_LOGGER = None

def build_engine(onnx_path, engine_path, precision='fp16', workspace_size=4, disable_optimizations=False, verbose=False, optimization_level=3):
    """
    Build TensorRT engine from ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
        workspace_size: Workspace size in GB (default: 4GB)
        disable_optimizations: Disable certain optimizations to avoid graph optimizer errors
        verbose: Enable verbose logging (VERBOSE level)
        optimization_level: Builder optimization level 0-5 (default: 3, higher=better performance but slower build)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Configure logger based on verbosity
    global TRT_LOGGER
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        print("VERBOSE logging enabled - detailed build information will be displayed")
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    print(f"\n{'='*60}")
    print(f"Building TensorRT Engine")
    print(f"{'='*60}")
    print(f"ONNX: {onnx_path}")
    print(f"Engine: {engine_path}")
    print(f"Precision: {precision.upper()}")
    print(f"Workspace: {workspace_size}GB")
    
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
        print("\n[1/5] Parsing ONNX model...")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse ONNX file")
                for error in range(parser.num_errors):
                    print(f"  {parser.get_error(error)}")
                return False
        print("SUCCESS: ONNX parsed successfully")
        
        # Configure builder
        print(f"\n[2/5] Configuring builder...")
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
        
        # Try to disable problematic optimizations
        try:
            # Disable timing cache which can sometimes cause issues
            config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        except AttributeError:
            pass
        
        # Set profiling verbosity for detailed layer timing information
        try:
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            if verbose:
                print("Profiling verbosity set to DETAILED (enables layer-by-layer timing)")
        except AttributeError:
            if verbose:
                print("WARNING: ProfilingVerbosity not available in this TensorRT version")
        
        # Set builder optimization level (0-5, higher is better but slower to build)
        try:
            config.builder_optimization_level = optimization_level
            print(f"Builder optimization level set to {optimization_level} (0=fastest build, 5=best performance)")
        except AttributeError:
            if verbose:
                print("WARNING: builder_optimization_level not available in this TensorRT version")
        
        # Set precision
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
        
        # Build engine (TensorRT 8.0+ uses build_serialized_network)
        print(f"\n[3/5] Building TensorRT engine (this may take 5-15 minutes)...")
        
        # Save engine path (use absolute path for output)
        if not os.path.isabs(engine_path):
            engine_path = os.path.join(original_dir, engine_path)
        engine_path = os.path.abspath(engine_path)
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        
        try:
            # TensorRT 8.0+ API: build_serialized_network returns serialized engine
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                # If FP16 failed, try automatic fallback to FP32
                if precision == 'fp16':
                    print("\n" + "="*60)
                    print("FP16 BUILD RETURNED None - ATTEMPTING FP32 FALLBACK")
                    print("="*60 + "\n")
                    
                    try:
                        # Create a new config for FP32
                        config_fp32 = builder.create_builder_config()
                        workspace_bytes = workspace_size * (1 << 30)
                        try:
                            config_fp32.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
                        except (AttributeError, TypeError):
                            try:
                                config_fp32.max_workspace_size = workspace_bytes
                            except AttributeError:
                                pass
                        
                        # Update engine path for FP32
                        fp32_engine_path = engine_path.replace('fp16', 'fp32').replace('_fp16', '_fp32')
                        if fp32_engine_path == engine_path:
                            fp32_engine_path = engine_path.replace('.engine', '_fp32.engine')
                        
                        print(f"Retrying with FP32 precision...")
                        print(f"Output will be saved to: {fp32_engine_path}")
                        
                        # Build with FP32
                        serialized_engine = builder.build_serialized_network(network, config_fp32)
                        
                        if serialized_engine is None:
                            print("ERROR: FP32 build also returned None")
                            return False
                        
                        # Update engine_path for saving
                        engine_path = fp32_engine_path
                        print("SUCCESS: FP32 engine built successfully")
                        
                    except Exception as e2:
                        print(f"\nERROR: FP32 fallback failed: {e2}")
                        return False
                else:
                    print("\nERROR: Engine build returned None")
                    print("\nTroubleshooting suggestions:")
                    print("1. Try with different precision (fp32, fp16, int8)")
                    print("2. The ONNX model might have compatibility issues with TensorRT")
                    print("3. Try re-exporting the ONNX model with a different opset version")
                    return False
            
            # Save engine
            print(f"\n[4/5] Saving engine to {engine_path}...")
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
            
            # Deserialize engine to get information (optional, for display)
            print(f"\n[5/5] Engine Information:")
            try:
                runtime = trt.Runtime(TRT_LOGGER)
                engine = runtime.deserialize_cuda_engine(serialized_engine)
                
                if engine:
                    # TensorRT 10.x API: use num_io_tensors instead of num_bindings
                    try:
                        num_tensors = engine.num_io_tensors
                        print(f"  Number of I/O tensors: {num_tensors}")
                        for i in range(num_tensors):
                            name = engine.get_tensor_name(i)
                            shape = engine.get_tensor_shape(name)
                            dtype = engine.get_tensor_dtype(name)
                            mode = engine.get_tensor_mode(name)
                            io_type = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
                            print(f"  [{i}] {name}: {shape} ({dtype}) [{io_type}]")
                    except AttributeError:
                        # Fallback for TensorRT < 10.0
                        num_bindings = engine.num_bindings
                        print(f"  Number of bindings: {num_bindings}")
                        for i in range(num_bindings):
                            name = engine.get_binding_name(i)
                            shape = engine.get_binding_shape(i)
                            dtype = trt.nptype(engine.get_binding_dtype(i))
                            is_input = engine.binding_is_input(i)
                            io_type = "INPUT" if is_input else "OUTPUT"
                            print(f"  [{i}] {name}: {shape} ({dtype}) [{io_type}]")
            except Exception as e:
                print(f"  Could not deserialize engine for info: {e}")
                print(f"  Engine saved successfully, but info display skipped")
            
            engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
            print(f"  Engine size: {engine_size_mb:.2f} MB")
            
        except AttributeError:
            # Fallback for older TensorRT versions (< 8.0) that use build_engine
            try:
                engine = builder.build_engine(network, config)
                if engine is None:
                    print("ERROR: Engine build returned None")
                    return False
                
                print(f"\n[4/5] Saving engine to {engine_path}...")
                with open(engine_path, 'wb') as f:
                    f.write(engine.serialize())
                
                # Print engine info
                print(f"\n[5/5] Engine Information:")
                print(f"  Number of bindings: {engine.num_bindings}")
                for i in range(engine.num_bindings):
                    name = engine.get_binding_name(i)
                    shape = engine.get_binding_shape(i)
                    dtype = trt.nptype(engine.get_binding_dtype(i))
                    is_input = engine.binding_is_input(i)
                    io_type = "INPUT" if is_input else "OUTPUT"
                    print(f"  [{i}] {name}: {shape} ({dtype}) [{io_type}]")
                
                engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
                print(f"  Engine size: {engine_size_mb:.2f} MB")
                
            except Exception as e:
                print(f"ERROR: Engine build failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            error_msg = str(e)
            print(f"\nERROR: Engine build failed: {error_msg}")
            
            # Check if it's a graph optimizer error with FP16
            if ("graphOptimizer" in error_msg or "Internal Error" in error_msg) and precision == 'fp16':
                print("\n" + "="*60)
                print("GRAPH OPTIMIZER ERROR DETECTED WITH FP16")
                print("="*60)
                print("This error often occurs with FP16 precision due to graph optimization issues.")
                print("\nAttempting automatic fallback to FP32...")
                print("="*60 + "\n")
                
                # Try again with FP32
                try:
                    # Create a new config for FP32
                    config_fp32 = builder.create_builder_config()
                    workspace_bytes = workspace_size * (1 << 30)
                    try:
                        config_fp32.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
                    except (AttributeError, TypeError):
                        try:
                            config_fp32.max_workspace_size = workspace_bytes
                        except AttributeError:
                            pass
                    
                    # Update engine path for FP32
                    fp32_engine_path = engine_path.replace('fp16', 'fp32').replace('_fp16', '_fp32')
                    if fp32_engine_path == engine_path:
                        fp32_engine_path = engine_path.replace('.engine', '_fp32.engine')
                    
                    print(f"Retrying with FP32 precision...")
                    print(f"Output will be saved to: {fp32_engine_path}")
                    
                    # Build with FP32
                    serialized_engine = builder.build_serialized_network(network, config_fp32)
                    
                    if serialized_engine is None:
                        print("ERROR: FP32 build also returned None")
                        return False
                    
                    # Save FP32 engine
                    os.makedirs(os.path.dirname(fp32_engine_path), exist_ok=True)
                    with open(fp32_engine_path, 'wb') as f:
                        f.write(serialized_engine)
                    
                    print(f"\nSUCCESS: FP32 engine built and saved to {fp32_engine_path}")
                    print("NOTE: FP16 failed due to graph optimizer issues, but FP32 works.")
                    print("FP32 will be slower but more stable.")
                    
                    # Try to get engine info
                    try:
                        runtime = trt.Runtime(TRT_LOGGER)
                        engine = runtime.deserialize_cuda_engine(serialized_engine)
                        if engine:
                            engine_size_mb = os.path.getsize(fp32_engine_path) / (1024 * 1024)
                            print(f"Engine size: {engine_size_mb:.2f} MB")
                    except:
                        pass
                    
                    return True
                    
                except Exception as e2:
                        error_msg_fp32 = str(e2)
                        print(f"\nERROR: FP32 fallback also failed: {error_msg_fp32}")
                        
                        if "graphOptimizer" in error_msg_fp32 or "Internal Error" in error_msg_fp32:
                            print("\n" + "="*60)
                            print("CRITICAL: ONNX MODEL COMPATIBILITY ISSUE")
                            print("="*60)
                            print("Both FP16 and FP32 failed with the same graph optimizer error.")
                            print("This indicates the ONNX model has compatibility issues with TensorRT.")
                            print("\nSOLUTION: Re-export the ONNX model with opset 11")
                            print("="*60)
                            print("\nSteps to fix:")
                            print("1. On your PC, run the ONNX export script:")
                            print("   python3 export_stgcn_onnx.py")
                            print("\n2. When prompted for 'ONNX opset version', enter: 11")
                            print("   (instead of the default 12)")
                            print("\n3. Transfer the new ONNX file to Jetson and try again")
                            print("="*60)
                        else:
                            print("\nManual solutions:")
                            print("1. Try re-exporting ONNX with opset 11 instead of 12")
                            print("2. Check TensorRT version compatibility")
                            import traceback
                            traceback.print_exc()
                        return False
            else:
                # Not a graph optimizer error or not FP16, show normal error
                print("\nTroubleshooting suggestions:")
                print("1. Try with different precision (fp32, fp16, int8)")
                print("2. The ONNX model might have compatibility issues with TensorRT")
                print("3. Try re-exporting the ONNX model with a different opset version")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\nSUCCESS: TensorRT engine built successfully!")
        return True
        
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert ONNX models to TensorRT engines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Convert single-person model with FP16
                python3 onnx_to_tensorrt.py \\
                    --onnx models/stgcn_single_correct.onnx \\
                    --engine models/stgcn_single_fp16.engine \\
                    --precision fp16 \\
                    --workspace 4

                # Convert multi-person model with FP16
                python3 onnx_to_tensorrt.py \\
                    --onnx models/stgcn_multi_correct.onnx \\
                    --engine models/stgcn_multi_fp16.engine \\
                    --precision fp16 \\
                    --workspace 4
        """
    )
    parser.add_argument('--onnx', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--engine', type=str, required=True,
                       help='Output path for TensorRT engine file')
    parser.add_argument('--precision', type=str, choices=['fp32', 'fp16', 'int8'],
                       default='fp16', help='Precision mode (default: fp16)')
    parser.add_argument('--workspace', type=int, default=4,
                       help='Workspace size in GB (default: 4)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging for detailed build information')
    parser.add_argument('--optimization-level', type=int, default=3, choices=[0, 1, 2, 3, 4, 5],
                       help='Builder optimization level (0-5): 0=fastest build, 5=best runtime performance (default: 3)')
    
    args = parser.parse_args()
    
    success = build_engine(args.onnx, args.engine, args.precision, args.workspace, 
                          verbose=args.verbose, optimization_level=args.optimization_level)
    
    if not success:
        exit(1)

