#!/usr/bin/env python3
import ctypes
import os
import platform
import subprocess
import sys


def print_section(title):
    """Print a section header"""
    print(f"\n{'=' * 80}")
    print(f"   {title}")
    print(f"{'=' * 80}")


def check_libraries(libraries):
    """Check if libraries can be loaded"""
    results = {}
    for lib_name in libraries:
        try:
            ctypes.CDLL(lib_name)
            results[lib_name] = "✅ LOADED"
        except Exception as e:
            results[lib_name] = f"❌ FAILED: {str(e)}"
    return results


def run_command(cmd, timeout=2):
    """Run shell command and return output with timeout"""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except subprocess.CalledProcessError as e:
        return f"Error (exit code {e.returncode}): {e.output}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_memory_info():
    """Get total memory information"""
    with open("/proc/meminfo", "r") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_kb = int(line.split()[1])
                mem_gb = mem_kb / (1024 * 1024)
                return mem_gb
    raise Exception("Failed to read memory info")


def get_cpu_info():
    """Get detailed CPU information"""
    cpu_info = {}

    try:
        # Get core count
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
            processors = content.count("processor")
            cpu_info["core_count"] = processors

        # Try to determine CPU types
        cpu_types = {}
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "CPU part" in line:
                        part = line.strip().split(": ")[1]
                        cpu_types[part] = cpu_types.get(part, 0) + 1

            # Map CPU parts to known CPU types
            cpu_type_map = {
                "0x0a0": "Cortex-A35",
                "0x0d0": "Cortex-A77",
                "0x0d1": "Cortex-A78",
                "0x0d2": "Cortex-A78AE",  # Used in Orin
                "0x0d4": "Cortex-A78C",
                "0xd40": "Cortex-A76",
                "0xd41": "Cortex-A77",
                "0xd42": "Cortex-A78",
                "0xd43": "Cortex-A65AE",
                "0x4e0": "Carmel",  # Used in Xavier
            }

            cpu_models = {}
            for part, count in cpu_types.items():
                model = cpu_type_map.get(part, f"Unknown CPU (part: {part})")
                cpu_models[model] = count

            cpu_info["cpu_types"] = cpu_models

        except Exception as e:
            cpu_info["cpu_types_error"] = str(e)

    except Exception as e:
        return f"Error reading CPU info: {e}"

    return cpu_info


def get_jetson_info():
    """Get detailed information about Jetson hardware"""
    info = {}

    # Get SoC model from device tree
    if os.path.exists("/proc/device-tree/compatible"):
        with open("/proc/device-tree/compatible", "rb") as f:
            compat = f.read().strip(b"\x00").decode("utf-8", errors="ignore")
            info["compatible"] = compat

            # Parse chip information
            if "tegra234" in compat:
                info["soc"] = "Tegra234 (Orin)"

                # Determine Orin variant
                if "p3767" in compat:
                    mem_gb = get_memory_info()
                    if mem_gb > 14:
                        info["som_variant"] = "Orin NX 16GB (estimated from memory size)"
                    elif mem_gb > 7:
                        info["som_variant"] = "Orin NX/Nano 8GB (estimated from memory size)"
                    else:
                        info["som_variant"] = "Orin Nano 4GB (estimated from memory size)"
                elif "p3701" in compat:
                    info["som_variant"] = "Orin AGX"

            elif "tegra194" in compat:
                info["soc"] = "Tegra194 (Xavier)"

                if "p2888" in compat or "galen" in compat:
                    info["som_variant"] = "Xavier AGX"
                elif "p3668" in compat:
                    if "p3509" in compat:
                        info["som_variant"] = "Xavier NX"
                    else:
                        info["som_variant"] = "Xavier NX (eMMC)"
            elif "tegra210" in compat:
                info["soc"] = "Tegra210 (TX1/Nano)"
            else:
                info["soc"] = "Unknown Tegra SoC"

    # Get total memory
    mem_gb = get_memory_info()
    if mem_gb:
        info["total_memory"] = f"{mem_gb:.2f} GB"

    # Get CPU information
    cpu_info = get_cpu_info()
    if isinstance(cpu_info, dict):
        info.update(cpu_info)
    else:
        info["cpu_info_error"] = cpu_info

    # Get hardware model
    if os.path.exists("/proc/device-tree/model"):
        with open("/proc/device-tree/model", "rb") as f:
            info["model"] = f.read().strip(b"\x00").decode("utf-8", errors="ignore")

    # Get serial number if available
    if os.path.exists("/proc/device-tree/serial-number"):
        with open("/proc/device-tree/serial-number", "rb") as f:
            info["serial_number"] = f.read().strip(b"\x00").decode("utf-8", errors="ignore")

    # Get machine info from DMI if available
    if os.path.exists("/sys/class/dmi/id/product_name"):
        with open("/sys/class/dmi/id/product_name", "r") as f:
            info["product_name"] = f.read().strip()

    # Get L4T version
    if os.path.exists("/etc/nv_tegra_release"):
        with open("/etc/nv_tegra_release", "r") as f:
            release_info = f.read().strip()
            info["tegra_release"] = release_info

            # Parse L4T version
            if "R" in release_info and "REVISION" in release_info:
                l4t_version = release_info.split("R")[1].split()[0]
                info["l4t_version"] = l4t_version

    return info


def main():
    print_section("SYSTEM INFORMATION")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"OS: {platform.system()} {platform.release()}")

    # Get CPU info
    cpu_info = run_command("cat /proc/cpuinfo | grep 'model name' | uniq")
    print(f"CPU: {cpu_info.split(':')[1].strip()}")

    print_section("ENVIRONMENT VARIABLES")
    for var in ["LD_LIBRARY_PATH", "CUDA_OVERRIDE_DRIVER_VERSION", "PATH"]:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

    # Check CUDA libraries
    print_section("CUDA LIBRARY CHECK")
    cuda_libs = [
        "libcuda.so.1",  # CUDA driver
        "libcudart.so.11.0",  # CUDA runtime
        "libnvinfer.so.8",  # TensorRT
        "libcudnn.so.8",  # cuDNN
        "libnvonnxparser.so.8",  # ONNX parser for TensorRT
    ]
    lib_results = check_libraries(cuda_libs)
    for lib, result in lib_results.items():
        print(f"{lib}: {result}")

    # Check for NVIDIA devices
    print_section("NVIDIA DEVICES")
    print(run_command("ls -l /dev/nv*"))

    # Try to get CUDA version information
    print_section("CUDA VERSION INFORMATION")
    try:
        cuda = ctypes.CDLL("libcudart.so")
        cuda_version = ctypes.c_int()
        driver_version = ctypes.c_int()

        if hasattr(cuda, "cudaRuntimeGetVersion"):
            result = cuda.cudaRuntimeGetVersion(ctypes.byref(cuda_version))
            if result == 0:  # cudaSuccess
                major = cuda_version.value // 1000
                minor = (cuda_version.value % 1000) // 10
                print(f"CUDA Runtime Version: {major}.{minor}")

        if hasattr(cuda, "cudaDriverGetVersion"):
            result = cuda.cudaDriverGetVersion(ctypes.byref(driver_version))
            if result == 0:  # cudaSuccess
                major = driver_version.value // 1000
                minor = (driver_version.value % 1000) // 10
                print(f"CUDA Driver Version: {major}.{minor}")
    except Exception as e:
        print(f"Error getting CUDA version: {e}")

    # Check NVIDIA modules
    print_section("NVIDIA KERNEL MODULES")
    print(run_command("lsmod | grep -i -E 'nvidia|nv|cuda|tegra'"))

    # Check for TensorRT
    print_section("TENSORRT CHECK")
    try:
        import tensorrt as trt

        print(f"TensorRT version: {trt.__version__}")
        print(f"TensorRT path: {trt.__file__}")
    except ImportError:
        print("TensorRT Python package not found")
    except Exception as e:
        print(f"Error importing TensorRT: {e}")

    # Check for ONNX Runtime
    print_section("ONNX RUNTIME CHECK")
    try:
        import onnxruntime as ort

        print(f"ONNX Runtime version: {ort.__version__}")
        print(f"ONNX Runtime path: {ort.__file__}")
        print(f"Available providers: {ort.get_available_providers()}")

        # Skip the actual inference test as it's too slow
        print("\nProviders are available. Skipping inference test for speed.")
    except ImportError:
        print("ONNX Runtime package not found")
    except Exception as e:
        print(f"Error with ONNX Runtime: {e}")

    # Check for Jetson-specific information
    print_section("JETSON HARDWARE INFORMATION")
    jetson_info = get_jetson_info()

    # Display the hardware information in a readable way
    if "soc" in jetson_info:
        print(f"SoC: {jetson_info['soc']}")

    if "som_variant" in jetson_info:
        print(f"SoM Variant: {jetson_info['som_variant']}")

    if "model" in jetson_info:
        print(f"Model: {jetson_info['model']}")

    if "total_memory" in jetson_info:
        print(f"Total Memory: {jetson_info['total_memory']}")

    if "core_count" in jetson_info:
        print(f"CPU Cores: {jetson_info['core_count']}")

    if "cpu_types" in jetson_info:
        print("CPU Types:")
        for cpu_type, count in jetson_info["cpu_types"].items():
            print(f"  - {cpu_type}: {count} cores")

    if "l4t_version" in jetson_info:
        print(f"L4T Version: {jetson_info['l4t_version']}")

    if os.path.exists("/sys/devices/virtual/dmi/id/bios_version"):
        print("\nBIOS/Firmware version:")
        print(run_command("cat /sys/devices/virtual/dmi/id/bios_version"))

    # Try Jetson-specific tools
    print_section("JETSON GPU STATUS")
    print("Checking Jetson GPU status...")

    # Try nvpmodel (most reliable and quick tool)
    nvpmodel = run_command("nvpmodel -q")
    if "Error" not in nvpmodel and len(nvpmodel.strip()) > 0:
        print("\nnvpmodel output:")
        print(nvpmodel)
    else:
        print("nvpmodel not available or requires root access")

    # Try jetson_clocks (quick status check)
    jetson_clocks = run_command("jetson_clocks --show")
    if "Error" not in jetson_clocks and len(jetson_clocks.strip()) > 0:
        gpu_clock_lines = []
        for line in jetson_clocks.splitlines():
            if "GPU" in line:
                gpu_clock_lines.append(line)

        if gpu_clock_lines:
            print("\nGPU Clock Information:")
            for line in gpu_clock_lines:
                print(line)
        else:
            print("\nNo GPU clock information available")
    else:
        print("jetson_clocks not available or requires root access")

    # Simple test to check if CUDA can initialize
    print_section("CUDA INITIALIZATION TEST")
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        result = cuda.cuInit(0)
        if result == 0:
            print("✅ CUDA initialized successfully")
        else:
            print(f"❌ CUDA initialization failed with error code {result}")

        # Try to get device count
        device_count = ctypes.c_int()
        if hasattr(cuda, "cuDeviceGetCount"):
            result = cuda.cuDeviceGetCount(ctypes.byref(device_count))
            if result == 0:
                print(f"CUDA device count: {device_count.value}")

                # Try to get device name
                if device_count.value > 0:
                    device = ctypes.c_int()
                    result = cuda.cuDeviceGet(ctypes.byref(device), 0)
                    if result == 0:
                        name_buffer = ctypes.create_string_buffer(100)
                        if hasattr(cuda, "cuDeviceGetName"):
                            result = cuda.cuDeviceGetName(name_buffer, 100, device)
                            if result == 0:
                                print(f"CUDA device name: {name_buffer.value.decode('utf-8')}")
    except Exception as e:
        print(f"Error in CUDA initialization test: {e}")
    print("\nDiagnostics completed.")


if __name__ == "__main__":
    main()
