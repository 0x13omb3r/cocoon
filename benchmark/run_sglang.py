#!/usr/bin/env python3
"""
Start sglang server in Docker with automatic cleanup.

Usage:
    python run_sglang.py                           # Default: tencent/Hunyuan-MT-Chimera-7B
    python run_sglang.py --model tencent/Hunyuan-MT-7B
    python run_sglang.py --port 8001 --tp 2
"""

import argparse
import atexit
import os
import signal
import subprocess
import sys


def download_model(model_name: str):
    """Download model using hf CLI."""
    print(f"[*] Downloading model: {model_name}")
    try:
        subprocess.run(["hf", "download", model_name], check=True)
        print(f"[+] Model downloaded: {model_name}")
    except subprocess.CalledProcessError as e:
        print(f"[!] Failed to download model: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("[!] hf CLI not found. Install with: pip install huggingface_hub[cli]")
        sys.exit(1)


def run_docker(model_name: str, port: int, tp: int, extra_args: list):
    """Run sglang in Docker and return the process."""
    container_name = f"sglang-{model_name.replace('/', '-')}-{port}"
    
    cmd = [
        "docker", "run",
        "--rm",  # Auto-remove container on exit
        f"--name={container_name}",
        "--entrypoint=python3",
        "--gpus", "all",
        "--shm-size", "32g",
        "-p", f"{port}:8000",
        "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        "--ulimit", "nproc=10000",
        "--privileged",
        "--ipc=host",
        "lmsysorg/sglang:latest",
        "-m", "sglang.launch_server",
        "--model-path", model_name,
        "--tp", str(tp),
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", model_name,
        *extra_args
    ]
    
    print(f"[*] Starting Docker container: {container_name}")
    print(f"[*] Command: {' '.join(cmd)}")
    print(f"[*] Server will be available at http://localhost:{port}")
    print("-" * 60)
    
    # Start Docker process
    process = subprocess.Popen(cmd)
    
    # Setup cleanup handlers
    def cleanup():
        print(f"\n[*] Stopping container: {container_name}")
        subprocess.run(["docker", "stop", container_name], 
                      capture_output=True, timeout=30)
        print("[+] Container stopped")
    
    def signal_handler(signum, frame):
        cleanup()
        sys.exit(0)
    
    # Register cleanup on exit and signals
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    return process, container_name


def main():
    parser = argparse.ArgumentParser(description='Run sglang server in Docker')
    parser.add_argument('--model', default='tencent/Hunyuan-MT-Chimera-7B',
                        help='Model name (default: tencent/Hunyuan-MT-Chimera-7B)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Host port (default: 8000)')
    parser.add_argument('--tp', type=int, default=1,
                        help='Tensor parallelism (default: 1)')
    parser.add_argument('--no-download', action='store_true',
                        help='Skip model download')
    parser.add_argument('extra', nargs='*',
                        help='Extra arguments to pass to sglang')
    
    args = parser.parse_args()
    
    # Download model first
    if not args.no_download:
        download_model(args.model)
    
    # Run Docker
    process, container_name = run_docker(args.model, args.port, args.tp, args.extra)
    
    # Wait for process
    try:
        exit_code = process.wait()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        pass  # Cleanup handled by signal handler


if __name__ == "__main__":
    main()

