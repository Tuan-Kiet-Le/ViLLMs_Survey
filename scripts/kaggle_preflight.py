"""Check the Kaggle runtime before downloading model weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

from villms_inference.registry import get_models


def print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def run_nvidia_smi() -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        print("nvidia-smi: unavailable")
        return
    print(result.stdout.strip() or result.stderr.strip() or "nvidia-smi returned no output")


def print_python_environment() -> None:
    print(f"Python: {sys.version.split()[0]}")
    for package in ("torch", "transformers", "accelerate", "huggingface-hub"):
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: NOT INSTALLED")


def print_torch_environment() -> None:
    try:
        import torch
    except ImportError:
        print("torch: unable to import")
        return
    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print(f"Torch CUDA version: {torch.version.cuda}")
    print(f"Visible CUDA devices: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"GPU {index}: {properties.name}; "
            f"VRAM={properties.total_memory / 1024**3:.2f} GiB; "
            f"compute_capability={properties.major}.{properties.minor}"
        )


def print_disk_space(path: Path) -> None:
    usage = shutil.disk_usage(path)
    print(f"Disk ({path}): free={usage.free / 1024**3:.2f} GiB, total={usage.total / 1024**3:.2f} GiB")


def check_model_access() -> None:
    from transformers import AutoConfig

    for model in get_models():
        try:
            config = AutoConfig.from_pretrained(model.model_id, trust_remote_code=model.trust_remote_code)
            print(f"{model.key}: accessible ({config.model_type})")
        except Exception as exc:
            print(f"{model.key}: ERROR {type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-models", action="store_true", help="Download/check model configs, not model weights")
    parser.add_argument("--path", type=Path, default=Path.cwd(), help="Filesystem path for disk-space check")
    args = parser.parse_args()

    print_header("Python packages")
    print_python_environment()
    print_header("nvidia-smi")
    run_nvidia_smi()
    print_header("PyTorch")
    print_torch_environment()
    print_header("Disk space")
    print_disk_space(args.path)
    if args.check_models:
        print_header("Model repository access")
        check_model_access()


if __name__ == "__main__":
    main()
