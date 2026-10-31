"""Read-only runtime facts for reproducible benchmarks; no model initialization."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path


def environment_record() -> dict:
    packages = {}
    for name in ("torch", "torchvision", "ultralytics", "numpy", "pandas", "pillow",
                 "onnx", "onnxruntime", "fastapi", "uvicorn", "httpx"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu = platform.processor() or None
    info = Path("/proc/cpuinfo")
    if info.exists():
        cpu = next((s.split(":", 1)[1].strip() for s in info.read_text().splitlines()
                    if s.startswith("model name")), cpu)
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = None
    record = {"python": sys.version, "platform": platform.platform(), "cpu_model": cpu,
              "logical_cpus": os.cpu_count(), "host_physical_ram_bytes": ram,
              "packages": packages}
    try:
        import torch
        record["torch_threads"] = torch.get_num_threads()
        record["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        record["torch_threads"] = None
        record["cuda_available"] = None
    return record
