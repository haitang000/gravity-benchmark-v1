from __future__ import annotations

import platform
from typing import Any


def detect_hardware() -> dict[str, Any]:
    result: dict[str, Any] = {"platform": platform.platform(), "accelerator": "cpu"}
    try:
        import torch
        result.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "hip": getattr(torch.version, "hip", None)})
        if torch.cuda.is_available():
            result["accelerator"] = "rocm" if torch.version.hip else "cuda"
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        result["torch"] = None
    return result
