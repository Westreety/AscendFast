"""Repository-wide Python startup hygiene for Ascend/NPU runs.

Python 在仓库根目录位于 sys.path 时会自动 import 本模块。保持轻量：只做两件
Ascend 必需的"启动卫生"，两者缺一这套管线就 import torch 即崩。

1. 关闭 torch_npu 后端自动加载：裸 `import torch` 会触发 torch 自动加载
   torch_npu 后端扩展而崩（RuntimeError: Failed to load the backend
   extension: torch_npu）。设 TORCH_DEVICE_BACKEND_AUTOLOAD=0 让 torch_npu
   按需显式加载，规避此崩溃。
2. 修复 stdlib `profile` 被遮蔽：本仓库有个 profile.py（run_profile 模块），
   会遮蔽标准库的 profile；而 torch._dynamo → cProfile 依赖的是标准库版本，
   冲突时报 "module 'profile' has no attribute 'run'"。这里把标准库 profile.py
   强制装回 sys.modules["profile"]。

刻意不做 autokernel 那套 autokernel_runtime_patches 自动加载——本仓库无此包。
"""
from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _ensure_stdlib_profile_module() -> None:
    current = sys.modules.get("profile")
    current_file = (
        Path(getattr(current, "__file__", "")).resolve()
        if current is not None and getattr(current, "__file__", None)
        else None
    )
    stdlib_profile = Path(sysconfig.get_path("stdlib") or "") / "profile.py"
    if current_file == stdlib_profile and hasattr(current, "run"):
        return
    if not stdlib_profile.exists():
        return
    spec = importlib.util.spec_from_file_location("profile", stdlib_profile)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["profile"] = module
    spec.loader.exec_module(module)


_ensure_stdlib_profile_module()
