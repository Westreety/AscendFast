"""把 add_demo 的 PyTorch 适配层 (adapter_add_demo.cpp) 即时编成扩展 .so。

路线 A：adapter 手写 aclnn 两段式，只依赖公开头。本脚本负责把它和
torch_npu / CANN / 本地编出的 libcust_opapi.so 链在一起。

用法（必须先 source CANN 环境，让 ASCEND_TOOLKIT_HOME 生效）：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    .venv/bin/python kernels/csrc/build_adapter.py

import 返回的扩展后，torch.ops.ascendfast.add_demo 即注册进 PyTorch。
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# 算子库工程(ascendfast_custom_ops:msopgen -lan cpp 生成的标准工程,所有自定义
# 算子都加在这一个工程里)的编译产物目录。新增算子不需要改这里。
_OPS_BUILD = _HERE.parent / "ascendc_ops" / "ascendfast_custom_ops" / "build_out"


def _cann_home() -> Path:
    home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        raise RuntimeError(
            "ASCEND_TOOLKIT_HOME 未设置。先 source "
            "/usr/local/Ascend/ascend-toolkit/set_env.sh"
        )
    return Path(home)


def _torch_npu_dir() -> Path:
    import torch_npu

    return Path(torch_npu.__file__).resolve().parent


def _adapter_op_name(src: Path) -> str:
    """adapter_<op>.cpp -> <op>。算子名同时用于 .so 名和 aclnn 头名。"""
    return src.stem[len("adapter_"):]


def _aclnn_header_name(op: str) -> str:
    """<op> -> aclnn_<op>.h。autogen 的 aclnn 头按算子名（小写下划线）命名。"""
    return f"aclnn_{op}.h"


def build():
    from torch.utils.cpp_extension import load, _get_build_directory
    import shutil

    cann = _cann_home()
    tnpu = _torch_npu_dir()

    # 工程 build.sh 的产物布局:aclnn 头在 autogen/,host 库（所有算子共用一份
    # libcust_opapi.so）在 op_host/。
    aclnn_inc = _OPS_BUILD / "autogen"
    opapi_lib = _OPS_BUILD / "op_host"
    if not (opapi_lib / "libcust_opapi.so").exists():
        raise RuntimeError(
            f"缺少算子 host 库：{opapi_lib / 'libcust_opapi.so'}"
            f"（先在 ascendfast_custom_ops/ 下 build.sh）"
        )

    include_dirs = [
        str(tnpu / "include"),
        str(tnpu / "include" / "third_party" / "acl" / "inc"),
        str(cann / "include"),
        str(cann / "include" / "aclnn"),
        str(aclnn_inc),
    ]
    library_dirs = [
        str(tnpu / "lib"),
        str(cann / "lib64"),
        str(opapi_lib),
    ]
    # cust_opapi: 本地算子的 host 入口（aclnn<Op>*，整个工程一份）。
    # ascendcl/nnopbase: aclCreateTensor / aclnn 执行器底座。
    # torch_npu: getCurrentNPUStream / NPUWorkspaceAllocator。
    libraries = ["torch_npu", "cust_opapi", "ascendcl", "nnopbase"]

    lib_dir = _HERE.parent / "src" / "ascendfast_ops" / "lib"
    lib_dir.mkdir(exist_ok=True)

    # 遍历 csrc/adapter_*.cpp，每个算子各编一个 .so。加新算子只要丢一个
    # adapter_<op>.cpp 进来（且 B 链已编出对应 aclnn_<op>.h），这里无需改动。
    adapters = sorted(_HERE.glob("adapter_*.cpp"))
    if not adapters:
        raise RuntimeError(f"未找到任何 adapter_*.cpp（在 {_HERE}）")

    targets = []
    for src in adapters:
        op = _adapter_op_name(src)
        header = aclnn_inc / _aclnn_header_name(op)
        if not header.exists():
            raise RuntimeError(
                f"缺少算子 {op} 的 aclnn 头：{header}"
                f"（先在 ascendfast_custom_ops/ 下 build.sh 生成它）"
            )
        name = f"ascendfast_adapter_{op}"
        # is_python_module=False：adapter 不是 Python 模块（没有 PYBIND11_MODULE），
        # 只靠 TORCH_LIBRARY/_FRAGMENT 在加载时注册算子。load 返回 None，产物 .so 落在
        # torch 的扩展缓存目录里。
        load(
            name=name,
            sources=[str(src)],
            extra_include_paths=include_dirs,
            extra_ldflags=(
                [f"-L{d}" for d in library_dirs]
                + [f"-Wl,-rpath,{d}" for d in library_dirs]
                + [f"-l{l}" for l in libraries]
            ),
            is_python_module=False,
            verbose=True,
        )
        build_dir = Path(_get_build_directory(name, verbose=False))
        so_path = build_dir / f"{name}.so"
        if not so_path.exists():
            raise RuntimeError(f"编译产物未找到: {so_path}")
        target = lib_dir / so_path.name
        shutil.copy2(so_path, target)
        print(f"[build_adapter] built and installed: {target}")
        targets.append(target)

    return targets


if __name__ == "__main__":
    build()
