"""从 ExecutionMode 的 workspace 物化 (model, tokenizer) 的公共加载器。

每个 ExecutionMode 都是一个自包含、可运行的目录，通过 entrypoint（默认
build_model.py）暴露统一入口：

    build_model() -> (model, tokenizer)

无论 workspace 里嵌的是哪种优化（forward patch / 算子融合 / 量化 / ...），
correctness / profile / benchmark 都**只**通过这个入口加载模型——这是全项目
唯一的模型真相源。加载逻辑本身与 profiling/benchmark 无关，所以独立成模块，
避免各功能去 import profile_runner.py 的私有实现。

注意：本模块不依赖 torch，只负责 import workspace 的 build_model.py 并调用它；
具体 device / dtype 由 build_model() 自身决定。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from models import ExecutionMode


def load_build_model(mode: ExecutionMode) -> tuple[Any, Any]:
    """加载 mode.workspace_dir 的 build_model()，返回 (model, tokenizer)。

    Args:
        mode: 一个自包含可运行的 ExecutionMode；其 entrypoint 必须暴露
              build_model() -> (model, tokenizer)。

    Raises:
        FileNotFoundError: entrypoint 文件不存在。
        ImportError:       entrypoint 无法作为模块加载。
        AttributeError:    entrypoint 未暴露 build_model()。
    """
    ws = Path(mode.workspace_dir).resolve()
    entry = ws / mode.entrypoint
    if not entry.is_file():
        raise FileNotFoundError(f"entrypoint not found: {entry}")

    # 隔离：每个 fork 的 build_model.py 可能 import 同名辅助文件（patches/config/...）。
    # 若把 ws 永久留在 sys.path、把这些裸名模块永久留在 sys.modules，下一个 fork 会
    # 拿到上一个 fork 缓存的同名模块——profile 到错的代码，静默污染"比较延迟"本身。
    # 因此进来前快照、用完后还原：只清掉本次 import 新引入的裸名模块与本次加进的路径。
    path_added = str(ws) not in sys.path
    if path_added:
        sys.path.insert(0, str(ws))
    modules_before = set(sys.modules)
    try:
        spec = importlib.util.spec_from_file_location(f"_mode_entry_{ws.name}", entry)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load entrypoint: {entry}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "build_model"):
            raise AttributeError(f"{entry} does not expose build_model()")
        model, tokenizer = module.build_model()
        return model, tokenizer
    finally:
        # 还原：删掉"定位落在本 workspace 内"的新缓存模块（fork 自带的入口与裸名
        # 辅助包 patches/config/...），让下一个 fork 不会拿到本 fork 的同名模块。
        # 关键：既看 __file__（普通模块/有 __init__.py 的包），也看 __path__（包，
        # 含缺 __init__.py 的 namespace package——它 __file__ 为 None，只有 __path__
        # 指向 ws）。只看 __file__ 会漏掉 namespace package，使其永久残留：下一个
        # fork 的 `from patches import X` 会命中这个陈旧缓存——静默跑错代码，或
        # 报 `(unknown location)` ImportError。
        # 刻意不动 transformers_modules.* 等 trust_remote_code 动态模块——它们的定位
        # 在 HF 缓存目录、不在 ws 内，刚建好的 model 仍依赖它们留在 sys.modules。
        for name in set(sys.modules) - modules_before:
            mod = sys.modules.get(name)
            # 关键：从 mod.__dict__ 直接取，不用 getattr。某些惰性模块（如
            # torch_npu.dynamo）定义了 __getattr__，getattr(mod, "__path__") 会
            # 惊动它触发子 import（torchair → pkg_resources）从而抛错。__dict__
            # 是普通字典读取，只看模块"已实际设置"的属性，无副作用。
            mod_dict = getattr(mod, "__dict__", None)
            if not mod_dict:
                continue
            locations = []
            mod_file = mod_dict.get("__file__")
            if mod_file:
                locations.append(mod_file)
            # __path__ 多数是路径字符串的可迭代对象，但有些库会把非标准对象塞进
            # __path__（如 torch_npu 的 _ClassNamespace 不可迭代）。安全收集：迭代
            # 失败或元素非字符串都跳过，绝不让清理逻辑因这种模块整体抛错。
            mod_path = mod_dict.get("__path__")
            if mod_path is not None:
                try:
                    locations.extend(p for p in mod_path if isinstance(p, (str, bytes)))
                except TypeError:
                    pass
            for loc in locations:
                try:
                    if Path(loc).resolve().is_relative_to(ws):
                        del sys.modules[name]
                        break
                except (ValueError, OSError, TypeError):
                    continue
        if path_added and str(ws) in sys.path:
            sys.path.remove(str(ws))
