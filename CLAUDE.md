# AscendFast — 所有 agent 的项目上下文

> 本项目受 **govctl** 治理：`spec → impl → test → stable`，RFC 至上，不静默偏离。
> 治理产物在 `gov/`，命令用 `govctl`。详见下方「govctl 治理」。

## 硬件 & 运行时

- 设备：Ascend 910 NPU (`npu:0`)
- torch_npu: 2.7.1.post2
- transformers: 4.57.1
- Python: 3.10
- venv: `/models/share/userdata/cb/AscendFast/.venv`
- 命令从项目根目录运行：`/models/share/userdata/cb/AscendFast`

## 环境配置（运行任何命令前必做）

在运行任何 Ascend/NPU Python 命令前，**必须先 source 环境脚本**：

```bash
source scripts/ascend-env.sh
```

它会一次性配好 CANN 运行时变量、项目 venv、以及必需的 torch_npu 启动开关。
未 source 直接跑会因为缺少 CANN/torch_npu 环境而失败。

## transformers 4.57.1 — 可用的 Qwen2 类

```python
# AVAILABLE
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

# NOT AVAILABLE — do not import, do not reference
# Qwen2FlashAttention2   (added in transformers ≥ 4.40, not present here)
# Qwen2SdpaAttention     (added in transformers ≥ 4.40, not present here)
```

## torch_npu fused ops（使用前先检查）

```python
import torch_npu
hasattr(torch_npu, "npu_rms_norm")      # True — safe to use
hasattr(torch_npu, "npu_rotary_mul")    # check at runtime
# Always guard with hasattr() — never assume a fused op exists
```

## build_model.py 的关键 import 规则

所有 `from patches import ...` 语句必须写在 `build_model()` 函数体内，**不能**
放在模块顶层：

```python
# CORRECT
def build_model(...):
    from patches import my_patch   # inside function
    my_patch.apply()
    ...

# WRONG — causes sys.modules pollution across workspace isolation
from patches import my_patch       # top-level
```

这是因为 `workspace_loader.py` 通过在 import 前后快照 `sys.modules` 来隔离每个
workspace 的模块。顶层的 patch import 会在 `exec_module()` 时执行，可能与之前
加载过的某个 workspace 里同名的 `patches` 包发生冲突。

## apply-agent 的强制 smoke test

写完任何代码后，返回 JSON 之前先运行这个：

```bash
cd /models/share/userdata/cb/AscendFast && python -c "
import importlib.util, sys
ws = '<absolute workspace path>'
sys.path.insert(0, ws)
spec = importlib.util.spec_from_file_location('bm', ws + '/build_model.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('smoke OK, build_model:', type(m.build_model))
"
```

若它抛出任何异常，先修代码。不要在 workspace 损坏的情况下返回 JSON。

## 项目布局

```
model/                  # original weights (read-only)
adaptations/            # ExecutionMode workspaces
  <model_id>/
    baseline/           # baseline ExecutionMode
    mode_<...>/         # optimized forks
data/
  prompts_real.jsonl    # profile dataset (simulated, small)
  prompts_sharegpt.jsonl # benchmark dataset (real, 64 samples)
runs/                   # RunLedger JSON files (one per optimization run)
```

## Agent 职责（不要越界）

| Agent | 做什么 | 不做什么 |
|---|---|---|
| profile-agent | 跑 profiler，返回路径 + 延迟 | 解读结果 |
| analysis-agent | 找出时间花在**哪里** | 提出修复方案 |
| strategy-agent | 提出优化**什么**、**为什么**（WHAT/WHY） | 写代码、把 HOW 定死 |
| apply-agent | 写代码 + 验证（HOW） | 发明策略 |

## govctl 治理（本项目受治理）

本项目由 govctl 治理。治理产物（source of truth）在 `gov/` 下：`rfc/ adr/ work/ guard/`。
命令用 **`govctl`** 二进制（已装在 `/usr/local/bin/govctl`）——注意不是 `cargo run`。

核心法则（来自 govctl 治理模型，对本项目同样适用）：

- **RFC 至上**：与 normative RFC 冲突的代码是 bug。不要静默偏离——要么改代码，要么提 RFC 修订。
- **阶段纪律**：`spec → impl → test → stable`，禁止跳过。新功能在 RFC 进入 normative 前不得开始实现。
- **不静默发明**：行为未定义或有歧义时，升级 / 提 issue，不要自己编。
- **产物职责**：RFC 定义「必须满足什么」（normative），ADR 解释「为什么这样选」，Work Item 跟踪「这个任务在干什么」。不要混用。

常用命令：

```bash
govctl status            # 项目概览
govctl check             # 校验所有治理产物（提交前必跑）
govctl search <query>    # 检索治理产物
govctl rfc|adr|work new "Title"
govctl work list
```

> 注意：govctl 仓库自身的 `AGENTS.md`（`/models/share/userdata/cb/govctl/`）是它**作为 Rust 项目的开发指南**，
> 其中 `cargo run`、`src/`、Cargo 等内容**不适用**于 AscendFast，不要照搬。
