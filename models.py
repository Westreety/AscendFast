from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ChangeRecord:
    """一次优化动作的自描述记录，喂给下一轮 Agent 和人阅读。

    优化方案本身是异构的（算子融合 / 图改写 / forward patch / kvcache /
    并行 / 量化 / 尚未想到的方案），但每一步都收敛成这条统一记录：
    做了什么（summary/details）、动了哪些文件（files）、怎么回退（revert_cmd）。
    """
    mode_uid: str               # 引入本次修改的 ExecutionMode.uid
    strategy_uid: str           # 来源 OptimizationStrategy.uid
    kind: str                   # forward_patch | operator_fusion | graph_rewrite
                                #  | kvcache | parallelism | quantize | config | custom
    summary: str                # 一句话：这一步做了什么
    details: str                # 详细：动了哪些模块/算子、为什么、有何约束
    files: list[str]            # 本步新增/修改的文件（相对 workspace_dir）
    revert_cmd: str | None = None
    metadata: dict | None = None


@dataclass
class OptimizationStrategy:
    uid: str
    local_speedup_ratio: float
    measures: list[str]
    prompt_instruction: str
    extra: dict | None = None


@dataclass
class AnalysisResult:
    uid: str
    total_latency: float
    top_ops: list[str]
    hot_groups: dict[str, list[str]]
    extra: dict | None = None
    model_id: str | None = None
    device_kind: str | None = None
    device_name: str | None = None
    dtype: str | None = None
    profile_report_path: str | None = None
    latency_stats_ms: dict | None = None
    dataset: dict | None = None
    top_kernels: list[dict] = field(default_factory=list)
    op_type_totals: dict[str, dict] = field(default_factory=dict)
    roofline_summary: dict[str, float] = field(default_factory=dict)
    profile_findings: list[str] = field(default_factory=list)


@dataclass
class ExecutionMode:
    """一个自包含、可运行的"模型变体快照"。

    workspace_dir 是 fork 自父 mode 的完整可运行目录；entrypoint 暴露统一入口
    build_model() -> (model, tokenizer)，无论里面是何种优化，correctness/profile
    都只通过这个入口加载，对优化方案本身无知。change_log 是从 root 累积到本步
    的全部修改（append-only），下一轮 apply 会把它注入 Agent 以便叠加优化。
    """
    uid: str
    model_id: str                                   # 基础模型标识
    strategy_uid: str                               # 产生本步的 strategy（baseline 为 "baseline"）
    workspace_dir: str                              # 自包含可运行目录（物化后的优化模型）
    parent_uid: str | None = None                   # 父 mode.uid；None = baseline 根节点
    entrypoint: str = "build_model.py"              # 统一入口文件（相对 workspace_dir）
    change_log: list[ChangeRecord] = field(default_factory=list)
    correctness_passed: bool | None = None
    extra: dict | None = None


@dataclass
class ProfileResult:
    uid: str
    execution_mode_uid: str
    latency_before: float
    latency_after: float
    extra: dict | None = None
    profile_report_path: str | None = None
    profiler_output_dir: str | None = None
    profile_report: dict | None = None
