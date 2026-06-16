# Data entities. LEVER_KINDS is the single source of truth per [[ADR-0005]];
# ExecutionMode + ChangeRecord realize the workspace model of [[ADR-0004]] /
# [[RFC-0001]]; StageOutcome + RunLedger realize the observability layer of [[ADR-0007]].
from __future__ import annotations
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# 优化杠杆（lever）的权威枚举——单一真相源。
# strategy 选 lever、apply 记 kind、ledger 归因都引用这里；新增/改名 lever 只动
# 这一处，其余 Python 文件 import 本常量，不再各自硬编码字符串（文档侧手工对齐）。
#
# 四个 canonical lever 对应 build_model 的四个改动层级（详见 npu-strategy skill）：
#   forward_patch   — monkey-patch 某个 nn.Module.forward（最窄，治单算子）
#   operator_fusion — 改 config / attn_implementation，整条路径切融合后端
#   graph_rewrite   — 在 build_model() 里包整模型（torch.compile / NPU 图模式）
#   loading_time    — 加载期一次性处理（权重 ND→NZ、dtype 清理、静态 KV cache、padding）
# kvcache / quantize / config / parallelism 不是平级 lever：前三者都是 loading_time
# 的子情况，parallelism 单卡 NPU 用不到——不要把它们当独立 kind。
# --------------------------------------------------------------------------- #
LEVER_KINDS = ("forward_patch", "operator_fusion", "graph_rewrite", "loading_time")
# apply 侧合法的 kind 全集：四个 lever + custom（实在无法归类时的兜底，strategy 不用）。
CHANGE_KINDS = LEVER_KINDS + ("custom",)


@dataclass
class ChangeRecord:
    """一次优化动作的自描述记录，喂给下一轮 Agent 和人阅读。

    优化方案本身是异构的（落在四个 lever 之一：forward_patch / operator_fusion /
    graph_rewrite / loading_time，或尚未归类的 custom），但每一步都收敛成这条统一
    记录：做了什么（summary/details）、动了哪些文件（files）、怎么回退（revert_cmd）。
    """
    mode_uid: str               # 引入本次修改的 ExecutionMode.uid
    strategy_uid: str           # 来源 OptimizationStrategy.uid
    kind: str                   # CHANGE_KINDS 之一：forward_patch | operator_fusion
                                #  | graph_rewrite | loading_time | custom
    summary: str                # 一句话：这一步做了什么
    details: str                # 详细：动了哪些模块/算子、为什么、有何约束
    files: list[str]            # 本步新增/修改的文件（相对 workspace_dir）
    revert_cmd: str | None = None
    metadata: dict | None = None


@dataclass
class StageOutcome:
    """一个环节（benchmark/profile/analyze/strategy/apply/correctness/agent_call）
    的成败判定，与 ChangeRecord 同构：每个环节"已经在做但形态不一"的成败判断，
    都收敛成这一条统一记录——做了什么环节（stage）、过没过（ok）、为什么（reason）。

    门禁是喂给 stage() 的纯函数；它们的返回值落到 ok/reason。异常被 stage()
    捕获后也落成一条 ok=False 的 StageOutcome，不再带着 stacktrace 炸穿整条 run。
    """
    stage: str                  # benchmark|profile|analyze|strategy|apply|correctness|agent_call|decision
    ok: bool
    reason: str = ""            # 失败原因；成功时为 ""
    mode_uid: str | None = None # 该环节作用/产出的 ExecutionMode.uid
    metadata: dict | None = None


@dataclass
class RunLedger:
    """一次 optimize() 的 run 级记录：这一次探索了哪棵树、每个环节成败、为什么停。

    mode 级产物（manifest/report）记录单个变体；RunLedger 记录贯穿整条递归的
    决策轨迹。确定性、离线可用，不沾 agent——agent_call 只是 outcomes 里一种
    stage，用来把"为什么没效果"从黑盒里捞出来。
    """
    run_uid: str
    model_id: str
    outcomes: list["StageOutcome"] = field(default_factory=list)
    stop_reason: str | None = None      # reached_2x|max_depth|no_strategies|exhausted|stage_failed:<stage>
    best_mode_uid: str | None = None
    best_latency: float | None = None
    baseline_latency: float | None = None


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
