# strategy-agent role (WHAT/WHY, names a lever, no code) per [[ADR-0006]]; levers come
# from the LEVER_KINDS single source of truth ([[ADR-0005]]); raises rather than
# degrading to an empty list when the agent runtime is unavailable ([[ADR-0008]]).
from __future__ import annotations

import json

from agent_client import AGENT_ENABLED, call_agent_json
from models import LEVER_KINDS, AnalysisResult, OptimizationStrategy


def generate_optimization_strategies(
    analysis: AnalysisResult,
    max_count: int = 5,
) -> list[OptimizationStrategy]:
    """LLM strategy agent (no rule-based fallback).

    Requires the agent runtime to be enabled; raises otherwise so the failure
    is visible instead of silently degrading to an empty strategy list.
    """
    try:
        if not AGENT_ENABLED:
            raise RuntimeError(
                f"strategy agent unavailable: AGENT_ENABLED={AGENT_ENABLED!r}"
            )
        return _llm_generate_optimization_strategies(analysis, max_count) or []
    except Exception as exc:
        print(f"[strategy] generate failed (AGENT_ENABLED={AGENT_ENABLED!r}): {exc}")
        raise


def _llm_generate_optimization_strategies(
    analysis: AnalysisResult,
    max_count: int,
) -> list[OptimizationStrategy] | None:
    top_ops = analysis.top_ops[:10] if analysis.top_ops else []
    prompt = (
        "You are an NPU optimization strategy expert.\n"
        "Given the AnalysisResult summary, generate up to "
        f"{max_count} ranked OptimizationStrategy candidates.\n\n"
        f"analysis_uid: {analysis.uid}\n"
        f"model_id: {analysis.model_id or 'unknown'}\n"
        f"device: {analysis.device_kind} {analysis.device_name}\n"
        f"dtype: {analysis.dtype}\n"
        f"total_latency_ms: {analysis.total_latency:.4f}\n"
        f"top_ops: {top_ops}\n"
        f"op_type_totals: {json.dumps(analysis.op_type_totals, ensure_ascii=False)}\n"
        f"roofline_summary: {json.dumps(analysis.roofline_summary, ensure_ascii=False)}\n"
        f"profile_findings: {json.dumps(analysis.profile_findings or [], ensure_ascii=False)}\n\n"
        "Each strategy must name a lever in `kind` (the build_model layer it touches):\n"
        "  forward_patch   — monkey-patch one nn.Module.forward (narrowest, single op)\n"
        "  operator_fusion — flip a config flag / attn_implementation to a fused backend\n"
        "  graph_rewrite   — wrap the whole model (torch.compile / NPU graph mode)\n"
        "  loading_time    — one-off at load (weight ND→NZ, dtype cleanup, static KV cache, padding)\n"
        "Beyond the official torch_npu.npu_* fused ops, a project-local custom operator\n"
        "library `torch.ops.ascendfast.*` (the kernels/ package) is also available: when a\n"
        "hot op has NO suitable official fused implementation, you MAY propose replacing it\n"
        "with a custom AscendC kernel. This is still a forward_patch or operator_fusion lever\n"
        "(it just swaps in a different op) — say WHAT/WHY only, never the HOW; the apply step\n"
        "decides whether such a kernel exists or must be written.\n"
        "When (and ONLY when) a custom kernel is genuinely warranted — the official op is\n"
        "missing OR a multi-op fusion the official library does not offer (e.g. RMSNorm+residual,\n"
        "RoPE+attention, QKV+bias) would cut launch/cast overhead — attach a `custom_operator`\n"
        "spec to that strategy. Do NOT attach one when an official fused op already covers the\n"
        "hot op well (a hand-written kernel rarely beats a tuned official single-op kernel).\n"
        "The spec is WHAT/WHY for an operator-agent that will design+compile+verify the kernel\n"
        "BEFORE apply runs; the operator-agent reads the model's own config for arch params.\n"
        "Do NOT default to forward_patch; across the candidates cover at least two distinct kinds.\n\n"
        "Return JSON:\n"
        '{"strategies": [{"focus": "<bottleneck + target mechanism, one line>", '
        '"measures": ["<mechanism step, describe what, not implementation code>"], '
        '"kind": "forward_patch|operator_fusion|graph_rewrite|loading_time", '
        '"local_speedup_ratio": 1.1, '
        '"custom_operator": {"op_name": "<snake_case>", "semantic": "<math/pseudocode>", '
        '"why_custom": "<why official torch_npu is insufficient>", '
        '"fusion_targets": ["<op>", "..."], "expected_signature": "<optional>"}}, ...]}\n'
        "local_speedup_ratio is a conservative estimate, >= 1.0. "
        "OMIT the custom_operator key entirely on strategies that don't need a custom kernel."
    )
    result = call_agent_json("strategy-agent", prompt, timeout=1000)
    if not isinstance(result, dict):
        return None
    raw_strategies = result.get("strategies")
    if not isinstance(raw_strategies, list) or not raw_strategies:
        return None

    strategies: list[OptimizationStrategy] = []
    for item in raw_strategies:
        if not isinstance(item, dict) or len(strategies) >= max_count:
            continue
        measures = item.get("measures") if isinstance(item.get("measures"), list) else ["see focus"]
        focus = str(item.get("focus") or "")
        speedup = _to_float(item.get("local_speedup_ratio")) or 1.05
        # lever 规范化到 LEVER_KINDS：未指定/未知值默认 forward_patch（最窄、最低风险）。
        raw_kind = str(item.get("kind") or "").strip()
        kind = raw_kind if raw_kind in LEVER_KINDS else "forward_patch"
        # 可选 custom_operator spec：仅在 agent 判断「官方无合适算子」时附带。规范化成
        # 一个干净 dict 存进 extra；optimization 据它决定要不要先跑 operator stage。
        custom_op = _normalize_custom_operator(
            item.get("custom_operator"), analysis, f"strategy:{analysis.uid}:{len(strategies) + 1}"
        )
        strategies.append(
            OptimizationStrategy(
                uid=f"strategy:{analysis.uid}:{len(strategies) + 1}",
                local_speedup_ratio=round(speedup, 4),
                measures=measures,
                prompt_instruction=_build_strategy_prompt(analysis, focus, measures, kind),
                extra={
                    "kind": kind,
                    "source_analysis_uid": analysis.uid,
                    "model_id": analysis.model_id,
                    "device_kind": analysis.device_kind,
                    "device_name": analysis.device_name,
                    "dtype": analysis.dtype,
                    "custom_operator": custom_op,   # None 或一个 spec dict
                },
            )
        )
    return strategies if strategies else None


def _normalize_custom_operator(
    raw: object,
    analysis: AnalysisResult,
    strategy_uid: str,
) -> dict | None:
    """把 agent 给的 custom_operator 字段规范化成一个干净 spec dict（缺/坏则 None）。

    存的是 dict 而非 OperatorSpec 实例，让 OptimizationStrategy.extra 保持可 JSON 序列化
    （manifest/ledger 落盘）；optimization 在要用时再 OperatorSpec(**spec) 物化。op_name
    与 semantic 是底线，缺任一就当没提（返回 None，走官方算子路径）。
    """
    if not isinstance(raw, dict):
        return None
    op_name = str(raw.get("op_name") or "").strip()
    semantic = str(raw.get("semantic") or "").strip()
    if not op_name or not semantic:
        return None
    fusion = raw.get("fusion_targets")
    return {
        "op_name": op_name,
        "semantic": semantic,
        "why_custom": str(raw.get("why_custom") or "").strip(),
        "fusion_targets": [str(t) for t in fusion] if isinstance(fusion, list) else [],
        "expected_signature": (str(raw["expected_signature"]).strip()
                               if raw.get("expected_signature") else None),
        "source_strategy_uid": strategy_uid,
        "source_analysis_uid": analysis.uid,
    }


def _build_strategy_prompt(
    analysis: AnalysisResult,
    focus: str,
    measures: list[str],
    kind: str,
) -> str:
    measures_text = "\n".join(f"- {measure}" for measure in measures)
    top_ops = ", ".join(analysis.top_ops[:10]) if analysis.top_ops else "none"
    return (
        "Implement the optimization strategy below against the model execution.\n"
        "The focus and measures are already chosen — your job is the HOW: select the "
        "concrete API, signature, and guards; decide where to apply the patch and how "
        "to wire build_model; and verify functional equivalence. Preserve correctness "
        "while delivering the targeted latency reduction.\n\n"
        f"Lever (kind): {kind}\n"
        f"{_LEVER_HINTS.get(kind, '')}\n\n"
        f"Focus:\n{focus}\n\n"
        f"Measures:\n{measures_text}\n\n"
        "Profile context:\n"
        f"- analysis_uid: {analysis.uid}\n"
        f"- model_id: {analysis.model_id or 'unknown'}\n"
        f"- device: {analysis.device_kind or 'unknown'} {analysis.device_name or ''}\n"
        f"- dtype: {analysis.dtype or 'unknown'}\n"
        f"- total_latency_ms: {analysis.total_latency:.6f}\n"
        f"- top_ops: {top_ops}\n"
        f"- op_type_totals: {analysis.op_type_totals}\n"
        f"- roofline_summary: {analysis.roofline_summary}\n"
        f"Report this same kind ({kind}) in your ChangeRecord JSON unless the actual "
        "change ended up on a different lever.\n"
    )


# 每个 lever 给 apply-agent 的落点提示——让它改对位置，而不是默认去 patch forward。
_LEVER_HINTS = {
    "forward_patch": "Monkey-patch the target nn.Module.forward from inside build_model().",
    "operator_fusion": "Set the config flag (e.g. attn_implementation) at load; do not "
    "hand-patch the attention forward.",
    "graph_rewrite": "Edit build_model.py inside build_model() (after from_pretrained, "
    "before return); wrap the whole model. Do not patch any forward.",
    "loading_time": "Edit build_model.py inside build_model() (after from_pretrained, "
    "before return); do not patch any forward.",
}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
