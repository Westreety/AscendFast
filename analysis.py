from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import AnalysisResult, ProfileResult

def analyze_profile(
    profile: ProfileResult,
) -> AnalysisResult:
    """
    将 ProfileResult 整理汇总为 AnalysisResult，
    结果可反馈至下一轮策略生成。

    Args:
        profile:  本轮 profile 结果

    Returns:
        AnalysisResult
    """
    report, report_path = _load_profile_report(profile)
    extra = profile.extra if isinstance(profile.extra, dict) else {}

    top_kernels = _profile_top_kernels(report)
    top_ops = [str(kernel.get("name", "")) for kernel in top_kernels if kernel.get("name")]
    hot_groups = _hot_groups_by_op_type(top_kernels)
    op_type_totals = _op_type_totals(top_kernels)
    roofline_summary = _roofline_summary(top_kernels)
    unsupported_hotspots = _unsupported_hotspots(top_kernels)
    latency_stats = report.get("latency_stats_ms") if isinstance(report.get("latency_stats_ms"), dict) else None
    dataset = _dataset_manifest(report)
    total_latency = _infer_total_latency_ms(profile, report, latency_stats)
    optimization_hints = _optimization_hints(report, op_type_totals, unsupported_hotspots, latency_stats)

    analysis_extra = {
        "source_profile_uid": profile.uid,
        "execution_mode_uid": profile.execution_mode_uid,
        "latency_before": profile.latency_before,
        "latency_after": profile.latency_after,
        "total_device_time_ms": report.get("total_device_time_ms"),
        "profile_iters": report.get("profile_iters"),
        "total_kernels": report.get("total_kernels"),
        "optimization_summary": report.get("optimization_summary"),
        "analysis_readiness": report.get("analysis_readiness"),
        "artifacts": report.get("artifacts"),
    }
    if extra:
        analysis_extra["profile_extra"] = extra

    return AnalysisResult(
        uid=f"analysis:{profile.uid}",
        total_latency=total_latency,
        top_ops=top_ops,
        hot_groups=hot_groups,
        extra=analysis_extra,
        model_id=str(report.get("pretrained") or report.get("model") or ""),
        device_kind=_optional_str(report.get("device_kind")),
        device_name=_optional_str(report.get("device_name")),
        dtype=_optional_str(report.get("dtype")),
        profile_report_path=str(report_path) if report_path is not None else None,
        latency_stats_ms=latency_stats,
        dataset=dataset,
        top_kernels=top_kernels,
        op_type_totals=op_type_totals,
        roofline_summary=roofline_summary,
        unsupported_hotspots=unsupported_hotspots,
        optimization_hints=optimization_hints,
    )

def _load_profile_report(profile: ProfileResult) -> tuple[dict[str, Any], Path | None]:
    if isinstance(profile.profile_report, dict):
        return profile.profile_report, Path(profile.profile_report_path) if profile.profile_report_path else None

    for value in (profile.profile_report_path, profile.profiler_output_dir):
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_dir():
                path = path / "profile_report.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8")), path

    extra = profile.extra if isinstance(profile.extra, dict) else {}
    direct_report = extra.get("profile_report")
    if isinstance(direct_report, dict):
        return direct_report, _path_from_extra(extra)

    for key in ("profile_report_path", "report_path", "output_path", "profile_report"):
        value = extra.get(key)
        if isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_dir():
                path = path / "profile_report.json"
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8")), path

    profiler_output_dir = extra.get("profiler_output_dir")
    if isinstance(profiler_output_dir, (str, Path)):
        path = Path(profiler_output_dir) / "profile_report.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), path

    return {}, None


def _path_from_extra(extra: dict[str, Any]) -> Path | None:
    for key in ("profile_report_path", "report_path", "output_path"):
        value = extra.get(key)
        if isinstance(value, (str, Path)):
            return Path(value)
    return None


def _profile_top_kernels(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("top_kernels")
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _hot_groups_by_op_type(top_kernels: list[dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for kernel in top_kernels:
        op_type = str(kernel.get("op_type") or "unknown")
        name = str(kernel.get("name") or "")
        if name:
            groups.setdefault(op_type, []).append(name)
    return groups


def _op_type_totals(top_kernels: list[dict[str, Any]]) -> dict[str, dict]:
    totals: dict[str, dict[str, float | int]] = {}
    for kernel in top_kernels:
        op_type = str(kernel.get("op_type") or "unknown")
        item = totals.setdefault(
            op_type,
            {
                "device_time_ms": 0.0,
                "pct_total": 0.0,
                "call_count": 0,
                "supported_time_ms": 0.0,
                "kernel_count": 0,
            },
        )
        duration_ms = _float_or_zero(kernel.get("device_time_ms"))
        pct_total = _float_or_zero(kernel.get("pct_total"))
        call_count = int(_float_or_zero(kernel.get("call_count")))
        item["device_time_ms"] = float(item["device_time_ms"]) + duration_ms
        item["pct_total"] = float(item["pct_total"]) + pct_total
        item["call_count"] = int(item["call_count"]) + call_count
        item["kernel_count"] = int(item["kernel_count"]) + 1
        if bool(kernel.get("autokernel_supported")):
            item["supported_time_ms"] = float(item["supported_time_ms"]) + duration_ms
    return {
        key: {
            "device_time_ms": round(float(value["device_time_ms"]), 6),
            "pct_total": round(float(value["pct_total"]), 6),
            "call_count": int(value["call_count"]),
            "supported_time_ms": round(float(value["supported_time_ms"]), 6),
            "kernel_count": int(value["kernel_count"]),
        }
        for key, value in sorted(totals.items(), key=lambda item: float(item[1]["device_time_ms"]), reverse=True)
    }


def _roofline_summary(top_kernels: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for kernel in top_kernels:
        roofline = str(kernel.get("roofline") or "unknown")
        summary[roofline] = summary.get(roofline, 0.0) + _float_or_zero(kernel.get("device_time_ms"))
    return {
        key: round(value, 6)
        for key, value in sorted(summary.items(), key=lambda item: item[1], reverse=True)
    }


def _unsupported_hotspots(top_kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unsupported = [kernel for kernel in top_kernels if not bool(kernel.get("autokernel_supported"))]
    return sorted(unsupported, key=lambda item: _float_or_zero(item.get("device_time_ms")), reverse=True)


def _dataset_manifest(report: dict[str, Any]) -> dict | None:
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    dataset = artifacts.get("dataset")
    return dict(dataset) if isinstance(dataset, dict) else None


def _infer_total_latency_ms(
    profile: ProfileResult,
    report: dict[str, Any],
    latency_stats: dict | None,
) -> float:
    if latency_stats:
        mean = _float_or_zero(latency_stats.get("mean"))
        if mean > 0.0:
            return mean
    if profile.latency_after > 0.0:
        return float(profile.latency_after)
    total_device_time = _float_or_zero(report.get("total_device_time_ms"))
    profile_iters = int(_float_or_zero(report.get("profile_iters")))
    if total_device_time > 0.0 and profile_iters > 0:
        return total_device_time / profile_iters
    return total_device_time


def _optimization_hints(
    report: dict[str, Any],
    op_type_totals: dict[str, dict],
    unsupported_hotspots: list[dict[str, Any]],
    latency_stats: dict | None,
) -> list[str]:
    hints: list[str] = []
    matmul_pct = _pct(op_type_totals, "matmul")
    attention_pct = _pct(op_type_totals, "flash_attention")
    copy_cast_pct = _pct(op_type_totals, "copy_cast")
    rmsnorm_pct = _pct(op_type_totals, "rmsnorm")
    reduce_pct = _pct(op_type_totals, "reduce")

    if matmul_pct >= 35.0:
        hints.append(
            f"matmul dominates {matmul_pct:.1f}% of top-kernel time; prioritize GEMM/layout/dtype/batch-shape optimizations."
        )
    if attention_pct >= 5.0:
        hints.append(
            f"flash_attention accounts for {attention_pct:.1f}%; inspect attention mask layout, sequence length, and fused attention path."
        )
    if copy_cast_pct >= 2.0:
        hints.append(
            f"copy_cast accounts for {copy_cast_pct:.1f}%; remove redundant dtype/layout conversions before kernel tuning."
        )
    if rmsnorm_pct + reduce_pct >= 5.0:
        hints.append(
            f"normalization/reduce kernels account for {rmsnorm_pct + reduce_pct:.1f}%; consider fused RMSNorm or reduce cleanup."
        )
    if unsupported_hotspots:
        top = unsupported_hotspots[0]
        hints.append(
            "largest unsupported hotspot: "
            f"{top.get('op_type', 'unknown')} {top.get('name', '')} "
            f"({_float_or_zero(top.get('pct_total')):.1f}%)."
        )
    summary = report.get("optimization_summary")
    if isinstance(summary, dict) and summary.get("estimated_max_speedup"):
        hints.append(f"profile-estimated max speedup: {summary['estimated_max_speedup']}.")
    if latency_stats:
        noise = _float_or_zero(latency_stats.get("noise_relative"))
        if noise > 0.05:
            hints.append(f"latency noise is high ({noise:.2%}); rerun profile before ranking small gains.")
    return hints


def _pct(op_type_totals: dict[str, dict], op_type: str) -> float:
    item = op_type_totals.get(op_type)
    if not isinstance(item, dict):
        return 0.0
    return _float_or_zero(item.get("pct_total"))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
