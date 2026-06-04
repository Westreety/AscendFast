#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch_npu  # noqa: F401
from torch_npu.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler
from transformers import AutoModelForCausalLM, AutoTokenizer

from _hf_common import (
    PROJECT_ROOT,
    adaptation_cache_dir,
    configure_hf_endpoint,
    local_or_remote_model_ref,
    sanitize_model_name,
)


DEFAULT_MODEL = PROJECT_ROOT / "model" / "yi_6b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile a Hugging Face causal LM with torch_npu.profiler."
    )
    parser.add_argument(
        "model",
        nargs="?",
        default=str(DEFAULT_MODEL),
        help="Local model path or Hugging Face model id. Default: model/yi_6b",
    )
    parser.add_argument("--prompt", default="Hello, please summarize this model in one sentence.")
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional JSONL file. Each line must contain a prompt field.",
    )
    parser.add_argument("--mode", choices=["forward", "generate"], default="forward")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--sort-by", default="npu_time_total")
    parser.add_argument("--trace-dir", default=None, help="Default: profiles/<model-name>")
    parser.add_argument("--table-file", default=None, help="Default: <trace-dir>/key_averages.txt")
    parser.add_argument("--chrome-trace", default=None, help="Optional path for prof.export_chrome_trace().")
    return parser.parse_args()


def load_prompts(prompt_file: str | None, fallback_prompt: str) -> list[str]:
    if prompt_file is None:
        return [fallback_prompt]

    path = Path(prompt_file)
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{path}:{line_no} must contain a non-empty prompt string")
            prompts.append(prompt)
    if not prompts:
        raise ValueError(f"{path} does not contain any prompts")
    return prompts


def resolve_dtype(value: str) -> str | torch.dtype:
    if value == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[value]


def reset_peak_memory() -> None:
    if hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()


def max_memory_gb() -> float | None:
    if not hasattr(torch.npu, "max_memory_allocated"):
        return None
    return torch.npu.max_memory_allocated() / 1024**3


def parse_us(value: str | None) -> float:
    if not value:
        return 0.0
    return float(value.strip().replace(",", ""))


def latest_profiler_output(trace_dir: Path) -> Path | None:
    outputs = list(trace_dir.glob("*_ascend_pt/ASCEND_PROFILER_OUTPUT"))
    if not outputs:
        return None
    return max(outputs, key=lambda path: path.stat().st_mtime)


def render_csv_operator_table(trace_dir: Path, sort_by: str, row_limit: int) -> str:
    output_dir = latest_profiler_output(trace_dir)
    if output_dir is None:
        return f"No ASCEND_PROFILER_OUTPUT directory found under {trace_dir}"
    operator_csv = output_dir / "operator_details.csv"
    if not operator_csv.exists():
        return f"No operator_details.csv found under {output_dir}"

    stats: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with operator_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Name") or "<unknown>"
            item = stats[name]
            item["calls"] += 1
            item["host_self_us"] += parse_us(row.get("Host Self Duration(us)"))
            item["host_total_us"] += parse_us(row.get("Host Total Duration(us)"))
            item["device_self_us"] += parse_us(row.get("Device Self Duration(us)"))
            item["device_total_us"] += parse_us(row.get("Device Total Duration(us)"))
            item["aicore_self_us"] += parse_us(row.get("Device Self Duration With AICore(us)"))
            item["aicore_total_us"] += parse_us(row.get("Device Total Duration With AICore(us)"))

    sort_map = {
        "npu_time_total": "device_total_us",
        "npu_time": "device_self_us",
        "device_total_us": "device_total_us",
        "device_self_us": "device_self_us",
        "host_total_us": "host_total_us",
        "host_self_us": "host_self_us",
        "aicore_total_us": "aicore_total_us",
        "aicore_self_us": "aicore_self_us",
    }
    sort_key = sort_map.get(sort_by, "device_total_us")
    rows = sorted(stats.items(), key=lambda item: item[1][sort_key], reverse=True)

    headers = [
        "Name",
        "Calls",
        "Device Total(us)",
        "Device Self(us)",
        "Host Total(us)",
        "Host Self(us)",
        "AICore Total(us)",
    ]
    table_rows = []
    for name, item in rows[:row_limit]:
        table_rows.append(
            [
                name,
                str(int(item["calls"])),
                f"{item['device_total_us']:.3f}",
                f"{item['device_self_us']:.3f}",
                f"{item['host_total_us']:.3f}",
                f"{item['host_self_us']:.3f}",
                f"{item['aicore_total_us']:.3f}",
            ]
        )

    widths = [
        max(len(row[index]) for row in [headers, *table_rows])
        for index in range(len(headers))
    ]
    lines = [
        f"torch_npu.profile has no key_averages(); aggregated from {operator_csv}",
        " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in table_rows:
        lines.append(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    return "\n".join(lines)


def render_table(prof: Any, trace_dir: Path, sort_by: str, row_limit: int) -> str:
    if hasattr(prof, "key_averages"):
        try:
            return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
        except Exception as exc:  # noqa: BLE001
            fallback = "self_cpu_time_total"
            return (
                f"Failed to sort by {sort_by}: {exc}\n"
                + prof.key_averages().table(sort_by=fallback, row_limit=row_limit)
            )
    try:
        return render_csv_operator_table(trace_dir, sort_by, row_limit)
    except Exception as exc:  # noqa: BLE001
        fallback = "self_cpu_time_total"
        return f"Failed to build operator table from profiler CSVs: {exc}\nfallback_sort: {fallback}"


def main() -> int:
    args = parse_args()
    endpoint = configure_hf_endpoint()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch.npu is not available. Check Ascend runtime and torch_npu installation.")

    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")

    model_ref = local_or_remote_model_ref(args.model)
    model_name = sanitize_model_name(args.model)
    cache_dir = Path(args.cache_dir) if args.cache_dir else adaptation_cache_dir(args.model)
    trace_dir = Path(args.trace_dir) if args.trace_dir else PROJECT_ROOT / "profiles" / model_name
    table_file = Path(args.table_file) if args.table_file else trace_dir / "key_averages.txt"
    trace_dir.mkdir(parents=True, exist_ok=True)
    table_file.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {model_ref}")
    print(f"cache_dir: {cache_dir}")
    print(f"trace_dir: {trace_dir}")
    print(f"hf_endpoint: {endpoint}")
    print(f"device: {device}")

    prompts = load_prompts(args.prompt_file, args.prompt)
    print(f"prompt_count: {len(prompts)}")
    if args.prompt_file:
        print(f"prompt_file: {args.prompt_file}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_ref,
        trust_remote_code=args.trust_remote_code,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    dtype = resolve_dtype(args.dtype)
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "cache_dir": cache_dir,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "dtype": dtype,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model_kwargs.pop("dtype")
        model_kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_ref, **model_kwargs)
    model.eval()
    model.to(device)

    encoded_inputs = []
    for prompt in prompts:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        encoded_inputs.append({name: tensor.to(device) for name, tensor in encoded.items()})

    prompt_index = 0

    def run_once() -> Any:
        nonlocal prompt_index
        inputs = encoded_inputs[prompt_index % len(encoded_inputs)]
        prompt_index += 1
        if args.mode == "generate":
            return model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=args.use_cache,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return model(**inputs, use_cache=args.use_cache)

    with torch.inference_mode():
        for _ in range(args.warmup):
            run_once()
            torch.npu.synchronize()

        reset_peak_memory()
        start = time.perf_counter()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            schedule=schedule(wait=0, warmup=args.warmup, active=args.steps, repeat=1),
            record_shapes=args.record_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.with_stack,
            on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        ) as prof:
            for _ in range(args.warmup + args.steps):
                run_once()
                torch.npu.synchronize()
                prof.step()
        elapsed_ms = (time.perf_counter() - start) * 1000

    table = render_table(prof, trace_dir, args.sort_by, args.row_limit)
    table_file.write_text(table, encoding="utf-8")
    print(table)
    print(f"profiled_steps: {args.steps}")
    print(f"elapsed_ms: {elapsed_ms:.3f}")
    print(f"avg_step_ms: {elapsed_ms / max(args.steps, 1):.3f}")
    peak = max_memory_gb()
    if peak is not None:
        print(f"peak_npu_memory_gb: {peak:.3f}")
    print(f"table_file: {table_file}")

    if args.chrome_trace:
        chrome_trace = Path(args.chrome_trace)
        chrome_trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(chrome_trace))
        print(f"chrome_trace: {chrome_trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
