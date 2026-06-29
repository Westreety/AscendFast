"""Small dataset helpers for configured model-level inference sessions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptDataset:
    path: Path
    prompts: tuple[str, ...]
    prompt_field: str = "prompt"
    format: str = "jsonl"

    def manifest(self) -> dict[str, Any]:
        lengths = [len(prompt) for prompt in self.prompts]
        return {
            "path": str(self.path),
            "format": self.format,
            "prompt_field": self.prompt_field,
            "num_prompts": len(self.prompts),
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
            "avg_chars": (sum(lengths) / len(lengths)) if lengths else 0.0,
        }


def load_prompt_dataset(
    path: str | Path,
    *,
    prompt_field: str = "prompt",
    max_samples: int | None = None,
) -> PromptDataset:
    """Load a local JSONL prompt dataset.

    Each non-empty line must be a JSON object containing ``prompt_field``.  The
    helper is intentionally small and deterministic so profile and verify can
    replay the same prompts without pulling in dataset libraries.
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Prompt dataset does not exist: {dataset_path}")
    if dataset_path.suffix.lower() != ".jsonl":
        raise ValueError(f"Only JSONL prompt datasets are supported for now: {dataset_path}")

    prompts: list[str] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {dataset_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {dataset_path}:{line_no}")
            value = row.get(prompt_field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing non-empty field {prompt_field!r} at {dataset_path}:{line_no}")
            prompts.append(value)
            if max_samples is not None and len(prompts) >= max_samples:
                break

    if not prompts:
        raise ValueError(f"No prompts loaded from {dataset_path}")
    return PromptDataset(path=dataset_path, prompts=tuple(prompts), prompt_field=prompt_field)


def load_tokenizer(module_name: str, pretrained: str | None, *, trust_remote_code: bool = True) -> Any | None:
    """Load an AutoTokenizer when available; return None for non-HF-style models."""

    if not pretrained:
        return None
    try:
        module = __import__(module_name, fromlist=["AutoTokenizer"])
    except ImportError:
        return None
    tokenizer_cls = getattr(module, "AutoTokenizer", None)
    if tokenizer_cls is None or not hasattr(tokenizer_cls, "from_pretrained"):
        return None
    kwargs = {"trust_remote_code": trust_remote_code} if module_name == "transformers" else {}
    if module_name == "transformers" and Path(pretrained).expanduser().exists():
        kwargs["local_files_only"] = True
    return tokenizer_cls.from_pretrained(pretrained, **kwargs)


def tokenize_prompts(
    torch: Any,
    tokenizer: Any,
    prompts: tuple[str, ...],
    *,
    device: str,
    max_length: int,
) -> dict[str, Any]:
    _ensure_padding_token(tokenizer)
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}


def _ensure_padding_token(tokenizer: Any) -> None:
    """Use the EOS token for padding when decoder-only tokenizers omit one."""

    if getattr(tokenizer, "pad_token", None) is not None:
        return
    eos_token = getattr(tokenizer, "eos_token", None)
    if eos_token is None:
        return
    try:
        tokenizer.pad_token = eos_token
    except Exception:
        return
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and getattr(tokenizer, "pad_token_id", None) is None:
        try:
            tokenizer.pad_token_id = eos_token_id
        except Exception:
            return


# --------------------------------------------------------------------------- #
# Trace-to-training dataset exports.
# --------------------------------------------------------------------------- #
def export_sft_agent_calls(
    trace_db: str | Path,
    output_path: str | Path,
    *,
    statuses: tuple[str, ...] = ("ok",),
) -> Path:
    """Export agent prompt/response pairs from trace.sqlite as JSONL.

    Output rows use the common training shape:
    ``input``, ``output``, ``metadata``, ``reward``, ``refs``.
    """
    rows = _query(
        trace_db,
        """
        SELECT agent_call_id, run_uid, agent_name, stage, status, prompt,
               raw_response, parsed_response_json, mode_uid, strategy_uid,
               analysis_uid, duration_ms
          FROM agent_calls
         WHERE status IN ({placeholders})
           AND raw_response IS NOT NULL
         ORDER BY completed_at, agent_call_id
        """.format(placeholders=",".join("?" for _ in statuses)),
        tuple(statuses),
    )
    samples = []
    for row in rows:
        samples.append({
            "input": row["prompt"],
            "output": _json_loads(row["parsed_response_json"], row["raw_response"]),
            "metadata": {
                "task": f"{row['stage']}_agent_sft",
                "agent": row["agent_name"],
                "status": row["status"],
                "duration_ms": row["duration_ms"],
            },
            "reward": None,
            "refs": _refs(row),
        })
    return _write_jsonl(output_path, samples)


def export_strategy_preferences(trace_db: str | Path, output_path: str | Path) -> Path:
    """Export DPO-style pairs among strategies from the same analysis."""
    strategies = _query(
        trace_db,
        """
        SELECT s.*, COALESCE(MAX(r.value), -999999.0) AS reward_value
          FROM strategies s
          LEFT JOIN rewards r
            ON r.run_uid = s.run_uid AND r.strategy_uid = s.strategy_uid
         GROUP BY s.run_uid, s.strategy_uid
         ORDER BY s.run_uid, s.analysis_uid, reward_value DESC, s.rank ASC
        """,
    )
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in strategies:
        grouped.setdefault((row["run_uid"], row["analysis_uid"]), []).append(row)

    samples = []
    for (_run_uid, analysis_uid), group in grouped.items():
        if len(group) < 2:
            continue
        chosen = group[0]
        for rejected in group[1:]:
            if float(chosen["reward_value"]) == float(rejected["reward_value"]):
                continue
            samples.append({
                "input": {"analysis_uid": analysis_uid},
                "output": {
                    "chosen": _json_loads(chosen["payload_json"], {}),
                    "rejected": _json_loads(rejected["payload_json"], {}),
                },
                "metadata": {
                    "task": "strategy_preference",
                    "chosen_reward": chosen["reward_value"],
                    "rejected_reward": rejected["reward_value"],
                },
                "reward": {
                    "chosen": chosen["reward_value"],
                    "rejected": rejected["reward_value"],
                },
                "refs": {
                    "run_uid": chosen["run_uid"],
                    "analysis_uid": analysis_uid,
                    "strategy_uid": chosen["strategy_uid"],
                    "rejected_strategy_uid": rejected["strategy_uid"],
                },
            })
    return _write_jsonl(output_path, samples)


def export_apply_preferences(trace_db: str | Path, output_path: str | Path) -> Path:
    """Export apply success/failure preference pairs for the same strategy."""
    actions = _query(
        trace_db,
        """
        SELECT *
          FROM apply_actions
         ORDER BY run_uid, strategy_uid, phase, status DESC
        """,
    )
    grouped: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
    for row in actions:
        grouped.setdefault((row["run_uid"], row["strategy_uid"], row["phase"]), []).append(row)

    samples = []
    for (_run_uid, strategy_uid, phase), group in grouped.items():
        successes = [row for row in group if row["status"] in {"applied", "operator_pending"}]
        failures = [row for row in group if row["status"] not in {"applied", "operator_pending"}]
        for chosen in successes:
            for rejected in failures:
                samples.append({
                    "input": {"strategy_uid": strategy_uid, "phase": phase},
                    "output": {
                        "chosen": _apply_payload(chosen),
                        "rejected": _apply_payload(rejected),
                    },
                    "metadata": {"task": "apply_preference", "phase": phase},
                    "reward": {"chosen": 1.0, "rejected": -1.0},
                    "refs": {
                        "run_uid": chosen["run_uid"],
                        "mode_uid": chosen["mode_uid"],
                        "parent_mode_uid": chosen["parent_mode_uid"],
                        "strategy_uid": strategy_uid,
                        "action_id": chosen["action_id"],
                        "rejected_action_id": rejected["action_id"],
                    },
                })
    return _write_jsonl(output_path, samples)


def export_rl_trajectories(trace_db: str | Path, output_path: str | Path) -> Path:
    """Export mode/action/evaluation/reward trajectory rows."""
    modes = _query(
        trace_db,
        """
        SELECT m.*, COALESCE(MAX(r.value), 0.0) AS reward_value
          FROM modes m
          LEFT JOIN rewards r
            ON r.run_uid = m.run_uid AND r.mode_uid = m.mode_uid
         GROUP BY m.run_uid, m.mode_uid
         ORDER BY m.run_uid, m.mode_uid
        """,
    )
    samples = []
    for mode in modes:
        actions = _query(
            trace_db,
            """
            SELECT *
              FROM apply_actions
             WHERE run_uid = ? AND mode_uid = ?
             ORDER BY action_id
            """,
            (mode["run_uid"], mode["mode_uid"]),
        )
        evals = _query(
            trace_db,
            """
            SELECT *
              FROM evaluations
             WHERE run_uid = ? AND mode_uid = ?
             ORDER BY eval_type, eval_id
            """,
            (mode["run_uid"], mode["mode_uid"]),
        )
        samples.append({
            "input": {
                "state": _json_loads(mode["payload_json"], {}),
                "actions": [_apply_payload(row) for row in actions],
            },
            "output": {
                "observations": [_json_loads(row["payload_json"], {}) for row in evals],
            },
            "metadata": {"task": "rl_trajectory"},
            "reward": mode["reward_value"],
            "refs": {
                "run_uid": mode["run_uid"],
                "mode_uid": mode["mode_uid"],
                "parent_mode_uid": mode["parent_mode_uid"],
                "strategy_uid": mode["strategy_uid"],
            },
        })
    return _write_jsonl(output_path, samples)


def _query(
    trace_db: str | Path,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[sqlite3.Row]:
    conn = sqlite3.connect(trace_db)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(sql, params))
    finally:
        conn.close()


def _write_jsonl(output_path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _json_loads(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _refs(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "run_uid": row["run_uid"],
        "mode_uid": row["mode_uid"],
        "parent_mode_uid": None,
        "analysis_uid": row["analysis_uid"],
        "strategy_uid": row["strategy_uid"],
        "agent_call_id": row["agent_call_id"],
        "artifact_refs": [],
    }


def _apply_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "action_id": row["action_id"],
        "phase": row["phase"],
        "status": row["status"],
        "change_record": _json_loads(row["change_record_json"], None),
        "operator_spec": _json_loads(row["operator_spec_json"], None),
        "operator_artifact": _json_loads(row["operator_artifact_json"], None),
        "forward_gate_ok": row["forward_gate_ok"],
        "error": row["error"],
    }
