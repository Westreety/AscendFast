"""Append-only trace storage for optimization runs.

The RunLedger in verify.py is intentionally small and human-oriented.  This
module is the training/RL data plane: it records raw events to JSONL and indexes
the important entities into SQLite so later jobs can export SFT, preference, or
trajectory datasets.

All public record functions are best-effort.  Trace failures must never change
the optimization pipeline's behavior.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent
_TRACE_DIR = _PROJECT_ROOT / "traces"  # 训练数据专用目录，不应被清空

_CURRENT: "_TraceContext | None" = None
_LOCK = threading.RLock()


class _TraceContext:
    def __init__(self, run_uid: str, model_id: str, config: dict[str, Any] | None) -> None:
        self.run_uid = run_uid
        self.model_id = model_id
        self.safe_uid = _safe_id(run_uid)
        self.run_dir = _TRACE_DIR / self.safe_uid
        self.events_path = self.run_dir / "events.jsonl"
        self.db_path = self.run_dir / "trace.sqlite"
        self.seq = 0
        self.started_at = _now()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        _init_schema(self.conn)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs
                (run_uid, model_id, started_at, completed_at, status,
                 config_json, summary_json, rewards_json)
            VALUES (?, ?, ?, NULL, 'running', ?, NULL, NULL)
            """,
            (run_uid, model_id, self.started_at, _json_dumps(config or {})),
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def start_run_trace(
    run_uid: str,
    model_id: str,
    config: dict[str, Any] | None = None,
) -> Path | None:
    """Start a run-local trace under runs/<safe-run-uid>/."""
    global _CURRENT
    try:
        with _LOCK:
            if _CURRENT is not None:
                _CURRENT.close()
            _CURRENT = _TraceContext(run_uid, model_id, config)
            record_event(
                "run_started",
                {"model_id": model_id, "config": config or {}},
                refs={"run_uid": run_uid},
            )
            return _CURRENT.run_dir
    except Exception as exc:  # noqa: BLE001 - tracing is observability only
        print(f"[trace] start_run_trace failed: {type(exc).__name__}: {exc}")
        _CURRENT = None
        return None


def finalize_run_trace(
    summary: dict[str, Any] | None = None,
    rewards: dict[str, Any] | None = None,
) -> None:
    """Write the terminal run event, update SQLite, and close the trace."""
    global _CURRENT
    ctx = _CURRENT
    if ctx is None:
        return
    try:
        with _LOCK:
            record_event("run_completed", summary or {}, refs={"run_uid": ctx.run_uid})
            ctx.conn.execute(
                """
                UPDATE runs
                   SET completed_at = ?, status = ?, summary_json = ?, rewards_json = ?
                 WHERE run_uid = ?
                """,
                (
                    _now(),
                    "completed",
                    _json_dumps(summary or {}),
                    _json_dumps(rewards or {}),
                    ctx.run_uid,
                ),
            )
            ctx.conn.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] finalize_run_trace failed: {type(exc).__name__}: {exc}")
    finally:
        ctx.close()
        _CURRENT = None


def current_run_uid() -> str | None:
    ctx = _CURRENT
    return ctx.run_uid if ctx is not None else None


def current_trace_dir() -> Path | None:
    ctx = _CURRENT
    return ctx.run_dir if ctx is not None else None


def record_event(
    event_type: str,
    payload: Any | None = None,
    refs: dict[str, Any] | None = None,
) -> str | None:
    """Append a raw trace event and index it into the generic events table."""
    ctx = _CURRENT
    if ctx is None:
        return None
    try:
        with _LOCK:
            ctx.seq += 1
            event_id = f"event:{uuid.uuid4().hex}"
            refs_obj = _jsonable(refs or {})
            payload_obj = _jsonable(payload or {})
            row = {
                "event_id": event_id,
                "run_uid": ctx.run_uid,
                "seq": ctx.seq,
                "timestamp": _now(),
                "event_type": event_type,
                "payload": payload_obj,
                "refs": refs_obj,
            }
            with ctx.events_path.open("a", encoding="utf-8") as handle:
                handle.write(_json_dumps(row) + "\n")
            ctx.conn.execute(
                """
                INSERT INTO events
                    (event_id, run_uid, seq, timestamp, event_type, stage, mode_uid,
                     payload_json, refs_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    ctx.run_uid,
                    ctx.seq,
                    row["timestamp"],
                    event_type,
                    _optional_str(refs_obj.get("stage")),
                    _optional_str(refs_obj.get("mode_uid")),
                    _json_dumps(payload_obj),
                    _json_dumps(refs_obj),
                ),
            )
            ctx.conn.commit()
            return event_id
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_event failed: {type(exc).__name__}: {exc}")
        return None


def record_agent_io(
    agent_name: str,
    stage: str,
    prompt: str,
    raw_response: str | None,
    parsed_response: Any | None,
    status: str,
    *,
    detail: str = "",
    duration_ms: float | None = None,
    refs: dict[str, Any] | None = None,
) -> str | None:
    """Record one agent call with prompt, raw text, parsed JSON, and status."""
    ctx = _CURRENT
    if ctx is None:
        return None
    agent_call_id = f"agent_call:{uuid.uuid4().hex}"
    stage = stage or _stage_from_agent(agent_name)
    refs_obj = dict(refs or {})
    refs_obj.update({"agent_call_id": agent_call_id, "agent": agent_name, "stage": stage})
    prompt_event = _prompt_event_for(stage)
    done_event = _done_event_for(stage)
    try:
        with _LOCK:
            record_event(
                prompt_event,
                {
                    "agent_call_id": agent_call_id,
                    "agent": agent_name,
                    "prompt": prompt,
                },
                refs=refs_obj,
            )
            done_payload = {
                "agent_call_id": agent_call_id,
                "agent": agent_name,
                "status": status,
                "detail": detail,
                "duration_ms": duration_ms,
                "raw_response": raw_response,
                "parsed_response": parsed_response,
            }
            event_id = record_event(done_event, done_payload, refs=refs_obj)
            now = _now()
            duration = float(duration_ms) if duration_ms is not None else None
            started_at = now - (duration / 1000.0) if duration is not None else None
            ctx.conn.execute(
                """
                INSERT INTO agent_calls
                    (agent_call_id, run_uid, event_id, agent_name, stage, status,
                     started_at, completed_at, duration_ms, prompt, raw_response,
                     parsed_response_json, error, mode_uid, strategy_uid, analysis_uid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_call_id,
                    ctx.run_uid,
                    event_id,
                    agent_name,
                    stage,
                    status,
                    started_at,
                    now,
                    duration,
                    prompt,
                    raw_response,
                    _json_dumps(parsed_response),
                    detail,
                    _optional_str(refs_obj.get("mode_uid")),
                    _optional_str(refs_obj.get("strategy_uid")),
                    _optional_str(refs_obj.get("analysis_uid")),
                ),
            )
            ctx.conn.commit()
            return agent_call_id
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_agent_io failed: {type(exc).__name__}: {exc}")
        return None


def record_mode(mode: Any, *, payload: dict[str, Any] | None = None) -> None:
    ctx = _CURRENT
    if ctx is None or mode is None:
        return
    try:
        data = _jsonable(mode)
        if payload:
            data["payload"] = _jsonable(payload)
        mode_uid = _optional_str(getattr(mode, "uid", None))
        parent_uid = _optional_str(getattr(mode, "parent_uid", None))
        strategy_uid = _optional_str(getattr(mode, "strategy_uid", None))
        record_event("mode_evaluated", data, refs={"mode_uid": mode_uid})
        with _LOCK:
            ctx.conn.execute(
                """
                INSERT OR REPLACE INTO modes
                    (run_uid, mode_uid, parent_mode_uid, strategy_uid, workspace_dir,
                     correctness_passed, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ctx.run_uid,
                    mode_uid,
                    parent_uid,
                    strategy_uid,
                    _optional_str(getattr(mode, "workspace_dir", None)),
                    _bool_to_int(getattr(mode, "correctness_passed", None)),
                    _json_dumps(data),
                ),
            )
            ctx.conn.commit()
        workspace_dir = getattr(mode, "workspace_dir", None)
        if workspace_dir:
            record_artifact(
                Path(workspace_dir) / "mode_manifest.json",
                kind="mode_manifest",
                mode_uid=mode_uid,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_mode failed: {type(exc).__name__}: {exc}")


def record_analysis_result(analysis: Any, profile: Any | None = None) -> None:
    ctx = _CURRENT
    if ctx is None or analysis is None:
        return
    try:
        payload = _jsonable(analysis)
        profile_uid = _optional_str(getattr(profile, "uid", None))
        mode_uid = _optional_str(getattr(profile, "execution_mode_uid", None))
        payload["source_profile_uid"] = profile_uid
        record_event(
            "analysis_completed",
            payload,
            refs={
                "analysis_uid": _optional_str(getattr(analysis, "uid", None)),
                "profile_uid": profile_uid,
                "mode_uid": mode_uid,
                "stage": "analysis",
            },
        )
        with _LOCK:
            ctx.conn.execute(
                """
                INSERT OR REPLACE INTO analyses
                    (run_uid, analysis_uid, mode_uid, profile_uid, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    ctx.run_uid,
                    _optional_str(getattr(analysis, "uid", None)),
                    mode_uid,
                    profile_uid,
                    _json_dumps(payload),
                ),
            )
            ctx.conn.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_analysis_result failed: {type(exc).__name__}: {exc}")


def record_strategies(analysis: Any, strategies: list[Any] | None) -> None:
    ctx = _CURRENT
    if ctx is None:
        return
    try:
        analysis_uid = _optional_str(getattr(analysis, "uid", None))
        mode_uid = _optional_str(getattr(analysis, "execution_mode_uid", None))
        extra = getattr(analysis, "extra", None)
        if mode_uid is None and isinstance(extra, dict):
            mode_uid = _optional_str(extra.get("execution_mode_uid"))
        payload = {
            "analysis_uid": analysis_uid,
            "count": len(strategies or []),
            "strategies": [_jsonable(item) for item in strategies or []],
        }
        record_event(
            "strategy_generated",
            payload,
            refs={"analysis_uid": analysis_uid, "mode_uid": mode_uid, "stage": "strategy"},
        )
        with _LOCK:
            for rank, strategy in enumerate(strategies or [], 1):
                item = _jsonable(strategy)
                extra = item.get("extra") if isinstance(item.get("extra"), dict) else {}
                ctx.conn.execute(
                    """
                    INSERT OR REPLACE INTO strategies
                        (run_uid, strategy_uid, analysis_uid, mode_uid, rank, kind,
                         local_speedup_ratio, payload_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ctx.run_uid,
                        _optional_str(item.get("uid")),
                        analysis_uid,
                        mode_uid,
                        rank,
                        _optional_str(extra.get("kind")),
                        _float_or_none(item.get("local_speedup_ratio")),
                        _json_dumps(item),
                    ),
                )
            ctx.conn.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_strategies failed: {type(exc).__name__}: {exc}")


def record_apply_action(
    *,
    strategy: Any,
    mode: Any | None,
    base_mode: Any | None,
    phase: str,
    status: str,
    change_record: Any | None = None,
    operator_spec: Any | None = None,
    operator_artifact: Any | None = None,
    workspace_dir: str | Path | None = None,
    forward_gate_ok: bool | None = None,
    error: str = "",
) -> None:
    ctx = _CURRENT
    if ctx is None:
        return
    try:
        action_id = f"apply:{uuid.uuid4().hex}"
        mode_uid = _optional_str(getattr(mode, "uid", None))
        parent_uid = _optional_str(getattr(base_mode, "uid", None) or getattr(mode, "parent_uid", None))
        strategy_uid = _optional_str(getattr(strategy, "uid", None))
        payload = {
            "action_id": action_id,
            "phase": phase,
            "status": status,
            "mode_uid": mode_uid,
            "parent_mode_uid": parent_uid,
            "strategy_uid": strategy_uid,
            "workspace_dir": str(workspace_dir) if workspace_dir is not None else None,
            "forward_gate_ok": forward_gate_ok,
            "change_record": _jsonable(change_record),
            "operator_spec": _jsonable(operator_spec),
            "operator_artifact": _jsonable(operator_artifact),
            "error": error,
        }
        record_event(
            "apply_completed",
            payload,
            refs={
                "mode_uid": mode_uid,
                "parent_mode_uid": parent_uid,
                "strategy_uid": strategy_uid,
                "stage": "apply",
            },
        )
        with _LOCK:
            ctx.conn.execute(
                """
                INSERT INTO apply_actions
                    (action_id, run_uid, mode_uid, parent_mode_uid, strategy_uid,
                     phase, status, change_record_json, operator_spec_json,
                     operator_artifact_json, workspace_dir, forward_gate_ok, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    ctx.run_uid,
                    mode_uid,
                    parent_uid,
                    strategy_uid,
                    phase,
                    status,
                    _json_dumps(change_record),
                    _json_dumps(operator_spec),
                    _json_dumps(operator_artifact),
                    str(workspace_dir) if workspace_dir is not None else None,
                    _bool_to_int(forward_gate_ok),
                    error,
                ),
            )
            ctx.conn.commit()
        if workspace_dir is not None and change_record is not None:
            files = _change_record_files(change_record)
            for rel_path in files:
                record_artifact(Path(workspace_dir) / rel_path, kind="workspace_file", mode_uid=mode_uid)
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_apply_action failed: {type(exc).__name__}: {exc}")


def record_evaluation(
    mode_uid: str | None,
    eval_type: str,
    *,
    ok: bool | None = None,
    metric_name: str | None = None,
    metric_value: float | None = None,
    payload: Any | None = None,
) -> None:
    ctx = _CURRENT
    if ctx is None:
        return
    try:
        eval_id = f"eval:{uuid.uuid4().hex}"
        event_type = "correctness_completed" if eval_type == "correctness" else "mode_benchmarked"
        event_payload = {
            "eval_id": eval_id,
            "eval_type": eval_type,
            "ok": ok,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "payload": _jsonable(payload),
        }
        record_event(
            event_type,
            event_payload,
            refs={"mode_uid": mode_uid, "stage": eval_type},
        )
        with _LOCK:
            ctx.conn.execute(
                """
                INSERT INTO evaluations
                    (eval_id, run_uid, mode_uid, eval_type, ok, metric_name,
                     metric_value, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    eval_id,
                    ctx.run_uid,
                    mode_uid,
                    eval_type,
                    _bool_to_int(ok),
                    metric_name,
                    metric_value,
                    _json_dumps(event_payload),
                ),
            )
            ctx.conn.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_evaluation failed: {type(exc).__name__}: {exc}")


def record_reward(
    mode_uid: str | None,
    *,
    parent_mode_uid: str | None = None,
    strategy_uid: str | None = None,
    reward_type: str = "speedup_reward",
    value: float | None = None,
    payload: Any | None = None,
) -> None:
    ctx = _CURRENT
    if ctx is None:
        return
    try:
        reward_id = f"reward:{uuid.uuid4().hex}"
        event_payload = {
            "reward_id": reward_id,
            "reward_type": reward_type,
            "value": value,
            "payload": _jsonable(payload),
        }
        record_event(
            "reward_recorded",
            event_payload,
            refs={
                "mode_uid": mode_uid,
                "parent_mode_uid": parent_mode_uid,
                "strategy_uid": strategy_uid,
            },
        )
        with _LOCK:
            ctx.conn.execute(
                """
                INSERT INTO rewards
                    (reward_id, run_uid, mode_uid, parent_mode_uid, strategy_uid,
                     reward_type, value, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reward_id,
                    ctx.run_uid,
                    mode_uid,
                    parent_mode_uid,
                    strategy_uid,
                    reward_type,
                    value,
                    _json_dumps(event_payload),
                ),
            )
            ctx.conn.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_reward failed: {type(exc).__name__}: {exc}")


def record_artifact(
    path: str | Path | None,
    *,
    kind: str,
    mode_uid: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    ctx = _CURRENT
    if ctx is None or path is None:
        return
    try:
        artifact_path = Path(path)
        if not artifact_path.exists() or not artifact_path.is_file():
            return
        artifact_id = f"artifact:{uuid.uuid4().hex}"
        digest = _sha256_file(artifact_path)
        size = artifact_path.stat().st_size
        payload = {
            "artifact_id": artifact_id,
            "kind": kind,
            "path": str(artifact_path),
            "sha256": digest,
            "size_bytes": size,
            "metadata": metadata or {},
        }
        record_event("artifact_recorded", payload, refs={"mode_uid": mode_uid, "kind": kind})
        with _LOCK:
            ctx.conn.execute(
                """
                INSERT INTO artifacts
                    (artifact_id, run_uid, mode_uid, kind, path, sha256,
                     size_bytes, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact_id,
                    ctx.run_uid,
                    mode_uid,
                    kind,
                    str(artifact_path),
                    digest,
                    size,
                    _json_dumps(metadata or {}),
                ),
            )
            ctx.conn.commit()
    except Exception as exc:  # noqa: BLE001
        print(f"[trace] record_artifact failed: {type(exc).__name__}: {exc}")


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_uid TEXT PRIMARY KEY,
            model_id TEXT,
            started_at REAL,
            completed_at REAL,
            status TEXT,
            config_json TEXT,
            summary_json TEXT,
            rewards_json TEXT
        );

        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            run_uid TEXT,
            seq INTEGER,
            timestamp REAL,
            event_type TEXT,
            stage TEXT,
            mode_uid TEXT,
            payload_json TEXT,
            refs_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_events_run_seq ON events(run_uid, seq);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_events_mode ON events(mode_uid);

        CREATE TABLE IF NOT EXISTS modes (
            run_uid TEXT,
            mode_uid TEXT,
            parent_mode_uid TEXT,
            strategy_uid TEXT,
            workspace_dir TEXT,
            correctness_passed INTEGER,
            payload_json TEXT,
            PRIMARY KEY (run_uid, mode_uid)
        );

        CREATE TABLE IF NOT EXISTS agent_calls (
            agent_call_id TEXT PRIMARY KEY,
            run_uid TEXT,
            event_id TEXT,
            agent_name TEXT,
            stage TEXT,
            status TEXT,
            started_at REAL,
            completed_at REAL,
            duration_ms REAL,
            prompt TEXT,
            raw_response TEXT,
            parsed_response_json TEXT,
            error TEXT,
            mode_uid TEXT,
            strategy_uid TEXT,
            analysis_uid TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_agent_calls_run_stage ON agent_calls(run_uid, stage);

        CREATE TABLE IF NOT EXISTS analyses (
            run_uid TEXT,
            analysis_uid TEXT,
            mode_uid TEXT,
            profile_uid TEXT,
            payload_json TEXT,
            PRIMARY KEY (run_uid, analysis_uid)
        );

        CREATE TABLE IF NOT EXISTS strategies (
            run_uid TEXT,
            strategy_uid TEXT,
            analysis_uid TEXT,
            mode_uid TEXT,
            rank INTEGER,
            kind TEXT,
            local_speedup_ratio REAL,
            payload_json TEXT,
            PRIMARY KEY (run_uid, strategy_uid)
        );

        CREATE TABLE IF NOT EXISTS apply_actions (
            action_id TEXT PRIMARY KEY,
            run_uid TEXT,
            mode_uid TEXT,
            parent_mode_uid TEXT,
            strategy_uid TEXT,
            phase TEXT,
            status TEXT,
            change_record_json TEXT,
            operator_spec_json TEXT,
            operator_artifact_json TEXT,
            workspace_dir TEXT,
            forward_gate_ok INTEGER,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_apply_strategy ON apply_actions(run_uid, strategy_uid);

        CREATE TABLE IF NOT EXISTS evaluations (
            eval_id TEXT PRIMARY KEY,
            run_uid TEXT,
            mode_uid TEXT,
            eval_type TEXT,
            ok INTEGER,
            metric_name TEXT,
            metric_value REAL,
            payload_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_evaluations_mode ON evaluations(run_uid, mode_uid);

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id TEXT PRIMARY KEY,
            run_uid TEXT,
            mode_uid TEXT,
            kind TEXT,
            path TEXT,
            sha256 TEXT,
            size_bytes INTEGER,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS rewards (
            reward_id TEXT PRIMARY KEY,
            run_uid TEXT,
            mode_uid TEXT,
            parent_mode_uid TEXT,
            strategy_uid TEXT,
            reward_type TEXT,
            value REAL,
            payload_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_rewards_strategy ON rewards(run_uid, strategy_uid);
        """
    )
    conn.commit()


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_id(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def _now() -> float:
    return time.time()


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bool_to_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


def _change_record_files(change_record: Any) -> list[str]:
    if change_record is None:
        return []
    if is_dataclass(change_record):
        files = getattr(change_record, "files", None)
    elif isinstance(change_record, dict):
        files = change_record.get("files")
    else:
        files = None
    return [str(path) for path in files] if isinstance(files, list) else []


def _stage_from_agent(agent_name: str) -> str:
    if agent_name.endswith("-agent"):
        return agent_name[: -len("-agent")]
    return agent_name


def _prompt_event_for(stage: str) -> str:
    if stage in {"analysis", "strategy", "apply"}:
        return f"{stage}_prompted"
    return "agent_prompted"


def _done_event_for(stage: str) -> str:
    if stage == "strategy":
        return "strategy_generated"
    if stage in {"analysis", "apply"}:
        return f"{stage}_completed"
    if stage == "operator":
        return "operator_completed"
    return "agent_completed"
