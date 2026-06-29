from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import agent_client
import trace_store
from dataset import (
    export_apply_preferences,
    export_rl_trajectories,
    export_sft_agent_calls,
    export_strategy_preferences,
)
from models import AnalysisResult, ChangeRecord, ExecutionMode, OptimizationStrategy, ProfileResult


class TraceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.old_runs_dir = trace_store._RUNS_DIR
        trace_store._RUNS_DIR = Path(self.tmp.name) / "runs"

    def tearDown(self) -> None:
        trace_store.finalize_run_trace({"test": "tearDown"}, {})
        trace_store._RUNS_DIR = self.old_runs_dir
        self.tmp.cleanup()

    def test_trace_schema_and_training_exports(self) -> None:
        run_dir = trace_store.start_run_trace("run:unit-full", "unit-model", {"top_k": 2})
        self.assertIsNotNone(run_dir)

        workspace = Path(run_dir) / "workspace"
        workspace.mkdir()
        (workspace / "mode_manifest.json").write_text("{}", encoding="utf-8")
        (workspace / "build_model.py").write_text("# changed\n", encoding="utf-8")

        mode = ExecutionMode(
            uid="mode:child",
            model_id="unit-model",
            strategy_uid="strategy:1",
            workspace_dir=str(workspace),
            parent_uid="mode:baseline",
        )
        analysis = AnalysisResult(
            uid="analysis:1",
            execution_mode_uid=mode.uid,
            top_ops=["matmul"],
            profile_findings=["matmul dominates"],
        )
        profile = ProfileResult(uid="profile:1", execution_mode_uid=mode.uid)
        strategy1 = OptimizationStrategy(
            uid="strategy:1",
            execution_mode_uid=mode.uid,
            local_speedup_ratio=1.2,
            measures=["fuse op"],
            prompt_instruction="apply strategy 1",
            extra={"kind": "operator_fusion"},
        )
        strategy2 = OptimizationStrategy(
            uid="strategy:2",
            execution_mode_uid=mode.uid,
            local_speedup_ratio=1.05,
            measures=["patch forward"],
            prompt_instruction="apply strategy 2",
            extra={"kind": "forward_patch"},
        )
        change = ChangeRecord(
            mode_uid=mode.uid,
            strategy_uid=strategy1.uid,
            kind="operator_fusion",
            summary="changed build_model",
            details="test details",
            files=["build_model.py"],
        )

        trace_store.record_agent_io(
            "analysis-agent",
            "analysis",
            "prompt",
            '{"hints":["matmul"]}',
            {"hints": ["matmul"]},
            "ok",
            refs={"mode_uid": mode.uid, "analysis_uid": analysis.uid},
        )
        trace_store.record_mode(mode)
        trace_store.record_analysis_result(analysis, profile)
        trace_store.record_strategies(analysis, [strategy1, strategy2])
        trace_store.record_apply_action(
            strategy=strategy1,
            mode=mode,
            base_mode=None,
            phase="apply",
            status="applied",
            change_record=change,
            workspace_dir=workspace,
            forward_gate_ok=True,
        )
        trace_store.record_apply_action(
            strategy=strategy1,
            mode=mode,
            base_mode=None,
            phase="apply",
            status="failed",
            error="forward gate failed",
        )
        trace_store.record_evaluation(
            mode.uid, "benchmark", ok=True, metric_name="latency_ms", metric_value=10.0
        )
        trace_store.record_reward(
            mode.uid, parent_mode_uid=mode.parent_uid, strategy_uid=strategy1.uid, value=0.25
        )
        trace_store.record_reward(
            "mode:other", parent_mode_uid=mode.parent_uid, strategy_uid=strategy2.uid, value=-0.5
        )
        trace_store.finalize_run_trace({"stop_reason": "unit"}, {"speedup": 1.25})

        trace_db = Path(run_dir) / "trace.sqlite"
        self.assertTrue(trace_db.exists())
        conn = sqlite3.connect(trace_db)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            self.assertTrue(
                {
                    "runs",
                    "modes",
                    "agent_calls",
                    "analyses",
                    "strategies",
                    "apply_actions",
                    "evaluations",
                    "artifacts",
                    "rewards",
                }.issubset(tables)
            )
            self.assertGreater(conn.execute("SELECT count(*) FROM artifacts").fetchone()[0], 0)
        finally:
            conn.close()

        sft = Path(run_dir) / "sft.jsonl"
        strategy_pref = Path(run_dir) / "strategy_pref.jsonl"
        apply_pref = Path(run_dir) / "apply_pref.jsonl"
        rl = Path(run_dir) / "rl.jsonl"
        export_sft_agent_calls(trace_db, sft)
        export_strategy_preferences(trace_db, strategy_pref)
        export_apply_preferences(trace_db, apply_pref)
        export_rl_trajectories(trace_db, rl)

        for path in (sft, strategy_pref, apply_pref, rl):
            self.assertTrue(path.exists(), path)
            self.assertTrue(path.read_text(encoding="utf-8").strip(), path)

    def test_call_agent_json_disabled_records_status(self) -> None:
        old_enabled = agent_client.AGENT_ENABLED
        agent_client.AGENT_ENABLED = False
        try:
            run_dir = trace_store.start_run_trace("run:unit-disabled", "unit-model", {})
            self.assertIsNone(agent_client.call_agent_json("analysis-agent", "return json"))
            trace_store.finalize_run_trace({"stop_reason": "disabled"}, {})

            conn = sqlite3.connect(Path(run_dir) / "trace.sqlite")
            try:
                status = conn.execute("SELECT status FROM agent_calls").fetchone()[0]
            finally:
                conn.close()
            self.assertEqual(status, "disabled")
        finally:
            agent_client.AGENT_ENABLED = old_enabled


if __name__ == "__main__":
    os.environ.setdefault("ASCENDFAST_USE_LLM_AGENT", "0")
    unittest.main()
