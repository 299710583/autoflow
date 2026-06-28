from __future__ import annotations

from pathlib import Path
import unittest

from autoflow.executor.ssh_executor import CommandResult
from autoflow.flows.models import AssessmentFlow
from autoflow.tools.dispatcher import ToolDispatcher


class FakeScriptRunner:
    def __init__(self) -> None:
        self.calls = []

    def run_script(self, **kwargs):
        self.calls.append(kwargs)
        return CommandResult(
            command=["python3", "script.py"],
            command_text="fake-script",
            exit_code=0,
            stdout='{"ok": true, "evidence": "custom probe ran"}',
            stderr="",
        )


class ToolDispatcherPathTests(unittest.TestCase):
    def test_source_scan_path_is_limited_to_autoflow_data_dirs(self) -> None:
        allowed = Path("data/source_audit/dispatcher-path-test")
        allowed.mkdir(parents=True, exist_ok=True)

        resolved = ToolDispatcher()._safe_container_scan_path(str(allowed))

        self.assertTrue(resolved.endswith("data/source_audit/dispatcher-path-test"))

    def test_source_scan_path_rejects_outside_workspace_data_dirs(self) -> None:
        outside = Path.cwd().parent

        with self.assertRaisesRegex(ValueError, "must be under one of"):
            ToolDispatcher()._safe_container_scan_path(str(outside))

    def test_custom_validation_script_executes_through_script_runner(self) -> None:
        runner = FakeScriptRunner()
        dispatcher = ToolDispatcher(script_runner=runner)
        state = {
            "flow": AssessmentFlow(name="demo", target_scope=["http://example.test"]),
            "target_scope": ["http://example.test"],
            "tool_observations": [],
            "executed_tasks": [],
        }

        result = dispatcher.dispatch(
            "run_script__custom_validation",
            {
                "target": "http://example.test",
                "script_source": "import json\nprint(json.dumps({'target': TARGET, 'ok': True}))",
                "policy_profile": "low_readonly_http",
                "timeout": "30",
            },
            state,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(runner.calls[0]["target"], "http://example.test")
        self.assertIn("TARGET = 'http://example.test'", runner.calls[0]["script"])
        self.assertEqual(runner.calls[0]["policy_profile"], "low_readonly_http")
        self.assertEqual(state["executed_tasks"][-1]["task"]["profile"], "custom_validation")
        self.assertEqual(state["tool_observations"][-1]["profile"], "custom_validation")


if __name__ == "__main__":
    unittest.main()
