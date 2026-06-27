from __future__ import annotations

import asyncio
import unittest

from autoflow.agents.executor import ValidationExecutorAgent
from autoflow.agents.validation_reasoner import ValidationReasoningDecision
from autoflow.flows.models import (
    Finding,
    FindingConfidence,
    FindingSeverity,
    RiskLevel,
    TestPlanAction,
    ValidationPlan,
    ValidationResultStatus,
)


class ValidationExecutorAgentTests(unittest.TestCase):
    def test_marks_precovered_validation_action_completed(self) -> None:
        action = TestPlanAction(
            name="Confirm response security headers",
            action_kind="script",
            tool="script_runner",
            profile="security_headers_check",
            target="http://example.test",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            script_template="security_headers_check",
        )
        plan = ValidationPlan(
            finding_id="finding_1",
            target="http://example.test",
            objective="Confirm missing header",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            actions=[action],
        )

        updated = ValidationExecutorAgent()._mark_validation_plan_statuses(
            [plan.model_dump(mode="json")],
            new_results={},
            precovered_actions={action.id},
        )

        self.assertEqual(updated[0]["status"], "completed")
        self.assertEqual(updated[0]["execution_results"][0]["action_id"], action.id)
        self.assertIn("already executed", updated[0]["execution_results"][0]["summary"])

    def test_evaluates_api_validation_result_and_updates_finding(self) -> None:
        finding = Finding(
            title="Exposed API endpoint",
            severity=FindingSeverity.MEDIUM,
            target="http://example.test/api",
            description="API endpoint exposed.",
            evidence=["nuclei observed API"],
            metadata={"category": "api_exposure"},
        )
        action = TestPlanAction(
            name="Probe API",
            action_kind="script",
            tool="script_runner",
            profile="api_endpoint_probe",
            target="http://example.test/api",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            script_template="api_endpoint_probe",
        )
        plan = ValidationPlan(
            finding_id=finding.id,
            target="http://example.test/api",
            objective="Verify API exposure",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            actions=[action],
            metadata={"finding": finding.model_dump(mode="json"), "category": "api_exposure"},
        ).model_dump(mode="json")
        new_results = {
            action.id: {
                "action_id": action.id,
                "status": "completed",
                "summary": "script_runner completed",
                "stdout": '{"status": 200, "content_type": "application/json", "json_keys": ["user"]}',
                "stderr": "",
            }
        }
        agent = ValidationExecutorAgent()
        updated_plans = agent._mark_validation_plan_statuses([plan], new_results, set())
        validation_results = agent._build_validation_results(updated_plans, new_results)
        state = {"findings": [finding.model_dump(mode="json")]}

        agent._apply_validation_results_to_findings(state, validation_results)

        self.assertEqual(validation_results[0].status.value, "validated")
        self.assertEqual(state["findings"][0]["status"], "validated")
        self.assertIn("validation_result_ids", state["findings"][0]["metadata"])

    def test_evaluates_false_positive_when_completed_without_indicators(self) -> None:
        finding = Finding(
            title="Wildcard CORS header",
            severity=FindingSeverity.LOW,
            target="http://example.test",
            description="Candidate CORS issue.",
            evidence=["candidate"],
            metadata={"category": "cors_wildcard"},
        )
        action = TestPlanAction(
            name="Probe CORS",
            action_kind="script",
            tool="script_runner",
            profile="cors_probe",
            target="http://example.test",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            script_template="cors_probe",
        )
        plan = ValidationPlan(
            finding_id=finding.id,
            target="http://example.test",
            objective="Verify CORS",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            actions=[action],
            metadata={"finding": finding.model_dump(mode="json"), "category": "cors_wildcard"},
        ).model_dump(mode="json")
        new_results = {
            action.id: {
                "action_id": action.id,
                "status": "completed",
                "summary": "script_runner completed",
                "stdout": '{"access_control_allow_origin": "https://example.test"}',
                "stderr": "",
            }
        }
        agent = ValidationExecutorAgent()
        updated_plans = agent._mark_validation_plan_statuses([plan], new_results, set())
        validation_results = agent._build_validation_results(updated_plans, new_results)
        state = {"findings": [finding.model_dump(mode="json")]}

        agent._apply_validation_results_to_findings(state, validation_results)

        self.assertEqual(validation_results[0].status.value, "false_positive")
        self.assertEqual(state["findings"][0]["status"], "false_positive")

    def test_evaluates_cors_json_header_as_validated(self) -> None:
        finding = Finding(
            title="Wildcard CORS header",
            severity=FindingSeverity.LOW,
            target="http://example.test",
            description="Candidate CORS issue.",
            evidence=["candidate"],
            metadata={"category": "cors_wildcard"},
        )
        action = TestPlanAction(
            name="Confirm response security headers",
            action_kind="script",
            tool="script_runner",
            profile="security_headers_check",
            target="http://example.test",
            risk_level=RiskLevel.LOW,
            requires_approval=False,
            script_template="security_headers_check",
        )
        plan = ValidationPlan(
            finding_id=finding.id,
            target="http://example.test",
            objective="Verify CORS",
            risk_level=RiskLevel.MEDIUM,
            requires_approval=False,
            actions=[action],
            metadata={"finding": finding.model_dump(mode="json"), "category": "cors_wildcard"},
        ).model_dump(mode="json")
        new_results = {
            action.id: {
                "action_id": action.id,
                "status": "completed",
                "summary": "script_runner completed",
                "stdout": '{"headers": {"access-control-allow-origin": "*"}, "status": 200}',
                "stderr": "",
            }
        }
        agent = ValidationExecutorAgent()
        updated_plans = agent._mark_validation_plan_statuses([plan], new_results, set())
        validation_results = agent._build_validation_results(updated_plans, new_results)

        self.assertEqual(validation_results[0].status.value, "validated")

    def test_react_decision_overrides_rule_fallback(self) -> None:
        finding = Finding(
            title="Exposed API endpoint",
            severity=FindingSeverity.MEDIUM,
            target="http://example.test/api",
            description="API endpoint exposed.",
            evidence=["candidate"],
            metadata={"category": "api_exposure"},
        )
        action = TestPlanAction(
            name="Probe API",
            action_kind="script",
            tool="script_runner",
            profile="api_endpoint_probe",
            target="http://example.test/api",
            risk_level=RiskLevel.MEDIUM,
            script_template="api_endpoint_probe",
        )
        plan = ValidationPlan(
            finding_id=finding.id,
            target="http://example.test/api",
            objective="Verify API exposure",
            risk_level=RiskLevel.MEDIUM,
            actions=[action],
            metadata={"finding": finding.model_dump(mode="json"), "category": "api_exposure"},
        ).model_dump(mode="json")
        action_results = [
            {
                "action_id": action.id,
                "status": "completed",
                "summary": "script_runner completed without obvious positive indicator",
                "stdout": "{}",
                "stderr": "",
            }
        ]
        decision = ValidationReasoningDecision(
            decision=ValidationResultStatus.VALIDATED,
            confidence=FindingConfidence.HIGH,
            reasoning="The endpoint is reachable and returns unauthenticated JSON data.",
            impact="Unauthenticated API exposure is confirmed.",
            evidence=["HTTP 200 with unauthenticated JSON response."],
            reproduction_steps=["GET http://example.test/api without authentication."],
            raw={"decision": "confirmed", "confidence": "high"},
            tool_results=[{"ok": True, "tool_call": "curl", "summary": "HTTP 200"}],
        )

        result = ValidationExecutorAgent(use_validation_react=False)._evaluate_validation_plan(
            plan,
            action_results,
            react_decision=decision,
        )

        self.assertEqual(result.status, ValidationResultStatus.VALIDATED)
        self.assertEqual(result.confidence, FindingConfidence.HIGH)
        self.assertEqual(result.reasoning, decision.reasoning)
        self.assertIn("HTTP 200 with unauthenticated JSON response.", result.evidence)
        self.assertEqual(result.metadata["decision_source"], "validation_react")
        self.assertEqual(result.metadata["react_decision"]["decision"], "confirmed")

    def test_build_validation_results_records_react_reasoning(self) -> None:
        class FakeReasoner:
            def reason(self, *, state, plan, action_results):
                return ValidationReasoningDecision(
                    decision=ValidationResultStatus.INCONCLUSIVE,
                    confidence=FindingConfidence.MEDIUM,
                    reasoning="Need a response body sample before confirmation.",
                    missing_evidence=["response body sample"],
                    next_actions=[
                        {
                            "name": "Fetch endpoint body",
                            "action_kind": "tool",
                            "tool": "curl",
                            "target": plan["target"],
                            "risk_level": "low",
                        }
                    ],
                    raw={"decision": "need_more_evidence"},
                )

        finding = Finding(
            title="Exposed API endpoint",
            severity=FindingSeverity.MEDIUM,
            target="http://example.test/api",
            description="API endpoint exposed.",
            evidence=["candidate"],
            metadata={"category": "api_exposure"},
        )
        action = TestPlanAction(
            name="Probe API",
            action_kind="script",
            tool="script_runner",
            profile="api_endpoint_probe",
            target="http://example.test/api",
            risk_level=RiskLevel.MEDIUM,
            script_template="api_endpoint_probe",
        )
        plan = ValidationPlan(
            finding_id=finding.id,
            target="http://example.test/api",
            objective="Verify API exposure",
            risk_level=RiskLevel.MEDIUM,
            actions=[action],
            metadata={"finding": finding.model_dump(mode="json"), "category": "api_exposure"},
        ).model_dump(mode="json")
        new_results = {
            action.id: {
                "action_id": action.id,
                "status": "completed",
                "summary": "script_runner completed",
                "stdout": "{}",
                "stderr": "",
            }
        }
        state = {"rules_of_engagement": {"validation_react_enabled": True}}
        agent = ValidationExecutorAgent(validation_reasoner=FakeReasoner(), use_validation_react=True)
        updated_plans = agent._mark_validation_plan_statuses([plan], new_results, set())

        results = agent._build_validation_results(updated_plans, new_results, state=state)

        self.assertEqual(results[0].status, ValidationResultStatus.INCONCLUSIVE)
        self.assertEqual(results[0].metadata["decision_source"], "validation_react")
        self.assertEqual(state["validation_react_results"][0]["decision"], "inconclusive")
        self.assertEqual(state["validation_next_actions"][0]["tool"], "curl")

    def test_agentic_react_validation_updates_finding(self) -> None:
        class FakeReasoner:
            def validate(self, *, state, finding, plan, previous_results):
                return ValidationReasoningDecision(
                    decision=ValidationResultStatus.VALIDATED,
                    confidence=FindingConfidence.HIGH,
                    reasoning="The ReAct loop fetched the endpoint and observed unauthenticated JSON.",
                    impact="Unauthenticated API data exposure is confirmed.",
                    evidence=["curl returned HTTP 200 with unauthenticated JSON data."],
                    reproduction_steps=["GET http://example.test/api without credentials."],
                    raw={"decision": "confirmed", "confidence": "high"},
                    tool_results=[
                        {
                            "ok": True,
                            "tool_call": "run_curl__get_with_headers",
                            "summary": "curl completed: HTTP/1.1 200 OK",
                            "result": {"action_id": "action_react_1"},
                        }
                    ],
                    messages=[{"role": "assistant", "content": "{}"}],
                )

        finding = Finding(
            title="Exposed API endpoint",
            severity=FindingSeverity.MEDIUM,
            target="http://example.test/api",
            description="API endpoint exposed.",
            evidence=["candidate"],
            metadata={"category": "api_exposure"},
        )
        plan = ValidationPlan(
            finding_id=finding.id,
            target=finding.target,
            objective="Verify API exposure",
            risk_level=RiskLevel.MEDIUM,
            actions=[],
            metadata={"finding": finding.model_dump(mode="json"), "category": "api_exposure"},
        ).model_dump(mode="json")
        state = {
            "findings": [finding.model_dump(mode="json")],
            "validation_plans": [plan],
            "rules_of_engagement": {"validation_react_agent_enabled": True},
        }
        agent = ValidationExecutorAgent(validation_reasoner=FakeReasoner(), use_validation_react=True)

        updated_state = asyncio.run(agent.run(state))

        self.assertEqual(updated_state["findings"][0]["status"], "validated")
        self.assertEqual(updated_state["validation_results"][0]["metadata"]["decision_source"], "validation_react_agent")
        self.assertEqual(updated_state["validation_results"][0]["executed_action_ids"], ["action_react_1"])
        self.assertEqual(updated_state["validation_react_results"][0]["decision"], "validated")

    def test_agentic_react_requires_tool_call_for_confirmation(self) -> None:
        finding = Finding(
            title="Exposed API endpoint",
            severity=FindingSeverity.MEDIUM,
            target="http://example.test/api",
            description="API endpoint exposed.",
            evidence=["candidate"],
            metadata={"category": "api_exposure"},
        )
        decision = ValidationReasoningDecision(
            decision=ValidationResultStatus.VALIDATED,
            confidence=FindingConfidence.HIGH,
            reasoning="Looks confirmed.",
            evidence=["Model assertion without a validation tool call."],
            raw={"decision": "confirmed"},
            tool_results=[],
        )

        result = ValidationExecutorAgent(use_validation_react=False)._validation_result_from_react_decision(
            finding=finding.model_dump(mode="json"),
            plan={},
            decision=decision,
        )

        self.assertEqual(result.status, ValidationResultStatus.INCONCLUSIVE)
        self.assertIn("did not execute a tool call", result.reasoning)

    def test_prioritizes_high_value_validation_actions_with_budget(self) -> None:
        info_finding = Finding(
            title="Informational x-recruiting header exposed",
            severity=FindingSeverity.INFO,
            target="http://example.test",
            description="Header hint.",
            evidence=["x-recruiting"],
            metadata={"category": "informational_header:x-recruiting"},
        )
        api_finding = Finding(
            title="Exposed API endpoint",
            severity=FindingSeverity.MEDIUM,
            target="http://example.test/api/Challenges",
            description="API endpoint exposed.",
            evidence=["api"],
            metadata={"category": "api_exposure"},
        )
        debug_finding = Finding(
            title="Debug endpoint",
            severity=FindingSeverity.MEDIUM,
            target="http://example.test/metrics",
            description="Metrics exposed.",
            evidence=["metrics"],
            metadata={"category": "debug_endpoint_exposed"},
        )

        info_action = TestPlanAction(
            name="Confirm info header",
            action_kind="web_recon",
            tool="web_recon",
            profile="fetch_page",
            target="http://example.test",
            risk_level=RiskLevel.LOW,
        )
        api_action = TestPlanAction(
            name="Probe API",
            action_kind="script",
            tool="script_runner",
            profile="api_endpoint_probe",
            target="http://example.test/api/Challenges",
            risk_level=RiskLevel.MEDIUM,
            script_template="api_endpoint_probe",
        )
        debug_action = TestPlanAction(
            name="Probe debug endpoint",
            action_kind="script",
            tool="script_runner",
            profile="debug_endpoint_probe",
            target="http://example.test/metrics",
            risk_level=RiskLevel.MEDIUM,
            script_template="debug_endpoint_probe",
        )
        plans = [
            ValidationPlan(
                finding_id=info_finding.id,
                target=info_finding.target,
                objective="Confirm informational header.",
                risk_level=RiskLevel.LOW,
                actions=[info_action],
                metadata={"finding": info_finding.model_dump(mode="json"), "category": "informational_header:x-recruiting"},
            ).model_dump(mode="json"),
            ValidationPlan(
                finding_id=debug_finding.id,
                target=debug_finding.target,
                objective="Verify debug endpoint.",
                risk_level=RiskLevel.MEDIUM,
                actions=[debug_action],
                metadata={"finding": debug_finding.model_dump(mode="json"), "category": "debug_endpoint_exposed"},
            ).model_dump(mode="json"),
            ValidationPlan(
                finding_id=api_finding.id,
                target=api_finding.target,
                objective="Verify API exposure.",
                risk_level=RiskLevel.MEDIUM,
                actions=[api_action],
                metadata={"finding": api_finding.model_dump(mode="json"), "category": "api_exposure"},
            ).model_dump(mode="json"),
        ]

        candidates = ValidationExecutorAgent()._candidate_actions(
            {
                "validation_plans": plans,
                "validation_action_budget": 2,
                "executed_tasks": [],
                "executed_action_fingerprints": [],
            }
        )

        self.assertEqual([item["finding_id"] for item in candidates], [api_finding.id, debug_finding.id])


if __name__ == "__main__":
    unittest.main()
