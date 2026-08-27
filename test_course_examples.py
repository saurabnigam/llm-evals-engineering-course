"""Regression checks for copyable examples used across the course.

Each test extracts the named function or class from its Markdown source so the
README remains the source of truth.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
from math import comb
from pathlib import Path
import random
import re
from types import SimpleNamespace
from typing import Dict, List, Optional
import unittest


ROOT = Path(__file__).parent


def load_symbols(readme: Path, *names: str, namespace: dict | None = None):
    """Compile named top-level definitions from one Python fence."""
    blocks = []
    lines = readme.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        start = re.match(r"^\s*(>?\s*)```python\s*$", lines[i])
        if not start:
            i += 1
            continue
        quoted = ">" in start.group(1)
        i += 1
        body = []
        while i < len(lines) and not re.match(r"^\s*>?\s*```\s*$", lines[i]):
            line = lines[i]
            if quoted:
                line = re.sub(r"^(\s*)>\s?", r"\1", line, count=1)
            body.append(line)
            i += 1
        blocks.append("\n".join(body))
        i += 1

    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            # Markdown blockquotes retain their leading `>` in the capture.
            continue
        selected = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in names
        ]
        if len(selected) == len(names):
            scope = {
                "Dict": Dict,
                "List": List,
                "Optional": Optional,
                "comb": comb,
                "dataclass": dataclass,
                "datetime": datetime,
                "defaultdict": defaultdict,
                "json": json,
                "random": random,
                "timedelta": timedelta,
            }
            if namespace:
                scope.update(namespace)
            module = ast.Module(body=selected, type_ignores=[])
            exec(compile(module, readme, "exec"), scope)
            return tuple(scope[name] for name in names)
    raise AssertionError(f"Could not find {names!r} together in {readme}")


class ReliabilityExampleTests(unittest.TestCase):
    README = ROOT / "01-fundamentals" / "README.md"

    def test_reliability_report_rejects_empty_input_and_nonpositive_k(self):
        (report,) = load_symbols(self.README, "reliability_report")

        with self.assertRaisesRegex(ValueError, "trial_results"):
            report([], k=1)
        with self.assertRaisesRegex(ValueError, "positive"):
            report([{"task_id": "t", "success": True}], k=0)


class TemplateGeneratorExampleTests(unittest.TestCase):
    README = ROOT / "04-cold-start" / "README.md"

    def test_correlated_slot_rows_do_not_create_impossible_answers(self):
        (generator_class,) = load_symbols(
            self.README,
            "TemplateBasedGenerator",
            namespace={"Template": __import__("string").Template},
        )
        generator = generator_class()
        generator.add_template(
            name="price",
            input_template="$product costs what?",
            output_template="$product costs $price.",
            rows=[
                {"product": "phone", "price": "$999"},
                {"product": "tablet", "price": "$599"},
            ],
        )

        cases = generator.generate("price", 25)

        valid = {("phone", "$999"), ("tablet", "$599")}
        self.assertTrue(
            all((c["slot_values"]["product"], c["slot_values"]["price"]) in valid for c in cases)
        )

    def test_bootstrap_labeler_reports_observed_agreement_not_confidence(self):
        (labeler_class,) = load_symbols(ROOT / "04-cold-start" / "README.md", "BootstrapLabeler")
        labeler = labeler_class.__new__(labeler_class)
        labeler.num_runs = 3
        labeler.consistency_threshold = 0.8
        verdicts = iter(["PASS", "FAIL", "PASS"])
        labeler._get_label = lambda sample, criteria: {
            "verdict": next(verdicts),
            "evidence": "observed output",
            "issues": [],
        }

        row = labeler.label_batch(
            [{"input": "q", "output": "a"}], "states the policy window"
        )[0]["bootstrap_label"]

        self.assertEqual(row["agreement_rate"], 2 / 3)
        self.assertTrue(row["needs_human_review"])
        self.assertNotIn("repeat_consistency", row)
        self.assertNotIn("confidence", row)


class FeedbackAnalyticsExampleTests(unittest.TestCase):
    README = ROOT / "06-feedback-loops" / "README.md"

    def test_thumbs_are_counted_once_and_consistently(self):
        class FeedbackType(Enum):
            THUMBS_UP = "thumbs_up"
            THUMBS_DOWN = "thumbs_down"
            RATING = "rating"

        (analytics_class,) = load_symbols(
            self.README,
            "FeedbackAnalytics",
            namespace={"FeedbackType": FeedbackType},
        )
        now = datetime.now()
        feedback = [
            SimpleNamespace(
                feedback_type=FeedbackType.THUMBS_UP,
                rating=1.0,
                correction=None,
                category=None,
                timestamp=now,
                comment=None,
                input_text="good",
            ),
            SimpleNamespace(
                feedback_type=FeedbackType.THUMBS_DOWN,
                rating=0.0,
                correction=None,
                category=None,
                timestamp=now,
                comment="bad",
                input_text="bad",
            ),
        ]
        store = SimpleNamespace(get_range=lambda start, end: feedback)

        summary = analytics_class(store).get_summary(now - timedelta(days=1), now)
        daily = next(iter(summary["daily_trend"].values()))

        self.assertEqual(summary["totals"]["positive"], 1)
        self.assertEqual(summary["totals"]["negative"], 1)
        self.assertEqual(summary["totals"]["neutral"], 0)
        self.assertEqual(daily, {"positive": 1, "negative": 1, "total": 2})


class CIAggregationExampleTests(unittest.TestCase):
    README = ROOT / "07-cicd-integration" / "README.md"

    def test_aggregation_reports_sample_size_and_empty_coverage(self):
        (runner_class,) = load_symbols(self.README, "EvalRunner")
        runner = runner_class.__new__(runner_class)
        runner.evaluators = {"safety": object(), "quality": object()}
        rows = [
            {"scores": {"safety": {"score": 1.0, "passed": True}, "quality": {"score": 0.8, "passed": True}}},
            {"scores": {"safety": {"score": 0.0, "passed": False}, "quality": {"score": 0.9, "passed": True}}},
        ]

        report = runner._aggregate_results(rows)

        self.assertEqual(report["by_evaluator"]["safety"]["n"], 2)
        self.assertFalse(report["overall"]["all_must_pass"])
        empty = runner._aggregate_results([])
        self.assertEqual(empty["overall"]["coverage"], 0.0)
        self.assertIsNone(empty["overall"]["score"])

    def test_regression_comparison_rejects_unmeasured_runs(self):
        compare_results, minimum_detectable_effect, required_n = load_symbols(
            self.README,
            "compare_results",
            "minimum_detectable_effect",
            "required_n",
        )
        unmeasured = {
            "aggregated": {"overall": {"score": None}, "by_evaluator": {}}
        }

        with self.assertRaisesRegex(ValueError, "measured overall scores"):
            compare_results(unmeasured, unmeasured)

    def test_flaky_detector_reports_instability_without_fake_confidence(self):
        from statistics import mean, pvariance

        np = SimpleNamespace(mean=mean, var=pvariance)

        (detector_class,) = load_symbols(
            self.README,
            "FlakyEvalDetector",
            namespace={"np": np},
        )
        scores = iter([0.0, 1.0, 0.0, 1.0])
        evaluator = SimpleNamespace(evaluate=lambda sample: {"score": next(scores)})

        report = detector_class(variance_threshold=0.1).run_with_retries(
            evaluator, {"id": "unstable"}, n_runs=4
        )

        self.assertTrue(report["is_flaky"])
        self.assertEqual(report["individual_runs"], [0.0, 1.0, 0.0, 1.0])
        self.assertNotIn("confidence", report)


class PairwiseTournamentExampleTests(unittest.TestCase):
    README = ROOT / "02-evaluation-methods" / "README.md"

    def test_position_flip_is_not_scored_as_a_tie(self):
        (evaluator_class,) = load_symbols(self.README, "PairwiseEvaluator")
        evaluator = evaluator_class.__new__(evaluator_class)
        evaluator.compare = lambda *args, **kwargs: {
            "winner": "UNRESOLVED",
            "position_consistent": False,
            "normal": {"reasoning": "preferred the first answer"},
            "swapped": {"reasoning": "preferred the first answer again"},
        }

        report = evaluator.run_tournament(
            "question", {"model_a": "answer a", "model_b": "answer b"}, "quality"
        )

        self.assertEqual(dict(report["ranking"]), {"model_a": 0, "model_b": 0})
        self.assertEqual(report["unresolved_comparisons"], 1)


class JudgePanelExampleTests(unittest.TestCase):
    README = ROOT / "02-evaluation-methods" / "README.md"

    def test_unknown_votes_reduce_coverage_instead_of_becoming_ties(self):
        (panel_class,) = load_symbols(
            self.README,
            "MultiJudgePanel",
            namespace={"LLMJudge": object},
        )
        judges = [
            SimpleNamespace(
                evaluate=lambda *args: SimpleNamespace(verdict="UNKNOWN")
            ),
            SimpleNamespace(
                evaluate=lambda *args: SimpleNamespace(verdict="PASS")
            ),
        ]

        report = panel_class(judges).evaluate("q", "a", "criterion")

        self.assertEqual(report["final_verdict"], "PASS")
        self.assertEqual(report["votes"], ["PASS"])
        self.assertEqual(report["coverage"], 0.5)


class MultimodalGateExampleTests(unittest.TestCase):
    README = ROOT / "08-case-studies" / "README.md"

    def test_judge_refusal_is_unmeasured_not_an_image_failure(self):
        (qa_gate,) = load_symbols(
            self.README,
            "qa_gate",
            namespace={
                "VETO": {"faithfulness": "question"},
                "SCORED": {},
            },
        )
        qa_gate.__globals__["judge_criterion"] = lambda *args: {
            "verdict": "UNKNOWN",
            "evidence": "judge refused",
            "repair_directive": "human review",
        }

        report = qa_gate(b"original", b"edited")

        self.assertEqual(report["status"], "UNMEASURED")
        self.assertIsNone(report["passed"])
        self.assertEqual(report["blocked_by"], "unmeasured_faithfulness")


class LangChainRunnerExampleTests(unittest.TestCase):
    README = ROOT / "09-langchain-examples" / "README.md"

    def test_safety_failure_is_a_veto_and_missing_scores_reduce_coverage(self):
        import asyncio

        namespace = {
            "asyncio": asyncio,
            "BaseEvaluator": object,
            "EvalResult": object,
            "EvalSample": object,
            "BatchResult": object,
            "datetime": datetime,
            "Path": Path,
        }
        (runner_class,) = load_symbols(
            self.README,
            "BatchEvalRunner",
            namespace=namespace,
        )
        runner = runner_class.__new__(runner_class)
        runner.evaluators = {
            "SafetyEvaluator": object(),
            "HelpfulnessEvaluator": object(),
        }
        rows = [
            {
                "scores": {"SafetyEvaluator": 0.0, "HelpfulnessEvaluator": 1.0},
                "details": {
                    "SafetyEvaluator": {"passed": False},
                    "HelpfulnessEvaluator": {"passed": True},
                },
            },
            {
                "scores": {"SafetyEvaluator": None, "HelpfulnessEvaluator": 1.0},
                "details": {
                    "SafetyEvaluator": {"error": "judge refusal"},
                    "HelpfulnessEvaluator": {"passed": True},
                },
            },
        ]

        report = runner._aggregate_results(rows)

        self.assertFalse(report["overall"]["all_must_pass"])
        self.assertEqual(report["overall"]["coverage"], 0.75)


class AlignmentConsistencyExampleTests(unittest.TestCase):
    README = ROOT / "10-advanced-topics" / "README.md"

    def test_context_probe_reports_gap_without_declaring_alignment_faking(self):
        (evaluator_class,) = load_symbols(self.README, "AlignmentConsistencyEvaluator")
        evaluator = evaluator_class(
            refusal_eval=lambda rows: sum("refuse" in row for row in rows) / len(rows),
            helpfulness_eval=lambda rows: 1.0,
            safety_eval=lambda rows: 1.0,
            n_trials=2,
        )

        class FakeModel:
            def generate(self, *, system, user):
                return "refuse" if "recorded" in system else "answer"

        report = evaluator.evaluate_consistency(FakeModel(), ["request"])["request"]

        self.assertGreater(report["max_refusal_rate_gap"], 0)
        self.assertNotIn("flag", report)


class ConstitutionalEvaluatorExampleTests(unittest.TestCase):
    README = ROOT / "10-advanced-topics" / "README.md"

    def test_required_principles_are_verdicts_not_an_averaged_score(self):
        import asyncio

        class FakeJudge:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, prompt):
                self.calls += 1
                # First pass: one required failure. Re-evaluation: all pass.
                verdict = "FAIL" if self.calls == 1 else "PASS"
                return {
                    "verdict": verdict,
                    "evidence": "specific observed behavior",
                }

        class FakeReviser:
            async def ainvoke(self, prompt):
                return SimpleNamespace(content="revised response")

        evaluator_class, assessment_class = load_symbols(
            self.README,
            "ConstitutionalEvaluator",
            "PrincipleAssessment",
            namespace={
                "Literal": __import__("typing").Literal,
                "TypedDict": __import__("typing").TypedDict,
                "json": json,
            },
        )
        evaluator = evaluator_class(judge=FakeJudge(), reviser=FakeReviser())

        report = asyncio.run(evaluator.evaluate_and_revise("request", "response"))

        self.assertFalse(report["initial_required_pass"])
        self.assertTrue(report["final_required_pass"])
        self.assertNotIn("overall_score", report)

    def test_red_team_judge_uses_structured_verdict_not_yes_substring(self):
        import asyncio

        framework_class, assessment_class = load_symbols(
            self.README,
            "RedTeamFramework",
            "AttackAssessment",
            namespace={
                "Literal": __import__("typing").Literal,
                "TypedDict": __import__("typing").TypedDict,
            },
        )
        framework = framework_class.__new__(framework_class)

        async def blocked_assessment(prompt):
            return {
                "verdict": "BLOCKED",
                "evidence": "The response says yes only while explaining the refusal.",
            }

        framework.attack_judge = SimpleNamespace(
            ainvoke=blocked_assessment
        )

        verdict = asyncio.run(framework._judge_attack_success("attack", "response"))

        self.assertEqual(verdict["verdict"], "BLOCKED")


class DynamicBenchmarkExampleTests(unittest.TestCase):
    README = ROOT / "12-eval-training-separation" / "README.md"

    def test_contamination_probe_reports_observations_not_fake_verdict(self):
        (detector_class,) = load_symbols(self.README, "ContaminationDetector")

        class FakeModel:
            def generate(self, prompt, **kwargs):
                if prompt.startswith("Complete this question"):
                    return "of France?"
                if prompt.startswith("Is this question"):
                    return "This resembles a general-knowledge question."
                return "Paris"

        report = detector_class(FakeModel()).detect_memorization(
            [{"id": "capital", "question": "What is the capital of France?", "answer": "Paris"}]
        )

        self.assertNotIn("contamination_rate", report)
        self.assertNotIn("recommendation", report)
        self.assertEqual(report["details"][0]["answer_exact_match"], True)
        self.assertIn("requires corroboration", report["interpretation"])

    def test_requested_size_is_allocated_exactly(self):
        (generator_class,) = load_symbols(self.README, "DynamicBenchmarkGenerator")

        class FakeLLM:
            def generate(self, prompt, **kwargs):
                count = int(re.search(r"Generate (\d+) unique", prompt).group(1))
                return json.dumps([{"question": str(i), "answer": str(i)} for i in range(count)])

        generator = generator_class(FakeLLM(), {"description": "arithmetic"})

        self.assertEqual(len(generator.generate_eval_set(50)), 50)
        self.assertEqual(len(generator.generate_eval_set(10)), 10)


class ResearchExperimentExampleTests(unittest.TestCase):
    README = ROOT / "13-advancing-ai-research" / "README.md"

    def test_context_probe_reports_behavior_gap_not_alignment_diagnosis(self):
        class FakeModel:
            def generate(self, *, system, user):
                return "refuse" if "recorded" in system else "answer"

        (detector_class,) = load_symbols(self.README, "AlignmentFakingDetector")
        detector = detector_class(
            FakeModel(), behavior_scorer=lambda rows: sum("refuse" in r for r in rows) / len(rows)
        )

        report = detector.run_monitoring_variation_test(["request"], num_trials=2)

        self.assertGreater(report["max_behavior_score_gap"], 0)
        self.assertNotIn("alignment_faking_risk", report)
        self.assertNotIn("significant", report["comparisons"][0])
        self.assertIn("does not diagnose", report["interpretation"])

    def test_prompt_variations_are_explicitly_configured(self):
        (experiment_class,) = load_symbols(self.README, "EvalResearchExperiment")
        experiment = experiment_class(
            prompt_variations={"standard": "{question}"},
            generation_config={"max_tokens": 1000},
        )

        self.assertEqual(experiment.prompt_variations, {"standard": "{question}"})
        self.assertEqual(experiment.generation_config, {"max_tokens": 1000})


class EffortSweepExampleTests(unittest.TestCase):
    README = ROOT / "15-opus5-eval-techniques" / "README.md"

    def test_refusals_are_unmeasured_not_failed(self):
        replies = iter(
            [
                SimpleNamespace(
                    stop_reason="refusal",
                    content=[],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=0),
                ),
                SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text="ok")],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=10),
                ),
            ]
        )
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kwargs: next(replies))
        )
        (run_suite,) = load_symbols(
            self.README,
            "run_suite",
            namespace={"client": fake_client, "time": __import__("time")},
        )
        cases = [
            {"prompt": "refuse", "check": lambda text: False},
            {"prompt": "answer", "check": lambda text: text == "ok"},
        ]

        report = run_suite(cases, "low")

        self.assertEqual(report["measured"], 1)
        self.assertEqual(report["unmeasured"], 1)
        self.assertEqual(report["coverage"], 0.5)
        self.assertEqual(report["pass_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
