"""Executable checks for the Python examples in Module 14.

The chapter is the source of truth: tests extract named functions from its
Python fences and exercise the code learners are asked to copy.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import re
from statistics import mean
import time
from types import SimpleNamespace
from typing import Callable
import unittest


README = Path(__file__).with_name("README.md")


def load_example_symbols(*names: str):
    """Compile named functions or classes directly from one Python fence."""
    markdown = README.read_text(encoding="utf-8")
    for block in re.findall(r"```python\n(.*?)```", markdown, flags=re.DOTALL):
        tree = ast.parse(block)
        selected = [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name in names
        ]
        if len(selected) == len(names):
            namespace = {
                "Any": object,
                "Callable": Callable,
                "Counter": Counter,
                "LoopResult": object,
                "dataclass": dataclass,
                "field": field,
                "hashlib": hashlib,
                "mean": mean,
                "time": time,
            }
            module = ast.Module(body=selected, type_ignores=[])
            exec(compile(module, README, "exec"), namespace)
            return tuple(namespace[name] for name in names)
    raise AssertionError(f"Python examples {names!r} were not found together in {README}")


def load_example_function(name: str):
    return load_example_symbols(name)[0]


def attempt(score: float = 1.0):
    return SimpleNamespace(
        input_tokens=1_000,
        cache_read_tokens=0,
        cache_creation_5m_tokens=0,
        cache_creation_1h_tokens=0,
        output_tokens=100,
        score=score,
    )


def result(outcome: str, accepted_at: int | None, attempts: int):
    return SimpleNamespace(
        outcome=outcome,
        accepted_at=accepted_at,
        attempts=[attempt() for _ in range(attempts)],
        cost_usd=0.01 * attempts,
    )


class VerifierAsymmetryExampleTests(unittest.TestCase):
    def test_tpr_must_exceed_fpr_to_improve_accepted_precision(self):
        """The worked condition must distinguish useful and inverted gates."""
        loop_quality = load_example_function("loop_quality")

        useful = loop_quality(p_generate=0.70, tpr=0.55, fpr=0.35, k=3)
        inverted = loop_quality(p_generate=0.70, tpr=0.25, fpr=0.40, k=3)

        self.assertGreater(useful["precision_of_accepted"], 0.70)
        self.assertLess(inverted["precision_of_accepted"], 0.70)

    def test_reviewer_refusal_blocks_without_claiming_refutation(self):
        adversarially_verify = load_example_function("adversarially_verify")
        adversarially_verify.__globals__.update({
            "MODEL": "test-model",
            "REFUTE_SCHEMA": {},
            "client": SimpleNamespace(
                messages=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(stop_reason="refusal")
                )
            ),
        })

        report = adversarially_verify("claim", n=1)

        self.assertEqual(report["status"], "UNMEASURED")
        self.assertEqual(report["coverage"], 0.0)
        self.assertEqual(report["reviews"][0]["verdict"], "UNKNOWN")


class LoopMetricsExampleTests(unittest.TestCase):
    def test_marginal_yield_uses_only_runs_that_reached_the_attempt(self):
        """An early-exhausted run must not dilute attempt 2's conversion rate."""
        loop_metrics = load_example_function("loop_metrics")
        runs = [
            result("accepted", accepted_at=1, attempts=1),
            result("accepted", accepted_at=2, attempts=2),
            result("exhausted", accepted_at=None, attempts=1),
        ]

        metrics = loop_metrics(runs, k=2)

        self.assertAlmostEqual(metrics["accepted_by_k"], 2 / 3)
        self.assertEqual(metrics["accepted_at"], {1: 1 / 3, 2: 1 / 3})
        self.assertEqual(metrics["reached_attempt"], {1: 1.0, 2: 1 / 3})
        self.assertEqual(metrics["marginal_yield"][2], 1.0)

    def test_metrics_include_preflight_exhaustion_with_zero_attempts(self):
        """Budget-blocked runs belong in rates even though they have no attempt 1."""
        loop_metrics = load_example_function("loop_metrics")
        runs = [
            result("accepted", accepted_at=1, attempts=1),
            result("exhausted", accepted_at=None, attempts=0),
        ]

        metrics = loop_metrics(runs, k=1)

        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["accepted_by_k"], 0.5)
        self.assertEqual(metrics["mean_attempts"], 0.5)

    def test_accepted_by_k_excludes_acceptances_after_k(self):
        """A later success must not be credited to an earlier attempt budget."""
        loop_metrics = load_example_function("loop_metrics")
        runs = [
            result("accepted", accepted_at=1, attempts=1),
            result("accepted", accepted_at=3, attempts=3),
        ]

        metrics = loop_metrics(runs, k=2)

        self.assertEqual(metrics["accepted_by_k"], 0.5)
        self.assertEqual(metrics["accepted_at"], {1: 0.5, 2: 0.0})


class LoopResultExampleTests(unittest.TestCase):
    def test_observability_emits_one_run_row_and_one_row_per_attempt(self):
        """Adding retries must add rows, not ever-wider blocked_by_N columns."""
        Attempt, LoopResult = load_example_symbols("Attempt", "LoopResult")
        attempts = [
            Attempt(1, "draft", False, "grounding", "cite source", 0.4,
                    1_000, 100, 25, "hash-1", cache_read_tokens=800),
            Attempt(2, "revision", True, None, "", 0.9,
                    1_100, 120, 30, "hash-2", cache_read_tokens=900),
        ]
        loop_result = LoopResult(
            run_id="run-1", attempts=attempts, outcome="accepted", accepted_at=2
        )

        run_row, attempt_rows = loop_result.to_rows()

        self.assertEqual(run_row["run_id"], "run-1")
        self.assertEqual(run_row["attempts"], 2)
        self.assertNotIn("blocked_by_1", run_row)
        self.assertEqual([row["attempt_index"] for row in attempt_rows], [1, 2])
        self.assertEqual(attempt_rows[0]["blocked_by"], "grounding")
        self.assertEqual(attempt_rows[0]["cache_read_tokens"], 800)
        self.assertAlmostEqual(loop_result.cost_usd, 0.01685)

    def test_run_loop_records_cache_usage_returned_by_generator(self):
        """The callback contract must carry every token category used for cost."""
        Attempt, LoopResult, run_loop = load_example_symbols(
            "Attempt", "LoopResult", "run_loop"
        )

        loop_result = run_loop(
            run_id="run-cache",
            generate=lambda critique: ("answer", 1_000, 800, 200, 100, 100),
            verify=lambda output: {
                "passed": True,
                "blocked_by": None,
                "critique": "",
                "score": 1.0,
            },
        )

        self.assertEqual(loop_result.attempts[0].cache_read_tokens, 800)
        self.assertEqual(loop_result.attempts[0].cache_creation_5m_tokens, 200)
        self.assertEqual(loop_result.attempts[0].cache_creation_1h_tokens, 100)
        self.assertAlmostEqual(loop_result.cost_usd, 0.01015)

    def test_cost_includes_cache_writes_at_each_ttl(self):
        """Cold or refreshed caches must not disappear from loop cost."""
        Attempt, LoopResult = load_example_symbols("Attempt", "LoopResult")
        cached_attempt = Attempt(
            1, "answer", True, None, "", 1.0,
            1_000, 100, 25, "hash-cache-write",
            cache_read_tokens=800,
            cache_creation_5m_tokens=200,
            cache_creation_1h_tokens=100,
        )
        loop_result = LoopResult(
            run_id="run-cache-write",
            attempts=[cached_attempt],
            outcome="accepted",
            accepted_at=1,
        )

        self.assertAlmostEqual(loop_result.cost_usd, 0.01015)
        _, attempt_rows = loop_result.to_rows()
        self.assertEqual(attempt_rows[0]["cache_creation_5m_tokens"], 200)
        self.assertEqual(attempt_rows[0]["cache_creation_1h_tokens"], 100)

    def test_run_loop_does_not_start_attempt_without_reserved_budget(self):
        """A hard ceiling must be checked before generation spends the money."""
        Attempt, LoopResult, run_loop = load_example_symbols(
            "Attempt", "LoopResult", "run_loop"
        )
        calls = 0

        def generate(critique):
            nonlocal calls
            calls += 1
            return "answer", 1_000, 0, 0, 0, 100

        loop_result = run_loop(
            run_id="run-budget",
            generate=generate,
            verify=lambda output: {"passed": True},
            cost_ceiling_usd=0.01,
            attempt_cost_reserve_usd=0.02,
        )

        self.assertEqual(calls, 0)
        self.assertEqual(loop_result.outcome, "exhausted")
        self.assertEqual(loop_result.attempts, [])

    def test_rejected_verdict_requires_a_blocker_reason(self):
        """A bare False verdict cannot support retry or failure analysis."""
        Attempt, LoopResult, run_loop = load_example_symbols(
            "Attempt", "LoopResult", "run_loop"
        )

        with self.assertRaisesRegex(ValueError, "blocked_by"):
            run_loop(
                run_id="run-unlabeled-rejection",
                generate=lambda critique: ("answer", 1_000, 0, 0, 0, 100),
                verify=lambda output: {"passed": False},
            )


class LayerCorrelationExampleTests(unittest.TestCase):
    def test_shared_gate_holes_use_symmetric_jaccard_overlap(self):
        """Swapping gate A and B must not change their reported miss overlap."""
        layer_correlation = load_example_function("layer_correlation")
        labels = [True, True, True, False]
        layer_a = [False, False, True, True]   # misses defective items 0 and 1
        layer_b = [True, False, False, True]   # misses defective items 1 and 2

        overlap = layer_correlation(labels, layer_a, layer_b)

        self.assertAlmostEqual(overlap["miss_jaccard"], 1 / 3)

    def test_all_clean_labels_have_zero_miss_rates(self):
        """A clean calibration slice has no true defects for either gate to miss."""
        layer_correlation = load_example_function("layer_correlation")

        overlap = layer_correlation(
            labels=[False, False],
            layer_a=[True, True],
            layer_b=[True, False],
        )

        self.assertEqual(overlap["a_miss_rate"], 0.0)
        self.assertEqual(overlap["b_miss_rate"], 0.0)
        self.assertEqual(overlap["miss_jaccard"], 0.0)


class StandaloneExampleTests(unittest.TestCase):
    def test_cheap_gate_labels_rejections_and_passes_valid_output(self):
        cheap_gate = load_example_function("cheap_gate")

        self.assertEqual(cheap_gate("   ")["blocked_by"], "empty")
        self.assertIsNone(cheap_gate("usable output"))

    def test_only_promotion_gate_can_read_the_frozen_holdout(self):
        GoldenSet = load_example_symbols("GoldenSet")[0]
        golden = GoldenSet(visible=["visible"], holdout=["secret"])

        with self.assertRaises(PermissionError):
            golden.read_holdout("optimizer")
        self.assertEqual(golden.read_holdout("promotion_gate"), ["secret"])
        self.assertEqual(golden._holdout_reads, 1)


class PromotionGateExampleTests(unittest.TestCase):
    def test_overlapping_quality_interval_cannot_promote(self):
        """A raw score increase is not a demonstrated improvement when its CI crosses zero."""
        can_promote = load_example_function("can_promote")
        candidate = {
            "holdout_score": 0.82,
            "holdout_delta_ci_low": -0.01,
            "minimum_effect": 0.01,
            "cost_per_accepted": 0.10,
        }
        baseline = {"holdout_score": 0.80, "cost_per_accepted": 0.10}

        promoted, reason = can_promote(candidate, baseline, guardrails={})

        self.assertFalse(promoted)
        self.assertIn("uncertain", reason)

    def test_product_owned_cost_limit_blocks_an_expensive_candidate(self):
        """The reusable gate must not hide a fixed 25% cost policy."""
        can_promote = load_example_function("can_promote")
        candidate = {
            "holdout_score": 0.84,
            "holdout_delta_ci_low": 0.02,
            "minimum_effect": 0.01,
            "cost_per_accepted": 1.21,
            "max_cost_multiplier": 1.10,
        }
        baseline = {"holdout_score": 0.80, "cost_per_accepted": 1.00}

        promoted, reason = can_promote(candidate, baseline, guardrails={})

        self.assertFalse(promoted)
        self.assertIn("cost regression", reason)


if __name__ == "__main__":
    unittest.main()
