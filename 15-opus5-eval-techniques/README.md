# Module 15: Evaluating in the Opus 5 Era

> **The eval harness you wrote in 2024 does not run in 2026, and the parts that still run are measuring something subtly different.**
>
> On the current Claude models covered here, `temperature=0` is rejected. Extended thinking with a fixed `budget_tokens` and assistant prefill are also rejected. Thinking defaults changed, so an old eval config may measure a different and more expensive system. Prompt caching and batches can materially reduce cost, while structured outputs can enforce response shape.
>
> This module is the API-layer half of eval engineering: what changed, what breaks, and what is now possible.
>
> **Prerequisites:** Module 02 (judges), Module 05 (scaling and cost), Module 14 (loop engineering).

---

## In Plain English (start here if you don't write the code)

An **eval harness** is the test suite for an AI feature: a set of saved examples, run against the model, scored automatically. The models changed underneath these harnesses in 2026, and three of the changes matter to anyone who reads the resulting numbers — not just to the people maintaining the code.

**1. AI test results are now a range, not a number.** Harnesses used to have a setting that made the model behave as consistently as possible; that setting no longer exists, and it never worked as well as people believed. So the same test suite run twice gives slightly different answers. This is not a bug to be fixed — it is how the technology works, and the fix is to report results the way medicine reports trial results: an estimate plus a margin of error.

The practical consequence is uncomfortable and worth internalizing: **a 100-example test suite genuinely cannot tell 87% apart from 79%.** If your team argues about whether a four-point movement is real, the answer is usually "there is no way to know" — the suite is too small. Detecting a five-point change reliably takes roughly 900 examples.

**2. The model has a "how hard should I think" dial.** It runs from low to max, and it changes both quality and cost—often dramatically and not proportionally. In the illustrative frontier later in this module, the first twelve points cost about 32¢ each and the last increment costs $41.75. Your frontier will differ, so the setting is an empirical product decision; any quality claim that omits it is incomplete.

**3. A refusal must be classified according to the question your eval asks.** If
you are estimating task capability *conditional on an attempted answer*, a
safeguard refusal is **unmeasured** and reduces coverage. If the product must
answer, or the eval is testing correct refusal behavior, that same response is a
measured failure or success. Silently treating every refusal as an ordinary
wrong answer—or silently dropping it—changes the claim.

The rest of the module shows how to implement this and how to measure potential savings rather than assume them. **Terms you'll meet:** *judge* — a model scoring another model's output. *effort* — the think-harder dial. *coverage* — the share of test cases that produced a real result. *contamination* — when the system under test has somehow seen the answers.

Primary references for the changing API claims: [Claude model overview](https://platform.claude.com/docs/en/about-claude/models/overview), [migration guide](https://platform.claude.com/docs/en/about-claude/models/migration-guide), [effort](https://platform.claude.com/docs/en/build-with-claude/effort), [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), and [pricing](https://platform.claude.com/docs/en/about-claude/pricing).

### Opus-era harness checks: what they cover, catch, and enable

| Eval or harness check | What it covers | What it catches | Decision it enables |
|---|---|---|---|
| Repeated trials plus interval | Outcome variability under a pinned configuration | A headline movement that is compatible with sampling noise | Increase sample size, widen the gate, or investigate a real effect |
| Effort sweep | Quality, latency, tokens, and tool turns at each effort level | Paying more for no material gain—or underthinking hard cases | Choose effort by measured value, not model prestige |
| Refusal/coverage accounting | Which cases produced a score and why | Selective missingness hidden inside a pass rate | Report conditional capability separately from product availability |
| Structured judge output | Whether a judge returned the required fields and evidence | Regex/parser failures and missing verdicts | Mark malformed results unmeasured and repair the judge contract |
| Position-flipped, diverse panel | Judge bias across order, lens, and model family | Position bias and correlated blind spots | Keep only panel members that add calibrated coverage |
| Cache telemetry and batch reconciliation | Whether promised cost optimizations actually occurred | Silent cache misses and results joined in the wrong order | Fix the harness before trusting cost or score comparisons |
| Context/memory A/B | The exact long-horizon system configuration | Gains caused by compaction, hidden warm state, or answer carryover | Report configurations separately and isolate trials |
| Migration smoke suite | Request validity, truncation, stop reasons, and served model | API changes that look like capability regressions | Repair the harness before comparing model versions |

---

## 15.1 The Model Landscape for Eval Harnesses

| Model | ID | Context | Max output | Input / Output per MTok | Where it belongs in an eval stack |
|---|---|---|---|---|---|
| Claude Fable 5.1 | `claude-fable-5-1` | 1M | 128K | $10 / $50 (cache read $0.25) | Ceiling-setting; cheaper to replay long transcripts than Fable 5 because of the 0.025× cache-read rate ([pricing](https://platform.claude.com/docs/en/about-claude/pricing)) |
| Claude Fable 5 | `claude-fable-5` | 1M | 128K | $10 / $50 | Ceiling-setting: hardest reference judgments, adversarial verification (legacy tier, still active) |
| **Claude Opus 5** | **`claude-opus-5`** | **1M** | **128K** | **$5 / $25** | **Default judge and arbiter; agent-under-test for hard tasks** |
| Claude Opus 4.8 | `claude-opus-4-8` | 1M | 128K | $5 / $25 | Fallback target on refusals; A/B baseline |
| Claude Sonnet 5 | `claude-sonnet-5` | 1M | 128K | $2 / $10 | High-volume judging where κ against humans holds up |
| Claude Haiku 4.5 | `claude-haiku-4-5` | 200K | 64K | $1 / $5 | First-stage screen in a cascade; deterministic-ish rule checks |

Facts that change harness design, not just the model string:

- **Thinking is on by default on Opus 5.** Omitting `thinking` runs adaptive thinking — unlike Opus 4.8/4.7, where omitting it meant no thinking. If your harness never set `thinking`, it just got more capable *and* more expensive, and `max_tokens` now caps thinking **plus** response text together. A judge with `max_tokens=512` that used to be fine can now truncate.
- **Effort has five levels** — `low`, `medium`, `high` (default), `xhigh`, `max` — set inside `output_config`, not top-level.
- **Disabling thinking is capped at `high` effort.** `thinking: {"type": "disabled"}` with `xhigh` or `max` is a 400, validated per request.
- **Prompt-cache minimum is 512 tokens** on Opus 5 (down from 1024 on Opus 4.8, and 4096 on Opus 4.6/Haiku 4.5). Judge prompts that were previously too short to cache now cache.
- **Opus 5 has its own rate-limit bucket**, separate from the combined Opus 4.x pool. Moving an eval suite over does not inherit your old headroom.
- **Safety classifiers can decline**, returning HTTP 200 with `stop_reason: "refusal"`. This has a specific and nasty consequence for evals — see §15.4.

**September 2026 additions (Fable 5.1 / Mythos 5.1 and platform-wide), all per the [Claude Platform release notes](https://platform.claude.com/docs/en/release-notes/overview):**

- **Fable 5.1 / Mythos 5.1 reject forced tool calls.** `tool_choice: {"type": "any"}` and `{"type": "tool", ...}` now return **HTTP 400** — only `auto` and `none` remain. Any harness that force-calls a grader or JSON-extraction tool on these models must migrate to [strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use) or [structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) (see below, and §15.4).
- **Per-message `effort` (beta, header `mid-conversation-output-config-2026-07-01`).** `output_config.effort` can now change mid-conversation via a `role: "system"` message **without invalidating the prompt cache** — the pattern this enables is cheap-effort exploration turns followed by a single high-effort judge turn, in one cached session, instead of two separate calls.
- **Thinking-block replay got stricter.** On accounts created after Aug 31, 2026, replaying a thinking block after the system prompt, tools, or earlier messages changed now returns a **400** instead of being silently accepted. This is a classic harness bug: mutating prompts between runs while reusing a cached thinking trace from a prior run.
- **Messages API compaction (beta `compact-2026-09-04`).** A new *on-demand* request — separate from your conversation turns, so it can run in the background — returns a signed summary block covering everything you send it; swap that block in for those messages on your next call. Unlike the threshold-triggered compaction in §15.6, on-demand compaction summarizes the whole request you send it, not just an older portion — there is no automatic "keep the last N turns verbatim" split, so decide what to include before calling it.
- **Computer/browser toolsets went GA Aug 19, 2026** as `computer_toolset_20260801` / `browser_toolset_20260801`, with breaking changes from the `computer_20251124` beta. A computer-use eval suite still pinned to the old tool schema needs updating before it will run.

**Fable 5.1 forced-tool-call migration, before/after:**

```python
# Before (Fable 5, Opus 5, etc.) — 400 on Fable 5.1 / Mythos 5.1:
resp = client.messages.create(
    model="claude-fable-5-1", tools=[GRADER_TOOL],
    tool_choice={"type": "tool", "name": "record_verdict"},   # ← 400 on 5.1
    messages=[{"role": "user", "content": judge_prompt}],
)

# After — structured outputs, no forced tool call needed:
resp = client.messages.create(
    model="claude-fable-5-1",
    output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    messages=[{"role": "user", "content": judge_prompt}],
)
```

---

## 15.2 Why One Run Is Not Evidence

For years, the first line of every eval harness was `temperature=0`, and the second was a comment claiming this made runs reproducible. Both are now gone: **`temperature`, `top_p`, and `top_k` are rejected with a 400** on Opus 5, Opus 4.8/4.7, Fable 5, and (for non-default values) Sonnet 5.

This is less of a loss than it looks, because `temperature=0` never guaranteed
identical outputs; it reduced variance without eliminating it. It did not make
**n=1** statistically valid—it merely made repeated differences easier to
overlook. The correct response is to measure run-to-run variation explicitly.

```python
# pip install anthropic scipy
"""Run each eval case n times; report the interval, not the point estimate."""
import statistics
from dataclasses import dataclass
from scipy.stats import beta as beta_dist


def clopper_pearson(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson (exact) bounds — correct near 0% and 100%, unlike normal-approx."""
    alpha = 1 - conf
    lo = beta_dist.ppf(alpha / 2, successes, n - successes + 1) if successes else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, successes + 1, n - successes) if successes < n else 1.0
    return float(lo), float(hi)


@dataclass
class EvalRun:
    """A pass rate you can actually defend in a review."""
    successes: int
    n: int

    def rate(self) -> float:
        return self.successes / self.n

    def interval(self) -> tuple[float, float]:
        return clopper_pearson(self.successes, self.n)

    def __str__(self) -> str:
        lo, hi = self.interval()
        return f"{self.rate():.1%} (95% CI {lo:.1%}–{hi:.1%}, n={self.n})"


print(EvalRun(successes=87, n=100))    # 87.0% (95% CI 78.8%–92.9%, n=100)
print(EvalRun(successes=870, n=1000))  # 87.0% (95% CI 84.8%–89.0%, n=1000)
```

Look at the first line before shipping any regression gate. Its one-sample 95%
interval still includes roughly 79%, so the point estimate alone does not
establish an eight-point difference. A paired baseline/candidate comparison can
have different power because it uses per-case differences; calculate that
design rather than comparing two rounded pass rates. The interval and minimum
detectable effect tell you which movements the suite can support.

### How many cases do you actually need?

```python
def required_n(baseline: float, mde: float, power: float = 0.80, alpha: float = 0.05) -> int:
    """Cases needed to detect a `mde`-sized drop from `baseline` (two-proportion, per arm)."""
    from math import sqrt
    from scipy.stats import norm
    p1, p2 = baseline, baseline - mde
    pbar = (p1 + p2) / 2
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    num = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(-(-num / (mde ** 2) // 1))


for mde in (0.10, 0.05, 0.02):
    print(f"detect a {mde:.0%} drop from 85%: n={required_n(0.85, mde):,} per arm")
```

```
detect a 10% drop from 85%: n=250 per arm
detect a  5% drop from 85%: n=906 per arm
detect a  2% drop from 85%: n=5,274 per arm
```

That table is the honest answer to "how big should our eval set be", and it explains why serious CI gates are set at loose thresholds on small suites and tight thresholds only on large ones. **Publish the MDE of your eval suite next to its pass rate** — the *minimum detectable effect* is the smallest change your suite can reliably tell apart from noise, and a gate set tighter than its own MDE is firing on randomness. A gate that fires on a 3-point drop over 150 cases is a random-number generator wearing a lab coat.

### Two variance-related consequences

1. **Reliability metrics matter more.** With sampling variance irreducible, `pass^k` (succeeds on *all* k trials) becomes the metric that reflects deployed reality, while `pass@k` (succeeds on *at least one*) flatters. Module 01 §1.3b has the arithmetic; the point here is that the API change removed the last excuse for reporting only pass@1.
2. **Cache your seeds elsewhere.** For genuinely deterministic regression tests — schema validity, refusal behavior on a fixed red-team set, tool-call shape — move the determinism into the *check*, not the sampler: assert invariants that hold across samples rather than asserting on an exact string.

---

## 15.3 Effort Is an Eval Axis, Not a Setting

The same model at `low` and at `max` effort is, for evaluation purposes, **two different systems** — different scores, different costs, different latencies, sometimes different failure modes. A benchmark number without an effort setting attached is not reproducible.

> **Rule: record `(model, effort, thinking mode, scaffold version)` with every score. All four. A "SOTA" claim missing any one of them is not checkable.**

The productive use of this is a deliberate sweep, producing a cost-quality frontier rather than a single number:

```python
# pip install anthropic
"""Effort sweep: the output is a frontier, not a score."""
import anthropic, json, time

client = anthropic.Anthropic()

def run_suite(cases, effort: str, model: str = "claude-opus-5") -> dict:
    passed = cost = 0.0
    measured = unmeasured = 0
    t0 = time.monotonic()
    for case in cases:
        r = client.messages.create(
            model=model, max_tokens=8000,
            output_config={"effort": effort},           # low | medium | high | xhigh | max
            messages=[{"role": "user", "content": case["prompt"]}],
        )
        if r.stop_reason == "refusal":                  # never silently score a refusal
            unmeasured += 1
            continue
        text = next((b.text for b in r.content if b.type == "text"), "")
        passed += case["check"](text)
        measured += 1
        cost += r.usage.input_tokens * 5e-6 + r.usage.output_tokens * 25e-6
    return {"effort": effort,
            "pass_rate": passed / measured if measured else None,
            "measured": measured, "unmeasured": unmeasured,
            "coverage": measured / len(cases) if cases else 0.0,
            "cost_usd": cost,
            "wall_clock_s": time.monotonic() - t0}

frontier = [run_suite(CASES, e) for e in ("low", "medium", "high", "xhigh", "max")]
print(json.dumps(frontier, indent=2))
```

An illustrative frontier — and the shape of it is the point:

| Effort | Pass rate | Cost | Latency (p50) | $ per additional point |
|---|---|---|---|---|
| `low` | 71.2% | $4.10 | 3.1 s | — |
| `medium` | 82.6% | $7.80 | 6.4 s | $0.32 |
| `high` | 87.1% | $14.20 | 12.9 s | $1.42 |
| `xhigh` | 89.0% | $24.60 | 24.1 s | $5.47 |
| `max` | 89.4% | $41.30 | 47.8 s | $41.75 |

Three readings, all of which people get wrong by default:

- **The last column is the decision.** Points 71→83 cost 32¢ each; the point from `xhigh` to `max` costs $41.75. Nothing about "we use max effort because quality matters" survives contact with that column.
- **Sweep down, not up.** On Opus 5, `low` and `medium` are unusually strong — often matching a previous generation's top settings. Prior-generation effort defaults rarely transfer; re-tune them rather than carrying them over.
- **Choose a starting point, then sweep.** `xhigh` is a defensible capability-first starting point for hard coding/agentic work and `high` for many other tasks, but neither is a universal optimum. Higher effort can reduce total turn count or merely add cost; measure end-to-end task cost, latency, coverage, and success.

### The thinking-disabled trap for eval harnesses

A common cost optimization is to disable thinking on the judge. On Opus 5 this has two documented failure modes that are uniquely destructive **in an eval context**:

| Failure | Why it is worse in an eval harness |
|---|---|
| Tool calls emitted as plain text instead of a `tool_use` block | The turn succeeds, the call never runs, no error is raised. In a **tool-use eval**, this scores as "the model failed the task" when what actually happened is that your config broke the tool channel. You will file a capability bug against a model that did nothing wrong. |
| `<thinking>` tags leaking into visible output | String-matching checks and structured parsers silently mis-score. |

Mitigation, in priority order: **turn thinking back on and lower `effort` instead** — that gets most of the cost saving without either failure mode. If a route must stay thinking-off, add *"You may say a brief sentence before using a tool"*, **delete** any "do not think / do not reason" instruction (it makes tag leakage worse, counterintuitively), and use a generic *"Do not include internal or system XML tags in your response"* rather than naming thinking tags.

---

## 15.4 Judges: Structured Outputs and Refusal-Aware Measurement

### Structured outputs replace prefill-and-regex parsing

Assistant prefill returns a 400 on Opus 5, Opus 4.8/4.7/4.6, Sonnet 5/4.6,
and Fable 5. A judge that depended on
`{"role": "assistant", "content": "{"}` must therefore migrate. Structured
outputs enforce the response shape, although they do not prove that the rubric
or verdict is correct:

```python
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL", "UNKNOWN"]},
        "evidence": {"type": "string", "description": "Quote from the response you relied on."},
    },
    "required": ["verdict", "evidence"],
    "additionalProperties": False,
}

resp = client.messages.create(
    model="claude-opus-5", max_tokens=2000,
    output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
                   "effort": "high"},
    messages=[{"role": "user", "content": judge_prompt}],
)
```

You can remove stop sequences guarding JSON, the regex extractor, and
"output ONLY valid JSON" prompting. Keep telemetry for refusals, truncation,
transport failures, `UNKNOWN` verdicts, and schema/API errors: valid JSON is
not the same as a measured judgment. Two API notes: the first request with a
new schema pays a one-time compilation cost (cached for 24 hours), and
structured outputs are **incompatible with citations** (400).

**Migration note for Fable 5.1 / Mythos 5.1.** If your schema-constrained judge
instead forces the call with `tool_choice: {"type": "tool", ...}` (or
`"any"`), that pattern now 400s on Fable 5.1 / Mythos 5.1 — see the
before/after snippet in §15.1. Structured outputs (above) or [strict tool
use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)
are the two supported replacements; there is no forced-tool-call path left on
these models.

### Refusals: classify them from the estimand

For a capability eval conditional on the target model actually answering, a
safeguard refusal is missing measurement. For an end-to-end product eval where
availability matters, it is a product outcome. For a refusal-policy eval, it
may be the correct answer. Decide this before running the suite and report both
the conditional score and refusal rate when they answer different questions.

This is the eval-integrity issue of the Opus 5 era, and it is easy to get silently wrong.

Safety classifiers can decline a request. The response is **HTTP 200** with `stop_reason: "refusal"` and empty or partial `content`. Code that does `content[0].text` throws; code that does `next((b.text for b in r.content), "")` returns an empty string — and an empty string fails almost every check you have written.

> **A refused judge call is a hole in your data. Scoring it as FAIL manufactures a result that no one measured.**

The consequences compound. Refusals are not uniformly distributed — they cluster in security, biology, and adjacent domains — so silently counting them as failures produces a suite that reports systematically depressed scores **on exactly the categories a safety-relevant eval exists to measure**. You will conclude the model is bad at the thing your harness merely declined to look at.

```python
from enum import Enum

class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNMEASURED = "unmeasured"      # ← the third state most harnesses lack

def judge(prompt: str) -> tuple[Outcome, dict]:
    r = client.beta.messages.create(
        model="claude-opus-5", max_tokens=2000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",       # policy declines get re-served by the routed fallback
        output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
                       "effort": "high"},
        messages=[{"role": "user", "content": prompt}],
    )
    if r.stop_reason == "refusal":          # check BEFORE touching r.content
        cat = r.stop_details.category if r.stop_details else None
        return Outcome.UNMEASURED, {"reason": "refusal", "category": cat}
    v = json.loads(next(b.text for b in r.content if b.type == "text"))
    return (Outcome.PASS if v["verdict"] == "PASS" else Outcome.FAIL), v


def report(outcomes: list[Outcome]) -> str:
    if not outcomes:
        return "UNMEASURED (0 cases, coverage 0.0%)"
    measured = [o for o in outcomes if o is not Outcome.UNMEASURED]
    unmeasured = len(outcomes) - len(measured)
    rate = (
        sum(o is Outcome.PASS for o in measured) / len(measured)
        if measured else None
    )
    coverage = len(measured) / len(outcomes)
    rate_text = f"{rate:.1%}" if rate is not None else "UNMEASURED"
    return (f"pass {rate_text} on {len(measured)} measured cases "
            f"({unmeasured} unmeasured, coverage {coverage:.1%})")
```

**Always report coverage and refusal rate next to a conditional score.**
"91.4% pass, coverage 99.8%" and "91.4% pass, coverage 71%" are different
claims. The second can still be evidence about attempted cases, but it is weak
evidence about end-to-end product behavior and invites a missingness audit.

`fallbacks: "default"` (beta header `server-side-fallback-2026-07-01`) can
route declined requests to a recommended fallback by refusal category. That
may recover product coverage, but it changes the system under test: record
`response.model` and report primary-model and routed-system results separately.
The option is Claude API only and rejected on the Batches API, so a batched eval
suite still needs an explicit refusal outcome.

### Panels: diversity beats redundancy

Three judges of the same model and prompt are one judge billed three times, exactly as with correlated gate layers (Module 14 §14.6). A useful panel varies at least one axis:

| Axis to vary | Cheap version | What it catches |
|---|---|---|
| **Model family** | Opus 5 + Sonnet 5 + a non-Anthropic model | Family-specific blind spots and self-enhancement bias |
| **Lens** | correctness / safety / does-it-reproduce | Failure modes a single framing never asks about |
| **Position** | swap A/B order and re-run | Position bias in pairwise comparisons |
| **Tier** | Haiku screen → Opus arbiter | Cost, at equal quality on the easy majority |

---

## 15.5 Running a Large Eval Suite for a Tenth of the Cost

Four independent multipliers, all composable. For 10,000 judge calls at 2,000 input / 300 output tokens each, with 1,500 of the input tokens being a stable rubric prefix:

| Strategy | Cost | vs. naive |
|---|---|---|
| Naive Opus 5 | $175.00 | 100% |
| + prompt caching | $107.51 | 61% |
| + Batch API (50% off) | $87.50 | 50% |
| Caching + batch | $53.75 | 31% |
| Haiku 4.5 screen only | $35.00 | 20% |
| Cascade (Haiku → Opus 5 on 20%) | $70.00 | 40% |
| **Cascade + caching + batch** | **$21.50** | **12%** |

**1. Prompt caching, with the prefix designed for it.** Caching is a *prefix match* — one changed byte invalidates everything after it. For a judge, that means: rubric and instructions first (stable, cached), the case under evaluation last (varies every call).

```python
resp = client.messages.create(
    model="claude-opus-5", max_tokens=2000,
    system=[{"type": "text", "text": RUBRIC_AND_INSTRUCTIONS,       # stable → cached
             "cache_control": {"type": "ephemeral"}}],
    output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    messages=[{"role": "user", "content": case_text}],               # varies → after the breakpoint
)
assert resp.usage.cache_read_input_tokens > 0, "cache miss — hunt the invalidator"
```

That assertion belongs in your harness permanently. The classic invalidators are a timestamp or run-ID interpolated into the system prompt, `json.dumps()` without `sort_keys=True`, and a tool list whose order varies between runs. Each one silently costs you the entire discount while every other metric looks normal. The 512-token minimum on Opus 5 also means short judge prompts that never cached on older models now can — re-check anything you previously wrote off.

**2. Batch API.** Offline eval suites are the ideal batch workload: latency-insensitive, embarrassingly parallel, 50% off. Results arrive in **any order** — key by `custom_id`, never by position.

**3. Cascade.** Screen with Haiku 4.5, then escalate the unresolved region to
Opus 5. Learn the acceptance and escalation boundaries on held-out human
labels; disagreement with the stronger judge is a diagnostic, not ground
truth. Report each tier's error and coverage by slice.

**4. Effort.** Lower effort can materially reduce cost on straightforward
criteria, but the saving and validity change depend on output length and task.
Measure cost, coverage, and agreement with humans per criterion rather than
assuming a fixed percentage reduction.

> **Count Claude tokens with `client.messages.count_tokens`, not a tokenizer
> built for another model family.** Token boundaries differ, so a foreign
> tokenizer can bias cost and truncation estimates. Validate estimates against
> the usage fields returned by the API.

---

## 15.6 Long-Horizon Eval Runs: Context Management Changes What You're Measuring

Agent evals now routinely exceed any context window, and there are two distinct mechanisms for coping. **They are not interchangeable, and both change the system under test.**

| Mechanism | What it does | Beta | Effect on the agent |
|---|---|---|---|
| **Compaction** | Summarizes earlier context server-side into a `compaction` block | `compact-2026-01-12` | Agent retains a *summary* of its history — lossy, and the loss is model-chosen |
| **Context editing** | *Clears* old tool results (`clear_tool_uses_20250919`) or thinking blocks (`clear_thinking_20251015`) | `context-management-2025-06-27` | Agent loses that content entirely |

```python
resp = client.beta.messages.create(
    model="claude-opus-5", max_tokens=16000,
    betas=["compact-2026-01-12"],
    context_management={"edits": [{"type": "compact_20260112"}]},
    messages=messages,
)
messages.append({"role": "assistant", "content": resp.content})   # full content, not just text
```

That last line is the one people get wrong: **append `resp.content`, not the extracted text.** The compaction block must be preserved — the API uses it to replace the compacted history on the next request. Extracting only the text silently loses the compaction state, and the failure looks like an agent that mysteriously forgets things mid-run.

**The eval-methodology point.** Long-horizon agent failures documented in the literature — context degradation, forgetting earlier commitments, spiraling "meltdowns" (Module 08 Case Study 9) — are *precisely* the failures that context management interacts with. An agent evaluated with compaction enabled and an agent evaluated without it are different systems, and the difference shows up exactly in the metric you care about.

> **Record the context-management configuration alongside every long-horizon result, and treat a change to it as a change to the system under test — requiring a re-run of the baseline, not just the candidate.**

---

## 15.7 Memory Is a Contamination Vector

Agents can now persist knowledge across sessions — the memory tool writing to a `/memories` directory, or a workspace-scoped **memory store** mounted into a session's filesystem. This is a genuine capability improvement and a **direct threat to eval validity** that most harnesses do not defend against.

The mechanism is simple and, once you see it, obvious:

```
Trial 1: agent attempts benchmark task X, fails, writes to memory:
         "task X: the answer is 47; the trick is to check the units first"
Trial 2: agent reads memory, answers instantly.

Reported pass^5: 80%.   Actual capability: it solved it once and took notes.
```

This is benchmark contamination (Module 12) arriving through a new door — not through the training set, but through the **runtime state**. It inflates exactly the reliability metrics that are supposed to be contamination-resistant, because `pass^k` assumes k *independent* trials and memory makes them dependent by design.

**Mitigations, in order of rigor:**

| Approach | Rigor | Cost |
|---|---|---|
| Fresh memory store per trial (or no memory resource at all) | Highest — restores independence | Provisioning per trial |
| Same store, but audit memory contents between trials for task-specific answers | Medium — catches the blatant cases | Human or LLM review |
| Report **both** configurations: cold-memory and warm-memory pass rates | Best for decisions — the delta *is* a capability measurement | 2× runs |

The third option is the one worth internalizing. "Cold pass@1 = 34%, warm pass@1 = 71%" is not a contaminated number — it is a *measurement of how well the agent learns from its own experience*, which is a real and increasingly important capability. It becomes contamination only when you report the warm number and call it the cold one.

Two adjacent hygiene rules, both of which have bitten teams:

- **Never store credentials in memory**; memories are replayed verbatim into every later session that mounts the store.
- **Memory versions are an audit surface.** Every mutation produces an immutable version record, which means "did this agent write the answer to itself?" is an answerable question — use it in your contamination audits rather than guessing.

---

## 15.8 Harness-Building Techniques Worth Knowing

**Programmatic tool calling** — let the model compose tool calls into a script that runs in the code-execution container, with intermediate results staying *in the script* rather than entering context. For an eval harness that fans out over hundreds of retrievals or checks, this collapses many round trips into one and keeps the context window free for the judgment itself.

**Tool search** (`tool_search_tool_regex_20251119` / `_bm25_`, with other tools marked `defer_loading: true`) — for evals over large tool libraries, load only relevant schemas. Critically for cost: discovered schemas are *appended*, not swapped, so **the prompt cache survives** — unlike editing the `tools` array, which invalidates everything.

**Mid-conversation tool changes** (beta `mid-conversation-tool-changes-2026-07-01`, Opus 5+) — add or remove tools between turns via `tool_addition` / `tool_removal` blocks on a `role: "system"` message, without invalidating the cached prefix. This makes **tool-ablation evals** cheap: previously, measuring "how much worse is the agent without the search tool?" meant a separate cold-cache run per configuration.

**The advisor tool** — pair a cheaper executor model with a stronger advisor consulted mid-generation. A natural fit for cascaded judging where you want Haiku-tier throughput with Opus-tier judgment on the hard calls. The advisor model must be at least as capable as the executor, or the request 400s. Note the payload shape differs by advisor: on Opus 5 / Fable 5 the result content is `advisor_redacted_result` carrying `encrypted_content`, not readable `text` — code that reads `.text` unconditionally gets nothing.

**Instrumentation worth capturing on every eval call:**

| Field | Why |
|---|---|
| `usage.cache_read_input_tokens` | Cache health; zero across repeated runs = a silent invalidator |
| `usage.input_tokens` | The **uncached remainder only** — total prompt = input + cache_creation + cache_read |
| `stop_reason` / `stop_details.category` | Coverage accounting (§15.4) |
| `usage.iterations` | Per-attempt billing when fallbacks fire |
| `response.model` | Which model actually served it — not necessarily the one you asked for, under fallbacks |

That last row matters more than it looks. Under `fallbacks`, a response can be served by a different model than the one in your request, and sticky routing means later turns in the same conversation may keep going there. **An eval that attributes a fallback-served result to the requested model is reporting the wrong model's score.**

---

## 15.9 Migrating an Eval Harness to Opus 5

| Change | Severity | Action |
|---|---|---|
| `temperature` / `top_p` / `top_k` | **400** | Delete. Move to n-run estimates with intervals (§15.2) |
| `thinking: {"type": "enabled", "budget_tokens": N}` | **400** | Replace with `output_config.effort` |
| Assistant-turn prefill | **400** | Replace with `output_config.format` (structured outputs) |
| `thinking: disabled` + `effort: xhigh`/`max` | **400** | Lower effort to `high`, or enable thinking |
| Harness never set `thinking` | **Silent** | Thinking is now on by default; `max_tokens` caps thinking + text together — raise it or expect truncation |
| `content[0].text` without a `stop_reason` check | **Silent** | Refusals become fake failures (§15.4) |
| `tiktoken`-based cost model | **Silent** | Switch to `messages.count_tokens` |
| Effort defaults carried from a prior model | **Silent** | Re-sweep; `low`/`medium` are unusually strong on Opus 5 |
| Fixed `max_tokens` on judges | **Silent** | Thinking shares the budget now |
| Rate-limit assumptions | **Silent** | Opus 5 is a separate bucket from Opus 4.x |
| `tool_choice: "any"` / `"tool"` (Fable 5.1 / Mythos 5.1 only) | **400** | Migrate the forced grader/JSON tool call to structured outputs or strict tool use (§15.1, §15.4) |
| Replaying a cached thinking block after mutating the system prompt/tools/earlier messages (Fable 5.1 / Mythos 5.1, accounts created after Aug 31, 2026) | **400** | Don't reuse a cached thinking trace across a changed prompt — re-run rather than replay |
| Cache-read cost model assumes 0.1× on Fable 5.1 / Mythos 5.1 | **Silent** | Actual rate is 0.025× ($0.25/MTok flat) — re-check any harness cost projection built before Sept 1, 2026 |
| Harness has no way to cheapen exploration turns without a fresh call | **New capability, not a break** | Per-message `effort` (beta) lets a single cached session mix cheap-effort turns with a high-effort judge turn |

**Verification after migration** — one call, three assertions:

```python
r = client.messages.create(model="claude-opus-5", max_tokens=64,
                           messages=[{"role": "user", "content": "Reply with OK."}])
assert r.model.startswith("claude-opus-5"), r.model      # not silently a fallback
assert r.stop_reason != "refusal"
assert r.usage.output_tokens > 0
```

---

## 15.10 Exercises

### Exercise 1: Put an interval on your headline number
Take your current eval suite's pass rate. Compute its 95% interval and its MDE at 80% power. Then look at your CI gate threshold. If the threshold is smaller than the MDE, your gate has been firing on noise — write up how many cases you would need for the threshold you actually want.

### Exercise 2: Build the frontier
Run your suite at all five effort levels. Produce the cost-per-additional-point column. Identify the level where marginal cost per point exceeds what a point is worth to your product, and change your default to it.

### Exercise 3: Find your unmeasured cases
Add the `UNMEASURED` outcome to your harness and re-run. Report coverage next
to every pass rate. Compare it with a predeclared minimum based on risk and
failure cost—99% may be appropriate for some high-stakes suites but is not a
universal threshold. Regardless of the aggregate, inspect which categories the
holes occupy; that distribution is often more informative than the score.

### Exercise 4: Hunt the cache invalidator
Assert `cache_read_input_tokens > 0` in your harness and run it. If it fails, bisect the prompt-building path for the invalidator. Compute the annualized cost of the bug you find.

### Exercise 5: Measure your memory contamination
Run an agent eval twice: once with a fresh memory store per trial, once with a
shared store across trials. Report both pass rates and inspect the stored
content. The delta is an observation consistent with several explanations:
answer leakage, intended cross-session learning, order effects, or a state-reset
bug. Design the next probe to distinguish them.

---

## Next Steps

You have now covered the loop (Module 14) and the platform (Module 15). Both feed back into the earlier modules:

- **Module 02** — every judge in this module is still subject to position bias, self-enhancement bias, and κ calibration. New API features do not fix old judge pathologies.
- **Module 12** — memory stores and agent state are the newest contamination vectors; the decontamination discipline there applies to runtime state, not just training data.
- **Module 08 Case Study 10** — the Uber Eats image agent is every technique in Modules 14 and 15 assembled into one production system.

The API surface will keep moving. The measurement discipline is what carries forward.
