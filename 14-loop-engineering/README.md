# Module 14: Loop Engineering

> **For systems that critique, retry, or learn from use, evaluate the loop—not
> only the final call.**
>
> A simple system can still take a prompt and return an answer. A looped system
> *attempts* something, *checks its work*, *repairs*, and *retries*—and a slower
> outer loop may use production evidence to revise its configuration. A
> single-response metric cannot tell you which attempt added value, whether the
> checker rejected good work, or what the retries cost.
>
> **Prerequisites:** Module 01 (pass@k vs pass^k), Module 02 (rubrics, judges, judge calibration), Module 06 (feedback loops), Module 08 Case Study 10 (the Uber Eats image agent, which is a worked example of everything in this module).

---

## In Plain English (start here if you don't write the code)

A **loop** is a system that checks its own work and tries again when it is not good enough. It writes a draft, something inspects the draft, and if the draft fails inspection it writes another one — up to some limit. This pattern can improve an agent without changing the underlying model, but only when the checker is trustworthy and another attempt is worth its cost.

This module is about the question a final pass rate cannot answer: **is the checking actually helping, and what is it costing?**

Four ideas carry the whole module, and none of them require reading a line of code:

1. **Checker quality limits loop quality.** If the inspector is unreliable, retrying can make things *worse* — it throws away good work, lets bad work through, and adds cost. A loop cannot repair errors its checker cannot recognize.
2. **"It passed eventually" hides everything.** A product that succeeds 91% of the time might get 88% of that on the first try, with the retries adding almost nothing while adding most of the cost. Same headline number, completely different business decision.
3. **Trying again can make things worse.** Sometimes the third draft is worse than the first — the system "improves" itself into a worse answer. This is measurable; the chapter calls it **regression rate**.
4. **Log the reason, not just the failure.** "This failed" is useless. "This failed because the lighting check rejected it" is a work item with an owner. Everything else in this module depends on that one habit.

The rest of the module makes each of these measurable. One distinction matters from the start:

- **Attempt 1, 2, 3 inside one run** are dependent revisions. Measure them with **accepted by attempt K**, the **accepted-at distribution**, and marginal yield.
- **Trial 1, 2, 3 started independently from a clean state** measure stochastic reliability. Use standard **pass@k** (at least one trial succeeds) and **pass^k** (every trial succeeds).

Calling both of these "pass@k" hides whether retries repaired one run or whether repeated fresh runs were merely lucky. Other terms you will meet: *gate* or *verifier* — the automated checker; *escalation* — handing an unresolved run to a human or safe fallback.

---

## 14.0 Why This Module Exists

### A documented failure: the agent succeeded, but the eval said it failed

Anthropic reports that Claude Opus 4.5 found a policy loophole while solving a flight-booking task in **τ2-bench**. The written eval marked the trial as a failure, even though the agent found a better solution for the user. A final score alone would have blamed the model. Reading the transcript showed that the **grader was too rigid**. ([Anthropic, “Demystifying evals for AI agents,” 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents))

| Eval | What it covered | What it caught | What should improve |
|---|---|---|---|
| Outcome grader | Whether the requested state was achieved | The benchmark rejected an unexpectedly valid solution | Grade the real environment outcome and repair the task specification |
| Transcript review | Why the agent took each action | The model had not simply ignored the task; it found a path the grader did not anticipate | Keep trajectory constraints only for genuinely forbidden actions, not one preferred route |
| Grader calibration | Agreement between automated verdicts and expert review | A false rejection by the checker | Add this case to the grader's regression set and re-measure false rejects |

This is the verifier asymmetry in concrete form: a checker can force a capable generator to redo correct work. To evaluate the *loop* rather than only its final answer, add the measurements in the right-hand column:

| A final-only dashboard shows | A loop dashboard must also show |
|---|---|
| Final acceptance rate | Acceptance **by attempt index** — where does the yield actually come from? |
| Mean cost per call | Cost per **accepted output**, including the attempts that failed |
| Gate pass rate | Gate **precision and recall against human labels** — is the gate right? |
| — | **Regression rate**: fraction of cases where attempt 3 scored *worse* than attempt 1 |
| — | **Oscillation rate**: loops that alternate between two failure modes until the budget runs out |
| — | **Escape rate**: defects that passed every gate |

This module is about that right-hand column.

---

## 14.1 The Three Loops

Looped AI systems can contain three nested loops running at different
timescales. They fail differently and need different instrumentation; conflating
them can hide which layer produced a misleading aggregate result.

```
┌──────────────────────────────────────────────────────────────────────┐
│ OUTER LOOP — days to weeks — "is the system still right?"            │
│   production traces → sampling → golden benchmark → diagnosis →      │
│   prompt/config change → offline gate → canary → promote             │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ TASK LOOP — seconds to minutes — "is this output good enough?" │  │
│  │   attempt → QA gate → critique → re-attempt  (accept by K)     │  │
│  │                                                                 │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │ TURN LOOP — one request — "do I have what I need?"       │  │  │
│  │  │   think → call tool → read result → think → answer       │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

| Loop | Controlled by | Iterations | Primary metric | Primary failure |
|---|---|---|---|---|
| **Turn** | The model (thinking + tool use inside one request) | 1–50 tool calls | Task success, tool-call efficiency | Thrash: repeated identical tool calls |
| **Task** | Your orchestration code | 1–5 dependent attempts | Accepted by K, marginal yield, cost per accepted output | Oscillation, over-editing regression |
| **Outer** | Your eval + deployment pipeline | Weekly-ish | Drift detection latency, promotion precision | Goodharting the benchmark |

**The rule that saves the most debugging time:** *every loop needs its own stop condition, its own budget, and its own metric.* When a request is slow and expensive, the first question is *which loop* ran away. If your logging cannot answer that in one query, fix the logging before you tune anything.

---

## 14.2 Anatomy of a Task Loop

Every well-built task loop has six components. Missing any one of them produces a characteristic pathology.

| Component | Job | If missing… |
|---|---|---|
| **Generator** | Produce a candidate | — |
| **Verifier (gate)** | Decide accept/reject, with a reason | Loop is a retry-on-exception, not self-correction |
| **Critique channel** | Turn rejection into an actionable directive for the next attempt | Blind retries: attempt N+1 makes the same mistake |
| **Budget** | Cap total spend (attempts, tokens, wall-clock) | Cost blowup; runaway loops in the tail |
| **Stop conditions** | Terminate on success, exhaustion, *and* no-progress | Oscillation until budget death |
| **Escalation path** | Where a loop that fails goes (human, stronger model, safe default) | Silent failures published as successes |

Here is a reference implementation. It is short on purpose — the value is in what it *records*, not what it does.

```python
# pip install anthropic
"""Reference task loop: generate → verify → critique → retry, fully instrumented."""
import time, json, hashlib
from dataclasses import dataclass, field
from typing import Callable, Any

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-opus-5"


@dataclass
class Attempt:
    index: int
    output: Any
    passed: bool
    blocked_by: str | None       # WHICH criterion failed — never just False
    critique: str
    score: float | None          # graded quality, independent of pass/fail
    input_tokens: int
    output_tokens: int
    latency_ms: int
    output_hash: str             # for oscillation detection
    cache_read_tokens: int = 0   # repeated-prefix cache use on this attempt
    cache_creation_5m_tokens: int = 0
    cache_creation_1h_tokens: int = 0


@dataclass
class LoopResult:
    run_id: str
    attempts: list[Attempt] = field(default_factory=list)
    outcome: str = "pending"     # accepted | exhausted | oscillated | no_progress
    accepted_at: int | None = None
    escalated: bool = False      # orthogonal to outcome — never overwrite the reason

    @property
    def cost_usd(self) -> float:
        # claude-opus-5 standard global rates per MTok: $5 uncached input,
        # $6.25 for 5-minute cache writes, $10 for 1-hour cache writes,
        # $0.50 for cache reads, and $25 output.
        return sum(a.input_tokens * 5e-6
                   + a.cache_creation_5m_tokens * 6.25e-6
                   + a.cache_creation_1h_tokens * 10e-6
                   + a.cache_read_tokens * 0.5e-6
                   + a.output_tokens * 25e-6
                   for a in self.attempts)

    def to_rows(self) -> tuple[dict, list[dict]]:
        """Return one run row plus one row for every attempt (see §14.7)."""
        run_row = {
            "run_id": self.run_id,
            "outcome": self.outcome,
            "escalated": self.escalated,
            "attempts": len(self.attempts),
            "accepted_at": self.accepted_at,
            "cost_usd": round(self.cost_usd, 5),
            "total_latency_ms": sum(a.latency_ms for a in self.attempts),
            "first_score": self.attempts[0].score if self.attempts else None,
            "final_score": self.attempts[-1].score if self.attempts else None,
        }
        attempt_rows = [{
            "run_id": self.run_id,
            "attempt_index": a.index,
            "passed": a.passed,
            "blocked_by": a.blocked_by,
            "critique": a.critique,
            "score": a.score,
            "input_tokens": a.input_tokens,
            "output_tokens": a.output_tokens,
            "cache_read_tokens": a.cache_read_tokens,
            "cache_creation_5m_tokens": a.cache_creation_5m_tokens,
            "cache_creation_1h_tokens": a.cache_creation_1h_tokens,
            "latency_ms": a.latency_ms,
            "output_hash": a.output_hash,
        } for a in self.attempts]
        return run_row, attempt_rows


def run_loop(
    run_id: str,
    generate: Callable[[str | None], tuple[Any, int, int, int, int, int]],
    # critique -> (output, uncached_input_tokens, cache_read_tokens,
    #              cache_write_5m_tokens, cache_write_1h_tokens, output_tokens)
    verify: Callable[[Any], dict],                            # output -> {passed, blocked_by, critique, score}
    max_attempts: int = 3,
    cost_ceiling_usd: float = 0.50,
    attempt_cost_reserve_usd: float = 0.10,
    escalate: Callable[[LoopResult], Any] | None = None,
) -> LoopResult:
    result = LoopResult(run_id=run_id)
    critique: str | None = None
    seen_hashes: set[str] = set()

    for i in range(1, max_attempts + 1):
        # Reserve before spending. Set this to a conservative upper bound for
        # one attempt (from token caps and current model/cache prices).
        if result.cost_usd + attempt_cost_reserve_usd > cost_ceiling_usd:
            result.outcome = "exhausted"
            break

        t0 = time.monotonic()
        output, in_tok, cache_read_tok, cache_5m_tok, cache_1h_tok, out_tok = generate(critique)
        v = verify(output)
        if not v["passed"] and not v.get("blocked_by"):
            raise ValueError("Rejected verifier result requires a non-empty 'blocked_by' reason")
        h = hashlib.sha256(str(output).encode()).hexdigest()[:16]

        result.attempts.append(Attempt(
            index=i, output=output, passed=v["passed"], blocked_by=v.get("blocked_by"),
            critique=v.get("critique", ""), score=v.get("score"),
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=int((time.monotonic() - t0) * 1000), output_hash=h,
            cache_read_tokens=cache_read_tok,
            cache_creation_5m_tokens=cache_5m_tok,
            cache_creation_1h_tokens=cache_1h_tok,
        ))

        # --- Stop condition 1: success ---
        if v["passed"]:
            result.outcome, result.accepted_at = "accepted", i
            return result

        # --- Stop condition 2: oscillation (this output was already produced & rejected) ---
        if h in seen_hashes:
            result.outcome = "oscillated"
            break
        seen_hashes.add(h)

        # --- Stop condition 3: no progress (same blocker twice in a row) ---
        if i >= 2 and result.attempts[-1].blocked_by == result.attempts[-2].blocked_by:
            result.outcome = "no_progress"
            break

        # --- Stop condition 4: budget ---
        if result.cost_usd >= cost_ceiling_usd:
            result.outcome = "exhausted"
            break

        critique = v.get("critique")

    if result.outcome == "pending":
        result.outcome = "exhausted"
    if escalate:
        escalate(result)
        result.escalated = True      # keep WHY it failed; escalation is a separate fact
    return result
```

Several details in this small example carry most of its safety and diagnostic value:

The cost property uses Claude Opus 5's August 2026 standard global rates: $5/MTok uncached input, $6.25/MTok for 5-minute cache writes, $10/MTok for 1-hour cache writes, $0.50/MTok cache reads, and $25/MTok output. The API reports the two write TTLs separately under `usage.cache_creation`; keep both because one aggregate write count cannot be priced correctly when a request mixes TTLs. ([Claude prompt-caching pricing and usage fields](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)) In production, load rates from versioned configuration so a pricing change does not silently corrupt historical comparisons.

`attempt_cost_reserve_usd` makes the ceiling a preflight decision rather than an after-the-fact alert. Derive the reserve from the maximum tokens and tool spend one attempt is allowed to consume. If actual attempts can exceed the reservation, the ceiling is not hard; tighten the underlying token/tool limits or call it a target.

**Stop condition 3 (no progress) is not the same as the budget.** If a loop fails `lighting` twice with effectively unchanged evidence, another blind attempt has little justification. The cause might be a critique that is not actionable, an ambiguous verifier, or a generator that cannot satisfy the request. Stopping with the label `no_progress` preserves the remaining budget and gives the team a category it can inspect instead of guessing from `failed`.

**`blocked_by` is a string, never a boolean.** Refuse to log `passed: false` without a reason next to it. The blocker and recovery analyses in §14.7 depend on that reason existing.

---

## 14.3 Verifier Asymmetry: Separate Quality from Cost

> **Under the independent-attempt model below, the verifier improves the
> precision of accepted outputs when it rejects bad outputs more often than
> good ones (`TPR > FPR`). Whether that improvement is worth buying is a
> separate cost-and-latency decision.**

This distinction explains why some loops improve accepted quality but still
lose on economics, while an inverted verifier can make both quality and cost
worse. The worked arithmetic assumes independent, identically distributed
attempts; correlated revisions need the empirical within-run metrics in §14.4.

Let generation succeed with probability *p*. The verifier has two error rates, and the whole argument turns on keeping them separate:

- **TPR** (true-positive rate) — how often it **catches a genuinely bad output**. Low TPR = defects slip through.
- **FPR** (false-positive rate) — how often it **rejects a perfectly good output**. High FPR = good work thrown away and re-done, which is where loop cost comes from.

After a rejection you retry. What does the loop actually deliver?

```python
def loop_quality(p_generate: float, tpr: float, fpr: float, k: int) -> dict:
    """Outcome distribution after up to k attempts, assuming attempts are iid.

    tpr = P(verifier rejects | output is bad)    — catches real defects
    fpr = P(verifier rejects | output is good)   — throws away good work
    """
    accept_good = p_generate * (1 - fpr)          # good output the verifier let through
    accept_bad = (1 - p_generate) * (1 - tpr)     # bad output the verifier missed
    retry = 1 - accept_good - accept_bad          # anything rejected -> another attempt

    good = bad = 0.0
    p_reach = 1.0                                 # P(we are still looping at attempt i)
    for _ in range(k):
        good += p_reach * accept_good
        bad += p_reach * accept_bad
        p_reach *= retry
    return {"accepted_good": good, "accepted_bad": bad, "unresolved": 1 - good - bad,
            "precision_of_accepted": good / (good + bad) if good + bad else 0.0,
            "expected_attempts": (1 - retry ** k) / (1 - retry) if retry < 1 else k}


P = 0.70
print(f"{'no verifier (ship attempt 1)':28} good {P:.1%}  bad {1-P:.1%}  "
      f"precision {P:.1%}  attempts 1.00")
for tpr, fpr, label in [(0.95, 0.05, "strong verifier"),
                        (0.70, 0.20, "mediocre verifier"),
                        (0.55, 0.35, "weak verifier"),
                        (0.25, 0.40, "inverted verifier")]:
    r = loop_quality(p_generate=P, tpr=tpr, fpr=fpr, k=3)
    print(f"{label:28} good {r['accepted_good']:.1%}  bad {r['accepted_bad']:.1%}  "
          f"precision {r['precision_of_accepted']:.1%}  "
          f"attempts {r['expected_attempts']:.2f}")
```

```
no verifier (ship attempt 1) good 70.0%  bad 30.0%  precision 70.0%  attempts 1.00
strong verifier              good 94.6%  bad 2.1%  precision 97.8%  attempts 1.42
mediocre verifier            good 82.5%  bad 13.3%  precision 86.2%  attempts 1.47
weak verifier                good 71.8%  bad 21.3%  precision 77.1%  attempts 1.58
inverted verifier            good 62.2%  bad 33.3%  precision 65.1%  attempts 1.48
```

Read the precision column against the no-verifier baseline. The weak verifier
still buys about seven points of accepted-output precision, but at a 58%
increase in expected attempts and with 21.3% of all requests accepting a bad
output. The inverted verifier has `TPR < FPR`; it pushes precision below the
70% no-verifier baseline while also adding work. Retry count changes eventual
acceptance and cost, but under these iid assumptions it does not repair a
verifier whose acceptance signal points in the wrong direction.

Worse, the mechanism of the collapse is invisible in aggregate metrics. On a *single* attempt, the weak verifier rejects 35% of good outputs — 24.5 percentage points of all attempts (`0.70 × 0.35`). Retries recover some discarded work, so the final acceptance number can look respectable while the system spends 1.58 attempts to end up only modestly ahead of shipping attempt 1 unchecked. **A high false-positive rate converts directly into loop cost.**

Three practical corollaries:

1. **Evaluate the verifier before you deploy the loop.** Gate precision and recall against expert labels (Module 02 §2.3.6) is a prerequisite. Report uncertainty and agreement per criterion. Cohen's κ is useful for chance-adjusted agreement, but there is no universal κ cutoff: the required precision depends on the cost of a false accept or false reject. A safety veto and a style suggestion should not share one threshold.
2. **Prefer verification tasks that are easier and more objective than generation.** Running a well-designed test suite is usually easier to grade than writing the code; checking whether an edited image added a garnish can be narrower than producing the edit. When you cannot find that asymmetry, you may have a resampler rather than a self-correcting loop.
3. **A resampler is a legitimate design, but call it one.** If your "verifier" is just the generator scoring itself, you are drawing best-of-K samples. That is fine and often effective — but its metric is best-of-K quality, not self-correction, and it does not justify a critique channel.

> **Diagnostic:** compare score changes *within the same run* (`score_i - score_1`) for runs that reached attempt `i`. Do not simply compare the mean of all attempt-1 outputs with all attempt-3 outputs: only the harder cases reach attempt 3, so that survivor population can make a useful loop look worse. A rising within-run delta supports self-correction; a flat distribution suggests resampling; a negative tail is over-editing (§14.8 failure mode 3).

---

## 14.4 Loop Metrics: What to Actually Put on the Dashboard

A loop needs metrics at two levels. **Within one run**, attempts are revisions that share history. **Across clean reruns**, trials measure stochastic reliability. Do not mix the two denominators.

| Metric / eval | What it covers | Issue it catches | Decision it enables |
|---|---|---|---|
| **Accepted by attempt K** | One task-loop run, including its dependent revisions | A good final number that required too many retries | Keep, shorten, or remove the retry loop |
| **Accepted-at distribution** | Which attempt first passed | Most value arriving on attempt 1 while later attempts add cost | Put effort into the first prompt or retain later attempts |
| **Marginal yield** `P(accepted at i \| reached i)` | Conversion among runs that actually reached attempt `i` | A final attempt whose extra successes no longer pay for it | Choose K from value versus marginal cost |
| **Within-run regression rate** | `final_score < first_score` on the same run | The checker editing good work into worse work | Keep the best attempt and recalibrate false rejects |
| **Oscillation / no-progress rate** | Repeated outputs or blockers | Two criteria fighting, or critique that changes nothing | Stop early and repair the rubric or feedback channel |
| **Cost per accepted output** | All successful and failed attempt spend | Cheap calls forming an expensive product result | Compare the loop with a stronger first attempt or human review |
| **Gate escape rate** | Human-confirmed defects among accepted outputs | Bad work that every automated layer missed | Add a different gate or change the existing criterion |
| **Gate false-reject rate** | Human-confirmed good outputs that the gate rejected | Correct work repeatedly discarded | Relax or rewrite the gate before increasing K |
| **Loop tax** | Cost and latency versus one attempt | Tail retries breaking capacity assumptions | Set per-run limits and forecast real serving capacity |
| **pass@k / pass^k across independent trials** | Fresh reruns from clean state | A system that can succeed once but is not dependable | Decide whether reliability is sufficient for unattended use |

```python
"""Loop metrics from a list of LoopResult rows. Drop-in for a nightly job."""
from collections import Counter
from statistics import mean


def loop_metrics(results: list[LoopResult], k: int) -> dict:
    n = len(results)
    accepted = [r for r in results if r.outcome == "accepted"]

    # Where the yield comes from
    k_hist = Counter(r.accepted_at for r in accepted)
    marginal = {}
    reached_hist = {}
    for i in range(1, k + 1):
        won = k_hist.get(i, 0)
        reached = sum(len(r.attempts) >= i for r in results)
        reached_hist[i] = reached / n
        marginal[i] = won / reached if reached else 0.0

    # Did iteration ever make things worse? (needs a score on every attempt)
    scored = [r for r in results if len(r.attempts) > 1
              and r.attempts[0].score is not None and r.attempts[-1].score is not None]
    regressed = [r for r in scored if r.attempts[-1].score < r.attempts[0].score]

    first_attempts = [r.attempts[0] for r in results if r.attempts]
    single_attempt_cost = (mean(a.input_tokens * 5e-6
                                + a.cache_creation_5m_tokens * 6.25e-6
                                + a.cache_creation_1h_tokens * 10e-6
                                + a.cache_read_tokens * 0.5e-6
                                + a.output_tokens * 25e-6
                                for a in first_attempts)
                           if first_attempts else None)

    return {
        "n": n,
        "accepted_by_k": sum(
            r.outcome == "accepted"
            and r.accepted_at is not None
            and r.accepted_at <= k
            for r in results
        ) / n,
        "accepted_at": {i: k_hist.get(i, 0) / n for i in range(1, k + 1)},
        "reached_attempt": reached_hist,
        "marginal_yield": marginal,
        "regression_rate": len(regressed) / len(scored) if scored else None,
        "oscillation_rate": sum(r.outcome == "oscillated" for r in results) / n,
        "no_progress_rate": sum(r.outcome == "no_progress" for r in results) / n,
        "cost_per_accepted": (sum(r.cost_usd for r in results) / len(accepted)
                              if accepted else float("inf")),
        "loop_tax": ((sum(r.cost_usd for r in results) / n) / single_attempt_cost
                     if single_attempt_cost else None),
        "mean_attempts": mean(len(r.attempts) for r in results),
    }
```

### Reading a worked dataset

The following numbers are **illustrative arithmetic**, not reported production results. Their purpose is to show how the metrics lead to decisions.

```
n                  4,812
accepted_by_k      0.913
accepted_at        {1: 0.804, 2: 0.081, 3: 0.028}
reached_attempt    {1: 1.000, 2: 0.196, 3: 0.115}
marginal_yield     {1: 0.804, 2: 0.413, 3: 0.243}
regression_rate    0.061
oscillation_rate   0.018
no_progress_rate   0.033
cost_per_accepted  $0.0374
loop_tax           1.31×
mean_attempts      1.29
```

Everything you need to make three decisions is in there:

- **Is attempt 3 worth keeping?** 11.5% of traffic reaches it and it converts 24.3% of those — about **135 extra successes on 4,812 requests**, bought by running a third generate-and-verify cycle on about 553 of them. Compare the value of those successes with their marginal cost, then inspect the separate 6.1% within-run regression signal before deciding.
- **Is the loop paying for itself?** Acceptance rises from 80.4% on attempt 1 to 91.3% by attempt 3. Whether eleven points for a 1.31× cost multiplier is worthwhile depends on the value of a success and the harm of an escaped defect. The metric makes that trade-off explicit; it does not decide it for you.
- **Is anything pathological?** In this scenario, 6.1% regression and 1.8% oscillation need investigation (§14.8). Both are invisible in the 91.3% final-acceptance number.

> **The one-number trap.** "Accepted by attempt 3 = 91.3%" hides all three findings above. Report it with the accepted-at distribution, marginal yield, regression rate, and cost per accepted output. Separately report pass@k or pass^k from clean independent trials when reliability matters.

---

## 14.5 Budgets and Stop Conditions

Loops need hard controls and soft guidance. Hard controls terminate work even if the model would continue. Soft guidance lets the model pace itself and finish gracefully, but it is not a guarantee.

Opus 5-era models expose this as a first-class parameter — a **task budget** — which is distinct from `max_tokens`:

| Control | Nature | Model aware? | Behavior at the limit |
|---|---|---|---|
| `max_tokens` | Hard API cap **per request** | No | Truncates with `stop_reason: "max_tokens"` |
| `output_config.task_budget` | **Advisory** beta signal across the agentic loop | **Yes** — server injects a countdown | Model usually prioritizes and wraps up, but may exceed the target |
| Loop `max_attempts` | Hard client-side cap | No | Orchestrator exits before another attempt |
| Loop cost ceiling | Client-side accounting rule | No | Orchestrator refuses another attempt when the reserved budget is insufficient |

```python
# pip install anthropic
# Task budget: tell the model how much room it has for a whole agentic turn.
# Beta `task-budgets-2026-03-13`; minimum total is 20,000 tokens.
# Stream, because a large max_tokens on a non-streaming call risks HTTP timeouts.
with client.beta.messages.stream(
    model="claude-opus-5",
    max_tokens=128000,                       # hard ceiling, invisible to the model
    output_config={
        "effort": "high",
        "task_budget": {"type": "tokens", "total": 64000},   # soft ceiling, visible
    },
    betas=["task-budgets-2026-03-13"],
    tools=TOOLS,
    messages=messages,
) as stream:
    response = stream.get_final_message()
```

The task budget counts thinking, tool calls, tool results, and text across the agentic loop — not the full history your client resends on each request. Leave `remaining` unset in a normal loop so the server tracks the countdown. Pass it explicitly only when carrying a budget across client-side compaction. Most importantly, Anthropic documents task budgets as **advisory, not enforced**; pair them with `max_tokens` and client-side attempt/cost controls. ([Claude Platform task-budget documentation](https://platform.claude.com/docs/en/build-with-claude/task-budgets))

**An eval-relevant warning about budget countdowns.** Surfacing a remaining-token count to the model can change how it prioritizes work. If you A/B a budget change, treat it as a **behavioral change, not just a cost control**, and re-run quality evals on the budgeted configuration. Report quality and cost together.

### The stop-condition ladder

Order matters — check cheap conditions first, and always leave a labeled outcome behind:

```
1. accepted        → verifier passed                     → publish
2. oscillated      → output hash repeated                → escalate (loop is stuck)
3. no_progress     → same blocker N times consecutively  → escalate (critique not landing)
4. exhausted       → attempts or cost ceiling reached    → escalate or safe default

   `escalated` is a FLAG, not a fifth outcome — a run that was handed to a human
   is still an `oscillated` run or an `exhausted` run, and you need both facts.
```

Never collapse 2–4 into a single `failed`. They point to different investigations: oscillation may expose conflicting criteria; no-progress may expose weak critique, an ambiguous gate, or a capability limit; exhaustion may expose an undersized budget or an unexpectedly hard input. The label narrows diagnosis, but it does not prove one root cause.

---

## 14.6 Gate Architecture

The verifier is where most of the eval engineering actually lives. Four patterns, roughly in order of how much you should reach for them.

### Pattern 1 — Deterministic pre-checks first

The cheapest gate that can reject an output should run first: schema validation, length bounds, forbidden-token checks, file existence, exit codes, or pixel differences. These checks are deterministic for the rule encoded in code and are usually cheaper and easier to debug than an LLM judge. They are not infallible: a wrong rule or buggy implementation can still reject good work or miss a defect.

```python
def cheap_gate(output) -> dict | None:
    """Return a rejection dict, or None to pass to the expensive gate."""
    if not output.strip():
        return {"passed": False, "blocked_by": "empty", "critique": "Output was empty."}
    if len(output) > 50_000:
        return {"passed": False, "blocked_by": "length", "critique": "Output exceeded the length budget."}
    return None
```

A surprising amount of "our judge is expensive" turns out to be a judge being asked to notice that the output is empty.

### Pattern 2 — Veto criteria above scored criteria

Covered in depth in Module 08 Case Study 10: integrity constraints (safety,
factual grounding, policy) produce **PASS / FAIL / UNMEASURED decisions whose
aggregation is enforced in code**, not weighted terms in an average. A measured
failure vetoes; missing measurement blocks automated acceptance. The moment a
must-not-violate constraint has a weight, it has an exchange rate, and an
optimizer can trade it away for gains elsewhere.

```python
score = 0.4 * plating + 0.3 * faithfulness + 0.3 * realism    # ❌ buys past a violation
status = ("FAIL" if "FAIL" in vetoes
          else "UNMEASURED" if "UNKNOWN" in vetoes
          else "PASS" if sum(scored) >= bar
          else "FAIL")                                        # ✅ no exchange rate exists
```

### Pattern 3 — Isolated per-criterion judge calls

Use one judge call per criterion when criteria can influence one another or need separate calibration. Isolation costs more calls, but it makes each rejection attributable and lets you measure precision and recall per criterion instead of hiding disagreement inside one omnibus score.

### Pattern 4 — Adversarial verification for high-stakes accepts

For findings where a false accept is expensive, flip the judge's job from “is
this good?” to “**try to refute this**.” Keep refusal and insufficient evidence
separate from a measured refutation:

```python
REFUTE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["REFUTED", "NOT_REFUTED", "UNKNOWN"],
        },
        "evidence": {"type": "string"},
    },
    "required": ["verdict", "evidence"],
    "additionalProperties": False,
}


def adversarially_verify(claim: str, n: int = 3) -> dict:
    """Require complete coverage and a majority of measured non-refutations."""
    if not 1 <= n <= 3:
        raise ValueError("n must be between 1 and 3")
    lenses = ["correctness", "security", "does-it-actually-reproduce"][:n]
    reviews = []
    for lens in lenses:
        r = client.messages.create(
            model=MODEL, max_tokens=2000,
            output_config={"effort": "high",
                           "format": {"type": "json_schema", "schema": REFUTE_SCHEMA}},
            messages=[{"role": "user", "content":
                       f"Try to REFUTE this claim through the {lens} lens: {claim}\n"
                       "Return UNKNOWN when the supplied evidence cannot decide."}],
        )
        if r.stop_reason == "refusal":
            reviews.append({"lens": lens, "verdict": "UNKNOWN",
                            "evidence": "reviewer refused"})
            continue
        review = json.loads(next(b.text for b in r.content if b.type == "text"))
        reviews.append({"lens": lens, **review})

    unknown = sum(row["verdict"] == "UNKNOWN" for row in reviews)
    not_refuted = sum(row["verdict"] == "NOT_REFUTED" for row in reviews)
    status = (
        "UNMEASURED" if unknown
        else "PASS" if not_refuted > len(reviews) / 2
        else "FAIL"
    )
    return {"status": status, "coverage": (len(reviews) - unknown) / len(lenses),
            "reviews": reviews}
```

`UNMEASURED` still blocks a high-stakes promotion, but it does not manufacture
a claim that the finding was disproved. A measured `REFUTED` review must carry
the evidence that supports that decision.

The three prompts use different **perspectives**, but that does not make their errors independent: they still share a model and much of the context. Measure miss overlap on labeled cases. Keep the extra reviews only if they catch different defects; otherwise they add cost without meaningful coverage.

### Measuring layer independence

If you stack gates, measure whether they are genuinely independent. On a labeled set, compute how often layers make the *same* mistake:

```python
def layer_correlation(labels: list[bool], layer_a: list[bool], layer_b: list[bool]) -> dict:
    """Do these two gates fail on the same items? (labels: True = truly defective)"""
    a_missed = {i for i, t in enumerate(labels) if t and layer_a[i] is False}
    b_missed = {i for i, t in enumerate(labels) if t and layer_b[i] is False}
    both = a_missed & b_missed
    either = a_missed | b_missed
    defective = sum(labels)
    return {
        "a_miss_rate": len(a_missed) / defective if defective else 0.0,
        "b_miss_rate": len(b_missed) / defective if defective else 0.0,
        "miss_jaccard": len(both) / len(either) if either else 0.0,
        "a_miss_given_b_miss": len(both) / len(b_missed) if b_missed else 0.0,
        "b_miss_given_a_miss": len(both) / len(a_missed) if a_missed else 0.0,
    }
```

`miss_jaccard` is the intersection of the two miss sets divided by their union. It is symmetric: swapping layer A and B cannot change it. A value near 1.0 means the layers leave almost the same holes; a value near 0 means their observed misses differ. The two conditional rates show whether one layer's misses are mostly a subset of the other's. Change model family, modality, or framing only when the labeled evidence shows that the added layer is redundant.

---

## 14.7 Loop Observability

You cannot debug a loop from its final output. Keep three complementary records:

1. **Run table — one row per run.** Outcome, accepted attempt, total cost, total latency, model/config version. Use it for product-level rates and cost.
2. **Attempt table — one row per attempt.** Gate verdict, `blocked_by`, score, tokens, latency, and output hash. Use it for retry and failure analysis.
3. **Raw nested trace.** Messages, tool calls, tool results, and environment events. Use it to understand one surprising run.

Do not create `blocked_by_1`, `blocked_by_2`, … forever. Attempts are rows, not new columns. The `to_rows()` method in §14.2 emits the first two shapes.

**Run row:**

```json
{
  "run_id": "r_8812fa", "ts": "2026-08-06T09:14:22Z",
  "pipeline_version": "2026-08-01.2", "model": "claude-opus-5", "effort": "high",
  "outcome": "accepted", "attempts": 2, "accepted_at": 2,
  "total_latency_ms": 12930, "cost_usd": 0.31120,
  "escalated": false, "human_review": false
}
```

**Attempt rows (JSON Lines):**

Here `input_tokens` means uncached input tokens. Cache reads and writes at each TTL have their own fields and rates.

```json
{"run_id":"r_8812fa","attempt_index":1,"passed":false,"blocked_by":"grounding","score":0.61,"input_tokens":22100,"output_tokens":1710,"cache_read_tokens":19800,"cache_creation_5m_tokens":1000,"cache_creation_1h_tokens":0,"latency_ms":6040,"output_hash":"ae91..."}
{"run_id":"r_8812fa","attempt_index":2,"passed":true,"blocked_by":null,"score":0.88,"input_tokens":19100,"output_tokens":1470,"cache_read_tokens":19100,"cache_creation_5m_tokens":0,"cache_creation_1h_tokens":0,"latency_ms":6890,"output_hash":"70bc..."}
```

The tabular views let an analyst use `GROUP BY`; the raw trace lets an engineer inspect causality. Neither replaces the other.

### A documented failure: one trial left clues for the next

Anthropic reports that, in internal agent evals, Claude sometimes gained an unfair advantage by reading Git history left by previous trials. The score then measured cross-trial state leakage as if it were agent capability. The fix is not a better model: start every trial in a clean environment and treat shared files, caches, and resource exhaustion as harness failures. ([Anthropic, “Demystifying evals for AI agents,” 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents))

| Eval | What it covered | What it caught | What to improve |
|---|---|---|---|
| Environment-isolation check | Files, database state, caches, and resources at trial start | Git history from earlier trials leaked useful information | Reset from a known snapshot before every independent trial |
| Trace review | What evidence the agent actually used | The agent solved part of the task from leaked history | Separate model success from harness contamination |
| Clean rerun | Same task after state reset | Whether the apparent gain survives without leaked clues | Replace the inflated score with a valid capability estimate |

### Four diagnostic queries

```sql
-- 1. Where does the loop actually spend its retries?
SELECT a.blocked_by, COUNT(*) AS rejected_attempts,
       AVG(CASE WHEN next_a.passed THEN 1.0 ELSE 0.0 END) AS next_attempt_recovery
FROM attempts a
LEFT JOIN attempts next_a
  ON next_a.run_id = a.run_id AND next_a.attempt_index = a.attempt_index + 1
WHERE NOT a.passed
GROUP BY a.blocked_by ORDER BY rejected_attempts DESC;
-- Recovery is a triage signal, not a root-cause verdict. Read sample traces.

-- 2. Is the final attempt worse than the first on the same run?
SELECT AVG(CASE WHEN last_a.score < first_a.score THEN 1.0 ELSE 0.0 END) AS regression_rate
FROM runs r
JOIN attempts first_a ON first_a.run_id = r.run_id AND first_a.attempt_index = 1
JOIN attempts last_a  ON last_a.run_id = r.run_id AND last_a.attempt_index = r.attempts
WHERE r.attempts >= 2 AND first_a.score IS NOT NULL AND last_a.score IS NOT NULL;

-- 3. What does one shippable output cost, including failures?
SELECT SUM(cost_usd)
       / NULLIF(SUM(CASE WHEN outcome = 'accepted' THEN 1 ELSE 0 END), 0)
       AS cost_per_accepted
FROM runs;

-- 4. Is prompt caching actually working in the loop?
SELECT AVG(cache_read_tokens * 1.0
           / NULLIF(input_tokens + cache_read_tokens
                    + cache_creation_5m_tokens + cache_creation_1h_tokens, 0)) AS cache_hit_ratio
FROM attempts;   -- near 0 across repeated prefixes => inspect cache invalidation (module 15 §15.5)
```

Loops often resend a large shared prefix on every attempt, so cache behavior can materially change their cost. A low hit ratio does not itself prove a bug—the prefix may genuinely differ—but it tells you which traces to inspect for accidental invalidators such as timestamps or reordered content.

---

## 14.8 The Loop Failure Catalog

Nine failure modes, their signatures in the run/attempt tables, and the next action each suggests.

| # | Failure | Log signature | Fix |
|---|---|---|---|
| 1 | **Blind retry** | Consecutive attempt rows have the same blocker and little output change | Critique may be missing or generic ("try harder"). Feed specific evidence into the next prompt, then re-test. |
| 2 | **Oscillation** | `output_hash` repeats; blockers alternate A/B/A | Two criteria are in tension. Rank them explicitly, or merge into one criterion with a stated trade-off. |
| 3 | **Over-editing regression** | `final_score < first_score` on a material share of multi-attempt runs | Measure false rejects, preserve the best valid attempt, and inspect whether the critique causes unrelated edits. |
| 4 | **Correlated verifier error** | Automated acceptance is much higher than human acceptance on the same sample | Generator and verifier may share blind spots. Recalibrate first; add a genuinely different verifier only if labeled misses justify it. |
| 5 | **Budget starvation** | `outcome = "exhausted"` clusters on long inputs | Budget scaled to the mean, not the tail. Make the budget input-size dependent. |
| 6 | **Retry amplification** | Tail cost is many times the median and concentrated in a small slice | Add per-run reservation/ceiling and a circuit breaker; inspect the expensive slice. |
| 7 | **Gate drift** | Gate pass rate moves with no code change | Model version, upstream data, or judge prompt changed. Pin versions; run the gate's own eval set on a schedule. |
| 8 | **Reward hacking the gate** | Automated acceptance rises while blinded human quality is flat or down | Keep held-out criteria, audit accepted outputs, and prevent the optimizer from reading the promotion holdout. |
| 9 | **Loop masking an upstream bug** | High attempt counts on one input segment | The loop is compensating for bad routing/retrieval. Fix upstream — a loop is an expensive way to paper over a broken input. |

Failure 9 can remain hidden because the loop *works*: final quality looks acceptable, so nobody asks why one input slice needs more attempts. The useful signal is **attempt count correlated with an input attribute**. If documents from one source consistently need three attempts, sample their traces and test the upstream extraction path before spending more on retries.

```sql
-- Failure 9 detector: run this weekly, on whatever segments you have.
SELECT source, AVG(attempts) avg_attempts, COUNT(*) n
FROM runs GROUP BY source HAVING COUNT(*) > 100 ORDER BY avg_attempts DESC;
```

---

## 14.9 The Outer Loop: Self-Improving Systems That Don't Fool Themselves

The outer loop closes over production. Its canonical form — the one Uber's team describes for their image agent, and the one that appears in mature agent platforms generally — is:

```
production traces
   → stratified sampling
   → benchmark against golden dataset (human-labeled)
   → diagnosis agent: what regressed, and why?
   → proposed prompt / config / threshold change
   → offline gate on frozen holdout
   → canary on live traffic slice
   → promote (or auto-rollback)
```

Two components carry all the risk.

### The diagnosis agent

An agent that reads the eval delta and proposes a cause. It is enormously useful and it is **an LLM making a causal claim from correlational data**, which means it needs the same skepticism as any other judge:

```python
DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "regressed_slice": {"type": "string"},
        "hypothesis": {"type": "string"},
        "evidence_run_ids": {"type": "array", "items": {"type": "string"}},
        "alternative_hypotheses": {
            "type": "array", "items": {"type": "string"}
        },
        "proposed_change": {"type": "string"},
        "falsifying_test": {"type": "string",
                            "description": "What result would prove this hypothesis WRONG?"},
    },
    "required": ["regressed_slice", "hypothesis", "evidence_run_ids",
                 "alternative_hypotheses", "proposed_change", "falsifying_test"],
    "additionalProperties": False,
}
```

The `falsifying_test` field is not decoration. Requiring the diagnosis agent to state what would disprove its hypothesis turns a plausible story into something the offline gate can test. Treat the diagnosis as a hypothesis until that test runs.

### The Goodhart guardrails

> **A system that optimizes prompts against a benchmark, and also decides when the benchmark is satisfied, will eventually optimize the benchmark.**

Five useful guardrails:

| Guardrail | Prevents |
|---|---|
| **Frozen holdout the optimizer never queries** — read only by the promotion gate | Overfitting to the specific examples in the visible benchmark |
| **Human re-label cadence** on a rotating sample of the golden set | Labels quietly drifting toward what the system already does |
| **Never optimizer → production**: always offline gate → canary → promote | A confident bad change shipping at full scale |
| **Guardrail metrics with hard floors** (diversity, complaint rate, cost) alongside the quality metric | Quality gains bought with something that isn't in the objective |
| **Every config version pinned to the eval run that promoted it** | Un-diagnosable regressions six weeks later |

```python
@dataclass
class GoldenSet:
    """Split enforced in code, not in a doc nobody reads."""
    visible: list       # optimizer may query freely
    holdout: list       # ONLY the promotion gate touches this
    _holdout_reads: int = 0

    def read_holdout(self, caller: str) -> list:
        if caller != "promotion_gate":
            raise PermissionError(
                f"{caller} attempted to read the frozen holdout. "
                "If the optimizer can see it, it is not a holdout.")
        self._holdout_reads += 1
        return self.holdout
```

The access check prevents an optimization process from repeatedly querying the promotion holdout. Once candidate changes are selected using those examples, they are no longer an unbiased final test; rotate or replace them and record every read.

### Promotion gate

```python
def can_promote(candidate: dict, baseline: dict, guardrails: dict) -> tuple[bool, str]:
    """Require a meaningful, statistically supported win and intact guardrails."""
    delta = candidate["holdout_score"] - baseline["holdout_score"]
    if delta < candidate["minimum_effect"]:
        return False, "no material gain on frozen holdout"
    if candidate["holdout_delta_ci_low"] <= 0:
        return False, "quality gain is uncertain: paired interval crosses zero"
    for name, (value, floor) in guardrails.items():
        if value < floor:
            return False, f"guardrail breach: {name} = {value:.3f} < floor {floor:.3f}"
    if (candidate["cost_per_accepted"]
            > baseline["cost_per_accepted"] * candidate["max_cost_multiplier"]):
        allowed = candidate["max_cost_multiplier"] - 1
        return False, f"cost regression exceeds policy limit of {allowed:.0%}"
    return True, "promote to canary"
```

`minimum_effect` is the smallest gain the product considers worth shipping. `holdout_delta_ci_low` should come from a **paired** confidence interval because the candidate and baseline are evaluated on the same cases; bootstrap the per-case score differences rather than treating the two scores as unrelated samples. `max_cost_multiplier` is also product policy, not a universal 25% rule. A candidate must clear all three questions: *is the gain large enough, is it supported by the sample, and did it preserve guardrails?*

---

## 14.10 The Loop as a Platform Primitive

Loop engineering is being productized. Anthropic's beta **Managed Agents** API exposes an *Outcome*: you state what "done" looks like as a gradeable rubric, and the platform can run evaluate → revise cycles. The API currently allows 3 cycles by default and at most 20. ([Claude Managed Agents API reference](https://platform.claude.com/docs/en/api/typescript/beta/sessions/threads))

```python
# The task loop of §14.2, as a platform call.
session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENVIRONMENT_ID,
    initial_events=[{
        "type": "user.define_outcome",
        "description": "Produce a reconciliation report for Q3 as .xlsx",
        "rubric": {"type": "text", "content": RUBRIC_MD},   # explicit, gradeable criteria
        "max_iterations": 5,                                 # dependent revision cycles
    }],
)
```

**This does not remove the eval work — it relocates it.** You now own the rubric, and the rubric *is* the verifier, which means every principle in §14.3 and §14.6 still applies:

- Vague criteria ("the report should look professional") produce noisy loops. Gradeable criteria ("every row has a numeric `variance` column; totals tie to the GL within $1") produce convergent ones.
- The grader's verdicts arrive as `span.outcome_evaluation_end` events with a `result` field. Aggregate those events into the same loop dashboard used for a custom orchestrator:

| `result` | Meaning | What a high rate tells you |
|---|---|---|
| `satisfied` | Rubric met | Healthy |
| `needs_revision` | Iterating | Normal in moderation; check the k-distribution |
| `max_iterations_reached` | Criteria still unmet when revision budget ended | Inspect failed criteria: budget, capability, or task difficulty may be responsible |
| `failed` | Grader says the rubric does not apply to the deliverables | Inspect the event explanation for a task/rubric/deliverable mismatch |
| `interrupted` | Client interrupted | — |

A rising `failed` rate is different from a rising `needs_revision` rate. The former points to a rubric that does not apply to what was delivered; the latter says the rubric applies but criteria were not met. Keep them separate so a specification problem is not filed as a model-quality regression.

---

## 14.11 The Loop Engineering Checklist

**Before shipping a loop:**

- [ ] The verifier has been calibrated against human labels (precision, recall, κ per criterion)
- [ ] Verification is demonstrably cheaper *and* more reliable than generation (§14.3)
- [ ] Integrity constraints are vetoes in code, not weighted rubric terms
- [ ] Every rejection produces a `blocked_by` reason and an actionable critique
- [ ] Stop conditions cover: success, oscillation, no-progress, budget — each with a distinct label
- [ ] Per-run cost policy exists; the orchestrator reserves enough budget before starting another attempt
- [ ] An escalation path exists and is instrumented (never a silent fallthrough)
- [ ] One run row, one row per attempt, and a raw trace linked by `run_id`
- [ ] Maximum attempts chosen from a measured marginal-yield curve, not by default
- [ ] Standard pass@k/pass^k measured separately on clean independent trials when reliability matters

**Weekly, in production:**

- [ ] Accepted-at distribution and marginal yield (is the last attempt still earning its keep?)
- [ ] Regression rate — is iteration making outputs worse?
- [ ] Oscillation and no-progress rates
- [ ] Cost per accepted output and loop tax
- [ ] Gate pass rate vs. a human-labeled sample (drift detection)
- [ ] Attempt count by input segment (upstream bug detector, §14.8 #9)
- [ ] Cache hit ratio inside the loop

**For the outer loop:**

- [ ] Frozen holdout, enforced in code, readable only by the promotion gate
- [ ] Human re-label cadence on a rotating sample
- [ ] Guardrail metrics with hard floors, evaluated on every promotion
- [ ] Canary before promote; auto-rollback wired
- [ ] Config version pinned to the eval run that promoted it
- [ ] Diagnosis agent outputs include a falsifying test

---

## 14.12 Exercises

### Exercise 1: Instrument an existing loop
Take any retry logic you have in production. Add `blocked_by`, per-attempt scores, one run row, and one row per attempt. Run it on a representative sample, then compute the accepted-at distribution and regression rate. Predict both before looking; write down which engineering decision would change at each plausible result.

### Exercise 2: Verifier asymmetry
Measure your gate's precision and recall against an expert-labeled sample large enough to show useful uncertainty bounds. Plug the point estimates and plausible low/high values into `loop_quality()` from §14.3. Compute the accepted-good rate with and without the loop. If the conclusion changes across the interval, collect more labels before shipping a reject-and-retry policy.

### Exercise 3: Choose K from data
Compute marginal yield per attempt index. Find the attempt where marginal successes × value-per-success drops below marginal cost. Compare to your current K. Write the one-paragraph justification you would give a finance partner.

### Exercise 4: Break your own Swiss cheese
Take two stacked gates. Compute `miss_jaccard` and both conditional miss rates on a labeled set. Inspect the shared misses. Redesign one layer only if the evidence shows redundant coverage, then re-measure escape rate and cost.

### Exercise 5: Attack your own outer loop
Write a prompt change that improves the visible golden-set score while making the product worse (hint: exploit an under-specified rubric criterion). If you succeed easily, your guardrail metrics have a gap. Fix the gap, not the prompt.

---

## Next Module

**[Module 15: Evaluating in the Opus 5 Era](../15-opus5-eval-techniques/)** — what changed at the API layer for eval harnesses: effort as an eval axis, the death of `temperature=0` reproducibility, structured-output judges, refusal handling, and how to run large eval suites for a tenth of the cost.
