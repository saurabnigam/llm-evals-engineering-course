# Module 14: Loop Engineering

> **The unit of production AI is no longer the call. It is the loop.**
>
> A 2023 system took a prompt and returned an answer, and eval engineering meant scoring answers. A 2026 system *attempts* something, *checks its own work*, *repairs*, and *retries* — and then a second, slower loop takes what happened in production and rewrites the first loop's configuration. Neither of those loops is evaluable with a metric designed for single responses.
>
> **Prerequisites:** Module 01 (pass@k vs pass^k), Module 02 (rubrics, judges, judge calibration), Module 06 (feedback loops), Module 08 Case Study 10 (the Uber Eats image agent, which is a worked example of everything in this module).

---

## 14.0 Why This Module Exists

Here is the failure that motivates loop engineering as a discipline.

A team ships an agent with a self-correction loop: generate, check, retry up to 3 times. Offline pass rate goes from 71% to 89%. They ship it. Three weeks later:

- Costs are **4.1× forecast**, not the 1.3× they modeled.
- p99 latency is 40 seconds and nobody can say why.
- The quality complaints did not go away — they changed shape. Users now report outputs that are *worse* than the un-retried version would have been.
- Nobody can answer "did the loop help on *this* request?" because the only thing logged is the final output.

Every one of those is a measurement failure, not a modeling failure. The team evaluated the loop's *output* and never evaluated the *loop*.

| What they measured | What they needed to measure |
|---|---|
| Final pass rate | Pass rate **per attempt index** — where does the yield actually come from? |
| Mean cost per call | Cost per **accepted output**, including the attempts that failed |
| Gate pass rate | Gate **precision and recall against human labels** — is the gate right? |
| — | **Regression rate**: fraction of cases where attempt 3 scored *worse* than attempt 1 |
| — | **Oscillation rate**: loops that alternate between two failure modes until the budget runs out |
| — | **Escape rate**: defects that passed every gate |

This module is about that right-hand column.

---

## 14.1 The Three Loops

Production AI systems contain three nested loops running at three timescales. They fail differently, they need different instrumentation, and conflating them is the root cause of most "our evals looked fine" incidents.

```
┌──────────────────────────────────────────────────────────────────────┐
│ OUTER LOOP — days to weeks — "is the system still right?"            │
│   production traces → sampling → golden benchmark → diagnosis →      │
│   prompt/config change → offline gate → canary → promote             │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ TASK LOOP — seconds to minutes — "is this output good enough?" │  │
│  │   attempt → QA gate → critique → re-attempt  (pass@K)          │  │
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
| **Task** | Your orchestration code | 1–5 attempts | pass@K, cost per accepted output | Oscillation, over-editing regression |
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
from dataclasses import dataclass, field, asdict
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


@dataclass
class LoopResult:
    run_id: str
    attempts: list[Attempt] = field(default_factory=list)
    outcome: str = "pending"     # accepted | exhausted | oscillated | no_progress
    accepted_at: int | None = None
    escalated: bool = False      # orthogonal to outcome — never overwrite the reason

    @property
    def cost_usd(self) -> float:
        # claude-opus-5: $5 / MTok in, $25 / MTok out
        return sum(a.input_tokens * 5e-6 + a.output_tokens * 25e-6 for a in self.attempts)

    def to_row(self) -> dict:
        """Flat, one row per run — see §14.7 on why flatness is non-negotiable."""
        row = {
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
        for a in self.attempts:                    # flatten, don't nest
            row[f"blocked_by_{a.index}"] = a.blocked_by
        return row


def run_loop(
    run_id: str,
    generate: Callable[[str | None], tuple[Any, int, int]],   # critique -> (output, in_tok, out_tok)
    verify: Callable[[Any], dict],                            # output -> {passed, blocked_by, critique, score}
    max_attempts: int = 3,
    cost_ceiling_usd: float = 0.50,
    escalate: Callable[[LoopResult], Any] | None = None,
) -> LoopResult:
    result = LoopResult(run_id=run_id)
    critique: str | None = None
    seen_hashes: set[str] = set()

    for i in range(1, max_attempts + 1):
        t0 = time.monotonic()
        output, in_tok, out_tok = generate(critique)
        v = verify(output)
        h = hashlib.sha256(str(output).encode()).hexdigest()[:16]

        result.attempts.append(Attempt(
            index=i, output=output, passed=v["passed"], blocked_by=v.get("blocked_by"),
            critique=v.get("critique", ""), score=v.get("score"),
            input_tokens=in_tok, output_tokens=out_tok,
            latency_ms=int((time.monotonic() - t0) * 1000), output_hash=h,
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

Two design choices in there are load-bearing and frequently omitted in real systems:

**Stop condition 3 (no progress) is not the same as the budget.** A loop that fails `lighting` three times in a row is not going to succeed on the fourth try — the critique is not landing, and the remaining budget is pure waste. Detecting *"same blocker twice"* typically recovers 15–30% of loop spend in systems that previously only stopped on `max_attempts`, and it converts an invisible failure into a labeled one (`no_progress`) that your dashboard can count.

**`blocked_by` is a string, never a boolean.** The single highest-leverage instrumentation decision in loop engineering is refusing to log `passed: false` without a reason next to it. Every diagnostic query in §14.7 depends on that column existing.

---

## 14.3 The Verifier Asymmetry Law

> **A self-correcting loop improves output only if verification is both cheaper and more reliable than generation. When it isn't, the loop amplifies error instead of removing it.**

This is the theoretical core of the module, and it explains why some loops work spectacularly and others make things worse.

Let generation succeed with probability *p*, and let the verifier have true-positive rate *TPR* (catches a real defect) and false-positive rate *FPR* (rejects a good output). After a rejection you retry. What does the loop actually deliver?

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
                        (0.55, 0.35, "weak verifier")]:
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
```

Read the precision column against the no-verifier baseline. **The loop's entire value is a function of verifier quality, and it degrades far faster than the verifier does.** Dropping TPR from 0.95 to 0.55 — a verifier that is still better than a coin flip — takes precision from 97.8% to 77.1%: the loop now buys **seven points of precision for a 58% increase in attempts**, and ships a bad output one time in five.

Worse, the mechanism of the collapse is invisible in aggregate metrics. Look at what the weak verifier does on a *single* attempt: it accepts only 45.5% of the good outputs the generator produced (0.70 × 0.65), throwing away a quarter of the correct work it was handed. The retries claw that back to 71.8% — so the pass rate looks fine while the system spends 1.58 attempts to end up barely ahead of shipping attempt 1 unchecked. **A high false-positive rate converts directly into loop cost**, and cost is where you will notice it first.

Three practical corollaries:

1. **Evaluate the verifier before you deploy the loop.** Gate precision/recall against human labels (Module 02 §2.3.6) is a *prerequisite*, not a nice-to-have. A gate with κ < 0.4 should not be allowed to reject anything.
2. **Prefer verification tasks that are structurally easier than generation.** Running the test suite is a strictly easier problem than writing the code; checking whether an edited image added a garnish is strictly easier than producing the edit. When you cannot find an asymmetric verifier, you probably do not have a loop — you have a resampler.
3. **A resampler is a legitimate design, but call it one.** If your "verifier" is just the generator scoring itself, you are drawing best-of-K samples. That is fine and often effective — but its metric is best-of-K quality, not self-correction, and it does not justify a critique channel.

> **Diagnostic:** plot mean score by attempt index. Genuine self-correction rises monotonically. A resampler produces a flat line with variance. If your line is *falling*, see §14.8 failure mode 3.

---

## 14.4 Loop Metrics: What to Actually Put on the Dashboard

Single-response metrics do not survive contact with loops. Here is the replacement set, with the question each one answers.

| Metric | Question it answers | Danger if unmeasured |
|---|---|---|
| **pass@K** | Does the loop eventually succeed? | — (usually the only one measured) |
| **pass^K** | Does it succeed *every* time on the same input? | Ships luck as reliability (Module 01 §1.3b) |
| **k-to-pass distribution** | Where does yield come from? | You fund attempt 3 when 96% of value is in attempt 1 |
| **Marginal yield** `p(pass at i \| failed i−1)` | Is the next attempt worth buying? | K chosen by superstition |
| **Regression rate** | How often does iterating make it *worse*? | Over-editing ships as improvement |
| **Oscillation rate** | How often does the loop cycle between two states? | Invisible budget burn in the tail |
| **Cost per accepted output** | What does one shippable result cost? | Per-call cost hides the 4× loop tax |
| **Gate escape rate** | What fraction of defects got through everything? | The only metric users actually feel |
| **False reject rate** | How much good work is the gate throwing away? | Silent quality ceiling |
| **Loop tax** | Cost/latency multiple vs. a single attempt | Capacity planning by surprise |

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
    still_running = n
    for i in range(1, k + 1):
        won = k_hist.get(i, 0)
        marginal[i] = won / still_running if still_running else 0.0
        still_running -= won

    # Did iteration ever make things worse? (needs a score on every attempt)
    scored = [r for r in results if len(r.attempts) > 1
              and r.attempts[0].score is not None and r.attempts[-1].score is not None]
    regressed = [r for r in scored if r.attempts[-1].score < r.attempts[0].score]

    single_attempt_cost = mean(r.attempts[0].input_tokens * 5e-6
                               + r.attempts[0].output_tokens * 25e-6 for r in results)

    return {
        "n": n,
        "pass_at_k": len(accepted) / n,
        "k_to_pass": {i: k_hist.get(i, 0) / n for i in range(1, k + 1)},
        "marginal_yield": marginal,
        "regression_rate": len(regressed) / len(scored) if scored else None,
        "oscillation_rate": sum(r.outcome == "oscillated" for r in results) / n,
        "no_progress_rate": sum(r.outcome == "no_progress" for r in results) / n,
        "cost_per_accepted": (sum(r.cost_usd for r in results) / len(accepted)
                              if accepted else float("inf")),
        "loop_tax": (sum(r.cost_usd for r in results) / n) / single_attempt_cost,
        "mean_attempts": mean(len(r.attempts) for r in results),
    }
```

### Reading a real-looking output

```
n                  4,812
pass_at_k          0.913
k_to_pass          {1: 0.804, 2: 0.081, 3: 0.028}
marginal_yield     {1: 0.804, 2: 0.413, 3: 0.243}
regression_rate    0.061
oscillation_rate   0.018
no_progress_rate   0.033
cost_per_accepted  $0.0374
loop_tax           1.31×
mean_attempts      1.29
```

Everything you need to make three decisions is in there:

- **Is attempt 3 worth keeping?** 11.5% of traffic reaches it and it converts 24.3% of those — **135 extra successes on 4,812 requests**, bought by running a third generate-and-verify cycle on 553 of them. Weigh that against the 6.1% regression rate, which is doing damage on the same population. This is now an arithmetic decision, not a taste one.
- **Is the loop paying for itself?** pass@1 is 80.4%; pass@3 is 91.3%. Eleven points of quality for a 1.31× cost multiplier is usually an easy yes. If the loop tax were 3.8× for the same 11 points, it is usually a no — spend it on a better first attempt (higher `effort`) instead.
- **Is anything pathological?** 6.1% regression and 1.8% oscillation are both real bugs with real fixes (§14.8), and both were completely invisible in the pass@3 number that a naive dashboard would have shown.

> **The one-number trap.** "pass@3 = 91.3%" is the number every stakeholder wants and the number that hides all three findings above. Report pass@K *with* k-to-pass and regression rate, always, on the same slide.

---

## 14.5 Budgets and Stop Conditions

Loops need a hard ceiling and a soft one. The hard ceiling is enforced by your code and the model cannot see it. The soft ceiling is *told to the model* so it can pace itself and finish gracefully rather than being guillotined mid-thought.

Opus 5-era models expose this as a first-class parameter — a **task budget** — which is distinct from `max_tokens`:

| Control | Enforced by | Model aware? | Behavior at the limit |
|---|---|---|---|
| `max_tokens` | API | No | Hard truncation, `stop_reason: "max_tokens"` |
| `output_config.task_budget` | API (beta) | **Yes** — server injects a countdown | Model prioritizes and wraps up |
| Loop `max_attempts` | Your code | No | Loop exits |
| Loop cost ceiling | Your code | No | Loop exits |

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

The budget counts what the model generates plus the tool results it reads *this turn* — not the full history you resend. Leave `remaining` unset in a normal loop; the server tracks the countdown. Only pass it explicitly if you compact or rewrite history between requests, because then the server can no longer derive prior spend.

**An eval-relevant warning about budget countdowns.** Surfacing a remaining-token count to the model changes behavior — models can wrap up prematurely or start worrying about running out of room. If you are A/B-ing a budget change, treat it as a **behavioral change, not just a cost control**, and re-run your quality evals. A budget that cuts cost 30% and quality 8% is a product decision, and you will only find out if you measured quality on the budgeted configuration.

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

Never collapse 2–4 into a single `failed`. They have different fixes: oscillation is a *critique quality* bug, no-progress is a *generator capability* bug, exhaustion is a *budget calibration* bug. A dashboard that shows only "failed: 8.7%" cannot tell you which of your three teams should be working on it.

---

## 14.6 Gate Architecture

The verifier is where most of the eval engineering actually lives. Four patterns, roughly in order of how much you should reach for them.

### Pattern 1 — Deterministic pre-checks first

The cheapest gate that can reject an output should run first. Schema validation, length bounds, forbidden-token checks, file-exists, exit-code, pixel-diff. These cost microseconds, have zero false-negative rate on the things they check, and are perfectly reliable — which makes them strictly better than an LLM gate for anything they can express.

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

Covered in depth in Module 08 Case Study 10: integrity constraints (safety, factual grounding, policy) are **binary vetoes evaluated in code**, never weighted terms in an average. The moment a constraint has a weight, it has an exchange rate, and the optimizer will find it.

```python
score = 0.4 * plating + 0.3 * faithfulness + 0.3 * realism    # ❌ buys past a violation
passed = all(vetoes) and (sum(scored) >= bar)                  # ✅ no exchange rate exists
```

### Pattern 3 — Isolated per-criterion judge calls

One judge call per criterion, each seeing only its own question. Halo effects are real and large: an eloquent answer scores higher on *factual accuracy* when the judge is also asked about eloquence in the same call. Isolation costs more calls and buys interpretability plus resistance to contamination between dimensions.

### Pattern 4 — Adversarial verification for high-stakes accepts

For findings where a false accept is expensive, flip the judge's job from "is this good?" to "**refute this**" and require a majority to fail at refuting:

```python
def adversarially_verify(claim: str, n: int = 3) -> bool:
    """Survives only if a majority of independent skeptics cannot refute it."""
    votes = []
    for lens in ["correctness", "security", "does-it-actually-reproduce"][:n]:
        r = client.messages.create(
            model=MODEL, max_tokens=2000,
            output_config={"effort": "high",
                           "format": {"type": "json_schema", "schema": REFUTE_SCHEMA}},
            messages=[{"role": "user", "content":
                       f"Try to REFUTE this claim through the {lens} lens: {claim}\n"
                       "Default to refuted=true if you are uncertain."}],
        )
        if r.stop_reason == "refusal":
            votes.append(True)                     # treat a refusal as a refutation
            continue
        votes.append(json.loads(next(b.text for b in r.content if b.type == "text"))["refuted"])
    return sum(not v for v in votes) > len(votes) / 2
```

Note the **perspective diversity**: three verifiers with three different lenses catch failure modes that three identical verifiers cannot, for the same token cost. Redundancy without diversity is the correlated-layers mistake from Case Study 10 in miniature.

### Measuring layer independence

If you stack gates, measure whether they are genuinely independent. On a labeled set, compute how often layers make the *same* mistake:

```python
def layer_correlation(labels: list[bool], layer_a: list[bool], layer_b: list[bool]) -> dict:
    """Do these two gates fail on the same items? (labels: True = truly defective)"""
    a_missed = [i for i, t in enumerate(labels) if t and layer_a[i] is False]
    b_missed = [i for i, t in enumerate(labels) if t and layer_b[i] is False]
    both = set(a_missed) & set(b_missed)
    return {
        "a_miss_rate": len(a_missed) / sum(labels),
        "b_miss_rate": len(b_missed) / sum(labels),
        "overlap_of_misses": len(both) / max(len(a_missed), 1),   # 1.0 = same hole
    }
```

`overlap_of_misses` near 1.0 means your second layer is decorative. Vary the model family, the modality, or the framing until it drops.

---

## 14.7 Loop Observability

You cannot debug a loop from its output. You debug it from a **flat, one-row-per-run table** where every decision has a column and every failure has a reason next to it.

```json
{
  "run_id": "r_8812fa", "ts": "2026-08-06T09:14:22Z",
  "pipeline_version": "2026-08-01.2", "model": "claude-opus-5", "effort": "high",
  "outcome": "accepted", "attempts": 2, "accepted_at": 2,
  "blocked_by_1": "grounding", "blocked_by_2": null,
  "score_1": 0.61, "score_2": 0.88,
  "gate_latency_ms_1": 1840, "gate_latency_ms_2": 1790,
  "generate_latency_ms_1": 4200, "generate_latency_ms_2": 5100,
  "input_tokens": 41200, "output_tokens": 3180,
  "cache_read_tokens": 38900, "cost_usd": 0.0812,
  "escalated": false, "human_review": false
}
```

**Why flat and not nested.** The people who need to answer questions about your loop include ops analysts, support leads, and PMs. A nested trace object requires a JSON path expression and a mental model of your orchestration; a flat row requires `GROUP BY`. Nested traces are for engineers debugging one run; flat rows are for everyone diagnosing patterns across a million. Emit both, but if you only build one, build the flat one — this is precisely the point Uber's team makes about their own pipeline logging (Case Study 10).

### The four queries that diagnose 80% of loop problems

```sql
-- 1. Where does the loop actually spend its retries?
SELECT blocked_by_1, COUNT(*) n, AVG(attempts) avg_attempts,
       AVG(CASE WHEN outcome='accepted' THEN 1.0 ELSE 0 END) recovery_rate
FROM runs WHERE pipeline_version = '2026-08-01.2'
GROUP BY 1 ORDER BY n DESC;
-- A blocker with a LOW recovery rate is a generator capability gap.
-- A blocker with a HIGH recovery rate and high volume is a first-attempt prompt bug.

-- 2. Is iteration making things worse? (the metric nobody logs)
SELECT COUNT(*) FILTER (WHERE score_2 < score_1) * 1.0 / COUNT(*) AS regression_rate
FROM runs WHERE attempts >= 2;

-- 3. What does one shippable output cost, including failures?
SELECT SUM(cost_usd) / COUNT(*) FILTER (WHERE outcome = 'accepted') AS cost_per_accepted
FROM runs;

-- 4. Is prompt caching actually working in the loop?
SELECT AVG(cache_read_tokens * 1.0 / NULLIF(input_tokens, 0)) AS cache_hit_ratio
FROM runs;   -- near 0 across repeated runs => a silent cache invalidator (module 15 §15.5)
```

Query 4 deserves a note. Loops re-send a large shared prefix on every attempt, so they are the single biggest beneficiary of prompt caching in any AI system — and also the place where a stray timestamp in the system prompt costs the most. A loop with a broken cache pays full price for the same 40K-token prefix three times per request.

---

## 14.8 The Loop Failure Catalog

Nine failure modes, their signatures in the flat log, and their fixes.

| # | Failure | Log signature | Fix |
|---|---|---|---|
| 1 | **Blind retry** | `blocked_by_1 == blocked_by_2 == blocked_by_3` | Critique channel isn't wired, or the critique is generic ("try harder"). Feed the specific evidence into the next prompt. |
| 2 | **Oscillation** | `output_hash` repeats; blockers alternate A/B/A | Two criteria are in tension. Rank them explicitly, or merge into one criterion with a stated trade-off. |
| 3 | **Over-editing regression** | `score_final < score_first` on 5%+ of runs | Gate rejects too aggressively (high FPR). Add "if the current output already satisfies the criterion, PASS" to the judge prompt, and keep the best-scoring attempt rather than the last. |
| 4 | **Verifier collusion** | Gate pass rate ≫ human pass rate on the same items | Generator and verifier share a model and prompt lineage. Vary model family or framing; re-calibrate against human labels. |
| 5 | **Budget starvation** | `outcome = "exhausted"` clusters on long inputs | Budget scaled to the mean, not the tail. Make the budget input-size dependent. |
| 6 | **Retry amplification** | p99 cost ≫ 10× median | No cost ceiling per run; one pathological input eats the daily budget. Add per-run ceiling + circuit breaker. |
| 7 | **Gate drift** | Gate pass rate moves with no code change | Model version, upstream data, or judge prompt changed. Pin versions; run the gate's own eval set on a schedule. |
| 8 | **Reward hacking the gate** | Pass rate up, human quality flat or down | The generator learned the gate's tells. Rotate held-out criteria; audit accepted outputs by hand. |
| 9 | **Loop masking an upstream bug** | High attempt counts on one input segment | The loop is compensating for bad routing/retrieval. Fix upstream — a loop is an expensive way to paper over a broken input. |

Failure 9 is the most expensive one in practice and the hardest to see, because the loop *works*: quality is acceptable and nobody investigates. The tell is always the same — **attempt count correlated with an input attribute**. If documents from one source consistently need three attempts, the loop is silently paying to repair an upstream extraction bug that costs nothing to fix at the source.

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
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "proposed_change": {"type": "string"},
        "falsifying_test": {"type": "string",
                            "description": "What result would prove this hypothesis WRONG?"},
    },
    "required": ["regressed_slice", "hypothesis", "evidence_run_ids",
                 "confidence", "proposed_change", "falsifying_test"],
    "additionalProperties": False,
}
```

The `falsifying_test` field is not decoration. Requiring the diagnosis agent to state what would disprove its own hypothesis is the cheapest available defense against confident, plausible, wrong causal stories — and it gives your offline gate something concrete to run.

### The Goodhart guardrails

> **A system that optimizes prompts against a benchmark, and also decides when the benchmark is satisfied, will eventually optimize the benchmark.**

Five guardrails, in descending order of importance:

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

That exception is not paranoia. The failure it prevents — an engineer adding "just a quick check" of the holdout to the optimization loop under deadline — is how holdouts die, and it dies silently: every metric keeps improving while the product stops improving.

### Promotion gate

```python
def can_promote(candidate: dict, baseline: dict, guardrails: dict) -> tuple[bool, str]:
    """A promotion needs a real quality win AND no guardrail breach. Both."""
    if candidate["holdout_score"] <= baseline["holdout_score"] + 0.01:
        return False, "no material gain on frozen holdout"
    for name, (value, floor) in guardrails.items():
        if value < floor:
            return False, f"guardrail breach: {name} = {value:.3f} < floor {floor:.3f}"
    if candidate["cost_per_accepted"] > baseline["cost_per_accepted"] * 1.25:
        return False, "cost regression > 25%"
    return True, "promote to canary"
```

---

## 14.10 The Loop as a Platform Primitive

Loop engineering is being productized. Anthropic's **Managed Agents** exposes an *Outcome*: you state what "done" looks like as a gradeable rubric, and the platform runs the iterate → grade → revise loop for you, with an independent grader in its own context window.

```python
# The task loop of §14.2, as a platform call.
session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENVIRONMENT_ID,
    initial_events=[{
        "type": "user.define_outcome",
        "description": "Produce a reconciliation report for Q3 as .xlsx",
        "rubric": {"type": "text", "content": RUBRIC_MD},   # explicit, gradeable criteria
        "max_iterations": 5,                                 # the K in pass@K
    }],
)
```

**This does not remove the eval work — it relocates it.** You now own the rubric, and the rubric *is* the verifier, which means every principle in §14.3 and §14.6 still applies:

- Vague criteria ("the report should look professional") produce noisy loops. Gradeable criteria ("every row has a numeric `variance` column; totals tie to the GL within $1") produce convergent ones.
- The grader's verdicts arrive as `span.outcome_evaluation_end` events with a `result` field. **The distribution of those results is your loop dashboard**, for free:

| `result` | Meaning | What a high rate tells you |
|---|---|---|
| `satisfied` | Rubric met | Healthy |
| `needs_revision` | Iterating | Normal in moderation; check the k-distribution |
| `max_iterations_reached` | Ran out of attempts | Budget too tight, or rubric unachievable |
| `failed` | Rubric doesn't match the task | **Your rubric contradicts the description** — a spec bug, not a model bug |
| `interrupted` | Client interrupted | — |

A rising `failed` rate is the single most useful early signal a platform loop gives you, because it is almost never the model — it is your specification disagreeing with itself.

---

## 14.11 The Loop Engineering Checklist

**Before shipping a loop:**

- [ ] The verifier has been calibrated against human labels (precision, recall, κ per criterion)
- [ ] Verification is demonstrably cheaper *and* more reliable than generation (§14.3)
- [ ] Integrity constraints are vetoes in code, not weighted rubric terms
- [ ] Every rejection produces a `blocked_by` reason and an actionable critique
- [ ] Stop conditions cover: success, oscillation, no-progress, budget — each with a distinct label
- [ ] Per-run cost ceiling exists and is enforced independently of attempt count
- [ ] An escalation path exists and is instrumented (never a silent fallthrough)
- [ ] Flat, one-row-per-run logging with a column per decision
- [ ] `k` chosen from a measured marginal-yield curve, not by default

**Weekly, in production:**

- [ ] k-to-pass distribution and marginal yield (is the last attempt still earning its keep?)
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
Take any retry logic you have in production. Add `blocked_by`, per-attempt scores, and a flat log row. Run it for a week, then compute the k-to-pass distribution and the regression rate. Predict both numbers *before* you look. Most people are wrong about the regression rate by 3–5×.

### Exercise 2: Verifier asymmetry
Measure your gate's precision and recall against 200 human labels. Plug them into `loop_quality()` from §14.3. Compute the accepted-good rate with and without the loop. If the loop lowers it, you have found a real bug worth more than the rest of the exercise.

### Exercise 3: Choose K from data
Compute marginal yield per attempt index. Find the attempt where marginal successes × value-per-success drops below marginal cost. Compare to your current K. Write the one-paragraph justification you would give a finance partner.

### Exercise 4: Break your own Swiss cheese
Take two stacked gates. Compute `overlap_of_misses` on a labeled set. If it exceeds 0.6, redesign one layer (different model family, different modality, or different framing) and re-measure. Report the escape-rate change.

### Exercise 5: Attack your own outer loop
Write a prompt change that improves the visible golden-set score while making the product worse (hint: exploit an under-specified rubric criterion). If you succeed easily, your guardrail metrics have a gap. Fix the gap, not the prompt.

---

## Next Module

**[Module 15: Evaluating in the Opus 5 Era](../15-opus5-eval-techniques/)** — what changed at the API layer for eval harnesses: effort as an eval axis, the death of `temperature=0` reproducibility, structured-output judges, refusal handling, and how to run large eval suites for a tenth of the cost.
