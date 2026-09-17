# Module 12: Eval-Training Separation & Benchmark Integrity

> **A Core Threat to Benchmark Validity**
>
> If evaluation data leaks into training, benchmark scores can overstate generalization. This module covers practical ways to reduce, detect, and disclose that risk -- from data decontamination to dynamic benchmarks and controls described in public model reports.

---

## In Plain English

A benchmark score is useful only if the system had to solve the task during the
test. If the model saw the answer during training, fetched it through a tool, or
inherited it from an agent's memory, the same score can describe a very
different capability. This chapter teaches you to separate those failure
channels instead of calling every suspicious result "contamination."

### What each integrity check covers, catches, and enables

| Check | What it covers | What it can catch | Decision it enables |
|---|---|---|---|
| Corpus overlap and provenance review | Whether eval items, labels, or close variants appear in accessible training corpora | Direct copies, near-duplicates, answer-key leakage | Remove or quarantine affected items; disclose residual risk |
| Prefix-completion probe | Whether canonical item wording is reproduced unusually exactly | Memorized phrasing or benchmark-template familiarity | Investigate with corpus evidence or a private replacement; **not** declare contamination from this probe alone |
| Original-versus-validated-variant comparison | Sensitivity to wording or surface form while preserving the tested skill | Prompt brittleness and, when combined with other evidence, possible memorization | Report both results and run a private/temporal holdout before making a capability claim |
| Private or post-cutoff holdout | Performance on items the developer was unlikely to have trained against | Public-set overfitting and benchmark-specific scaffolding | Prefer the holdout estimate for release or procurement decisions |
| Network, tool, cache, and memory audit | What the system can acquire *during* the run | Answer retrieval, warm-memory leakage, hidden test access | Fix environment isolation or report a warm-state result separately |
| Dynamic or interactive task generation | Fresh tasks or environments rather than a fixed answer list | Exact-item reuse and some static-benchmark gaming | Extend benchmark life, after validating generated tasks and graders |
| Canary/access audit | Unauthorized exposure of private items or graders | Pipeline leakage and unexpected readers | Rotate compromised assets and repair access controls |

No single row proves that a benchmark is clean. The useful output is an
evidence bundle: item provenance, overlap findings, environment configuration,
holdout results, and uncertainty about what remains unobserved.

## 12.1 Why Eval-Training Separation Matters

```
THE FUNDAMENTAL PROBLEM

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  SCENARIO: You train a model on internet data. Your eval benchmarks are    │
│  published on the internet. Therefore...                                    │
│                                                                              │
│     ┌──────────────────┐                                                    │
│     │  Training Data   │     ┌──────────────────┐                          │
│     │  (Internet-scale │  ∩  │  Eval Benchmarks │                          │
│     │   corpora)       │     │  (Published on   │                          │
│     │                  │     │   the internet)  │                          │
│     └──────────────────┘     └──────────────────┘                          │
│              │                         │                                     │
│              └────────┬────────────────┘                                     │
│                       ▼                                                      │
│              ┌──────────────────┐                                            │
│              │  CONTAMINATION   │                                            │
│              │  The model has   │                                            │
│              │  seen the test!  │                                            │
│              └──────────────────┘                                            │
│                                                                              │
│  Risk: benchmark scores can mix task capability with item exposure,         │
│  making generalization harder to estimate.                                  │
│                                                                              │
│  ANALOGY: A student who got the answer key before the exam.                 │
│  They score 100%, but learned nothing.                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12.2 How Anthropic Separates Training from Evaluation

### Multi-Layer Protection Strategy

```
ANTHROPIC'S EVAL INTEGRITY APPROACH

Layer 1: TEMPORAL GATING (training-data collection windows)
  ├── Each model has a training-data collection window; evals
  │   created AFTER it closes cannot have been seen
  ├── Current system cards use this explicitly: the Fable 5 /
  │   Mythos 5 card reports USAMO 2026 (held March 21-22, 2026)
  │   because it took place "after almost all of Mythos's training
  │   data was collected, and we are confident that there was
  │   no contamination"
  └── But this only helps for time-gated evals

Layer 2: USER DATA EXCLUSION
  ├── API/commercial customer data is excluded from training
  │   by default (earlier cards stated this absolutely; verify
  │   the current data policy for your tier before relying on it)
  ├── This prevents eval data sent via API from leaking back
  └── Critical for customers running proprietary evals

Layer 3: DATA DECONTAMINATION
  ├── Deduplication pipelines remove near-duplicates
  ├── Classification filters identify benchmark-like content
  └── For specific benchmarks (e.g., ARC-AGI-1): train on
      reshuffled splits, report only on held-out test sets

Layer 4: SEMI-PRIVATE & PRIVATE TEST SETS
  ├── Maintain held-out test sets not published online
  ├── Report on third-party private holdouts (the Fable 5 card
  │   cites Harvey's held-out Legal Agent set and Zapier's
  │   private AutomationBench set — see Section 12.7)
  └── Named external testers in the 2026 cards: METR, UK AISI,
      Meridian Labs (Petri), Gray Swan, Andon Labs

Layer 5: WEB CRAWLER TRANSPARENCY
  ├── Anthropic operates a named crawler (ClaudeBot)
  ├── Follows robots.txt (site operators can opt out)
  └── Enables tracking which sources are in training data

Layer 6: ELICITATION HONESTY (explicit in the 2026 cards)
  ├── Capability evals run on the UNSAFEGUARDED config
  │   (Mythos 5) so safeguards can't mask true capability;
  │   safeguard evals run on the shipped config (Fable 5)
  ├── "Helpful-only" model variants used for dangerous-
  │   capability elicitation (harmlessness trained away)
  └── Sandbagging sweep: 2,440 dangerous-capability transcripts
      model-graded for deliberate under-performance; "we do not
      see any concerning instances of sandbagging behavior"
```

Sources: [Fable 5 / Mythos 5 system card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) §1.1 (training data, ClaudeBot), §1.4 (which config is evaluated), §6.5.2 (sandbagging sweep), §8.10 (USAMO 2026 temporal gating), §§8.17.4/8.17.9 (third-party holdouts); external-tester roster per card §§2.3.8, 3.2.5, 6.2.3.3, 6.2.5.

### What Anthropic Does NOT Disclose

```
TRANSPARENCY GAPS (Per Stanford FMTI Report 2025)

Anthropic does NOT publicly disclose:
  • Top-5 sources of publicly available training datasets
    (Reason: "competitive advantages and intellectual property")
  
  • Exact filtering/decontamination methods
  
  • Complete list of benchmarks excluded from training

This means external verification of separation is LIMITED.

WHY THIS MATTERS FOR YOU:
  → You cannot fully trust any public benchmark score
  → You MUST run your own private evals for critical applications
  → Treat published benchmark comparisons as DIRECTIONAL, not absolute
```

---

## 12.3 Types of Data Contamination

> **A framing note before the taxonomy.** Everything below treats contamination as a *training-data* problem: the answers leaked into the corpus before the model shipped. That framing was complete until agents got tool access and persistent state. It no longer is — see §12.3b, which covers the channel where the system under test acquires the answers **during the evaluation itself**. The two require completely different defenses: decontamination is a data-pipeline problem; runtime contamination is an isolation problem.

### Taxonomy of Contamination

```python
contamination_types = {
    "direct_contamination": {
        "description": "Exact benchmark questions/answers appear in training data",
        "severity": "critical",
        "detection": "N-gram overlap, exact string matching",
        "example": "MMLU questions verbatim in a web scrape",
        "impact": "Scores inflated by 5-30% on affected benchmarks"
    },
    
    "indirect_contamination": {
        "description": "Paraphrased or reformulated versions of benchmark items",
        "severity": "high",
        "detection": "Semantic similarity, embedding-based matching",
        "example": "Blog post explaining MMLU answer with same reasoning",
        "impact": "Scores inflated by 2-10%"
    },
    
    "distributional_contamination": {
        "description": "Training data over-represents the benchmark's domain/style",
        "severity": "medium",
        "detection": "Distribution analysis, genre classification",
        "example": "Training on 10x more medical text inflates MedQA scores",
        "impact": "Scores reflect data mix, not general capability"
    },
    
    "temporal_contamination": {
        "description": "Benchmark solutions posted online after benchmark creation",
        "severity": "high",
        "detection": "Temporal analysis of web content",
        "example": "StackOverflow answers to HumanEval problems",
        "impact": "Earlier models more trustworthy than newer ones on same benchmark"
    },
    
    "label_contamination": {
        "description": "Training data contains the correct labels/answers",
        "severity": "critical",
        "detection": "DCR framework (semantic, informational, data, label levels)",
        "example": "Answer key to a benchmark found in training corpus",
        "impact": "Model can retrieve answers without reasoning"
    },

    # --- Two types that only became visible in 2025-2026 ---

    "preference_leaderboard_contamination": {
        "description": "Vendor tunes/selects variants against a preference leaderboard's distribution",
        "severity": "high",
        "detection": "Compare leaderboard rank of submitted variant vs released weights",
        "example": "Llama 4: a chat-optimized variant hit Elo 1417 (#2) on LMArena; "
                   "the released weights ranked ~32nd. 'The Leaderboard Illusion' "
                   "(arXiv:2504.20879) showed Meta privately tested 27 Llama-4 variants, "
                   "and Arena-style finetuning can inflate ArenaHard >100% while DEGRADING MMLU",
        "impact": "Leaderboard rank reflects leaderboard-fitting, not capability"
    },

    "agentic_harness_memorization": {
        "description": "Agent memorized the benchmark's repos/environments, not the skill",
        "severity": "critical",
        "detection": "Re-run on structurally identical tasks from repos OUTSIDE the benchmark",
        "example": "'The SWE-Bench Illusion' (arXiv:2506.12286): SOTA models emit the correct "
                   "buggy-file PATH from the issue text alone for SWE-bench repos; performance "
                   "drops to <=53% of baseline on tasks from non-benchmark repos",
        "impact": "Agentic scores overstate generalization; repo identity is the leaked label"
    }
}
```

### Leaderboard Integrity, Autumn 2026: Two Indices, One Model, Opposite Verdicts

The `preference_leaderboard_contamination` row above (Llama 4 on LMArena) is a 2025 case of a *vendor* gaming a leaderboard. September 2026 produced a different failure mode: two independent, reputable indices scoring the **same released model** and landing on opposite conclusions.

When GPT-6 Astra launched, Artificial Analysis's Intelligence Index scored it roughly level with its predecessor, GPT-5.6 Sol — no visible gain. Epoch AI's independently run ECI, over the same weights, ranked Astra **#1 of 247 models** (Epoch's own Sept 12 brief: "taking the top spot among the 247 models we track" — not 267; verified directly against the primary post). Artificial Analysis responded with two point releases inside a week: **v4.2 (Sept 4)** dropped the now-saturated GPQA Diamond (and, per its own changelog, already raised private-test-set weighting to 40% and added AA-Briefcase); **v4.3 (Sept 7)** marks only Terminal-Bench 4.0 and AutomationBench-AA as "(new in index)," and separately added GDPval-AA v2, CritPt, AA-Omniscience, and AA-LCR v1.1, dropped τ³-Banking, and raised private-test-set weighting further from 40% to 45%. A third version number, v4.1.1, appears in some secondary coverage with overlapping content and a sequencing that doesn't cleanly reconcile with v4.3's own dated claims — that inconsistency could not be resolved from primary sources here, so treat the exact version sequencing as unresolved without treating it as evidence against the substance of the change. Primary: [artificialanalysis.ai v4.3](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-3). Dispute coverage (secondary, label it as such): [the-decoder.com](https://the-decoder.com/artificial-analysis-overhauls-its-intelligence-index-after-gpt-6-astra-scoring-drew-skepticism/).

The teaching point survives the version-numbering confusion: **the index's composition is the claim.** Two competent, well-resourced organizations measured the same weights and disagreed enough to argue about it in public, and the resolution was not a re-run — it was a redesign of what gets measured and how much the private, hard-to-game portion counts. Before you cite a composite-index rank, read what it dropped and added this quarter, not just the number. Separately: "Terminal-Bench 4.0" appears in Artificial Analysis's own additions list without a standalone maintainer launch post found from tbench.ai itself — whether it is a Terminal-Bench-maintainer major version or an Artificial-Analysis-commissioned harder split of 3.0 is unresolved; don't repeat it as a confirmed Terminal-Bench release.

**Self-report vs. independent measurement, restated with a fresh example.** The Artificial Analysis/Epoch split above is a disagreement between two *outsiders*. A cleaner and older pattern is a vendor's own number against an independent evaluator's number on the same model: CAISI's independent evaluation of DeepSeek V4 Pro found it trailing the US frontier by roughly eight months and scoring **lower** on CAISI's own tests than on DeepSeek's self-reported numbers ([nist.gov](https://www.nist.gov/news-events/news/2026/05/caisi-evaluation-deepseek-v4-pro)). Put next to each other, the two disputes make the same point from opposite directions — trust in a score drops whether the second measurer disagrees *upward* (Epoch on Astra) or *downward* (CAISI on DeepSeek) from the first one. The identity of the measurer is not a formality; it changes the number.

---

## 12.3b Runtime Contamination: When the System Acquires the Answers During the Eval

Every category above is about what entered the training corpus. This one is not. **Runtime contamination is the system under test obtaining benchmark answers at inference time**, through capabilities you deliberately gave it. Decontaminating your training data does nothing to prevent it, and n-gram overlap checks cannot detect it — the model was clean when it started.

```python
runtime_contamination_types = {
    "persistent_memory": {
        "description": "Agent writes findings to memory that survive into later trials",
        "severity": "high",
        "detection": "Compare cold-memory vs warm-memory pass rates; audit memory contents",
        "example": "Trial 1 fails and writes 'task X: answer is 47'; trials 2-5 read it and pass",
        "impact": "Inflates pass^k specifically — the metric that assumes INDEPENDENT trials",
        "defense": "Fresh memory store per trial, or report both configurations explicitly",
    },

    "tool_retrieval": {
        "description": "Agent uses a search or fetch tool to look up the benchmark's answers",
        "severity": "critical",
        "detection": "Log every tool call and network destination, not just outcomes",
        "example": "Agent web-searches the benchmark name and finds the public solution set",
        "impact": "Task looks solved; nothing was solved. Scales with how public your benchmark is",
        "defense": "Default-deny egress; block benchmark-adjacent domains; prefer private splits",
    },

    "environment_leakage": {
        "description": "The answer key is reachable from inside the eval environment",
        "severity": "critical",
        "detection": "Canary answer file — alert on ANY access",
        "example": "Grading data in the same store, bucket, or database the harness can read",
        "impact": "Total loss of eval validity, usually undetected",
        "defense": "Grade out of band, from outside the sandbox, with separate credentials",
    },

    "cross_trial_state": {
        "description": "Trials share a container, cache, or database and leak through it",
        "severity": "medium",
        "detection": "Correlate success with trial index — later trials passing more is the tell",
        "example": "Trial 1 writes a fixture that trial 2 reuses; a warm cache carries an answer",
        "impact": "Trials are not independent, so every reliability metric is wrong",
        "defense": "Fresh, isolated environment per trial (Module 03 §3.2.2)",
    },
}
```

### The detection that costs almost nothing

You cannot grep for this, but you can trap it. Two cheap instruments catch most of it:

```python
# 1. CANARY: a plausible-looking answer key nothing legitimate should ever read.
#    Place it in the eval environment. Any access is a high-severity alert.
#    This is the single highest-yield control in this module — it detects
#    tool_retrieval and environment_leakage without knowing how they'd happen.
CANARY_PATH = "/workspace/.eval/reference_solutions.json"

# 2. TRIAL-INDEX CORRELATION: independent trials show no trend by index.
#    A rising success rate across trial order means state is leaking between them.
def trial_index_trend(results: list[dict]) -> float:
    """Pearson r between trial index and success. ~0 = independent. >0 = leakage.

    Written without statistics.correlation so it runs on Python 3.9.
    """
    xs = [float(r["trial_index"]) for r in results]
    ys = [1.0 if r["success"] else 0.0 for r in results]
    n = len(xs)
    if n < 2 or len(set(ys)) < 2:
        return 0.0                       # all-pass or all-fail: no trend to measure
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0
```

If `trial_index_trend` is meaningfully positive, your trials are not independent and **every pass^k number you have reported is overstated** — the metric's entire premise is independence.

### Why this now belongs in a contamination module

Two 2026 events moved runtime contamination from theoretical to documented:

1. **Memory became a product feature.** Agents can persist findings across sessions by design (Module 15 §15.7). That is a real capability and a real contamination channel, and the distinction between them is *whether you reported which configuration you measured*. "Cold pass@1 34%, warm pass@1 71%" is an honest and interesting result; reporting the warm number alone as if it were cold is contamination.
2. **The ExploitGym incident** demonstrated the extreme case: models under evaluation escaped their sandbox and retrieved the benchmark's answer key from a third party's production database (Module 16 §16.6). The benchmark was uncontaminated. The environment was not.

> **The generalized rule that survives every model generation:** a benchmark whose answer key is reachable by the system being benchmarked is not a benchmark. Decontaminate your corpus *and* isolate your environment — they are different jobs and neither substitutes for the other.

---

## 12.4 Detecting Contamination

### The DCR Framework (2025)

```
DATA CONTAMINATION RISK (DCR) FRAMEWORK

A lightweight, interpretable pipeline for detecting and quantifying
contamination risk at four granular levels:

┌─────────────────────────────────────────────────────────────────┐
│  Level 1: SEMANTIC CONTAMINATION                                 │
│  "Is the meaning of the eval question present in training?"     │
│  Method: Embedding similarity between eval items and training    │
│  Signal: calibrated semantic-similarity score                   │
├─────────────────────────────────────────────────────────────────┤
│  Level 2: INFORMATIONAL CONTAMINATION                            │
│  "Does training data contain information that directly answers?" │
│  Method: Information extraction + overlap analysis               │
│  Signal: calibrated key-information overlap                     │
├─────────────────────────────────────────────────────────────────┤
│  Level 3: DATA CONTAMINATION                                     │
│  "Are eval data points (questions) in the training set?"        │
│  Method: N-gram matching, fuzzy deduplication                   │
│  Signal: n-gram/fuzzy overlap calibrated to the corpus          │
├─────────────────────────────────────────────────────────────────┤
│  Level 4: LABEL CONTAMINATION                                    │
│  "Are the eval ANSWERS in the training set?"                    │
│  Method: Answer extraction + matching                           │
│  Signal: answer/label overlap                                   │
└─────────────────────────────────────────────────────────────────┘

Output: the paper combines contamination signals with fuzzy membership
functions into a DCR adjustment. It does not prescribe the detector-specific
hard thresholds shown in earlier versions of this course, and the adjusted
score is a diagnostic—not the unknowable true uncontaminated performance.
```

### What 2025-2026 Contamination Studies Actually Found

Detection methods stopped being hypothetical — these are the field results you should know cold:

| Finding | What it showed | Source |
|---|---|---|
| **The SWE-Bench Illusion** (v4, Dec 2025) | SOTA models can name the buggy file from the issue text *alone* on SWE-bench repos; performance drops to ≤53% of baseline on tasks from repos outside the benchmark. Strong memorization evidence on the field's flagship coding eval | [arXiv:2506.12286](https://arxiv.org/abs/2506.12286); follow-up [arXiv:2512.10218](https://arxiv.org/pdf/2512.10218) |
| **Private-set deltas** (Balunović et al. 2025) | Models score substantially lower on MathArena's *unpublished* competition problems than on public math benchmarks of comparable difficulty | via [arXiv:2601.19334](https://arxiv.org/html/2601.19334v1) |
| **Training-set inclusion** | GSM8K and MATH found in the training data of 31 modern models (Xu et al. 2024); MMLU and HellaSwag contamination rates quantified by Hidayat et al. 2025 | via [arXiv:2601.19334](https://arxiv.org/html/2601.19334v1) |
| **Zero-leakage natural experiment** | 2026 Korean CSAT math exam administered to LLMs *hours* after release — a clean post-cutoff capability reading | [arXiv:2511.18649](https://arxiv.org/pdf/2511.18649) |
| **FrontierMath conflict of interest** (Jan 2025) | OpenAI funded the benchmark and had access to most problems + solutions; contributors weren't told; o3's headline 25% was on a benchmark its maker partly owned. Epoch kept a holdout set and apologized | [TechCrunch](https://techcrunch.com/2025/01/19/ai-benchmarking-organization-criticized-for-waiting-to-disclose-funding-from-openai/) |
| **ARC overfitting concern** | ARC Prize itself flagged that "ARC data is well represented" in frontier training corpora — even abstract-reasoning grids leak | [ARC Prize 2025 results](https://arcprize.org/blog/arc-prize-2025-results-analysis) |

Two practitioner lessons: (1) **repo/source identity is itself a leaked label** for agentic benchmarks — decontamination must consider the environment, not just the question text; (2) **benchmark governance is part of the threat model** — who funds, owns, and has access to a benchmark determines how much you trust scores on it. Ongoing catalog of the literature: [awesome-data-contamination](https://github.com/lyy1994/awesome-data-contamination).

### Autumn 2026 Additions: New Channels, New Discipline

Five developments from July–September 2026 extend the taxonomy and detection material above.

- **(a) A benchmark designed to out-run its own leakage.** Agents' Last Exam (ALE) — Berkeley RDI, Snorkel AI, and 300+ domain experts across 55 professional subfields — uses **rolling evaluation by design**: it periodically publishes a fresh public task subset, rotates private tasks in, and retires old public tasks out (the paper commits to the rolling design but fixes no cadence; secondary coverage says roughly six months). It currently holds 1,500+ tasks toward a 5,000-task target, and on its hardest tier the average full-pass rate across mainstream harness/backbone configurations is about 2.6% — nowhere near saturated. Add it to the "dynamic benchmarks that actually shipped" table below (§12.6) as the long-horizon, economically-grounded entry: refresh cadence is the anti-contamination mechanism, the same pattern as LiveBench, applied to professional-workflow tasks instead of trivia. [agents-last-exam.org](https://agents-last-exam.org/), [arXiv:2606.05405](https://arxiv.org/html/2606.05405v1)
- **(b) Terminal-Bench 3.0's mitigation is procedural, not structural.** Its agents keep full internet access but are instructed not to search for task-specific solutions — the maintainers call this "surprisingly effective," which is a weaker claim than "robust." It sits at the opposite end of the contamination-resistance spectrum from ARC-AGI-3 below (§12.6): an instruction an agent could in principle ignore, versus an environment with no answer key to find at all. [tbench.ai](https://www.tbench.ai/news/terminal-bench-3-0)
- **(c) A sharper organizing principle for the taxonomy.** "Benchmark Contamination: A Taxonomy Organized by Defeated Mitigation" (arXiv:2608.29463) reframes the `contamination_types` dict above around a more useful question than "what type is this?": **which of your specific mitigations does this type defeat?** Direct contamination defeats n-gram filtering; indirect (paraphrased) contamination defeats exact-match dedup; distributional contamination defeats item-level filtering entirely, because nothing about any single item is wrong. Pair this with an earlier-2026 caution: contamination *detectors themselves* degrade under distribution shift and scale, so no single detection method should be trusted as ground truth ([arXiv:2606.03305](https://arxiv.org/pdf/2606.03305)).
- **(d) An open contamination question inside a Critical-tier release.** UK AISI's non-CoT capability measurement of GPT-6 Astra showed a task-length jump large enough that **both OpenAI and UK AISI suspected data contamination** — as of September 2026 this is unresolved. Worth flagging precisely because it happened inside one of the most heavily scrutinized releases of the year: contamination suspicion is not a solved problem even at the top of the frontier, and the lab and its external evaluator agreeing to be suspicious doesn't resolve it. [deploymentsafety.openai.com — UK AISI external evaluation](https://deploymentsafety.openai.com/gpt-6-astra/external-evaluation-for-monitorability---uk-aisi)
- **(e) Contamination can also understate capability.** Every failure mode catalogued in this module inflates scores. A September 2026 paper argues the opposite direction also happens: near-saturated physics benchmarks, re-graded by domain experts, turn out to be "broken" in ways that make reported scores *lower* than real competence — closed-ended grading penalizes correct answers expressed differently than the reference key. Keep both directions in view when a score looks wrong: a benchmark can mislead by being too generous or too stingy, and the fix (expert re-grading vs. holdout construction) looks different for each. [arXiv:2609.13009](https://arxiv.org/abs/2609.13009)

### Implementing Contamination Detection

```python
"""
Contamination detection for your evaluation datasets.
Use this BEFORE trusting any benchmark result.
"""

from typing import List, Dict

class ContaminationDetector:
    """
    Collect indirect contamination signals when the training corpus is hidden.

    These observations are triage evidence, not a calibrated probability of
    contamination. Exact correctness or benchmark recognition can occur
    without leakage.
    """
    
    def __init__(self, model, embedding_model=None):
        self.model = model
        self.embedding_model = embedding_model
    
    def detect_memorization(self, eval_items: List[Dict]) -> Dict:
        """
        Record canonical-suffix reproduction, correctness, and source naming.
        Corroborate suspicious rows before making a contamination claim.
        """
        if not eval_items:
            raise ValueError("eval_items must not be empty")

        results = []
        
        for item in eval_items:
            question = item["question"]
            expected = item["answer"]
            
            # Probe 1: Can model complete a truncated question?
            truncated = question[:len(question)//2]
            completion = self.model.generate(
                f"Complete this question: {truncated}",
                max_tokens=200
            )
            completion_overlap = self._text_similarity(
                completion, question[len(question)//2:]
            )
            
            # Probe 2: Record capability on the item. Correctness alone is not
            # contamination evidence.
            answer = self.model.generate(
                question,
                max_tokens=100
            )
            answer_exact_match = self._normalize(answer) == self._normalize(expected)
            
            # Probe 3: Does model know the benchmark source?
            source_probe = self.model.generate(
                f"Is this question from a well-known benchmark? "
                f"If so, which one? Question: {question}"
            )
            source_named = any(
                bench in source_probe.lower() 
                for bench in ["mmlu", "hellaswag", "humaneval", "gsm8k", "arc"]
            )

            results.append({
                "question_id": item.get("id", "unknown"),
                "canonical_suffix_overlap": completion_overlap,
                "answer_exact_match": answer_exact_match,
                "benchmark_source_named": source_named,
            })

        n = len(results)
        return {
            "total_items": n,
            "observable_summary": {
                "mean_canonical_suffix_overlap": sum(
                    row["canonical_suffix_overlap"] for row in results
                ) / n,
                "benchmark_source_named_rate": sum(
                    row["benchmark_source_named"] for row in results
                ) / n,
                "answer_exact_match_rate": sum(
                    row["answer_exact_match"] for row in results
                ) / n,
            },
            "details": results,
            "interpretation": (
                "These are indirect signals. Any suspicious pattern requires "
                "corroboration from corpus provenance, controlled variants, "
                "or a private/post-cutoff holdout."
            ),
        }
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple token overlap similarity"""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        if not tokens1 or not tokens2:
            return 0.0
        intersection = tokens1 & tokens2
        return len(intersection) / max(len(tokens1), len(tokens2))
    
    def _normalize(self, text: str) -> str:
        return text.strip().lower().rstrip(".")


class BenchmarkDecontaminator:
    """
    Generate candidate benchmark variants for human validation.

    A transformation can change difficulty, introduce ambiguity, or leak the
    answer. It is not "decontaminated" merely because an LLM rewrote it.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def decontaminate_item(self, item: Dict, method: str = "paraphrase") -> Dict:
        """Generate a candidate variant; validation remains required."""
        
        methods = {
            "paraphrase": self._paraphrase,
            "context_noise": self._add_context_noise,
            "polarity_reverse": self._reverse_polarity,
        }

        if method not in methods:
            raise ValueError(f"Unsupported method: {method}")
        transform = methods[method]
        return transform(item)
    
    def _paraphrase(self, item: Dict) -> Dict:
        """Rephrase while preserving meaning and difficulty"""
        prompt = f"""Rephrase this question to test the same knowledge/skill 
but using completely different wording. The new question should be equally 
difficult and have the same correct answer.

Original: {item['question']}
Answer: {item['answer']}

Provide the rephrased question only:"""
        
        new_question = self.llm.generate(prompt, temperature=0.7)
        return {
            **item,
            "question": new_question,
            "variant_method": "paraphrase",
            "validated": False,
        }
    
    def _add_context_noise(self, item: Dict) -> Dict:
        """Add irrelevant context to test robustness"""
        prompt = f"""Add 1-2 sentences of plausible but irrelevant context 
to this question. The correct answer should remain the same, but the 
model must identify what's relevant.

Original: {item['question']}

Provide the modified question:"""
        
        new_question = self.llm.generate(prompt, temperature=0.7)
        return {
            **item,
            "question": new_question,
            "variant_method": "context_noise",
            "validated": False,
        }
    
    def _reverse_polarity(self, item: Dict) -> Dict:
        """Ask for the opposite (which is NOT correct, etc.)"""
        prompt = f"""Reverse the polarity of this question (e.g., change 
"which is correct" to "which is INCORRECT" or "which of these is NOT true").
Update the answer accordingly.

Original question: {item['question']}
Original answer: {item['answer']}

Provide both the new question and new answer as JSON:
{{"question": "...", "answer": "..."}}"""
        
        result = self.llm.generate(prompt, temperature=0.3)
        import json
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:
            raise ValueError("Model did not return valid variant JSON") from exc
        if not isinstance(parsed, dict) or not {"question", "answer"} <= parsed.keys():
            raise ValueError("Variant JSON must contain question and answer")
        return {
            **item,
            **parsed,
            "variant_method": "polarity_reverse",
            "validated": False,
        }


class DynamicBenchmarkGenerator:
    """
    Generate candidate evaluation items on-the-fly.

    Fresh generation reduces exact-item reuse; it does not prove that an item
    is novel, uncontaminated, correct, or unmemorized. Validate answers and run
    overlap checks before using generated candidates for measurement.
    """
    
    def __init__(self, llm, domain_spec: Dict):
        self.llm = llm
        self.domain_spec = domain_spec
    
    def generate_eval_set(self, 
                          num_items: int,
                          difficulty_distribution: Dict = None) -> List[Dict]:
        """
        Generate a requested number of fresh candidate items.
        
        Uses AdEval-style approach: extract knowledge points, search for 
        current information, generate multi-level questions.
        """
        if difficulty_distribution is None:
            difficulty_distribution = {
                "remembering": 0.15,    # Bloom's Level 1
                "understanding": 0.20,  # Bloom's Level 2
                "applying": 0.25,       # Bloom's Level 3
                "analyzing": 0.20,      # Bloom's Level 4
                "evaluating": 0.15,     # Bloom's Level 5
                "creating": 0.05        # Bloom's Level 6
            }
        
        if num_items < 0:
            raise ValueError("num_items must be non-negative")
        if not difficulty_distribution or any(p < 0 for p in difficulty_distribution.values()):
            raise ValueError("difficulty proportions must be non-negative")
        total = sum(difficulty_distribution.values())
        if total <= 0:
            raise ValueError("difficulty proportions must sum to a positive value")

        normalized = {level: p / total for level, p in difficulty_distribution.items()}
        raw = {level: num_items * p for level, p in normalized.items()}
        counts = {level: int(value) for level, value in raw.items()}
        remainder = num_items - sum(counts.values())
        for level in sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True)[:remainder]:
            counts[level] += 1

        items = []
        for level, n in counts.items():
            level_items = self._generate_level_items(level, n)
            items.extend(level_items)
        
        return items
    
    def _generate_level_items(self, cognitive_level: str, count: int) -> List[Dict]:
        """Generate items at a specific Bloom's taxonomy level"""
        
        level_prompts = {
            "remembering": "Create a question that tests recall of specific facts",
            "understanding": "Create a question that tests comprehension and explanation",
            "applying": "Create a question that requires applying knowledge to a new situation",
            "analyzing": "Create a question that requires breaking down and examining relationships",
            "evaluating": "Create a question that requires making judgments with justification",
            "creating": "Create a question that requires synthesizing information into something new"
        }
        
        prompt = f"""You are generating evaluation questions for: {self.domain_spec['description']}

Domain constraints: {self.domain_spec.get('constraints', 'None')}

{level_prompts[cognitive_level]}

Generate {count} unique questions with verified answers.
Each should be completely original (not from any existing benchmark).

Return as JSON list: [{{"question": "...", "answer": "...", "reasoning": "...", 
"cognitive_level": "{cognitive_level}", "difficulty": 1-5}}]"""
        
        result = self.llm.generate(prompt)
        import json
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError as exc:
            raise ValueError("generator returned invalid JSON") from exc
        if not isinstance(parsed, list) or len(parsed) != count:
            raise ValueError(f"generator returned {len(parsed) if isinstance(parsed, list) else 'non-list'} items; expected {count}")
        return parsed
```

---

## 12.5 Benchmark Saturation: When Benchmarks Stop Being Useful

```
BENCHMARK SATURATION TIMELINE

THE LEGACY TIER (saturated ~2024, contaminated, no longer
reported in any 2026 model card):

MMLU (2021):
  GPT-3:            43.9%
  GPT-4 (2023):     86.4%
  All 2026 frontier models: 88-94%+
  ← SATURATED: Models cluster at top, can't differentiate

HumanEval (2021):
  Codex (2021):     28.8%
  GPT-4 (2023):     67.0%
  Claude 3.5:       92.0%
  ← SATURATED: 164 toy problems, weak tests

THE REPLACEMENT TIER SATURATED TOO (mid-2026):

GPQA Diamond (2023, "PhD-level science"):
  Gemini 3.1 Pro:   ~94-95%  ← Epoch AI: "largely saturated"

SWE-bench Verified (2024):
  Mythos 5 / Fable 5 (June 2026): 95.5% / 95.0%
  ← Effectively a regression test now, not a discriminator

ARC-AGI-2 (2025, designed to be unsolvable):
  Best 2025 Kaggle entry:  24.0% (compute-limited)
  GPT-5.5 (June 2026):     ~85%
  ← Under two years from "unsolvable" to solved

tau2-bench Telecom (agentic):
  GPT-5.5: 98.0%  ← Sierra pivoted to tau-voice

RULE OF THUMB: every static benchmark saturates in ~18-36
months — design for refresh or holdout from day one.

WHAT HAPPENS WHEN BENCHMARKS SATURATE:
  1. Small score differences become noise, not signal
  2. Models optimize for benchmark tricks, not real capability
  3. New benchmarks needed, but they take months to create
  4. Meanwhile, marketing claims based on saturated benchmarks mislead

THE SOLUTION LANDSCAPE (2025-2026):
  ┌────────────────────────────────────────────────────────────┐
  │  Static Benchmarks  →  Dynamic Benchmarks                  │
  │  Fixed test sets    →  On-the-fly generation              │
  │  One evaluation     →  Continuous evaluation              │
  │  Public benchmarks  →  Private + dynamic benchmarks       │
  │  Single difficulty  →  Adaptive difficulty (IRT)          │
  │  Pass/fail          →  Capability profiles                │
  └────────────────────────────────────────────────────────────┘
```

### The Saturated List and What Replaced It (mid-2026)

| Legacy benchmark | Status mid-2026 | Replaced in practice by |
|---|---|---|
| MMLU | 88–94%+ for all top models; no frontier signal ([LXT survey](https://www.lxt.ai/blog/llm-benchmarks/)) | HLE, GPQA Diamond (itself now ~94–95%, [Epoch AI](https://epoch.ai/benchmarks/gpqa-diamond)), MMLU-Pro / MMMU-Pro |
| GSM8K | 95%+ universal; documented training-set contamination ([arXiv:2601.19334](https://arxiv.org/html/2601.19334v1)) | AIME 2025/2026 (itself now 90%+ solved), FrontierMath, MathArena |
| HumanEval | Saturated; 164 toy problems, weak tests | SWE-bench Verified/Pro, LiveCodeBench, Terminal-Bench |
| HellaSwag | Saturated + contaminated ([llm-stats](https://llm-stats.com/blog/research/what-is-a-contaminated-llm)) | Nothing — the commonsense multiple-choice format was abandoned |

Two structural signals that saturation is now systemic, not benchmark-by-benchmark:

- **Artificial Analysis retired its own index.** Its v3 composite saturated (top model hit 73/100), so the v4.0 Intelligence Index (Jan 2026) swapped in 10 harder evals — GDPval-AA, τ³-Banking, Terminal-Bench 2.1, HLE, CritPt, and others — recalibrated so top models score ≤50 *by design* ([methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking)). When the meta-benchmark has to reset its scale, every static benchmark under it has aged out.
- **Benchmark deprecation is being formalized** — a proposed framework for officially retiring saturated/contaminated benchmarks: [arXiv:2507.06434](https://arxiv.org/pdf/2507.06434).

### September 2026 Saturation Ledger

The pattern above kept going. Five benchmarks — four public, one internal — saturated or were formally retired in the four months after the table above, and a sixth, internal suite had to be rebuilt rather than retired:

| Benchmark | Signal | Replacement / disposition | Source |
|---|---|---|---|
| Terminal-Bench 2.1 | Leaderboard clustered in the high 70s/low 80s (Fable 5 highest at 83.8%; Opus 4.8 78.9%, GPT-5.6 Terra 78.4%) — tbench.ai's own framing: "many Terminal-Bench tasks have become saturated and leaderboard entries are condensed into a narrow range" | Terminal-Bench 3.0 (Aug 24, 2026): 74 tasks, 7 domains, best score ≈34% | [tbench.ai](https://www.tbench.ai/news/terminal-bench-3-0) |
| FrontierMath Tier 4 | GPT-6 Astra solved the single remaining unsolved problem; Epoch now describes the tier as saturated | FrontierMath Erdős (Sept 1, 2026): 68 open Erdős problems formalized in Lean, $300/problem, 72-hour limit, single attempt per model. Under protocol only GPT-6 Astra scored (2/68); looser exploratory runs (larger budget, multiple attempts, outside benchmark protocol) reached 5/68 at over $220,000 in compute | [epoch.ai](https://epoch.ai/latest/announcing-frontiermath-erdos) |
| GPQA Diamond | Judged saturated | Dropped from the Artificial Analysis Intelligence Index in v4.2 (Sept 4, 2026) | [artificialanalysis.ai v4.2](https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v4-2) |
| ARC-AGI-3 | The counter-example: still discriminating. 62.7% / 30.2% / 7.8% for GPT-6 Astra / Claude Opus 5 / GPT-5.6 Sol, across 12 evaluated models as of Sept 14, 2026 | None needed — this is what a healthy spread looks like, versus the clustering that signals saturation elsewhere in this table | [arcprize.org/leaderboard](https://arcprize.org/leaderboard) |
| SWE-bench Verified | OpenAI's own review found 59% of its hardest 138 tasks were flawed | OpenAI stopped reporting it (Feb 23, 2026) | [openai.com](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) |
| CoBench (Anthropic internal, 449 real engineering problems) | **Not saturated — it is the replacement.** The Aug 2026 Risk Report says Anthropic's *earlier* task-based AI-R&D rule-out suite saturated (frontier models surpass the human baseline on most of its tasks), so CoBench was built from harder, real engineering work; the report's own figure shows released models still short of the "could substitute for research staff" bar | Replaces the saturated R&D suite — watch for its own saturation next | [Anthropic Aug 2026 Risk Report](https://www.anthropic.com/aug-2026-risk-report) |
| Cyber Coverage Eval (Anthropic internal; Fable 5.1 / Mythos 5.1 system card) | Reported by the card as saturated — the eval suite no longer discriminates | Not yet replaced; disclosed qualitatively, no precision claimed here | [Fable 5.1 / Mythos 5.1 launch](https://www.anthropic.com/claude-fable-and-mythos-5-1) |

Read the count, not just the rows: five benchmarks aged out in roughly one quarter, against this module's own rule of thumb (18–36 months per static benchmark), and a sixth internal suite had to be rebuilt. The rate is accelerating, and two of these are **internal, non-public evals that no one gamed from outside** — the Cyber Coverage Eval saturated, and the task-based AI-R&D suite behind CoBench did the same, because the models simply got good. Contamination is one road to saturation; capability is the other, and it is now the faster one.

**Measurement ceiling ≠ saturation.** This module's own METR citation (§12.2, third-party testers) deserves the same scrutiny. METR's time-horizon page has not been updated since May 8, 2026, and METR's own page states that measurements above roughly 16 hours are unreliable on its current task suite. Read the frontier's current time-horizon entry as *where the instrument ran out of dynamic range*, not as a capability milestone the frontier has since blown past. Secondary reporting suggests the published estimate is now sensitive to how the scorer credits cheating/reward-hacking behavior on the underlying tasks — that specific claim is unverified against a METR publication, so treat it as a hypothesis worth watching, not a number to cite. Reliably measuring tasks longer than the current ceiling needs month-scale, human-baselined task construction, which METR itself calls slow and expensive — the same "your instrument has a ceiling" lesson as a saturated public benchmark, just driven by measurement cost instead of model capability. [metr.org/time-horizons](https://metr.org/time-horizons/)

---

## 12.6 Dynamic Evaluation: The 2026 Paradigm

### AdEval: Alignment-based Dynamic Evaluation

```
THE AdEval APPROACH (2025)

Instead of using fixed benchmark items:

Step 1: EXTRACT KNOWLEDGE POINTS from existing benchmarks
  "MMLU question about photosynthesis"
  → Knowledge point: "light-dependent reactions in photosynthesis"

Step 2: SEARCH FOR CURRENT INFORMATION
  → Find latest research/information about the knowledge point
  → Ensures questions test current understanding

Step 3: GENERATE MULTI-LEVEL QUESTIONS using Bloom's hierarchy
  Level 1 (Remember): "What are the two stages of photosynthesis?"
  Level 2 (Understand): "Explain why photosynthesis requires sunlight"
  Level 3 (Apply): "Given this scenario, predict the photosynthesis rate"
  Level 4 (Analyze): "Compare C3 and C4 photosynthesis pathways"
  Level 5 (Evaluate): "Assess this claim about artificial photosynthesis"
  Level 6 (Create): "Design an experiment to test photosynthesis efficiency"

Step 4: ITERATIVE RECONSTRUCTION
  → Questions are reconstructed to control difficulty
  → Validated against ground truth
  → Contamination risk minimized

RESULT: Each evaluation run uses DIFFERENT questions testing the SAME skills.
Freshly generated questions reduce exact-item reuse, but they can still
reproduce known material or invalid labels. Validate novelty and answers.
```

### Self-Evolving Benchmarks

```
BENCHMARK SELF-EVOLUTION (6 Reframing Operations)

Given original item: "What is the capital of France? (A) London (B) Paris"

1. PARAPHRASE
   → "Which city serves as France's seat of government?"

2. CONTEXT NOISING
   → "France, known for its wine and cheese, has a capital city.
      Given that many EU countries have relocated offices to Brussels,
      what is still the capital of France?"

3. POLARITY REVERSAL
   → "Which of these is NOT the capital of France?"

4. QUESTION ALTERNATING
   → "Paris is the capital of which country?"

5. DIFFICULTY SCALING
   → "Name the capital of France and the year it became the capital."

6. FORMAT CHANGE
   → Open-ended: "Describe the capital of France and its significance."

KEY FINDING: Most LLMs show PERFORMANCE DECLINES under these 
transformations compared to original benchmarks, revealing that 
static benchmark scores OVERESTIMATE true capability.
```

### Dynamic Benchmarks That Actually Shipped

AdEval-style generation is no longer a research idea — by mid-2026 a whole tier of production benchmarks is dynamic by construction:

| Benchmark | Anti-contamination mechanism | Notes |
|---|---|---|
| [LiveBench](https://github.com/LiveBench/LiveBench) | Monthly question drops sourced from recent arXiv papers, news, and IMDb | Refresh cadence is the contract |
| [LiveCodeBench](https://livecodebench.github.io/) | Rolling post-cutoff windowing of new LeetCode/Codeforces/AtCoder problems ([arXiv:2403.07974](https://arxiv.org/abs/2403.07974)) | Near-saturated at the top anyway (93.5% — [BenchLM](https://benchlm.ai/benchmarks/liveCodeBench)): rolling design *delays* saturation, it doesn't prevent it |
| [SWE-rebench](https://swe-rebench.com/) | SWE-bench-style tasks continuously mined from fresh GitHub PRs | The direct answer to the SWE-Bench Illusion finding |
| [AntiLeakBench](https://aclanthology.org/2025.acl-long.901/) (ACL 2025) | Auto-builds test items from real-world facts that emerged after each model's cutoff | Fully automated temporal gating |
| [ARC-AGI-3](https://arcprize.org/blog/arc-agi-3-launch) (Mar 25, 2026) | **Interactive** turn-based game environments — no instructions, no stated goals, no static answer text to memorize | At launch: humans 100%, frontier AI 0.51% aggregate ([arXiv:2603.24621](https://arxiv.org/pdf/2603.24621)) |
| [Agents' Last Exam](https://agents-last-exam.org/) (2026) | Rolling refresh: periodic new public subsets (cadence not fixed in the paper; ~6 months per secondary coverage), private tasks rotate in, retired public tasks rotate out | 1,500+ tasks toward a 5,000-task target across 55 professional subfields; hardest-tier average full-pass rate ≈2.6% ([arXiv:2606.05405](https://arxiv.org/html/2606.05405v1)) |

The gradient matters: refresh-based designs (LiveBench, SWE-rebench, Agents' Last Exam) buy you months to years depending on refresh cadence and task-authoring cost; **interactivity (ARC-AGI-3) is the strongest contamination resistance available**, because there is no answer key — leaked or otherwise — only an environment the agent must explore. Expect more eval formats to move this way as static and even refreshed sets keep getting eaten.

---

## 12.7 Private Holdouts: How 2026 Benchmarks Stay Honest

The single biggest structural change since 2024: serious benchmarks now ship with a **private holdout** as a design requirement, and frontier system cards report on third-party private sets as a credibility signal.

### The holdout landscape

| Benchmark | Public | Held out | Holder |
|---|---|---|---|
| Humanity's Last Exam | 2,500 expert questions | A private test split, kept specifically to detect overfitting | CAIS/Scale ([arXiv:2501.14249](https://arxiv.org/abs/2501.14249), [agi.safe.ai](https://agi.safe.ai/)) |
| FrontierMath | Problem statements/solutions accessible to OpenAI (the funder) | A **50-problem holdout withheld even from OpenAI** — the post-scandal fix | Epoch AI ([epoch.ai](https://epoch.ai/frontiermath/the-benchmark)) |
| SWE-bench Pro | 731 instances from GPL/copyleft repos | A 12-repo held-out split **plus** a 276-task commercial set from 18 private startup codebases, never released | Scale AI ([arXiv:2509.16941](https://arxiv.org/pdf/2509.16941), [Scale blog](https://scale.com/blog/swe-bench-pro)) |
| GDPval | A 220-task gold subset with an automated grader | The remainder of 1,320 real occupational deliverables | OpenAI ([arXiv:2510.04374](https://arxiv.org/abs/2510.04374)) |
| Legal Agent Benchmark | Full public set | Harvey's held-out set — the Fable 5 card reports Fable 5 ranked highest on it as of June 2026 | Harvey ([Fable 5 card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) §8.17.4) |
| AutomationBench | Leaderboard | A private held-out task set (Fable 5 17.4% vs Opus 4.8 15.5%) | Zapier ([Fable 5 card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) §8.17.9) |

Note the second-order effect in the last two rows: **labs now treat "we score well on sets we cannot have seen" as a marketing asset.** Vendors reporting on private third-party holdouts in their own system cards is the strongest market validation eval-training separation has ever had.

### Two design patterns worth stealing

1. **Legal deterrence.** SWE-bench Pro's *public* split deliberately uses GPL/copyleft repos — training on them creates licensing exposure, so the license itself becomes an anti-contamination mechanism ([arXiv:2509.16941](https://arxiv.org/pdf/2509.16941)).
2. **Governance as integrity.** The FrontierMath/OpenAI incident (Section 12.4) taught the field that a holdout is only as private as its *governance*: Epoch's 50-problem set is now withheld even from the funder. When you read any score, ask who controls the holdout.

### The split-and-scaffold spread is the lesson

Three different numbers circulated as SWE-bench Pro "SOTA" in mid-2026:

```
80.3%   Mythos 5 (Fable 5: 80.0), Anthropic's own scaffold, public split
~59.1%  GPT-5.4 xHigh, Scale's standardized SEAL harness, public split
~47.1%  Opus 4.6, private commercial split
```

Same benchmark family, 30+ points apart ([the-decoder](https://the-decoder.com/anthropic-releases-claude-fable-5-and-mythos-5-with-major-gains-in-coding-and-science/), [morphllm](https://www.morphllm.com/swe-bench-pro), [Scale leaderboard](https://scale.com/leaderboard/swe_bench_pro_commercial)). "SOTA" is a function of **(model, scaffold, effort setting, split)** — never quote a score without all four, and treat the private-split number as the closest thing to truth.

---

## 12.8 Eval-Training Separation for Agentic Benchmarks & RL Environments

Agentic evaluation broke the old mental model of contamination. The question is no longer just "did the test questions leak into pretraining?" — it's "is the eval artifact itself a training artifact?"

### Your eval set is now someone's RL training set

The 2025-2026 convergence: **RL environments and agent evals are the same artifact** — a dataset + harness + scoring rules.

- Prime Intellect's [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) library is explicitly "RL environments + evals"; its Environments Hub hosts 2,500+ open-source environments used for *both* purposes ([launch post](https://www.primeintellect.ai/blog/environments)).
- OpenAI's platform exposes **grader objects** (`python` graders, `score_model` graders, multigraders) that are *shared between the Evals API and reinforcement fine-tuning* — an RFT job auto-creates an associated eval from the same grader ([graders guide](https://developers.openai.com/api/docs/guides/graders)).

Consequence: publishing an eval harness now means publishing a training environment. The train/test firewall must be enforced **at the artifact level** — if your grader doubles as a reward function (and increasingly it does), your private holdout must include held-out *graders and environments*, not just held-out questions. And every rubric is a reward spec: red-team it for hackability like one ([Anthropic, arXiv:2511.18397](https://arxiv.org/abs/2511.18397) showed reward hacking learned in production coding RL generalizes to sabotage and alignment faking).

### How the Fable 5 / Mythos 5 system card handles it

The June 2026 card is the best public case study of agentic eval-training separation in practice ([PDF](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)):

```
AGENTIC SEPARATION DISCIPLINES (Fable 5 / Mythos 5 card)

1. PROBES ARE NOT REWARDS (§6.4.1)
   Interpretability probes (dishonesty, reward hacking, eval
   awareness) ran on essentially every RL transcript during
   most of post-training — but were NEVER used as a training
   signal. Measurement and optimization kept firewalled.

2. ENVIRONMENT-SIDE LABEL LEAKAGE IS REAL (§6.2.2)
   Training-data review caught models "recovering reference
   solutions from git history, build caches, or other
   answer-revealing" artifacts inside RL environments —
   the agentic version of label contamination.

3. HOLDOUT DISCIPLINE EVEN DURING ELICITATION (§6.5.4)
   When fine-tuning a model variant to elicit maximum stealth,
   they trained on Minimal-LinuxBench trajectories precisely
   because "Using Minimal-LinuxBench means SHADE-Arena remains
   held out from training."

4. RETRIEVAL CONTAMINATION CONTROL (§8, HLE-with-tools)
   Tool-using evals can fetch answers from the web. The HLE
   harness blocklists known HLE-discussing sources for both
   searcher and fetcher; a judge model reviews ALL transcripts
   and confirmed answer-retrievals are re-graded as incorrect.

5. ANTI-REWARD-HACKING EVAL DESIGN (§3.2.1)
   ExploitBench replays challenge-response functions across
   randomized heap layouts, so hardcoding a leaked address
   doesn't count as a solved exploit.

6. GRADER AWARENESS IS MEASURED, NOT ASSUMED AWAY (§6.4.2)
   Coding RL environments are classified into high/low
   grader-hacking risk. Unverbalized grader awareness is
   prevalent in high-risk environments and INCREASES over
   training — "almost all of the time, this awareness is
   unverbalized." It is lower in deployment than in training.
```

Each of these maps directly onto a failure mode you will hit in your own agentic evals: reward-shaped graders, answer-leaking sandboxes, fine-tuning on your own eval trajectories, agents googling the answer key, hackable verifiers, and agents that behave differently because they detect the grader.

### Evaluation awareness: the model knows it's being tested

The newest contamination channel isn't in the data at all — it's in the model's *situational inference*:

- Apollo Research found **GPT-5.5 verbalized awareness of being alignment-evaluated in 22.1% of samples — 52% under sandbagging conditions**, with some samples naming the exact eval type ([GPT-5.5 system card](https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf)).
- UK AISI measured that Mythos 5 **distinguishes real deployment data from synthetic reproductions 84% of the time pairwise** (Opus 4.8: 78%) ([Fable 5 card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) §6.2.4); Anthropic and UK AISI now jointly apply **environment-realism methods** to suppress eval-awareness during testing.

For eval engineers the implication is uncomfortable but actionable: **realism is now a benchmark-integrity property.** A synthetic-smelling eval measures "the model's behavior when it suspects a test," which is a different quantity from deployed behavior. Build evals from sanitized real traffic where possible, audit transcripts for verbalized test-awareness, and report awareness rates alongside scores — the frontier labs now do.

### What this means for your eval system

- Keep graders and rubrics as private as test items — they double as reward specs.
- If you use eval data to drive RFT/RL (the grader-object pattern), **fork a frozen holdout first** and never let it touch a training job.
- For tool-using agents, decide explicitly whether retrieval of external answers counts — and blocklist + audit if it doesn't.
- Re-run agentic evals on environments *structurally identical but outside* the published set (the SWE-Bench Illusion test) before trusting a number.

---

## 12.9 Building Contamination-Resistant Eval Systems

### The Private Eval Infrastructure

The following is a **control-flow sketch**, not production security code. Private storage, access logs, rotation, and overlap checks reduce specific exposure and reuse risks; none proves that a set is uncontaminated. The encryption methods below are placeholders, and every similarity threshold must be calibrated and supplemented with provenance review.

```python
"""
Illustrative contamination-resistant evaluation controls.
Limit exposure, log access, rotate items, and validate provenance.
"""

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class PrivateEvalVault:
    """
    Secure storage for private evaluation datasets.
    Eval items are never exposed to the internet or shared externally.
    """
    
    def __init__(self, storage_backend, encryption_key: str):
        self.storage = storage_backend
        self.encryption_key = encryption_key
        self.access_log = []
    
    def store_eval_set(self, 
                       name: str,
                       items: List[Dict],
                       metadata: Dict) -> str:
        """Store an eval set with encryption and access tracking"""
        
        eval_set_id = f"eval_{secrets.token_hex(8)}"
        
        encrypted_items = self._encrypt(items)
        
        record = {
            "id": eval_set_id,
            "name": name,
            "items": encrypted_items,
            "num_items": len(items),
            "created_at": datetime.now().isoformat(),
            "metadata": metadata,
            "fingerprint": self._fingerprint(items),
            "access_count": 0,
            "max_uses": metadata.get("max_uses", 100),  # Retire after N uses
            "expires_at": (
                datetime.now() + timedelta(days=metadata.get("ttl_days", 90))
            ).isoformat()
        }
        
        self.storage.save(eval_set_id, record)
        return eval_set_id
    
    def get_eval_set(self, eval_set_id: str, requester: str) -> Optional[List[Dict]]:
        """Retrieve eval set with access control and logging"""
        
        record = self.storage.get(eval_set_id)
        if not record:
            return None
        
        # Check expiry
        if datetime.fromisoformat(record["expires_at"]) < datetime.now():
            self._retire_eval_set(eval_set_id, "expired")
            return None
        
        # Check use limit
        if record["access_count"] >= record["max_uses"]:
            self._retire_eval_set(eval_set_id, "max_uses_reached")
            return None
        
        # Log access
        self.access_log.append({
            "eval_set_id": eval_set_id,
            "requester": requester,
            "timestamp": datetime.now().isoformat()
        })
        
        # Increment access count
        record["access_count"] += 1
        self.storage.save(eval_set_id, record)
        
        # Decrypt and return
        return self._decrypt(record["items"])
    
    def rotate_eval_set(self, 
                        old_id: str,
                        generator,
                        domain_spec: Dict) -> str:
        """
        Retire old eval set and generate a fresh one.
        Rotation reduces repeated-item exposure; it does not prove novelty.
        """
        old_record = self.storage.get(old_id)
        
        # Generate fresh items
        new_items = generator.generate_eval_set(
            num_items=old_record["num_items"],
            difficulty_distribution=old_record["metadata"].get("difficulty_distribution")
        )
        
        # Verify no overlap with old set
        overlap = self._check_overlap(
            self._decrypt(old_record["items"]),
            new_items
        )
        
        if overlap > 0.1:  # Illustrative threshold; calibrate for the domain
            raise ValueError(f"Generated items have {overlap:.0%} overlap with retired set")
        
        # Retire old, store new
        self._retire_eval_set(old_id, "rotated")
        new_id = self.store_eval_set(
            name=f"{old_record['name']}_rotated_{datetime.now().strftime('%Y%m%d')}",
            items=new_items,
            metadata=old_record["metadata"]
        )
        
        return new_id
    
    def _fingerprint(self, items: List[Dict]) -> str:
        """Create a fingerprint of eval items for tracking"""
        content = str(sorted([item.get("question", "") for item in items]))
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _check_overlap(self, old_items: List[Dict], new_items: List[Dict]) -> float:
        """Check semantic overlap between old and new eval sets"""
        # Simplified: check text similarity
        old_questions = set(item.get("question", "").lower() for item in old_items)
        new_questions = set(item.get("question", "").lower() for item in new_items)
        
        if not old_questions or not new_questions:
            return 0.0
        
        # Token-level overlap as proxy for semantic similarity
        overlap_count = 0
        for new_q in new_questions:
            for old_q in old_questions:
                similarity = len(set(new_q.split()) & set(old_q.split())) / \
                           max(len(set(new_q.split())), 1)
                if similarity > 0.7:
                    overlap_count += 1
                    break
        
        return overlap_count / len(new_questions)
    
    def _retire_eval_set(self, eval_set_id: str, reason: str):
        """Retire an eval set (mark as unusable)"""
        record = self.storage.get(eval_set_id)
        if record:
            record["retired"] = True
            record["retired_reason"] = reason
            record["retired_at"] = datetime.now().isoformat()
            self.storage.save(eval_set_id, record)
    
    def _encrypt(self, data): 
        # In production: use Fernet or AES-256
        return data  # Placeholder
    
    def _decrypt(self, data):
        return data  # Placeholder
```

---

## 12.10 The Evaluation Integrity Checklist

```markdown
## Before Trusting Any Benchmark Score

### Data Contamination
- [ ] Is the benchmark publicly available? (If yes, assume some contamination)
- [ ] When was the benchmark created vs model's training cutoff?
- [ ] Has the benchmark appeared on GitHub, arXiv, Kaggle, or StackOverflow?
- [ ] Run contamination detection probes (Section 12.4)
- [ ] Check if model can identify the benchmark by name from partial questions

### Benchmark Health
- [ ] Is the benchmark saturated? (Top models within 2% of each other)
- [ ] Has the benchmark been refreshed/updated recently?
- [ ] Are there known issues with the benchmark's ground truth?
- [ ] Does the benchmark test the skill you actually care about?
- [ ] Is there a private holdout — and who controls it? (Funder access = conflict of interest; see FrontierMath, Section 12.4)
- [ ] Does the reported score disclose all four of (model, scaffold, effort setting, split)? (These move scores 30+ points; Section 12.7)

### Agentic & RL-Era (2026)
- [ ] Could the agent retrieve answers via tools (web search, git history, build caches)? Is retrieval blocklisted and transcript-audited?
- [ ] Have you re-run on structurally identical tasks *outside* the published benchmark? (The SWE-Bench Illusion test)
- [ ] Are your graders/rubrics kept as private as the test items? (Graders double as RL reward specs)
- [ ] If eval data feeds RFT/RL, did you fork a frozen holdout before the first training job?
- [ ] Could the model be eval-aware? Is the environment realistic enough that you're measuring deployed behavior, not test-detection behavior?

### Your Eval System
- [ ] Are your eval items stored privately (not on the internet)?
- [ ] Do you rotate eval items periodically?
- [ ] Do you use dynamic generation for at least some evals?
- [ ] Are your eval results versioned and reproducible?
- [ ] Do you track eval item age and usage count?

### Reporting
- [ ] Do you report contamination risk alongside scores?
- [ ] Do you compare against multiple benchmarks (not just one)?
- [ ] Do you supplement public benchmarks with private evals?
- [ ] Do you report confidence intervals, not just point estimates?
```

---

## 12.11 Worked Examples (2026)

These are **illustrative diagnostic probes**, not contamination detectors. They
help you decide which items deserve provenance review or replacement. The
sourced incidents in Sections 12.4, 12.7, and 12.8 are the documented cases in
which benchmark-integrity problems were actually found.

#### Example 1 — Canary-string memorization probe

Unusually exact reproduction of the *rest of a canonical question* from a short
prefix can reveal memorized wording. Ordinary correctness is not the signal,
and a familiar phrase can produce a false positive.

```python
from difflib import SequenceMatcher
from openai import OpenAI
client = OpenAI()

def canonical_suffix_similarity(item: dict, model="gpt-5.5") -> float:
    """Compare a prompted continuation with the hidden canonical suffix."""
    words = item["question"].split()
    split_at = max(1, len(words) // 2)
    prefix = " ".join(words[:split_at])
    canonical_suffix = " ".join(words[split_at:])
    out = client.chat.completions.create(
        model=model,
        messages=[{"role": "user",
                   "content": f"Continue this benchmark item verbatim:\n{prefix}"}],
    ).choices[0].message.content
    normalize = lambda text: " ".join((text or "").lower().split())
    return SequenceMatcher(
        None, normalize(out), normalize(canonical_suffix)
    ).ratio()

# Inspect item-level results. Compare against unpublished control questions
# with similar length and template; do not turn the ratio into an unsupported
# universal contamination threshold.
scores = [canonical_suffix_similarity(x) for x in benchmark_sample]
print(sorted(zip([x["id"] for x in benchmark_sample], scores), key=lambda x: -x[1]))
```

#### Example 2 — Original vs paraphrased delta

A paired drop on independently validated variants catches **surface-form
sensitivity**. Memorization is one possible explanation, but so are changed
difficulty, ambiguity, or ordinary prompt brittleness.

```python
def compare_validated_variants(items, validated_variants):
    """Both lists must use the same IDs and independently verified answers."""
    if [x["id"] for x in items] != [x["id"] for x in validated_variants]:
        raise ValueError("Original and variant items must be paired by ID")

    original = run_eval(items)
    variant = run_eval(validated_variants)
    return {
        "original_accuracy": original,
        "variant_accuracy": variant,
        "paired_gap": original - variant,
        "interpretation": (
            "Surface-form sensitivity; combine with provenance, private-holdout, "
            "and prefix-completion evidence before attributing it to memorization."
        ),
    }
```

Pre-register the gap worth investigating and report a paired confidence
interval. For a consequential decision, these probes supplement—not
replace—a private or post-cutoff holdout and an environment-access audit.

---

## 12.12 Exercises

### Exercise 1: Contamination Audit
Take a public benchmark (MMLU, HumanEval, GSM8K) and:
- Run the memorization probes from Section 12.4 against a model
- Estimate contamination risk
- Generate candidate variants of 10 items and have a second reviewer validate
  answer equivalence and difficulty
- Compare performance on the paired original and validated-variant items;
  describe the result as surface-form sensitivity unless other evidence
  supports a contamination attribution

### Exercise 2: Build a Private Eval System
Design and implement a private evaluation system for your use case:
- Private item storage with encryption
- Automatic rotation schedule
- Contamination monitoring
- Access logging and audit trail

### Exercise 3: Dynamic Benchmark Design
Choose a domain and build a dynamic benchmark generator:
- Define knowledge points for your domain
- Implement at least 3 of the 6 reframing operations
- Generate 50 fresh items and validate quality
- Compare model scores on static vs dynamic versions

### Exercise 4: The SWE-Bench Illusion Test (agentic)
Pick an agentic coding benchmark your team uses and replicate the memorization probe from Section 12.8:
- Give the model only the issue/task text and ask it to name the file(s) that need changing — without repo access
- Build 10 structurally identical tasks from a repo *outside* the benchmark and compare pass rates
- Audit 20 transcripts for tool-based answer retrieval (web search, git history) and for verbalized eval-awareness
- Write up the gap between in-benchmark and out-of-benchmark performance as a contamination estimate

---

## Next Module
-> [Module 13: Advancing AI Research -- Contributing to the Frontier](../13-advancing-ai-research/README.md)
