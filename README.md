# Eval Engineering: Study Guide

> **The Art and Science of Evaluating AI/LLM Systems at Scale**
>
> A practical guide for software engineers and aspiring AI researchers learning
> to build decision-oriented evaluation systems for LLMs, RAG systems, and AI
> agents. Updated **September 2026** for the Claude Fable 5.1 / GPT-6 Astra
> release wave — benchmark saturation and leaderboard churn, and new
> eval-science on chain-of-thought monitorability and judge reliability —
> with current API behavior, recent model reports, and evaluation practice
> from primary research and evaluation groups.
>
> **Companion reading:** Hamel Husain's ["Your AI Product Needs Evals"](https://hamel.dev/blog/posts/evals/) and Eugene Yan's ["Evaluating LLM-Evaluators"](https://eugeneyan.com/writing/llm-evaluators/) are useful practitioner introductions to read alongside this guide.

## Table of Contents

| Module | Topic | Time | Difficulty |
|--------|-------|------|------------|
| [00-prerequisites](./00-prerequisites/) | ML/AI Basics for Software Engineers | 1 hour | Beginner |
| [01-fundamentals](./01-fundamentals/) | Core Concepts & Terminology | 2 hours | Beginner |
| [02-evaluation-methods](./02-evaluation-methods/) | Core Evaluation Approaches + Psychometric & Dynamic Evals | 4 hours | Intermediate |
| [03-pipeline-architecture](./03-pipeline-architecture/) | Building Robust Pipelines | 3 hours | Intermediate |
| [04-cold-start](./04-cold-start/) | Bootstrapping Evaluations | 2 hours | Advanced |
| [05-scaling](./05-scaling/) | Cost & Performance Optimization | 3 hours | Advanced |
| [06-feedback-loops](./06-feedback-loops/) | Continuous Improvement | 3 hours | Advanced |
| [07-cicd-integration](./07-cicd-integration/) | Production Integration | 4 hours | Expert |
| [08-case-studies](./08-case-studies/) | Worked Composites + Sourced Case Studies | 2 hours | Intermediate |
| [09-langchain-examples](./09-langchain-examples/) | Python/LangChain Implementation | 3 hours | Intermediate |
| [10-advanced-topics](./10-advanced-topics/) | Enterprise Patterns, Alignment Faking, EDDOps | 4 hours | Expert |
| **[11-how-frontier-models-are-trained](./11-how-frontier-models-are-trained/)** | **The Training Pipeline: Pretraining → SFT → RLHF/RLVR → Constitutional AI, and how the latest models were evaluated** | **4 hours** | **Expert** |
| **[12-eval-training-separation](./12-eval-training-separation/)** | **Benchmark Integrity, Contamination, Dynamic Evals** | **3 hours** | **Expert** |
| **[13-advancing-ai-research](./13-advancing-ai-research/)** | **Contributing to the Frontier: Alignment, Safety, Research Skills** | **3 hours** | **Expert** |
| **[14-loop-engineering](./14-loop-engineering/)** | **Loop Engineering: Evaluating Self-Correcting Systems** | **3 hours** | **Advanced** |
| **[15-opus5-eval-techniques](./15-opus5-eval-techniques/)** | **Evaluating in the Opus 5 Era: API-Native Eval Techniques** | **2 hours** | **Advanced** |
| **[16-frontier-architectures-and-research-thinking](./16-frontier-architectures-and-research-thinking/)** | **Reading the Frontier: DeepSeek-V4, Kimi K3, GPT-5.6 — and How to Think Like a Researcher** | **2 hours** | **All levels** |

**Total Study Time: ~48 hours**

---

## How to Read the Course

Every chapter now uses the same learner contract:

- **Covers** — the behavior or risk the eval actually observes.
- **Catches** — the failure pattern that can become visible.
- **Decision enabled** — what you can reasonably change after seeing the
  result.
- **Evidence status** — a sourced documented case, an explicitly labeled
  teaching example, or a control-flow/code sketch.

An eval result is an observation, not automatically a cause. A prompt gap does
not prove memorization, a monitoring-context gap does not prove scheming, and a
judge score does not prove quality until the judge is calibrated. The chapters
state the next discriminating check when attribution needs more evidence.

### Coverage inventory: where each decision is taught

| Eval family / learner decision | Primary module(s) |
|---|---|
| Deterministic checks, labels, schema, code tests | [00](./00-prerequisites/), [02](./02-evaluation-methods/), [09](./09-langchain-examples/) |
| Single judges, pairwise comparison, panels, calibration | [02](./02-evaluation-methods/), [07](./07-cicd-integration/), [15](./15-opus5-eval-techniques/) |
| Human review, rubric design, disagreement, grader audits | [02](./02-evaluation-methods/), [04](./04-cold-start/), [10](./10-advanced-topics/) |
| Repeated reliability, intervals, MDE, regression gates | [01](./01-fundamentals/), [07](./07-cicd-integration/), [15](./15-opus5-eval-techniques/) |
| Retrieval, groundedness, answer quality | [02](./02-evaluation-methods/), [09](./09-langchain-examples/) |
| Agent outcomes, trajectories, tools, sandboxes, long horizons | [02](./02-evaluation-methods/), [03](./03-pipeline-architecture/), [08](./08-case-studies/), [14](./14-loop-engineering/) |
| Safety, red teams, control/sabotage, automated audits | [10](./10-advanced-topics/), [13](./13-advancing-ai-research/), [16](./16-frontier-architectures-and-research-thinking/) |
| Feedback, online measurement, random audits, experiments | [06](./06-feedback-loops/), [13](./13-advancing-ai-research/) |
| Contamination, private/post-cutoff holdouts, dynamic evals, memory leakage | [12](./12-eval-training-separation/), [15](./15-opus5-eval-techniques/) |
| Retry/verifier value, marginal yield, regressions, loop cost | [14](./14-loop-engineering/) |
| Current effort, refusal, structured-output, caching, batch, migration behavior | [15](./15-opus5-eval-techniques/) |
| Multimodal generation without one reference answer | [08 Case 10](./08-case-studies/#case-study-10-uber-eats-multimodal-image-agent--evaluating-generation-with-no-ground-truth) |
| Architecture/vendor claims, reproductions, containment, research inference | [13](./13-advancing-ai-research/), [16](./16-frontier-architectures-and-research-thinking/) |

The inventory is intentionally organized by **decision**, not by fashionable
metric name. Modules 14–16 exist because loops, changing API semantics, and
frontier evidence each create a distinct decision. The remaining eval families
fit the existing chapters, so adding another chapter would duplicate material
rather than close a learner gap.

---

## What's New in the September 2026 Edition

This revision tracks the Claude Fable 5.1 / GPT-6 Astra release wave
(Sept 1–3, 2026), the benchmark suites that saturated or got reworked in
response, and new eval-science on judge validity and chain-of-thought
monitorability:

- **[Module 05](./05-scaling/) & [Module 15](./15-opus5-eval-techniques/) (pricing and API changes)**: Module 05 adds Claude Fable 5.1 pricing (`claude-fable-5-1`, $10/$50, with cache reads at a 0.025× rate instead of the standard 0.1×) and confirms Sonnet 5's $2/$10 pricing is now permanent — the scheduled Sept 1 increase to $3/$15 was cancelled. Module 15 covers five API changes that break existing harnesses: Fable 5.1/Mythos 5.1 reject forced tool calls (`tool_choice: "any"`/`"tool"` now return HTTP 400 — migrate to strict tool use or structured outputs), per-message `effort` (beta), stricter thinking-block replay rules on newer accounts, Messages API compaction (beta), and the now-GA computer/browser toolsets.
- **[Module 08](./08-case-studies/) (Case Study 7 addendum)**: a new "what a point release re-evaluates" addendum covers the Fable 5.1 / Mythos 5.1 system card — which safety thresholds were re-confirmed versus newly assessed-but-not-triggered, the changed third-party evaluator roster, and a checklist for reading a point-release card.
- **[Module 10](./10-advanced-topics/) (alignment evals and evaluators)**: adds GPT-6 Astra as the first model rated Critical for cybersecurity under a published framework, the UK AISI external evaluation of Astra's monitorability, and updates to the red-team and third-party evaluator tables (Meridian Labs' Petri 3.0, CAISI, UK AISI).
- **[Module 11](./11-how-frontier-models-are-trained/) (training pipeline)**: covers OpenAI's Aug 18, 2026 pause of its largest planned RL run on GPT-6 Astra pending safety mitigations, DeepMind's Frontier Safety Framework v3.1, and CAISI's independent evaluation of DeepSeek V4 Pro as a self-report-vs-independent-measurement case study.
- **[Module 12](./12-eval-training-separation/) (saturation and leaderboard integrity)**: a September 2026 saturation ledger (Terminal-Bench 3.0, FrontierMath Erdős, ARC-AGI-3, GPQA Diamond dropped from the Artificial Analysis index) plus a new section on the Artificial Analysis index's v4.2/v4.3 overhaul after it and Epoch's ECI ranked GPT-6 Astra very differently.
- **[Module 13](./13-advancing-ai-research/) (research frontier)**: Anthropic's RSP v3.4, Petri's donation to Meridian Labs, and a 12-paper "Autumn 2026" reading list on eval-awareness, sandbagging, and chain-of-thought monitor reliability.
- **[Module 16](./16-frontier-architectures-and-research-thinking/) (reading the frontier)**: a new GPT-6 Astra reading case, and "Reading Benchmark Claims Without Being Had" extended from four questions to seven — run-to-run noise, saturation vs. instrument ceilings, and self-reported vs. independent measurement.
- **[Module 02](./02-evaluation-methods/) (judge and agent-eval science)**: two 2026 findings that change how you validate a judge (rubric-artifact leakage; reliability is not validity), plus Senior SWE-Bench's layered grading, MemoryArena, and Agents' Last Exam as agent-eval references.
- **[Module 03](./03-pipeline-architecture/), [Module 06](./06-feedback-loops/), [Module 07](./07-cicd-integration/) & [Module 09](./09-langchain-examples/) (tooling sweep)**: the OpenAI Evals platform's sunset (read-only Oct 31, 2026; shut down Nov 30), Google's Agent + Model Evaluations GA, Langfuse's stable evaluator API, Arize Phoenix's session-level PII evaluator, and Claude Code's `claude plugin eval` are threaded through the pipeline, feedback-loop, CI/CD, and LangChain-example modules, each of which also flags the Fable 5.1 forced-tool-call break where it applies to an existing example.

## What's New in the August 2026 Edition

This revision adds the two things practitioners kept asking for — **how do you evaluate a system that evaluates itself**, and **what changed at the API layer** — plus a production case study that exercises both.

- **Module 14 (new) — [Loop Engineering](./14-loop-engineering/)**: when a system retries or self-corrects, evaluate the loop rather than only an isolated call. The chapter covers three nested loops (turn / task / outer), six task-loop components, **verifier asymmetry** (how a weak gate can make retries worse, with the arithmetic), metrics beyond pass@k alone (marginal yield, regression rate, oscillation rate, cost per accepted output, loop tax), stop conditions, gate architecture, a failure catalog with log signatures, and Goodhart guardrails for an outer improvement loop.
- **Module 15 (new) — [Evaluating in the Opus 5 Era](./15-opus5-eval-techniques/)**: current Claude sampling/API changes, intervals and MDE, effort as an eval axis, schema-constrained judges, **refusal-aware measurement** (unmeasured for conditional capability; a product outcome when availability or refusal policy is the target), measured cost optimization, context management, and **memory stores as a contamination vector**.
- **Case Study 10 (new) — [Uber Eats multimodal image agent](./08-case-studies/#case-study-10-uber-eats-multimodal-image-agent--evaluating-generation-with-no-ground-truth)**: a production generative pipeline with no ground truth, from a talk by Soumya Gupta and Jai Chopra (Uber). Covers the routing gate that **censors your own dataset**, faithfulness as a veto rather than a weighted score, Swiss-cheese guardrails and what correlated layers cost, flat-JSON observability, and closing the loop to conversion rate without Goodharting the golden set.
- **Module 16 (new) — [Reading the Frontier](./16-frontier-architectures-and-research-thinking/)**: a deliberate bridge from evals to evidence reading. It separates paper-reported architecture results, vendor comparisons, and downstream hypotheses; teaches a 30-minute triage protocol; and uses the documented **ExploitGym incident** to connect capability elicitation, containment, and benchmark integrity.
- **Plain-English entry points across Modules 00–16**, so a PM, designer, or ops lead can get each chapter's argument before choosing whether to read the code.
- **Claude Opus 5** (`claude-opus-5`) threaded through the code and the tech stack: thinking on by default, the five-level effort ladder, 512-token prompt-cache minimum, task budgets, `fallbacks: "default"`.

## What's New in the June 2026 Edition

This revision is anchored on the question *"how were the newest frontier models actually evaluated?"* and threads the answer through every module:

- **Module 11 (reasoning-RL-era rewrite)**: the training pipeline updated for **RLVR (RL from verifiable rewards)**, RL environments, distillation, and training-time reward-hacking monitoring — plus an end-to-end account of how the latest generation (Claude Fable 5/Mythos 5, Opus 4.8, GPT-5.x, Gemini 3.x) was trained *and* evaluated, with corrected model dates.
- **Module 08 (new case studies)**: a full anatomy of the **Claude Fable 5 / Mythos 5 system card** (capability + RSP/ASL-3 + alignment + Petri auditing), plus **GDPval** and **METR time-horizon** studies as worked cases.
- **Module 02**: rubric-based grading (the **HealthBench/GDPval** pattern), reasoning-model judges, isolated per-dimension judge calls, and agent **trajectory/outcome** evals with the Anthropic task/trial/transcript/outcome vocabulary.
- **Module 10**: the newest alignment-eval practice from the system cards — **evaluation awareness**, **grader-/reward-hacking measurement**, interpretability-assisted audits (NLAs), automated behavioral auditing (**Petri**), and third-party testing (**METR / UK AISI / Meridian / Gray Swan**).
- **Module 12**: 2025-26 contamination findings (the **SWE-Bench Illusion**, MathArena private deltas), private holdouts in current cards, the saturated-benchmark list, and contamination-resistant agentic benchmarks.
- **Modules 01 / 03 / 06 / 07 / 09**: **pass@k vs pass^k** reliability, **OpenTelemetry GenAI** tracing + sandboxed agent harnesses (Inspect AI / Harbor), the **trace → dataset flywheel**, agentic-coding-era CI gates, and current model IDs throughout the runnable code.
- **Craft sections (July 2026)** — taught, not just cited: **§4.2b "From Blank Page to First 30 Cases"** (failure-taxonomy generation, grid sampling, fault-injection ground truth, anchored partial-credit rubrics, error-clustering — one dbt scenario end-to-end), **§1.4b** worked error-analysis walkthrough (open coding → axial coding → what to build), and **§2.3.6** the TPR/TNR arithmetic that a single "accuracy" number hides.
- **Latest research** from Anthropic (reward-hacking → misalignment generalization, Petri, Constitutional AI), OpenAI (CoT monitoring, GDPval, Preparedness Framework v2), UK AISI (Inspect / ControlArena), Prime Intellect ("environments are the new datasets"), and ICLR/NeurIPS 2025-2026.

---

## What is Eval Engineering?

**Eval Engineering** is the discipline of designing, building, and maintaining systems that measure the quality, safety, and performance of AI systems. It turns "it seems to work" into explicit, uncertainty-aware evidence for a decision; it does not prove universal correctness.

```
┌─────────────────────────────────────────────────────────────────┐
│                    THE EVAL ENGINEERING LOOP                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐ │
│    │  Build  │────▶│  Eval   │────▶│ Analyze │────▶│ Improve │ │
│    │  Model  │     │ System  │     │ Results │     │  Model  │ │
│    └─────────┘     └─────────┘     └─────────┘     └─────────┘ │
│         ▲                                               │        │
│         └───────────────────────────────────────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📖 Read It as One File

Prefer one document over 17 module folders? [`book.md`](./book.md) is the whole
course — this README plus every module's README, concatenated in order — and
[`book.html`](./book.html) is the same content as a single self-contained,
print-friendly page (renders client-side via [marked.js](https://marked.js.org/),
no build step). Good for offline reading, printing, or feeding the whole
course to another tool in one shot. Regenerate either after a course update
with:

```bash
python3 build_book.py . book
```

Both files track the per-module READMEs, which remain the source of truth —
if they ever disagree, the module READMEs win; re-run the script above.

---

## 🚀 Quick Start

### For Complete Beginners (New to ML/AI)
```bash
# Start here
open 00-prerequisites/README.md   # ML/AI basics for software engineers
open 01-fundamentals/README.md    # Core eval concepts
```

### For Experienced Engineers (Know ML, New to Evals)
```bash
# Skip prerequisites, start with fundamentals
open 01-fundamentals/README.md
open 02-evaluation-methods/README.md
open 09-langchain-examples/README.md  # Hands-on code
```

### For Immediate Practical Use
```bash
# Jump to implementation
open 09-langchain-examples/README.md  # LangChain/OpenAI code
open 07-cicd-integration/README.md    # CI/CD setup
```

---

## Learning Paths

### Path 1: Complete Course (6 weeks)
```
Week 1: Modules 00-02 (Foundations + Eval Methods)
Week 2: Modules 03-04 (Pipelines & Cold Start)
Week 3: Modules 05-06 (Scaling & Feedback)
Week 4: Modules 07-10 (Production & Advanced)
Week 5: Modules 11-13 (Training, Separation & Research)
Week 6: Modules 14-16 (Loop Engineering, Opus 5-Era Techniques & Reading the Frontier)
```

### Path 5: Agentic Systems Track (2 weeks) -- NEW
```
Week 1: Modules 01 (§1.3b pass@k), 14 (Loop Engineering), 08 (Case Studies 5, 9, 10)
Week 2: Modules 15 (Opus 5-era techniques), 03 (§3.3.4 trace flywheel), 06 (Feedback loops)
```

### Path 2: Practical Focus (2 weeks)
```
Week 1: Modules 00, 01, 09 (Basics + Code)
Week 2: Modules 07, 08 (CI/CD + Case Studies)
```

### Path 3: Enterprise Focus (3 weeks)
```
Week 1: Modules 01, 03, 05 (Foundations + Scale)
Week 2: Modules 07, 10 (CI/CD + Enterprise Patterns)
Week 3: Modules 11, 12 (Training Pipeline + Benchmark Integrity)
```

### Path 4: AI Researcher Track (3 weeks) -- NEW
```
Week 1: Modules 01, 02, 11 (Fundamentals + Training Pipeline)
Week 2: Modules 10, 12 (Advanced Topics + Contamination)
Week 3: Module 13 (Research Frontier + Project Planning)
```

---

## Key Topics Covered

### Evaluation Methods
- Rule-based evaluation (regex, exact match, structured output)
- LLM-as-Judge (single judge, multi-judge panels, pairwise comparison)
- Human evaluation (crowdsourcing, expert review, annotation quality)
- Rubric-based evaluation with detailed scoring
- **Psychometric evaluation (IRT, Bloom's taxonomy, adaptive testing)** -- NEW
- **Dynamic benchmark generation and reframing** -- NEW

### System Types
- Simple chatbots
- RAG (Retrieval-Augmented Generation)
- AI Agents
- Multi-step pipelines
- Code generation
- Content moderation

### Production Concerns
- Cold start bootstrapping
- Cost optimization with measured batching, caching, and routing tradeoffs
- Scaling to millions of evaluations
- CI/CD integration (GitHub Actions)
- Feedback loops and active learning
- **Evaluation-Driven Development (EDD/EDDOps)** -- NEW
- **Self-evolving evaluation pipelines** -- NEW
- **Loop metrics: marginal yield, regression rate, oscillation, cost per accepted output** -- NEW
- **Statistical eval design: intervals, MDE, and sizing your suite** -- NEW
- **Refusal-aware estimands and coverage reporting (conditional capability versus end-to-end availability)** -- NEW
- **Memory and agent state as contamination vectors** -- NEW

### Enterprise Patterns
- Constitutional AI evaluation
- Red teaming & adversarial testing
- Distributed evaluation systems
- Evaluator calibration
- Governance and compliance
- **Alignment faking detection** -- NEW
- **Modular attack scaffolds (SHADE-Arena)** -- NEW

### Frontier AI Research -- NEW
- **How models are trained (Pretraining -> SFT -> RLHF -> Constitutional AI)**
- **Eval-training data separation and decontamination**
- **Benchmark contamination detection (DCR framework)**
- **Dynamic benchmarks and adaptive evaluation**
- **Alignment faking and strategic model behavior**
- **Path from eval engineer to AI researcher**

---

## Tech Stack

This guide uses:
- **Python 3.11+**
- **LangChain / LangGraph** for LLM and agent orchestration
- **Anthropic API** (Claude Fable 5.1 `claude-fable-5-1`, Fable 5, Opus 5, Sonnet 5, and Haiku 4.5 — Sonnet 5's $2/$10 pricing is standard, not introductory) and **OpenAI API** (GPT-6 Astra `gpt-6-astra`, GPT-5.6 Sol/Terra/Luna, plus pinned smaller models). Reasoning defaults, effort levels, and sampling-parameter support differ by provider and model; record the exact configuration and check current docs (see Module 15)
- **Pydantic** for data validation and structured outputs
- **Redis/Celery** for distributed processing
- **GitHub Actions** for CI/CD
- **NumPy/SciPy/scikit-learn** for statistical analysis and calibration

### Modern eval tooling landscape (2026)

| Category | Tools |
|----------|-------|
| **Eval frameworks** | [Inspect AI](https://inspect.aisi.org.uk/) (UK AISI — agent-first, sandboxed, 200+ benchmarks), [OpenAI Evals](https://github.com/openai/evals) (hosted platform [sunsetting](https://developers.openai.com/api/docs/deprecations): read-only Oct 31, 2026, shut down Nov 30, 2026 — OpenAI points early-stage users to Datasets, durable path is Promptfoo), [Promptfoo](https://www.promptfoo.dev/), [DeepEval](https://github.com/confident-ai/deepeval), [Google Agent + Model Evaluations](https://developers.googleblog.com/agent-and-model-evaluations-in-gemini-enterprise-agent-platform-are-now-ga/) (GA Jul 31, 2026 — Gemini Enterprise Agent Platform), Claude Code [`claude plugin eval`](https://code.claude.com/docs/en/plugin-evals) (Sept 2026 — proposes cases/graders, runs each 3x with and 3x without the plugin) |
| **Hosted eval + tracing** | [LangSmith](https://docs.langchain.com/langsmith/evaluation), [Braintrust](https://www.braintrust.dev/), [Arize Phoenix](https://phoenix.arize.com/) (evals 3.6.0, Aug 28 2026, adds a session-level PII evaluator), [Weights & Biases Weave](https://wandb.ai/site/weave), [Langfuse](https://langfuse.com/) ([stable evaluator API](https://langfuse.com/changelog/2026-08-27-stable-evaluator-api), Aug 2026 — old endpoints migrate by Nov 16, 2026), [Helicone](https://www.helicone.ai/) |
| **RAG-specific** | [RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/), [DeepEval RAG metrics](https://github.com/confident-ai/deepeval) |
| **Tracing standards** | [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), [OpenLLMetry / Traceloop](https://github.com/traceloop/openllmetry) |
| **Safety / red-team** | [Inspect Evals safety suite](https://inspect.aisi.org.uk/evals/), [Garak](https://github.com/NVIDIA/garak), [PyRIT](https://github.com/Azure/PyRIT), Anthropic's [SHADE-Arena](https://alignment.anthropic.com/2025/strengthening-red-teams/) |
| **Public leaderboards** | [Arena (formerly LMArena/LMSYS)](https://lmarena.ai/) — note "The Leaderboard Illusion" critique; [Epoch AI Benchmarking Hub + Capabilities Index (ECI)](https://epoch.ai/benchmarks/eci); [Artificial Analysis Intelligence Index](https://artificialanalysis.ai/methodology/intelligence-benchmarking) (recomposed twice in Sept 2026 — v4.2 then v4.3 — after AA and Epoch's ECI ranked GPT-6 Astra very differently; see [Module 12](./12-eval-training-separation/)); [SEAL / SEAL Showdown (Scale)](https://scale.com/leaderboard); [Terminal-Bench 3.0](https://www.tbench.ai/news/terminal-bench-3-0) (Aug 24, 2026 — replaces the saturated 2.1 suite), [LiveBench](https://livebench.ai/), [SWE-bench Verified/Pro](https://www.swebench.com/), [ARC-AGI-3 leaderboard](https://arcprize.org/leaderboard), [FrontierMath Erdős](https://epoch.ai/latest/announcing-frontiermath-erdos), [Agents' Last Exam](https://agents-last-exam.org/) |

---

## What You'll Build

By the end of this guide, you'll be able to:

1. **Design** evaluation frameworks tied to explicit AI-system decisions
2. **Implement** automated evaluators (rule-based, LLM, hybrid, psychometric)
3. **Scale** to handle millions of evaluations cost-effectively
4. **Integrate** evals into CI/CD pipelines with EDDOps practices
5. **Analyze** results and drive improvements
6. **Operate** production evaluation systems
7. **Understand** how frontier models are trained and what that means for evals
8. **Collect evidence about and reduce** benchmark-contamination risk with private holdouts, overlap checks, and isolated environments
9. **Contribute** to AI safety research and alignment science

---

## Who This Is For

| Background | Recommended Path |
|------------|------------------|
| Software Engineer (new to ML) | Start at Module 00 |
| ML Engineer (new to LLMs) | Start at Module 01 |
| Startup building AI product | Focus on Modules 01-04, 07-08 |
| Enterprise AI team | Focus on Modules 03, 05-07, 10-12 |
| Engineering Manager | Skim all, deep dive 03, 07, 10 |
| **Aspiring AI Researcher** | **Path 4: Modules 01-02, 10-13** |
| **AI Safety Engineer** | **Modules 10-13, then 02-03** |
| **Shipping an agent or self-correcting pipeline** | **Path 5: Modules 14-16, then 08 Case Studies 5/9/10** |

---

## 🔗 Quick Reference

### Key Patterns

```python
# Pattern 1: Binary judge + evidence (NOT "rate 1-5" — Likert scales give
# spurious precision and poor human alignment; see module 01 §1.4d)
verdict = llm.invoke(
    f"Did the response resolve the user's request? "
    f"Answer PASS or FAIL, then quote the evidence.\n\n{response}")

# Pattern 2: Pairwise comparison — always swap positions (position bias)
r1 = judge(f"Which is better?\nA: {a}\nB: {b}")
r2 = judge(f"Which is better?\nA: {b}\nB: {a}")   # order swapped
winner = a if (r1 == "A" and r2 == "B") else tie_or_rejudge(r1, r2)

# Pattern 3: Rubric — isolated judge calls are a robust default when criteria
# can contaminate one another. Batch only after validating equivalent quality.
# Keep non-tradeable safety/policy/grounding constraints as vetoes, outside a
# weighted score for preferences that are genuinely allowed to trade off.
results = {c.id: judge_criterion(c, response) for c in rubric.criteria}
score = sum(c.weight for c in rubric.criteria if results[c.id].met)

# Pattern 4: Cascade — cheap screen first, strong judge only when uncertain
s = cheap_judge.score(response)            # score once, reuse the result
final = s if calibrated_route.accept(s) else strong_judge.score(response)
```

### Key Metrics

| Metric | Use When |
|--------|----------|
| Accuracy | Factual correctness matters |
| Helpfulness | User satisfaction is key |
| Safety | Content could cause harm |
| Groundedness | RAG systems (hallucination prevention) |
| Task Completion | Agent/workflow success |

---

## Repository Structure

```
llm-evals-engineering-course/
├── README.md                              # This file
├── COURSE_AUDIT.md                        # Chapter-by-chapter QA log and rubric
├── test_course_examples.py                # Regression tests for copyable code
├── book.md / book.html                    # Whole course as one file (see below)
├── build_book.py                          # Generates book.md / book.html
├── 00-prerequisites/                      # ML/AI basics
├── 01-fundamentals/                       # Core concepts
├── 02-evaluation-methods/                 # All techniques + psychometric & dynamic
├── 03-pipeline-architecture/              # Building pipelines
├── 04-cold-start/                         # Bootstrapping
├── 05-scaling/                            # Optimization
├── 06-feedback-loops/                     # Continuous improvement
├── 07-cicd-integration/                   # Production CI/CD
├── 08-case-studies/                       # Worked composites + sourced cases
├── 09-langchain-examples/                 # Python code
├── 10-advanced-topics/                    # Enterprise patterns + alignment faking
├── 11-how-frontier-models-are-trained/    # Training pipeline deep dive
├── 12-eval-training-separation/           # Contamination & dynamic evals
├── 13-advancing-ai-research/              # Research frontier & career path
├── 14-loop-engineering/                   # Self-correcting loops & their metrics
├── 15-opus5-eval-techniques/              # Opus 5-era harness techniques
└── 16-frontier-architectures-and-research-thinking/   # Reading papers, research habits
```

---

## Quality Control

Every module's copyable Python is extracted straight from its Markdown source and regression-tested, so the README stays the executable source of truth rather than drifting from it:

```bash
pip install pytest
pytest test_course_examples.py -v
```

Each revision pass is also logged against a fixed rubric — human entry point, eval semantics, teaching evidence, no ornamental examples, current coverage, executable integrity — in [`COURSE_AUDIT.md`](./COURSE_AUDIT.md), so you can see exactly what was checked and when, not just that it "looks done."

---

## 🤝 Contributing

This is a living document, refreshed roughly every 6–8 weeks against the frontier (see the "What's New" sections above for the cadence). Suggestions and corrections welcome via issues or PRs — if you're flagging a factual claim, a source link makes it actionable immediately.

---

## Additional Resources

### Foundational Papers
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) -- Anthropic
- [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) -- Anthropic
- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) -- Stanford
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) -- Kaplan et al.
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) -- Liu et al.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) -- Zheng et al.
- [Replacing Judges with Juries (Panel of LLM evaluators / PoLL)](https://arxiv.org/abs/2404.18796) -- Verga et al.

### Safety & Alignment Research
- [Alignment Faking in Large Language Models](https://www.anthropic.com/research/alignment-faking) -- Anthropic & Redwood Research (Dec 2024, [arXiv:2412.14093](https://arxiv.org/abs/2412.14093))
- [Sabotage Evaluations for Frontier Models](https://www.anthropic.com/research/sabotage-evaluations) -- Anthropic
- [Many-Shot Jailbreaking](https://www.anthropic.com/research/many-shot-jailbreaking) -- Anthropic
- [Strengthening Red Teams: Modular Scaffold for Control Evaluations](https://alignment.anthropic.com/2025/strengthening-red-teams/) -- Anthropic (2025)
- [Claude's Constitution](https://www.anthropic.com/constitution) -- Anthropic
- [Apollo Research: Scheming evaluations](https://www.apolloresearch.ai/research)

### Evaluation Research (2025-2026)
- [AdEval: Alignment-based Dynamic Evaluation](https://arxiv.org/abs/2501.13983) -- 2025
- [Benchmarking LLMs Under Data Contamination](https://aclanthology.org/2025.emnlp-main.511/) -- EMNLP 2025
- [The Art of Scaling RL Compute for LLMs](https://openreview.net/forum?id=FMjeC9Msws) -- ICLR 2026
- [Front-Loading Reasoning](https://research.nvidia.com/labs/adlr/Synergy/) -- NVIDIA 2025

### Tools & Frameworks
- [OpenAI's Eval Framework](https://github.com/openai/evals)
- [LangSmith Evaluation Docs](https://docs.langchain.com/langsmith/evaluation)
- [Braintrust AI Evaluation](https://www.braintrust.dev/docs/evaluate)
- [Inspect AI (UK AISI)](https://inspect.aisi.org.uk/)
- [RAGAS](https://docs.ragas.io/) — RAG and agent evaluation metrics
- [Arize Phoenix](https://phoenix.arize.com/) — open-source observability
- [Weights & Biases Weave](https://wandb.ai/site/weave)
- [Anthropic's Transparency Hub](https://www.anthropic.com/transparency)

### Practitioner Essays (must-read)
- Hamel Husain — [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)
- Eugene Yan — [Evaluating LLM-Evaluators (LLM-as-Judge)](https://eugeneyan.com/writing/llm-evaluators/)
- Shreya Shankar et al. — [Who Validates the Validators? (EvalGen)](https://arxiv.org/abs/2404.12272)

### System Cards & Transparency Reports (primary sources for this edition)
- [Claude Fable 5 / Mythos 5 System Card (June 2026, 319 pp)](https://www.anthropic.com/claude-fable-5-mythos-5-system-card) — the flagship case study in Module 08; capability + RSP/ASL-3 + ~120 pp alignment assessment
- [Claude Fable 5.1 / Mythos 5.1 launch & system card (Sept 1, 2026, 212 pp)](https://www.anthropic.com/claude-fable-and-mythos-5-1) — same underlying weights as Fable 5/Mythos 5 under two safeguard configurations; addendum in Module 08
- [Claude Opus 4.8 announcement (May 2026)](https://www.anthropic.com/news/claude-opus-4-8) and [Claude Sonnet 4.6 announcement (Feb 2026)](https://www.anthropic.com/news/claude-sonnet-4-6) — each links to its system card
- [GPT-6 Astra Deployment Safety Hub](https://deploymentsafety.openai.com/gpt-6-astra), [Path to Astra](https://openai.com/index/path-to-astra/), and [Pacing model development on cyber capabilities](https://openai.com/index/pacing-model-development-cyber-capabilities/) — OpenAI; first model rated Critical for cybersecurity under a published framework
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy) (v3.4 effective Jul 8, 2026 — see the version history on that page; the separate [updates hub](https://www.anthropic.com/responsible-scaling-policy/updates) tracks policy changes) and [OpenAI Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf)
- [Anthropic August 2026 Risk Report](https://www.anthropic.com/aug-2026-risk-report) and [DeepMind Frontier Safety Framework v3.1](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) (Apr 17, 2026 — Tracked Capability Levels, Security Level 2+)
- [Anthropic FMTI Transparency Report (December 2025)](https://crfm.stanford.edu/fmti/December-2025/company-reports/Anthropic_FinalReport_FMTI2025.html)

### Frontier model evaluation (2025–2026)
- [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic (Jan 2026); the de-facto agent-evals playbook (task/trial/transcript/outcome, pass@k vs pass^k)
- [GDPval](https://openai.com/index/gdpval/) and [HealthBench](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf) — OpenAI; rubric-based and economically-grounded evals
- [METR time horizons](https://metr.org/time-horizons/) — the "AGI progress" task-length metric used for autonomy assessment
- [Petri: open-source automated auditing](https://www.anthropic.com/research/petri-open-source-auditing) (now maintained by Meridian Labs) and [Inspect AI / ControlArena](https://inspect.aisi.org.uk/) — UK AISI
- [Petri 3.0](https://meridianlabs.ai/blog/posts/introducing-petri-3/) — Meridian Labs; auditor/target split, "Dish" realism harness, "Bloom" single-behavior suite generator, following Anthropic's [donation of Petri](https://www.anthropic.com/research/donating-open-source-petri) (May 7, 2026)
- [Natural Emergent Misalignment from Reward Hacking](https://arxiv.org/abs/2511.18397) — Anthropic (Nov 2025); [Chain-of-thought monitoring](https://openai.com/index/chain-of-thought-monitoring/) — OpenAI
- [The SWE-Bench Illusion](https://arxiv.org/abs/2506.12286) and ["environments are the new datasets"](https://www.primeintellect.ai/blog/environments) — Prime Intellect
- [Terminal-Bench 3.0](https://www.tbench.ai/news/terminal-bench-3-0) (Aug 24, 2026), [FrontierMath Erdős](https://epoch.ai/latest/announcing-frontiermath-erdos) (Sept 1, 2026, 68 Lean-formalized open Erdős problems), and [Agents' Last Exam](https://agents-last-exam.org/) (rolling refresh with private-task rotation) — the September 2026 saturation-resistant benchmark generation, detailed in Module 12

---

**Evaluation engineering is one practical discipline for making AI-system decisions more evidence-based. Treat the evaluator with the same skepticism you apply to the system it measures.**
