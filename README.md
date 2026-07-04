# Eval Engineering: Complete Study Guide

> **The Art and Science of Evaluating AI/LLM Systems at Scale**
>
> A comprehensive guide for software engineers and aspiring AI researchers learning to build production-grade evaluation systems for LLMs, RAG systems, and AI agents. Updated **June 2026** with a deep pass on *how the latest frontier models were actually evaluated* — drawn from the Claude Fable 5 / Mythos 5, Opus 4.8, and Sonnet 4.6 system cards, OpenAI's GPT-5.x Preparedness work, Google's Frontier Safety Framework, plus practice from the UK AI Security Institute (AISI), METR, and the broader AI safety community.
>
> **Companion reading:** Hamel Husain's ["Your AI Product Needs Evals"](https://hamel.dev/blog/posts/evals/) and Eugene Yan's ["Evaluating LLM-Evaluators"](https://eugeneyan.com/writing/llm-evaluators/) are the two best practitioner essays in the field — read them alongside this guide.

## Table of Contents

| Module | Topic | Time | Difficulty |
|--------|-------|------|------------|
| [00-prerequisites](./00-prerequisites/) | ML/AI Basics for Software Engineers | 1 hour | Beginner |
| [01-fundamentals](./01-fundamentals/) | Core Concepts & Terminology | 2 hours | Beginner |
| [02-evaluation-methods](./02-evaluation-methods/) | All Evaluation Approaches + Psychometric & Dynamic Evals | 4 hours | Intermediate |
| [03-pipeline-architecture](./03-pipeline-architecture/) | Building Robust Pipelines | 3 hours | Intermediate |
| [04-cold-start](./04-cold-start/) | Bootstrapping Evaluations | 2 hours | Advanced |
| [05-scaling](./05-scaling/) | Cost & Performance Optimization | 3 hours | Advanced |
| [06-feedback-loops](./06-feedback-loops/) | Continuous Improvement | 3 hours | Advanced |
| [07-cicd-integration](./07-cicd-integration/) | Production Integration | 4 hours | Expert |
| [08-case-studies](./08-case-studies/) | Real-World Examples | 2 hours | Intermediate |
| [09-langchain-examples](./09-langchain-examples/) | Python/LangChain Implementation | 3 hours | Intermediate |
| [10-advanced-topics](./10-advanced-topics/) | Enterprise Patterns, Alignment Faking, EDDOps | 4 hours | Expert |
| **[11-how-frontier-models-are-trained](./11-how-frontier-models-are-trained/)** | **The Training Pipeline: Pretraining → SFT → RLHF/RLVR → Constitutional AI, and how the latest models were evaluated** | **4 hours** | **Expert** |
| **[12-eval-training-separation](./12-eval-training-separation/)** | **Benchmark Integrity, Contamination, Dynamic Evals** | **3 hours** | **Expert** |
| **[13-advancing-ai-research](./13-advancing-ai-research/)** | **Contributing to the Frontier: Alignment, Safety, Research Skills** | **3 hours** | **Expert** |

**Total Study Time: ~41 hours**

---

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

**Eval Engineering** is the discipline of designing, building, and maintaining systems that measure the quality, safety, and performance of AI systems. It's the bridge between "it seems to work" and "we can prove it works."

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

### Path 1: Complete Course (5 weeks)
```
Week 1: Modules 00-02 (Foundations + Eval Methods)
Week 2: Modules 03-04 (Pipelines & Cold Start)
Week 3: Modules 05-06 (Scaling & Feedback)
Week 4: Modules 07-10 (Production & Advanced)
Week 5: Modules 11-13 (Training, Separation & Research)
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
- Cost optimization (93% savings strategies)
- Scaling to millions of evaluations
- CI/CD integration (GitHub Actions)
- Feedback loops and active learning
- **Evaluation-Driven Development (EDD/EDDOps)** -- NEW
- **Self-evolving evaluation pipelines** -- NEW

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
- **Anthropic API** (Claude Fable 5 `claude-fable-5`, Opus 4.8 `claude-opus-4-8`, Sonnet 4.6 `claude-sonnet-4-6`, Haiku 4.5 `claude-haiku-4-5`) and **OpenAI API** (GPT-5.5, GPT-5.4-mini reasoning models) — frontier models reason by default; "effort"/"thinking budget" is a tunable knob that changes both score and cost
- **Pydantic** for data validation and structured outputs
- **Redis/Celery** for distributed processing
- **GitHub Actions** for CI/CD
- **NumPy/SciPy/scikit-learn** for statistical analysis and calibration

### Modern eval tooling landscape (2026)

| Category | Tools |
|----------|-------|
| **Eval frameworks** | [Inspect AI](https://inspect.aisi.org.uk/) (UK AISI — agent-first, sandboxed, 200+ benchmarks), [OpenAI Evals](https://github.com/openai/evals), [Promptfoo](https://www.promptfoo.dev/), [DeepEval](https://github.com/confident-ai/deepeval) |
| **Hosted eval + tracing** | [LangSmith](https://docs.langchain.com/langsmith/evaluation), [Braintrust](https://www.braintrust.dev/), [Arize Phoenix](https://phoenix.arize.com/), [Weights & Biases Weave](https://wandb.ai/site/weave), [Langfuse](https://langfuse.com/), [Helicone](https://www.helicone.ai/) |
| **RAG-specific** | [RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/), [DeepEval RAG metrics](https://github.com/confident-ai/deepeval) |
| **Tracing standards** | [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), [OpenLLMetry / Traceloop](https://github.com/traceloop/openllmetry) |
| **Safety / red-team** | [Inspect Evals safety suite](https://inspect.aisi.org.uk/evals/), [Garak](https://github.com/NVIDIA/garak), [PyRIT](https://github.com/Azure/PyRIT), Anthropic's [SHADE-Arena](https://alignment.anthropic.com/2025/strengthening-red-teams/) |
| **Public leaderboards** | [Arena (formerly LMArena/LMSYS)](https://lmarena.ai/) — note "The Leaderboard Illusion" critique; [Epoch AI Benchmarking Hub + Capabilities Index (ECI)](https://epoch.ai/benchmarks/eci); [Artificial Analysis Intelligence Index v4](https://artificialanalysis.ai/methodology/intelligence-benchmarking); [SEAL / SEAL Showdown (Scale)](https://scale.com/leaderboard); [Terminal-Bench](https://www.tbench.ai/leaderboard/terminal-bench/2.0), [LiveBench](https://livebench.ai/), [SWE-bench Verified/Pro](https://www.swebench.com/) |

---

## What You'll Build

By the end of this guide, you'll be able to:

1. **Design** comprehensive evaluation frameworks for any AI use case
2. **Implement** automated evaluators (rule-based, LLM, hybrid, psychometric)
3. **Scale** to handle millions of evaluations cost-effectively
4. **Integrate** evals into CI/CD pipelines with EDDOps practices
5. **Analyze** results and drive improvements
6. **Operate** production evaluation systems
7. **Understand** how frontier models are trained and what that means for evals
8. **Detect** benchmark contamination and build contamination-proof eval systems
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

# Pattern 3: Rubric — one ISOLATED judge call per criterion, weighted
# aggregation. Never one omnibus call. (module 02 §2.3.4)
results = {c.id: judge_criterion(c, response) for c in rubric.criteria}
score = sum(c.weight for c in rubric.criteria if results[c.id].met)

# Pattern 4: Cascade — cheap screen first, strong judge only when uncertain
s = cheap_judge.score(response)            # score once, reuse the result
final = s if s.confidence > 0.8 else strong_judge.score(response)
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
eval-engineering/
├── README.md                              # This file
├── 00-prerequisites/                      # ML/AI basics
├── 01-fundamentals/                       # Core concepts
├── 02-evaluation-methods/                 # All techniques + psychometric & dynamic
├── 03-pipeline-architecture/              # Building pipelines
├── 04-cold-start/                         # Bootstrapping
├── 05-scaling/                            # Optimization
├── 06-feedback-loops/                     # Continuous improvement
├── 07-cicd-integration/                   # Production CI/CD
├── 08-case-studies/                       # Real examples
├── 09-langchain-examples/                 # Python code
├── 10-advanced-topics/                    # Enterprise patterns + alignment faking
├── 11-how-frontier-models-are-trained/    # Training pipeline deep dive (NEW)
├── 12-eval-training-separation/           # Contamination & dynamic evals (NEW)
└── 13-advancing-ai-research/              # Research frontier & career path (NEW)
```

---

## 🤝 Contributing

This is a living document. Suggestions welcome!

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
- [Claude Opus 4.8 announcement (May 2026)](https://www.anthropic.com/news/claude-opus-4-8) and [Claude Sonnet 4.6 announcement (Feb 2026)](https://www.anthropic.com/news/claude-sonnet-4-6) — each links to its system card
- [Anthropic Responsible Scaling Policy](https://www.anthropic.com/responsible-scaling-policy) and [OpenAI Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf)
- [Anthropic FMTI Transparency Report (December 2025)](https://crfm.stanford.edu/fmti/December-2025/company-reports/Anthropic_FinalReport_FMTI2025.html)

### Frontier model evaluation (2025–2026)
- [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) — Anthropic (Jan 2026); the de-facto agent-evals playbook (task/trial/transcript/outcome, pass@k vs pass^k)
- [GDPval](https://openai.com/index/gdpval/) and [HealthBench](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf) — OpenAI; rubric-based and economically-grounded evals
- [METR time horizons](https://metr.org/time-horizons/) — the "AGI progress" task-length metric used for autonomy assessment
- [Petri: open-source automated auditing](https://www.anthropic.com/research/petri-open-source-auditing) (now maintained by Meridian Labs) and [Inspect AI / ControlArena](https://inspect.aisi.org.uk/) — UK AISI
- [Natural Emergent Misalignment from Reward Hacking](https://arxiv.org/abs/2511.18397) — Anthropic (Nov 2025); [Chain-of-thought monitoring](https://openai.com/index/chain-of-thought-monitoring/) — OpenAI
- [The SWE-Bench Illusion](https://arxiv.org/abs/2506.12286) and ["environments are the new datasets"](https://www.primeintellect.ai/blog/environments) — Prime Intellect

---

**The field of evaluation engineering is the most critical skill for ensuring AI goes well for humanity. Master it.**