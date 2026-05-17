# Eval Engineering: Complete Study Guide

> **The Art and Science of Evaluating AI/LLM Systems at Scale**
>
> A comprehensive guide for software engineers and aspiring AI researchers learning to build production-grade evaluation systems for LLMs, RAG systems, and AI agents. Updated **May 2026** with the latest research and practice from Anthropic, OpenAI, the UK AI Security Institute (AISI), and the broader AI safety community.
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
| **[11-how-frontier-models-are-trained](./11-how-frontier-models-are-trained/)** | **The Complete Training Pipeline: Pretraining to RLHF to Constitutional AI** | **4 hours** | **Expert** |
| **[12-eval-training-separation](./12-eval-training-separation/)** | **Benchmark Integrity, Contamination, Dynamic Evals** | **3 hours** | **Expert** |
| **[13-advancing-ai-research](./13-advancing-ai-research/)** | **Contributing to the Frontier: Alignment, Safety, Research Skills** | **3 hours** | **Expert** |

**Total Study Time: ~41 hours**

---

## What's New in the 2026 Edition

This guide has been substantially updated with:

- **Module 11**: Deep dive into how frontier models (Claude Opus 4.5/Sonnet 4.5, GPT-4o/o-series, Gemini 2.x) are actually trained -- pretraining, SFT, reward modeling, RLHF, Constitutional AI, and safety fine-tuning
- **Module 12**: Eval-training separation, benchmark contamination detection, dynamic benchmarks, and the DCR framework
- **Module 13**: Research frontier -- alignment faking detection, open problems, how to contribute to AI safety research, and a complete path from eval engineer to AI researcher
- **Module 10 update**: Alignment faking evaluation, sandbagging and sabotage evals, SHADE-Arena, EDDOps (Evaluation-Driven Development and Operations), self-evolving eval systems
- **Module 02 update**: Modern LLM-as-judge practice (pairwise vs. direct, panel of judges, calibration, bias controls), agent trajectory evals, reasoning-trace / CoT-faithfulness evals, and dynamic benchmark generation
- **Module 06 update**: Online (production) evals with async LLM-judge scoring, drift detection, and the trace → dataset → eval flywheel
- **Latest research** from Anthropic (Constitutional AI revisions, alignment faking, sabotage evaluations, modular red-team scaffolds), UK AISI (Inspect framework), NVIDIA (front-loading reasoning), Apollo Research (scheming evals), and ICLR/NeurIPS 2025-2026

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
- **OpenAI API** (GPT-4o, GPT-4o-mini, o3/o4-mini reasoning models) and **Anthropic API** (Claude Opus 4.5, Sonnet 4.5, Haiku 4.5)
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
| **Public leaderboards** | [Chatbot Arena (LMSYS)](https://lmarena.ai/), [LiveBench](https://livebench.ai/), [SWE-Bench Verified](https://www.swebench.com/), [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard), [SEAL leaderboards (Scale)](https://scale.com/leaderboard) |

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
# Pattern 1: Simple LLM-as-Judge
score = llm.invoke(f"Rate this response 1-5: {response}")

# Pattern 2: Pairwise Comparison
winner = llm.invoke(f"Which is better? A: {a} or B: {b}")

# Pattern 3: Rubric-Based
scores = llm.invoke(f"Score on accuracy, helpfulness, safety: {response}")

# Pattern 4: Hierarchical (Cost-Optimized)
if fast_filter.passes(response):
    if cheap_llm.score(response) > 0.7:
        return cheap_llm.score(response)
    else:
        return expensive_llm.detailed_score(response)
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

### Transparency Reports
- [Anthropic FMTI Transparency Report (December 2025)](https://crfm.stanford.edu/fmti/December-2025/company-reports/Anthropic_FinalReport_FMTI2025.html)
- [Claude Opus 4.5 System Card](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf)

---

**The field of evaluation engineering is the most critical skill for ensuring AI goes well for humanity. Master it.**