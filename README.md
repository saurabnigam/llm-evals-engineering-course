# Eval Engineering: Complete Study Guide

> **The Art and Science of Evaluating AI/LLM Systems at Scale**
> 
> A comprehensive guide for software engineers and aspiring AI researchers learning to build production-grade evaluation systems for LLMs, RAG systems, and AI agents. Updated for 2026 with the latest research from Anthropic, OpenAI, and the broader AI safety community.

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

- **Module 11**: Deep dive into how frontier models (Claude Opus 4.6, Sonnet, GPT-4) are actually trained -- pretraining, SFT, reward modeling, RLHF, Constitutional AI, and safety fine-tuning
- **Module 12**: Eval-training separation, benchmark contamination detection, dynamic benchmarks, and the DCR framework
- **Module 13**: Research frontier -- alignment faking detection, open problems, how to contribute to AI safety research, and a complete path from eval engineer to AI researcher
- **Module 10 update**: Alignment faking evaluation, EDDOps (Evaluation-Driven Development and Operations), self-evolving eval systems
- **Module 02 update**: Psychometric evaluation (IRT, Bloom's taxonomy), adaptive testing, dynamic benchmark generation, evaluation-driven development workflow
- **Latest research** from Anthropic (Constitutional AI 2026 revision, modular red team scaffolds, SHADE-Arena, alignment faking), NVIDIA (front-loading reasoning), and ICLR 2026 (RL compute scaling laws)

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
- **LangChain** for LLM orchestration
- **OpenAI API** (GPT-4o, GPT-4o-mini) and **Anthropic API** (Claude Opus, Sonnet)
- **Pydantic** for data validation
- **Redis/Celery** for distributed processing
- **GitHub Actions** for CI/CD
- **NumPy/SciPy/scikit-learn** for statistical analysis and calibration

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

### Safety & Alignment Research
- [Alignment Faking in Large Language Models](https://www.anthropic.com/research/alignment-faking) -- Anthropic (2024)
- [Strengthening Red Teams: Modular Scaffold for Control Evaluations](https://alignment.anthropic.com/2025/strengthening-red-teams/) -- Anthropic (2025)
- [Claude's Constitution (2026 Revision)](https://www.anthropic.com/constitution) -- Anthropic

### Evaluation Research (2025-2026)
- [AdEval: Alignment-based Dynamic Evaluation](https://arxiv.org/abs/2501.13983) -- 2025
- [Benchmarking LLMs Under Data Contamination](https://aclanthology.org/2025.emnlp-main.511/) -- EMNLP 2025
- [The Art of Scaling RL Compute for LLMs](https://openreview.net/forum?id=FMjeC9Msws) -- ICLR 2026
- [Front-Loading Reasoning](https://research.nvidia.com/labs/adlr/Synergy/) -- NVIDIA 2025

### Tools & Frameworks
- [OpenAI's Eval Framework](https://github.com/openai/evals)
- [LangSmith Documentation](https://docs.smith.langchain.com/)
- [Braintrust AI Evaluation](https://www.braintrustdata.com/)
- [Anthropic's Transparency Hub](https://www.anthropic.com/transparency)

### Transparency Reports
- [Anthropic FMTI Transparency Report (December 2025)](https://crfm.stanford.edu/fmti/December-2025/company-reports/Anthropic_FinalReport_FMTI2025.html)
- [Claude Opus 4.5 System Card](https://assets.anthropic.com/m/64823ba7485345a7/Claude-Opus-4-5-System-Card.pdf)

---

**The field of evaluation engineering is the most critical skill for ensuring AI goes well for humanity. Master it.**