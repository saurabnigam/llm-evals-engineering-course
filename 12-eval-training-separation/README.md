# Module 12: Eval-Training Separation & Benchmark Integrity

> **The Most Critical Problem in AI Evaluation That Most People Ignore**
>
> If your evaluation data leaked into training, your benchmarks are meaningless. This module covers the science of keeping evals honest -- from data contamination to dynamic benchmarks to the techniques Anthropic uses to maintain eval integrity across Claude model generations.

---

## 12.1 Why Eval-Training Separation Is Existential

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
│  Result: High benchmark scores that DON'T reflect real capability.          │
│  The model "memorized the test" rather than learned the skill.              │
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
│  Threshold: cosine_similarity > 0.85                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 2: INFORMATIONAL CONTAMINATION                            │
│  "Does training data contain information that directly answers?" │
│  Method: Information extraction + overlap analysis               │
│  Threshold: key_info_overlap > 0.7                              │
├─────────────────────────────────────────────────────────────────┤
│  Level 3: DATA CONTAMINATION                                     │
│  "Are eval data points (questions) in the training set?"        │
│  Method: N-gram matching, fuzzy deduplication                   │
│  Threshold: n_gram_overlap(n=10) > 0.5                          │
├─────────────────────────────────────────────────────────────────┤
│  Level 4: LABEL CONTAMINATION                                    │
│  "Are the eval ANSWERS in the training set?"                    │
│  Method: Answer extraction + matching                           │
│  Threshold: exact_answer_match > 0.3                            │
└─────────────────────────────────────────────────────────────────┘

Output: Unified DCR Factor that adjusts raw accuracy to reflect 
contamination-aware performance.

Adjusted_Score = Raw_Score * (1 - DCR_Factor)
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

### Implementing Contamination Detection

```python
"""
Contamination detection for your evaluation datasets.
Use this BEFORE trusting any benchmark result.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

class ContaminationDetector:
    """
    Detect potential data contamination between eval sets and model training.
    Since we can't access training data directly, we use indirect signals.
    """
    
    def __init__(self, model, embedding_model=None):
        self.model = model
        self.embedding_model = embedding_model
    
    def detect_memorization(self, 
                           eval_items: List[Dict],
                           num_probes: int = 5) -> Dict:
        """
        Test if model has memorized eval items by checking if it can
        complete partial questions or predict exact answer formats.
        """
        results = []
        
        for item in eval_items:
            question = item["question"]
            expected = item["answer"]
            
            # Probe 1: Can model complete a truncated question?
            truncated = question[:len(question)//2]
            completion = self.model.generate(
                f"Complete this question: {truncated}",
                temperature=0.0,
                max_tokens=200
            )
            completion_overlap = self._text_similarity(
                completion, question[len(question)//2:]
            )
            
            # Probe 2: Does model give suspiciously exact answers?
            answer = self.model.generate(
                question,
                temperature=0.0,
                max_tokens=100
            )
            answer_exact_match = self._normalize(answer) == self._normalize(expected)
            
            # Probe 3: Does model know the benchmark source?
            source_probe = self.model.generate(
                f"Is this question from a well-known benchmark? "
                f"If so, which one? Question: {question}",
                temperature=0.0
            )
            knows_source = any(
                bench in source_probe.lower() 
                for bench in ["mmlu", "hellaswag", "humaneval", "gsm8k", "arc"]
            )
            
            # Probe 4: Confidence calibration
            # Contaminated items often have abnormally high confidence
            logprobs = self.model.generate_with_logprobs(question)
            avg_confidence = np.mean([lp for lp in logprobs if lp is not None])
            
            results.append({
                "question_id": item.get("id", "unknown"),
                "completion_overlap": completion_overlap,
                "exact_match": answer_exact_match,
                "knows_source": knows_source,
                "avg_confidence": avg_confidence,
                "contamination_risk": self._compute_risk(
                    completion_overlap, answer_exact_match, 
                    knows_source, avg_confidence
                )
            })
        
        # Aggregate
        high_risk = sum(1 for r in results if r["contamination_risk"] > 0.7)
        
        return {
            "total_items": len(results),
            "high_risk_items": high_risk,
            "contamination_rate": high_risk / len(results),
            "recommendation": self._recommend(high_risk / len(results)),
            "details": results
        }
    
    def _compute_risk(self, completion_overlap, exact_match, 
                      knows_source, avg_confidence) -> float:
        """Compute overall contamination risk score"""
        risk = 0.0
        risk += 0.3 * completion_overlap
        risk += 0.3 * (1.0 if exact_match else 0.0)
        risk += 0.2 * (1.0 if knows_source else 0.0)
        risk += 0.2 * min(1.0, max(0.0, (avg_confidence + 2) / 4))
        return risk
    
    def _recommend(self, contamination_rate: float) -> str:
        if contamination_rate > 0.3:
            return "HIGH CONTAMINATION: Do not use this benchmark. Create private eval set."
        elif contamination_rate > 0.1:
            return "MODERATE CONTAMINATION: Supplement with private evals. Discount scores by ~15%."
        else:
            return "LOW CONTAMINATION: Benchmark likely reliable. Continue monitoring."
    
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
    Create decontaminated versions of existing benchmarks.
    Transforms questions while preserving the skill being tested.
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def decontaminate_item(self, item: Dict, method: str = "paraphrase") -> Dict:
        """Transform a benchmark item to reduce contamination risk"""
        
        methods = {
            "paraphrase": self._paraphrase,
            "context_noise": self._add_context_noise,
            "polarity_reverse": self._reverse_polarity,
            "format_change": self._change_format,
            "difficulty_shift": self._shift_difficulty,
        }
        
        transform = methods.get(method, self._paraphrase)
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
        return {**item, "question": new_question, "decontaminated": True, "method": "paraphrase"}
    
    def _add_context_noise(self, item: Dict) -> Dict:
        """Add irrelevant context to test robustness"""
        prompt = f"""Add 1-2 sentences of plausible but irrelevant context 
to this question. The correct answer should remain the same, but the 
model must identify what's relevant.

Original: {item['question']}

Provide the modified question:"""
        
        new_question = self.llm.generate(prompt, temperature=0.7)
        return {**item, "question": new_question, "decontaminated": True, "method": "context_noise"}
    
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
            return {**item, **parsed, "decontaminated": True, "method": "polarity_reverse"}
        except:
            return item


class DynamicBenchmarkGenerator:
    """
    Generate fresh evaluation items on-the-fly.
    The gold standard for contamination-free evaluation.
    """
    
    def __init__(self, llm, domain_spec: Dict):
        self.llm = llm
        self.domain_spec = domain_spec
    
    def generate_eval_set(self, 
                          num_items: int,
                          difficulty_distribution: Dict = None) -> List[Dict]:
        """
        Generate a fresh evaluation set that has never been seen by any model.
        
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
        
        items = []
        for level, proportion in difficulty_distribution.items():
            n = int(num_items * proportion)
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
        
        result = self.llm.generate(prompt, temperature=0.8)
        import json
        try:
            return json.loads(result)
        except:
            return []
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
Models can't memorize because the questions are always new.
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

The gradient matters: refresh-based designs (LiveBench, SWE-rebench) buy you months; **interactivity (ARC-AGI-3) is the strongest contamination resistance available**, because there is no answer key — leaked or otherwise — only an environment the agent must explore. Expect more eval formats to move this way as static and even refreshed sets keep getting eaten.

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

## 12.9 Building Contamination-Proof Eval Systems

### The Private Eval Infrastructure

```python
"""
Enterprise-grade contamination-proof evaluation infrastructure.
Never publish your eval data. Rotate questions regularly.
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
        This is the key to contamination-proof evaluation.
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
        
        if overlap > 0.1:  # More than 10% semantic overlap
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

Two small probes that catch the most common contamination patterns in 5 minutes.

#### Example 1 — Canary-string memorization probe

If a model can complete a benchmark item from a tiny prefix, it has likely seen it during training.

```python
from openai import OpenAI
client = OpenAI()

def memorization_score(item: dict, model="gpt-5.5") -> float:
    """Return fraction of the GROUND-TRUTH answer the model regenerates
    given only the FIRST 8 WORDS of the canonical question."""
    prefix = " ".join(item["question"].split()[:8])
    out = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "user",
                   "content": f"Continue this benchmark item verbatim:\n{prefix}"}],
    ).choices[0].message.content
    # Token-overlap with the *exact* canonical answer is the smoking gun
    answer_tokens = set(item["answer"].lower().split())
    out_tokens    = set(out.lower().split())
    return len(answer_tokens & out_tokens) / max(1, len(answer_tokens))

# >0.5 on randomly sampled items → strong contamination signal
scores = [memorization_score(x) for x in mmlu_sample]
print("mean overlap:", sum(scores)/len(scores))
```

#### Example 2 — Original vs paraphrased delta

If the model scores meaningfully higher on the canonical wording than on a semantically-identical paraphrase, the gap is (mostly) memorization.

```python
from anthropic import Anthropic
client = Anthropic()

def paraphrase(q: str) -> str:
    r = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=200, temperature=0.4,
        messages=[{"role":"user","content":
            f"Paraphrase this question. Keep the answer the same. "
            f"Change wording, sentence structure, and any proper nouns that don't "
            f"affect the answer.\n\nQ: {q}"}])
    return r.content[0].text

orig_acc = run_eval(items)                              # canonical wording
para_acc = run_eval([{**i, "question": paraphrase(i["question"])} for i in items])
print(f"Original: {orig_acc:.2%}   Paraphrased: {para_acc:.2%}   Gap: {orig_acc-para_acc:+.2%}")
# Healthy gap: 0–3%. >5% gap → contamination strongly suspected.
# Always report BOTH numbers in any benchmark claim.
```

These two probes — plus the dynamic-rewording recipe in section 12.6 — are the minimum hygiene required to take any 2026 leaderboard score seriously.

---

## 12.12 Exercises

### Exercise 1: Contamination Audit
Take a public benchmark (MMLU, HumanEval, GSM8K) and:
- Run the memorization probes from Section 12.4 against a model
- Estimate contamination risk
- Generate a decontaminated version of 10 items
- Compare model performance on original vs decontaminated items

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
