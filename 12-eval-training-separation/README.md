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

Layer 1: KNOWLEDGE CUTOFF DATES
  ├── Each model has a clear temporal boundary
  │   • Opus 4.6: May 2025
  │   • Opus/Sonnet 4: March 2025
  │   • Sonnet 3.7: October 2024
  ├── Evals created AFTER the cutoff cannot have been seen
  └── But this only helps for time-gated evals

Layer 2: USER DATA EXCLUSION
  ├── "Claude models have NOT been trained on any user prompt
  │    or output data" from deployments
  ├── This prevents eval data sent via API from leaking back
  └── Critical for customers running proprietary evals

Layer 3: DATA DECONTAMINATION
  ├── Deduplication pipelines remove near-duplicates
  ├── Classification filters identify benchmark-like content
  └── For specific benchmarks (e.g., ARC-AGI-1): train on
      reshuffled splits, report only on held-out test sets

Layer 4: SEMI-PRIVATE & PRIVATE TEST SETS
  ├── Maintain held-out test sets not published online
  ├── Use third-party evaluation services (METR, etc.)
  └── Internal evals never published in training data

Layer 5: WEB CRAWLER TRANSPARENCY
  ├── Anthropic operates its own web crawler
  ├── Follows robots.txt (site operators can opt out)
  └── Enables tracking which sources are in training data
```

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
    }
}
```

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

MMLU (2021):
  GPT-3:          43.9%
  GPT-4 (2023):   86.4%
  Claude 3.5:     88.7%
  GPT-4o (2024):  88.7%
  ← SATURATED: Models cluster at top, can't differentiate

HumanEval (2021):
  Codex (2021):   28.8%
  GPT-4 (2023):   67.0%
  Claude 3.5:     92.0%
  Claude Opus 4:  ~95%+
  ← SATURATED: Near-ceiling performance

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

---

## 12.7 Building Contamination-Proof Eval Systems

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

## 12.8 The Evaluation Integrity Checklist

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

## 12.8b Worked Examples (2026)

Two small probes that catch the most common contamination patterns in 5 minutes.

#### Example 1 — Canary-string memorization probe

If a model can complete a benchmark item from a tiny prefix, it has likely seen it during training.

```python
from openai import OpenAI
client = OpenAI()

def memorization_score(item: dict, model="gpt-4o") -> float:
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
        model="claude-sonnet-4-5", max_tokens=200, temperature=0.4,
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

## 12.9 Exercises

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

---

## Next Module
-> [Module 13: Advancing AI Research -- Contributing to the Frontier](../13-advancing-ai-research/README.md)
