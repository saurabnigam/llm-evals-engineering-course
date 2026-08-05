# Module 2: Evaluation Methods & Techniques

## 2.1 Overview of Evaluation Approaches

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION METHODS TAXONOMY                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                        ┌─────────────────┐                              │
│                        │   Evaluation    │                              │
│                        │    Methods      │                              │
│                        └────────┬────────┘                              │
│                                 │                                        │
│            ┌────────────────────┼────────────────────┐                  │
│            ▼                    ▼                    ▼                  │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│   │   Rule-Based    │  │  Model-Based    │  │  Human-Based    │        │
│   │   (Automated)   │  │  (LLM-as-Judge) │  │  (Crowdsource)  │        │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘        │
│            │                    │                    │                  │
│    ┌───────┴───────┐    ┌───────┴───────┐    ┌───────┴───────┐         │
│    │ • Regex       │    │ • Single Judge│    │ • Expert      │         │
│    │ • Exact Match │    │ • Multi-Judge │    │ • Crowdworker │         │
│    │ • Keyword     │    │ • Pairwise    │    │ • A/B Testing │         │
│    │ • Structured  │    │ • Reference   │    │ • User Studies│         │
│    │ • Code Exec   │    │ • Rubric-based│    │               │         │
│    └───────────────┘    └───────────────┘    └───────────────┘         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2.2 Rule-Based Evaluation

### When to Use
- Deterministic, objective criteria
- High-volume, low-cost evaluation
- Clear pass/fail conditions
- Regression testing

### 2.2.1 Exact Match

```python
class ExactMatchEvaluator:
    """Simple but limited - useful for factual QA"""
    
    def evaluate(self, prediction: str, ground_truth: str) -> float:
        # Normalize both strings
        pred_normalized = self._normalize(prediction)
        truth_normalized = self._normalize(ground_truth)
        return 1.0 if pred_normalized == truth_normalized else 0.0
    
    def _normalize(self, text: str) -> str:
        return text.lower().strip()

# Example usage
evaluator = ExactMatchEvaluator()
print(evaluator.evaluate("Paris", "paris"))  # 1.0
print(evaluator.evaluate("Paris, France", "Paris"))  # 0.0
```

### 2.2.2 Fuzzy Matching

```python
from difflib import SequenceMatcher
from typing import List

class FuzzyMatchEvaluator:
    """Handles minor variations in text"""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
    
    def evaluate(self, prediction: str, ground_truth: str) -> float:
        ratio = SequenceMatcher(None, 
                                prediction.lower(), 
                                ground_truth.lower()).ratio()
        return ratio
    
    def evaluate_with_alternatives(self, 
                                   prediction: str, 
                                   alternatives: List[str]) -> float:
        """Check against multiple acceptable answers"""
        scores = [self.evaluate(prediction, alt) for alt in alternatives]
        return max(scores)

# Example
evaluator = FuzzyMatchEvaluator()
print(evaluator.evaluate("The Eiffel Tower", "Eiffel Tower"))  # ~0.93
print(evaluator.evaluate_with_alternatives(
    "ML", 
    ["Machine Learning", "ML", "machine learning"]
))  # 1.0
```

### 2.2.3 Regex-Based Evaluation

```python
import re

class RegexEvaluator:
    """For structured outputs with known patterns"""
    
    def __init__(self, patterns: dict):
        self.patterns = {k: re.compile(v) for k, v in patterns.items()}
    
    def evaluate(self, output: str, required_patterns: List[str]) -> dict:
        results = {}
        for pattern_name in required_patterns:
            pattern = self.patterns.get(pattern_name)
            if pattern:
                match = pattern.search(output)
                results[pattern_name] = {
                    'matched': bool(match),
                    'value': match.group() if match else None
                }
        return results

# Example: Validate code generation output
code_evaluator = RegexEvaluator({
    'has_function_def': r'def\s+\w+\s*\([^)]*\):',
    'has_docstring': r'"""[\s\S]*?"""',
    'has_return': r'return\s+.+',
    'has_type_hints': r'def\s+\w+\s*\([^)]*:\s*\w+[^)]*\)',
    'no_print_statements': r'^(?!.*print\().*$'
})

code_output = '''
def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers."""
    return a + b
'''

results = code_evaluator.evaluate(code_output, [
    'has_function_def', 
    'has_docstring', 
    'has_return', 
    'has_type_hints'
])
print(results)
```

### 2.2.4 Structured Output Validation

```python
from pydantic import BaseModel, ValidationError
from typing import Optional
import json

class StructuredOutputEvaluator:
    """Validate JSON/structured outputs against schemas"""
    
    def __init__(self, schema: type[BaseModel]):
        self.schema = schema
    
    def evaluate(self, output: str) -> dict:
        try:
            # Try to parse as JSON
            data = json.loads(output)
            # Validate against schema
            validated = self.schema(**data)
            return {
                'valid': True,
                'parsed': validated.model_dump(),
                'errors': None
            }
        except json.JSONDecodeError as e:
            return {
                'valid': False,
                'parsed': None,
                'errors': [f"JSON parse error: {e}"]
            }
        except ValidationError as e:
            return {
                'valid': False,
                'parsed': None,
                'errors': [str(err) for err in e.errors()]
            }

# Example: Validate API response schema
class ProductRecommendation(BaseModel):
    product_id: str
    product_name: str
    confidence: float
    reason: Optional[str] = None

evaluator = StructuredOutputEvaluator(ProductRecommendation)

# Valid output
valid_output = '{"product_id": "123", "product_name": "Widget", "confidence": 0.95}'
print(evaluator.evaluate(valid_output))  # {'valid': True, ...}

# Invalid output
invalid_output = '{"product_id": "123", "confidence": "high"}'
print(evaluator.evaluate(invalid_output))  # {'valid': False, 'errors': [...]}
```

### 2.2.5 Code Execution Evaluation

```python
import subprocess
import tempfile
import os
from typing import Tuple

class CodeExecutionEvaluator:
    """Evaluate code by actually running it"""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
    
    def evaluate_python(self, code: str, test_cases: list) -> dict:
        results = []
        
        for test in test_cases:
            # Create test script
            test_script = f"""
{code}

# Test case
result = {test['function_call']}
expected = {test['expected']}
assert result == expected, f"Expected {{expected}}, got {{result}}"
print("PASS")
"""
            success, output, error = self._run_code(test_script)
            results.append({
                'test': test['name'],
                'passed': success and 'PASS' in output,
                'output': output,
                'error': error
            })
        
        passed = sum(1 for r in results if r['passed'])
        return {
            'score': passed / len(results) if results else 0,
            'passed': passed,
            'total': len(results),
            'details': results
        }
    
    def _run_code(self, code: str) -> Tuple[bool, str, str]:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return (
                result.returncode == 0,
                result.stdout,
                result.stderr
            )
        except subprocess.TimeoutExpired:
            return (False, '', 'Timeout exceeded')
        finally:
            os.unlink(temp_path)

# Example usage
code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

test_cases = [
    {'name': 'fib_0', 'function_call': 'fibonacci(0)', 'expected': 0},
    {'name': 'fib_1', 'function_call': 'fibonacci(1)', 'expected': 1},
    {'name': 'fib_5', 'function_call': 'fibonacci(5)', 'expected': 5},
    {'name': 'fib_10', 'function_call': 'fibonacci(10)', 'expected': 55},
]

evaluator = CodeExecutionEvaluator()
results = evaluator.evaluate_python(code, test_cases)
print(f"Score: {results['score']}")  # 1.0
```

---

## 2.3 Model-Based Evaluation (LLM-as-Judge)

### The Core Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM-AS-JUDGE PATTERN                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │  Input   │     │  Model   │     │  Output  │               │
│   │  Query   │────▶│  Under   │────▶│  to      │               │
│   │          │     │  Test    │     │  Evaluate│               │
│   └──────────┘     └──────────┘     └──────────┘               │
│        │                                  │                      │
│        │                                  │                      │
│        ▼                                  ▼                      │
│   ┌─────────────────────────────────────────────────┐           │
│   │              JUDGE PROMPT                        │           │
│   │  ┌─────────────────────────────────────────┐   │           │
│   │  │ You are an expert evaluator. Rate the   │   │           │
│   │  │ following response on a scale of 1-5:   │   │           │
│   │  │                                         │   │           │
│   │  │ Question: {input_query}                 │   │           │
│   │  │ Response: {model_output}                │   │           │
│   │  │ Reference: {ground_truth} (optional)    │   │           │
│   │  │                                         │   │           │
│   │  │ Criteria:                               │   │           │
│   │  │ - Accuracy                              │   │           │
│   │  │ - Helpfulness                           │   │           │
│   │  │ - Clarity                               │   │           │
│   │  └─────────────────────────────────────────┘   │           │
│   └─────────────────────────────────────────────────┘           │
│                          │                                       │
│                          ▼                                       │
│   ┌──────────────────────────────────────────────┐              │
│   │ Judge LLM (GPT-5.5, Opus 4.8, etc.)         │              │
│   └──────────────────────────────────────────────┘              │
│                          │                                       │
│                          ▼                                       │
│   ┌──────────────────────────────────────────────┐              │
│   │ Score: 4/5                                   │              │
│   │ Reasoning: "The response accurately..."     │              │
│   └──────────────────────────────────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

> ### ⚠️ The diagram above is the pattern you will meet everywhere — and it has three defects
>
> It is drawn as most tutorials and most production code write it, because you need to recognise it. Do not copy it. Three things are wrong, and §1.4d, §2.3.6 and Module 00 §0.4 all circle back to them:
>
> | Defect in the diagram | Why it bites | Fix |
> |---|---|---|
> | **A 1–5 scale** | Judges bunch on 3 and 4, so five levels carry maybe two bits. Nobody can define the 3/4 boundary, so the number is not reproducible across judges, runs, or people — and you cannot act on "3.6" | **Binary verdict + quoted evidence.** If you need gradation, get it from *how many* binary criteria passed |
> | **Three criteria in one call** | Halo effect: a fluent, well-organised answer pulls its *accuracy* score up, because the judge is looking at all three at once | **One isolated call per criterion** (§2.3.4) |
> | **Score first, reasoning after** | The score gets generated before the justification exists, so the "reasoning" field is a post-hoc rationalisation of a number the model already committed to | **Evidence first, verdict last** — make the model quote the text before it judges it |
>
> Everything below is written the corrected way. When you see a 1–5 judge in someone else's harness, that is not automatically wrong — it is a signal to go check their κ against human labels (§2.3.6).

### 2.3.1 Single Judge Implementation

The current-correct shape: **one criterion per call, evidence before verdict, schema enforced by the API rather than by a parser**. Structured outputs (`output_config.format`) make the response shape a guarantee — no `json.loads` in a retry loop, no "output ONLY valid JSON" in the prompt, no assistant prefill (which returns a 400 on current models — Module 15 §15.4).

```python
# pip install anthropic pydantic
import anthropic, json
from pydantic import BaseModel
from typing import Optional

client = anthropic.Anthropic()

class Verdict(BaseModel):
    evidence: str      # ORDER MATTERS: quoted first...
    verdict: str       # ...so the judgement is derived from it, not rationalised after
    confidence: str

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {"type": "string",
                     "description": "Quote the exact span of the response you are judging."},
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["evidence", "verdict", "confidence"],
    "additionalProperties": False,
}

class LLMJudge:
    """Single-criterion LLM evaluator."""

    def __init__(self, model: str = "claude-opus-5", effort: str = "high"):
        self.model, self.effort = model, effort

    def evaluate(self,
                 question: str,
                 answer: str,              # renamed from `response` — the old code
                 criterion: str,           # shadowed this name with the API response
                 reference: Optional[str] = None) -> Verdict | None:
        ref = f"\nReference answer: {reference}" if reference else ""
        prompt = (
            f"Question asked: {question}\n"
            f"Response given: {answer}{ref}\n\n"
            f"Criterion — {criterion}\n\n"
            "First quote the exact span of the response that determines this "
            "criterion. Then give a PASS or FAIL verdict on that evidence alone. "
            "If the response already satisfies the criterion, PASS it — do not "
            "look for reasons to fail a good answer."
        )
        r = client.messages.create(
            model=self.model, max_tokens=2000,
            output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
                           "effort": self.effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if r.stop_reason == "refusal":
            return None       # UNMEASURED — not a FAIL. See Module 15 §15.4
        return Verdict(**json.loads(next(b.text for b in r.content if b.type == "text")))


# Example usage — note one call per criterion, not one call for all of them
judge = LLMJudge()
question = "Explain quantum entanglement to a 10-year-old"
answer = ("Quantum entanglement is like having two magic coins that always land "
          "the same way, no matter how far apart they are!")

for criterion in [
    "Is the explanation factually defensible as a simplification (not actively misleading)?",
    "Is the vocabulary appropriate for a 10-year-old?",
    "Does it avoid implying that information travels faster than light?",
]:
    v = judge.evaluate(question, answer, criterion)
    print(f"{v.verdict if v else 'UNMEASURED':10}  {criterion[:50]}...")
    if v and v.verdict == "FAIL":
        print(f"            evidence: {v.evidence}")
```

Three details in there are the whole lesson:

1. **`evidence` is declared before `verdict` in the schema.** Structured outputs are generated in field order, so this forces the model to find and quote the relevant text *before* committing to a judgement. Reversing these two lines measurably changes results — it turns the evidence field into a justification of a decision already made.
2. **The prompt explicitly permits passing.** Judges drift toward finding fault because criticism reads as rigour; without that sentence, false-positive rate climbs, and in a retry loop that directly burns budget (Module 14 §14.3).
3. **A refusal returns `None`, not `FAIL`.** Scoring a refusal as a failure invents a measurement nobody made, and refusals cluster by topic — so it depresses scores in precisely the categories you are trying to assess.

**Where a graded score IS legitimate:** ranking and triage, not gating. If you need to sort 500 outputs by quality to review the worst 20, a continuous score is fine — you only care about ordering. The moment a number becomes a release gate or a reported metric, switch to binary criteria you can define.

### 2.3.2 Multi-Judge Panel

```python
from typing import List
import statistics

class MultiJudgePanel:
    """Use multiple judges for more robust evaluation"""
    
    def __init__(self, judges: List[LLMJudge]):
        self.judges = judges
    
    def evaluate(self, 
                 question: str, 
                 response: str, 
                 criteria: str,
                 aggregation: str = "median") -> dict:
        
        results = []
        for judge in self.judges:
            result = judge.evaluate(question, response, criteria)
            results.append(result)
        
        scores = [r.score for r in results]
        
        if aggregation == "median":
            final_score = statistics.median(scores)
        elif aggregation == "mean":
            final_score = statistics.mean(scores)
        elif aggregation == "min":  # Conservative
            final_score = min(scores)
        else:
            final_score = statistics.mean(scores)
        
        return {
            'final_score': final_score,
            'individual_scores': scores,
            'agreement': 1 - statistics.stdev(scores) if len(scores) > 1 else 1.0,
            'individual_results': results
        }

# Example: Panel of diverse judges (mix providers AND sizes)
panel = MultiJudgePanel([
    LLMJudge(model="gpt-5.5"),
    LLMJudge(model="claude-sonnet-4-6"),
    LLMJudge(model="claude-haiku-4-5")  # Small judge: cheap dissenting vote
])
```

**Why panels beat a single big judge:** the "Replacing Judges with Juries" result (PoLL) showed a panel of 3 small, diverse judges outperforms a single GPT-4 judge with less intra-model bias at ~1/7 the cost ([arXiv 2404.18796](https://arxiv.org/abs/2404.18796)). This is now standard at frontier scale: the Petri 3.0 cross-lab behavioral audit scores every transcript with **three judges from different providers** (Opus 4.7, GPT-5.5, Gemini 3.1 Pro) and reports the average ([Fable 5 system card §6.2.3.3](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)). For composing larger judge pipelines, see [Verdict](https://arxiv.org/pdf/2502.18018).

### 2.3.3 Pairwise Comparison (A/B Evaluation)

```
┌─────────────────────────────────────────────────────────────────┐
│                  PAIRWISE COMPARISON FLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    ┌──────────────┐                             │
│                    │   Question   │                             │
│                    └──────┬───────┘                             │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              ▼                         ▼                        │
│       ┌────────────┐           ┌────────────┐                   │
│       │  Model A   │           │  Model B   │                   │
│       │  Response  │           │  Response  │                   │
│       └─────┬──────┘           └──────┬─────┘                   │
│             │                         │                          │
│             └───────────┬─────────────┘                          │
│                         ▼                                        │
│            ┌─────────────────────────┐                          │
│            │     JUDGE PROMPT        │                          │
│            │  "Which response is     │                          │
│            │   better? A, B, or TIE" │                          │
│            └───────────┬─────────────┘                          │
│                        ▼                                         │
│            ┌─────────────────────────┐                          │
│            │  Winner: A | B | TIE    │                          │
│            │  Reason: "..."          │                          │
│            └─────────────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
from enum import Enum

class ComparisonResult(Enum):
    A_BETTER = "a"
    B_BETTER = "b"
    TIE = "tie"

class PairwiseEvaluator:
    """Compare two responses directly"""
    
    def __init__(self, model: str = "gpt-5.5"):
        self.client = OpenAI()
        self.model = model
    
    def compare(self, 
                question: str, 
                response_a: str, 
                response_b: str,
                criteria: str) -> dict:
        
        # IMPORTANT: Randomize order to avoid position bias
        import random
        if random.random() > 0.5:
            first, second = response_a, response_b
            order = "normal"
        else:
            first, second = response_b, response_a
            order = "swapped"
        
        prompt = f"""Compare these two responses to the same question.

Question: {question}

Response A:
{first}

Response B:
{second}

Evaluation Criteria: {criteria}

Which response is better? Respond with JSON:
{{
    "winner": "A" | "B" | "TIE",
    "reasoning": "<explanation>",
    "a_strengths": ["..."],
    "b_strengths": ["..."]
}}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Correct for swapped order
        if order == "swapped":
            if result['winner'] == 'A':
                result['winner'] = 'B'
            elif result['winner'] == 'B':
                result['winner'] = 'A'
        
        return result
    
    def run_tournament(self, question: str, responses: dict, criteria: str) -> dict:
        """Run round-robin tournament between multiple responses"""
        from itertools import combinations
        
        scores = {name: 0 for name in responses}
        comparisons = []
        
        for (name_a, resp_a), (name_b, resp_b) in combinations(responses.items(), 2):
            result = self.compare(question, resp_a, resp_b, criteria)
            
            if result['winner'] == 'A':
                scores[name_a] += 1
            elif result['winner'] == 'B':
                scores[name_b] += 1
            else:  # TIE
                scores[name_a] += 0.5
                scores[name_b] += 0.5
            
            comparisons.append({
                'a': name_a,
                'b': name_b,
                'winner': result['winner'],
                'reasoning': result['reasoning']
            })
        
        # Rank by score
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'ranking': ranking,
            'comparisons': comparisons
        }
```

### 2.3.4 Rubric-Based Evaluation

```python
class RubricEvaluator:
    """Detailed rubric-based evaluation with specific criteria.

    NOTE ON THE 1-5 SCALES BELOW — this is NOT the anti-pattern from §2.3.
    The difference is anchoring. "Rate clarity 1-5" is unanchored: the levels
    mean whatever the judge decides today. The rubric here defines every level
    in observable terms ("Code works for basic cases but has some bugs"), which
    makes 3-vs-4 a question about the artifact rather than about taste, and
    makes the score reproducible across judges and runs.

    Test for whether your scale is anchored: could two people who have never
    met assign the same level to the same artifact, using only your level
    descriptions? If not, you have a Likert scale wearing a rubric costume ---
    collapse it to binary criteria instead.
    """
    
    def __init__(self, rubric: dict):
        self.rubric = rubric
        self.client = OpenAI()
    
    def evaluate(self, question: str, response: str) -> dict:
        rubric_text = self._format_rubric()
        
        prompt = f"""Evaluate the response using this detailed rubric.

Question: {question}

Response to evaluate:
{response}

RUBRIC:
{rubric_text}

For each criterion, provide:
1. The score (using the exact levels from the rubric)
2. Evidence from the response supporting your score
3. Specific suggestions for improvement

Respond in JSON format with a "scores" object containing each criterion.
"""
        
        # NOTE: named `api_result`, not `response` — `response` is this method's
        # own parameter, and shadowing it is how a later edit silently breaks.
        api_result = self.client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(api_result.choices[0].message.content)
    
    def _format_rubric(self) -> str:
        lines = []
        for criterion, levels in self.rubric.items():
            lines.append(f"\n{criterion.upper()}:")
            for score, description in sorted(levels.items(), reverse=True):
                lines.append(f"  {score}: {description}")
        return "\n".join(lines)

# Example rubric for code review
code_review_rubric = {
    "correctness": {
        5: "Code is completely correct and handles all edge cases",
        4: "Code is correct for main cases, minor edge cases missed",
        3: "Code works for basic cases but has some bugs",
        2: "Code has significant logic errors",
        1: "Code does not work at all"
    },
    "readability": {
        5: "Exceptionally clear, well-commented, follows best practices",
        4: "Clear and readable, good naming conventions",
        3: "Readable but could use better naming or comments",
        2: "Difficult to follow, poor variable names",
        1: "Incomprehensible"
    },
    "efficiency": {
        5: "Optimal time and space complexity",
        4: "Good efficiency, minor optimizations possible",
        3: "Acceptable efficiency for most use cases",
        2: "Inefficient, would struggle with large inputs",
        1: "Extremely inefficient, would timeout"
    }
}

evaluator = RubricEvaluator(code_review_rubric)
```

#### The per-example rubric pattern (HealthBench / GDPval — the 2025-2026 default)

The rubric above is *global*: one rubric for every sample. The pattern that took over in 2025 is **per-example rubrics** — each test case carries its own expert-written, weighted criteria, and the judge grades each criterion **independently** (met / not met), never as one omnibus quality call.

The two canonical references:

| | [HealthBench](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf) (OpenAI, May 2025) | [GDPval](https://openai.com/index/gdpval/) (OpenAI, Sept 2025) |
|---|---|---|
| **What** | 5,000 physician-refined multi-turn health conversations, 26 specialties, 49 languages | 1,320 real deliverable tasks from 44 occupations in 9 GDP-heavy industries |
| **Who writes the gold standard** | Physicians: each conversation gets its own rubric (mean ~11–12 criteria; ~48,562 unique weighted criteria total) | Professionals averaging 14 yrs experience; ~7–9 hrs of expert time and ~$400 of value per task ([arXiv 2510.04374](https://arxiv.org/abs/2510.04374)) |
| **How it's graded** | Judge model grades each criterion independently; weighted results aggregate to a score | **Blind expert pairwise comparison** vs. human deliverables; headline metric = win rate |
| **Automation** | Judge model throughout | Trained automated grader: 66% agreement with experts vs. 71% human-human agreement — within 5 points of human consistency ([GDPval paper](https://cdn.openai.com/pdf/d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf)) |

Both are now headline numbers in frontier system cards: the [Fable 5 system card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) reports HealthBench 62.7 / HealthBench Professional 66.0 for Mythos 5 (Opus 4.8: 59.3 / 56.9; GPT-5.5: 56.5 / 51.8) and a GDPval-AA Elo of 1932.

```python
# pip install anthropic
# HealthBench-style per-example rubric grading: one isolated judge
# call per criterion, weighted aggregation. ~30 lines, production-shaped.
import json
from anthropic import Anthropic

client = Anthropic()

# Each EXAMPLE carries its own expert-written, weighted criteria.
# Positive weights reward; negative weights penalize.
criteria = [
    {"id": "advises_er",   "weight": 10,
     "text": "Advises seeking emergency care for chest pain radiating to the arm"},
    {"id": "asks_history", "weight": 5,
     "text": "Asks about relevant history (cardiac risk factors, medications)"},
    {"id": "jargon",       "weight": -3,
     "text": "Uses unexplained medical jargon"},
]

GRADER = """Grade ONE criterion. Reply with ONLY JSON:
{{"met": true | false | "unknown", "evidence": "<exact quote or 'none'>"}}

Criterion: {criterion}

Conversation:
{conversation}

Response to grade:
{response}"""

def grade(conversation: str, response: str) -> dict:
    results = {}
    for c in criteria:   # one ISOLATED call per criterion — never omnibus
        msg = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=200, temperature=0,
            messages=[{"role": "user", "content": GRADER.format(
                criterion=c["text"], conversation=conversation,
                response=response)}])
        results[c["id"]] = json.loads(msg.content[0].text)

    earned  = sum(c["weight"] for c in criteria
                  if results[c["id"]]["met"] is True)
    maximum = sum(c["weight"] for c in criteria if c["weight"] > 0)
    return {"score": max(0, earned) / maximum, "criteria": results}
```

Design rules, straight from Anthropic's agent-evals playbook ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), Jan 2026):

1. **One isolated judge call per rubric dimension** — omnibus judges blur criteria together.
2. **Give the judge an "Unknown" escape hatch** — forced binary verdicts manufacture noise.
3. **Calibrate frequently against expert human judgment** — rubric scores drift from expert intent (see 2.3.6).

The 2026 frontier of this pattern is automating the rubric authorship itself ([Automated Rubrics for Medical Dialogue, arXiv 2601.15161](https://arxiv.org/html/2601.15161); [RubricRAG, arXiv 2603.20882](https://arxiv.org/html/2603.20882)) and meta-evaluating judges at rubric level ([RubricEval, arXiv 2603.25133](https://arxiv.org/html/2603.25133v1)).

### 2.3.5 Reasoning-Model Judges

Production judge prompts increasingly run on reasoning models with an explicit **plan-then-grade** structure, rather than a single forward pass:

- **EvalPlanner** (Meta, Jan 2025): the judge first *generates an evaluation plan*, then executes it, then issues a verdict; trained via preference optimization ([arXiv 2501.18099](https://arxiv.org/abs/2501.18099)).
- **J1** (Meta, May 2025): RL-trained "thinking judges" (8B–70B) that outline criteria and self-generate reference answers before judging; SOTA on judge benchmarks at release ([arXiv 2505.10320](https://arxiv.org/abs/2505.10320)).
- Meta-evaluation of judges is its own subfield now: [JudgeBench](https://www.emergentmind.com/topics/judgebench) and successors ([arXiv 2512.16041](https://arxiv.org/html/2512.16041v1)).

Two caveats specific to reasoning judges:

1. **Judge CoT is not a faithful audit log.** Reasoning models don't always say what they think ([Anthropic, arXiv 2505.05410](https://arxiv.org/pdf/2505.05410)) — treat the judge's written rationale as a debugging aid, not ground truth for *why* it scored that way.
2. **Cost scales with judge-time compute.** Reasoning judges are several times the price per verdict; reserve them for ambiguous dimensions and keep deterministic/code graders for everything objective.

### 2.3.6 Calibrating Judges Against Humans

A judge you haven't calibrated is a random number generator with good vibes. The converged 2025-2026 loop:

```
1. Build a golden set      20-50 examples, human-labeled (expand to
                           200-500 from production failures over time)
2. Run judge side-by-side  same examples, same rubric
3. Measure agreement       Cohen's κ for categorical verdicts;
                           TPR/TNR per failure mode for binary judges
4. Iterate the prompt      feed human corrections back as few-shot
                           examples in the judge prompt
5. Re-measure on held-out  never tune and report on the same slice
```

- [LangSmith Align Evals](https://blog.langchain.com/introducing-align-evals/) (July 2025) productized exactly this loop, including storing human corrections as few-shot examples for the judge.
- The Hamel Husain / Shreya Shankar school operationalizes step 3 as TPR/TNR of the judge against human labels derived from error analysis ([evals FAQ, Jan 2026](https://hamel.dev/blog/posts/evals-faq/evals-faq.pdf)) — a judge with great accuracy but poor TNR on your most expensive failure mode is worse than no judge.
- See Example A in 2.7.8 for runnable Cohen's κ calibration code. Bands (matching Module 01 §1.3): **κ ≥ 0.8** strong enough for the verdict to gate a release; **0.6–0.8** usable as telemetry and for triage; **0.4–0.6** iterate the prompt or the rubric; **< 0.4** redesign — the judge and your humans are answering different questions, and no amount of prompt tuning fixes an under-specified criterion.

> **A κ caveat worth knowing before you report one.** κ is deflated when the label distribution is heavily skewed — with 95% passes, even a good judge can post a mediocre κ simply because there is little chance-corrected room to move (the "kappa paradox"). If κ looks bad but TPR and TNR both look fine, trust TPR/TNR and report all three. κ is a summary; the confusion matrix is the evidence.

#### The TPR/TNR arithmetic, worked (why "the judge is 86% accurate" means nothing)

Take the binary judge proposed at the end of module 01's error-analysis example — it flags responses that *explain the task instead of doing it*. You human-label a 50-example golden set: 10 true failures, 40 true passes. The judge's verdicts:

```
                      Judge: FLAG      Judge: PASS
  Human: failure (10)   TP = 5           FN = 5     ← missed half!
  Human: pass    (40)   FP = 2           TN = 38

  Accuracy = (5 + 38) / 50 = 86%   ← sounds shippable
  TPR = TP / (TP+FN) = 5/10  = 50%  ← catches HALF the failure mode
                                       it exists to catch
  TNR = TN / (TN+FP) = 38/40 = 95%
```

Accuracy is dominated by the majority class (passes), so a judge that's nearly blind to your failure mode still "scores" 86%. **TPR is the number that matters for a failure-mode judge** — here it says: iterate the prompt (add the FN transcripts as few-shot examples — step 4 of the loop above) before trusting this judge in CI.

The mirror failure is just as expensive. Suppose prompt-iteration gets TPR to 90% but TNR slips to 60%. On 1,000 production traces with a 5% true failure rate: the judge catches 45 of 50 real failures — and false-flags 380 of the 950 good ones. Your review queue is now 425 items, 89% noise, and within two weeks nobody on the team opens it. That is what "poor TNR is worse than no judge" means concretely: the judge didn't just fail, it *burned the team's trust in the whole eval system*. Report TPR **and** TNR per failure mode, and pick the operating point by which error costs more — never by the single accuracy number.

---

## 2.4 Human Evaluation

### When to Use Human Eval

| Use Human Eval When | Use Automated Eval When |
|---------------------|------------------------|
| Subjective quality matters | Objective metrics exist |
| Nuance is critical | Scale is paramount |
| Safety/ethics review | Cost is a concern |
| New/novel tasks | Established benchmarks exist |
| Ground truth is ambiguous | Clear right/wrong answers |

**The 2026 shift: from crowdworkers to domain experts.** The human-data market moved from commodity labeling to **expert rubric-writing and grading** — physicians authoring HealthBench rubrics, occupational experts doing blind pairwise grading for GDPval ([digest: GDPval](https://openai.com/index/gdpval/)). Human data teams now primarily feed *judge calibration* (golden sets, rubric authorship, disagreement review), not bulk SFT labels. Expert-network firms like Mercor reached a $450M/yr run-rate by mid-2025 supplying exactly this ([industry overview](https://www.herohunt.ai/blog/the-ultimate-ai-data-labeling-industry-overview/)). The crowdsourcing pipeline below still applies, but the scarce resource is the expert who writes the rubric, not the annotator who applies it.

### 2.4.1 Annotation Interface Design

```python
class AnnotationTask:
    """Structure for human annotation tasks"""
    
    def __init__(self, task_config: dict):
        self.config = task_config
    
    def generate_task(self, sample: dict) -> dict:
        return {
            'task_id': sample['id'],
            'instructions': self.config['instructions'],
            'input': sample['input'],
            'model_output': sample['output'],
            'questions': self.config['questions'],
            'metadata': {
                'estimated_time': self.config.get('estimated_time', '2 min'),
                'reward': self.config.get('reward', '$0.10')
            }
        }

# Example annotation config
annotation_config = {
    'instructions': """
        You will be shown a question and an AI-generated answer.
        Please rate the answer on several dimensions.
        Be honest and rigorous in your assessment.
    """,
    'questions': [
        {
            'id': 'accuracy',
            'type': 'likert_5',
            'prompt': 'How accurate is this response?',
            'labels': ['Very inaccurate', 'Inaccurate', 'Neutral', 
                      'Accurate', 'Very accurate']
        },
        {
            'id': 'helpful',
            'type': 'likert_5',
            'prompt': 'How helpful is this response?',
            'labels': ['Not helpful', 'Slightly helpful', 'Neutral',
                      'Helpful', 'Very helpful']
        },
        {
            'id': 'issues',
            'type': 'multi_select',
            'prompt': 'Select all issues that apply:',
            'options': [
                'Factual error',
                'Incomplete answer',
                'Irrelevant content',
                'Poor formatting',
                'Harmful content',
                'None of the above'
            ]
        },
        {
            'id': 'feedback',
            'type': 'free_text',
            'prompt': 'Any additional feedback? (optional)'
        }
    ],
    'estimated_time': '3 min',
    'reward': '$0.25'
}
```

### 2.4.2 Quality Control for Human Annotators

```python
class AnnotationQualityControl:
    """Ensure high-quality human annotations"""
    
    def __init__(self):
        self.gold_standards = []  # Known-answer questions
        self.annotator_stats = {}
    
    def add_gold_standard(self, task: dict, expected_answers: dict):
        """Add a quality check question with known answers"""
        self.gold_standards.append({
            'task': task,
            'expected': expected_answers,
            'tolerance': 0.2  # Allow some variance
        })
    
    def inject_quality_checks(self, task_batch: list, frequency: float = 0.1) -> list:
        """Insert gold standard questions into a batch"""
        import random
        
        enhanced_batch = task_batch.copy()
        num_checks = max(1, int(len(task_batch) * frequency))
        
        for _ in range(num_checks):
            gold = random.choice(self.gold_standards)
            position = random.randint(0, len(enhanced_batch))
            enhanced_batch.insert(position, {
                **gold['task'],
                '_is_gold': True,
                '_expected': gold['expected']
            })
        
        return enhanced_batch
    
    def evaluate_annotator(self, annotator_id: str, responses: list) -> dict:
        """Calculate annotator reliability"""
        gold_responses = [r for r in responses if r.get('_is_gold')]
        
        if not gold_responses:
            return {'reliable': True, 'score': None, 'message': 'No gold standards'}
        
        correct = sum(
            1 for r in gold_responses 
            if self._check_gold_response(r)
        )
        
        accuracy = correct / len(gold_responses)
        
        self.annotator_stats[annotator_id] = {
            'gold_accuracy': accuracy,
            'total_gold': len(gold_responses),
            'reliable': accuracy >= 0.8
        }
        
        return self.annotator_stats[annotator_id]
    
    def _check_gold_response(self, response: dict) -> bool:
        # Compare response to expected with tolerance
        expected = response['_expected']
        for key, expected_value in expected.items():
            actual = response.get(key)
            if isinstance(expected_value, (int, float)):
                if abs(actual - expected_value) > self.gold_standards[0]['tolerance']:
                    return False
            elif actual != expected_value:
                return False
        return True
    
    def calculate_inter_annotator_agreement(self, 
                                            annotations: list, 
                                            method: str = 'krippendorff') -> float:
        """Calculate agreement between multiple annotators"""
        # Group annotations by task
        tasks = {}
        for ann in annotations:
            task_id = ann['task_id']
            if task_id not in tasks:
                tasks[task_id] = []
            tasks[task_id].append(ann)
        
        # Calculate agreement (simplified Cohen's Kappa)
        agreements = []
        for task_id, task_annotations in tasks.items():
            if len(task_annotations) >= 2:
                # Compare first two annotators
                a1, a2 = task_annotations[0], task_annotations[1]
                agreement = sum(
                    1 for k in a1 if k in a2 and a1[k] == a2[k]
                ) / len(a1)
                agreements.append(agreement)
        
        return sum(agreements) / len(agreements) if agreements else 0
```

### 2.4.3 Crowdsourcing Best Practices

```
┌─────────────────────────────────────────────────────────────────┐
│              CROWDSOURCING EVALUATION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. QUALIFICATION                                                │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ • Screening test (10 gold standard questions)       │     │
│     │ • Require >80% accuracy to proceed                  │     │
│     │ • Filter by demographics/expertise if needed        │     │
│     └─────────────────────────────────────────────────────┘     │
│                         │                                        │
│                         ▼                                        │
│  2. CALIBRATION                                                  │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ • Provide detailed examples with explanations       │     │
│     │ • Show edge cases and how to handle them            │     │
│     │ • Allow practice round with feedback                │     │
│     └─────────────────────────────────────────────────────┘     │
│                         │                                        │
│                         ▼                                        │
│  3. ANNOTATION                                                   │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ • 3+ annotators per item (for voting/agreement)     │     │
│     │ • 10% hidden gold standards                         │     │
│     │ • Real-time quality monitoring                      │     │
│     └─────────────────────────────────────────────────────┘     │
│                         │                                        │
│                         ▼                                        │
│  4. AGGREGATION                                                  │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ • Majority voting (simple)                          │     │
│     │ • Weighted voting (by annotator reliability)        │     │
│     │ • MACE/Dawid-Skene (sophisticated)                  │     │
│     └─────────────────────────────────────────────────────┘     │
│                         │                                        │
│                         ▼                                        │
│  5. REVIEW                                                       │
│     ┌─────────────────────────────────────────────────────┐     │
│     │ • Expert review of disagreements                    │     │
│     │ • Identify systematic issues                        │     │
│     │ • Update guidelines if needed                       │     │
│     └─────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.5 Combining Evaluation Methods

### The Evaluation Pyramid

```
                    ▲
                   /│\
                  / │ \          Human Expert Review
                 /  │  \         (Small sample, high stakes)
                /   │   \        
               /────┼────\       
              /     │     \      Human Crowdsource
             /      │      \     (Medium sample, calibration)
            /       │       \    
           /────────┼────────\   
          /         │         \  LLM-as-Judge
         /          │          \ (Large sample, nuanced)
        /           │           \
       /────────────┼────────────\
      /             │             \  Automated Rules
     /              │              \ (All samples, fast)
    ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔

    VOLUME: High ◄───────────────► Low
    COST:   Low  ◄───────────────► High
    DEPTH:  Shallow ◄─────────────► Deep
```

### 2.5.1 Hybrid Evaluation Strategy

```python
class HybridEvaluator:
    """Combine multiple evaluation methods intelligently"""
    
    def __init__(self):
        self.rule_evaluator = RuleBasedEvaluator()
        self.llm_evaluator = LLMJudge()
        self.human_queue = []
    
    def evaluate(self, sample: dict) -> dict:
        results = {}
        
        # Stage 1: Rule-based (fast, cheap)
        rule_results = self.rule_evaluator.evaluate(
            sample['output'],
            sample.get('expected')
        )
        results['rule_based'] = rule_results
        
        # Early exit if critical failure
        if rule_results.get('critical_failure'):
            return {
                'final_score': 0,
                'method': 'rule_based_critical_fail',
                'details': results
            }
        
        # Stage 2: LLM evaluation (if rules pass)
        llm_results = self.llm_evaluator.evaluate(
            question=sample['input'],
            response=sample['output'],
            criteria="accuracy, helpfulness, safety"
        )
        results['llm_based'] = llm_results
        
        # Stage 3: Route to human if uncertain
        if self._needs_human_review(llm_results):
            self.human_queue.append({
                'sample': sample,
                'llm_results': llm_results,
                'reason': 'uncertain_score'
            })
            results['needs_human_review'] = True
        
        return {
            'final_score': llm_results.score,
            'method': 'hybrid',
            'details': results
        }
    
    def _needs_human_review(self, llm_results) -> bool:
        # Low confidence or borderline scores
        return llm_results.score < 0.3 or (0.45 < llm_results.score < 0.55)
```

---

## 2.6 Common Pitfalls and How to Avoid Them

### Pitfall 1: Position Bias in LLM Judges
LLMs tend to prefer responses in certain positions (usually first).

**Solution:** Randomize order and aggregate.

```python
def evaluate_with_position_debiasing(evaluator, question, responses):
    import random
    
    results = []
    for _ in range(3):  # Multiple evaluations
        shuffled = responses.copy()
        random.shuffle(shuffled)
        result = evaluator.compare_list(question, shuffled)
        # Map back to original positions
        results.append(result)
    
    # Aggregate across permutations
    return aggregate_results(results)
```

### Pitfall 2: Self-Enhancement Bias
Models rate their own outputs higher.

**Solution:** Use different models for generation and evaluation.

### Pitfall 3: Style Bias (the verbosity story, revised)
The classic claim was "longer responses get higher scores." The 2026 picture is subtler: a systematic evaluation of bias mitigations ([Judging the Judges, arXiv 2604.23178](https://arxiv.org/html/2604.23178)) found **style/formatting bias is the most robust bias across models (severity 0.76–0.92)** — and, notably, all five tested judge models preferred *concise* responses over padded ones. The bias is toward confident, well-formatted, list-heavy *style*, not raw length.

**Solution:** Anchor rubric criteria to content ("states X", "cites Y"), not presentation; strip or normalize formatting before judging where feasible; check judge agreement separately on style-mismatched pairs.

### Pitfall 4: Annotator Fatigue
Human annotators lose focus over time.

**Solution:** Limit session length, mix in attention checks, vary task difficulty.

### Pitfall 5: Evaluation Awareness
Frontier models increasingly detect that they are being evaluated and behave differently. Apollo Research found GPT-5.5 verbalized awareness of being alignment-evaluated in **22.1% of samples (52% under sandbagging conditions)**, some naming the exact eval type ([GPT-5.5 system card](https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf)). UK AISI found Mythos 5 distinguishes real deployment data from synthetic reproductions **84%** of the time pairwise ([Fable 5 system card §6.2.4](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)). If your scenario smells synthetic, you are measuring "behavior when watched."

**Solution:** Invest in environment realism (real codebases, real tool outputs, plausible stakes); measure eval-awareness alongside the target behavior and report it as a confound — exactly what frontier system cards now do (see 2.7.9).

### Pitfall 6: Your Grader Is a Reward Spec
Graders are converging with training rewards: OpenAI's platform shares the same grader objects between the Evals API and reinforcement fine-tuning ([graders guide](https://developers.openai.com/api/docs/guides/graders)). That means every rubric you write can be Goodharted. Anthropic showed models that learn to reward-hack production coding graders generalize to alignment faking and sabotage ([arXiv 2511.18397](https://arxiv.org/abs/2511.18397)); OpenAI showed that optimizing against a CoT monitor teaches *obfuscated* hacking rather than honest behavior ([chain-of-thought monitoring](https://openai.com/index/chain-of-thought-monitoring/)).

**Solution:** Red-team your rubrics like reward functions: ask "what's the cheapest output that maxes this score?" before a model finds it for you. Keep at least one held-out grader that never touches training.

---

---

## 2.7 Advanced Evaluation Methods (2025-2026 Research)

### 2.7.1 Psychometric Evaluation

Traditional software metrics don't capture the nuances of LLM behavior. The 2026 eval frontier integrates psychometric principles from human assessment science.

```python
class PsychometricEvaluator:
    """
    Apply Item Response Theory (IRT) and Bloom's Taxonomy to LLM evaluation.
    This provides more nuanced capability assessment than simple accuracy.
    """
    
    def __init__(self):
        # Bloom's cognitive hierarchy for question generation
        self.cognitive_levels = {
            "remembering": {"description": "Recall of facts", "weight": 0.10},
            "understanding": {"description": "Explain concepts", "weight": 0.15},
            "applying": {"description": "Use in new situations", "weight": 0.25},
            "analyzing": {"description": "Break down relationships", "weight": 0.25},
            "evaluating": {"description": "Make judgments", "weight": 0.15},
            "creating": {"description": "Produce original work", "weight": 0.10},
        }
    
    def compute_capability_profile(self, results: list) -> dict:
        """
        Instead of a single score, compute a PROFILE of capabilities
        across cognitive levels. This reveals WHERE the model fails,
        not just IF it fails.
        """
        profile = {}
        for level in self.cognitive_levels:
            level_results = [r for r in results if r.get("cognitive_level") == level]
            if level_results:
                profile[level] = {
                    "accuracy": sum(r["correct"] for r in level_results) / len(level_results),
                    "n_items": len(level_results),
                    "avg_difficulty": sum(r.get("difficulty", 3) for r in level_results) / len(level_results)
                }
        
        return profile
    
    def item_discrimination(self, item_results: list) -> float:
        """
        Compute how well an eval item discriminates between
        high-ability and low-ability models.
        
        High discrimination = item is useful for ranking models.
        Low discrimination = item doesn't help differentiate.
        """
        # Split into upper and lower performing groups
        sorted_results = sorted(item_results, key=lambda x: x["total_score"], reverse=True)
        n = len(sorted_results)
        upper = sorted_results[:n//3]
        lower = sorted_results[-(n//3):]
        
        upper_correct = sum(1 for r in upper if r["item_correct"]) / len(upper)
        lower_correct = sum(1 for r in lower if r["item_correct"]) / len(lower)
        
        # Discrimination index
        return upper_correct - lower_correct  # Should be > 0.3 for good items


class AdaptiveEvaluator:
    """
    Adaptive testing: adjust question difficulty based on model performance.
    More efficient than fixed-difficulty test sets.
    
    Concept: Like GRE/GMAT adaptive testing, but for LLMs.
    If model gets easy questions right, skip to harder ones.
    """
    
    def __init__(self, item_bank: list, model):
        self.item_bank = sorted(item_bank, key=lambda x: x["difficulty"])
        self.model = model
        self.ability_estimate = 0.5  # Start at medium
    
    def run_adaptive_eval(self, max_items: int = 50) -> dict:
        """Run an adaptive evaluation that efficiently estimates capability"""
        
        administered = []
        remaining = self.item_bank.copy()
        
        for i in range(min(max_items, len(remaining))):
            # Select item closest to current ability estimate
            next_item = min(
                remaining, 
                key=lambda x: abs(x["difficulty"] - self.ability_estimate)
            )
            remaining.remove(next_item)
            
            # Administer item
            response = self.model.generate(next_item["question"])
            correct = self._check_answer(response, next_item["answer"])
            
            administered.append({
                "item": next_item,
                "response": response,
                "correct": correct,
                "ability_at_time": self.ability_estimate
            })
            
            # Update ability estimate (simplified IRT)
            if correct:
                self.ability_estimate = min(1.0, self.ability_estimate + 0.1 * (1 - self.ability_estimate))
            else:
                self.ability_estimate = max(0.0, self.ability_estimate - 0.1 * self.ability_estimate)
            
            # Early stopping if estimate is stable
            if i > 10 and self._estimate_stable(administered[-10:]):
                break
        
        return {
            "final_ability": self.ability_estimate,
            "items_administered": len(administered),
            "efficiency": len(administered) / len(self.item_bank),
            "trajectory": [(a["ability_at_time"], a["correct"]) for a in administered]
        }
    
    def _check_answer(self, response, expected):
        return expected.lower().strip() in response.lower()
    
    def _estimate_stable(self, recent):
        abilities = [r["ability_at_time"] for r in recent]
        return max(abilities) - min(abilities) < 0.05
```

### 2.7.2 Dynamic Benchmark Evaluation

Static benchmarks are becoming obsolete due to contamination and saturation. Dynamic approaches are the 2026 standard.

```python
class DynamicEvaluator:
    """
    Generate fresh evaluation items for each run.
    Prevents contamination and benchmark gaming.
    """
    
    # Six reframing operations for transforming benchmark items
    REFRAMING_OPS = [
        "paraphrase",           # Same question, different words
        "context_noising",      # Add irrelevant context
        "polarity_reversal",    # "Which IS" → "Which is NOT"
        "question_alternation", # Reverse question/answer roles
        "difficulty_scaling",   # Make harder or easier
        "format_change"         # MCQ → open-ended, etc.
    ]
    
    def generate_dynamic_eval(self, 
                               static_items: list, 
                               reframing_ops: list = None,
                               llm=None) -> list:
        """
        Transform static benchmark items into fresh dynamic items.
        Each run produces different questions testing the same skills.
        """
        ops = reframing_ops or self.REFRAMING_OPS
        dynamic_items = []
        
        for item in static_items:
            # Randomly select reframing operation
            import random
            op = random.choice(ops)
            
            new_item = self._apply_reframing(item, op, llm)
            new_item["original_id"] = item.get("id")
            new_item["reframing_op"] = op
            new_item["dynamic"] = True
            dynamic_items.append(new_item)
        
        return dynamic_items
    
    def _apply_reframing(self, item, operation, llm):
        """Apply a specific reframing operation"""
        prompts = {
            "paraphrase": f"Rephrase this question to test the same skill "
                         f"using completely different wording:\n{item['question']}",
            "context_noising": f"Add 1-2 sentences of plausible but irrelevant context "
                              f"to this question (answer stays the same):\n{item['question']}",
            "polarity_reversal": f"Reverse the polarity of this question "
                                f"(e.g., 'correct' → 'incorrect'):\n{item['question']}",
        }
        
        prompt = prompts.get(operation, prompts["paraphrase"])
        if llm:
            new_question = llm.generate(prompt, temperature=0.7)
            return {**item, "question": new_question}
        return item
```

### 2.7.3 Evaluation-Driven Development (EDD)

The 2026 best practice: embed evaluation as a governing function throughout the entire LLM lifecycle.

```
EVALUATION-DRIVEN DEVELOPMENT WORKFLOW

┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Traditional: code → test → ship → (hope it works)              │
│                                                                  │
│  EDD: eval → code → eval → iterate → eval → ship → eval        │
│                                                                  │
│  Key principles:                                                │
│  1. Write evals BEFORE changing prompts/models (eval-first)     │
│  2. Run evals at EVERY stage (not just before release)          │
│  3. Use evals to GUIDE development decisions                    │
│  4. Monitor evals CONTINUOUSLY in production                    │
│  5. Feed production data back into eval datasets                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

EDD was formalized academically as a process model in [Evaluation-Driven Development and Operations of LLM Agents (arXiv 2411.13768)](https://arxiv.org/abs/2411.13768) — evals as the central artifact across dev *and* ops, because agent behavior is open-ended, probabilistic, and system-shaped. What it looks like in practice in 2026:

- **Golden datasets of ~200–500 examples** built from real production failures, not synthetic guesses.
- **Judges in CI**: e.g., Braintrust's GitHub Action runs evals on every PR, posts score summaries, and blocks merge below thresholds ([braintrust.dev](https://www.braintrust.dev/articles/langsmith-vs-braintrust)).
- **Error-analysis-first methodology** (the dominant applied workflow, per Hamel Husain & Shreya Shankar): open coding of real failures → axial coding into failure modes → judges per failure mode, validated by TPR/TNR against human labels → only then CI gates ([evals FAQ, Jan 2026](https://hamel.dev/blog/posts/evals-faq/evals-faq.pdf)).
- **The flywheel**: production traces → sampled async judging → failure clustering → promote failures to dataset rows → regression-test in CI → repeat ([OpenAI Cookbook evaluation flywheel](https://developers.openai.com/cookbook/examples/evaluation/building_resilient_prompts_using_an_evaluation_flywheel)).

### 2.7.4 RAG-Specific Evaluation (RAGAS metric set)

For Retrieval-Augmented Generation systems, evaluate the retriever and the generator separately, then end-to-end. The de-facto open-source library is [RAGAS](https://docs.ragas.io/).

| Component | Metric | What it measures |
|-----------|--------|------------------|
| **Retriever** | `context_precision` | Of retrieved chunks, fraction relevant to the question |
| **Retriever** | `context_recall` | Of relevant info available, fraction actually retrieved |
| **Retriever** | precision@k, recall@k, MRR, nDCG, hit rate | Classical IR metrics on the candidate list |
| **Generator** | `faithfulness` / `groundedness` | Are claims in the answer supported by the retrieved context? (hallucination check) |
| **Generator** | `answer_relevancy` / `response_relevancy` | Does the answer actually address the question? |
| **End-to-end** | `factual_correctness` | Match against ground truth answer |
| **End-to-end** | `noise_sensitivity` | Does adding irrelevant chunks degrade the answer? |
| **End-to-end** | citation accuracy | Are inline citations real and pointing to the right chunk? |

```python
# Minimal RAGAS example (2026 API)
from ragas import evaluate
from ragas.metrics import (
    Faithfulness, ResponseRelevancy,
    LLMContextPrecisionWithReference, LLMContextRecall,
    NoiseSensitivity,
)

result = evaluate(
    dataset=eval_dataset,  # columns: question, contexts, answer, ground_truth
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        NoiseSensitivity(),
    ],
)
```

**Test-set generation:** RAGAS can synthesize a knowledge-graph-grounded test set from your corpus, including single-hop and multi-hop queries with personas. Treat the generated set as a starting point — *always* hand-curate at least 30–50 items.

### 2.7.5 Agent Evaluation (trajectory + outcome)

Single-turn eval is insufficient for tool-using agents. You need to score the **whole trajectory** (steps, tool calls, intermediate states), not just the final answer.

**The canonical vocabulary** — standardized by Anthropic's [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (Jan 2026) and now used across the industry:

| Term | Meaning |
|------|---------|
| **Task** | Test case + success criteria |
| **Trial** | One stochastic run of a task |
| **Transcript / trajectory** | Full record of a trial, incl. tool calls and reasoning |
| **Outcome** | The actual end-state of the environment — *not* what the agent claims it did |

Three practice rules from the same playbook:

1. **Grade outcomes, not paths.** "Grade what the agent produced, not the path it took" — brittle step-sequence checks fail valid alternative solutions. Trajectory inspection is for *diagnosis*, outcome is for *the score*.
2. **Grader hierarchy:** deterministic/code graders where possible → LLM graders where necessary → humans judiciously (calibration and gold standards).
3. **Read the transcripts. Manual review is non-negotiable.** An internal Anthropic run scored Opus 4.5 at 42% on a benchmark until *harness* bugs were fixed — then 95%, same model. If you never read trajectories, you can't tell a model failure from an eval failure.

**Reliability: pass@k vs pass^k.** Leaderboards report pass@k (≥1 success in k trials). Deployed agents need **pass^k** (*all* k trials succeed) — introduced by tau-bench, which found GPT-4o pass^8 < 25% in retail ([arXiv 2406.12045](https://arxiv.org/abs/2406.12045)):

```python
def pass_at_k(p: float, k: int) -> float:   # P(>=1 of k trials succeeds)
    return 1 - (1 - p) ** k

def pass_hat_k(p: float, k: int) -> float:  # P(all k trials succeed)
    return p ** k

# A "90% agent" is not a 90% agent once users hit it repeatedly:
print(f"{pass_at_k(0.90, 8):.3f}")   # 1.000 — looks superb on a leaderboard
print(f"{pass_hat_k(0.90, 8):.3f}")  # 0.430 — most users see a failure
print(f"{pass_hat_k(0.75, 3):.3f}")  # 0.422 — a 75% agent is a coin flip at k=3
```

Use pass@k for "one success matters" tools (research, brainstorming); pass^k for anything customer-facing where consistency is the product.

| Dimension | Metric / approach |
|-----------|-------------------|
| **Outcome** | Did the agent achieve the goal? (`agent_goal_accuracy` in RAGAS) |
| **Tool selection** | `tool_call_accuracy` — was each call to the right tool? |
| **Tool arguments** | `tool_call_f1` — were the arguments structurally and semantically right? |
| **Topic adherence** | Does the agent stay within the allowed task scope across turns? |
| **Efficiency** | Steps to completion, tokens used, wall-clock time |
| **Recovery** | When a tool errors, does the agent recover or loop? |
| **Safety** | Did the agent attempt unauthorized actions? Probe with adversarial prompts. |

**Sandboxed task benchmarks (2026 standard):**
- [SWE-Bench Verified](https://www.swebench.com/) — real GitHub issues, repo-level coding agents. Saturated at the top (Fable 5: 95.0; Mythos 5: 95.5 — [llm-stats](https://llm-stats.com/benchmarks/swe-bench-verified)); now a regression test, not a discriminator. The discriminating successor is [SWE-bench Pro](https://scale.com/blog/swe-bench-pro) (1,865 long-horizon multi-file tasks with a never-released private commercial split).
- [Terminal-Bench 2.0](https://www.tbench.ai/) — 89 hand-crafted, human-verified end-to-end terminal tasks, each in an isolated Docker container; official harness is [Harbor](https://harborframework.com/docs/running-tbench), which can run Claude Code, Codex CLI, OpenHands, and Mini-SWE-Agent agents.
- [GAIA](https://huggingface.co/spaces/gaia-benchmark/leaderboard) — general assistant tasks across web/file/multimodal
- [τ²-bench](https://github.com/sierra-research/tau2-bench) — successor to τ-Bench: 279 multi-turn dialogues across retail/airline/telecom with **dual control** (both agent and simulated user mutate shared state via tools); success = task completion + policy adherence + database state match. Origin of the pass^k metric. Text domains are near-saturated (Sonnet 4.6: Retail 91.7 / Telecom 97.9 per its [system card](https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf)), so Sierra shifted to [τ-voice](https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks) for full-duplex voice agents.
- [WebArena](https://webarena.dev/) and [OSWorld](https://os-world.github.io/) — browser/desktop interaction. Note: computer-use agents crossed the OSWorld-Verified human baseline (~72%) in early 2026 ([coverage](https://coasty.ai/blog/osworld-benchmark-results-2026-computer-use-ranked)).
- [Cybench](https://cybench.github.io/) — cybersecurity CTF agents. Historically important, but saturated: Anthropic dropped it from the Fable 5 cyber suite in favor of harder internal ranges like ExploitBench ([Fable 5 system card §3.2](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)).
- [SHADE-Arena](https://alignment.anthropic.com/2025/strengthening-red-teams/) — Anthropic's modular control-evaluation scaffold for sabotage/control tests; used in the Fable 5 card's covert-capability evals (§6.5.4).

**Tooling:** [Inspect AI](https://inspect.aisi.org.uk/) is the de-facto framework for agent evals — first-class support for ReAct and multi-agent solvers, Docker/Kubernetes sandboxes, MCP tool integration, and an "agent bridge" that lets you score externally-built agents (Claude Code, Codex CLI, Gemini CLI) inside the Inspect harness.

### 2.7.6 Reasoning-Model Evaluation (CoT faithfulness + monitorability)

Reasoning models (GPT-5.5 thinking, Claude with extended/adaptive thinking, Gemini 3.1 Pro, and — historically the open-weights breakthrough — DeepSeek-R1) emit an explicit thinking trace before the final answer. This opens up a class of evals that simply did not exist before:

- **Outcome accuracy** — the usual answer-correctness metric.
- **CoT faithfulness** — does the chain actually reflect the computation that produced the answer, or is it post-hoc rationalization? Probe by perturbing the chain and checking whether the final answer changes coherently. (See Anthropic's [Reasoning Models Don't Always Say What They Think](https://www.anthropic.com/research/reasoning-models-dont-always-say-what-they-think).)
- **CoT monitorability — now a *reported safety property* in system cards.** Concrete examples of the genre:
  - OpenAI measures deception with a **CoT monitor over reasoning traces** (precision 81%, recall 84%); it flagged ~2.1% of gpt-5-thinking responses vs ~4.8% for o3 ([GPT-5 system card](https://arxiv.org/html/2601.03267v1)).
  - DeepMind's Gemini 3 Pro FSF report graded **99.98%** of sampled reasoning transcripts as comprehensible, using metrics from Emmons et al. 2025 ([FSF report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf)).
  - The Fable 5 card reports the opposite direction honestly: Mythos 5 is "likely harder to monitor than recent Claude Opus models" — its reasoning is denser, jargon-heavy, occasionally "almost completely illegible," and "technical errors led to accidental chain-of-thought supervision during training" of a small fraction of episodes ([§6.5.5](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)).
  - The standing caution: training *against* a CoT monitor teaches obfuscated misbehavior, not honesty — keep CoTs unrestricted and monitorable ([OpenAI](https://openai.com/index/chain-of-thought-monitoring/)).
- **Process supervision** — score every reasoning step (PRM-style), not just the final answer. Catches models that get the right answer for the wrong reason.
- **Reasoning-effort trade-off** — sweep `reasoning_effort` (low/medium/high) and plot accuracy vs. tokens vs. latency. Most production tasks plateau well below "high". (Frontier cards report effort settings for the same reason: Fable 5 capability numbers are "adaptive thinking at max effort, averaged over 5 trials" — a score without its effort setting is meaningless.)
- **Hidden-CoT integrity** — for models that hide raw CoT from users (OpenAI reasoning models), evaluate the *summary* shown to the user for fidelity to the underlying chain.
- **Reasoning leakage** — does the model accidentally reveal evaluation hints, system prompt, or tool outputs in its visible CoT?

### 2.7.7 Tooling Pointer (2026 Stack)

| Need | Pick |
|------|------|
| Safety / capability / agent benchmarks, sandboxed | [Inspect AI](https://inspect.aisi.org.uk/) |
| Hosted offline + CI + online (production) scoring | [Braintrust](https://www.braintrust.dev/docs/evaluate), [LangSmith](https://docs.langchain.com/langsmith/evaluation) |
| Open-source production observability | [Arize Phoenix](https://phoenix.arize.com/), [Langfuse](https://langfuse.com/), [Helicone](https://www.helicone.ai/) |
| Experiment tracking + LLM traces | [W&B Weave](https://wandb.ai/site/weave) |
| RAG and agent metrics | [RAGAS](https://docs.ragas.io/), [TruLens](https://www.trulens.org/) |
| Pytest-style assertions for LLMs | [DeepEval](https://github.com/confident-ai/deepeval) — note its [DAG metric](https://deepeval.com/docs/metrics-conversational-dag): deterministic LLM decision trees with hard-coded leaf scores |
| Judge calibration vs. human labels | [LangSmith Align Evals](https://blog.langchain.com/introducing-align-evals/) |
| Fast prompt A/B in YAML; red-team probes | [Promptfoo](https://www.promptfoo.dev/) (acquisition by OpenAI announced Mar 2026; remains open source — [announcement](https://openai.com/index/openai-to-acquire-promptfoo/)) |
| Reference framework, model registry | [OpenAI Evals](https://github.com/openai/evals) |
| Adversarial / red-team probes | [Garak](https://github.com/NVIDIA/garak), [PyRIT](https://github.com/Azure/PyRIT) |
| Tracing standard | [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), [OpenLLMetry](https://github.com/traceloop/openllmetry) |

### 2.7.8 Worked Examples

Three end-to-end snippets that put the 2026 patterns above into practice.

#### Example A — Pairwise judge with position-bias control + Cohen's κ

```python
# pip install anthropic scikit-learn
import random, json
from anthropic import Anthropic
from sklearn.metrics import cohen_kappa_score

client = Anthropic()

JUDGE_PROMPT = """You are comparing two assistant responses to the same user message.
Return ONLY JSON: {{"winner": "A" | "B" | "tie", "reason": "..."}}.

User message:
{user}

Response A:
{a}

Response B:
{b}"""

def judge_once(user, a, b):
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user",
                   "content": JUDGE_PROMPT.format(user=user, a=a, b=b)}],
    )
    return json.loads(msg.content[0].text)["winner"]

def pairwise(user, a, b):
    """Run BOTH orderings; only count agreements as a real win."""
    forward = judge_once(user, a, b)              # A in slot 1
    reverse = judge_once(user, b, a)              # A in slot 2
    # 'A' wins forward maps to 'B' wins reverse — flip and require agreement
    flipped = {"A": "B", "B": "A", "tie": "tie"}[reverse]
    if forward == flipped:
        return forward                            # consistent verdict
    return "tie"                                  # position-sensitive → tie

# Compare against human labels collected in your annotation tool
human   = ["A", "A", "tie", "B", "A", "B"]
judge   = [pairwise(*row) for row in eval_rows]   # eval_rows = [(user,a,b), ...]
print("Cohen's κ vs humans:", cohen_kappa_score(human, judge))
# Rule of thumb: κ ≥ 0.6 = ship the judge; 0.4–0.6 = iterate prompt; <0.4 = redesign.
```

#### Example B — Minimal RAG eval with RAGAS

```python
# pip install ragas datasets langchain-openai
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextRecall

data = Dataset.from_list([{
    "question": "What is the refund window for opened electronics?",
    "contexts": [
        "Electronics may be returned within 30 days if unopened.",
        "Opened electronics are eligible for store credit within 14 days.",
    ],
    "answer":       "You can get store credit within 14 days for opened electronics.",
    "ground_truth": "Opened electronics qualify for store credit within 14 days.",
}])

result = evaluate(data, metrics=[Faithfulness(),
                                 ResponseRelevancy(),
                                 LLMContextRecall()])
print(result.to_pandas())
# faithfulness  answer_relevancy  context_recall
#         1.00              0.93            1.00
# Interpret: answer is grounded (1.0), addresses the question (0.93),
# and the retrieved context contained the gold info (1.0).
```

#### Example C — Agent trajectory eval with Inspect AI (sandboxed)

```python
# pip install inspect-ai && inspect eval refund_agent.py --model anthropic/claude-sonnet-4-6
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import use_tools, generate, system_message
from inspect_ai.tool import tool
from inspect_ai.scorer import scorer, accuracy, Score, Target

@tool
def issue_refund():
    async def execute(order_id: str, amount: float) -> str:
        """Issue a refund. amount must be <= order total."""
        return f"refunded ${amount:.2f} for {order_id}"
    return execute

@scorer(metrics=[accuracy()])
def trajectory_scorer():
    """Score = 1 only if the agent called issue_refund with the right args
    AND the final user-facing message confirms the refund."""
    async def score(state, target: Target):
        calls = [m for m in state.messages if m.role == "tool"]
        used_refund = any("refunded" in (c.text or "") for c in calls)
        confirmed   = "refund" in state.output.completion.lower()
        ok = used_refund and confirmed
        return Score(value=1.0 if ok else 0.0,
                     explanation=f"used_refund={used_refund} confirmed={confirmed}")
    return score

@task
def refund_agent():
    return Task(
        dataset=[Sample(input="I want a refund on order A-42 for $19.99",
                        target="refund issued")],
        solver=[system_message("You are a support agent. Use tools."),
                use_tools(issue_refund()), generate()],
        scorer=trajectory_scorer(),
        sandbox="docker",   # <— isolates each task; safe for tool execution
    )
```

Run with `inspect view` to see per-sample traces, tool calls, and aggregate accuracy. The same harness scales to SWE-Bench Verified, GAIA, τ²-bench, etc., without code changes — swap the dataset and scorer.

### 2.7.9 Case Study: How Frontier System Cards Grade Behavior (Fable 5 / Opus 4.x)

Every method in this module shows up, industrialized, in how Anthropic evaluated Claude Fable 5 / Mythos 5 (June 2026) and the Opus 4.x line. The [Fable 5 system card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) is 319 pages, and its alignment assessment is essentially a giant LLM-as-judge pipeline with human triage. Worth studying as the most mature public example of model-graded evaluation.

#### The automated behavioral audit (card §6.2.3)

```
┌──────────────────────────────────────────────────────────────────────┐
│        AUTOMATED BEHAVIORAL AUDIT (Fable 5 / Mythos 5 pattern)       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ~1,450 seed scenarios (largely hand-written)                        │
│        │                                                             │
│        ▼                                                             │
│  ~2,900 INVESTIGATION SESSIONS per model under study                │
│        │                                                             │
│        ▼                                                             │
│  INVESTIGATOR agents          ──probe──▶   SUBJECT (model under     │
│  • helpful-only Opus 4.8                    study, e.g. Mythos 5)   │
│    (strong at eliciting misuse)                                      │
│  • Mythos 5 itself                                                   │
│    (strong at simulating ordinary use)                               │
│  Affordances: set system prompts, simulate tools, prefill,           │
│  rewind, drive REAL sandboxed computer-use & Claude Code             │
│  sessions seeded with internal codebases                             │
│        │                                                             │
│        ▼                                                             │
│  JUDGE (Mythos Preview) scores every transcript on a FIXED          │
│  set of several-dozen behavioral dimensions, applied to ALL         │
│  transcripts regardless of subject (lower = better)                 │
│        │                                                             │
│        ▼                                                             │
│  Humans review flagged tails + aggregate dashboards                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

Note the **strict three-role separation** — investigator ≠ subject ≠ judge, with different models in each seat. That is the self-enhancement-bias fix from 2.6 applied at frontier scale. The judge applies the *same* fixed set of several-dozen behavioral dimensions to every transcript regardless of subject, so the card cautions that absolute scores reflect the seed-scenario mix and are "valuable primarily for comparing models."

#### Petri: the audit pattern, open-sourced and cross-lab

[Petri](https://alignment.anthropic.com/2025/petri/) ("Parallel Exploration Tool for Risky Interactions") packages the auditor-agent + judge pattern as open source ([repo](https://github.com/safety-research/petri)). First released Oct 2025 against 14 frontier models with 111 seed instructions; now maintained by the independent nonprofit Meridian Labs. The Fable 5 card reports **Petri 3.0** runs by Meridian + UK AISI: 173 seed instructions, six graded metrics (misaligned behavior, verbalized evaluation awareness, cooperation with human misuse, deception toward the user, sycophancy, encouragement of user delusion), with each transcript scored by **three judge models from different providers** (Opus 4.7, GPT-5.5, Gemini 3.1 Pro) and the three scores averaged ([card §6.2.3.3](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)). Cross-provider judging is what makes a cross-lab comparison credible — no single lab's judge gets to referee its own model.

#### Model-graded triage: the sandbagging sweep (card §6.5.2)

The cleanest worked example of "LLM judge as triage, human as final arbiter" — the evaluation pyramid from 2.5, verbatim:

1. Sample **2,440 dangerous-capability eval transcripts** (240 automated AI R&D + 1,700 CB on a helpful-only variant + 500 cyber). Elicitation uses **helpful-only model variants** (harmlessness trained away) so refusal-based underperformance can't masquerade as incapability.
2. Ask a judge (Mythos Preview) to grade each transcript **0–10** for refusal, deliberate inaccuracy, low helpfulness, or consideration of such actions, plus a generic "anything unusual/concerning" score.
3. Human review triggers on thresholds: sandbagging > 1/10 or concerningness > 3/10. **Only 9 transcripts crossed them**, all manually classified (none were concerning sandbagging).

2,440 transcripts, 9 human reviews. That ratio — judge breadth, human depth — is the economics of the whole field.

#### Evaluation awareness, measured four ways (card §6.5.1)

Because eval-aware models corrupt behavioral evals (Pitfall 5), the card measures awareness itself with deliberately redundant instruments: (1) steering/probe vectors built from contrastive prompt pairs ("Testing vs. Conversation", "Monitoring Sensitivity", an explicit "Alignment Faking" vector); (2) interpretability-graded awareness over sampled tokens; (3) the model's own follow-up self-rating of "synthetic vs. real" (0–1); (4) spontaneous verbalized awareness. All measures rise with scenario "suspiciousness" (itself 1–10 graded by Mythos Preview) but are only **weakly correlated with each other** — eval awareness is a multifaceted construct, and one metric would have been a lie of precision.

#### What the Opus 4.x cards add

The [Opus 4.8 card](https://www-cdn.anthropic.com/0b4915911bb0d19eca5b5ee635c80fef830a37ea.pdf) grades *propensities*, not just abilities: reckless/destructive actions and over-refusals "substantially reduced" vs. Opus 4.7, agentic honesty "markedly improved" — operationalized as a much lower tendency to fail to report flaws in its own code (the launch blog quantifies "around four times less likely than its predecessor to allow flaws in code it has written to pass unremarked" — [blog](https://www.anthropic.com/news/claude-opus-4-8)). The Fable 5 card's targeted honesty suite (§6.3) shows what a behavior-grading battery looks like: factual honesty, false-premise questions, MASK, missing-context, lying-about-identity, code-summary honesty, lazy-investigation and overconfidence checks — each a separately graded eval, not one "honesty score."

#### Lessons for your own eval stack

1. **Three roles, three models** — investigator, subject, judge are never the same model.
2. **Anchored numeric rubrics** (0–10 with a written definition per dimension), never a generic "rate the quality."
3. **Judge for breadth, humans for the flagged tail** — define numeric triage thresholds *before* running.
4. **Cross-provider judge panels** when the result needs to be credible to outsiders.
5. **Measure eval-awareness as a first-class confound**, with multiple instruments.
6. **Elicit with refusals removed** when measuring capability ceilings, or refusals masquerade as incapability.

---

## 2.8 Exercises

### Exercise 1: Build a Multi-Modal Evaluator
Create an evaluation system that:
- Uses regex for format validation
- Uses code execution for functional testing
- Uses LLM-as-judge for style/quality
- Combines scores with weighted average

### Exercise 2: Position Bias Experiment
Run an experiment showing position bias in an LLM judge. Measure the effect and implement a debiasing strategy.

### Exercise 3: Design Human Annotation Guidelines
Create comprehensive annotation guidelines for evaluating customer service chatbot responses. Include:
- Clear rubric
- Example ratings with explanations
- Edge cases and how to handle them

### Exercise 4: Psychometric Eval Design (New)
Design an evaluation using Bloom's taxonomy:
- Create items at each of the 6 cognitive levels for a chosen domain
- Compute a capability profile (not just accuracy)
- Use adaptive testing to efficiently estimate model ability

### Exercise 5: Dynamic Benchmark (New)
Take 20 items from a public benchmark and:
- Apply all 6 reframing operations
- Compare model scores on original vs reframed items
- Report the "contamination discount" (how much scores drop)

### Exercise 6: Per-Example Rubric Grader (New)
Pick 10 prompts from a domain you know well and, HealthBench-style:
- Write 5-10 weighted criteria *per prompt* (include at least one negative weight)
- Grade with one isolated judge call per criterion (code in 2.3.4)
- Human-label the same outputs, then compute Cohen's κ between judge and yourself
- Iterate the judge prompt until κ ≥ 0.6 on a held-out half

### Exercise 7: Agent Reliability Audit (New)
Take one agent task (e.g., the Inspect refund agent from Example C):
- Run it k=10 times; report pass@10 and pass^10
- Read all 10 transcripts and classify each failure as model failure vs harness failure
- Triage like the Fable 5 sandbagging sweep: define a numeric judge threshold first, then have a judge flag transcripts that cross it, and manually review only the flagged ones

---

## Next Module
-> [Module 3: Building Evaluation Pipelines](../03-pipeline-architecture/README.md)

