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
│   │ Judge LLM (GPT-4, Claude, etc.)             │              │
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

### 2.3.1 Single Judge Implementation

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

class EvalResult(BaseModel):
    score: float
    reasoning: str
    strengths: list[str]
    weaknesses: list[str]

class LLMJudge:
    """Single LLM evaluator"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
    
    def evaluate(self, 
                 question: str, 
                 response: str, 
                 criteria: str,
                 reference: Optional[str] = None) -> EvalResult:
        
        reference_section = f"\nReference Answer: {reference}" if reference else ""
        
        prompt = f"""You are an expert evaluator. Assess the following response.

Question: {question}
Response: {response}{reference_section}

Evaluation Criteria: {criteria}

Provide your evaluation in the following JSON format:
{{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your score>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"]
}}

Be rigorous and objective. A score of 0.5 means adequate, 0.7 means good, 0.9+ means excellent.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result_json = json.loads(response.choices[0].message.content)
        return EvalResult(**result_json)

# Example usage
judge = LLMJudge()
result = judge.evaluate(
    question="Explain quantum entanglement to a 10-year-old",
    response="Quantum entanglement is like having two magic coins that always land the same way, no matter how far apart they are!",
    criteria="Accuracy, age-appropriateness, clarity, engagement"
)
print(f"Score: {result.score}")
print(f"Reasoning: {result.reasoning}")
```

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

# Example: Panel of diverse judges
panel = MultiJudgePanel([
    LLMJudge(model="gpt-4o"),
    LLMJudge(model="claude-3-opus-20240229"),
    LLMJudge(model="gpt-4o-mini")  # Different perspective
])
```

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
    
    def __init__(self, model: str = "gpt-4o"):
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
    """Detailed rubric-based evaluation with specific criteria"""
    
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
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
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

### Pitfall 3: Verbosity Bias
Longer responses often get higher scores regardless of quality.

**Solution:** Explicitly penalize unnecessary length or use length-normalized metrics.

### Pitfall 4: Annotator Fatigue
Human annotators lose focus over time.

**Solution:** Limit session length, mix in attention checks, vary task difficulty.

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
│  Result: 47x faster regression detection vs manual QA           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

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

---

## Next Module
-> [Module 3: Building Evaluation Pipelines](../03-pipeline-architecture/README.md)

