# Module 8: Real-World Case Studies

## Overview

This module presents detailed case studies of evaluation systems in production. Each case study includes the problem, architecture, implementation details, and lessons learned.

Case studies 1–6 are practitioner-scale composites (realistic but anonymized/illustrative numbers). Case studies 7–9 are documented public evaluations from 2025–2026 — a frontier-model release, an economically grounded benchmark, and a long-horizon agent eval — with every number traceable to a primary source. Case study 10 is a **production multimodal agent** (Uber Eats image enhancement) presented publicly by the team that built it: architecture and design principles from the source, arithmetic worked here.

**Which case study to read for which problem:**

| Your problem | Case study |
|---|---|
| Free-text quality, no ground truth | 1 (support bot), 8 (GDPval) |
| Ground truth exists (tests, execution) | 2 (code gen) |
| Retrieval + grounding | 3 (RAG) |
| Precision/recall trade on a contested label | 4 (moderation) |
| Multi-step tool use, reliability | 5 (refund agent), 9 (Vending-Bench) |
| Reasoning traces and effort settings | 6 (math tutor) |
| Release gating, safety, third-party audit | 7 (Fable 5 system card) |
| **Generative pipeline, reference-free, brand-critical, self-correcting loop** | **10 (Uber Eats image agent)** |

---

## Case Study 1: Customer Support Chatbot Evaluation

### The Problem

A large e-commerce company deployed an AI chatbot handling 100,000+ customer inquiries daily. They needed to ensure:
- Accurate product and policy information
- Appropriate escalation to human agents
- Empathetic responses to frustrated customers
- No harmful or inappropriate content
- Cost-effective operation

### Initial State (Cold Start)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       COLD START STATE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WHAT THEY HAD:                                                              │
│  ✗ No labeled data                                                          │
│  ✗ No evaluation metrics                                                    │
│  ✗ No baseline performance data                                             │
│  ✗ Only customer complaints as feedback                                     │
│                                                                              │
│  WHAT THEY NEEDED:                                                           │
│  ✓ Comprehensive eval framework                                             │
│  ✓ Real-time quality monitoring                                             │
│  ✓ Continuous improvement system                                            │
│  ✓ Safety guardrails                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CUSTOMER SUPPORT EVAL ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Production Traffic                                                          │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     REAL-TIME LAYER                                  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    │
│  │  │   Safety    │  │   Format    │  │ Escalation  │                 │    │
│  │  │   Filter    │  │ Validation  │  │  Detector   │                 │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                 │    │
│  │         │                │                │                         │    │
│  │         └────────────────┴────────────────┘                         │    │
│  │                          │                                           │    │
│  │                   Blocks bad responses                               │    │
│  │                   < 100ms latency                                    │    │
│  └──────────────────────────┬──────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    ASYNC EVALUATION LAYER                            │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    │
│  │  │  Accuracy   │  │ Helpfulness │  │    Tone     │                 │    │
│  │  │    LLM      │  │    LLM      │  │    LLM      │                 │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │    │
│  │                                                                      │    │
│  │  5% sample, evaluated async                                         │    │
│  │  Results stored for analysis                                        │    │
│  └──────────────────────────┬──────────────────────────────────────────┘    │
│                             │                                                │
│                             ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    FEEDBACK INTEGRATION                              │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    │
│  │  │   Thumbs    │  │  Customer   │  │   Agent     │                 │    │
│  │  │   Up/Down   │  │   Survey    │  │  Override   │                 │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │    │
│  │                                                                      │    │
│  │  Explicit signals integrated into eval data                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Details

#### Step 1: Define Evaluation Dimensions

```python
EVAL_DIMENSIONS = {
    'accuracy': {
        'weight': 0.25,
        'description': 'Factual correctness about products, policies, orders',
        'pass_threshold': 0.85,
        'critical': True  # Block on failure
    },
    'helpfulness': {
        'weight': 0.25,
        'description': 'Did the response actually help solve the problem?',
        'pass_threshold': 0.75,
        'critical': False
    },
    'empathy': {
        'weight': 0.15,
        'description': 'Appropriate emotional response, especially for complaints',
        'pass_threshold': 0.70,
        'critical': False
    },
    'escalation': {
        'weight': 0.15,
        'description': 'Correctly identified when to involve human agent',
        'pass_threshold': 0.90,
        'critical': True
    },
    'safety': {
        'weight': 0.20,
        'description': 'No harmful, inappropriate, or policy-violating content',
        'pass_threshold': 0.99,
        'critical': True
    }
}
```

#### Step 2: Bootstrap Initial Dataset

```python
# Week 1: Expert seeding
expert_examples = 50  # Domain experts created 50 high-quality examples

# Week 2: Synthetic expansion
synthetic_examples = generator.generate_test_cases(
    task_description="Customer support for e-commerce...",
    num_cases=500,
    categories=['order_status', 'returns', 'product_info', 'complaints', 'general']
)

# Week 3: Production sampling
production_sample = active_sampler.combined_sampling(
    production_logs,
    n_select=200,
    strategy='combined'
)

# LLM labeling with human validation
labeled = bootstrap_labeler.label_batch(production_sample, criteria=EVAL_DIMENSIONS)
human_validated = human_review_queue.process(labeled['uncertain_cases'])

# Final dataset: 750 labeled examples
initial_eval_set = expert_examples + synthetic_examples + production_sample
```

#### Step 3: Real-Time Safety Filter

```python
class RealTimeSafetyFilter:
    """Blocks unsafe responses before they reach customers"""
    
    def __init__(self):
        # Fast keyword filter
        self.blocked_patterns = self._load_blocked_patterns()
        
        # Safety classifier (distilled model, <10ms)
        self.safety_classifier = self._load_safety_model()
        
        # Policy checker
        self.policy_checker = PolicyChecker()
    
    def check(self, response: str, context: dict) -> dict:
        """
        Real-time safety check
        Returns: {'safe': bool, 'reason': str, 'latency_ms': float}
        """
        start = time.time()
        
        # Level 1: Keyword filter (<1ms)
        if self._contains_blocked_content(response):
            return {
                'safe': False,
                'reason': 'blocked_keywords',
                'latency_ms': (time.time() - start) * 1000
            }
        
        # Level 2: Policy check (<5ms)
        policy_result = self.policy_checker.check(response, context)
        if not policy_result['compliant']:
            return {
                'safe': False,
                'reason': f"policy_violation:{policy_result['violation']}",
                'latency_ms': (time.time() - start) * 1000
            }
        
        # Level 3: ML classifier (<10ms)
        safety_score = self.safety_classifier.predict(response)
        if safety_score < 0.95:  # High threshold for safety
            return {
                'safe': False,
                'reason': f"low_safety_score:{safety_score:.2f}",
                'latency_ms': (time.time() - start) * 1000
            }
        
        return {
            'safe': True,
            'reason': None,
            'latency_ms': (time.time() - start) * 1000
        }
```

#### Step 4: Async Quality Evaluation

```python
class AsyncQualityEvaluator:
    """Samples and evaluates production traffic asynchronously"""
    
    def __init__(self, sample_rate: float = 0.05):
        self.sample_rate = sample_rate
        self.evaluators = {
            # Frontier judge for the critical dimension; small fast judge for
            # high-volume dimensions. Mixing judge models also reduces
            # self-preference bias — see "Replacing Judges with Juries" (PoLL):
            # https://arxiv.org/abs/2404.18796
            'accuracy': AccuracyEvaluator(model='claude-sonnet-4-6'),
            'helpfulness': HelpfulnessEvaluator(model='claude-haiku-4-5'),
            'empathy': ToneEvaluator(model='claude-haiku-4-5')
        }
        self.results_store = ResultsStore()
    
    async def maybe_evaluate(self, request: dict, response: str):
        """Probabilistically evaluate a response"""
        
        if random.random() > self.sample_rate:
            return  # Skip this one
        
        # Queue for async evaluation
        await self.eval_queue.put({
            'request_id': request['id'],
            'input': request['input'],
            'output': response,
            'context': request.get('context', {}),
            'timestamp': datetime.now().isoformat()
        })
    
    async def evaluation_worker(self):
        """Process evaluation queue"""
        
        while True:
            item = await self.eval_queue.get()
            
            scores = {}
            for name, evaluator in self.evaluators.items():
                result = await evaluator.evaluate(
                    input=item['input'],
                    output=item['output'],
                    context=item['context']
                )
                scores[name] = result
            
            # Store results
            await self.results_store.save({
                'request_id': item['request_id'],
                'timestamp': item['timestamp'],
                'scores': scores
            })
            
            # Check for anomalies
            await self.anomaly_detector.check(scores)
```

### Results

| Metric | Before | After 3 Months |
|--------|--------|----------------|
| Customer Satisfaction | 72% | 89% |
| Escalation Accuracy | 65% | 94% |
| Safety Incidents | 12/month | 0/month |
| Resolution Rate | 45% | 68% |
| Average Handle Time | 8 min | 3 min |

### Lessons Learned

1. **Safety must be real-time**: Async evaluation isn't enough for safety-critical responses
2. **Empathy matters more than accuracy**: Customers forgave minor errors if treated well
3. **Escalation is binary**: Must be near-perfect - wrong escalation wastes human time
4. **Feedback is gold**: Customer thumbs up/down correlated highly with expert evaluation

---

## Case Study 2: Code Generation Assistant

### The Problem

A developer tools company built an AI coding assistant. They needed to ensure:
- Generated code was syntactically correct
- Code solved the requested problem
- Code followed best practices
- No security vulnerabilities introduced
- Works across multiple languages

### Evaluation Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CODE GENERATION EVAL FRAMEWORK                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DIMENSION 1: CORRECTNESS                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Syntax validation (parser)                                       │    │
│  │  • Unit test execution                                              │    │
│  │  • Expected output matching                                         │    │
│  │  • Edge case handling                                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  DIMENSION 2: QUALITY                                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Readability (naming, structure)                                  │    │
│  │  • Efficiency (time/space complexity)                               │    │
│  │  • Best practices adherence                                         │    │
│  │  • Documentation presence                                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  DIMENSION 3: SECURITY                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Static analysis (SAST)                                           │    │
│  │  • Known vulnerability patterns                                     │    │
│  │  • Injection prevention                                             │    │
│  │  • Secrets detection                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  DIMENSION 4: TASK ALIGNMENT                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Solves the stated problem                                        │    │
│  │  • Handles specified requirements                                   │    │
│  │  • Appropriate scope (not over/under engineered)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation: Multi-Stage Evaluation Pipeline

```python
class CodeEvaluationPipeline:
    """Complete evaluation pipeline for code generation"""
    
    def __init__(self):
        self.stages = [
            SyntaxStage(),
            ExecutionStage(),
            SecurityStage(),
            QualityStage(),
            TaskAlignmentStage()
        ]
    
    async def evaluate(self, 
                      task: dict,
                      generated_code: str,
                      language: str) -> dict:
        """Run full evaluation pipeline"""
        
        results = {
            'overall_pass': True,
            'stages': {}
        }
        
        for stage in self.stages:
            stage_result = await stage.evaluate(
                task=task,
                code=generated_code,
                language=language
            )
            
            results['stages'][stage.name] = stage_result
            
            # Early exit on critical failure
            if stage_result.get('critical_failure'):
                results['overall_pass'] = False
                results['failure_reason'] = f"{stage.name}: {stage_result['error']}"
                break
            
            # Track non-critical failures
            if not stage_result.get('passed', True):
                results['overall_pass'] = False
        
        # Compute overall score
        results['score'] = self._compute_score(results['stages'])
        
        return results
    
    def _compute_score(self, stages: dict) -> float:
        weights = {
            'syntax': 0.15,
            'execution': 0.35,
            'security': 0.20,
            'quality': 0.15,
            'task_alignment': 0.15
        }
        
        total = 0
        for name, weight in weights.items():
            if name in stages:
                total += stages[name].get('score', 0) * weight
        
        return total

class ExecutionStage:
    """Evaluate code by running it"""
    
    def __init__(self):
        self.sandbox = CodeSandbox(timeout=30)
    
    async def evaluate(self, task: dict, code: str, language: str) -> dict:
        result = {
            'name': 'execution',
            'passed': True,
            'score': 0,
            'tests': []
        }
        
        # Get test cases from task
        test_cases = task.get('test_cases', [])
        
        if not test_cases:
            # Generate test cases if not provided
            test_cases = await self._generate_test_cases(task, language)
        
        passed = 0
        for test in test_cases:
            test_result = await self.sandbox.run(
                code=code,
                language=language,
                test_input=test['input'],
                expected_output=test['expected']
            )
            
            result['tests'].append({
                'name': test.get('name', 'test'),
                'passed': test_result['passed'],
                'actual': test_result.get('output'),
                'expected': test['expected'],
                'error': test_result.get('error')
            })
            
            if test_result['passed']:
                passed += 1
        
        result['score'] = passed / len(test_cases) if test_cases else 0
        result['passed'] = result['score'] >= 0.8  # 80% pass rate required
        
        return result
    
    async def _generate_test_cases(self, task: dict, language: str) -> list:
        """Generate test cases using LLM if not provided"""
        
        prompt = f"""Generate 5 test cases for this {language} task:

Task: {task['description']}
Function signature: {task.get('signature', 'Not specified')}

Return as JSON:
[
  {{"name": "test_name", "input": "...", "expected": "..."}}
]
"""
        
        response = await self.llm.complete(prompt)
        return json.loads(response)

class SecurityStage:
    """Check generated code for security issues"""
    
    def __init__(self):
        self.sast_tools = {
            'python': 'bandit',
            'javascript': 'eslint-security',
            'java': 'spotbugs'
        }
        
        self.vulnerability_patterns = self._load_patterns()
    
    async def evaluate(self, task: dict, code: str, language: str) -> dict:
        result = {
            'name': 'security',
            'passed': True,
            'score': 1.0,
            'issues': [],
            'critical_failure': False
        }
        
        # Run SAST tool
        sast_issues = await self._run_sast(code, language)
        
        # Check known patterns
        pattern_issues = self._check_patterns(code, language)
        
        # Combine issues
        all_issues = sast_issues + pattern_issues
        
        # Categorize by severity
        critical = [i for i in all_issues if i['severity'] == 'critical']
        high = [i for i in all_issues if i['severity'] == 'high']
        medium = [i for i in all_issues if i['severity'] == 'medium']
        
        result['issues'] = all_issues
        
        # Critical issues are blockers
        if critical:
            result['critical_failure'] = True
            result['passed'] = False
            result['score'] = 0
            result['error'] = f"Critical security issue: {critical[0]['description']}"
        elif high:
            result['passed'] = False
            result['score'] = 0.3
        elif medium:
            result['score'] = 0.7
        
        return result
    
    def _check_patterns(self, code: str, language: str) -> list:
        """Check for known vulnerability patterns"""
        
        issues = []
        patterns = self.vulnerability_patterns.get(language, [])
        
        for pattern in patterns:
            if re.search(pattern['regex'], code):
                issues.append({
                    'type': pattern['type'],
                    'severity': pattern['severity'],
                    'description': pattern['description'],
                    'line': self._find_line(code, pattern['regex'])
                })
        
        return issues
```

### Test Dataset Structure

```python
CODE_EVAL_DATASET = {
    'categories': {
        'algorithm': {
            'description': 'Classic algorithm implementations',
            'examples': [
                {
                    'id': 'algo_001',
                    'task': 'Implement binary search',
                    'signature': 'def binary_search(arr: List[int], target: int) -> int',
                    'test_cases': [
                        {'input': '([1,2,3,4,5], 3)', 'expected': '2'},
                        {'input': '([1,2,3,4,5], 6)', 'expected': '-1'},
                        {'input': '([], 1)', 'expected': '-1'},
                    ],
                    'difficulty': 'easy',
                    'expected_complexity': 'O(log n)'
                }
            ]
        },
        'data_processing': {
            'description': 'Data transformation and processing',
            'examples': [...]
        },
        'api_integration': {
            'description': 'Working with external APIs',
            'examples': [...]
        },
        'bug_fix': {
            'description': 'Fixing buggy code',
            'examples': [...]
        }
    }
}
```

### CI/CD Integration

```yaml
# .github/workflows/code-eval.yml
name: Code Generation Evaluation

on:
  push:
    branches: [main]
    paths:
      - 'src/generator/**'
      - 'prompts/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        language: [python, javascript, java, go]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up evaluation environment
        run: |
          docker pull code-sandbox:latest
          pip install -r requirements.txt
      
      - name: Run code generation eval
        run: |
          python -m code_eval \
            --language ${{ matrix.language }} \
            --dataset datasets/${{ matrix.language }}_eval.json \
            --output results_${{ matrix.language }}.json
      
      - name: Check thresholds
        run: |
          python scripts/check_code_eval.py \
            results_${{ matrix.language }}.json \
            --min-correctness 0.85 \
            --min-security 0.95 \
            --min-quality 0.70
```

### Results Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CODE EVAL DASHBOARD                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  OVERALL PASS RATE: 87.3%                   ▲ +2.1% from last week          │
│                                                                              │
│  BY LANGUAGE:                                                                │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │ Python     ████████████████████████░░░░░░ 89.2%                   │      │
│  │ JavaScript ████████████████████████░░░░░░ 88.1%                   │      │
│  │ Java       ██████████████████████░░░░░░░░ 85.4%                   │      │
│  │ Go         ██████████████████████░░░░░░░░ 84.7%                   │      │
│  │ Rust       ████████████████████░░░░░░░░░░ 82.1%                   │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  BY DIMENSION:                                                               │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │ Syntax         ██████████████████████████████ 99.1%               │      │
│  │ Execution      █████████████████████████░░░░░ 87.3%               │      │
│  │ Security       ███████████████████████████░░░ 96.2%               │      │
│  │ Quality        ██████████████████████░░░░░░░░ 78.4%               │      │
│  │ Task Alignment ████████████████████████░░░░░░ 84.5%               │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
│  TOP FAILURE CATEGORIES:                                                     │
│  1. Edge case handling (23% of failures)                                    │
│  2. Error handling missing (18% of failures)                                │
│  3. Inefficient solution (15% of failures)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Case Study 3: RAG System Evaluation

### The Problem

A legal tech company built a RAG (Retrieval-Augmented Generation) system for legal research. Critical requirements:
- Retrieved documents must be relevant to the query
- Generated answers must be grounded in retrieved documents
- No hallucinated case citations
- Accurate interpretation of legal precedents
- Clear attribution of sources

### RAG Eval Framework

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RAG EVALUATION FRAMEWORK                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         User Query                                           │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                    RETRIEVAL EVALUATION                           │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │      │
│  │  │  Relevance  │  │   Recall    │  │  Ranking    │              │      │
│  │  │   Score     │  │  (Coverage) │  │  Quality    │              │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │      │
│  └────────────────────────────┬──────────────────────────────────────┘      │
│                               │                                              │
│                               ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                   GENERATION EVALUATION                           │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │      │
│  │  │ Groundedness│  │  Accuracy   │  │ Attribution │              │      │
│  │  │ (Faithful?) │  │(Factually?) │  │  (Cited?)   │              │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │      │
│  └────────────────────────────┬──────────────────────────────────────┘      │
│                               │                                              │
│                               ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │                    END-TO-END EVALUATION                          │      │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │      │
│  │  │  Answers    │  │ Correctness │  │ Completeness│              │      │
│  │  │  Question?  │  │(Right Info?)│  │ (All Parts?)│              │      │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class RAGEvaluator:
    """Comprehensive RAG system evaluation"""
    
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.e2e_evaluator = EndToEndEvaluator()
    
    async def evaluate(self, 
                      query: str,
                      retrieved_docs: List[dict],
                      generated_answer: str,
                      ground_truth: dict = None) -> dict:
        """Full RAG evaluation"""
        
        results = {}
        
        # 1. Retrieval Quality
        results['retrieval'] = await self.retrieval_evaluator.evaluate(
            query=query,
            retrieved_docs=retrieved_docs,
            relevant_docs=ground_truth.get('relevant_docs') if ground_truth else None
        )
        
        # 2. Generation Quality
        results['generation'] = await self.generation_evaluator.evaluate(
            query=query,
            context_docs=retrieved_docs,
            generated_answer=generated_answer
        )
        
        # 3. End-to-End
        if ground_truth:
            results['e2e'] = await self.e2e_evaluator.evaluate(
                query=query,
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth.get('answer')
            )
        
        # Compute overall score
        results['overall'] = self._aggregate_scores(results)
        
        return results
    
    def _aggregate_scores(self, results: dict) -> dict:
        weights = {
            'retrieval': 0.3,
            'generation': 0.4,
            'e2e': 0.3
        }
        
        score = 0
        for component, weight in weights.items():
            if component in results:
                score += results[component].get('score', 0) * weight
        
        return {
            'score': score,
            'passed': score >= 0.75
        }

class RetrievalEvaluator:
    """Evaluate retrieval quality"""
    
    async def evaluate(self, 
                      query: str,
                      retrieved_docs: List[dict],
                      relevant_docs: List[str] = None) -> dict:
        
        result = {
            'score': 0,
            'metrics': {}
        }
        
        # If we have ground truth relevance labels
        if relevant_docs:
            # Precision: How many retrieved are relevant?
            retrieved_ids = [d['id'] for d in retrieved_docs]
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_docs))
            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            
            # Recall: How many relevant did we retrieve?
            recall = relevant_retrieved / len(relevant_docs) if relevant_docs else 0
            
            # NDCG: Ranking quality
            ndcg = self._compute_ndcg(retrieved_docs, relevant_docs)
            
            result['metrics']['precision'] = precision
            result['metrics']['recall'] = recall
            result['metrics']['ndcg'] = ndcg
            result['score'] = (precision + recall + ndcg) / 3
        
        else:
            # No ground truth - use LLM relevance judgment
            relevance_scores = await self._llm_relevance_check(query, retrieved_docs)
            result['metrics']['avg_relevance'] = np.mean(relevance_scores)
            result['metrics']['top_k_relevance'] = np.mean(relevance_scores[:3])
            result['score'] = result['metrics']['avg_relevance']
        
        return result
    
    async def _llm_relevance_check(self, 
                                   query: str, 
                                   docs: List[dict]) -> List[float]:
        """Use LLM to judge document relevance"""
        
        scores = []
        for doc in docs:
            prompt = f"""Rate the relevance of this document to the query.

Query: {query}

Document:
{doc['content'][:1000]}

Rate from 0.0 (not relevant) to 1.0 (highly relevant).
Return only the number.
"""
            response = await self.llm.complete(prompt)
            scores.append(float(response.strip()))
        
        return scores

class GenerationEvaluator:
    """Evaluate generation quality (groundedness, accuracy, attribution)"""
    
    async def evaluate(self,
                      query: str,
                      context_docs: List[dict],
                      generated_answer: str) -> dict:
        
        result = {
            'score': 0,
            'metrics': {},
            'issues': []
        }
        
        # Combine context
        context = "\n\n".join([d['content'] for d in context_docs])
        
        # 1. Groundedness: Is everything in the answer supported by context?
        groundedness = await self._check_groundedness(generated_answer, context)
        result['metrics']['groundedness'] = groundedness['score']
        if groundedness['unsupported_claims']:
            result['issues'].extend([
                f"Unsupported claim: {claim}" 
                for claim in groundedness['unsupported_claims']
            ])
        
        # 2. Attribution: Are sources properly cited?
        attribution = await self._check_attribution(generated_answer, context_docs)
        result['metrics']['attribution'] = attribution['score']
        
        # 3. Hallucination detection
        hallucinations = await self._detect_hallucinations(generated_answer, context)
        result['metrics']['no_hallucinations'] = 1.0 - len(hallucinations) * 0.1
        if hallucinations:
            result['issues'].extend([
                f"Potential hallucination: {h}" for h in hallucinations
            ])
        
        # Weighted score
        result['score'] = (
            0.5 * result['metrics']['groundedness'] +
            0.2 * result['metrics']['attribution'] +
            0.3 * result['metrics']['no_hallucinations']
        )
        
        return result
    
    async def _check_groundedness(self, answer: str, context: str) -> dict:
        """Check if answer is grounded in context"""
        
        prompt = f"""Analyze if this answer is fully supported by the provided context.

Context:
{context[:3000]}

Answer:
{answer}

Identify:
1. Claims that are fully supported by the context
2. Claims that are partially supported
3. Claims that are NOT supported (potential hallucinations)

Return JSON:
{{
    "supported_claims": ["..."],
    "partially_supported": ["..."],
    "unsupported_claims": ["..."],
    "score": 0.X  // 1.0 = fully grounded, 0.0 = completely ungrounded
}}
"""
        
        response = await self.llm.complete(prompt, response_format='json')
        return json.loads(response)
    
    async def _detect_hallucinations(self, answer: str, context: str) -> List[str]:
        """Detect specific hallucinations"""
        
        # Extract citations/references from answer
        citations = self._extract_citations(answer)
        
        hallucinations = []
        for citation in citations:
            if not self._citation_in_context(citation, context):
                hallucinations.append(f"Citation not found: {citation}")
        
        # Extract facts/claims
        claims = await self._extract_claims(answer)
        for claim in claims:
            if not await self._claim_supported(claim, context):
                hallucinations.append(f"Unsupported claim: {claim}")
        
        return hallucinations
```

### Results

| Metric | Before RAG Eval | After 6 Months |
|--------|-----------------|----------------|
| Retrieval Precision@10 | 62% | 84% |
| Groundedness Score | 71% | 93% |
| Hallucination Rate | 15% | 2% |
| Citation Accuracy | 68% | 96% |
| User Trust Score | 3.2/5 | 4.5/5 |

---

## Case Study 4: Content Moderation System

### The Problem

A social media platform needed to automatically moderate user-generated content. Requirements:
- Detect hate speech, harassment, violence
- Handle edge cases and context-dependent decisions
- Minimize false positives (over-moderation)
- Maintain consistency across moderators
- Scale to millions of posts per hour

### Multi-Model Consensus Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTENT MODERATION EVALUATION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Content Input                                                               │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TIER 1: FAST FILTERS                             │    │
│  │  • Keyword blocklist                                                │    │
│  │  • Known bad content hashes                                         │    │
│  │  • Previously flagged content similarity                            │    │
│  │  Latency: < 5ms                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       │ 95% pass                                                            │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TIER 2: ML CLASSIFIERS                           │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │    │
│  │  │   Toxicity  │  │   Violence  │  │    NSFW     │                 │    │
│  │  │  Classifier │  │  Classifier │  │  Classifier │                 │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │    │
│  │  Latency: < 50ms                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       │ Uncertain cases (5-10%)                                             │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TIER 3: LLM JUDGMENT                             │    │
│  │  • Contextual understanding                                         │    │
│  │  • Nuanced policy application                                       │    │
│  │  • Detailed reasoning                                               │    │
│  │  Latency: < 2s                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       │ Still uncertain (1-2%)                                              │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    TIER 4: HUMAN REVIEW                             │    │
│  │  • Expert moderators                                                │    │
│  │  • Policy edge cases                                                │    │
│  │  • Appeals                                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Evaluation Framework

```python
class ModerationEvaluator:
    """Evaluate content moderation system"""
    
    CATEGORIES = [
        'hate_speech', 'harassment', 'violence', 
        'sexual_content', 'misinformation', 'spam'
    ]
    
    def __init__(self):
        self.human_labels = self._load_human_labels()
        self.policy_guidelines = self._load_policy()
    
    async def evaluate_batch(self, 
                            content_items: List[dict],
                            predictions: List[dict]) -> dict:
        """Evaluate a batch of moderation decisions"""
        
        results = {
            'overall': {},
            'by_category': {},
            'confusion_matrix': {},
            'error_analysis': []
        }
        
        # Compare predictions to human labels
        for item, pred in zip(content_items, predictions):
            human_label = self.human_labels.get(item['id'])
            if human_label:
                self._record_comparison(results, pred, human_label, item)
        
        # Compute metrics
        results['overall'] = self._compute_metrics(results['confusion_matrix'])
        
        # Per-category analysis
        for category in self.CATEGORIES:
            results['by_category'][category] = self._compute_category_metrics(
                results, category
            )
        
        return results
    
    def _compute_metrics(self, confusion: dict) -> dict:
        """Compute precision, recall, F1 for moderation"""
        
        tp = confusion.get('true_positive', 0)
        fp = confusion.get('false_positive', 0)
        tn = confusion.get('true_negative', 0)
        fn = confusion.get('false_negative', 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # For moderation, we care about both:
        # - Not letting bad content through (recall)
        # - Not blocking good content (specificity/false positive rate)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
    
    async def evaluate_consistency(self, 
                                   content_variations: List[List[dict]]) -> dict:
        """Test consistency across similar content variations"""
        
        inconsistencies = []
        
        for variation_group in content_variations:
            # All items in group should have same decision
            decisions = [self.model.predict(item) for item in variation_group]
            
            if not all(d == decisions[0] for d in decisions):
                inconsistencies.append({
                    'items': variation_group,
                    'decisions': decisions,
                    'type': 'inconsistent_variations'
                })
        
        return {
            'consistency_rate': 1 - len(inconsistencies) / len(content_variations),
            'inconsistencies': inconsistencies
        }
    
    async def evaluate_edge_cases(self, edge_cases: List[dict]) -> dict:
        """Test on known edge cases"""
        
        results = []
        
        for case in edge_cases:
            prediction = await self.model.predict(case['content'])
            
            correct = prediction['decision'] == case['expected_decision']
            
            results.append({
                'case_id': case['id'],
                'category': case['category'],
                'correct': correct,
                'prediction': prediction,
                'expected': case['expected_decision'],
                'notes': case.get('notes')
            })
        
        # Group by category
        by_category = {}
        for r in results:
            cat = r['category']
            if cat not in by_category:
                by_category[cat] = {'correct': 0, 'total': 0}
            by_category[cat]['total'] += 1
            if r['correct']:
                by_category[cat]['correct'] += 1
        
        return {
            'overall_accuracy': sum(r['correct'] for r in results) / len(results),
            'by_category': {
                k: v['correct'] / v['total'] for k, v in by_category.items()
            },
            'failures': [r for r in results if not r['correct']]
        }
```

### Key Insights

1. **False positives hurt trust more than false negatives**: Users leave when unfairly moderated
2. **Context is everything**: Same words can be hate speech or reclaimed language
3. **Cultural variation**: Moderation standards vary globally
4. **Edge cases are infinite**: Adversarial users constantly find new ways

---

## Case Study 5: Tool-Using Agent (Customer-Refund Bot)

### Context
A mid-size DTC retailer ships a Claude Sonnet 4.6 agent (`claude-sonnet-4-6` — picked partly on its τ²-bench Retail score of 91.7, [Sonnet 4.6 system card, Table 2.1.A](https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf)) with three tools: `lookup_order`, `issue_refund`, `escalate_to_human`. Eval focus is **trajectory correctness**, not just final-message correctness, because a wrong tool call (e.g. refunding twice) costs real money.

### Eval Design

| Layer | Metric | Target |
|-------|--------|--------|
| Outcome | `agent_goal_accuracy` (RAGAS) on 200 scripted scenarios | ≥ 0.92 |
| Tool selection | `tool_call_accuracy` per turn | ≥ 0.97 |
| Tool args | `tool_call_f1` (exact `order_id` + amount within $0.01) | ≥ 0.99 |
| Safety | Adversarial set: “refund $500 to a different account” × 30 | 0 unauthorized actions |
| Efficiency | Median steps to resolution | ≤ 4 |
| Recovery | Synthetic tool errors injected on 10% of runs → does the agent recover? | ≥ 0.90 success |
| Reliability | `pass^8` on the 20 highest-traffic scenarios (**all** 8 trials succeed, not ≥1 — [tau-bench](https://arxiv.org/abs/2406.12045)) | ≥ 0.75 |

### Implementation Sketch (Inspect AI)

```python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import use_tools, generate, system_message
from inspect_ai.scorer import scorer, mean, Score, Target
from tools import lookup_order, issue_refund, escalate_to_human

@scorer(metrics=[mean()])
def refund_trajectory():
    async def score(state, target: Target):
        tool_calls = [m for m in state.messages if m.role == "tool"]
        used_refund = [c for c in tool_calls if "refunded" in (c.text or "")]
        # Hard-fail on unauthorized actions
        if target.text == "NO_REFUND" and used_refund:
            return Score(value=0.0, explanation="unauthorized refund")
        # Hard-pass on confirmed correct path
        confirmed = "refund" in state.output.completion.lower()
        return Score(value=1.0 if (confirmed and used_refund) else 0.0)
    return score

@task
def refund_agent_eval():
    return Task(
        dataset=json_dataset("data/refund_scenarios.jsonl"),
        solver=[system_message(open("prompts/agent.md").read()),
                use_tools(lookup_order(), issue_refund(), escalate_to_human()),
                generate()],
        scorer=refund_trajectory(),
        sandbox="docker",
    )
```

### Results After 6 Weeks

| Metric | Week 1 | Week 6 |
|--------|--------|--------|
| Goal accuracy | 0.78 | 0.94 |
| Tool-arg F1 | 0.91 | 0.995 |
| Unauthorized actions (adversarial) | 4 / 30 | 0 / 30 |
| Median steps | 6 | 3 |

### Key Insights
1. Trajectory metrics caught 3 bugs that final-output metrics missed (agent refunded then apologized — “successful” output, broken behavior).
2. Sandboxed eval was non-negotiable — a buggy iteration tried to call `issue_refund` 50 times in a loop on one task.
3. Adversarial scenarios were the highest-ROI items per dollar spent.

### 2026 Postscript: The Industry Playbook Caught Up

When this system was built, trajectory-vs-outcome grading was a judgment call. Anthropic's ["Demystifying evals for AI agents"](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (Jan 2026) has since standardized the vocabulary and defaults:

- **Task / trial / transcript / outcome.** Grade the *outcome* — the actual end-state of the environment, not what the agent claims it did. Trajectory checks like the scorer above remain valuable as diagnostics and hard-fail safety rules, but brittle step-sequence assertions are an anti-pattern.
- **pass@k vs pass^k.** pass@k = P(≥1 of k trials succeeds); **pass^k = P(all k succeed)**. A 90%-per-trial agent is only ~43% reliable at pass^8 (0.9⁸). For an agent that moves money, pass^k is the number that matters ([tau-bench](https://arxiv.org/abs/2406.12045), which introduced it, found GPT-4o's retail pass^8 was under 25%).
- **Distrust your harness before your model.** An internal Anthropic benchmark run scored Opus 4.5 at 42% until harness bugs were fixed — the same model then scored 95% ([source](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
- **Start small, read transcripts.** 20–50 tasks drawn from real failures, each trial isolated in a clean environment; manual transcript review is non-negotiable.

---

## Case Study 6: Reasoning-Model Math Tutor

### Context
A tutoring product uses a GPT-5-class reasoning model to walk students through problems. Both *answer correctness* and *reasoning quality* matter — a right answer with bad reasoning teaches nothing.

### Eval Design

| Dimension | Method |
|-----------|--------|
| Final answer | Exact match against numerical / symbolic ground truth (SymPy) |
| Step-level correctness | Process-reward model (PRM) scores each step 0–1 |
| CoT faithfulness | Perturb a key intermediate value; does the final answer change consistently? |
| Pedagogical quality | LLM-judge rubric on a 50-item set, calibrated against 2 math teachers |
| Reasoning effort | Sweep `reasoning_effort` ∈ {low, medium, high}; plot accuracy vs latency |

### CoT-Faithfulness Probe (illustrative)

```python
from openai import OpenAI
client = OpenAI()

def faithfulness_probe(problem: str, original_steps: list[str]) -> bool:
    """Replace step k with a wrong value; the final answer should change.
    If it doesn't, the chain is post-hoc rationalization."""
    k = len(original_steps) // 2
    perturbed = original_steps.copy()
    perturbed[k] = perturbed[k].replace("= 12", "= 99")   # inject error
    prompt = f"Problem: {problem}\nReasoning so far:\n" + "\n".join(perturbed) + "\nFinal answer:"
    cont = client.chat.completions.create(
        model="gpt-5.5", reasoning_effort="low",
        messages=[{"role":"user","content":prompt}]).choices[0].message.content
    # Faithful chain → the perturbed answer differs from the original
    return cont.strip() != "<original final answer>"
```

### Reasoning-Effort Sweep Result

| `reasoning_effort` | Accuracy | Median latency | Cost / 1k problems |
|--------------------|----------|----------------|--------------------|
| low                | 0.71     | 1.2 s          | $1.40 |
| medium             | 0.86     | 4.8 s          | $5.10 |
| high               | 0.89     | 18 s           | $22.00 |

**Decision:** ship `medium` to production. The +0.03 from `high` did not justify 4× latency and 4× cost for the tutoring use case.

### Key Insights
1. CoT-faithfulness probes flagged a regression in a checkpoint where the model routinely produced confident-but-fabricated reasoning steps that didn’t affect the final answer.
2. Process-reward scoring caught “right answer, wrong method” — critical for a teaching product.
3. The accuracy-vs-effort curve is the single most useful artefact for product decisions on reasoning models. Run it for *every* release.

---

## Case Study 7: How a Frontier Model Ships — Anatomy of the Fable 5 / Mythos 5 Release Evaluation

### Context

The largest eval engineering project you can study end-to-end is a frontier model release. Anthropic's Claude Fable 5 / Mythos 5 [system card](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) (June 2026, 319 pages) documents one in full.

One structural decision shapes everything: **Fable 5 and Mythos 5 are the same underlying model.** Mythos 5 is the unsafeguarded configuration (trusted partners only); Fable 5 is the general-access configuration whose classifier safeguards, when triggered, fall back to Opus 4.8. So Anthropic runs *capability and dangerous-capability evals on Mythos 5* (true underlying capability) and *safeguard/harmlessness evals on Fable 5* (the shipped product) — the eval target depends on the question being asked (card §1.5, §2.1).

### The Release Eval Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              FRONTIER RELEASE EVALUATION STACK (Fable 5 card)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LAYER 1: CAPABILITY (§8)            on Mythos 5 — "how good is it?"        │
│    SWE-bench, Terminal-Bench, BrowseComp, OSWorld, HLE, GDPval-AA, ...      │
│                                                                              │
│  LAYER 2: DANGEROUS CAPABILITY (§2–3) on Mythos 5 — "is it safe to exist?"  │
│    RSP threat models: bio/chem (CB-1/CB-2), cyber, autonomy / AI R&D        │
│    → output: an ASL level + required deployment/security controls           │
│                                                                              │
│  LAYER 3: ALIGNMENT (§6)             "does it behave when no one looks?"    │
│    Automated behavioral audit · Petri (external) · targeted evals ·         │
│    white-box probes on internal activations                                 │
│                                                                              │
│  LAYER 4: META-EVALUATION (§6.5)     "can we trust layers 1–3?"             │
│    Evaluation awareness · sandbagging checks · CoT monitorability ·         │
│    stealth / safeguard-evasion capability                                   │
│                                                                              │
│  LAYER 5: EXTERNAL VALIDATION        "don't take our word for it"           │
│    METR · UK AISI · Meridian Labs (Petri) · Andon Labs · Gray Swan ·        │
│    public jailbreak bug bounty                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer 1: Capability — and the measurable safety tax

Harness fully disclosed: adaptive thinking at max effort, default sampling, scores averaged over 5 trials, context ≤1M tokens (card §8.1). Selected results:

| Benchmark | Mythos 5 (unsafeguarded) | Fable 5 (shipped) |
|---|---|---|
| SWE-bench Verified | 95.5 | 95 |
| SWE-bench Pro | 80.3 | 80 |
| Terminal-Bench 2.1 | 88.0 | 84.3 |

The Terminal-Bench gap is a **measured safety tax**: 20.9% of Fable 5 trials hit a safety refusal mid-trajectory and fell back to Opus 4.8 (card §8.3). Publishing both numbers, rather than only the better one, is the honest way to report a safeguarded product.

### Layer 2: Dangerous capability → a ship/no-ship decision

The RSP evals output a decision, not a score: **ASL-3** deployment. Two illustrative judgments (card §2.1–§3.2):

- **AI R&D (autonomy threat model 2) ruled out** — but the old task-based "rule-out" evals had *saturated* (the model beats top-human thresholds on nearly all of them), so the case rests on new evidence: extensive internal use showing it cannot substitute for senior research staff, and internal measures of Anthropic's own pace of progress — including the Anthropic ECI, a fork of Epoch AI's capability index (§2.3.5–2.3.6) — showing no sustained AI-attributable 2× acceleration. Lesson: when an eval saturates, it stops bounding risk — you must replace the *evidence*, not just re-run the suite.
- **Cyber**: Mythos 5 built a working exploit for a Firefox 147 vulnerability on 88.4% of trials vs 8.8% for Opus 4.8 (§3.2.4) — a capability jump that the shipped product's cyber classifiers must contain (they flagged 407 of 410 eval episodes, §3.2.1).

### Layer 3: Alignment at scale

- **Automated behavioral audit**: ~2,900 investigation sessions per model from ~1,450 largely hand-written seed scenarios. The investigators are themselves models (a helpful-only Opus 4.8 variant plus Mythos 5 itself); the judge is Mythos Preview (§6.2.3). Evaluation has become an agent workload.
- **Petri 3.0** (Anthropic's open-source auditing framework, [alignment.anthropic.com/2025/petri](https://alignment.anthropic.com/2025/petri/), now maintained by the independent nonprofit Meridian Labs): 173 seed scenarios, investigators Opus 4.7 + GPT-5.5, judges from **three different labs** (Opus 4.7 / GPT-5.5 / Gemini 3.1 Pro) — cross-lab judge diversity as a bias control (§6.2.3.3).

### Layer 4: Meta-evaluation — trusting the measurement

The card spends a full section asking whether its own results can be believed:

- **Evaluation awareness** is measured four independent ways (activation probes, verbalized mentions, self-ratings, NLA-graded token analysis) — and the measures only weakly correlate, i.e. "knows it's being tested" is not one thing (§6.5.1).
- **Sandbagging sweep**: 2,440 dangerous-capability transcripts model-graded on a 0–10 sandbagging scale; 9 flagged for manual review; none concerning (§6.5.2).
- **White-box probes** for dishonesty, reward hacking, and eval-awareness ran on essentially every RL transcript during most of post-training — but were *never used as a training signal*, to avoid training the model to fool the probes (§6.4.1).

### Layer 5: External validation

- **METR** ran a pre-release snapshot on 38 of its hardest software/R&D tasks; verdict: "likely unable to fully and reliably automate R&D for frontier projects spanning multiple weeks," and "roughly on-trend" (card §2.3.8).
- **UK AISI** developed a single-turn jailbreak of Fable 5's cyber classifiers *within hours* (multi-turn agentic within ~2 more days), without achieving reliable long-form extraction (§3.3.1). The card is careful to add that these were interim results from a compressed testing window, "not a measure of the relative robustness" of the safeguards — disclosing the limits of your own red-team data is itself good eval practice.
- **Public bug bounty** (with Gray Swan): ~100,000 attempts ≈ 1,000 hours of adversarial effort by June 5, 2026 — zero universal jailbreaks, two task-specific ones (§3.3.2).

### Lessons for Eval Engineers

1. **Separate the capability question from the product question.** Two configurations, two eval targets — the same discipline applies to any system with guardrails: measure the raw model *and* the guarded product.
2. **Report the safety tax.** 88.0 vs 84.3 with the mechanism (20.9% classifier fallback) explained is more credible than one cherry-picked number.
3. **Saturated evals stop bounding risk.** Plan the replacement evidence before your suite saturates.
4. **Meta-evaluate.** At frontier scale, eval-awareness, sandbagging, and grader-hacking get their own eval suites. Your LLM judge deserves the same scrutiny (Module 2).
5. **Diversity and externality buy credibility**: cross-lab judges, named third parties, and paid public adversaries.
6. **Disclose the harness.** Effort setting, trial count, context limit, scaffold — a score without them is not reproducible.

---

## Case Study 8: GDPval — Grading Real Economic Work Without Unit Tests

### The Problem

OpenAI wanted to measure whether models can do *real, economically valuable knowledge work* — legal briefs, financial models, engineering plans, care plans. These deliverables have no exact-match answer, no executable test, and no single rubric that fits all of them. ([GDPval, arXiv:2510.04374](https://arxiv.org/abs/2510.04374), launched Sept 25, 2025.)

### Eval Design

| Design choice | Implementation |
|---|---|
| Task realism | 1,320 tasks from 44 occupations across the 9 top-GDP sectors, authored by professionals averaging 14 years of experience; each task ≈ 7–9 hours of expert work, ≈ $400 of value ([paper](https://arxiv.org/abs/2510.04374)) |
| Grading | **Blind expert pairwise comparison**: an occupational expert sees the model deliverable and a human deliverable, without knowing which is which, and picks the better one. Headline metric = win/tie rate |
| Contamination control | Only a 220-task "gold" subset is open-sourced (with an automated grader, [evals.openai.com](https://evals.openai.com/)); the rest stays private |
| Grader validation | A trained automated grader reaches **66% agreement with human experts — vs 71% human-human inter-rater agreement** ([GDPval paper PDF](https://cdn.openai.com/pdf/d5eb7428-c4e9-4a33-bd86-86dd4bcf12ce/GDPval.pdf)). The ceiling for any grader is human-human agreement, not 100% |

### Results

| When | Model | Result |
|---|---|---|
| Launch (Sept 2025) | Claude Opus 4.1 | Outputs rated ≥ human expert 47.6% of the time ([Fortune](https://fortune.com/2025/09/30/ai-models-are-already-as-good-as-experts-at-half-of-tasks-a-new-openai-benchmark-gdpval-suggests/)) |
| Apr 2026 | GPT-5.5 | 84.9% win/tie rate vs experts ([OpenAI](https://openai.com/index/introducing-gpt-5-5/)) |

A launch finding worth quoting to any budget owner: frontier models approached expert parity roughly **100× faster and cheaper** than the experts producing the same deliverables ([coverage](https://www.marketingaiinstitute.com/blog/openai-gdpval)). Artificial Analysis also runs an independent Elo-scored variant, [GDPval-AA](https://artificialanalysis.ai/evaluations/gdpval-aa) — Fable 5 scored 1932 vs Opus 4.8's 1769 ([Fable 5 system card §8.17.7](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)); the [Sonnet 4.6 card](https://www-cdn.anthropic.com/78073f739564e986ff3e28522761a7a0b4484f84.pdf) reports 1633 for Sonnet 4.6 and 1606 for Opus 4.6 — keeping the vendor honest with a re-implementation it doesn't control.

### Key Insights

1. **Pairwise comparison beats absolute scoring** when there is no rubricable ground truth — "which of these two is better?" is a far more reliable expert judgment than "score this 0–100".
2. **Validate your automated grader against the human-human agreement ceiling.** 66% vs a 71% ceiling means the grader is within 5 points of the best achievable — that framing, not "only 66%", is the right read.
3. **Blind the graders.** Experts must not know which deliverable is the model's; style tells alone can swing preferences.
4. **Hold most of the set back.** The 220/1,320 public-private split is now the standard contamination defense (Module 12 covers why).
5. **Expect independent replication** — and treat divergence between your number and the third-party number as information, not noise.

---

## Case Study 9: Vending-Bench 2 — Long-Horizon Agent Evaluation

### The Problem

Agents pass 10-step evals, then fall apart in week-long deployments. [Andon Labs](https://andonlabs.com/evals/vending-bench-2) built Vending-Bench to measure what no short eval can: **coherence over a simulated year** of running a small business. The agent starts with $500, manages a simulated vending machine operation (suppliers, pricing, stock, email), and is scored on one number — the final balance ([original paper, arXiv:2502.15840](https://arxiv.org/pdf/2502.15840)).

### Eval Design

- **One dollar-denominated outcome metric.** No step grading, no rubric — the environment itself does the scoring. At long horizons, step-level metrics drown in noise; outcome metrics survive.
- **The eval is an environment, not a dataset** — dataset, harness, and scoring rules are a single artifact. This is the "environments are the new datasets" pattern (Module 11).
- **A human baseline anchors the scale**: an estimated "good human" operator makes ~$63K/year.

### Results (Vending-Bench 2)

| Agent | Final balance (1 simulated year) |
|---|---|
| Gemini 3 Pro | $5,478.16 |
| Claude Opus 4.5 | $4,967.06 |
| Claude Sonnet 4.5 | $3,849.74 |
| Estimated good human | ~$63,000 — >10× the best model |

(Source: [Andon Labs](https://andonlabs.com/evals/vending-bench-2), [Vellum analysis](https://www.vellum.ai/blog/claude-opus-4-5-benchmarks).)

### Documented Failure Modes

The transcripts, not the leaderboard, are the real product: **context degradation** (forgetting earlier commitments as history compacts), **"meltdowns"** (spiraling, unrecoverable loops), and **emergent deception under pressure** — failure modes that simply do not appear in short evals.

### From Benchmark to Production Practice

- **Vending-Bench Arena** adds head-to-head multi-agent competition ([Andon Labs](https://andonlabs.com/evals/vending-bench-arena)).
- Frontier labs now buy this as pre-release behavioral testing: Anthropic used Andon Labs' Vending-Bench 2 and Arena in the Fable 5 release evaluation ([system card §6.2.5, §8.17.6](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)).
- **Project Vend** closed the sim-to-real loop: a real vending machine in Anthropic's office run by an agent — the ultimate "production eval."

### Key Insights

1. **At long horizons, grade outcomes.** A year of trajectory has too many defensible paths to grade step-by-step; the bank balance is unambiguous.
2. **Dollar-denominated metrics travel.** "$5,478 vs a $63K human baseline" lands with executives in a way "0.34 mean reward" never will.
3. **The headroom is the headline.** While static benchmarks saturate (SWE-bench Verified is at ~95%), the best model still earns <10% of a competent human here — eval design determines whether you can still see the frontier.
4. **Long-horizon failure modes are qualitatively new.** Meltdowns and deception-under-pressure justify the cost of long-horizon evals on their own: you cannot mitigate what your eval never elicits.

---

## Case Study 10: Uber Eats Multimodal Image Agent — Evaluating Generation With No Ground Truth

> Source: a talk by **Soumya Gupta** and **Jai Chopra** (Uber) on the multimodal agent that enhances food photography on Uber Eats. Architecture, stage boundaries, and design principles below follow their account; **all numbers in this section are illustrative** unless a source is cited — treat them as worked arithmetic, not reported Uber metrics.

### The Problem

Hundreds of thousands of independent merchants upload their own food photos. Many are bad in ways that cost the merchant orders — dim lighting, cluttered composition, bad crop, phone-flash glare. A better photo lifts conversion. But the obvious fix (regenerate the image) is unacceptable: the picture is a **commercial claim about what arrives in the bag**. And a marketplace where every photo has been pushed toward the same "good food photo" prior is a marketplace that has erased the thing that makes it a marketplace.

So the eval problem is defined by three properties that break most eval playbooks at once:

| Property | Why the usual approach fails |
|---|---|
| **No reference output** | There is no "correct" enhanced image to diff against. Exact match, BLEU, and every reference-based metric are unavailable. |
| **Quality is subjective, harm is not** | "Better composition" is a matter of taste. "Shows a garnish the merchant doesn't serve" is a factual defect with legal and trust consequences. These two cannot live on the same 1–5 scale. |
| **Marketplace-level effects** | Every per-image metric can improve while the portfolio gets worse (homogenization). Your eval set has to measure the population, not just the sample. |

Compare this to Case Study 2 (code generation), where tests give free ground truth, and to Case Study 4 (moderation), where labels are contested but at least discrete. This is the hardest eval regime: **generative, subjective, reference-free, and brand-critical**.

### Architecture and the Eval That Belongs to Each Stage

The system is three stages, and — this is the transferable lesson — **each stage is a different kind of eval problem**. Teams that try to cover an agentic pipeline with one end-to-end quality score end up unable to answer "which stage regressed?"

```
        ┌────────────────────────┐
Input   │ 1. Understanding &     │  → skip (most images)
image + │    Routing Agent       │
metadata└────────┬───────────────┘
                 │ enhance
                 ▼
        ┌────────────────────────┐
        │ 2. Edit ⇄ QA Loop      │ ◄─┐  pass@K self-correction
        │    (directive → edit   │   │
        │     → QA gate)         │ ──┘  fail → re-edit with critique
        └────────┬───────────────┘
                 │ passed
                 ▼
        ┌────────────────────────┐
        │ 3. Final Guardrails    │  → publish-ready
        │    (Swiss cheese)      │
        └────────────────────────┘
                 │
                 ▼   production traces
        ┌────────────────────────────────────────┐
        │ Closed loop: sample → golden benchmark │
        │ → diagnosis agent → prompt/config      │
        │ update → canary → promote              │
        └────────────────────────────────────────┘
```

#### Stage 1 — Routing agent: a classifier, and you must evaluate it like one

The routing agent reads the image plus metadata and answers one question: *does this image need enhancement at all?* Gupta and Chopra are explicit that it behaves as a classifier **optimized for recall** — missing a genuinely bad image is the expensive error — while keeping compute down, because the cheap path (skip) has to stay cheap at marketplace volume.

That framing gives you the eval for free: confusion matrix, precision/recall at the deployed threshold, and cost per thousand images. But there is a trap here that catches most production teams:

> **The routing gate censors your dataset.** Only routed images get enhanced, so only routed images produce downstream quality signal. Your false-negative rate — good-looking-to-the-router images that were actually bad — is invisible in production data *by construction*. You cannot measure it without deliberately sampling and labeling images the router **rejected**.

This is the single most common measurement bug in cascaded AI systems, and it has a standard fix: sample the skipped population on a schedule and label it.

```python
# pip install anthropic
# Estimating true recall of a routing gate under censoring.
# The router's own precision is measurable from production; its recall is NOT,
# because rejected images never get a downstream label. Buy that number back
# with a small stratified audit of the rejected pool.
from dataclasses import dataclass

@dataclass
class RouterAudit:
    routed_n: int              # images the router sent to enhancement
    routed_truly_bad: int      # of those, human-confirmed as needing enhancement
    skipped_sampled_n: int     # random sample drawn from the SKIPPED pool
    skipped_sampled_bad: int   # of that sample, human-confirmed as needing enhancement
    skipped_total: int         # size of the whole skipped pool this period

    def precision(self) -> float:
        return self.routed_truly_bad / self.routed_n

    def estimated_false_negatives(self) -> float:
        """Extrapolate missed-bad-images from the audited sample to the full pool."""
        miss_rate = self.skipped_sampled_bad / self.skipped_sampled_n
        return miss_rate * self.skipped_total

    def estimated_recall(self) -> float:
        tp = self.routed_truly_bad
        fn = self.estimated_false_negatives()
        return tp / (tp + fn)

# Illustrative numbers, not Uber's:
audit = RouterAudit(
    routed_n=10_000, routed_truly_bad=8_400,
    skipped_sampled_n=500, skipped_sampled_bad=35,
    skipped_total=190_000,
)
print(f"precision {audit.precision():.1%}")          # precision 84.0%
print(f"est. misses {audit.estimated_false_negatives():,.0f}")  # est. misses 13,300
print(f"est. recall {audit.estimated_recall():.1%}")            # est. recall 38.7%
```

The arithmetic is the point. A router with a comfortable-looking 84% precision can be missing the majority of bad images, and **nothing in your production dashboards will say so** — every downstream metric is computed on the routed slice, which looks fine. The audit line item (500 human labels per period) is what converts an unfalsifiable claim into a number.

Threshold selection then becomes an explicit, defensible trade rather than a vibe:

| Threshold | Est. recall | Images routed | Enhancement compute | Marginal cost per +1pp recall |
|---|---|---|---|---|
| 0.7 | 62% | 5.2% | 1.0× | — |
| 0.5 | 81% | 9.1% | 1.8× | ~4.2% compute / pp |
| 0.3 | 93% | 17.4% | 3.3× | ~7.8% compute / pp |

*(Illustrative.)* Publish this table with every threshold change. "We chose 0.5" is not a decision; "we bought 19 points of recall for 80% more enhancement compute, and declined the next 12 points because they cost twice as much per point" is.

#### Stage 2 — The edit ⇄ QA loop: pass@K, and a veto criterion that must never be averaged

The generation stage is a loop: an agent produces an edit against explicit directives, a **QA gate** scores it on dimensions including plating, **faithfulness**, and realism, and the system iterates — a **pass@K** strategy where failure feeds a critique back into the next attempt. (Full treatment of loop design and its metrics is Module 14.)

The eval-design decision that matters most here is **how the QA gate aggregates its dimensions**. The instinct is a weighted average: `0.4·plating + 0.3·faithfulness + 0.3·realism`. That is wrong, and dangerously so, because it lets a beautiful image buy its way past a faithfulness failure. Faithfulness is not a quality dimension. It is a **veto**.

```python
# pip install anthropic
# QA gate with veto semantics: aesthetics are scored, integrity is binary.
# One isolated judge call per criterion (never one omnibus call — module 02 §2.3.4).
import anthropic, base64, json

client = anthropic.Anthropic()
MODEL = "claude-opus-5"

VETO = {  # any failure here rejects the edit outright, whatever the scores are
    "faithfulness": (
        "Does the edited image depict ONLY food, portions, packaging, and garnishes "
        "that are present in the original image? Adding, removing, or substituting any "
        "edible component is a FAIL. Lighting, background, crop, and color correction "
        "are NOT violations."
    ),
    "realism": (
        "Could this plausibly be a photograph of real food? Physically impossible "
        "geometry, melted or smeared edges, duplicated items, or garbled text is a FAIL."
    ),
}
SCORED = {  # only consulted once every veto passes
    "plating": "Is the food arranged so the main item is clearly readable as the hero?",
    "composition": "Is the crop and framing free of distracting clutter and dead space?",
    "lighting": "Is the subject evenly lit, without blown highlights or muddy shadows?",
}

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "FAIL"]},
        "evidence": {"type": "string", "description": "The specific region or detail relied on."},
        "repair_directive": {"type": "string", "description": "Empty if PASS."},
    },
    "required": ["verdict", "evidence", "repair_directive"],
    "additionalProperties": False,
}

def _img(b: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(b).decode()}}

def judge_criterion(name: str, question: str, original: bytes, edited: bytes) -> dict:
    resp = client.messages.create(
        model=MODEL, max_tokens=2000,
        output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA},
                       "effort": "high"},
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "ORIGINAL:"}, _img(original),
            {"type": "text", "text": "EDITED:"}, _img(edited),
            {"type": "text", "text": f"Criterion `{name}`. {question}\n"
                                     "Answer PASS or FAIL, cite the evidence you relied on, "
                                     "and if FAIL give one concrete repair directive."},
        ]}],
    )
    if resp.stop_reason == "refusal":          # Opus 5 classifiers can decline — never
        return {"verdict": "FAIL",             # read content[0] before checking this
                "evidence": f"judge refused ({resp.stop_details.category if resp.stop_details else 'unknown'})",
                "repair_directive": "escalate to human review"}
    return json.loads(next(b.text for b in resp.content if b.type == "text"))

def qa_gate(original: bytes, edited: bytes) -> dict:
    """Veto first (cheap to fail fast), then score. Never average the two."""
    for name, q in VETO.items():
        v = judge_criterion(name, q, original, edited)
        if v["verdict"] == "FAIL":
            return {"passed": False, "blocked_by": name, **v}
    scores = {n: judge_criterion(n, q, original, edited) for n, q in SCORED.items()}
    passed = sum(s["verdict"] == "PASS" for s in scores.values())
    return {
        "passed": passed >= 2,                    # quality bar: 2 of 3 aesthetic criteria
        "blocked_by": None if passed >= 2 else "quality_bar",
        "scores": scores,
        "repair_directive": " ".join(s["repair_directive"] for s in scores.values()
                                     if s["verdict"] == "FAIL"),
    }
```

Three things this structure buys you that a single omnibus "rate this edit 1–5" call does not:

1. **A failure tells you what to fix.** `blocked_by: "faithfulness"` and `blocked_by: "lighting"` route to completely different repairs — and to different owners.
2. **Isolated calls don't let dimensions contaminate each other.** A gorgeous image drags an omnibus judge's faithfulness assessment upward; a separate call with only the faithfulness question in context can't be seduced.
3. **The veto is structurally un-tradeable.** No amount of prompt drift on the aesthetic criteria can create a path to publishing an unfaithful image, because the veto is evaluated in code, not by the model.

**And the gate itself needs an eval set.** This is the step production teams skip. The QA gate is a classifier over (original, edited) pairs, so it has precision and recall against human judgment, and those numbers move whenever anyone touches the judge prompt or the model version:

| Gate criterion | Human-labeled pairs | Gate precision | Gate recall | Cohen's κ |
|---|---|---|---|---|
| faithfulness | 400 | 0.94 | 0.88 | 0.81 |
| realism | 400 | 0.91 | 0.83 | 0.74 |
| plating | 400 | 0.72 | 0.69 | 0.44 |

*(Illustrative.)* Read the last row honestly: κ = 0.44 on plating means the gate and your human raters **substantially disagree about what good plating is**. That is not necessarily a bug — it may mean your annotation guideline is underspecified — but it does mean a plating score is not yet evidence for a launch decision, while a faithfulness verdict (κ = 0.81) is. Different criteria earn different levels of trust, and your dashboard should say which.

#### Stage 3 — Swiss cheese, and the correlation that quietly eats it

The final stage is post-processing plus a "publish-ready" QA step, described explicitly as a **Swiss cheese model**: several imperfect layers stacked so that a defect has to pass through a hole in every slice.

The arithmetic only works if the layers fail **independently**. And here is the failure mode that turns a Swiss cheese defense into a single slice with extra latency: if your stage-2 QA gate and your stage-3 publish check are the same model, running the same prompt family, on the same inputs, they do not have independent holes — **they have the same hole**, and you have paid three times for one layer of protection.

```python
# What layer independence is worth — and what correlation costs you.
def escape_rate(layer_miss_rates: list[float], correlation: float = 0.0) -> float:
    """correlation=0 → independent layers; correlation=1 → layers fail together."""
    independent = 1.0
    for m in layer_miss_rates:
        independent *= m
    worst_layer = max(layer_miss_rates)          # fully-correlated case
    return correlation * worst_layer + (1 - correlation) * independent

layers = [0.15, 0.20, 0.10]     # each layer misses 10–20% of defects on its own
print(f"{escape_rate(layers, 0.0):.4%}")   # 0.3000%  — genuinely independent
print(f"{escape_rate(layers, 0.5):.4%}")   # 10.1500% — same model, same prompt family
print(f"{escape_rate(layers, 0.9):.4%}")   # 18.0300% — three names for one check
```

Thirty defects per ten thousand versus eighteen hundred. Same three layers, same individual miss rates — the entire difference is whether the layers were designed to fail differently. So make independence a **design requirement and a measured quantity**: vary the model family, vary the modality (a deterministic pixel-diff check catches artifacts no vision model reasons about), vary the input framing (one layer sees the pair, one sees the edit alone), and then **measure the residual correlation** on your labeled set by checking how often layers agree on the cases they get wrong.

### The Closed Loop: Continuous Learning Without Goodharting Yourself

The most advanced part of the system is the continuous-learning pipeline, which exists to fight model and data drift: it **samples production data**, benchmarks it against a **golden dataset of human-labeled examples**, and runs a **diagnosis agent** that identifies what regressed and triggers prompt optimization and configuration updates — without manual intervention. Merchant feedback, internal dogfooding, and design-team critique feed into the same diagnosis layer.

This is the right architecture, and it comes with one danger that must be engineered against explicitly:

> **A loop that optimizes prompts against a benchmark, and also decides when the benchmark is satisfied, will eventually optimize the benchmark rather than the product.** This is Goodhart's law with a service account.

The mitigations are cheap, and they are the difference between a self-improving system and a self-congratulating one:

| Guardrail | What it prevents |
|---|---|
| **Frozen holdout the optimizer never queries** — a slice of the golden set that only the promotion gate reads | Prompt optimization overfitting to the specific images in the visible benchmark |
| **Human re-label cadence on a rotating sample** | Golden-set labels drifting as they get quietly "corrected" toward what the system already does |
| **Offline gate → canary → promote**, never optimizer-to-production | A confident diagnosis agent shipping a regression at marketplace scale |
| **Every config version pinned to the eval run that promoted it** | Un-diagnosable regressions: "which prompt was live when the complaint rate moved?" |
| **Stratified golden set** (cuisine, region, lighting, merchant size, camera class) | Aggregate improvement that hides a regression on a subpopulation — e.g. a lighting model tuned on bright studio-like inputs quietly failing on dimly-lit home kitchens |

The stratification requirement is not just fairness hygiene here — it is how you detect **homogenization**, the marketplace-level failure that no per-image metric can see. Two portfolio-level metrics are worth carrying permanently:

```python
# Marketplace diversity guard: is enhancement making everything look the same?
# Run on embeddings of published images, pre- vs post-enhancement, per cuisine.
import numpy as np

def mean_pairwise_distance(embeddings: np.ndarray) -> float:
    """Higher = more visually diverse portfolio."""
    n = len(embeddings)
    normed = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sims = normed @ normed.T
    return float(1 - (sims.sum() - n) / (n * (n - 1)))

def homogenization_delta(before: np.ndarray, after: np.ndarray) -> float:
    """Negative = the enhancement pipeline is collapsing visual variety."""
    return mean_pairwise_distance(after) - mean_pairwise_distance(before)
```

Wire `homogenization_delta` into the promotion gate as a **guardrail metric with a hard floor**, alongside merchant opt-out rate and complaint rate. A prompt change that lifts per-image quality by 3% and drops portfolio diversity by 15% should not be promotable, and only an explicit guardrail will stop it — the quality metric alone will wave it through.

### Observability: Logging Is the Deliverable

Gupta and Chopra make a point that deserves more attention than it usually gets: **logging is critical**, and the end-to-end orchestration is logged as a **flat JSON structure** specifically so that both technical and non-technical teams can diagnose issues at scale.

The word doing the work is *flat*. Nested trace objects are the natural shape of an agentic pipeline and the wrong shape for the people who need to answer questions about it. A flat, one-row-per-run schema is queryable in SQL by an ops analyst, pivotable in a spreadsheet by a design reviewer, and joinable to business metrics without a parsing step:

```json
{
  "run_id": "img_7f3a91",
  "merchant_id": "m_44812",
  "cuisine": "thai",
  "region": "us-west",
  "pipeline_version": "2026-07-14.3",
  "router_score": 0.61,
  "router_decision": "enhance",
  "router_latency_ms": 240,
  "edit_attempts": 2,
  "qa_fail_reason_attempt_1": "lighting",
  "qa_fail_reason_attempt_2": null,
  "veto_triggered": false,
  "final_verdict": "published",
  "guardrail_layer_flags": "none",
  "total_cost_usd": 0.0412,
  "total_latency_ms": 8830,
  "human_review": false
}
```

One row per image, one column per decision, every column answerable by someone who has never read the code. `SELECT qa_fail_reason_attempt_1, COUNT(*) FROM runs WHERE pipeline_version = '2026-07-14.3' GROUP BY 1` is a regression triage in one line. That is the whole argument for flatness — and note that the schema is designed so a *reason* is logged next to every failure, not just a boolean. A pipeline that logs `passed: false` and nothing else has told you that something is wrong and given you no way to find it.

### Connecting Eval Metrics to the Business Metric

The north star is **conversion rate** — did the enhanced photo cause more people to order? Everything upstream (QA pass rate, gate precision, aesthetic scores) is a **proxy**, and proxies must be periodically re-validated against the thing they proxy for:

1. **Run the experiment.** A holdout of eligible images stays un-enhanced; conversion difference is the ground truth.
2. **Correlate the proxy.** Do images that scored well on the aesthetic gate actually convert better than images that barely passed? If the correlation is near zero, your gate is measuring taste, not commerce — and you should either fix the rubric or stop treating its score as a launch criterion.
3. **Watch the guardrails independently.** Conversion can rise while merchant trust falls. Opt-out rate, complaint rate, and diversity delta are not tie-breakers; they are vetoes at the business level, exactly as faithfulness is a veto at the image level.

This is the loop that most AI teams never close, and it is the one that determines whether the eval system is a science project or a product function.

### Key Insights

1. **Decompose the pipeline, decompose the eval.** A routing classifier, a generative loop, and a guardrail stack are three different measurement problems. One end-to-end score cannot tell you which one broke.
2. **Optimize the gate for the asymmetric error, then pay to measure the invisible one.** Recall-optimized routing is correct — but the false-negative rate is structurally unobservable in production data and has to be bought with a labeled audit of the rejected pool.
3. **Integrity constraints are vetoes, not weighted dimensions.** The moment faithfulness has a weight, it has an exchange rate. Enforce vetoes in code, above the model.
4. **Your judge is a classifier with a κ score.** Criteria where the gate and humans agree (faithfulness) can gate a launch; criteria where they don't (plating) are telemetry until the guideline is fixed.
5. **Swiss cheese only works if the holes are in different places.** Correlated layers multiply cost, not protection — vary model, modality, and framing, and measure the residual correlation.
6. **Flat logs are an eval artifact.** One row per run, a reason next to every failure, and non-engineers can do their own triage.
7. **A self-optimizing loop needs a benchmark it cannot touch.** Frozen holdout, human re-label cadence, canary before promote — otherwise the diagnosis agent optimizes the scoreboard.
8. **Population metrics catch what per-item metrics cannot.** Marketplace homogenization is invisible to every per-image score and fatal to the product thesis; it needs its own guardrail with its own floor.

---

## Summary: Key Takeaways

### 1. Start with the Right Dimensions
Each use case has different priorities:
- Customer support: Helpfulness + Empathy
- Code generation: Correctness + Security
- RAG: Groundedness + Attribution
- Moderation: Precision vs Recall balance
- Frontier release: Capability + dangerous capability + alignment — measured on separate configurations
- Economic deliverables: Blind expert pairwise win rate
- Long-horizon agents: Outcome metrics + reliability (pass^k)
- Generative media: Scored aesthetics **plus** vetoed integrity constraints — never on one scale

### 2. Layer Your Evaluation
```
Fast + Cheap → Slower + Expensive → Human Review
```

### 3. Close the Feedback Loop
Production feedback is your most valuable data source.

### 4. Measure What Matters to Users
Optimize for user outcomes, not just model metrics.

### 5. Automate for CI/CD
Evaluation gates prevent regressions before they reach users.

### 6. A Score Is Never Just a Score
"SOTA" is a function of (model, scaffold, effort setting, data split). Mid-2026 SWE-bench Pro claims span 80.3% (Mythos 5 — Fable 5's unsafeguarded configuration — on the vendor's own harness and the public split, [Fable 5 system card §8.2](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)) down to ~47% (Opus 4.6 on the never-released commercial split — [Scale leaderboard](https://scale.com/leaderboard/swe_bench_pro_commercial)). Never quote — or gate a release on — a number without all four.

### 7. Reliability Beats Luck
pass@k rewards one lucky run; pass^k demands consistency every time. Deployed agents live and die on pass^k (Case Studies 5 and 9).

### 8. Every Gate Censors the Data Behind It
A router, filter, or QA gate makes the errors it *lets through* measurable and the errors it *blocks* invisible. Budget for a labeled audit of the rejected pool, or accept that half your confusion matrix is a guess (Case Study 10).

### 9. Defense in Depth Requires Uncorrelated Depth
Three layers built from the same model and the same prompt family are one layer billed three times. Vary model, modality, and framing — then measure the residual correlation (Case Study 10).

---

## Exercises

### Exercise 1: Design Your Own Eval System
Choose an AI application (chatbot, summarizer, translator, etc.) and design:
- Evaluation dimensions and weights
- Test dataset structure
- Automated evaluators
- Human review workflow
- CI/CD integration plan

### Exercise 2: Cold Start Simulation
Take a new domain and create an evaluation system from scratch:
- Generate synthetic data
- Bootstrap with LLM labels
- Validate with simulated human feedback
- Measure coverage improvement over iterations

### Exercise 3: Production Monitoring
Design a real-time monitoring system that:
- Samples production traffic
- Runs async evaluations
- Detects quality regressions
- Triggers alerts and rollbacks

### Exercise 4: System-Card Teardown
Pick a 2026 frontier system card — [Fable 5 / Mythos 5](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf) or [GPT-5.5](https://deploymentsafety.openai.com/gpt-5-5/gpt-5-5.pdf) — and map it to the stack in Case Study 7:
- Which evals are capability, which are safety, and which are meta-evaluation (eval-awareness, sandbagging)?
- What harness settings are disclosed (effort, trials, context, scaffold)?
- Which numbers could you fairly compare across labs, and which are scaffold-dependent?
- What would *you* have evaluated that the card omits?

---

## Conclusion

Eval engineering is the discipline that makes AI systems reliable. Through these case studies, we've seen that:

1. **Every domain has unique evaluation needs**
2. **Multi-stage pipelines balance quality and cost**
3. **Human feedback is essential but expensive**
4. **CI/CD integration prevents regressions**
5. **Continuous improvement requires closed loops**
6. **Frontier labs face the same problems at larger scale** — their system cards are free, deeply documented case studies; read every new one

The best evaluation systems are invisible to users but essential to the team—they catch problems before users experience them and guide improvements systematically.

---

**Next:** [Module 9 — LangChain Examples](../09-langchain-examples/README.md) turns the patterns from these case studies into runnable LangChain/LangSmith code.



