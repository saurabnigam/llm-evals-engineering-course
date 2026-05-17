# Module 8: Real-World Case Studies

## Overview

This module presents detailed case studies of evaluation systems in production. Each case study includes the problem, architecture, implementation details, and lessons learned.

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
            'accuracy': AccuracyEvaluator(model='gpt-4o'),
            'helpfulness': HelpfulnessEvaluator(model='gpt-4o-mini'),
            'empathy': ToneEvaluator(model='gpt-4o-mini')
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
A mid-size DTC retailer ships a Claude-Sonnet-4.5 agent with three tools: `lookup_order`, `issue_refund`, `escalate_to_human`. Eval focus is **trajectory correctness**, not just final-message correctness, because a wrong tool call (e.g. refunding twice) costs real money.

### Eval Design

| Layer | Metric | Target |
|-------|--------|--------|
| Outcome | `agent_goal_accuracy` (RAGAS) on 200 scripted scenarios | ≥ 0.92 |
| Tool selection | `tool_call_accuracy` per turn | ≥ 0.97 |
| Tool args | `tool_call_f1` (exact `order_id` + amount within $0.01) | ≥ 0.99 |
| Safety | Adversarial set: “refund $500 to a different account” × 30 | 0 unauthorized actions |
| Efficiency | Median steps to resolution | ≤ 4 |
| Recovery | Synthetic tool errors injected on 10% of runs → does the agent recover? | ≥ 0.90 success |

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

---

## Case Study 6: Reasoning-Model Math Tutor

### Context
A tutoring product uses an o-series reasoning model to walk students through problems. Both *answer correctness* and *reasoning quality* matter — a right answer with bad reasoning teaches nothing.

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
        model="o4-mini", reasoning_effort="low",
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

## Summary: Key Takeaways

### 1. Start with the Right Dimensions
Each use case has different priorities:
- Customer support: Helpfulness + Empathy
- Code generation: Correctness + Security
- RAG: Groundedness + Attribution
- Moderation: Precision vs Recall balance

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

---

## Conclusion

Eval engineering is the discipline that makes AI systems reliable. Through these case studies, we've seen that:

1. **Every domain has unique evaluation needs**
2. **Multi-stage pipelines balance quality and cost**
3. **Human feedback is essential but expensive**
4. **CI/CD integration prevents regressions**
5. **Continuous improvement requires closed loops**

The best evaluation systems are invisible to users but essential to the team—they catch problems before users experience them and guide improvements systematically.

---

**Congratulations!** You've completed the Eval Engineering study guide. Continue learning by building your own evaluation systems and iterating based on real-world feedback.



