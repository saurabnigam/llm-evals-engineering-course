# Module 5: Scaling & Optimization

## 5.1 The Scaling Challenge

As your AI system grows, your evaluation system must scale with it. This module covers strategies for running evaluations efficiently at scale.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SCALING DIMENSIONS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            ▲                                                 │
│                           /│\                                                │
│                          / │ \                                               │
│                         /  │  \                                              │
│                        /   │   \      Volume                                 │
│                       /    │    \     (samples/day)                          │
│                      /     │     \                                           │
│                     /      │      \                                          │
│                    /───────┼───────\                                         │
│                   /        │        \                                        │
│                  /         │         \    Complexity                         │
│                 /          │          \   (evaluators)                       │
│                /           │           \                                     │
│               /────────────┼────────────\                                    │
│              /             │             \                                   │
│             ◄──────────────┼──────────────►                                 │
│                            │                                                 │
│                       Latency                                                │
│                    (time to result)                                          │
│                                                                              │
│  THE TRILEMMA: You can optimize for two, but the third suffers              │
│  • High Volume + Low Latency = Simple evaluators only                       │
│  • High Volume + Complex Evals = High latency                               │
│  • Low Latency + Complex Evals = Low volume                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 Cost Optimization

### 5.2.1 The Cost Equation

```
Total Eval Cost = (LLM Calls × Cost per Call) + (Compute × Hours) + (Human Labels × Cost per Label)
```

### Cost Breakdown Example

| Component | Unit Cost | Daily Volume | Daily Cost |
|-----------|-----------|--------------|------------|
| GPT-4o Judge | $0.015/call | 10,000 calls | $150 |
| GPT-4o-mini | $0.0003/call | 50,000 calls | $15 |
| Compute (GPU) | $2/hour | 24 hours | $48 |
| Human Labels | $0.50/label | 200 labels | $100 |
| **Total** | | | **$313/day** |

### 5.2.2 Cost Reduction Strategies

```python
class CostOptimizedEvaluator:
    """Minimize cost while maintaining quality"""
    
    def __init__(self):
        # Tiered models by cost
        self.models = {
            'cheap': {'name': 'gpt-4o-mini', 'cost': 0.0003},
            'medium': {'name': 'gpt-4o', 'cost': 0.005},
            'expensive': {'name': 'gpt-4o', 'cost': 0.015}  # with more context
        }
        
        # Cache for repeated queries
        self.cache = {}
        
        # Budget tracking
        self.daily_budget = 100.0  # dollars
        self.daily_spend = 0.0
    
    def evaluate_with_budget(self, samples: List[dict]) -> List[dict]:
        """Evaluate within budget constraints"""
        
        results = []
        remaining_budget = self.daily_budget - self.daily_spend
        
        # Prioritize samples (most important first)
        prioritized = self.prioritize_samples(samples)
        
        for sample in prioritized:
            if remaining_budget <= 0:
                results.append({
                    'sample_id': sample['id'],
                    'status': 'budget_exceeded',
                    'score': None
                })
                continue
            
            # Choose model tier based on remaining budget
            tier = self.select_model_tier(remaining_budget, len(prioritized) - len(results))
            
            # Check cache first
            cache_key = self.get_cache_key(sample)
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
                continue
            
            # Run evaluation
            result = self.evaluate_single(sample, tier)
            remaining_budget -= self.models[tier]['cost']
            self.daily_spend += self.models[tier]['cost']
            
            # Cache result
            self.cache[cache_key] = result
            results.append(result)
        
        return results
    
    def prioritize_samples(self, samples: List[dict]) -> List[dict]:
        """Order samples by importance"""
        
        def priority_score(sample):
            score = 0
            
            # Higher priority for certain categories
            if sample.get('category') == 'safety':
                score += 100
            if sample.get('category') == 'production_issue':
                score += 50
            
            # Recent samples are more important
            if sample.get('is_recent'):
                score += 20
            
            # New patterns we haven't seen
            if sample.get('is_novel'):
                score += 30
            
            return score
        
        return sorted(samples, key=priority_score, reverse=True)
    
    def select_model_tier(self, remaining_budget: float, remaining_samples: int) -> str:
        """Dynamically select model tier based on budget"""
        
        if remaining_samples == 0:
            return 'cheap'
        
        budget_per_sample = remaining_budget / remaining_samples
        
        if budget_per_sample >= 0.01:
            return 'expensive'
        elif budget_per_sample >= 0.003:
            return 'medium'
        else:
            return 'cheap'
    
    def get_cache_key(self, sample: dict) -> str:
        """Generate cache key for a sample"""
        import hashlib
        
        content = f"{sample['input']}|{sample.get('output', '')}"
        return hashlib.sha256(content.encode()).hexdigest()
```

### 5.2.3 Caching Strategies

```python
import redis
from functools import lru_cache
import hashlib
import json

class EvalCache:
    """Multi-level caching for evaluations"""
    
    def __init__(self, redis_url: str = None):
        # L1: In-memory LRU cache
        self.memory_cache = {}
        
        # L2: Redis for distributed caching
        self.redis = redis.from_url(redis_url) if redis_url else None
        
        # Cache TTLs
        self.memory_ttl = 3600  # 1 hour
        self.redis_ttl = 86400  # 24 hours
    
    def get(self, key: str) -> Optional[dict]:
        """Get from cache (L1 -> L2)"""
        
        # Check L1
        if key in self.memory_cache:
            return self.memory_cache[key]['value']
        
        # Check L2
        if self.redis:
            cached = self.redis.get(f"eval:{key}")
            if cached:
                value = json.loads(cached)
                # Promote to L1
                self.memory_cache[key] = {'value': value}
                return value
        
        return None
    
    def set(self, key: str, value: dict):
        """Set in cache (both levels)"""
        
        # L1
        self.memory_cache[key] = {'value': value}
        
        # L2
        if self.redis:
            self.redis.setex(
                f"eval:{key}", 
                self.redis_ttl, 
                json.dumps(value)
            )
    
    @staticmethod
    def cache_key(input_text: str, model: str, evaluator: str) -> str:
        """Generate unique cache key"""
        content = f"{model}|{evaluator}|{input_text}"
        return hashlib.sha256(content.encode()).hexdigest()

class CachedLLMJudge:
    """LLM Judge with caching"""
    
    def __init__(self, cache: EvalCache, model: str = "gpt-4o"):
        self.cache = cache
        self.model = model
        self.client = OpenAI()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def evaluate(self, input_text: str, output: str, criteria: str) -> dict:
        # Generate cache key
        key = self.cache.cache_key(
            input_text=f"{input_text}|{output}|{criteria}",
            model=self.model,
            evaluator="llm_judge"
        )
        
        # Check cache
        cached = self.cache.get(key)
        if cached:
            self.hits += 1
            return cached
        
        # Cache miss - run evaluation
        self.misses += 1
        result = self._run_evaluation(input_text, output, criteria)
        
        # Cache result
        self.cache.set(key, result)
        
        return result
    
    def _run_evaluation(self, input_text: str, output: str, criteria: str) -> dict:
        # Actual LLM call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": f"Evaluate: {output}\nCriteria: {criteria}\nReturn JSON with 'score' and 'reasoning'"
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def get_cache_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0,
            'estimated_savings': self.hits * 0.015  # Assuming $0.015 per call
        }
```

---

## 5.3 Parallel Processing

### 5.3.1 Async Evaluation Pipeline

```python
import asyncio
from asyncio import Semaphore
from typing import List
import aiohttp

class AsyncEvalPipeline:
    """Asynchronous evaluation for high throughput"""
    
    def __init__(self, 
                 max_concurrent: int = 100,
                 timeout: float = 30.0):
        self.semaphore = Semaphore(max_concurrent)
        self.timeout = timeout
        self.session = None
    
    async def evaluate_batch(self, samples: List[dict]) -> List[dict]:
        """Evaluate a batch of samples concurrently"""
        
        async with aiohttp.ClientSession() as self.session:
            tasks = [
                self.evaluate_single(sample)
                for sample in samples
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed = []
        for sample, result in zip(samples, results):
            if isinstance(result, Exception):
                processed.append({
                    'sample_id': sample['id'],
                    'status': 'error',
                    'error': str(result)
                })
            else:
                processed.append(result)
        
        return processed
    
    async def evaluate_single(self, sample: dict) -> dict:
        """Evaluate a single sample with rate limiting"""
        
        async with self.semaphore:
            try:
                result = await asyncio.wait_for(
                    self._call_evaluator(sample),
                    timeout=self.timeout
                )
                return result
            except asyncio.TimeoutError:
                return {
                    'sample_id': sample['id'],
                    'status': 'timeout'
                }
    
    async def _call_evaluator(self, sample: dict) -> dict:
        """Make async API call to evaluator"""
        
        # Using OpenAI's async client
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Evaluate this output: {sample['output']}"
            }]
        )
        
        return {
            'sample_id': sample['id'],
            'status': 'success',
            'result': response.choices[0].message.content
        }

# Usage
async def main():
    pipeline = AsyncEvalPipeline(max_concurrent=50)
    
    samples = [{'id': i, 'output': f'Output {i}'} for i in range(1000)]
    
    import time
    start = time.time()
    results = await pipeline.evaluate_batch(samples)
    elapsed = time.time() - start
    
    print(f"Evaluated {len(results)} samples in {elapsed:.2f}s")
    print(f"Throughput: {len(results)/elapsed:.1f} samples/second")

# asyncio.run(main())
```

### 5.3.2 Distributed Evaluation with Workers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTED EVALUATION ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                           ┌─────────────┐                                   │
│                           │   Master    │                                   │
│                           │  Scheduler  │                                   │
│                           └──────┬──────┘                                   │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                      │
│              │                   │                   │                      │
│              ▼                   ▼                   ▼                      │
│        ┌──────────┐        ┌──────────┐        ┌──────────┐                │
│        │  Queue   │        │  Queue   │        │  Queue   │                │
│        │ (Redis)  │        │ (Redis)  │        │ (Redis)  │                │
│        └────┬─────┘        └────┬─────┘        └────┬─────┘                │
│             │                   │                   │                       │
│     ┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐             │
│     │               │   │               │   │               │             │
│     ▼               ▼   ▼               ▼   ▼               ▼             │
│  ┌──────┐       ┌──────┐ ┌──────┐   ┌──────┐ ┌──────┐   ┌──────┐         │
│  │Worker│       │Worker│ │Worker│   │Worker│ │Worker│   │Worker│         │
│  │  1   │       │  2   │ │  3   │   │  4   │ │  5   │   │  N   │         │
│  └──────┘       └──────┘ └──────┘   └──────┘ └──────┘   └──────┘         │
│                                                                              │
│  SCALING: Add/remove workers based on queue depth                           │
│  RESILIENCE: Failed jobs automatically retry                                │
│  MONITORING: Track throughput, latency, error rates                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
from celery import Celery
from celery.result import ResultSet
import redis

# Celery configuration
app = Celery('eval_tasks',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,  # For reliability
    worker_prefetch_multiplier=1,  # Fair distribution
    task_time_limit=300,  # 5 minute timeout
    task_soft_time_limit=240,  # Soft limit for graceful handling
)

@app.task(bind=True, max_retries=3)
def evaluate_sample(self, sample: dict, evaluator_config: dict) -> dict:
    """Celery task for evaluating a single sample"""
    
    try:
        # Load evaluator
        evaluator = load_evaluator(evaluator_config)
        
        # Run evaluation
        result = evaluator.evaluate(sample)
        
        return {
            'sample_id': sample['id'],
            'status': 'success',
            'result': result
        }
    
    except Exception as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=2 ** self.request.retries)

class DistributedEvalRunner:
    """Orchestrate distributed evaluations"""
    
    def __init__(self, evaluator_config: dict):
        self.evaluator_config = evaluator_config
        self.redis = redis.Redis()
    
    def run_batch(self, samples: List[dict], 
                  chunk_size: int = 100) -> 'EvalRun':
        """Submit batch for distributed evaluation"""
        
        # Create eval run
        run_id = f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Submit tasks in chunks
        all_tasks = []
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            tasks = [
                evaluate_sample.delay(sample, self.evaluator_config)
                for sample in chunk
            ]
            all_tasks.extend(tasks)
        
        # Store run info
        self.redis.hset(f"run:{run_id}", mapping={
            'total_tasks': len(all_tasks),
            'started_at': datetime.now().isoformat(),
            'status': 'running'
        })
        
        return DistributedEvalRun(run_id, all_tasks, self.redis)

class DistributedEvalRun:
    """Track and manage a distributed eval run"""
    
    def __init__(self, run_id: str, tasks: List, redis_client):
        self.run_id = run_id
        self.tasks = tasks
        self.redis = redis_client
        self.result_set = ResultSet(tasks)
    
    def get_progress(self) -> dict:
        """Get current progress"""
        
        completed = sum(1 for t in self.tasks if t.ready())
        failed = sum(1 for t in self.tasks if t.failed())
        
        return {
            'total': len(self.tasks),
            'completed': completed,
            'failed': failed,
            'pending': len(self.tasks) - completed,
            'progress_pct': completed / len(self.tasks) * 100
        }
    
    def wait_for_completion(self, timeout: int = None) -> List[dict]:
        """Block until all tasks complete"""
        
        return self.result_set.join(timeout=timeout)
    
    def get_results(self) -> List[dict]:
        """Get all results (only call after completion)"""
        
        results = []
        for task in self.tasks:
            if task.successful():
                results.append(task.result)
            else:
                results.append({
                    'status': 'failed',
                    'error': str(task.result)
                })
        
        return results
```

---

## 5.4 Batch Processing Optimization

### 5.4.1 Batched LLM Calls

```python
class BatchedLLMEvaluator:
    """Batch multiple evaluations into single API calls"""
    
    def __init__(self, 
                 model: str = "gpt-4o-mini",
                 batch_size: int = 10):
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
    
    def evaluate_batch(self, samples: List[dict]) -> List[dict]:
        """Evaluate multiple samples in batched calls"""
        
        all_results = []
        
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]
            batch_results = self._evaluate_batch_single_call(batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def _evaluate_batch_single_call(self, batch: List[dict]) -> List[dict]:
        """Evaluate a batch in a single LLM call"""
        
        # Construct batched prompt
        prompt = """Evaluate each of the following outputs. Return a JSON array with one result per output.

"""
        for i, sample in enumerate(batch):
            prompt += f"""
---
OUTPUT {i+1}:
Input: {sample['input']}
Response: {sample['output']}
---
"""
        
        prompt += """
For each output, provide:
- "index": the output number
- "score": 0.0 to 1.0
- "issues": list of any problems found

Return as: {"results": [...]}
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        parsed = json.loads(response.choices[0].message.content)
        
        # Map results back to samples
        results = []
        for i, sample in enumerate(batch):
            matching = next(
                (r for r in parsed['results'] if r.get('index') == i + 1),
                {'score': None, 'issues': ['No result returned']}
            )
            results.append({
                'sample_id': sample['id'],
                'score': matching.get('score'),
                'issues': matching.get('issues', [])
            })
        
        return results

# Cost comparison
# Without batching: 100 samples × $0.015 = $1.50
# With batching (10 per call): 10 calls × $0.03 = $0.30
# Savings: 80%
```

### 5.4.2 Smart Batching by Similarity

```python
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

class SimilarityBatcher:
    """Group similar samples for more efficient batched evaluation"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 batch_size: int = 10):
        self.embedder = SentenceTransformer(embedding_model)
        self.batch_size = batch_size
    
    def create_smart_batches(self, samples: List[dict]) -> List[List[dict]]:
        """Group similar samples into batches"""
        
        # Embed all samples
        texts = [s['input'] + ' ' + s.get('output', '') for s in samples]
        embeddings = self.embedder.encode(texts)
        
        # Cluster into groups
        n_clusters = len(samples) // self.batch_size
        clustering = AgglomerativeClustering(
            n_clusters=max(1, n_clusters),
            linkage='ward'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Group samples by cluster
        clusters = {}
        for sample, label in zip(samples, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(sample)
        
        # Create batches from clusters
        batches = []
        current_batch = []
        
        for cluster_samples in clusters.values():
            for sample in cluster_samples:
                current_batch.append(sample)
                if len(current_batch) >= self.batch_size:
                    batches.append(current_batch)
                    current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def evaluate_with_smart_batching(self, 
                                     samples: List[dict],
                                     evaluator: 'BatchedLLMEvaluator') -> List[dict]:
        """Evaluate using smart batching"""
        
        batches = self.create_smart_batches(samples)
        
        all_results = []
        for batch in batches:
            # Create context-aware prompt
            common_theme = self._extract_common_theme(batch)
            results = evaluator.evaluate_batch_with_context(batch, common_theme)
            all_results.extend(results)
        
        return all_results
    
    def _extract_common_theme(self, batch: List[dict]) -> str:
        """Extract common theme from a batch for better prompting"""
        
        # Simple: use most common category
        categories = [s.get('category', 'general') for s in batch]
        most_common = max(set(categories), key=categories.count)
        return most_common
```

---

## 5.5 Hierarchical Evaluation for Scale

### 5.5.1 Progressive Filtering

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROGRESSIVE FILTERING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  100,000 samples                                                             │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 1: Fast Heuristics (1ms per sample)                          │    │
│  │  - Length checks                                                    │    │
│  │  - Format validation                                                │    │
│  │  - Keyword filters                                                  │    │
│  │  Cost: ~$0 | Time: 100 seconds                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       │ 80,000 pass (80%)                                                   │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 2: Cheap ML Classifier (10ms per sample)                     │    │
│  │  - Distilled quality classifier                                     │    │
│  │  - Binary pass/fail                                                 │    │
│  │  Cost: ~$10 | Time: 15 minutes                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       │ 30,000 pass (38%)                                                   │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 3: Fast LLM (GPT-4o-mini, 50ms per sample)                   │    │
│  │  - Quick quality check                                              │    │
│  │  - Score 0-1                                                        │    │
│  │  Cost: ~$9 | Time: 25 minutes                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       │ 5,000 uncertain (17%)                                               │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 4: Full LLM Judge (GPT-4o, 500ms per sample)                 │    │
│  │  - Detailed evaluation                                              │    │
│  │  - Rubric-based scoring                                             │    │
│  │  Cost: ~$75 | Time: 45 minutes                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  TOTAL: $94 instead of $1,500 (93% savings)                                 │
│  TIME: ~90 minutes instead of ~14 hours                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class ProgressiveEvalPipeline:
    """Multi-stage evaluation with early exits"""
    
    def __init__(self):
        self.stages = [
            HeuristicFilter(name='heuristics'),
            ClassifierStage(name='classifier', model='quality-classifier-v1'),
            LLMStage(name='quick_llm', model='gpt-4o-mini'),
            LLMStage(name='full_llm', model='gpt-4o')
        ]
        
        # Thresholds for early exit
        self.pass_thresholds = {
            'heuristics': 1.0,  # Binary
            'classifier': 0.9,
            'quick_llm': 0.85,
            'full_llm': 0.5
        }
        
        self.fail_thresholds = {
            'heuristics': 0.0,
            'classifier': 0.1,
            'quick_llm': 0.2,
            'full_llm': 0.0
        }
    
    def evaluate(self, samples: List[dict]) -> List[dict]:
        """Run progressive evaluation"""
        
        current_samples = samples
        results = {s['id']: {'stages': {}, 'final_score': None, 'final_stage': None} 
                   for s in samples}
        
        for stage in self.stages:
            if not current_samples:
                break
            
            # Run stage
            stage_results = stage.evaluate_batch(current_samples)
            
            # Process results
            next_samples = []
            for sample, result in zip(current_samples, stage_results):
                results[sample['id']]['stages'][stage.name] = result
                
                score = result['score']
                
                # Early exit - pass
                if score >= self.pass_thresholds[stage.name]:
                    results[sample['id']]['final_score'] = score
                    results[sample['id']]['final_stage'] = stage.name
                    results[sample['id']]['outcome'] = 'pass'
                
                # Early exit - fail
                elif score <= self.fail_thresholds[stage.name]:
                    results[sample['id']]['final_score'] = score
                    results[sample['id']]['final_stage'] = stage.name
                    results[sample['id']]['outcome'] = 'fail'
                
                # Continue to next stage
                else:
                    next_samples.append(sample)
            
            current_samples = next_samples
        
        # Any remaining samples get last stage score
        for sample in current_samples:
            last_stage = self.stages[-1].name
            results[sample['id']]['final_score'] = results[sample['id']]['stages'][last_stage]['score']
            results[sample['id']]['final_stage'] = last_stage
        
        return list(results.values())
    
    def get_efficiency_report(self, results: List[dict]) -> dict:
        """Report on filtering efficiency"""
        
        stage_counts = {}
        for result in results:
            stage = result['final_stage']
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        return {
            'total_samples': len(results),
            'final_stage_distribution': stage_counts,
            'efficiency': {
                'early_exits': len(results) - stage_counts.get(self.stages[-1].name, 0),
                'full_evaluation': stage_counts.get(self.stages[-1].name, 0),
                'early_exit_rate': (len(results) - stage_counts.get(self.stages[-1].name, 0)) / len(results) * 100
            }
        }
```

---

## 5.6 Infrastructure for Scale

### 5.6.1 Kubernetes Deployment

```yaml
# eval-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: eval-worker
spec:
  replicas: 10  # Scale based on queue depth
  selector:
    matchLabels:
      app: eval-worker
  template:
    metadata:
      labels:
        app: eval-worker
    spec:
      containers:
      - name: worker
        image: eval-worker:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: eval-secrets
              key: redis-url
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: eval-secrets
              key: openai-key
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: eval-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: eval-worker
  minReplicas: 2
  maxReplicas: 50
  metrics:
  - type: External
    external:
      metric:
        name: redis_queue_length
      target:
        type: AverageValue
        averageValue: 100  # Scale up when queue > 100 per worker
```

### 5.6.2 Monitoring and Alerting

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Metrics
eval_requests_total = Counter(
    'eval_requests_total',
    'Total evaluation requests',
    ['evaluator', 'status']
)

eval_latency_seconds = Histogram(
    'eval_latency_seconds',
    'Evaluation latency in seconds',
    ['evaluator'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

eval_queue_depth = Gauge(
    'eval_queue_depth',
    'Current queue depth'
)

eval_score_distribution = Histogram(
    'eval_score_distribution',
    'Distribution of evaluation scores',
    ['evaluator'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

class InstrumentedEvaluator:
    """Evaluator with Prometheus metrics"""
    
    def __init__(self, evaluator, name: str):
        self.evaluator = evaluator
        self.name = name
    
    def evaluate(self, sample: dict) -> dict:
        start_time = time.time()
        
        try:
            result = self.evaluator.evaluate(sample)
            
            # Record metrics
            eval_requests_total.labels(
                evaluator=self.name, 
                status='success'
            ).inc()
            
            eval_score_distribution.labels(
                evaluator=self.name
            ).observe(result.get('score', 0))
            
            return result
            
        except Exception as e:
            eval_requests_total.labels(
                evaluator=self.name,
                status='error'
            ).inc()
            raise
            
        finally:
            elapsed = time.time() - start_time
            eval_latency_seconds.labels(
                evaluator=self.name
            ).observe(elapsed)
```

---

## 5.6b Worked Examples (2026)

Three levers that compound to drop eval cost 10–20× without sacrificing signal.

#### Example 1 — Judge cascade (route by difficulty)

Most samples don’t need the smartest judge. Cascade from cheapest → strongest, escalating only on ambiguity.

```python
from anthropic import Anthropic
from openai import OpenAI

anthropic, openai = Anthropic(), OpenAI()

CHEAP_JUDGE = ("openai",   "gpt-4o-mini",        0.0002)   # $/1k input tokens
MID_JUDGE   = ("anthropic","claude-haiku-4-5",   0.0008)
STRONG      = ("anthropic","claude-sonnet-4-5",  0.003)

def judge(prompt: str, model_tuple) -> tuple[float, float]:
    """Return (score 0-1, judge confidence 0-1)."""
    vendor, model, _ = model_tuple
    if vendor == "openai":
        r = openai.chat.completions.create(model=model, temperature=0,
                logprobs=True, top_logprobs=2,
                messages=[{"role":"user","content":prompt}])
        # Use top-2 logprob gap as a proxy for confidence
        top = r.choices[0].logprobs.content[0].top_logprobs
        conf = abs(top[0].logprob - top[1].logprob)
        return (1.0 if "good" in r.choices[0].message.content.lower() else 0.0,
                min(1.0, conf))
    # ... analogous Anthropic branch

def cascading_judge(prompt: str) -> float:
    score, conf = judge(prompt, CHEAP_JUDGE)
    if conf > 0.8:        return score
    score, conf = judge(prompt, MID_JUDGE)
    if conf > 0.8:        return score
    score, _    = judge(prompt, STRONG)
    return score

# Empirically: ~75% resolved at cheap, ~20% at mid, ~5% at strong.
# Blended cost ≈ 0.75*0.0002 + 0.20*0.0008 + 0.05*0.003 = $0.00046 / sample
# vs flat strong-judge $0.003 / sample → 6.5× cheaper, same final agreement.
```

#### Example 2 — OpenAI Batch API + Anthropic Message Batches (50% off)

Both providers offer ~50%-discounted batch endpoints with 24h SLA — perfect for nightly eval runs.

```python
# OpenAI Batch — prepare JSONL, submit, poll, download.
import json, time
from openai import OpenAI
client = OpenAI()

with open("requests.jsonl", "w") as f:
    for i, sample in enumerate(eval_dataset):
        f.write(json.dumps({
            "custom_id": f"sample-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-4o-mini", "temperature": 0,
                     "messages": [{"role":"user","content":sample["input"]}]},
        }) + "\n")

file = client.files.create(file=open("requests.jsonl","rb"), purpose="batch")
job  = client.batches.create(input_file_id=file.id,
                             endpoint="/v1/chat/completions",
                             completion_window="24h")
while (job := client.batches.retrieve(job.id)).status not in ("completed","failed"):
    time.sleep(60)
results = client.files.content(job.output_file_id).text
# 50% off list price — use this for *any* eval that doesn't need synchronous results.
```

For Anthropic, use `client.messages.batches.create(...)` with the same shape — also 50% off.

#### Example 3 — Prompt-cache aware judge (Anthropic prompt caching)

If 90% of your judge prompt is a long rubric, *cache* it. You pay full price once, then 10% on subsequent calls within 5 minutes.

```python
from anthropic import Anthropic
client = Anthropic()

LONG_RUBRIC = open("rubric.md").read()   # 8–12k tokens of detailed criteria + few-shots

def judge(sample_input: str, sample_output: str) -> str:
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        system=[
            {"type": "text",
             "text": LONG_RUBRIC,
             "cache_control": {"type": "ephemeral"}},   # <— the magic
        ],
        messages=[{"role": "user",
                   "content": f"INPUT:\n{sample_input}\n\nOUTPUT:\n{sample_output}\n\n"
                              "Score per the rubric. Return JSON."}],
    )
    return msg.content[0].text

# First call:  full price on 12k system tokens + small user.
# Calls 2..N within 5 min: 10% of system token price + full user.
# On a 1,000-sample run → ~85% reduction in input-token spend.
```

Combine all three (cascade + batch + prompt cache) and a $300 nightly eval becomes a $15 nightly eval. That’s often the difference between “eval-driven development” and “we’ll run it before release”.

---

## 5.7 Exercises

### Exercise 1: Cost Optimization
Given a budget of $100/day and 50,000 samples to evaluate:
- Design a cost-optimized evaluation strategy
- Calculate expected cost savings vs naive approach
- Implement the batching and caching components

### Exercise 2: Scaling Experiment
Benchmark your evaluation pipeline:
- Measure throughput at different concurrency levels
- Identify bottlenecks
- Implement optimizations and measure improvement

### Exercise 3: Distributed System Design
Design a distributed evaluation system that can:
- Handle 1M samples/day
- Maintain < 5 minute latency for priority samples
- Cost less than $500/day
- Include architecture diagram and component specifications

---

## Next Module
→ [Module 6: Feedback Systems & Active Learning](../06-feedback-loops/README.md)

