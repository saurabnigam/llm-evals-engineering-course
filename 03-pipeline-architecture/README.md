# Module 3: Building Evaluation Pipelines

## 3.1 Pipeline Architecture Overview

A robust evaluation pipeline is the backbone of any AI quality system. It must be reliable, scalable, and maintainable.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐            │
│  │  DATA SOURCES  │    │  EVAL ENGINE   │    │   REPORTING    │            │
│  ├────────────────┤    ├────────────────┤    ├────────────────┤            │
│  │ • Golden Sets  │───▶│ • Orchestrator │───▶│ • Dashboards   │            │
│  │ • Prod Samples │    │ • Evaluators   │    │ • Alerts       │            │
│  │ • Synth Data   │    │ • Aggregators  │    │ • Reports      │            │
│  │ • User Feedback│    │ • Validators   │    │ • Exports      │            │
│  └────────────────┘    └────────────────┘    └────────────────┘            │
│         │                     │                      │                      │
│         │                     │                      │                      │
│         ▼                     ▼                      ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         STORAGE LAYER                                │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│  │  │ Test Sets │  │ Results   │  │ Metrics   │  │ Artifacts │        │   │
│  │  │ (S3/GCS)  │  │ (DB)      │  │ (TimeSer) │  │ (Storage) │        │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         INFRASTRUCTURE                               │   │
│  │  • Task Queue (Celery/SQS)  • Compute (K8s/Lambda)  • Monitoring    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Core Components

### 3.2.1 Data Layer

```python
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class EvalSample:
    """Single evaluation sample"""
    id: str
    input: str
    expected_output: Optional[str]
    metadata: dict
    category: str
    difficulty: str
    created_at: datetime
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'input': self.input,
            'expected_output': self.expected_output,
            'metadata': self.metadata,
            'category': self.category,
            'difficulty': self.difficulty,
            'created_at': self.created_at.isoformat()
        }

class DataSource(ABC):
    """Abstract base for evaluation data sources"""
    
    @abstractmethod
    def load(self, filters: Optional[dict] = None) -> Iterator[EvalSample]:
        pass
    
    @abstractmethod
    def count(self, filters: Optional[dict] = None) -> int:
        pass

class GoldenDatasetSource(DataSource):
    """Load curated golden test sets"""
    
    def __init__(self, path: str):
        self.path = path
        self._data = None
    
    def load(self, filters: Optional[dict] = None) -> Iterator[EvalSample]:
        if self._data is None:
            self._data = self._load_from_storage()
        
        for item in self._data:
            if self._matches_filter(item, filters):
                yield EvalSample(
                    id=item['id'],
                    input=item['input'],
                    expected_output=item.get('expected_output'),
                    metadata=item.get('metadata', {}),
                    category=item.get('category', 'general'),
                    difficulty=item.get('difficulty', 'medium'),
                    created_at=datetime.fromisoformat(
                        item.get('created_at', datetime.now().isoformat())
                    )
                )
    
    def _load_from_storage(self) -> list:
        # Could be S3, GCS, local file, etc.
        with open(self.path, 'r') as f:
            return json.load(f)
    
    def _matches_filter(self, item: dict, filters: Optional[dict]) -> bool:
        if not filters:
            return True
        for key, value in filters.items():
            if item.get(key) != value:
                return False
        return True
    
    def count(self, filters: Optional[dict] = None) -> int:
        return sum(1 for _ in self.load(filters))

class ProductionSampleSource(DataSource):
    """Sample from production traffic"""
    
    def __init__(self, db_connection, sample_rate: float = 0.01):
        self.db = db_connection
        self.sample_rate = sample_rate
    
    def load(self, filters: Optional[dict] = None) -> Iterator[EvalSample]:
        # Query production logs with sampling
        query = """
            SELECT id, input, output, metadata, created_at
            FROM production_logs
            WHERE random() < %s
            AND created_at > NOW() - INTERVAL '7 days'
        """
        
        for row in self.db.execute(query, (self.sample_rate,)):
            yield EvalSample(
                id=row['id'],
                input=row['input'],
                expected_output=None,  # No ground truth for prod
                metadata=row['metadata'],
                category='production',
                difficulty='unknown',
                created_at=row['created_at']
            )
    
    def count(self, filters: Optional[dict] = None) -> int:
        return self.db.execute(
            "SELECT COUNT(*) * %s FROM production_logs",
            (self.sample_rate,)
        ).fetchone()[0]
```

### 3.2.2 Evaluation Orchestrator

```python
from typing import Dict, List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class EvalResult:
    """Result of a single evaluation"""
    sample_id: str
    scores: Dict[str, float]
    model_output: str
    evaluator_outputs: Dict[str, dict]
    latency_ms: float
    status: str  # 'success', 'error', 'timeout'
    error_message: Optional[str] = None

class EvalOrchestrator:
    """Coordinates the evaluation pipeline"""
    
    def __init__(self, 
                 model,
                 evaluators: Dict[str, 'Evaluator'],
                 max_workers: int = 10,
                 timeout: float = 30.0):
        self.model = model
        self.evaluators = evaluators
        self.max_workers = max_workers
        self.timeout = timeout
    
    def run_evaluation(self, 
                       data_source: DataSource,
                       run_name: str,
                       filters: Optional[dict] = None) -> 'EvalRun':
        """Execute full evaluation run"""
        
        run = EvalRun(
            name=run_name,
            started_at=datetime.now(),
            config={
                'evaluators': list(self.evaluators.keys()),
                'filters': filters,
                'model': str(self.model)
            }
        )
        
        samples = list(data_source.load(filters))
        logger.info(f"Starting eval run '{run_name}' with {len(samples)} samples")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_sample = {
                executor.submit(self._evaluate_sample, sample): sample
                for sample in samples
            }
            
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error evaluating {sample.id}: {e}")
                    results.append(EvalResult(
                        sample_id=sample.id,
                        scores={},
                        model_output='',
                        evaluator_outputs={},
                        latency_ms=0,
                        status='error',
                        error_message=str(e)
                    ))
                
                # Progress logging
                if len(results) % 100 == 0:
                    logger.info(f"Progress: {len(results)}/{len(samples)}")
        
        run.results = results
        run.finished_at = datetime.now()
        run.compute_aggregates()
        
        return run
    
    def _evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single sample"""
        start_time = time.time()
        
        # Get model output
        model_output = self.model.generate(sample.input)
        
        # Run all evaluators
        scores = {}
        evaluator_outputs = {}
        
        for name, evaluator in self.evaluators.items():
            try:
                result = evaluator.evaluate(
                    input=sample.input,
                    output=model_output,
                    expected=sample.expected_output,
                    metadata=sample.metadata
                )
                scores[name] = result['score']
                evaluator_outputs[name] = result
            except Exception as e:
                logger.warning(f"Evaluator {name} failed: {e}")
                scores[name] = None
                evaluator_outputs[name] = {'error': str(e)}
        
        latency_ms = (time.time() - start_time) * 1000
        
        return EvalResult(
            sample_id=sample.id,
            scores=scores,
            model_output=model_output,
            evaluator_outputs=evaluator_outputs,
            latency_ms=latency_ms,
            status='success'
        )

@dataclass
class EvalRun:
    """Container for an evaluation run"""
    name: str
    started_at: datetime
    config: dict
    results: List[EvalResult] = None
    finished_at: Optional[datetime] = None
    aggregates: Optional[dict] = None
    
    def compute_aggregates(self):
        """Compute aggregate statistics"""
        if not self.results:
            return
        
        successful = [r for r in self.results if r.status == 'success']
        
        # Aggregate by evaluator
        evaluator_scores = {}
        for result in successful:
            for evaluator, score in result.scores.items():
                if score is not None:
                    if evaluator not in evaluator_scores:
                        evaluator_scores[evaluator] = []
                    evaluator_scores[evaluator].append(score)
        
        self.aggregates = {
            'total_samples': len(self.results),
            'successful': len(successful),
            'failed': len(self.results) - len(successful),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
            'duration_seconds': (self.finished_at - self.started_at).total_seconds(),
            'evaluator_means': {
                name: sum(scores) / len(scores)
                for name, scores in evaluator_scores.items()
            },
            'evaluator_medians': {
                name: sorted(scores)[len(scores) // 2]
                for name, scores in evaluator_scores.items()
            }
        }
```

**Agent-aware orchestration: trials, not samples.** If the system under test is an *agent* (multi-step, tool-using, stochastic), the orchestrator above needs one structural change: run **k independent trials per task** and aggregate per-task, because a single run tells you almost nothing about reliability. The vocabulary standardized by Anthropic's [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (Jan 2026): a **task** is the test case + success criteria, a **trial** is one stochastic run, the **transcript** is the full record including tool calls, and the **outcome** is the actual end-state of the environment — grade that, not what the agent claims. Report **pass@k** (≥1 of k trials succeeds) when one success is enough, and **pass^k** (all k succeed) when consistency matters — a 75% per-trial agent is only ~42% reliable at pass^3. Two pipeline consequences: (1) `EvalResult` gains a `trial_index` and aggregation happens per `(task_id)` group, and (2) each trial must start from a clean, isolated environment so failures aren't correlated through shared infra (see sandboxing in 3.5).

```python
@dataclass
class TaskAggregate:
    """Per-task aggregate over k trials of an agent eval"""
    task_id: str
    trials: int
    successes: int

    @property
    def pass_at_k(self) -> bool:          # at least one success
        return self.successes >= 1

    @property
    def pass_hat_k(self) -> bool:         # every trial succeeded (pass^k)
        return self.successes == self.trials
```

### 3.2.3 Result Storage and Retrieval

```python
from abc import ABC, abstractmethod
import sqlite3
from datetime import datetime

class ResultStore(ABC):
    """Abstract storage for evaluation results"""
    
    @abstractmethod
    def save_run(self, run: EvalRun) -> str:
        pass
    
    @abstractmethod
    def load_run(self, run_id: str) -> EvalRun:
        pass
    
    @abstractmethod
    def list_runs(self, filters: Optional[dict] = None) -> List[dict]:
        pass
    
    @abstractmethod
    def compare_runs(self, run_ids: List[str]) -> dict:
        pass

class SQLiteResultStore(ResultStore):
    """SQLite-based result storage for development"""
    
    def __init__(self, db_path: str = "eval_results.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id TEXT PRIMARY KEY,
                name TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                config TEXT,
                aggregates TEXT,
                status TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                sample_id TEXT,
                scores TEXT,
                model_output TEXT,
                evaluator_outputs TEXT,
                latency_ms REAL,
                status TEXT,
                FOREIGN KEY (run_id) REFERENCES eval_runs(id)
            )
        """)
        conn.commit()
        conn.close()
    
    def save_run(self, run: EvalRun) -> str:
        import uuid
        run_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        
        # Save run metadata
        conn.execute("""
            INSERT INTO eval_runs (id, name, started_at, finished_at, config, aggregates, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            run.name,
            run.started_at.isoformat(),
            run.finished_at.isoformat() if run.finished_at else None,
            json.dumps(run.config),
            json.dumps(run.aggregates),
            'completed'
        ))
        
        # Save individual results
        for result in run.results:
            conn.execute("""
                INSERT INTO eval_results 
                (id, run_id, sample_id, scores, model_output, evaluator_outputs, latency_ms, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                run_id,
                result.sample_id,
                json.dumps(result.scores),
                result.model_output,
                json.dumps(result.evaluator_outputs),
                result.latency_ms,
                result.status
            ))
        
        conn.commit()
        conn.close()
        
        return run_id
    
    def load_run(self, run_id: str) -> EvalRun:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM eval_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            conn.close()
            raise KeyError(f"No run with id {run_id}")
        
        run = EvalRun(
            name=row['name'],
            started_at=datetime.fromisoformat(row['started_at']),
            config=json.loads(row['config']),
            finished_at=datetime.fromisoformat(row['finished_at'])
                        if row['finished_at'] else None,
            aggregates=json.loads(row['aggregates']),
        )
        run.results = [
            EvalResult(
                sample_id=r['sample_id'],
                scores=json.loads(r['scores']),
                model_output=r['model_output'],
                evaluator_outputs=json.loads(r['evaluator_outputs']),
                latency_ms=r['latency_ms'],
                status=r['status'],
            )
            for r in conn.execute(
                "SELECT * FROM eval_results WHERE run_id = ?", (run_id,)
            )
        ]
        conn.close()
        return run
    
    def list_runs(self, filters: Optional[dict] = None) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, name, started_at, finished_at, status "
            "FROM eval_runs ORDER BY started_at DESC"
        ).fetchall()
        conn.close()
        runs = [dict(r) for r in rows]
        if filters:
            runs = [r for r in runs
                    if all(r.get(k) == v for k, v in filters.items())]
        return runs
    
    def compare_runs(self, run_ids: List[str]) -> dict:
        """Compare metrics across multiple runs"""
        runs = [self.load_run(run_id) for run_id in run_ids]
        
        comparison = {
            'runs': [],
            'metrics': {}
        }
        
        for run in runs:
            comparison['runs'].append({
                'id': run_ids[runs.index(run)],
                'name': run.name,
                'date': run.started_at.isoformat()
            })
            
            for metric, value in run.aggregates.get('evaluator_means', {}).items():
                if metric not in comparison['metrics']:
                    comparison['metrics'][metric] = []
                comparison['metrics'][metric].append(value)
        
        # Calculate deltas
        comparison['deltas'] = {}
        for metric, values in comparison['metrics'].items():
            if len(values) >= 2:
                comparison['deltas'][metric] = values[-1] - values[-2]
        
        return comparison
```

---

## 3.3 Pipeline Patterns

### 3.3.1 Batch Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      BATCH EVALUATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐              │
│   │ Trigger │────▶│  Load   │────▶│ Process │────▶│ Store   │              │
│   │         │     │  Data   │     │ Batch   │     │ Results │              │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘              │
│       │                                                │                     │
│       │ • Scheduled (cron)                            │                     │
│       │ • On-demand (API)                             ▼                     │
│       │ • Git push (CI/CD)                     ┌─────────────┐             │
│       │                                        │   Report    │             │
│                                                │  Generate   │             │
│                                                └─────────────┘             │
│                                                      │                      │
│                                          ┌──────────┼──────────┐           │
│                                          ▼          ▼          ▼           │
│                                    ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│                                    │Dashboard│ │ Slack   │ │  Email  │    │
│                                    │         │ │ Alert   │ │ Report  │    │
│                                    └─────────┘ └─────────┘ └─────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
from dataclasses import dataclass
from enum import Enum
import schedule
import time

class TriggerType(Enum):
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"
    GIT_PUSH = "git_push"

@dataclass  
class PipelineConfig:
    name: str
    data_sources: List[DataSource]
    evaluators: Dict[str, 'Evaluator']
    schedule: Optional[str] = None  # Cron expression
    notification_channels: List[str] = None
    
class BatchEvalPipeline:
    """Scheduled batch evaluation pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.orchestrator = EvalOrchestrator(
            model=self._load_model(),
            evaluators=config.evaluators
        )
        self.result_store = SQLiteResultStore()
        self.notifier = Notifier(config.notification_channels)
    
    def run(self, trigger: TriggerType = TriggerType.ON_DEMAND) -> str:
        """Execute the pipeline"""
        
        run_name = f"{self.config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Combine all data sources
        all_samples = []
        for source in self.config.data_sources:
            all_samples.extend(list(source.load()))
        
        # Create a unified data source
        unified_source = InMemoryDataSource(all_samples)
        
        # Run evaluation
        run = self.orchestrator.run_evaluation(
            data_source=unified_source,
            run_name=run_name
        )
        
        # Store results
        run_id = self.result_store.save_run(run)
        
        # Generate and send report
        report = self._generate_report(run)
        self.notifier.send(report)
        
        return run_id
    
    def schedule_pipeline(self):
        """Set up scheduled execution"""
        if self.config.schedule:
            schedule.every().day.at(self.config.schedule).do(
                self.run, 
                trigger=TriggerType.SCHEDULED
            )
            
            while True:
                schedule.run_pending()
                time.sleep(60)
    
    def _generate_report(self, run: EvalRun) -> dict:
        return {
            'title': f"Eval Run: {run.name}",
            'summary': {
                'samples': run.aggregates['total_samples'],
                'success_rate': f"{run.aggregates['success_rate']*100:.1f}%",
                'duration': f"{run.aggregates['duration_seconds']:.1f}s"
            },
            'scores': run.aggregates['evaluator_means'],
            'link': f"/evals/{run.name}"
        }
```

**The "Git push" trigger grew teeth in 2025–2026.** Evals-as-CI-gates is now a first-class product feature rather than a custom script: [Braintrust's GitHub Action](https://www.braintrust.dev/articles/langsmith-vs-braintrust) runs the eval suite on every PR, posts score summaries as PR comments, and blocks merge when scores fall below configured thresholds. The practice pattern that goes with it ([MLAI Digital](https://www.mlaidigital.com/blogs/llm-evaluation-frameworks-2025-vs-2026-what-matters-now-2026)): a golden dataset of roughly 200–500 examples built from real production failures, with automated quality gates per pipeline stage. Treat the eval suite like a test suite — it runs on every change to prompts, retrieval config, or model version, not on a nightly cron alone.

### 3.3.2 Streaming Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STREAMING EVALUATION PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Production Traffic                                                          │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────┐                                                            │
│  │   Sampler   │ ◄── Sample 1% of traffic                                   │
│  │   (1%)      │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                    │
│         ▼                                                                    │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   Message   │────▶│   Worker    │────▶│  Real-time  │                   │
│  │   Queue     │     │   Pool      │     │  Store      │                   │
│  │  (Kafka)    │     │  (K8s)      │     │  (Redis)    │                   │
│  └─────────────┘     └─────────────┘     └─────────────┘                   │
│                            │                    │                           │
│                            │                    ▼                           │
│                            │             ┌─────────────┐                    │
│                            │             │ Live        │                    │
│                            │             │ Dashboard   │                    │
│                            │             └─────────────┘                    │
│                            │                                                 │
│                            ▼                                                 │
│                     ┌─────────────┐                                         │
│                     │   Anomaly   │──── Alert if scores drop                │
│                     │   Detector  │                                         │
│                     └─────────────┘                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
from kafka import KafkaConsumer, KafkaProducer
import redis
import json
from threading import Thread

class StreamingEvalPipeline:
    """Real-time evaluation on production traffic"""
    
    def __init__(self, 
                 kafka_broker: str,
                 input_topic: str,
                 output_topic: str,
                 redis_host: str,
                 evaluators: Dict[str, 'Evaluator'],
                 sample_rate: float = 0.01):
        
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=[kafka_broker],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_broker],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.redis = redis.Redis(host=redis_host)
        self.evaluators = evaluators
        self.sample_rate = sample_rate
        self.output_topic = output_topic
        
        # Anomaly detection
        self.window_size = 100
        self.alert_threshold = 0.1  # Alert if score drops 10%
    
    def start(self, num_workers: int = 4):
        """Start the streaming pipeline"""
        
        workers = []
        for i in range(num_workers):
            worker = Thread(target=self._worker_loop, args=(i,))
            worker.start()
            workers.append(worker)
        
        # Start anomaly detection thread
        anomaly_thread = Thread(target=self._anomaly_detection_loop)
        anomaly_thread.start()
        
        for worker in workers:
            worker.join()
    
    def _worker_loop(self, worker_id: int):
        """Process messages from the queue"""
        import random
        
        for message in self.consumer:
            # Probabilistic sampling
            if random.random() > self.sample_rate:
                continue
            
            data = message.value
            
            # Run evaluation
            scores = {}
            for name, evaluator in self.evaluators.items():
                try:
                    result = evaluator.evaluate(
                        input=data['input'],
                        output=data['output'],
                        expected=None
                    )
                    scores[name] = result['score']
                except Exception as e:
                    scores[name] = None
            
            # Store in Redis for real-time dashboard
            self._store_result(data['request_id'], scores)
            
            # Publish to output topic
            self.producer.send(self.output_topic, {
                'request_id': data['request_id'],
                'scores': scores,
                'timestamp': datetime.now().isoformat()
            })
    
    def _store_result(self, request_id: str, scores: dict):
        """Store result in Redis with windowing"""
        
        pipe = self.redis.pipeline()
        
        # Store individual result
        pipe.hset(f"eval:{request_id}", mapping=scores)
        pipe.expire(f"eval:{request_id}", 3600)  # 1 hour TTL
        
        # Update rolling window for each metric
        for metric, score in scores.items():
            if score is not None:
                pipe.lpush(f"window:{metric}", score)
                pipe.ltrim(f"window:{metric}", 0, self.window_size - 1)
        
        pipe.execute()
    
    def _anomaly_detection_loop(self):
        """Monitor for anomalies in scores"""
        import time
        
        baseline = {}
        
        while True:
            for metric in self.evaluators.keys():
                # Get current window
                scores = self.redis.lrange(f"window:{metric}", 0, -1)
                scores = [float(s) for s in scores]
                
                if len(scores) < self.window_size:
                    continue
                
                current_mean = sum(scores) / len(scores)
                
                # Initialize or compare to baseline
                if metric not in baseline:
                    baseline[metric] = current_mean
                else:
                    drop = (baseline[metric] - current_mean) / baseline[metric]
                    if drop > self.alert_threshold:
                        self._send_alert(metric, baseline[metric], current_mean)
                    
                    # Update baseline with exponential smoothing
                    alpha = 0.1
                    baseline[metric] = alpha * current_mean + (1 - alpha) * baseline[metric]
            
            time.sleep(10)  # Check every 10 seconds
    
    def _send_alert(self, metric: str, baseline: float, current: float):
        """Send alert for score degradation"""
        print(f"ALERT: {metric} dropped from {baseline:.3f} to {current:.3f}")
        # Integration with PagerDuty, Slack, etc.
```

### 3.3.3 Hierarchical Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   HIERARCHICAL EVALUATION PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ LEVEL 1: Fast Filters (< 10ms)                                       │   │
│  │ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                     │   │
│  │ │ Format  │ │ Length  │ │ Regex   │ │ Keyword │                     │   │
│  │ │ Check   │ │ Check   │ │ Match   │ │ Filter  │                     │   │
│  │ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘                     │   │
│  │      └───────────┴───────────┴───────────┘                           │   │
│  │                          │                                            │   │
│  │              PASS? ──────┴─────── FAIL → Reject (80% filtered)       │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             │                                                │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ LEVEL 2: Semantic Evaluation (< 1s)                                  │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                     │   │
│  │ │ Embedding   │ │ Fast LLM    │ │ Classifier  │                     │   │
│  │ │ Similarity  │ │ Judge       │ │ (safety)    │                     │   │
│  │ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘                     │   │
│  │        └───────────────┴───────────────┘                             │   │
│  │                          │                                            │   │
│  │            UNCERTAIN? ───┴───── PASS/FAIL → Final (95% resolved)     │   │
│  └──────────────────────────┬───────────────────────────────────────────┘   │
│                             │                                                │
│                             ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │ LEVEL 3: Deep Evaluation (< 30s)                                     │   │
│  │ ┌─────────────┐ ┌─────────────┐                                     │   │
│  │ │ Judge Panel │ │ Human Queue │                                     │   │
│  │ │ (3 judges)  │ │ (if needed) │                                     │   │
│  │ └─────────────┘ └─────────────┘                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class HierarchicalPipeline:
    """Multi-level evaluation with early exits"""
    
    def __init__(self):
        # Level 1: Fast filters
        self.level1_filters = [
            FormatValidator(),
            LengthValidator(min_len=10, max_len=10000),
            SafetyKeywordFilter(),
        ]
        
        # Level 2: Semantic evaluation
        self.level2_evaluators = {
            'similarity': EmbeddingSimilarity(),
            'quick_judge': FastLLMJudge(model='claude-haiku-4-5'),
            'safety': SafetyClassifier()
        }
        
        # Level 3: Deep evaluation.
        # A panel of small, diverse judges beats a single large judge:
        # the PoLL result (https://arxiv.org/abs/2404.18796) found a
        # 3-judge panel of small models outperformed a single GPT-4 judge
        # with less intra-model bias at ~1/7 the cost. Diversify vendors
        # to dilute self-preference bias.
        self.level3_evaluators = {
            'expert_panel': MultiJudgePanel(
                models=['claude-sonnet-4-6', 'gpt-5.5', 'gemini-3.1-pro']
            ),
        }
        
        # Thresholds
        self.level2_pass_threshold = 0.8
        self.level2_fail_threshold = 0.3
    
    def evaluate(self, sample: EvalSample) -> EvalResult:
        result = {
            'sample_id': sample.id,
            'level': 0,
            'scores': {},
            'status': 'pending'
        }
        
        # Level 1: Fast Filters
        for filter in self.level1_filters:
            if not filter.check(sample):
                result['level'] = 1
                result['status'] = 'rejected'
                result['rejection_reason'] = filter.name
                return result
        
        # Level 2: Semantic Evaluation
        level2_scores = {}
        for name, evaluator in self.level2_evaluators.items():
            level2_scores[name] = evaluator.evaluate(sample)['score']
        
        avg_score = sum(level2_scores.values()) / len(level2_scores)
        result['scores'].update(level2_scores)
        result['level'] = 2
        
        if avg_score >= self.level2_pass_threshold:
            result['status'] = 'passed'
            result['final_score'] = avg_score
            return result
        
        if avg_score <= self.level2_fail_threshold:
            result['status'] = 'failed'
            result['final_score'] = avg_score
            return result
        
        # Level 3: Deep Evaluation (uncertain cases only)
        level3_scores = {}
        for name, evaluator in self.level3_evaluators.items():
            level3_scores[name] = evaluator.evaluate(sample)['score']
        
        result['scores'].update(level3_scores)
        result['level'] = 3
        result['final_score'] = sum(level3_scores.values()) / len(level3_scores)
        result['status'] = 'passed' if result['final_score'] >= 0.5 else 'failed'
        
        return result
```

### 3.3.4 Trace-to-Dataset Flywheel

The batch and streaming patterns above treat the test set as a fixed input. The pattern that became the 2025–2026 consensus closes the loop: production traces continuously *become* new test cases. Offline evals gate releases; online evals score a sample of live traffic asynchronously; failures get clustered, triaged, and promoted into the golden set — which the CI gate then enforces forever ([OpenAI Cookbook: evaluation flywheel](https://developers.openai.com/cookbook/examples/evaluation/building_resilient_prompts_using_an_evaluation_flywheel); [LangChain, "LLM Evals"](https://www.langchain.com/articles/llm-evals)).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     TRACE-TO-DATASET FLYWHEEL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐   sample    ┌──────────────┐   low-score  ┌───────────┐ │
│   │  Production  │────────────▶│  Async LLM   │─────────────▶│  Failure  │ │
│   │  Traces      │  (async,    │  Judge       │   traces     │  Cluster  │ │
│   │  (OTel)      │  off the    │  (sampled)   │              │  + Triage │ │
│   └──────────────┘  hot path)  └──────────────┘              └─────┬─────┘ │
│          ▲                                                          │       │
│          │                                              human       │       │
│          │                                              approves    ▼       │
│   ┌──────┴───────┐             ┌──────────────┐  gate   ┌───────────────┐  │
│   │   Deploy     │◀────────────│   CI Eval    │◀────────│  Golden Set   │  │
│   │              │  pass       │   Gate (PR)  │         │  += new rows  │  │
│   └──────────────┘             └──────────────┘         └───────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

Two of the loop's stages got productized in 2025–2026 — failure clustering by [LangSmith's Insights Agent](https://latitude.so/blog/best-llm-observability-tools-agents-latitude-vs-langfuse-langsmith) (GA Oct 2025) and [Braintrust Loop](https://www.braintrust.dev/) (an AI assistant over logs that surfaces failure modes and generates scorers from plain-English descriptions) — but the promotion step should stay human-approved: a person decides which failures become permanent regression tests ([NVIDIA data-flywheel blueprint](https://github.com/NVIDIA-AI-Blueprints/data-flywheel)).

```python
class DatasetPromoter:
    """Promote judged production failures into the versioned golden set."""

    def __init__(self, test_set_manager: 'TestSetManager',
                 score_threshold: float = 0.5):
        self.manager = test_set_manager
        self.score_threshold = score_threshold
        self.candidates: List[EvalSample] = []

    def ingest(self, trace: dict, judge_scores: Dict[str, float]):
        """Called by the async judging worker for each sampled trace."""
        if min(s for s in judge_scores.values() if s is not None) \
                >= self.score_threshold:
            return  # not a failure, nothing to learn
        self.candidates.append(EvalSample(
            id=trace['trace_id'],
            input=trace['input'],
            expected_output=None,        # human adds the target during triage
            metadata={'judge_scores': judge_scores,
                      'source_trace': trace['trace_id']},
            category='production_failure',
            difficulty='unknown',
            created_at=datetime.now(),
        ))

    def review_queue(self) -> List[EvalSample]:
        """Surface candidates for human triage (dedupe / label / reject)."""
        return self.candidates

    def promote(self, approved: List[EvalSample],
                name: str, current_version: str) -> str:
        """Human-approved failures become a new golden-set version (3.4.1)."""
        existing = self.manager.load_version(current_version)
        return self.manager.create_version(
            name=name,
            samples=existing + approved,
            description=f"Promoted {len(approved)} production failures",
        )
```

The flywheel is also where this module connects to the previous one: the failure clusters are exactly the "open coding → axial coding" error-analysis artifacts, and every promoted row should carry a human label that the LLM judge is later calibrated against ([Hamel Husain & Shreya Shankar, evals FAQ](https://hamel.dev/blog/posts/evals-faq/evals-faq.pdf)).

---

## 3.4 Data Management

### 3.4.1 Test Set Versioning

```python
import hashlib
from datetime import datetime
import git

class TestSetManager:
    """Version control for evaluation datasets"""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.repo = git.Repo(base_path)
    
    def create_version(self, 
                       name: str, 
                       samples: List[EvalSample],
                       description: str) -> str:
        """Create a new version of a test set"""
        
        # Calculate content hash
        content = json.dumps([s.to_dict() for s in samples], sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
        
        version = f"{name}_v{datetime.now().strftime('%Y%m%d')}_{content_hash}"
        
        # Save to file
        file_path = f"{self.base_path}/datasets/{version}.json"
        with open(file_path, 'w') as f:
            json.dump({
                'version': version,
                'name': name,
                'description': description,
                'created_at': datetime.now().isoformat(),
                'num_samples': len(samples),
                'samples': [s.to_dict() for s in samples]
            }, f, indent=2)
        
        # Commit to git
        self.repo.index.add([file_path])
        self.repo.index.commit(f"Add test set version: {version}")
        
        return version
    
    def load_version(self, version: str) -> List[EvalSample]:
        """Load a specific version of a test set"""
        file_path = f"{self.base_path}/datasets/{version}.json"
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return [
            EvalSample(
                id=s['id'],
                input=s['input'],
                expected_output=s.get('expected_output'),
                metadata=s.get('metadata', {}),
                category=s.get('category', 'general'),
                difficulty=s.get('difficulty', 'medium'),
                created_at=datetime.fromisoformat(s['created_at'])
            )
            for s in data['samples']
        ]
    
    def diff_versions(self, version_a: str, version_b: str) -> dict:
        """Compare two versions of a test set"""
        samples_a = {s.id: s for s in self.load_version(version_a)}
        samples_b = {s.id: s for s in self.load_version(version_b)}
        
        added = set(samples_b.keys()) - set(samples_a.keys())
        removed = set(samples_a.keys()) - set(samples_b.keys())
        
        modified = []
        for id in set(samples_a.keys()) & set(samples_b.keys()):
            if samples_a[id].to_dict() != samples_b[id].to_dict():
                modified.append(id)
        
        return {
            'added': list(added),
            'removed': list(removed),
            'modified': modified,
            'unchanged': len(samples_a) - len(removed) - len(modified)
        }
```

### 3.4.2 Sample Stratification

```python
from collections import defaultdict
import random

class StratifiedSampler:
    """Ensure balanced sampling across categories"""
    
    def __init__(self, stratify_by: List[str]):
        self.stratify_by = stratify_by
    
    def sample(self, 
               samples: List[EvalSample], 
               n: int,
               strategy: str = 'proportional') -> List[EvalSample]:
        """
        Sample n items with stratification.
        
        strategy: 
            - 'proportional': maintain original distribution
            - 'balanced': equal samples per stratum
            - 'min_per_stratum': ensure minimum per stratum, then proportional
        """
        
        # Group by strata
        strata = defaultdict(list)
        for sample in samples:
            key = tuple(getattr(sample, attr, 'unknown') for attr in self.stratify_by)
            strata[key].append(sample)
        
        if strategy == 'balanced':
            per_stratum = n // len(strata)
            selected = []
            for key, stratum_samples in strata.items():
                selected.extend(random.sample(
                    stratum_samples, 
                    min(per_stratum, len(stratum_samples))
                ))
        
        elif strategy == 'proportional':
            selected = []
            for key, stratum_samples in strata.items():
                stratum_n = int(n * len(stratum_samples) / len(samples))
                selected.extend(random.sample(
                    stratum_samples,
                    min(stratum_n, len(stratum_samples))
                ))
        
        elif strategy == 'min_per_stratum':
            min_per = 5
            selected = []
            remaining = n
            
            # First, get minimum per stratum
            for key, stratum_samples in strata.items():
                take = min(min_per, len(stratum_samples), remaining)
                selected.extend(random.sample(stratum_samples, take))
                remaining -= take
            
            # Then fill proportionally
            if remaining > 0:
                unused = [s for s in samples if s not in selected]
                selected.extend(random.sample(unused, min(remaining, len(unused))))
        
        return selected

# Example usage
sampler = StratifiedSampler(stratify_by=['category', 'difficulty'])
balanced_sample = sampler.sample(all_samples, n=1000, strategy='balanced')
```

---

## 3.5 Worked Examples (2026)

Four pipeline patterns you can drop into a real project today.

#### Example 1 — Hosted offline-eval pipeline with Braintrust

Braintrust packages dataset → task → scorers → experiment view in one API. Good fit when you want a hosted UI without building it yourself. Since 2025 it also ships the CI half of the loop — a GitHub Action that posts experiment scores on PRs and blocks merge below thresholds — and [Loop](https://www.braintrust.dev/), an assistant that clusters log failures and generates scorers from plain-English descriptions.

```python
# pip install braintrust autoevals anthropic
from braintrust import Eval
from autoevals import Factuality, AnswerRelevancy
from anthropic import Anthropic

client = Anthropic()

def task(input: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5",      # cheap model under test
        max_tokens=512,
        messages=[{"role": "user", "content": input}],
    )
    return r.content[0].text

Eval(
    "support-bot-v3",            # project name in Braintrust
    data=lambda: [
        {"input": "How do I reset my password?",
         "expected": "Click 'Forgot password' on the login page."},
        {"input": "What are your hours?",
         "expected": "We are open 9am–9pm ET, 7 days a week."},
    ],
    task=task,
    scores=[Factuality, AnswerRelevancy],
)
# Run: `braintrust eval pipeline.py` — produces a diffable experiment view in the web UI.
```

#### Example 2 — Inspect AI task graph for a multi-evaluator pipeline

[Inspect AI](https://inspect.aisi.org.uk/) (UK AI Security Institute) matured into the reference open-source harness: the companion [`inspect_evals`](https://github.com/UKGovernmentBEIS/inspect_evals) repo ships 200+ prebuilt evals, and METR migrated its time-horizon suite onto Inspect in Jan 2026 ([METR Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/)).

```python
# Inspect lets you compose Solvers + Scorers — each Sample flows through them.
from inspect_ai import Task, task, eval as run_eval
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import generate, system_message
from inspect_ai.scorer import (
    model_graded_qa, includes, match, mean, stderr,
)

@task
def faq_pipeline():
    return Task(
        dataset=json_dataset("data/faq.jsonl"),     # {input, target} per line
        solver=[system_message("Answer concisely."), generate()],
        scorer=[
            includes(),                              # cheap rule-based first
            match("answer", ignore_case=True),       # exact-match
            model_graded_qa(model="anthropic/claude-sonnet-4-6"),  # judge fallback
        ],
        metrics=[mean(), stderr()],
    )

# Run 5 epochs to get standard error bands for non-deterministic outputs:
# inspect eval pipeline.py --epochs 5 --model anthropic/claude-sonnet-4-6
```

#### Example 3 — OpenTelemetry-traced evaluator (vendor-neutral)

Emit GenAI-conventions traces so the same evaluator works against Phoenix, Langfuse, LangSmith, or Braintrust by swapping the OTel exporter — no code changes.

The [OTel GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) became the de facto trace schema for LLM systems in 2025–2026, and they matter to pipeline design because your eval pipeline and your production observability can now share one span format. The taxonomy ([OTel GenAI observability blog, 2026](https://opentelemetry.io/blog/2026/genai-observability/); [agent-spans spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/)):

```
invoke_agent  (top-level agent span)
 ├── chat            (one span per LLM call)
 │     gen_ai.request.model, gen_ai.usage.input_tokens / output_tokens,
 │     gen_ai.response.finish_reasons, gen_ai.system_instructions,
 │     gen_ai.input.messages, gen_ai.output.messages
 └── execute_tool    (one span per tool invocation)
```

| Attribute | What it carries | Why evals care |
|---|---|---|
| `gen_ai.request.model` | requested model id | slice scores by model/version |
| `gen_ai.usage.input_tokens` / `output_tokens` | token counts | cost-per-eval, judge-cost budgets |
| `gen_ai.response.finish_reasons` | stop / length / tool_use … | detect truncation masquerading as failure |
| `gen_ai.input.messages` / `gen_ai.output.messages` | full conversation payloads | replay traces as eval samples (flywheel, 3.3.4) |
| `gen_ai.system_instructions` | system prompt | catch prompt-version drift between runs |

One honest caveat for anything you build on this: as of mid-2026 the GenAI conventions are **still marked "Development" (experimental), not stable** — there is no committed stabilization timeline, and you opt into the latest experimental version via an environment variable ([gen-ai semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/); [Greptime explainer, May 2026](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions)). Vendors adopted them anyway — MLflow documents native GenAI-semconv tracing ([mlflow.org](https://mlflow.org/docs/latest/genai/tracing/opentelemetry/genai-semconv/)), and Claude Code trace support is noted as beta in the OTel blog. Practical consequence: pin the semconv version in your pipeline config and treat attribute renames as a schema migration, exactly like a DB.

```python
# pip install opentelemetry-api opentelemetry-sdk openinference-instrumentation-openai
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation.openai import OpenAIInstrumentor
from openai import OpenAI

trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())   # OTEL_EXPORTER_OTLP_ENDPOINT picks the backend
)
OpenAIInstrumentor().instrument()            # auto-traces every OpenAI call w/ GenAI semconv
tracer = trace.get_tracer("evals")

client = OpenAI()

def evaluate_one(sample):
    with tracer.start_as_current_span("eval.sample") as span:
        span.set_attribute("eval.sample_id", sample["id"])
        span.set_attribute("eval.suite", "faq-v3")
        out = client.chat.completions.create(
            model="gpt-5.5",
            messages=[{"role": "user", "content": sample["input"]}],
        ).choices[0].message.content
        score = float(sample["expected"].lower() in out.lower())
        span.set_attribute("eval.score", score)
        return score
```

#### Example 4 — Sandboxed agent-eval pipeline (Inspect AI + Docker)

Everything earlier in this module assumed the system under test only *returns text*. Agent evals break that assumption: the agent runs shell commands, edits files, and mutates state — so the pipeline must provide an **isolated, disposable environment per trial**. Three reasons this is non-negotiable:

1. **Safety** — an agent that can run `rm -rf` or exfiltrate credentials must not do it on the eval host.
2. **Statistical validity** — trials that share an environment have correlated failures (one trial's leftover files break the next), corrupting pass@k/pass^k estimates. Anthropic's agent-evals guidance is explicit: isolate each trial in a clean environment ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
3. **Outcome grading** — to "grade what the agent produced, not the path it took," the scorer must inspect the *end state of the environment*, which requires owning that environment.

This is how the serious benchmarks are built: Terminal-Bench 2.0's 89 hand-crafted, human-verified terminal tasks each run in isolated Docker containers — and frontier models still fail 18–35% of them ([Terminal-Bench paper, ICLR 2026](https://openreview.net/pdf?id=a7Qa4CcHak)). Its official harness, [Harbor](https://harborframework.com/docs/running-tbench), is an open-source framework for container-based agent rollouts that can drive Claude Code, Codex CLI, OpenHands, and Mini-SWE-Agent. Inspect AI ships the same capability natively via its sandbox abstraction, plus [ControlArena](https://www.aisi.gov.uk/blog/our-2025-year-in-review) for control/sabotage evals.

```python
# pip install inspect-ai   (plus Docker running locally)
from inspect_ai import Task, task
from inspect_ai.agent import react
from inspect_ai.dataset import Sample
from inspect_ai.scorer import scorer, Score, accuracy, stderr, Target
from inspect_ai.solver import TaskState
from inspect_ai.tool import bash, text_editor
from inspect_ai.util import sandbox

@scorer(metrics=[accuracy(), stderr()])
def tests_pass():
    """Outcome-first grading: run the test suite INSIDE the sandbox
    and grade the end state — never trust the agent's own claim."""
    async def score(state: TaskState, target: Target) -> Score:
        result = await sandbox().exec(
            ["python", "-m", "pytest", "/workspace/tests", "-q"],
            timeout=120,
        )
        return Score(
            value=1.0 if result.success else 0.0,
            explanation=result.stdout[-2000:],   # keep evidence for triage
        )
    return score

@task
def fix_failing_build():
    return Task(
        dataset=[
            Sample(
                input="The test suite in /workspace is failing. "
                      "Find the bug and fix it. Do not modify the tests.",
                files={"/workspace": "fixtures/broken_project.zip"},
            ),
        ],
        solver=react(tools=[bash(timeout=180), text_editor()]),
        scorer=tests_pass(),
        sandbox="docker",          # fresh container per trial, torn down after
    )

# 5 trials per task -> pass@5 and pass^5 from one run:
# inspect eval agent_pipeline.py --epochs 5 --model anthropic/claude-sonnet-4-6
```

**Your harness is part of the system under test.** Two cautionary tales from frontier-lab pipelines. First, Claude Opus 4.5 initially scored 42% on CORE-Bench — until an Anthropic researcher found rigid grading, ambiguous task specs, and irreproducible stochastic tasks in the harness; with the bugs fixed and a less constrained scaffold, the same model scored 95% ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). Second, the Fable 5 system card switched Terminal-Bench 2.1 to the mini-SWE-agent harness because the previous Terminus-2 harness hit 2.7× more timeouts at the highest effort setting — and reports Fable 5 at 84.3% mean reward *with 20.9% of trials hitting a safety refusal and falling back to Opus 4.8 mid-trajectory* ([Fable 5 system card §8.3](https://www-cdn.anthropic.com/d00db56fa754a1b115b6dd7cb2e3c342ee809620.pdf)). If the best-resourced eval teams in the industry see double-digit score swings from harness choice, timeouts, and safeguard fallbacks, budget pipeline-debugging time accordingly: log per-trial infra errors separately from task failures, and treat a sudden score change as a harness bug until proven otherwise.

The four examples above cover the **deployment shapes** you will actually encounter: hosted SaaS (Braintrust), self-hosted research-grade (Inspect AI), vendor-neutral OSS observability (OpenTelemetry), and sandboxed agent rollouts (Inspect + Docker / Harbor). Pick one for offline + CI; pair with an online tracing backend for production (see module 06).

---

## 3.6 Tooling Landscape (mid-2026)

You will rarely build every box in the 3.1 diagram yourself. What changed in 2025–2026, tool by tool:

| Tool | What changed 2025–2026 | Source |
|---|---|---|
| **Inspect AI** (UK AISI) | 200+ prebuilt evals in `inspect_evals`; agent/multi-agent support, sandboxing, ControlArena; Bayesian evaluator-reliability stats; community-eval registry with automated review from May 2026 | [inspect.aisi.org.uk](https://inspect.aisi.org.uk/), [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals) |
| **LangSmith** | Align Evals judge-calibration loop (Jul 2025); Insights Agent failure clustering (GA Oct 2025) | [Align Evals](https://blog.langchain.com/introducing-align-evals/) |
| **Braintrust** | Loop (clusters log failures, generates scorers from plain English); CI GitHub Action with merge blocking; remote evals | [braintrust.dev](https://www.braintrust.dev/) |
| **Langfuse** (OSS) | Managed LLM-as-judge evaluators on traces/experiments; observation-level judge evals (Feb 2026) for per-tool-call precision; OTel integration | [docs](https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge), [changelog](https://langfuse.com/changelog/2026-02-13-observation-level-evals) |
| **Arize Phoenix** (OSS) | OpenTelemetry-native tracing + LLM-judge evals; the OTel-first option | [comparison](https://www.comet.com/site/blog/llm-evaluation-frameworks/) |
| **W&B Weave** | Rebuilt for production agents: online evaluations on live traffic (preview), Guardrails scorers (toxicity, PII, hallucination) | [announcement](https://wandb.ai/wandb_fc/product-announcements-fc/reports/New-in-W-B-Weave-Observability-and-continuous-improvement-for-production-agents--VmlldzoxNzAzMTcxNg) |
| **DeepEval** (Confident AI) | "Pytest for LLMs"; DAG metric (deterministic LLM decision trees with hard-coded leaf scores); conversational metrics | [docs](https://deepeval.com/docs/metrics-introduction) |
| **Promptfoo** | Security/red-team focus; **OpenAI announced acquisition Mar 9, 2026** (stays open source, integrates into OpenAI Frontier) | [openai.com](https://openai.com/index/openai-to-acquire-promptfoo/) |
| **Harbor** | Container-based agent rollouts; official Terminal-Bench 2.0 harness; drives Claude Code, Codex CLI, OpenHands | [harborframework.com](https://harborframework.com/docs/running-tbench) |

The stack pattern that recurs across 2026 comparisons: an **OSS framework (DeepEval / Promptfoo / Inspect) for PR-level CI checks** plus a **commercial platform (LangSmith / Braintrust / Weave) for production trace annotation, dataset management, and audit** ([Confident AI comparison](https://www.confident-ai.com/knowledge-base/compare/top-langsmith-alternatives-and-competitors-compared)). The Promptfoo acquisition is also a consolidation signal worth registering: eval tooling is being pulled into the model vendors themselves — keep your golden sets and result schemas portable (plain JSONL + OTel spans) so the pipeline survives a vendor change.

One more 2026 shift that affects architecture: **graders are converging with training infrastructure.** OpenAI's platform exposes grader objects (python graders, score-model graders, multigraders) shared between its Evals API and reinforcement fine-tuning jobs ([graders guide](https://developers.openai.com/api/docs/guides/graders)), and Prime Intellect's `verifiers` library treats RL environments and evals as one artifact — dataset + harness + scoring ([Environments Hub](https://www.primeintellect.ai/blog/environments)). If your evaluators may someday double as reward functions, every grader is a reward spec — red-team it like one (modules 10 and 12 cover reward hacking and eval–training separation in depth).

---

## 3.7 Exercises

### Exercise 1: Build a Complete Pipeline
Implement a batch evaluation pipeline with:
- Multiple data sources (golden set + production samples)
- At least 3 evaluators (rule-based, LLM, and semantic)
- Result storage with comparison capability
- Basic alerting on score regression

### Exercise 2: Implement Streaming Evaluation
Build a streaming pipeline that:
- Samples 5% of production traffic
- Evaluates in real-time
- Maintains a rolling 1-hour window of scores
- Alerts if any metric drops by more than 15%

### Exercise 3: Design a Data Schema
Design a comprehensive data schema for storing evaluation results that supports:
- Efficient querying by time range
- Aggregation by category
- Drill-down to individual samples
- Historical trending

### Exercise 4: Sandboxed Agent Eval
Using Inspect AI (or Harbor), build an agent eval that:
- Gives the agent `bash` + file-editing tools inside a Docker sandbox
- Grades the *outcome* (environment end-state) rather than the agent's transcript claims
- Runs 5 trials per task and reports both pass@5 and pass^5
- Separates infra errors (container/timeout failures) from genuine task failures in its report — then deliberately introduce a harness bug (e.g., a too-short tool timeout) and measure how much it moves your score

---

## Next Module
→ [Module 4: The Cold Start Problem](../04-cold-start/README.md)

