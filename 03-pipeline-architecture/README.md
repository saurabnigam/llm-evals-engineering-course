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
from typing import Dict, List, Callable
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
│  │ │ GPT-4 Panel │ │ Human Queue │                                     │   │
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
            'quick_judge': FastLLMJudge(model='gpt-4o-mini'),
            'safety': SafetyClassifier()
        }
        
        # Level 3: Deep evaluation
        self.level3_evaluators = {
            'expert_panel': MultiJudgePanel(models=['gpt-4o', 'claude-3-opus']),
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

Three pipeline patterns you can drop into a real project today.

#### Example 1 — Hosted offline-eval pipeline with Braintrust

Braintrust packages dataset → task → scorers → experiment view in one API. Good fit when you want a hosted UI without building it yourself.

```python
# pip install braintrust autoevals openai
from braintrust import Eval
from autoevals import Factuality, AnswerRelevancy
from openai import OpenAI

client = OpenAI()

def task(input: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input}],
        temperature=0,
    )
    return r.choices[0].message.content

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
            model_graded_qa(model="openai/gpt-4o"),  # LLM-judge fallback
        ],
        metrics=[mean(), stderr()],
    )

# Run 5 epochs to get standard error bands for non-deterministic outputs:
# inspect eval pipeline.py --epochs 5 --model anthropic/claude-sonnet-4-5
```

#### Example 3 — OpenTelemetry-traced evaluator (vendor-neutral)

Emit GenAI-conventions traces so the same evaluator works against Phoenix, Langfuse, LangSmith, or Braintrust by swapping the OTel exporter — no code changes.

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
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": sample["input"]}],
        ).choices[0].message.content
        score = float(sample["expected"].lower() in out.lower())
        span.set_attribute("eval.score", score)
        return score
```

The three examples above cover the **three deployment shapes** you will actually encounter: hosted SaaS (Braintrust), self-hosted research-grade (Inspect AI), and vendor-neutral OSS observability (OpenTelemetry). Pick one for offline + CI; pair with an online tracing backend for production (see module 06).

---

## 3.6 Exercises

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

---

## Next Module
→ [Module 4: The Cold Start Problem](../04-cold-start/README.md)

