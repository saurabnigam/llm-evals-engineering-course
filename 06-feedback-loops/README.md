# Module 6: Feedback Systems & Active Learning

## 6.1 The Feedback Loop Concept

Feedback loops are the mechanism by which your evaluation system learns and improves over time. They close the gap between production reality and offline evaluation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      THE CONTINUOUS FEEDBACK LOOP                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                         ┌─────────────────┐                                 │
│                         │   Production    │                                 │
│                         │     Model       │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌─────────────────┐                                 │
│                         │  User/System    │                                 │
│                         │   Feedback      │◄───── Implicit & Explicit       │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│              ┌───────────────────┼───────────────────┐                      │
│              ▼                   ▼                   ▼                      │
│     ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐            │
│     │    Collect &    │ │   Analyze &     │ │   Prioritize    │            │
│     │     Store       │ │    Aggregate    │ │   & Route       │            │
│     └────────┬────────┘ └────────┬────────┘ └────────┬────────┘            │
│              │                   │                   │                      │
│              └───────────────────┼───────────────────┘                      │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌─────────────────┐                                 │
│                         │ Update Evals &  │                                 │
│                         │  Training Data  │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  ▼                                          │
│                         ┌─────────────────┐                                 │
│                         │  Retrain/Update │                                 │
│                         │     Model       │                                 │
│                         └────────┬────────┘                                 │
│                                  │                                          │
│                                  └──────────► (back to Production)          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6.2 Types of Feedback

### 6.2.1 Explicit Feedback

Direct signals from users about output quality.

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List

class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    FLAG = "flag"
    CORRECTION = "correction"
    REGENERATE = "regenerate"

class FeedbackCategory(Enum):
    FACTUAL_ERROR = "factual_error"
    HARMFUL_CONTENT = "harmful_content"
    OFF_TOPIC = "off_topic"
    TOO_LONG = "too_long"
    TOO_SHORT = "too_short"
    CONFUSING = "confusing"
    PERFECT = "perfect"
    OTHER = "other"

@dataclass
class ExplicitFeedback:
    """User-provided feedback"""
    id: str
    request_id: str
    user_id: str
    feedback_type: FeedbackType
    rating: Optional[float]  # 1-5 scale
    category: Optional[FeedbackCategory]
    comment: Optional[str]
    correction: Optional[str]  # User's corrected version
    timestamp: datetime
    
    # Context
    input_text: str
    output_text: str
    model_version: str

class ExplicitFeedbackCollector:
    """Collect and process explicit user feedback"""
    
    def __init__(self, storage):
        self.storage = storage
    
    def record_thumbs(self, 
                      request_id: str,
                      user_id: str,
                      is_positive: bool,
                      context: dict) -> ExplicitFeedback:
        """Record thumbs up/down feedback"""
        
        feedback = ExplicitFeedback(
            id=self._generate_id(),
            request_id=request_id,
            user_id=user_id,
            feedback_type=FeedbackType.THUMBS_UP if is_positive else FeedbackType.THUMBS_DOWN,
            rating=1.0 if is_positive else 0.0,
            category=None,
            comment=None,
            correction=None,
            timestamp=datetime.now(),
            input_text=context['input'],
            output_text=context['output'],
            model_version=context['model_version']
        )
        
        self.storage.save(feedback)
        return feedback
    
    def record_detailed_feedback(self,
                                 request_id: str,
                                 user_id: str,
                                 rating: float,
                                 category: FeedbackCategory,
                                 comment: str,
                                 correction: Optional[str],
                                 context: dict) -> ExplicitFeedback:
        """Record detailed feedback with rating and comments"""
        
        feedback = ExplicitFeedback(
            id=self._generate_id(),
            request_id=request_id,
            user_id=user_id,
            feedback_type=FeedbackType.RATING,
            rating=rating,
            category=category,
            comment=comment,
            correction=correction,
            timestamp=datetime.now(),
            input_text=context['input'],
            output_text=context['output'],
            model_version=context['model_version']
        )
        
        self.storage.save(feedback)
        
        # If user provided correction, also create training example
        if correction:
            self._create_training_example(feedback)
        
        return feedback
    
    def _create_training_example(self, feedback: ExplicitFeedback):
        """Convert correction feedback into training data"""
        training_example = {
            'input': feedback.input_text,
            'bad_output': feedback.output_text,
            'good_output': feedback.correction,
            'source': 'user_correction',
            'feedback_id': feedback.id
        }
        self.storage.save_training_example(training_example)
    
    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())
```

### 6.2.2 Implicit Feedback

Behavioral signals that indicate quality without explicit user action.

```python
@dataclass
class ImplicitFeedback:
    """Inferred feedback from user behavior"""
    request_id: str
    user_id: str
    signal_type: str
    signal_value: float
    timestamp: datetime
    
    # Raw behavioral data
    time_to_first_action_ms: int
    session_continued: bool
    response_copied: bool
    response_shared: bool
    regeneration_requested: bool
    follow_up_question: bool
    task_completed: bool

class ImplicitFeedbackTracker:
    """Track and interpret implicit user signals"""
    
    SIGNALS = {
        'copy': {'weight': 0.8, 'positive': True},
        'share': {'weight': 0.9, 'positive': True},
        'regenerate': {'weight': 0.7, 'positive': False},
        'abandon': {'weight': 0.6, 'positive': False},
        'follow_up': {'weight': 0.5, 'positive': True},  # Ambiguous
        'dwell_time_long': {'weight': 0.4, 'positive': True},
        'immediate_exit': {'weight': 0.5, 'positive': False},
        'task_success': {'weight': 1.0, 'positive': True}
    }
    
    def __init__(self, storage, session_tracker):
        self.storage = storage
        self.session_tracker = session_tracker
    
    def track_response_event(self, 
                             request_id: str,
                             event_type: str,
                             event_data: dict):
        """Track a single behavioral event"""
        
        self.storage.append_event(request_id, {
            'type': event_type,
            'data': event_data,
            'timestamp': datetime.now().isoformat()
        })
    
    def compute_implicit_score(self, request_id: str) -> dict:
        """Compute overall implicit quality score from events"""
        
        events = self.storage.get_events(request_id)
        
        positive_signals = 0
        negative_signals = 0
        total_weight = 0
        
        triggered_signals = []
        
        for event in events:
            signal_name = self._event_to_signal(event)
            if signal_name and signal_name in self.SIGNALS:
                signal = self.SIGNALS[signal_name]
                triggered_signals.append(signal_name)
                
                if signal['positive']:
                    positive_signals += signal['weight']
                else:
                    negative_signals += signal['weight']
                total_weight += signal['weight']
        
        if total_weight == 0:
            return {'score': None, 'confidence': 0, 'signals': []}
        
        score = positive_signals / total_weight
        
        return {
            'score': score,
            'confidence': min(total_weight / 2.0, 1.0),  # More signals = higher confidence
            'signals': triggered_signals,
            'positive_weight': positive_signals,
            'negative_weight': negative_signals
        }
    
    def _event_to_signal(self, event: dict) -> Optional[str]:
        """Map event to signal name"""
        event_type = event['type']
        
        mapping = {
            'copy_click': 'copy',
            'share_click': 'share',
            'regenerate_click': 'regenerate',
            'session_end_no_action': 'abandon',
            'follow_up_message': 'follow_up',
            'task_marked_complete': 'task_success'
        }
        
        # Handle dwell time
        if event_type == 'dwell':
            dwell_ms = event['data'].get('duration_ms', 0)
            if dwell_ms > 5000:  # > 5 seconds
                return 'dwell_time_long'
            elif dwell_ms < 500:  # < 0.5 seconds
                return 'immediate_exit'
        
        return mapping.get(event_type)
```

### 6.2.3 Comparative Feedback

Feedback that compares multiple outputs.

```python
@dataclass  
class ComparativeFeedback:
    """Feedback comparing multiple responses"""
    id: str
    request_id: str
    user_id: str
    input_text: str
    options: List[dict]  # List of {id, text, model}
    selected_id: str
    ranking: List[str]  # Ordered list of option IDs
    tie: bool
    comment: Optional[str]
    timestamp: datetime

class ABTestFeedbackCollector:
    """Collect feedback from A/B tests"""
    
    def __init__(self, storage, experiment_tracker):
        self.storage = storage
        self.experiment_tracker = experiment_tracker
    
    def record_comparison(self,
                         experiment_id: str,
                         request_id: str,
                         user_id: str,
                         input_text: str,
                         options: List[dict],
                         selection: dict) -> ComparativeFeedback:
        """Record user's preference between options"""
        
        feedback = ComparativeFeedback(
            id=self._generate_id(),
            request_id=request_id,
            user_id=user_id,
            input_text=input_text,
            options=options,
            selected_id=selection['selected'],
            ranking=selection.get('ranking', [selection['selected']]),
            tie=selection.get('tie', False),
            comment=selection.get('comment'),
            timestamp=datetime.now()
        )
        
        self.storage.save(feedback)
        
        # Update experiment metrics
        self.experiment_tracker.record_preference(
            experiment_id=experiment_id,
            winner_variant=self._get_variant(feedback.selected_id, options),
            feedback=feedback
        )
        
        return feedback
    
    def _get_variant(self, option_id: str, options: List[dict]) -> str:
        for opt in options:
            if opt['id'] == option_id:
                return opt.get('variant', opt.get('model', 'unknown'))
        return 'unknown'
    
    def compute_win_rates(self, experiment_id: str) -> dict:
        """Compute win rates for an experiment"""
        
        feedbacks = self.storage.get_by_experiment(experiment_id)
        
        variant_stats = {}
        for fb in feedbacks:
            winner = self._get_variant(fb.selected_id, fb.options)
            
            for opt in fb.options:
                variant = opt.get('variant', opt.get('model'))
                if variant not in variant_stats:
                    variant_stats[variant] = {'wins': 0, 'total': 0}
                
                variant_stats[variant]['total'] += 1
                if variant == winner:
                    variant_stats[variant]['wins'] += 1
        
        # Calculate win rates with confidence intervals
        results = {}
        for variant, stats in variant_stats.items():
            win_rate = stats['wins'] / stats['total'] if stats['total'] > 0 else 0
            
            # Wilson score interval for confidence
            n = stats['total']
            if n > 0:
                z = 1.96  # 95% confidence
                p = win_rate
                denominator = 1 + z**2/n
                center = (p + z**2/(2*n)) / denominator
                spread = z * ((p*(1-p)/n + z**2/(4*n**2))**0.5) / denominator
                ci_low = max(0, center - spread)
                ci_high = min(1, center + spread)
            else:
                ci_low, ci_high = 0, 0
            
            results[variant] = {
                'win_rate': win_rate,
                'wins': stats['wins'],
                'total': stats['total'],
                'confidence_interval': [ci_low, ci_high]
            }
        
        return results
```

---

## 6.3 Feedback Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEEDBACK PROCESSING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Raw Feedback Stream                                                         │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. INGESTION & VALIDATION                                          │    │
│  │  - Schema validation                                                │    │
│  │  - Spam/abuse detection                                             │    │
│  │  - Deduplication                                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. ENRICHMENT                                                      │    │
│  │  - Add user context (history, expertise)                            │    │
│  │  - Add request context (model, prompt)                              │    │
│  │  - Compute implicit signals                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. CLASSIFICATION                                                  │    │
│  │  - Categorize feedback type                                         │    │
│  │  - Identify failure modes                                           │    │
│  │  - Detect patterns                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  4. AGGREGATION                                                     │    │
│  │  - Time-based aggregation                                           │    │
│  │  - Category aggregation                                             │    │
│  │  - Trend detection                                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  5. ROUTING                                                         │    │
│  │  - Urgent issues → Immediate alert                                  │    │
│  │  - Patterns → Eval dataset updates                                  │    │
│  │  - Corrections → Training data                                      │    │
│  │  - Trends → Dashboard/Reports                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

class FeedbackPriority(Enum):
    CRITICAL = 1  # Safety issues, immediate action needed
    HIGH = 2      # Significant quality issues
    MEDIUM = 3    # Standard feedback
    LOW = 4       # Minor issues, nice-to-have

@dataclass
class ProcessedFeedback:
    """Feedback after processing pipeline"""
    raw_feedback: ExplicitFeedback
    priority: FeedbackPriority
    categories: List[str]
    failure_modes: List[str]
    suggested_actions: List[str]
    enrichments: dict
    
class FeedbackProcessor:
    """Main feedback processing pipeline"""
    
    def __init__(self, 
                 classifier,
                 enricher,
                 aggregator,
                 router):
        self.classifier = classifier
        self.enricher = enricher
        self.aggregator = aggregator
        self.router = router
        
        # Spam detection
        self.spam_detector = SpamDetector()
    
    async def process(self, feedback: ExplicitFeedback) -> ProcessedFeedback:
        """Process a single feedback item"""
        
        # Step 1: Validate and filter
        if self.spam_detector.is_spam(feedback):
            return None
        
        # Step 2: Enrich with context
        enrichments = await self.enricher.enrich(feedback)
        
        # Step 3: Classify
        classification = await self.classifier.classify(feedback, enrichments)
        
        # Step 4: Determine priority
        priority = self._compute_priority(feedback, classification)
        
        # Step 5: Create processed feedback
        processed = ProcessedFeedback(
            raw_feedback=feedback,
            priority=priority,
            categories=classification['categories'],
            failure_modes=classification['failure_modes'],
            suggested_actions=classification['suggested_actions'],
            enrichments=enrichments
        )
        
        # Step 6: Route to appropriate handlers
        await self.router.route(processed)
        
        # Step 7: Update aggregations
        self.aggregator.add(processed)
        
        return processed
    
    async def process_batch(self, feedbacks: List[ExplicitFeedback]) -> List[ProcessedFeedback]:
        """Process a batch of feedback items"""
        tasks = [self.process(f) for f in feedbacks]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    
    def _compute_priority(self, 
                         feedback: ExplicitFeedback, 
                         classification: dict) -> FeedbackPriority:
        """Determine feedback priority"""
        
        # Critical: Safety issues
        if 'safety' in classification['failure_modes']:
            return FeedbackPriority.CRITICAL
        
        if feedback.category == FeedbackCategory.HARMFUL_CONTENT:
            return FeedbackPriority.CRITICAL
        
        # High: Very negative feedback with details
        if feedback.rating is not None and feedback.rating <= 1.0:
            if feedback.comment or feedback.correction:
                return FeedbackPriority.HIGH
        
        # Medium: Standard negative feedback
        if feedback.feedback_type == FeedbackType.THUMBS_DOWN:
            return FeedbackPriority.MEDIUM
        
        # Low: Positive feedback (still valuable but not urgent)
        return FeedbackPriority.LOW

class FeedbackEnricher:
    """Add context to feedback"""
    
    def __init__(self, user_store, request_store, model_registry):
        self.user_store = user_store
        self.request_store = request_store
        self.model_registry = model_registry
    
    async def enrich(self, feedback: ExplicitFeedback) -> dict:
        """Add contextual information to feedback"""
        
        # Get user context
        user = await self.user_store.get(feedback.user_id)
        
        # Get request context
        request = await self.request_store.get(feedback.request_id)
        
        # Get model info
        model = await self.model_registry.get(feedback.model_version)
        
        return {
            'user': {
                'expertise_level': user.get('expertise', 'unknown'),
                'feedback_history_count': user.get('feedback_count', 0),
                'reliability_score': user.get('reliability', 0.5),
                'user_segment': user.get('segment', 'general')
            },
            'request': {
                'category': request.get('category'),
                'complexity': request.get('complexity'),
                'latency_ms': request.get('latency_ms'),
                'prompt_version': request.get('prompt_version')
            },
            'model': {
                'name': model.get('name'),
                'version': model.get('version'),
                'deployment_date': model.get('deployed_at')
            }
        }

class FeedbackClassifier:
    """Classify feedback into categories and failure modes"""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def classify(self, 
                      feedback: ExplicitFeedback, 
                      enrichments: dict) -> dict:
        """Classify feedback using LLM"""
        
        prompt = f"""Analyze this user feedback for an AI assistant:

User Input: {feedback.input_text}
AI Response: {feedback.output_text}
User Rating: {feedback.rating}/5
User Comment: {feedback.comment or 'None provided'}
User Correction: {feedback.correction or 'None provided'}
Feedback Category: {feedback.category.value if feedback.category else 'None'}

Classify this feedback:
1. What categories does this fall into? (factual, style, completeness, safety, relevance)
2. What failure modes are present? (hallucination, incorrect, incomplete, harmful, off-topic, format)
3. What actions should be taken? (add_to_eval_set, update_prompt, flag_for_review, add_to_training)

Return as JSON:
{{
    "categories": ["..."],
    "failure_modes": ["..."],
    "suggested_actions": ["..."],
    "analysis": "brief explanation"
}}
"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)

class FeedbackRouter:
    """Route processed feedback to appropriate handlers"""
    
    def __init__(self):
        self.handlers = {}
    
    def register_handler(self, priority: FeedbackPriority, handler):
        self.handlers[priority] = handler
    
    async def route(self, feedback: ProcessedFeedback):
        """Route feedback to appropriate handlers"""
        
        # Priority-based routing
        handler = self.handlers.get(feedback.priority)
        if handler:
            await handler.handle(feedback)
        
        # Action-based routing
        for action in feedback.suggested_actions:
            if action == 'add_to_eval_set':
                await self._add_to_eval_set(feedback)
            elif action == 'update_prompt':
                await self._queue_prompt_review(feedback)
            elif action == 'flag_for_review':
                await self._flag_for_human_review(feedback)
            elif action == 'add_to_training':
                await self._add_to_training_data(feedback)
    
    async def _add_to_eval_set(self, feedback: ProcessedFeedback):
        """Add feedback example to evaluation dataset"""
        eval_example = {
            'input': feedback.raw_feedback.input_text,
            'bad_output': feedback.raw_feedback.output_text,
            'expected_output': feedback.raw_feedback.correction,
            'failure_modes': feedback.failure_modes,
            'source': 'user_feedback',
            'priority': feedback.priority.value
        }
        # Store for eval dataset update
        await self.eval_queue.add(eval_example)
    
    async def _add_to_training_data(self, feedback: ProcessedFeedback):
        """Add correction to training data"""
        if feedback.raw_feedback.correction:
            training_example = {
                'input': feedback.raw_feedback.input_text,
                'output': feedback.raw_feedback.correction,
                'source': 'user_correction'
            }
            await self.training_queue.add(training_example)
```

---

## 6.4 Active Learning

Active learning uses model uncertainty and feedback patterns to intelligently select which samples to label next.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ACTIVE LEARNING LOOP                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    ┌─────────────────────────────┐                          │
│                    │     Unlabeled Data Pool     │                          │
│                    │   (Production Traffic)      │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌─────────────────────────────┐                          │
│                    │    Selection Strategy       │                          │
│                    │  ┌─────────────────────┐   │                          │
│                    │  │ • Uncertainty       │   │                          │
│                    │  │ • Diversity         │   │                          │
│                    │  │ • Expected Impact   │   │                          │
│                    │  │ • Error Patterns    │   │                          │
│                    │  └─────────────────────┘   │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌─────────────────────────────┐                          │
│                    │    Selected Samples         │                          │
│                    │    for Labeling             │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                          │
│              ┌────────────────────┴────────────────────┐                    │
│              ▼                                         ▼                    │
│  ┌─────────────────────┐                 ┌─────────────────────┐           │
│  │   Human Labeling    │                 │   LLM Labeling      │           │
│  │   (High Value)      │                 │   (Bootstrapping)   │           │
│  └──────────┬──────────┘                 └──────────┬──────────┘           │
│             │                                       │                       │
│             └───────────────────┬───────────────────┘                       │
│                                 │                                            │
│                                 ▼                                            │
│                    ┌─────────────────────────────┐                          │
│                    │   Update Evaluation Set     │                          │
│                    │   Update Training Data      │                          │
│                    └──────────────┬──────────────┘                          │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌─────────────────────────────┐                          │
│                    │    Retrain/Update Model     │                          │
│                    └─────────────────────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class ActiveLearningSelector:
    """Select the most valuable samples for labeling"""
    
    def __init__(self, 
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 uncertainty_model = None):
        self.embedder = SentenceTransformer(embedding_model)
        self.uncertainty_model = uncertainty_model
        
        # Track what we've already labeled
        self.labeled_embeddings = []
    
    def select_samples(self,
                      pool: List[dict],
                      n_select: int,
                      strategy: str = 'combined') -> List[dict]:
        """Select n samples from pool using specified strategy"""
        
        if strategy == 'uncertainty':
            return self._uncertainty_sampling(pool, n_select)
        elif strategy == 'diversity':
            return self._diversity_sampling(pool, n_select)
        elif strategy == 'combined':
            return self._combined_sampling(pool, n_select)
        elif strategy == 'error_pattern':
            return self._error_pattern_sampling(pool, n_select)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _uncertainty_sampling(self, pool: List[dict], n: int) -> List[dict]:
        """Select samples where model is most uncertain"""
        
        scored = []
        for sample in pool:
            # Get model's confidence on this sample
            uncertainty = self._compute_uncertainty(sample)
            scored.append((sample, uncertainty))
        
        # Sort by uncertainty (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [s[0] for s in scored[:n]]
    
    def _compute_uncertainty(self, sample: dict) -> float:
        """Compute uncertainty score for a sample"""
        
        if self.uncertainty_model is None:
            # Fallback: use output length variance as proxy
            return 0.5
        
        # Get model's output distribution
        outputs = self.uncertainty_model.generate_n(sample['input'], n=5)
        
        # Compute variance in outputs (semantic diversity)
        embeddings = self.embedder.encode(outputs)
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        return float(np.mean(distances))
    
    def _diversity_sampling(self, pool: List[dict], n: int) -> List[dict]:
        """Select diverse samples using clustering"""
        
        # Embed all samples
        texts = [s['input'] for s in pool]
        embeddings = self.embedder.encode(texts)
        
        # Also consider already labeled samples
        if self.labeled_embeddings:
            all_embeddings = np.vstack([embeddings, np.array(self.labeled_embeddings)])
        else:
            all_embeddings = embeddings
        
        # Cluster
        n_clusters = min(n, len(pool))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Select one sample from each cluster (furthest from labeled data)
        selected = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(self.labeled_embeddings) > 0:
                # Find sample furthest from any labeled sample
                cluster_embeddings = embeddings[cluster_indices]
                labeled_array = np.array(self.labeled_embeddings)
                
                max_min_distance = -1
                best_idx = cluster_indices[0]
                
                for idx in cluster_indices:
                    min_dist = np.min(np.linalg.norm(
                        labeled_array - embeddings[idx], axis=1
                    ))
                    if min_dist > max_min_distance:
                        max_min_distance = min_dist
                        best_idx = idx
                
                selected.append(pool[best_idx])
            else:
                # Just pick closest to cluster center
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(
                    embeddings[cluster_indices] - center, axis=1
                )
                closest = cluster_indices[np.argmin(distances)]
                selected.append(pool[closest])
        
        return selected
    
    def _combined_sampling(self, pool: List[dict], n: int) -> List[dict]:
        """Combine uncertainty and diversity"""
        
        # Get uncertainty scores
        uncertainty_scores = []
        texts = []
        for sample in pool:
            texts.append(sample['input'])
            uncertainty_scores.append(self._compute_uncertainty(sample))
        
        # Get embeddings
        embeddings = self.embedder.encode(texts)
        
        # Normalize scores
        u_scores = np.array(uncertainty_scores)
        u_scores = (u_scores - u_scores.min()) / (u_scores.max() - u_scores.min() + 1e-8)
        
        # Compute diversity scores (distance from cluster centers of labeled data)
        if self.labeled_embeddings:
            labeled_array = np.array(self.labeled_embeddings)
            d_scores = np.array([
                np.min(np.linalg.norm(labeled_array - emb, axis=1))
                for emb in embeddings
            ])
        else:
            d_scores = np.ones(len(pool))
        
        d_scores = (d_scores - d_scores.min()) / (d_scores.max() - d_scores.min() + 1e-8)
        
        # Combined score (adjustable weights)
        combined = 0.6 * u_scores + 0.4 * d_scores
        
        # Select top n
        top_indices = np.argsort(combined)[-n:][::-1]
        
        return [pool[i] for i in top_indices]
    
    def _error_pattern_sampling(self, pool: List[dict], n: int) -> List[dict]:
        """Select samples similar to known error cases"""
        
        # Load known error patterns
        error_patterns = self._load_error_patterns()
        
        if not error_patterns:
            # Fallback to diversity
            return self._diversity_sampling(pool, n)
        
        # Embed pool and error patterns
        pool_texts = [s['input'] for s in pool]
        pool_embeddings = self.embedder.encode(pool_texts)
        
        error_embeddings = self.embedder.encode(error_patterns)
        
        # Score by similarity to error patterns
        similarity_scores = []
        for emb in pool_embeddings:
            max_sim = np.max(np.dot(error_embeddings, emb) / (
                np.linalg.norm(error_embeddings, axis=1) * np.linalg.norm(emb)
            ))
            similarity_scores.append(max_sim)
        
        # Select highest similarity
        top_indices = np.argsort(similarity_scores)[-n:][::-1]
        
        return [pool[i] for i in top_indices]
    
    def _load_error_patterns(self) -> List[str]:
        """Load known error patterns from feedback history"""
        # This would load from your feedback database
        # Return list of inputs that led to errors
        return []
    
    def mark_labeled(self, samples: List[dict]):
        """Mark samples as labeled (update internal state)"""
        texts = [s['input'] for s in samples]
        embeddings = self.embedder.encode(texts)
        self.labeled_embeddings.extend(embeddings.tolist())
```

---

## 6.5 Feedback-Driven Evaluation Updates

```python
class EvalSetUpdater:
    """Automatically update evaluation sets based on feedback"""
    
    def __init__(self, 
                 eval_store,
                 feedback_processor,
                 min_confidence: float = 0.8):
        self.eval_store = eval_store
        self.feedback_processor = feedback_processor
        self.min_confidence = min_confidence
    
    async def process_feedback_batch(self, 
                                    feedbacks: List[ProcessedFeedback]) -> dict:
        """Process a batch of feedback and update eval sets"""
        
        additions = []
        updates = []
        
        for fb in feedbacks:
            if self._should_add_to_eval(fb):
                eval_case = self._create_eval_case(fb)
                additions.append(eval_case)
            
            # Check if this invalidates existing eval cases
            invalidated = await self._check_invalidations(fb)
            updates.extend(invalidated)
        
        # Apply updates
        if additions:
            await self.eval_store.add_batch(additions)
        
        if updates:
            await self.eval_store.update_batch(updates)
        
        return {
            'added': len(additions),
            'updated': len(updates),
            'additions': additions,
            'updates': updates
        }
    
    def _should_add_to_eval(self, fb: ProcessedFeedback) -> bool:
        """Determine if feedback should become eval case"""
        
        # Must have user correction
        if not fb.raw_feedback.correction:
            return False
        
        # User must be reliable
        if fb.enrichments['user']['reliability_score'] < self.min_confidence:
            return False
        
        # Must be a clear failure
        if fb.raw_feedback.rating > 2:
            return False
        
        return True
    
    def _create_eval_case(self, fb: ProcessedFeedback) -> dict:
        """Create evaluation case from feedback"""
        
        return {
            'id': f"fb_{fb.raw_feedback.id}",
            'input': fb.raw_feedback.input_text,
            'expected_output': fb.raw_feedback.correction,
            'bad_outputs': [fb.raw_feedback.output_text],
            'failure_modes': fb.failure_modes,
            'source': 'user_feedback',
            'category': fb.categories[0] if fb.categories else 'general',
            'difficulty': self._infer_difficulty(fb),
            'created_at': datetime.now().isoformat(),
            'metadata': {
                'original_feedback_id': fb.raw_feedback.id,
                'user_reliability': fb.enrichments['user']['reliability_score']
            }
        }
    
    async def _check_invalidations(self, fb: ProcessedFeedback) -> List[dict]:
        """Check if feedback invalidates existing eval cases"""
        
        # Find similar existing cases
        similar = await self.eval_store.find_similar(
            fb.raw_feedback.input_text,
            threshold=0.9
        )
        
        updates = []
        for case in similar:
            # If user says our expected output is wrong
            if (fb.raw_feedback.correction and 
                self._is_similar(case['expected_output'], fb.raw_feedback.output_text)):
                updates.append({
                    'id': case['id'],
                    'action': 'review_required',
                    'reason': 'User correction suggests different answer',
                    'new_answer': fb.raw_feedback.correction
                })
        
        return updates
    
    def _infer_difficulty(self, fb: ProcessedFeedback) -> str:
        """Infer difficulty from feedback characteristics"""
        
        complexity = fb.enrichments['request'].get('complexity', 'medium')
        
        # Adjust based on failure severity
        if FeedbackPriority.CRITICAL in [fb.priority]:
            return 'hard'
        
        return complexity
    
    def _is_similar(self, text1: str, text2: str) -> bool:
        """Check if two texts are similar"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio() > 0.85
```

---

## 6.6 Feedback Analytics Dashboard

```python
class FeedbackAnalytics:
    """Analytics for feedback data"""
    
    def __init__(self, feedback_store):
        self.store = feedback_store
    
    def get_summary(self, 
                   start_date: datetime,
                   end_date: datetime) -> dict:
        """Get summary statistics for a time period"""
        
        feedbacks = self.store.get_range(start_date, end_date)
        
        total = len(feedbacks)
        if total == 0:
            return {'error': 'No feedback in range'}
        
        positive = sum(1 for f in feedbacks 
                      if f.feedback_type == FeedbackType.THUMBS_UP 
                      or (f.rating and f.rating >= 4))
        
        negative = sum(1 for f in feedbacks 
                      if f.feedback_type == FeedbackType.THUMBS_DOWN 
                      or (f.rating and f.rating <= 2))
        
        with_correction = sum(1 for f in feedbacks if f.correction)
        
        # Category breakdown
        categories = {}
        for f in feedbacks:
            if f.category:
                cat = f.category.value
                categories[cat] = categories.get(cat, 0) + 1
        
        # Daily trend
        daily = {}
        for f in feedbacks:
            day = f.timestamp.date().isoformat()
            if day not in daily:
                daily[day] = {'positive': 0, 'negative': 0, 'total': 0}
            daily[day]['total'] += 1
            if f.rating and f.rating >= 4:
                daily[day]['positive'] += 1
            elif f.rating and f.rating <= 2:
                daily[day]['negative'] += 1
        
        return {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'totals': {
                'total_feedback': total,
                'positive': positive,
                'negative': negative,
                'neutral': total - positive - negative,
                'with_correction': with_correction
            },
            'rates': {
                'positive_rate': positive / total,
                'negative_rate': negative / total,
                'correction_rate': with_correction / total
            },
            'categories': categories,
            'daily_trend': daily
        }
    
    def get_top_issues(self, 
                       n: int = 10,
                       days: int = 7) -> List[dict]:
        """Get most common issues from feedback"""
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        feedbacks = self.store.get_range(start, end)
        negative = [f for f in feedbacks if f.rating and f.rating <= 2]
        
        # Cluster similar issues
        if not negative:
            return []
        
        texts = [f.comment or f.input_text for f in negative if f.comment or f.input_text]
        
        if len(texts) < n:
            return [{'text': t, 'count': 1} for t in texts]
        
        # Use embeddings to cluster
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedder.encode(texts)
        
        kmeans = KMeans(n_clusters=min(n, len(texts)), random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # Get representative text from each cluster
        issues = []
        for cluster_id in range(kmeans.n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            
            # Find most central text
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(
                embeddings[cluster_indices] - center, axis=1
            )
            representative = cluster_texts[np.argmin(distances)]
            
            issues.append({
                'representative': representative,
                'count': len(cluster_indices),
                'examples': cluster_texts[:3]
            })
        
        return sorted(issues, key=lambda x: x['count'], reverse=True)
    
    def detect_regressions(self,
                          window_days: int = 7,
                          threshold: float = 0.1) -> List[dict]:
        """Detect quality regressions from feedback trends"""
        
        now = datetime.now()
        current_window = self.get_summary(
            now - timedelta(days=window_days), now
        )
        previous_window = self.get_summary(
            now - timedelta(days=window_days*2),
            now - timedelta(days=window_days)
        )
        
        regressions = []
        
        # Check positive rate drop
        current_positive = current_window['rates']['positive_rate']
        previous_positive = previous_window['rates']['positive_rate']
        
        if previous_positive - current_positive > threshold:
            regressions.append({
                'metric': 'positive_rate',
                'current': current_positive,
                'previous': previous_positive,
                'change': current_positive - previous_positive,
                'severity': 'high' if previous_positive - current_positive > 0.2 else 'medium'
            })
        
        # Check negative rate increase
        current_negative = current_window['rates']['negative_rate']
        previous_negative = previous_window['rates']['negative_rate']
        
        if current_negative - previous_negative > threshold:
            regressions.append({
                'metric': 'negative_rate',
                'current': current_negative,
                'previous': previous_negative,
                'change': current_negative - previous_negative,
                'severity': 'high' if current_negative - previous_negative > 0.2 else 'medium'
            })
        
        return regressions
```

---

## 6.6b Online Evals on Production Traces (2026 standard)

Offline evals on a fixed dataset will always lag what real users are sending you. The mature pattern is to run a *subset* of your evaluators **on live production traces** and treat the result as a continuous quality signal. This is what tools like [Braintrust online scoring](https://www.braintrust.dev/docs/guides/evals/online), [LangSmith online evaluators](https://docs.langchain.com/langsmith/online-evaluations), [Arize Phoenix](https://phoenix.arize.com/) and [Langfuse](https://langfuse.com/) all standardize on.

**What to score online:**

| Score | Frequency | Why |
|-------|-----------|-----|
| Cheap rule-based checks (schema, length, refusal regex) | 100% of traffic | Free, catches obvious breakage |
| LLM-judge faithfulness / groundedness (RAG) | 5–20% sample | Catches hallucination drift |
| LLM-judge tone / policy adherence | 5–20% sample | Catches voice/policy regressions |
| Tool-call correctness (agents) | 100% if cheap, else sample | Catches tool-schema drift |
| User-signal scores (👍/👎, dwell, edit, reopen) | 100% | Ground truth, eventually |

**The trace → dataset flywheel:**

```
Production trace
     │
     ├── Online judge scores it (e.g. faithfulness=0.4, low confidence)
     │
     ├── Filter: low score OR low judge-confidence OR user 👎
     │
     ├── Add to "review queue"
     │
     ├── Human labels it (correct answer + critique)
     │
     └── Promote to permanent eval dataset (versioned)
               │
               └── Next CI run catches this exact regression
```

**Implementation notes:**
- Run online judges **asynchronously** off the request path. They must never add latency to the user's response.
- Sample, don't score everything. 5–20% is plenty for trend detection; 100% is wasted spend.
- Track judge **confidence** alongside the score. Low-confidence scores are the highest-value items to send to humans.
- Always emit traces with [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) so you can swap eval/observability backends without re-instrumenting.
- Watch for **drift signals**: rolling mean of judge scores, distribution shift in input length / language / topic, rate of refusals, rate of tool errors.

### 6.6c Worked Examples (2026)

Three compact patterns that operationalize online evals + the trace→dataset flywheel.

#### Example 1 — Async online judge with sampling

Never block the user response. Fire-and-forget the judge call.

```python
import asyncio, random, json, time
from anthropic import AsyncAnthropic
client = AsyncAnthropic()

SAMPLE_RATE = 0.10  # 10% of traffic

async def online_judge(trace_id: str, user: str, output: str):
    """Background coroutine — awaits nothing on the request path."""
    if random.random() > SAMPLE_RATE:
        return
    msg = await client.messages.create(
        model="claude-haiku-4-5", max_tokens=120, temperature=0,
        messages=[{"role": "user", "content":
            f"Score 0-1 if the answer is grounded in policy.\nQ: {user}\nA: {output}\n"
            'Return JSON: {"score": float, "confidence": float, "reason": str}'}],
    )
    payload = json.loads(msg.content[0].text)
    await write_to_warehouse(trace_id, payload, ts=time.time())

# In your request handler:
async def handle(request):
    out = await llm_call(request.user_msg)
    asyncio.create_task(online_judge(request.id, request.user_msg, out))  # fire & forget
    return out
```

#### Example 2 — Drift-detection SQL on judge scores

Run this nightly against your trace warehouse. Alert if today’s mean drops >2σ vs the trailing 14-day baseline.

```sql
-- BigQuery / Snowflake / DuckDB compatible
WITH daily AS (
  SELECT DATE(ts) AS d,
         AVG(score)             AS mean_score,
         APPROX_QUANTILES(score, 100)[OFFSET(50)] AS p50,
         COUNT(*)               AS n
  FROM   prod_judge_scores
  WHERE  ts >= CURRENT_DATE - INTERVAL '15' DAY
  GROUP BY d
),
baseline AS (
  SELECT AVG(mean_score) AS mu,
         STDDEV(mean_score) AS sigma
  FROM   daily
  WHERE  d < CURRENT_DATE
)
SELECT d.d, d.mean_score, b.mu, b.sigma,
       (d.mean_score - b.mu) / NULLIF(b.sigma, 0) AS z_score
FROM   daily d, baseline b
WHERE  d.d = CURRENT_DATE
  AND  ABS((d.mean_score - b.mu) / NULLIF(b.sigma, 0)) > 2.0;  -- alert row
```

#### Example 3 — Trace → dataset promotion (LangSmith API)

When a trace is human-labeled “bad” in your review tool, push it into the regression dataset so CI catches it next time.

```python
from langsmith import Client
ls = Client()

DATASET = "support-bot-regressions"

def promote(trace_id: str, human_label: dict):
    run = ls.read_run(trace_id)
    ls.create_example(
        inputs={"question": run.inputs["question"]},
        outputs={"expected": human_label["corrected_answer"]},
        metadata={"source_trace": trace_id,
                  "failure_mode": human_label["failure_mode"],
                  "reviewer": human_label["reviewer"],
                  "promoted_at": human_label["ts"]},
        dataset_name=DATASET,
    )
# Next CI run via `langsmith eval support-bot-regressions` will catch this exact regression.
```

---

## 6.7 Exercises

### Exercise 1: Feedback Collection UI
Design a feedback collection interface that:
- Minimizes user friction (quick thumbs up/down)
- Allows detailed feedback when users want to give it
- Collects corrections for bad outputs
- Works on mobile and desktop

### Exercise 2: Active Learning Simulation
Create a simulation that:
- Starts with a pool of 10,000 unlabeled samples
- Compares random vs active learning selection
- Measures how many labels needed to reach 90% coverage
- Visualizes the learning curve

### Exercise 3: Feedback-to-Eval Pipeline
Build a complete pipeline that:
- Ingests user feedback
- Filters for high-quality corrections
- Creates new evaluation cases
- Updates evaluation sets
- Tracks coverage improvements

---

## Next Module
→ [Module 7: CI/CD Integration](../07-cicd-integration/README.md)



