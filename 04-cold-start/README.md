# Module 4: The Cold Start Problem

## 4.1 Understanding the Cold Start Problem

The **Cold Start Problem** in eval engineering is the challenge of building effective evaluations when you have:
- No historical data
- No labeled examples
- No baseline metrics
- No user feedback yet

This is the classic "chicken and egg" problem: you need good evals to improve your model, but you need model outputs to build good evals.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE COLD START DILEMMA                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│         Need good evals            Need model outputs                        │
│         to improve model           to build evals                            │
│               │                          │                                   │
│               ▼                          ▼                                   │
│         ┌─────────────────────────────────────────┐                         │
│         │                                         │                         │
│         │    🐔 ◄──────────────────────▶ 🥚      │                         │
│         │                                         │                         │
│         │   "Which comes first?"                  │                         │
│         │                                         │                         │
│         └─────────────────────────────────────────┘                         │
│                                                                              │
│  SYMPTOMS OF COLD START:                                                     │
│  • No golden dataset exists                                                  │
│  • Don't know what "good" looks like                                        │
│  • No baseline to compare against                                            │
│  • Flying blind on quality                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.2 Cold Start Strategies

### Strategy Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COLD START SOLUTION STRATEGIES                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IMMEDIATE (Day 1-7)                                                         │
│  ├── 1. Synthetic Data Generation                                            │
│  ├── 2. Transfer from Similar Domains                                        │
│  └── 3. Expert-Seeded Examples                                               │
│                                                                              │
│  SHORT-TERM (Week 2-4)                                                       │
│  ├── 4. Bootstrapping with LLM-as-Judge                                      │
│  ├── 5. Active Sampling from Production                                      │
│  └── 6. User Feedback Collection                                             │
│                                                                              │
│  MEDIUM-TERM (Month 1-3)                                                     │
│  ├── 7. Adversarial Data Generation                                          │
│  ├── 8. Edge Case Mining                                                     │
│  └── 9. Continuous Expansion                                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.3 Strategy 1: Synthetic Data Generation

Generate test cases using LLMs or rule-based systems.

### 4.3.1 LLM-Based Generation

```python
from openai import OpenAI
from typing import List
import json

class SyntheticDataGenerator:
    """Generate synthetic evaluation data using LLMs"""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
    
    def generate_test_cases(self,
                           task_description: str,
                           num_cases: int,
                           categories: List[str],
                           difficulty_levels: List[str]) -> List[dict]:
        """Generate diverse test cases for a given task"""
        
        prompt = f"""Generate {num_cases} diverse test cases for the following AI task:

TASK DESCRIPTION:
{task_description}

REQUIREMENTS:
1. Cover these categories: {', '.join(categories)}
2. Include these difficulty levels: {', '.join(difficulty_levels)}
3. Each test case should have:
   - A realistic user input
   - An ideal/expected output
   - Edge cases where applicable
   - Clear success criteria

Generate the test cases as a JSON array with this structure:
[
  {{
    "id": "unique_id",
    "input": "user query or input",
    "expected_output": "ideal response",
    "category": "one of the categories",
    "difficulty": "one of the difficulty levels",
    "success_criteria": ["criterion 1", "criterion 2"],
    "edge_case": true/false,
    "notes": "any special considerations"
  }}
]

Be creative and think about:
- Common real-world queries
- Edge cases and corner cases
- Potentially adversarial inputs
- Various user personas and contexts
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get('test_cases', result)
    
    def generate_variations(self, 
                           base_example: dict,
                           num_variations: int = 5) -> List[dict]:
        """Generate variations of a single example"""
        
        prompt = f"""Given this example:
Input: {base_example['input']}
Expected: {base_example['expected_output']}

Generate {num_variations} variations that test the same capability but with different:
- Phrasing
- Context
- Complexity levels
- Edge conditions

Return as JSON array with same structure as original.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content).get('variations', [])
    
    def generate_adversarial_cases(self,
                                   task_description: str,
                                   num_cases: int = 20) -> List[dict]:
        """Generate cases designed to find failure modes"""
        
        prompt = f"""Generate {num_cases} adversarial test cases for this AI system:

TASK: {task_description}

Generate inputs that might cause the AI to:
1. Make factual errors
2. Produce harmful content
3. Leak system prompts
4. Follow injection attacks
5. Get confused by ambiguity
6. Handle edge cases poorly
7. Produce inconsistent outputs
8. Miss important nuances

For each case, include:
- The adversarial input
- Why it might cause problems
- What the correct behavior should be
- Severity if the AI fails (low/medium/high/critical)

Return as JSON array.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content).get('adversarial_cases', [])

# Example usage for a customer service bot
generator = SyntheticDataGenerator()

task = """
A customer service chatbot for an e-commerce company.
It should:
- Answer questions about products, orders, and policies
- Handle refund and return requests
- Escalate complex issues to human agents
- Be helpful, professional, and empathetic
"""

test_cases = generator.generate_test_cases(
    task_description=task,
    num_cases=50,
    categories=['product_inquiry', 'order_status', 'refund', 'complaint', 'general'],
    difficulty_levels=['easy', 'medium', 'hard', 'adversarial']
)

print(f"Generated {len(test_cases)} test cases")
```

### 4.3.2 Template-Based Generation

```python
import random
from typing import List, Dict
from string import Template

class TemplateBasedGenerator:
    """Generate test cases using templates and slot filling"""
    
    def __init__(self):
        self.templates = {}
        self.slot_values = {}
    
    def add_template(self, 
                     name: str, 
                     input_template: str,
                     output_template: str,
                     slots: Dict[str, List[str]]):
        """Add a template with slot definitions"""
        self.templates[name] = {
            'input': Template(input_template),
            'output': Template(output_template),
            'slots': slots
        }
    
    def generate(self, template_name: str, num_cases: int) -> List[dict]:
        """Generate test cases from a template"""
        
        template = self.templates[template_name]
        cases = []
        
        for i in range(num_cases):
            # Fill slots with random values
            slot_values = {
                slot: random.choice(values)
                for slot, values in template['slots'].items()
            }
            
            case = {
                'id': f"{template_name}_{i}",
                'input': template['input'].safe_substitute(slot_values),
                'expected_output': template['output'].safe_substitute(slot_values),
                'template': template_name,
                'slot_values': slot_values
            }
            cases.append(case)
        
        return cases

# Example: E-commerce chatbot templates
generator = TemplateBasedGenerator()

generator.add_template(
    name='product_price',
    input_template="What's the price of the $product in $color?",
    output_template="The $product in $color costs $price.",
    slots={
        'product': ['iPhone 15', 'MacBook Pro', 'AirPods', 'iPad Air'],
        'color': ['black', 'silver', 'gold', 'space gray'],
        'price': ['$999', '$1299', '$199', '$599']
    }
)

generator.add_template(
    name='order_status',
    input_template="Where is my order $order_id? I ordered a $product $days_ago.",
    output_template="Your order $order_id for the $product is currently $status. Expected delivery: $delivery_date.",
    slots={
        'order_id': ['#12345', '#67890', '#11111', '#22222'],
        'product': ['laptop', 'phone', 'tablet', 'headphones'],
        'days_ago': ['yesterday', '3 days ago', 'last week', '2 weeks ago'],
        'status': ['in transit', 'out for delivery', 'processing', 'shipped'],
        'delivery_date': ['tomorrow', 'in 2 days', 'next Monday', 'this Friday']
    }
)

# Generate test cases
price_cases = generator.generate('product_price', 20)
order_cases = generator.generate('order_status', 20)
```

---

## 4.4 Strategy 2: Transfer from Similar Domains

Leverage existing datasets or benchmarks from related tasks.

```python
from datasets import load_dataset
from typing import List

class DomainTransfer:
    """Transfer and adapt evaluations from similar domains"""
    
    def __init__(self):
        self.available_benchmarks = {
            'qa': ['squad', 'natural_questions', 'triviaqa'],
            'summarization': ['cnn_dailymail', 'xsum'],
            'dialogue': ['personachat', 'dailydialog'],
            'code': ['humaneval', 'mbpp'],
            'math': ['gsm8k', 'math'],
            'safety': ['crows_pairs', 'winobias']
        }
    
    def load_benchmark(self, 
                       benchmark_name: str,
                       num_samples: int = 100) -> List[dict]:
        """Load samples from a public benchmark"""
        
        dataset = load_dataset(benchmark_name, split='test')
        samples = []
        
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            
            # Standard mapping - adjust per dataset
            sample = {
                'id': f"{benchmark_name}_{i}",
                'input': item.get('question', item.get('input', item.get('text'))),
                'expected_output': item.get('answer', item.get('output', item.get('label'))),
                'source': benchmark_name,
                'original_id': item.get('id')
            }
            samples.append(sample)
        
        return samples
    
    def adapt_to_domain(self,
                        samples: List[dict],
                        source_domain: str,
                        target_domain: str,
                        adapter_llm: str = "gpt-4o") -> List[dict]:
        """Adapt samples from one domain to another using LLM"""
        
        from openai import OpenAI
        client = OpenAI()
        
        adapted_samples = []
        
        for sample in samples:
            prompt = f"""Adapt this {source_domain} example to {target_domain}:

Original Input: {sample['input']}
Original Output: {sample['expected_output']}

Create a new example that:
1. Tests similar capabilities
2. Is relevant to {target_domain}
3. Has similar difficulty level
4. Maintains the same type of reasoning

Return JSON:
{{
    "adapted_input": "...",
    "adapted_output": "...",
    "adaptation_notes": "..."
}}
"""
            
            response = client.chat.completions.create(
                model=adapter_llm,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            adapted = json.loads(response.choices[0].message.content)
            adapted_samples.append({
                **sample,
                'adapted_input': adapted['adapted_input'],
                'adapted_output': adapted['adapted_output'],
                'target_domain': target_domain
            })
        
        return adapted_samples

# Example: Adapt general QA to customer service
transfer = DomainTransfer()

# Load from SQuAD
qa_samples = transfer.load_benchmark('squad', num_samples=50)

# Adapt to customer service domain
cs_samples = transfer.adapt_to_domain(
    samples=qa_samples,
    source_domain='general knowledge QA',
    target_domain='e-commerce customer service'
)
```

---

## 4.5 Strategy 3: Expert-Seeded Examples

Have domain experts create high-quality seed examples.

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class ExpertExample:
    """A hand-crafted expert example"""
    id: str
    input: str
    ideal_output: str
    reasoning: str  # Expert's explanation
    common_mistakes: List[str]
    difficulty: str
    importance: str  # How critical is this case?
    created_by: str
    created_at: datetime
    tags: List[str]

class ExpertSeedingWorkflow:
    """Structured workflow for expert example creation"""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.examples = []
        self.coverage_matrix = {}
    
    def define_coverage_requirements(self, 
                                     categories: List[str],
                                     capabilities: List[str],
                                     min_examples_per_cell: int = 3):
        """Define what we need to cover"""
        
        self.coverage_matrix = {
            (cat, cap): {
                'required': min_examples_per_cell,
                'current': 0,
                'examples': []
            }
            for cat in categories
            for cap in capabilities
        }
    
    def add_expert_example(self, example: ExpertExample, 
                          category: str, capability: str):
        """Add an expert-created example"""
        
        self.examples.append(example)
        
        key = (category, capability)
        if key in self.coverage_matrix:
            self.coverage_matrix[key]['current'] += 1
            self.coverage_matrix[key]['examples'].append(example.id)
    
    def get_coverage_report(self) -> dict:
        """Report on coverage status"""
        
        total_required = sum(c['required'] for c in self.coverage_matrix.values())
        total_current = sum(c['current'] for c in self.coverage_matrix.values())
        
        gaps = [
            {'category': k[0], 'capability': k[1], 
             'gap': v['required'] - v['current']}
            for k, v in self.coverage_matrix.items()
            if v['current'] < v['required']
        ]
        
        return {
            'coverage_percentage': total_current / total_required * 100,
            'total_examples': len(self.examples),
            'gaps': gaps,
            'next_priority': max(gaps, key=lambda x: x['gap']) if gaps else None
        }
    
    def export_for_annotation_tool(self, format: str = 'labelstudio') -> str:
        """Export examples for annotation tools"""
        
        if format == 'labelstudio':
            tasks = [
                {
                    'id': ex.id,
                    'data': {
                        'text': ex.input,
                        'ideal_output': ex.ideal_output,
                        'reasoning': ex.reasoning
                    },
                    'annotations': []
                }
                for ex in self.examples
            ]
            return json.dumps(tasks, indent=2)
        
        raise ValueError(f"Unknown format: {format}")

# Example usage
workflow = ExpertSeedingWorkflow('/path/to/storage')

# Define what we need to cover
workflow.define_coverage_requirements(
    categories=['product', 'order', 'refund', 'complaint', 'general'],
    capabilities=['factual_accuracy', 'empathy', 'escalation', 'policy_adherence'],
    min_examples_per_cell=3
)

# Expert adds an example
example = ExpertExample(
    id='exp_001',
    input="I've been waiting 3 weeks for my laptop and no one can tell me where it is!",
    ideal_output="""I completely understand your frustration, and I sincerely apologize for 
    this delay. Let me look into this right away. Could you please provide your order 
    number so I can track down exactly where your laptop is and give you a clear update?""",
    reasoning="""This response demonstrates empathy first (acknowledging frustration), 
    takes ownership (apologizing), and moves to action (asking for order number). 
    It doesn't make promises about delivery that can't be kept.""",
    common_mistakes=[
        "Being defensive or making excuses",
        "Promising a delivery date without checking",
        "Not acknowledging the customer's frustration"
    ],
    difficulty='medium',
    importance='high',
    created_by='jane_cs_expert',
    created_at=datetime.now(),
    tags=['frustrated_customer', 'order_delay', 'laptop']
)

workflow.add_expert_example(example, 'complaint', 'empathy')
print(workflow.get_coverage_report())
```

---

## 4.6 Strategy 4: Bootstrapping with LLM-as-Judge

Use LLMs to create initial labels, then refine.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BOOTSTRAPPING WITH LLM-AS-JUDGE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: Generate Samples                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Model under test generates outputs for diverse inputs               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  PHASE 2: LLM Labels                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Strong LLM (GPT-4) labels outputs with quality scores               │    │
│  │ Labels: good/bad, scores 1-5, specific issues                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  PHASE 3: Confidence Filtering                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Keep high-confidence labels (agreement across multiple runs)        │    │
│  │ Flag uncertain cases for human review                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  PHASE 4: Human Validation                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Experts review sample of labels (10-20%)                            │    │
│  │ Estimate label accuracy, refine prompts if needed                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│                                     ▼                                        │
│  PHASE 5: Golden Set Creation                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Combine validated LLM labels with human-verified examples           │    │
│  │ Create initial golden test set                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class BootstrapLabeler:
    """Bootstrap labels using LLM, then validate with humans"""
    
    def __init__(self, 
                 labeling_model: str = "gpt-4o",
                 confidence_threshold: float = 0.8,
                 num_labeling_runs: int = 3):
        self.client = OpenAI()
        self.labeling_model = labeling_model
        self.confidence_threshold = confidence_threshold
        self.num_runs = num_labeling_runs
    
    def label_batch(self, samples: List[dict], 
                   evaluation_criteria: str) -> List[dict]:
        """Label samples with confidence scores"""
        
        labeled = []
        
        for sample in samples:
            # Multiple labeling runs for confidence
            labels = []
            for _ in range(self.num_runs):
                label = self._get_label(sample, evaluation_criteria)
                labels.append(label)
            
            # Compute confidence as agreement
            scores = [l['score'] for l in labels]
            avg_score = sum(scores) / len(scores)
            score_variance = sum((s - avg_score)**2 for s in scores) / len(scores)
            confidence = 1.0 - min(score_variance, 1.0)  # Lower variance = higher confidence
            
            labeled.append({
                **sample,
                'bootstrap_label': {
                    'score': avg_score,
                    'confidence': confidence,
                    'needs_human_review': confidence < self.confidence_threshold,
                    'individual_labels': labels
                }
            })
        
        return labeled
    
    def _get_label(self, sample: dict, criteria: str) -> dict:
        """Get a single label from the LLM"""
        
        prompt = f"""Evaluate this AI output.

Input: {sample['input']}
Output: {sample['output']}

Evaluation Criteria: {criteria}

Provide:
1. A score from 0.0 to 1.0
2. A brief justification
3. Any specific issues found

Return as JSON:
{{
    "score": 0.X,
    "justification": "...",
    "issues": ["...", "..."]
}}
"""
        
        response = self.client.chat.completions.create(
            model=self.labeling_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3  # Lower temperature for more consistent labels
        )
        
        return json.loads(response.choices[0].message.content)
    
    def create_human_review_queue(self, 
                                  labeled_samples: List[dict]) -> dict:
        """Separate samples needing human review"""
        
        needs_review = [s for s in labeled_samples 
                       if s['bootstrap_label']['needs_human_review']]
        auto_labeled = [s for s in labeled_samples 
                       if not s['bootstrap_label']['needs_human_review']]
        
        # Also sample some auto-labeled for validation
        validation_sample_size = max(10, int(len(auto_labeled) * 0.1))
        validation_sample = random.sample(auto_labeled, 
                                          min(validation_sample_size, len(auto_labeled)))
        
        return {
            'uncertain_cases': needs_review,
            'validation_sample': validation_sample,
            'auto_labeled': auto_labeled,
            'stats': {
                'total': len(labeled_samples),
                'needs_review': len(needs_review),
                'auto_labeled': len(auto_labeled),
                'review_percentage': len(needs_review) / len(labeled_samples) * 100
            }
        }
```

---

## 4.7 Strategy 5: Active Sampling from Production

Intelligently select samples from production for labeling.

```python
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

class ActiveSampler:
    """Select the most informative samples for labeling"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(embedding_model)
    
    def diversity_sampling(self, 
                          samples: List[dict],
                          num_to_select: int,
                          num_clusters: int = None) -> List[dict]:
        """Select diverse samples using clustering"""
        
        if num_clusters is None:
            num_clusters = min(num_to_select, len(samples) // 5)
        
        # Embed all samples
        texts = [s['input'] + ' ' + s.get('output', '') for s in samples]
        embeddings = self.embedder.encode(texts)
        
        # Cluster
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Select samples closest to cluster centers
        selected = []
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = embeddings[cluster_indices]
            center = kmeans.cluster_centers_[cluster_id]
            
            # Find closest to center
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected.append(samples[closest_idx])
        
        # If we need more, sample uniformly from remaining
        if len(selected) < num_to_select:
            remaining = [s for s in samples if s not in selected]
            additional = random.sample(remaining, 
                                       min(num_to_select - len(selected), len(remaining)))
            selected.extend(additional)
        
        return selected[:num_to_select]
    
    def uncertainty_sampling(self,
                            samples: List[dict],
                            scorer,
                            num_to_select: int) -> List[dict]:
        """Select samples where the model is most uncertain"""
        
        scored_samples = []
        for sample in samples:
            # Get model's confidence/uncertainty
            result = scorer.score(sample)
            sample['uncertainty'] = 1.0 - result.get('confidence', 0.5)
            scored_samples.append(sample)
        
        # Sort by uncertainty (highest first)
        scored_samples.sort(key=lambda x: x['uncertainty'], reverse=True)
        
        return scored_samples[:num_to_select]
    
    def edge_case_detection(self,
                           samples: List[dict],
                           num_to_select: int) -> List[dict]:
        """Find potential edge cases based on outlier detection"""
        
        # Embed
        texts = [s['input'] for s in samples]
        embeddings = self.embedder.encode(texts)
        
        # Calculate distance from centroid
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        # Select outliers (furthest from centroid)
        outlier_indices = np.argsort(distances)[-num_to_select:]
        
        return [samples[i] for i in outlier_indices]
    
    def combined_sampling(self,
                         samples: List[dict],
                         num_to_select: int,
                         scorer = None) -> List[dict]:
        """Combine multiple strategies for balanced selection"""
        
        per_strategy = num_to_select // 3
        
        selected = []
        
        # Diversity
        selected.extend(self.diversity_sampling(samples, per_strategy))
        
        # Edge cases
        remaining = [s for s in samples if s not in selected]
        selected.extend(self.edge_case_detection(remaining, per_strategy))
        
        # Uncertainty (if scorer available)
        if scorer:
            remaining = [s for s in samples if s not in selected]
            selected.extend(self.uncertainty_sampling(remaining, scorer, per_strategy))
        else:
            # Random for remaining
            remaining = [s for s in samples if s not in selected]
            selected.extend(random.sample(remaining, 
                                          min(per_strategy, len(remaining))))
        
        return selected[:num_to_select]

# Example usage
sampler = ActiveSampler()

# Get production samples
production_samples = fetch_production_logs(limit=10000)

# Select most informative 500 for labeling
to_label = sampler.combined_sampling(
    samples=production_samples,
    num_to_select=500,
    scorer=quality_scorer
)
```

---

## 4.8 Cold Start Timeline

A practical timeline for bootstrapping evaluations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COLD START TIMELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DAY 1-2: Foundation                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Define evaluation dimensions (accuracy, safety, helpfulness)      │    │
│  │ • Set up basic infrastructure (storage, scoring)                    │    │
│  │ • Create 10-20 expert examples covering key categories              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     ↓                                        │
│  DAY 3-5: Synthetic Expansion                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Generate 200+ synthetic test cases                                │    │
│  │ • Use templates for high-volume generation                          │    │
│  │ • Create adversarial test cases                                     │    │
│  │ • Set up LLM-as-judge for automated scoring                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     ↓                                        │
│  WEEK 2: Production Data Integration                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Start collecting production samples (1-5% sampling)               │    │
│  │ • Run active sampling for diverse selection                         │    │
│  │ • Bootstrap labels with LLM                                         │    │
│  │ • Human review of uncertain cases                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     ↓                                        │
│  WEEK 3-4: Validation & Refinement                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Validate LLM labels against human judgments                       │    │
│  │ • Calculate inter-annotator agreement                               │    │
│  │ • Refine scoring prompts based on disagreements                     │    │
│  │ • Build first golden test set (100-500 examples)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     ↓                                        │
│  MONTH 2+: Continuous Improvement                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ • Integrate user feedback into eval data                            │    │
│  │ • Mine failure cases from production                                │    │
│  │ • Expand coverage systematically                                    │    │
│  │ • Set up regular eval runs in CI/CD                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4.8b Worked Examples (2026)

Three concrete patterns for getting from "empty dataset" to "100 trustworthy items" in a week.

#### Example 1 — Synthetic seed set with structured outputs

Frontier models in 2026 support guaranteed JSON schemas, which makes synthetic-data generation reliable instead of regex-prone.

```python
# pip install openai pydantic
from pydantic import BaseModel, Field
from typing import Literal
from openai import OpenAI

client = OpenAI()

class SupportTicket(BaseModel):
    user_message: str = Field(..., description="What the customer says")
    intent: Literal["refund", "shipping", "product_question", "complaint"]
    expected_action: str = Field(..., description="What the bot should do")
    difficulty: Literal["easy", "medium", "hard"]

class Batch(BaseModel):
    items: list[SupportTicket]

resp = client.chat.completions.parse(
    model="gpt-4o-2024-11-20",
    response_format=Batch,
    messages=[{"role": "user", "content":
        "Generate 30 diverse customer-support tickets for a DTC apparel brand. "
        "Cover all four intents and all three difficulties. Vary tone, length, "
        "non-native English, and include 3 adversarial / jailbreak attempts."}],
)
for t in resp.choices[0].message.parsed.items:
    print(t.intent, t.difficulty, "|", t.user_message[:80])
```

Always have a human spot-check at least 20% of synthetic items before promoting them into your eval set — LLM generators have their own distributional biases.

#### Example 2 — EvalGen-style criteria discovery

Don't guess your rubric. Grade outputs first, *then* let the LLM propose the criteria your grading implies.

```python
# 1. Have a human label ~30 outputs as good/bad with a one-line critique.
labels = [
    {"output": "Sure! Click 'Forgot password' on the login page.",
     "label": "good", "critique": "clear, actionable, brand voice"},
    {"output": "I cannot help with that.",
     "label": "bad",  "critique": "refuses a benign request"},
    # ...28 more
]

# 2. Ask Claude to extract a rubric from your labels.
from anthropic import Anthropic
a = Anthropic()
rubric = a.messages.create(
    model="claude-sonnet-4-5", max_tokens=600, temperature=0,
    messages=[{"role": "user", "content":
        f"Here are 30 graded support-bot outputs:\n\n{labels}\n\n"
        "Infer a 5-criterion rubric (each criterion binary, with a one-sentence "
        "definition) that would reproduce these grades. Return JSON."}],
).content[0].text
print(rubric)
# Now you have a HUMAN-derived rubric you can hand to an LLM-judge.
```

This is the EvalGen / "Who Validates the Validators" (Shankar et al. 2024) workflow: criteria emerge *from* the data instead of being imposed on it.

#### Example 3 — Bootstrap from production logs after week 1

Once you have any traffic, the cheapest gold examples are real ones. Sample from the *low-confidence* tail.

```python
import pandas as pd

logs = pd.read_parquet("prod_traces.parquet")  # cols: id, input, output, judge_score, judge_conf

# Stratified review queue: 60% low-confidence, 30% low-score, 10% random
low_conf  = logs.nsmallest(60, "judge_conf")
low_score = logs.nsmallest(30, "judge_score")
random_   = logs.sample(10, random_state=42)

review = pd.concat([low_conf, low_score, random_]).drop_duplicates("id")
review.to_csv("week2_human_review.csv", index=False)
print(f"{len(review)} items queued for human labeling — promote 'good' ones to eval set.")
```

---

## 4.9 Exercises

### Exercise 1: Cold Start Plan
Create a detailed cold start plan for evaluating a new code generation AI assistant. Include:
- Day-by-day timeline for first two weeks
- Specific data generation strategies
- Coverage requirements
- Success criteria for each phase

### Exercise 2: Synthetic Data Quality
Generate 50 synthetic test cases for a medical Q&A chatbot. Then:
- Have the LLM critique its own outputs
- Filter for quality
- Identify gaps in coverage

### Exercise 3: Active Sampling Implementation
Implement an active sampling system that:
- Uses embedding-based diversity sampling
- Incorporates uncertainty estimates
- Balances exploration vs exploitation

---

## Next Module
→ [Module 5: Scaling & Optimization](../05-scaling/README.md)

