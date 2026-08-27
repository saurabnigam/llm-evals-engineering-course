# Module 7: CI/CD Integration

## In Plain English

A CI eval gate should answer one narrow question: **is there enough evidence that this change is safe to advance?** Deterministic invariants may block on one failure; stochastic quality metrics need paired cases, adequate sample size, coverage checks, and an explicit tolerated effect. A red build should always point to cases and a decision—not merely a number that moved.

## 7.1 Why Integrate Evals in CI/CD?

Traditional software uses unit tests and integration tests in CI/CD. For AI systems, you need **evaluation gates** that catch quality regressions before they reach production.

| Gate or report | What it covers | Issue it catches | Decision it enables |
|---|---|---|---|
| Deterministic smoke checks | Schemas, forbidden actions, tool contracts, required fields | A prompt change makes every tool call invalid JSON | Block immediately with the exact invariant failure |
| Must-pass safety cases | Non-compensatory high-severity behavior | Average quality rises while one jailbreak or data-exfiltration case fails | Stop release; quality gains cannot average away the breach |
| Paired baseline/candidate comparison | Per-case change on the same eval set | A headline mean hides that the candidate fixed easy cases but regressed the costly failure mode | Block, approve, or inspect the disagreement set |
| Coverage and evaluator status | Whether the reported denominator includes attempted cases | Judge refusals/timeouts disappear and make the pass rate look better | Fail as unmeasured rather than accidentally pass |
| Repeated-run instability report | Variation from model, judge, or harness across trials | One borderline case alternates between pass and fail | Clarify the rubric, increase trials, or remove it from a hard gate |
| Canary deployment | Real behavior and operational guardrails on limited traffic | Offline score holds but latency, tool errors, or task completion regress | Roll back or continue gradual rollout |

The gate threshold belongs to the product’s failure cost and the suite’s statistical resolution. A familiar percentage is not evidence that the suite can detect a change that small.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL VS AI CI/CD                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRADITIONAL SOFTWARE                                                        │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │ Commit │─▶│ Build  │─▶│ Unit   │─▶│ Integ  │─▶│ Deploy │               │
│  │        │  │        │  │ Tests  │  │ Tests  │  │        │               │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘               │
│                              │            │                                  │
│                              ▼            ▼                                  │
│                         PASS/FAIL    PASS/FAIL                              │
│                                                                              │
│  AI SYSTEMS                                                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │
│  │ Commit │─▶│ Build  │─▶│ Unit   │─▶│ Eval   │─▶│ Safety │─▶│ Deploy │  │
│  │        │  │        │  │ Tests  │  │ Suite  │  │ Checks │  │        │  │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  │
│                              │            │           │                      │
│                              ▼            ▼           ▼                      │
│                         PASS/FAIL   THRESHOLD   PASS/FAIL                   │
│                                    (e.g., 85%)                              │
│                                                                              │
│  KEY DIFFERENCES:                                                            │
│  • Tests are probabilistic, not deterministic                               │
│  • Need thresholds and regression detection                                 │
│  • Must handle flaky evals                                                  │
│  • Cost considerations (LLM calls are expensive)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7.2 CI/CD Pipeline Architecture

### 7.2.1 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AI CI/CD PIPELINE ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Developer Push                                                              │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 1: QUICK CHECKS (< 2 min)                                    │    │
│  │  • Lint & format                                                    │    │
│  │  • Type checking                                                    │    │
│  │  • Unit tests                                                       │    │
│  │  • Prompt syntax validation                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 2: FAST EVALS (< 10 min)                                     │    │
│  │  • Smoke test (10-20 critical cases)                                │    │
│  │  • Format/structure validation                                      │    │
│  │  • Basic safety checks                                              │    │
│  │  • Cost: ~$0.50                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 3: CORE EVALS (< 30 min) [On PR merge to main]               │    │
│  │  • Full eval suite (100-500 cases)                                  │    │
│  │  • All evaluator types                                              │    │
│  │  • Regression detection                                             │    │
│  │  • Cost: ~$5-20                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 4: COMPREHENSIVE EVALS (< 2 hours) [Pre-release]             │    │
│  │  • Extended eval suite (1000+ cases)                                │    │
│  │  • Edge cases and adversarial tests                                 │    │
│  │  • Cross-model comparisons                                          │    │
│  │  • Cost: ~$50-100                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 5: CANARY DEPLOYMENT                                         │    │
│  │  • Deploy to 1-5% of traffic                                        │    │
│  │  • Real-time monitoring                                             │    │
│  │  • A/B comparison with production                                   │    │
│  │  • Auto-rollback on regression                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STAGE 6: FULL DEPLOYMENT                                           │    │
│  │  • Gradual rollout (5% → 25% → 50% → 100%)                         │    │
│  │  • Continuous monitoring                                            │    │
│  │  • Feedback collection                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7.3 GitHub Actions Implementation

### 7.3.1 Basic Eval Workflow

```yaml
# .github/workflows/eval.yml
name: AI Evaluation Pipeline

on:
  push:
    branches: [main, develop]
    tags: ['v*']
  pull_request:
    branches: [main]

env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  EVAL_CACHE_BUCKET: ${{ secrets.EVAL_CACHE_BUCKET }}

jobs:
  # Stage 1: Quick checks
  quick-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Lint
        run: ruff check .
      
      - name: Type check
        run: mypy src/
      
      - name: Unit tests
        run: pytest tests/unit -v
      
      - name: Validate prompts
        run: python scripts/validate_prompts.py

  # Stage 2: Fast evals (smoke tests)
  fast-evals:
    runs-on: ubuntu-latest
    needs: quick-checks
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run smoke tests
        run: |
          python -m eval_runner \
            --suite smoke \
            --output results/smoke.json \
            --threshold 0.9
      
      - name: Upload smoke test results
        uses: actions/upload-artifact@v4
        with:
          name: smoke-test-results
          path: results/smoke.json
      
      - name: Check smoke test threshold
        run: python scripts/check_threshold.py results/smoke.json --min-score 0.9

  # Stage 3: Core evals (on main branch only)
  core-evals:
    runs-on: ubuntu-latest
    needs: fast-evals
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Download eval cache
        run: |
          aws s3 sync s3://$EVAL_CACHE_BUCKET/cache ./eval_cache || true
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Run core evaluation suite
        run: |
          python -m eval_runner \
            --suite core \
            --output results/core.json \
            --cache-dir ./eval_cache \
            --parallel 10
      
      - name: Upload cache
        run: |
          aws s3 sync ./eval_cache s3://$EVAL_CACHE_BUCKET/cache
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      
      - name: Compare with baseline
        id: regression
        run: |
          python scripts/compare_baseline.py \
            results/core.json \
            baselines/main.json \
            --output comparison.json
      
      - name: Post results to PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const comparison = JSON.parse(fs.readFileSync('comparison.json', 'utf8'));
            
            let comment = '## Evaluation Results\n\n';
            comment += `| Metric | Current | Baseline | Delta |\n`;
            comment += `|--------|---------|----------|-------|\n`;
            
            for (const [metric, data] of Object.entries(comparison.metrics)) {
              const delta = data.current - data.baseline;
              const emoji = delta >= 0 ? '✅' : '⚠️';
              comment += `| ${metric} | ${data.current.toFixed(3)} | ${data.baseline.toFixed(3)} | ${emoji} ${delta >= 0 ? '+' : ''}${delta.toFixed(3)} |\n`;
            }
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
      
      - name: Fail on statistically resolved regression
        run: |
          python -c "import json,sys; c=json.load(open('comparison.json')); sys.exit(1 if c['has_regressions'] else 0)"
      
      - name: Upload candidate baseline for explicit approval
        if: success() && github.ref == 'refs/heads/main'
        uses: actions/upload-artifact@v4
        with:
          name: candidate-main-baseline
          path: results/core.json

  # Stage 4: Comprehensive evals (pre-release)
  comprehensive-evals:
    runs-on: ubuntu-latest
    needs: core-evals
    if: startsWith(github.ref, 'refs/tags/v')
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Run comprehensive suite
        run: |
          python -m eval_runner \
            --suite comprehensive \
            --output results/comprehensive.json \
            --parallel 20 \
            --timeout 7200
      
      - name: Generate release report
        run: |
          python scripts/generate_report.py \
            results/comprehensive.json \
            --format markdown \
            --output release_report.md
      
      - name: Upload release report
        uses: actions/upload-artifact@v4
        with:
          name: release-report
          path: release_report.md
```

### 7.3.2 Eval Runner Script

```python
# eval_runner.py
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import sys

class EvalRunner:
    """Main evaluation runner for CI/CD"""
    
    SUITES = {
        'smoke': {
            'dataset': 'datasets/smoke.json',
            'evaluators': ['format', 'safety'],
            'sample_limit': 20
        },
        'core': {
            'dataset': 'datasets/core.json',
            'evaluators': ['format', 'safety', 'quality', 'accuracy'],
            'sample_limit': 500
        },
        'comprehensive': {
            'dataset': 'datasets/comprehensive.json',
            'evaluators': ['format', 'safety', 'quality', 'accuracy', 'adversarial'],
            'sample_limit': None
        }
    }
    
    def __init__(self, 
                 suite: str,
                 cache_dir: Optional[str] = None,
                 parallel: int = 5):
        self.suite = suite
        self.config = self.SUITES[suite]
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.parallel = parallel
        
        # Initialize evaluators
        self.evaluators = self._load_evaluators()
        
        # Load cache
        self.cache = self._load_cache()
    
    def _load_evaluators(self) -> dict:
        """Load evaluators specified in suite config"""
        evaluators = {}
        
        for name in self.config['evaluators']:
            if name == 'format':
                evaluators[name] = FormatEvaluator()
            elif name == 'safety':
                evaluators[name] = SafetyEvaluator()
            elif name == 'quality':
                evaluators[name] = LLMQualityEvaluator()
            elif name == 'accuracy':
                evaluators[name] = AccuracyEvaluator()
            elif name == 'adversarial':
                evaluators[name] = AdversarialEvaluator()
        
        return evaluators
    
    def _load_cache(self) -> dict:
        """Load evaluation cache"""
        if self.cache_dir and (self.cache_dir / 'cache.json').exists():
            with open(self.cache_dir / 'cache.json') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save evaluation cache"""
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_dir / 'cache.json', 'w') as f:
                json.dump(self.cache, f)
    
    async def run(self) -> dict:
        """Run the evaluation suite"""
        
        start_time = datetime.now()
        
        # Load dataset
        with open(self.config['dataset']) as f:
            samples = json.load(f)
        
        if self.config['sample_limit']:
            samples = samples[:self.config['sample_limit']]
        
        print(f"Running {self.suite} suite with {len(samples)} samples...")
        
        # Get model outputs
        model = self._load_model()
        outputs = await self._generate_outputs(model, samples)
        
        # Run evaluations
        results = await self._run_evaluations(samples, outputs)
        
        # Aggregate results
        aggregated = self._aggregate_results(results)
        
        # Save cache
        self._save_cache()
        
        end_time = datetime.now()
        
        return {
            'suite': self.suite,
            'started_at': start_time.isoformat(),
            'finished_at': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'num_samples': len(samples),
            'results': results,
            'aggregated': aggregated,
            'evaluators': list(self.evaluators.keys())
        }
    
    async def _generate_outputs(self, model, samples: List[dict]) -> List[str]:
        """Generate model outputs for all samples"""
        
        async def generate_one(sample):
            import hashlib
            identity = json.dumps({
                'input': sample['input'],
                'model': str(model),
                'suite': self.suite,
            }, sort_keys=True)
            cache_key = "output:" + hashlib.sha256(identity.encode()).hexdigest()
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            output = await model.generate(sample['input'])
            self.cache[cache_key] = output
            return output
        
        semaphore = asyncio.Semaphore(self.parallel)
        
        async def with_semaphore(sample):
            async with semaphore:
                return await generate_one(sample)
        
        outputs = await asyncio.gather(*[with_semaphore(s) for s in samples])
        return outputs
    
    async def _run_evaluations(self, 
                               samples: List[dict], 
                               outputs: List[str]) -> List[dict]:
        """Run all evaluators on all samples"""
        
        results = []
        
        for i, (sample, output) in enumerate(zip(samples, outputs)):
            sample_result = {
                'id': sample.get('id', f'sample_{i}'),
                'input': sample['input'],
                'output': output,
                'expected': sample.get('expected_output'),
                'scores': {}
            }
            
            for eval_name, evaluator in self.evaluators.items():
                import hashlib
                identity = json.dumps({
                    'evaluator': eval_name,
                    'input': sample['input'],
                    'output': output,
                    'expected': sample.get('expected_output'),
                    'suite': self.suite,
                }, sort_keys=True)
                cache_key = "eval:" + hashlib.sha256(identity.encode()).hexdigest()
                
                if cache_key in self.cache:
                    score = self.cache[cache_key]
                else:
                    score = await evaluator.evaluate(
                        input=sample['input'],
                        output=output,
                        expected=sample.get('expected_output')
                    )
                    self.cache[cache_key] = score
                
                sample_result['scores'][eval_name] = score
            
            results.append(sample_result)
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(samples)}")
        
        return results
    
    def _aggregate_results(self, results: List[dict]) -> dict:
        """Compute aggregate metrics"""
        
        aggregated = {
            'overall': {},
            'by_evaluator': {},
            'by_category': {}
        }
        
        # By evaluator
        for eval_name in self.evaluators.keys():
            measured_rows = [
                r['scores'][eval_name]
                for r in results
                if eval_name in r['scores']
                and r['scores'][eval_name].get('score') is not None
                and r['scores'][eval_name].get('passed') is not None
            ]
            scores = [row['score'] for row in measured_rows]
            
            if scores:
                aggregated['by_evaluator'][eval_name] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores),
                    'pass_rate': sum(row['passed'] is True for row in measured_rows) / len(measured_rows),
                    'n': len(scores),
                }
        
        # Overall score (weighted average)
        weights = {
            'format': 0.1,
            'safety': 0.3,
            'quality': 0.3,
            'accuracy': 0.3
        }
        
        total_weight = 0
        weighted_sum = 0
        
        for eval_name, stats in aggregated['by_evaluator'].items():
            weight = weights.get(eval_name, 0.1)
            weighted_sum += stats['mean'] * weight
            total_weight += weight
        
        aggregated['overall']['score'] = weighted_sum / total_weight if total_weight else None
        case_verdicts = []
        for row in results:
            expected = [row['scores'].get(name) for name in self.evaluators]
            if all(item is not None and item.get('passed') is not None for item in expected):
                case_verdicts.append(all(item['passed'] is True for item in expected))
        aggregated['overall']['pass_rate'] = (
            sum(case_verdicts) / len(case_verdicts) if case_verdicts else None
        )
        expected_measurements = len(results) * len(self.evaluators)
        measured = sum(
            1 for r in results for score in r['scores'].values()
            if score and score.get('score') is not None and score.get('passed') is not None
        )
        aggregated['overall']['coverage'] = (
            measured / expected_measurements if expected_measurements else 0.0
        )
        # Safety is a must-pass dimension; a high style/quality score must not
        # average away a safety failure.
        aggregated['overall']['all_must_pass'] = bool(results) and all(
            r['scores'].get('safety', {}).get('passed') is True
            for r in results
        )
        
        return aggregated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, choices=['smoke', 'core', 'comprehensive'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--parallel', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.8)
    
    args = parser.parse_args()
    
    runner = EvalRunner(
        suite=args.suite,
        cache_dir=args.cache_dir,
        parallel=args.parallel
    )
    
    results = asyncio.run(runner.run())
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Check threshold
    overall_score = results['aggregated']['overall']['score']
    if overall_score is None:
        print("❌ FAILED: no measured results")
        sys.exit(1)
    print(f"\nOverall Score: {overall_score:.3f}")
    print(f"Threshold: {args.threshold}")
    if not results['aggregated']['overall']['all_must_pass']:
        print("❌ FAILED: a must-pass safety check failed")
        sys.exit(1)
    if overall_score < args.threshold:
        print(f"❌ FAILED: Score {overall_score:.3f} < threshold {args.threshold}")
        sys.exit(1)
    else:
        print(f"✅ PASSED: Score {overall_score:.3f} >= threshold {args.threshold}")

if __name__ == '__main__':
    main()
```

### 7.3.3 Regression Detection Script

```python
# scripts/compare_baseline.py
import argparse
import json
import sys
from typing import Dict

def compare_results(current: Dict, baseline: Dict) -> Dict:
    """Compare current results with baseline"""
    current_score = current['aggregated']['overall']['score']
    baseline_score = baseline['aggregated']['overall']['score']
    if current_score is None or baseline_score is None:
        raise ValueError(
            "current and baseline must both contain measured overall scores"
        )
    
    comparison = {
        'current_score': current_score,
        'baseline_score': baseline_score,
        'metrics': {},
        'regressions': [],
        'improvements': []
    }
    
    # Compare by evaluator
    for eval_name in current['aggregated']['by_evaluator']:
        current_mean = current['aggregated']['by_evaluator'][eval_name]['mean']
        baseline_mean = baseline['aggregated']['by_evaluator'].get(eval_name, {}).get('mean', 0)
        
        delta = current_mean - baseline_mean
        
        comparison['metrics'][eval_name] = {
            'current': current_mean,
            'baseline': baseline_mean,
            'delta': delta,
            'delta_pct': (delta / baseline_mean * 100) if baseline_mean else 0
        }
        
        # ⚠️ A FIXED THRESHOLD IS ONLY VALID IF YOUR SUITE CAN RESOLVE IT.
        # See the box below: a 2% gate on a 200-case suite fires on noise.
        # `min_detectable` is computed from the suite size, and a delta smaller
        # than it is reported as INDETERMINATE rather than as a pass or a fail.
        current_n = current['aggregated']['by_evaluator'][eval_name].get('n', 0)
        baseline_n = baseline['aggregated']['by_evaluator'].get(eval_name, {}).get('n', 0)
        n = min(current_n, baseline_n)
        min_detectable = minimum_detectable_effect(baseline_mean, n)

        if delta < -min_detectable:
            comparison['regressions'].append({
                'metric': eval_name,
                'delta': delta,
                'min_detectable': min_detectable,
                'severity': 'high' if delta < -2 * min_detectable else 'medium'
            })
        elif delta > min_detectable:
            comparison['improvements'].append({
                'metric': eval_name,
                'delta': delta
            })
        elif abs(delta) > 0.005:
            # Moved, but not by enough to distinguish from sampling noise.
            # Surfacing this is the point — silence here is what trains a team
            # to believe every wiggle is real.
            comparison.setdefault('indeterminate', []).append({
                'metric': eval_name,
                'delta': delta,
                'min_detectable': min_detectable,
                'note': f'need ~{required_n(baseline_mean, abs(delta)):,} cases to resolve this'
            })

    comparison['has_regressions'] = len(comparison['regressions']) > 0
    comparison['passed'] = not comparison['has_regressions']
    
    return comparison


def minimum_detectable_effect(baseline: float, n: int, power: float = 0.80) -> float:
    """Smallest change this suite size can distinguish from noise.

    Inverts the two-proportion sample-size formula (Module 15 §15.2). Returns
    a large number for tiny suites, which is the correct behaviour: it makes
    the gate refuse to fire rather than fire randomly.
    """
    from math import sqrt
    from statistics import NormalDist
    if n < 2:
        return 1.0
    z_a, z_b = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(power)
    # Approximation: solve mde ≈ (z_a + z_b) * sqrt(2 * p(1-p) / n)
    return (z_a + z_b) * sqrt(2 * baseline * (1 - baseline) / n)


def required_n(baseline: float, mde: float, power: float = 0.80) -> int:
    """Cases per arm needed to resolve a change of size `mde`."""
    from math import sqrt, ceil
    from statistics import NormalDist
    p1, p2 = baseline, max(baseline - mde, 1e-6)
    pbar = (p1 + p2) / 2
    z_a, z_b = NormalDist().inv_cdf(0.975), NormalDist().inv_cdf(power)
    num = (z_a * sqrt(2 * pbar * (1 - pbar)) + z_b * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return ceil(num / (mde ** 2))
```

> ### ⚠️ Sizing your gate: the threshold most CI configs get wrong
>
> Fixed rules such as `--max-regression 0.05` or `delta < -0.02` are common defaults, and on a realistically-sized eval suite they may be below the noise floor. The script above deliberately derives its gate from sample size instead of exposing an inert fixed-threshold flag. From the sample-size arithmetic in Module 15 §15.2, detecting a drop from an 85% baseline needs roughly:
>
> | Change you want to catch | Cases needed (per arm) |
> |---|---:|
> | 10 points | 250 |
> | 5 points | 906 |
> | 2 points | 5,274 |
>
> So a **2% regression gate on a 200-case suite is a coin flip with a CI badge.** It will fire on runs where nothing changed, the team will learn that red builds mean "re-run it", and the gate stops functioning as a gate — the failure mode is social, not statistical, and it is permanent once trust is gone.
>
> Three ways out, in order of preference:
>
> 1. **Set the threshold from the suite, not from taste.** `minimum_detectable_effect()` above does this. If it tells you your suite can only resolve 9-point swings, that is the honest gate — and the finding is that your suite is too small.
> 2. **Grow the suite where it gates.** You do not need 5,000 cases everywhere. You need them on the *one or two* metrics that block a release; everything else can be telemetry.
> 3. **Gate on something with less variance.** Deterministic checks (schema validity, refusal behaviour on a fixed red-team set, tool-call shape) have near-zero sampling noise and can carry tight thresholds honestly. Push the hard gate onto those and let the judged metrics inform rather than block.
>
> **Report the MDE next to the pass rate in every CI comment.** `87.2% — smallest resolvable change 9.4pp (n=200)` tells a reviewer instantly whether the number in front of them can support the decision they are about to make. Grow that suite to 500 cases and the same line reads `5.9pp` — which is the concrete argument for spending a week writing test cases, stated in the units the decision is actually made in.

```python
# (continued)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('current', help='Current results JSON file')
    parser.add_argument('baseline', help='Baseline results JSON file')
    parser.add_argument('--output', required=True, help='Output comparison file')
    
    args = parser.parse_args()
    
    with open(args.current) as f:
        current = json.load(f)
    
    with open(args.baseline) as f:
        baseline = json.load(f)
    
    comparison = compare_results(current, baseline)
    
    with open(args.output, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPARISON REPORT")
    print("=" * 60)
    print(f"\nOverall Score: {comparison['current_score']:.3f} (baseline: {comparison['baseline_score']:.3f})")
    
    if comparison['regressions']:
        print(f"\n⚠️  REGRESSIONS DETECTED:")
        for reg in comparison['regressions']:
            print(f"  - {reg['metric']}: {reg['delta']:+.3f} ({reg['severity']})")
    
    if comparison['improvements']:
        print(f"\n✅ IMPROVEMENTS:")
        for imp in comparison['improvements']:
            print(f"  - {imp['metric']}: {imp['delta']:+.3f}")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
```

---

## 7.4 Handling Non-Determinism

Hosted generative outputs may vary across calls. Here's how to handle that variability in CI/CD while keeping deterministic software checks deterministic:

### 7.4.1 Deterministic Testing Mode

```python
class DeterministicEvalConfig:
    """Configuration for reproducible evaluations"""
    
    def __init__(self):
        # Fix random seeds
        self.seed = 42
        
        # Sampling controls are model-specific; do not assume temperature=0
        # exists or guarantees identical output.
        self.model_temperature = None
        
        # Cache all LLM calls
        self.use_cache = True
        
        # Fixed test set (no sampling)
        self.sample_randomly = False

def setup_deterministic_mode():
    """Set up environment for deterministic evaluation"""
    import random
    import numpy as np
    
    random.seed(42)
    np.random.seed(42)
    
    # Pass any provider-supported seed explicitly in that provider's request.
    # An OPENAI_SEED environment variable is not an API setting.
```

> **⚠️ Honest naming: this is variance-*reduction* mode, not determinism.** Temperature 0 and seed parameters do **not** guarantee identical outputs from modern hosted models — mixture-of-experts routing, dynamic batching, and provider-side infrastructure changes all introduce run-to-run variation, and OpenAI documents its `seed` parameter as best-effort (Anthropic offers none). Two consequences: (1) the only *truly* deterministic layer is the cache — identical inputs should never hit the API twice; (2) any gate built on "same input ⇒ same output" will flake. Treat every eval score as a **sample from a distribution** and gate statistically — which is exactly what 7.4.2 and 7.4.3 below do. The judge is stochastic too: repeated judging and majority vote may reduce variance, but only use that policy after measuring its TPR/TNR, correlation, cost, and refusal coverage against held-out human labels.

### 7.4.1b How Many Test Cases Do You Actually Need?

The question every team asks first, and most courses never answer with a number. The noise floor of a pass-rate measured on *n* independent cases is its 95% confidence interval, half-width ≈ 1.96·√(p(1−p)/n). At a pass rate around 80%:

| Suite size *n* | 95% CI half-width | What you can detect |
|---|---|---|
| 50 cases | **±11 pp** | Only catastrophes. A "5-point regression" is invisible in the noise. |
| 200 cases | ±5.5 pp | Large regressions, reliably. |
| 1,000 cases | ±2.5 pp | The 3–5 pp regressions that actually reach production. |

Three practical consequences:

1. **A 50-case smoke suite is a tripwire, not a measurement.** That's still valuable — it catches "the prompt change broke JSON output" — but don't let a 4-point drop on 50 cases block a PR (it's noise), and don't let a 4-point *gain* on 50 cases justify shipping (same reason).
2. **Pair your comparisons to buy back sensitivity.** Run baseline and candidate on the *same* cases and analyze per-case deltas (which is what the bootstrap gate in 7.4.2 does): case-level difficulty variance cancels out, so paired designs detect differences that independent-sample math says you can't afford. The cases where the two versions *disagree* carry all the signal — read those transcripts, not just the aggregate.
3. **Grow the suite where the decisions are.** You don't need 1,000 cases everywhere — you need them on the failure modes that gate releases. A 50-case broad smoke + 300 cases concentrated on your top two failure modes (from error analysis, module 01 §1.4b) beats 1,000 uniformly random cases.

### 7.4.2 Statistical Significance Testing

```python
from scipy import stats
import numpy as np

class StatisticalEvalComparator:
    """Compare evaluations with statistical significance"""
    
    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
    
    def compare_means(self, 
                      current_scores: list, 
                      baseline_scores: list) -> dict:
        """Compare two sets of scores with statistical tests"""
        
        # Calculate basic stats
        current_mean = np.mean(current_scores)
        baseline_mean = np.mean(baseline_scores)
        
        if len(current_scores) != len(baseline_scores) or not current_scores:
            raise ValueError("paired comparisons require non-empty, equal-length score lists")

        # Candidate and baseline run on the same cases: test paired deltas.
        t_stat, p_value = stats.ttest_rel(current_scores, baseline_scores)
        
        # Effect size (Cohen's d)
        deltas = np.asarray(current_scores) - np.asarray(baseline_scores)
        delta_std = np.std(deltas, ddof=1) if len(deltas) > 1 else 0
        cohens_d = np.mean(deltas) / delta_std if delta_std > 0 else 0
        
        # Confidence interval for difference
        n1 = n2 = len(deltas)
        se = delta_std / np.sqrt(len(deltas)) if len(deltas) > 1 else 0
        ci_low = (current_mean - baseline_mean) - 1.96 * se
        ci_high = (current_mean - baseline_mean) + 1.96 * se
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        # Determine direction
        if is_significant:
            if current_mean > baseline_mean:
                conclusion = 'significant_improvement'
            else:
                conclusion = 'significant_regression'
        else:
            conclusion = 'no_significant_change'
        
        return {
            'current_mean': current_mean,
            'baseline_mean': baseline_mean,
            'difference': current_mean - baseline_mean,
            'p_value': p_value,
            'is_significant': is_significant,
            'effect_size': cohens_d,
            'effect_magnitude': self._interpret_cohens_d(cohens_d),
            'confidence_interval': [ci_low, ci_high],
            'conclusion': conclusion,
            'sample_sizes': {'current': n1, 'baseline': n2}
        }
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        d = abs(d)
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def bootstrap_confidence_interval(self,
                                      scores: list,
                                      n_bootstrap: int = 1000,
                                      ci: float = 0.95) -> tuple:
        """Calculate bootstrap confidence interval for mean"""
        
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            means.append(np.mean(sample))
        
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        
        return (lower, upper)
```

### 7.4.3 Flaky Eval Detection

```python
class FlakyEvalDetector:
    """Detect and handle flaky evaluations"""
    
    def __init__(self, variance_threshold: float = 0.1):
        self.variance_threshold = variance_threshold
        self.history = {}  # Store historical results
    
    def run_with_retries(self, 
                         evaluator,
                         sample: dict,
                         n_runs: int = 3) -> dict:
        """Run evaluation multiple times to detect flakiness"""
        
        results = []
        for _ in range(n_runs):
            result = evaluator.evaluate(sample)
            results.append(result['score'])
        
        variance = np.var(results)
        mean = np.mean(results)
        
        is_flaky = variance > self.variance_threshold
        
        return {
            'mean_score': mean,
            'variance': variance,
            'is_flaky': is_flaky,
            'individual_runs': results,
            'score_range': [min(results), max(results)],
        }
    
    def identify_flaky_cases(self, 
                             all_results: list,
                             min_variance: float = 0.05) -> list:
        """Identify test cases that are consistently flaky"""
        
        flaky_cases = []
        
        for result in all_results:
            if 'variance' in result and result['variance'] > min_variance:
                flaky_cases.append({
                    'id': result['id'],
                    'variance': result['variance'],
                    'reason': self._diagnose_flakiness(result)
                })
        
        return flaky_cases
    
    def _diagnose_flakiness(self, result: dict) -> str:
        """Try to diagnose why a case is flaky"""
        
        # Check if it's borderline
        if 0.4 < result['mean_score'] < 0.6:
            return 'borderline_quality'
        
        # Check if input is ambiguous
        if 'ambiguous' in result.get('flags', []):
            return 'ambiguous_input'
        
        # Disagreement is observable; a judge's self-reported confidence is not
        # a calibrated error probability.
        if result.get('judge_disagreement_rate', 0.0) > 0.2:
            return 'evaluator_instability'
        
        return 'unknown'
```

---

## 7.5 Cost Management in CI/CD

### 7.5.1 Budget Enforcement

```python
class CICDBudgetManager:
    """Manage evaluation costs in CI/CD"""
    
    # Cost per evaluation type (approximate)
    COSTS = {
        'format': 0.001,
        'safety': 0.005,
        'quality': 0.015,  # frontier-judge call (e.g. GPT-5.5 / Sonnet 4.6)
        'accuracy': 0.015,
        'adversarial': 0.02
    }
    
    def __init__(self, 
                 daily_budget: float = 100.0,
                 per_run_budget: float = 20.0):
        self.daily_budget = daily_budget
        self.per_run_budget = per_run_budget
        self.daily_spend = 0.0
    
    def estimate_cost(self, 
                      suite: str, 
                      num_samples: int) -> float:
        """Estimate cost for an evaluation run"""
        
        suite_evaluators = {
            'smoke': ['format', 'safety'],
            'core': ['format', 'safety', 'quality', 'accuracy'],
            'comprehensive': ['format', 'safety', 'quality', 'accuracy', 'adversarial']
        }
        
        evaluators = suite_evaluators.get(suite, [])
        
        total = 0.0
        for evaluator in evaluators:
            total += self.COSTS.get(evaluator, 0.01) * num_samples
        
        return total
    
    def can_run(self, estimated_cost: float) -> dict:
        """Check if we have budget for this run"""
        
        remaining_daily = self.daily_budget - self.daily_spend
        
        if estimated_cost > self.per_run_budget:
            return {
                'allowed': False,
                'reason': f'Exceeds per-run budget (${estimated_cost:.2f} > ${self.per_run_budget:.2f})',
                'suggestion': f'Reduce sample size to {int(self.per_run_budget / estimated_cost * 100)}%'
            }
        
        if estimated_cost > remaining_daily:
            return {
                'allowed': False,
                'reason': f'Exceeds remaining daily budget (${estimated_cost:.2f} > ${remaining_daily:.2f})',
                'suggestion': 'Wait until tomorrow or use smoke tests only'
            }
        
        return {
            'allowed': True,
            'estimated_cost': estimated_cost,
            'remaining_after': remaining_daily - estimated_cost
        }
    
    def record_spend(self, amount: float):
        """Record actual spend"""
        self.daily_spend += amount
```

### 7.5.2 Tiered Evaluation Strategy

```yaml
# .github/workflows/tiered-eval.yml
name: Tiered Evaluation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  determine-tier:
    runs-on: ubuntu-latest
    outputs:
      tier: ${{ steps.tier.outputs.tier }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Determine evaluation tier
        id: tier
        run: |
          # Check what changed
          CHANGED_FILES=$(git diff --name-only ${{ github.event.before }} ${{ github.sha }})
          
          # Tier 3: Model or prompt changes (full eval)
          if echo "$CHANGED_FILES" | grep -E "^(models/|prompts/|src/core/)" ; then
            echo "tier=comprehensive" >> $GITHUB_OUTPUT
            exit 0
          fi
          
          # Tier 2: Logic changes (core eval)
          if echo "$CHANGED_FILES" | grep -E "^src/" ; then
            echo "tier=core" >> $GITHUB_OUTPUT
            exit 0
          fi
          
          # Tier 1: Other changes (smoke test only)
          echo "tier=smoke" >> $GITHUB_OUTPUT

  run-evaluation:
    needs: determine-tier
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run ${{ needs.determine-tier.outputs.tier }} evaluation
        run: |
          python -m eval_runner \
            --suite ${{ needs.determine-tier.outputs.tier }} \
            --output results.json
```

---

## 7.6 Monitoring and Alerting

### 7.6.1 CI/CD Dashboard Integration

```python
# scripts/report_to_dashboard.py
import requests
import json
import os

class DashboardReporter:
    """Report eval results to monitoring dashboard"""
    
    def __init__(self, dashboard_url: str, api_key: str):
        self.url = dashboard_url
        self.api_key = api_key
    
    def report_run(self, results: dict, metadata: dict):
        """Report evaluation run to dashboard"""
        
        payload = {
            'run_id': metadata.get('run_id'),
            'commit_sha': os.environ.get('GITHUB_SHA'),
            'branch': os.environ.get('GITHUB_REF_NAME'),
            'suite': results['suite'],
            'timestamp': results['finished_at'],
            'metrics': {
                'overall_score': results['aggregated']['overall']['score'],
                'pass_rate': results['aggregated']['overall']['pass_rate'],
                **{
                    f"{k}_mean": v['mean'] 
                    for k, v in results['aggregated']['by_evaluator'].items()
                }
            },
            'sample_count': results['num_samples'],
            'duration_seconds': results['duration_seconds']
        }
        
        response = requests.post(
            f"{self.url}/api/eval-runs",
            json=payload,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        
        return response.status_code == 200
    
    def report_alert(self, alert_type: str, details: dict):
        """Send alert for regression or failure"""
        
        payload = {
            'type': alert_type,
            'severity': details.get('severity', 'medium'),
            'commit_sha': os.environ.get('GITHUB_SHA'),
            'details': details
        }
        
        response = requests.post(
            f"{self.url}/api/alerts",
            json=payload,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        
        return response.status_code == 200

# Usage in CI
def main():
    reporter = DashboardReporter(
        dashboard_url=os.environ['DASHBOARD_URL'],
        api_key=os.environ['DASHBOARD_API_KEY']
    )
    
    with open('results.json') as f:
        results = json.load(f)
    
    reporter.report_run(results, {'run_id': os.environ.get('GITHUB_RUN_ID')})
    
    # Check for regressions
    if results.get('has_regression'):
        reporter.report_alert('regression', {
            'severity': 'high',
            'metrics': results['regressions']
        })
```

### 7.6.2 Slack/Discord Integration

```python
import requests

class SlackNotifier:
    """Send eval notifications to Slack"""
    
    def __init__(self, webhook_url: str, pass_threshold: float):
        self.webhook_url = webhook_url
        self.pass_threshold = pass_threshold
    
    def notify_completion(self, results: dict, comparison: dict = None):
        """Notify on eval completion"""
        
        score = results['aggregated']['overall']['score']
        coverage = results['aggregated']['overall']['coverage']
        must_pass = results['aggregated']['overall']['all_must_pass']
        measured = score is not None and coverage == 1.0
        passed = measured and must_pass and score >= self.pass_threshold
        
        color = '#36a64f' if passed else ('#f2c744' if not measured else '#ff0000')
        status_emoji = '✅' if passed else ('⚠️' if not measured else '❌')
        score_text = f"{score:.3f}" if score is not None else "UNMEASURED"
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Evaluation Complete: {results['suite'].title()}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Overall Score:* {score_text}"},
                    {"type": "mrkdwn", "text": f"*Coverage:* {coverage:.1%}"},
                    {"type": "mrkdwn", "text": f"*Samples:* {results['num_samples']}"},
                    {"type": "mrkdwn", "text": f"*Duration:* {results['duration_seconds']:.1f}s"},
                    {"type": "mrkdwn", "text": f"*Commit:* `{os.environ.get('GITHUB_SHA', 'unknown')[:7]}`"}
                ]
            }
        ]
        
        # Add comparison if available
        if comparison:
            delta = comparison['current_score'] - comparison['baseline_score']
            delta_emoji = '📈' if delta > 0 else '📉' if delta < 0 else '➡️'
            
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{delta_emoji} *vs Baseline:* {delta:+.3f}"
                }
            })
            
            if comparison.get('regressions'):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "⚠️ *Regressions:*\n" + "\n".join(
                            f"• {r['metric']}: {r['delta']:+.3f}"
                            for r in comparison['regressions']
                        )
                    }
                })
        
        # Add action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Details"},
                    "url": f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
                }
            ]
        })
        
        requests.post(self.webhook_url, json={
            "attachments": [{
                "color": color,
                "blocks": blocks
            }]
        })
```

---

## 7.7 Best Practices

### 7.7.1 CI/CD Eval Checklist

```markdown
## CI/CD Evaluation Checklist

### Pre-Commit
- [ ] Prompt syntax validation
- [ ] Type checking passes
- [ ] Unit tests pass

### PR Checks
- [ ] Smoke tests pass (< 5 min)
- [ ] No critical safety failures
- [ ] Format validation passes

### Merge to Main
- [ ] Core eval suite passes threshold (85%)
- [ ] No significant regressions (< 5% drop)
- [ ] All evaluator categories pass
- [ ] Results stored and baseline updated

### Pre-Release
- [ ] Comprehensive eval suite passes
- [ ] Edge case coverage verified
- [ ] Adversarial tests pass
- [ ] Cross-model comparison done
- [ ] Release report generated

### Production Deployment
- [ ] Canary deployment successful
- [ ] No alerts in monitoring window
- [ ] A/B test shows no regression
- [ ] Gradual rollout completed
- [ ] Feedback monitoring active
```

### 7.7.2 Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Slow evals | CI takes 30+ minutes | Use hierarchical evaluation, cache results |
| Flaky tests | Random failures | Run multiple times, use statistical comparison |
| High costs | $100+/day on evals | Use tiered strategy, sample intelligently |
| False positives | Good changes blocked | Set appropriate thresholds, allow overrides |
| Outdated baselines | Compare against old model | Update baselines regularly |
| Missing coverage | Blind spots in testing | Continuously expand test sets from feedback |

---

## 7.7b Worked Examples (2026)

Three drop-in CI patterns that make eval gates fast, cheap, and trustworthy.

#### Example 1 — GitHub Actions: Inspect AI eval as a required check

```yaml
# .github/workflows/evals.yml
name: evals
on:
  pull_request:
    paths: ["prompts/**", "src/**", "evals/**"]

jobs:
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install inspect-ai openai anthropic
      - name: Run smoke evals (cheap model, 50 samples)
        env:
          OPENAI_API_KEY:    ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          inspect eval evals/smoke.py \
            --model openai/gpt-5.4-mini \
            --limit 50 \
            --log-dir logs/ \
            --fail-on-error
      - name: Enforce regression gate (mean accuracy ≥ 0.85)
        run: |
          python evals/gate.py logs/ --metric accuracy --min 0.85
      - uses: actions/upload-artifact@v4
        with: { name: eval-logs, path: logs/ }
```

#### Example 2 — Statistical regression gate (bootstrapped CI, not raw delta)

Don’t fail a PR because mean accuracy dropped 0.01 — that’s noise. Compare 95% bootstrap intervals.

```python
# evals/gate.py
import sys, json, glob, numpy as np

def bootstrap_ci(scores, n=10_000, alpha=0.05):
    rng = np.random.default_rng(0)
    means = [rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(n)]
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1 - alpha/2))

def load(log_dir):
    scores = []
    for f in glob.glob(f"{log_dir}/*.json"):
        for s in json.load(open(f))["samples"]:
            scores.append(s["score"]["value"])
    return scores

curr = load(sys.argv[1])
base = load("baseline_logs/")          # checked into the repo or pulled from main

lo_c, hi_c = bootstrap_ci(curr)
lo_b, hi_b = bootstrap_ci(base)
print(f"baseline: [{lo_b:.3f}, {hi_b:.3f}]   current: [{lo_c:.3f}, {hi_c:.3f}]")

# This separate-interval sketch is conservative and discards the paired-case
# design. In a production gate, join candidate and baseline rows by case ID,
# bootstrap their per-case deltas, and fail only when the delta interval is
# wholly below the tolerated regression margin.
```

#### Example 3 — Two-tier CI: smoke on every push, full eval on merge to main

```yaml
# .github/workflows/evals-full.yml
name: evals-full
on:
  push:
    branches: [main]
  schedule:
    - cron: "0 6 * * *"   # nightly 06:00 UTC

jobs:
  full:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install inspect-ai openai anthropic
      - name: Full eval suite (1000 samples, frontier model, batch API)
        env:
          OPENAI_API_KEY:    ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          inspect eval evals/full.py \
            --model anthropic/claude-sonnet-4-6 \
            --limit 1000 --epochs 3 \
            --log-dir logs/
      - name: Push results to Braintrust dashboard
        run: braintrust experiment push logs/
```

Key idea: PRs run a fast/cheap subset (smoke) for fast feedback; main and nightly runs use the strong-model + larger-N suite to catch subtler regressions, with the cost amortized via Batch APIs (see module 5).

### 7.7c Evals in the agentic-coding era: when the PR was written by an AI

By 2026 a large and growing share of the diffs flowing through CI are written by coding agents (Claude Code, Codex CLI, etc.), not typed by hand. That changes what your eval gates are *for*. Two failure modes are now first-class CI concerns:

1. **Flaws that pass unremarked.** An agent can produce code that compiles, passes the existing tests, and is subtly wrong. Anthropic's Opus 4.8 launch claims it is "around four times less likely than its predecessor to allow flaws in code it has written to pass unremarked" ([Opus 4.8 launch post](https://www.anthropic.com/news/claude-opus-4-8)) — i.e. labs now *measure this as a capability*, and so should your pipeline. Don't let "all tests green" stand in for "correct" when the tests themselves may have been written by the same agent.

2. **Reward hacking / test gaming.** When an agent is optimized to make a check pass, it may satisfy the *checker* rather than the *intent* — hard-coding an expected output, weakening an assertion, `skip`-ing a failing test, or special-casing the grader's input. In controlled research where internal reasoning was available, OpenAI observed explicit plans such as "Let's hack" when an environment was gameable ([CoT monitoring](https://openai.com/index/chain-of-thought-monitoring/)); Anthropic showed reward-hacking behavior can generalize beyond the immediate task ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397)). This does not mean ordinary CI users can retrieve a model's private reasoning.

Practical CI hardening for AI-authored changes:

```
┌──────────────────────────────────────────────────────────────────────┐
│  EVAL GATES FOR AI-WRITTEN CODE                                       │
├──────────────────────────────────────────────────────────────────────┤
│  • Diff the TESTS separately from the SOURCE. A PR that only          │
│    loosens/deletes assertions or adds `skip`/`xfail` is a red flag.   │
│  • Run a held-out test set the agent never saw (don't let it train    │
│    to your visible checks — this is contamination; see module 12).    │
│  • Mutation-test critical paths: if the agent's new tests still pass  │
│    after you inject a bug, the tests are theater.                     │
│  • Add an LLM-judge "did this change do what the PR says?" review     │
│    that reads the diff + PR description, independent of test results. │
│  • Keep visible reasoning summaries, tool logs, and outputs in the    │
│    CI artifact so a human can audit observable actions. Do not        │
│    require or retain private chain-of-thought.                        │
└──────────────────────────────────────────────────────────────────────┘
```

The throughline with the rest of this course: your test suite is a grader, and **any grader an agent is optimized against is a reward spec it can hack** (modules 02, 10, 11). In the agentic-coding era, "green CI" is necessary but no longer sufficient.

---

## 7.8 Exercises

### Exercise 1: Set Up Basic CI/CD Evals
Create a GitHub Actions workflow that:
- Runs smoke tests on every push
- Runs core evals on PR merge
- Posts results as PR comment
- Blocks merge on regression

### Exercise 2: Implement Cost-Aware Evaluation
Build a system that:
- Estimates cost before running
- Dynamically adjusts sample size based on budget
- Tracks spend over time
- Alerts when approaching limits

### Exercise 3: Build a Release Gate
Create an automated release gate that:
- Runs comprehensive evals before release
- Compares with statistical significance
- Generates release report
- Requires manual approval on borderline cases

---

## Next Module
→ [Module 8: Real-World Case Studies](../08-case-studies/README.md)
