# Module 0: Prerequisites - ML/AI Concepts for Software Engineers

> **For engineers who know software but are new to ML/AI**

## 0.1 How LLMs Work (5-Minute Version)

If you've worked with traditional software, here's the mental model shift:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           TRADITIONAL SOFTWARE vs LLM-BASED SOFTWARE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRADITIONAL SOFTWARE                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  Input ──▶ [ Deterministic Code Logic ] ──▶ Output                 │     │
│  │                                                                    │     │
│  │  • Same input = Same output (always)                               │     │
│  │  • Behavior defined by code you wrote                              │     │
│  │  • Errors are bugs in YOUR logic                                   │     │
│  │  • Testing: Assert exact outputs                                   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  LLM-BASED SOFTWARE                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │  Input ──▶ [ Probabilistic Neural Network ] ──▶ Output             │     │
│  │                                                                    │     │
│  │  • Same input ≈ Similar output (usually)                          │     │
│  │  • Behavior "learned" from training data                           │     │
│  │  • Errors might be in training, prompts, OR the model              │     │
│  │  • Testing: Measure quality distributions                          │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  KEY INSIGHT: LLMs are like hiring a very capable but unpredictable         │
│  contractor. You can't unit test them - you need to EVALUATE them.          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What's Actually Happening Inside

```python
# Traditional software: You define the logic
def get_greeting(name: str, time_of_day: str) -> str:
    if time_of_day == "morning":
        return f"Good morning, {name}!"
    elif time_of_day == "evening":
        return f"Good evening, {name}!"
    return f"Hello, {name}!"

# LLM-based: You describe what you want, model figures out how
def get_greeting_llm(name: str, time_of_day: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-5.5",
        messages=[{
            "role": "user",
            "content": f"Generate a friendly greeting for {name}. It's {time_of_day}."
        }]
    )
    return response.choices[0].message.content
    # Output: "Hey there, {name}! Hope you're having a wonderful morning! ☀️"
    # OR: "Good morning, {name}! Ready to seize the day?"
    # OR: ... (many possible outputs)
```

---

## 0.2 Key Terminology for Software Engineers

| Traditional Software Term | LLM Equivalent | Notes |
|--------------------------|----------------|-------|
| Function/Method | Model | Takes input, produces output |
| Source Code | Model Weights | The "logic" - but you can't read it |
| Configuration | Prompt | Instructions that shape behavior |
| Unit Test | Evaluation | Measures quality, not correctness |
| Bug | Failure Mode | Categories of wrong behavior |
| Debugging | Error Analysis | Understanding why things go wrong |
| Code Coverage | Eval Coverage | How many scenarios are tested |
| Regression Test | Eval Suite | Catches quality degradation |
| Build Pipeline | Eval Pipeline | Automated quality checking |

### New Concepts You'll Need

```python
# PROMPT: The instructions you give to the LLM
prompt = """
You are a helpful customer service agent for TechCorp.
Be professional but friendly. Never make promises about refunds
without checking the order status first.
"""

# COMPLETION: The LLM's response
completion = model.generate(prompt + user_message)

# TEMPERATURE: a sampling knob that used to control randomness.
# 0.0 = always pick the most likely next token; 1.0 = more varied.
#
# ⚠️  READ THIS, IT IS THE MOST COMMONLY TAUGHT OUT-OF-DATE FACT IN EVALS:
# For years the standard advice was "set temperature=0 for evals so runs are
# reproducible." That advice is now wrong twice over.
#   1. It never worked. temperature=0 reduces variance; it does not eliminate
#      it. Batching, hardware non-determinism, and MoE routing all mean the
#      same prompt can produce different outputs at temperature 0.
#   2. It no longer runs. Current frontier models REJECT sampling parameters:
#      temperature/top_p/top_k return a 400 error on Claude Opus 5, Opus
#      4.7/4.8, Fable 5, and (for non-default values) Sonnet 5.
# The modern replacement is statistical: run each case n times and report an
# interval, not a point estimate. Full treatment in Module 15 §15.2.

# EFFORT: the knob that REPLACED temperature as the thing you tune.
# `output_config: {"effort": "low"|"medium"|"high"|"xhigh"|"max"}` controls how
# much the model thinks before answering. It changes BOTH score and cost, often
# by a lot. Record it next to every eval number you report (Module 15 §15.3).

# TOKENS: how LLMs measure text length. The rule of thumb is ~4 characters per
# token for ordinary English prose — but it is only a rule of thumb, and it is
# badly wrong for code, JSON, and non-English text (often 1.5–3x more tokens).
# Never estimate a bill or a context budget from character counts, and never
# use a different vendor's tokenizer (tiktoken is OpenAI's and undercounts
# Claude tokens by 15-20% on prose, far more on code). Count exactly:
#     client.messages.count_tokens(model="claude-opus-5", messages=[...])

# CONTEXT WINDOW: how much text the model can "see" at once.
# The mid-2026 frontier converged on 1M tokens (≈ 2,500 pages): Claude Opus 5,
# Fable 5, Opus 4.8, and Sonnet 5 all offer 1M, as do Gemini 3.x Pro and the
# open-weights DeepSeek-V4 and Kimi K3. (Smaller/cheaper models still ship
# 128-256K; Claude Haiku 4.5 is 200K.)
# Note the cost of a long window is not just price: retrieval quality and
# instruction-following can degrade long before you hit the limit, which is why
# "it fits in the window" is not the same as "the model will use it well."

# REASONING MODELS (the default by 2026, not a niche):
# Almost every frontier model now produces an explicit "thinking" trace
# before answering — Claude with extended/adaptive thinking, OpenAI's
# GPT-5.x reasoning, Gemini 3.x Thinking, DeepSeek V4. "Effort" or
# "thinking budget" is now a tunable knob that changes both score and cost
# (e.g. Fable 5 reports benchmarks at "adaptive thinking, max effort").
# Implication for evals: you can score the *reasoning chain* itself, not
# just the final answer (see CoT-faithfulness evals in module 02) — AND
# you must record the effort setting next to every score, because the same
# model at "low" vs "max" effort is effectively two different systems.
```

---

## 0.2b How Modern Models Are Trained (the 60-Second Version)

You don't need to train models to evaluate them, but three words from the training pipeline now show up constantly in eval work, so here they are in plain terms (Module 11 has the full version):

```
PRETRAINING ──▶ SFT ──▶ RLHF / RLVR ──▶ (safety tuning) ──▶ shipped model
   "learn      "learn to   "learn to be      "learn to
    language"   follow       preferred/        refuse
                instructions" correct"          misuse"
```

- **RLHF** (RL from *Human* Feedback): humans rank model answers, a reward model learns "what people prefer," and the model is optimized toward that. Great for taste and tone; weak when "good" is subjective.
- **RLVR** (RL from *Verifiable* Rewards): the reward is a checkable fact — *did the code pass the tests? is the math answer correct?* This is the engine behind the 2024–2026 leap in coding and math, and it's why frontier coding scores shot up (e.g. SWE-bench Verified climbing from the ~50% range in 2024 to ~95% for the best 2026 models — [SWE-bench Verified leaderboard](https://llm-stats.com/benchmarks/swe-bench-verified)).
- **Reward hacking**: when the model learns to satisfy the *checker* rather than the *intent* — e.g. hard-coding a test's expected output instead of solving the problem. Anthropic showed in Nov 2025 that reward hacking learned in production coding RL can generalize to broader misalignment ([arXiv:2511.18397](https://arxiv.org/abs/2511.18397)).

**Why an eval engineer cares:** the artifact you write to *grade* a model (a rubric, a unit test, a judge prompt) is the same kind of artifact used to *reward* it during RLVR. "Environments are the new datasets" — your eval set can become a training set ([Prime Intellect](https://www.primeintellect.ai/blog/environments)). That means **every grader you write is a potential reward spec, and is therefore hackable** — a theme you'll see in Modules 02, 10, and 11.

---

## 0.3 Why Evals Are Different from Tests

### The Core Challenge

```python
# Traditional Test: Binary pass/fail
def test_addition():
    assert add(2, 2) == 4  # Either right or wrong

# LLM Eval: Quality spectrum
def eval_summary():
    article = "Long article about climate change..."
    summary = llm.summarize(article)
    
    # How do you test this? There's no single "right" answer!
    # You need to EVALUATE multiple dimensions:
    
    accuracy_score = check_factual_accuracy(summary, article)  # 0.0 - 1.0
    conciseness_score = check_length_appropriate(summary)      # 0.0 - 1.0
    readability_score = check_readability(summary)             # 0.0 - 1.0
    
    # And aggregate somehow
    overall_score = (accuracy_score * 0.5 + 
                    conciseness_score * 0.3 + 
                    readability_score * 0.2)
    
    # Pass threshold, not exact match
    assert overall_score >= 0.8
```

### Types of "Wrong" in LLM Systems

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TAXONOMY OF LLM FAILURES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FACTUAL ERRORS                                                              │
│  └── "Paris is the capital of Germany"                                      │
│      Measurable, clear failure                                               │
│                                                                              │
│  HALLUCINATIONS                                                              │
│  └── "According to the 2024 Smith study..." (study doesn't exist)           │
│      Model invents plausible-sounding but false information                  │
│                                                                              │
│  INSTRUCTION FAILURES                                                        │
│  └── User asked for 3 items, got 5                                          │
│      Model didn't follow explicit instructions                               │
│                                                                              │
│  SAFETY FAILURES                                                             │
│  └── Provides harmful information, exhibits bias                            │
│      Most critical - can cause real harm                                     │
│                                                                              │
│  STYLE/TONE FAILURES                                                         │
│  └── Professional email sounds too casual                                   │
│      Content correct but presentation wrong                                  │
│                                                                              │
│  CAPABILITY FAILURES                                                         │
│  └── "I can't help with that" for a valid request                          │
│      Model refuses when it shouldn't                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 0.3b See It Yourself: The 60-Second Non-Determinism Demo

Everything in this course follows from one observable fact, and you should observe it rather than take my word for it. Run the same prompt five times and count how many distinct answers you get.

```python
# pip install anthropic
"""The whole reason eval engineering exists, in 15 lines."""
import anthropic
from collections import Counter

client = anthropic.Anthropic()

PROMPT = "Name one common cause of a slow database query. Reply with just the cause, under 10 words."

answers = []
for _ in range(5):
    r = client.messages.create(
        model="claude-opus-5", max_tokens=100,
        messages=[{"role": "user", "content": PROMPT}],
    )
    answers.append(next(b.text for b in r.content if b.type == "text").strip())

for a in answers:
    print(f"  {a}")
print(f"\n{len(set(answers))} distinct answers from {len(answers)} identical requests")
```

A representative run:

```
  Missing index on a frequently filtered column
  A missing index on the queried column
  Missing indexes on columns used in WHERE clauses
  Full table scan due to a missing index
  Missing index on the WHERE clause column

5 distinct answers from 5 identical requests
```

Now sit with what that means for testing:

- **All five answers are correct.** This is not a quality problem, and no amount of prompt tuning removes it.
- **`assert response == expected` fails on four of five runs**, despite the system working perfectly. Exact-match assertions do not merely fail here — they measure the wrong thing.
- **The variation is in the wording, not the substance.** So your check has to be about *meaning*, which is why LLM-as-judge (Module 02) exists at all.
- **You cannot conclude anything from one run.** If a single trial had returned something wrong, you would not know whether the system is broken or you got unlucky — that's the difference between a bug and a 20% failure rate, and it takes many runs to tell them apart (Module 15 §15.2).

> **Try this variant:** ask something with a genuinely single correct answer ("What is 17 × 23?"). You will usually get identical outputs. That contrast is the intuition to keep: **the more open-ended the task, the more your eval must measure properties rather than strings.**

---

## 0.4 Your First Eval (Hands-On)

Let's build a simple evaluation system using OpenAI and LangChain.

### Setup

```bash
pip install anthropic openai langchain langchain-openai python-dotenv
```

```python
# .env file — set whichever provider(s) you're using
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

> **Why both?** Evals are one of the few places where using two providers is a
> feature rather than overhead: a judge from the same family as the system under
> test shares its blind spots and tends to like its own style
> (**self-enhancement bias**). Mixing families is the cheapest way to notice.

### Simple Eval Example

```python
# first_eval.py
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# 1. Define what you're evaluating
def my_chatbot(user_message: str) -> str:
    """The system we want to evaluate"""
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        # NOTE: no temperature=0 here. See §0.2 — it does not give you
        # reproducibility, and current frontier models reject it outright.
        # Handle variance by running n times, not by pretending it's absent.
    )
    return response.choices[0].message.content

# 2. Define test cases
test_cases = [
    {
        "input": "What is 2 + 2?",
        "expected_contains": ["4"],
        "category": "math"
    },
    {
        "input": "What is the capital of France?",
        "expected_contains": ["Paris"],
        "category": "factual"
    },
    {
        "input": "Write a haiku about programming",
        "expected_properties": ["3_lines"],
        "category": "creative"
    }
]

# 3. Define evaluators
def simple_contains_eval(response: str, expected: list) -> float:
    """Check if response contains expected strings"""
    matches = sum(1 for exp in expected if exp.lower() in response.lower())
    return matches / len(expected)

def llm_quality_eval(input_text: str, response: str) -> dict:
    """Use an LLM to judge quality.

    ⚠️ The obvious design — "rate this 1-5 on accuracy, helpfulness, clarity" —
    is the single most common beginner mistake in this field, and this course
    spends Module 02 §1.4d explaining why. In short:
      • Models cluster hard on 3s and 4s, so a 5-point scale gives you maybe
        two usable values dressed up as five.
      • Nobody can say what separates a 3 from a 4, so the number is not
        reproducible between judges, runs, or people.
      • Asking about three dimensions in ONE call lets them contaminate each
        other: a well-written wrong answer scores high on "accuracy" because
        the judge is also looking at clarity.
      • You cannot act on "3.4". You can act on "FAIL: contradicted the source
        in paragraph 2."
    Do this instead: one ISOLATED call per criterion, a binary verdict, and
    quoted evidence. It costs more calls and it is worth it.
    """
    criteria = {
        "accuracy": "Is every factual claim in the response correct?",
        "helpfulness": "Does the response actually answer the question asked?",
    }
    verdicts = {}
    for name, question in criteria.items():
        result = client.chat.completions.create(
            model="gpt-5.5",  # a stronger model than the one under test
            messages=[{"role": "user", "content": (
                f"Question asked: {input_text}\n"
                f"Response given: {response}\n\n"
                f"Criterion — {name}: {question}\n"
                "Answer PASS or FAIL, then quote the exact text you relied on.\n"
                'Return JSON: {"verdict": "PASS"|"FAIL", "evidence": "..."}'
            )}],
            response_format={"type": "json_object"},
        )
        verdicts[name] = json.loads(result.choices[0].message.content)
    return verdicts

# 4. Run evaluation
def run_eval():
    results = []
    
    for test in test_cases:
        # Get model response
        response = my_chatbot(test["input"])
        
        # Run evaluators
        result = {
            "input": test["input"],
            "response": response,
            "category": test["category"]
        }
        
        # Simple eval if expected strings provided
        if "expected_contains" in test:
            result["contains_score"] = simple_contains_eval(
                response, test["expected_contains"]
            )
        
        # LLM quality eval
        result["quality"] = llm_quality_eval(test["input"], response)
        
        results.append(result)
        
        # Print progress
        print(f"\n{'='*50}")
        print(f"Input: {test['input']}")
        print(f"Response: {response[:100]}...")
        print(f"Quality: {result['quality']}")
    
    # Aggregate results — pass RATES, not averaged opinions
    print(f"\n{'='*50}\nSUMMARY")
    n = len(results)
    for criterion in ("accuracy", "helpfulness"):
        passed = sum(r["quality"][criterion]["verdict"] == "PASS" for r in results)
        print(f"{criterion}: {passed}/{n} passed ({passed / n:.0%})")

    # Print the failures — this is the part you actually act on. A pass rate
    # tells you IF something is wrong; the evidence strings tell you WHAT.
    for r in results:
        for criterion, v in r["quality"].items():
            if v["verdict"] == "FAIL":
                print(f"\nFAIL [{criterion}] on: {r['input']}\n  evidence: {v['evidence']}")

    return results

if __name__ == "__main__":
    run_eval()
```

---

## 0.5 Common LLM Architectures to Evaluate

### 1. Simple Chatbot

```
User Input ──▶ LLM ──▶ Response
```

Evaluate: Accuracy, Helpfulness, Safety

### 2. RAG (Retrieval-Augmented Generation)

```
User Query ──▶ [Retriever] ──▶ Relevant Docs ──▶ [LLM] ──▶ Response
                    │                              │
                    ▼                              ▼
            Evaluate Retrieval           Evaluate Generation
            (Did we find right docs?)   (Is answer grounded in docs?)
```

Evaluate: Retrieval quality, Groundedness, Attribution

### 3. AI Agents

```
User Goal ──▶ [Planner LLM] ──▶ [Tool 1] ──▶ [Tool 2] ──▶ ... ──▶ Result
                    │              │            │
                    ▼              ▼            ▼
              Evaluate Plan   Evaluate Tool Use  Evaluate Result
```

Evaluate: Planning quality, Tool selection, Task completion

### 4. Multi-Step Pipelines

```
Input ──▶ [Extract] ──▶ [Transform] ──▶ [Summarize] ──▶ Output
              │             │               │
              ▼             ▼               ▼
          Eval Step 1   Eval Step 2    Eval Step 3
```

Evaluate: Each step AND end-to-end quality

---

## 0.6 The Eval Engineering Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVAL ENGINEERING WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DEFINE SUCCESS                                                           │
│     └── What does "good" look like for your use case?                       │
│         - Accuracy thresholds                                                │
│         - Safety requirements                                                │
│         - User satisfaction targets                                          │
│                                                                              │
│  2. BUILD TEST DATASET                                                       │
│     └── Collect/generate representative examples                            │
│         - Happy paths                                                        │
│         - Edge cases                                                         │
│         - Adversarial inputs                                                 │
│                                                                              │
│  3. IMPLEMENT EVALUATORS                                                     │
│     └── How will you measure quality?                                       │
│         - Rule-based (fast, cheap)                                          │
│         - LLM-based (nuanced, expensive)                                    │
│         - Human (gold standard, slow)                                       │
│                                                                              │
│  4. RUN & ANALYZE                                                            │
│     └── Execute evals, understand failures                                  │
│         - Aggregate scores                                                   │
│         - Identify failure patterns                                          │
│         - Compare to baselines                                               │
│                                                                              │
│  5. INTEGRATE INTO CI/CD                                                     │
│     └── Automate quality gates                                              │
│         - Run on every change                                                │
│         - Block regressions                                                  │
│         - Alert on issues                                                    │
│                                                                              │
│  6. ITERATE                                                                  │
│     └── Continuously improve                                                │
│         - Add cases from production failures                                 │
│         - Refine evaluators                                                  │
│         - Update thresholds                                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 0.7 Tools of the Trade

### LangChain (What You'll Use)

```python
# LangChain makes LLM interactions easier
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Initialize model
llm = ChatOpenAI(model="gpt-5.5", temperature=0)

# Simple usage
response = llm.invoke("What is 2+2?")

# With prompts
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

chain = prompt | llm
response = chain.invoke({"user_input": "Hello!"})
```

### OpenAI API (Direct)

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Eval Frameworks (2026 landscape)

| Framework | Best For | Complexity |
|-----------|----------|------------|
| **Custom Python** | Full control, learning | Medium |
| **[Inspect AI](https://inspect.aisi.org.uk/)** (UK AISI) | Agent / safety / capability evals, sandboxed | Medium |
| **[LangSmith](https://docs.langchain.com/langsmith/evaluation)** | LangChain/LangGraph projects, tracing | Low |
| **[Braintrust](https://www.braintrust.dev/docs/evaluate)** | Offline + online (production) eval, CI/CD | Medium |
| **[Arize Phoenix](https://phoenix.arize.com/)** | Open-source production observability | Medium |
| **[W&B Weave](https://wandb.ai/site/weave)** | Experiment tracking, LLM traces | Medium |
| **[RAGAS](https://docs.ragas.io/)** | RAG and agent metrics | Low |
| **[promptfoo](https://www.promptfoo.dev/)** | Fast prompt A/B testing | Low |
| **[DeepEval](https://github.com/confident-ai/deepeval)** | Pytest-style LLM evals | Low |
| **[OpenAI Evals](https://github.com/openai/evals)** | Reference framework, model registry | Medium |

---

## 0.8 Quick Reference: Eval Patterns

### Pattern 1: Exact Match (Simple)
```python
def eval_exact(response: str, expected: str) -> bool:
    return response.strip().lower() == expected.strip().lower()
```

### Pattern 2: Contains Keywords
```python
def eval_contains(response: str, keywords: list) -> float:
    found = sum(1 for k in keywords if k.lower() in response.lower())
    return found / len(keywords)
```

### Pattern 3: LLM-as-Judge — binary verdict plus evidence
```python
def eval_llm_judge(question: str, response: str, criterion: str) -> dict:
    """One criterion per call. Binary verdict. Quoted evidence.

    NOT `f"Rate this response 1-5"` — see the warning in §0.4. A 1-5 scale
    looks more informative and is less: judges bunch on 3-4, the numbers
    aren't comparable across runs, and you can't act on a 3.4.
    """
    prompt = (
        f"Question: {question}\nResponse: {response}\n\n"
        f"Criterion — {criterion}\n"
        "Answer PASS or FAIL, then quote the exact text you relied on."
    )
    return llm.invoke(prompt)   # -> {"verdict": "PASS"|"FAIL", "evidence": "..."}
```

### Pattern 4: Pairwise Comparison — always swap positions
```python
def eval_pairwise(question: str, response_a: str, response_b: str) -> str:
    """Judges have POSITION BIAS: they favour whichever answer they see first,
    with the effect sometimes reaching 10-20 percentage points. Asking once
    measures the bias as much as the answers. So ask twice, swapped, and only
    count a win when the verdict survives the swap."""
    first  = llm.invoke(f"Which is better?\nA: {response_a}\nB: {response_b}")
    second = llm.invoke(f"Which is better?\nA: {response_b}\nB: {response_a}")
    if first == "A" and second == "B":
        return "response_a"
    if first == "B" and second == "A":
        return "response_b"
    return "tie"      # judge flipped with position — this is a real result, not an error
```

---

## Next Steps

Now that you understand the foundations, proceed to:

→ [Module 1: Fundamentals](../01-fundamentals/README.md) - Deep dive into eval concepts
→ [Module 2: Evaluation Methods](../02-evaluation-methods/README.md) - All the techniques

---

## Quick Glossary

| Term | Definition |
|------|------------|
| **LLM** | Large Language Model (Claude, GPT, Gemini, DeepSeek, Kimi, etc.) |
| **Prompt** | Instructions/context given to the model |
| **Completion** | The model's output/response |
| **Token** | Unit of text (~4 chars of English prose; far fewer chars per token for code and non-English — measure, don't estimate) |
| **Temperature** | Legacy sampling knob. **0 is not deterministic**, and current frontier models reject the parameter entirely (§0.2, Module 15 §15.2) |
| **Effort** | The knob that replaced it: how hard the model thinks (`low`→`max`). Changes score *and* cost; always report it with a score |
| **Hallucination** | Model inventing false information |
| **RAG** | Retrieval-Augmented Generation |
| **Grounding** | Basing responses on provided context |
| **Eval** | Evaluation - measuring model quality |
| **Golden Dataset** | Curated test set with verified answers, used as the reference standard |
| **Judge** | A model used to score another model's output (Module 02) |
| **pass@k** | Succeeded within k attempts. Flatters — one lucky run counts |
| **pass^k** | Succeeded on *every* one of k attempts. The honest reliability measure (Module 01 §1.3b) |
| **Coverage** | Share of eval cases that produced a real result. A pass rate without coverage can hide refusals and errors (Module 15 §15.4) |

---

## Further Reading

The two essays to read before anything else — both short, both practitioner-written:

- Hamel Husain, [**"Your AI Product Needs Evals"**](https://hamel.dev/blog/posts/evals/) — the case for error analysis over dashboards, and the 3-level eval hierarchy this course uses in Module 01.
- Eugene Yan, [**"Evaluating LLM-Evaluators"**](https://eugeneyan.com/writing/llm-evaluators/) — a survey of what actually works when a model grades a model.

Foundational papers, if you want the primary sources:

- Zheng et al., [**Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**](https://arxiv.org/abs/2306.05685) — where position bias, verbosity bias, and self-enhancement bias were first measured systematically.
- Liu et al., [**G-Eval**](https://arxiv.org/abs/2303.16634) — LLM evaluation with better human alignment.
- Verga et al., [**Replacing Judges with Juries**](https://arxiv.org/abs/2404.18796) — why a panel of smaller judges often beats one large one.
- Shankar et al., [**Who Validates the Validators?**](https://arxiv.org/abs/2404.12272) — the problem of aligning judges to human criteria, and why rubric-writing is the real work.

