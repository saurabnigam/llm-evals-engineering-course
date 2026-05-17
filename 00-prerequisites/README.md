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
        model="gpt-4o",
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

# TEMPERATURE: How "creative" vs "deterministic" the output is
# 0.0 = Always pick most likely word (more consistent)
# 1.0 = More randomness (more creative/varied)
response = model.generate(prompt, temperature=0.0)  # For evals, use 0.0

# TOKENS: How LLMs measure text length (roughly ~4 chars = 1 token)
# "Hello world" ≈ 2 tokens
# Cost and limits are measured in tokens

# CONTEXT WINDOW: How much text the model can "see" at once
# GPT-4o: 128K tokens ≈ 300 pages
# Claude Opus/Sonnet 4.5: 200K tokens ≈ 500 pages
# Gemini 2.5 Pro: 1M+ tokens ≈ 2,500 pages

# REASONING MODELS (introduced 2024, mainstream by 2026):
# OpenAI o-series (o3, o4-mini), Claude with extended thinking,
# Gemini 2.5 Thinking, DeepSeek-R1.
# These models produce an explicit "thinking" trace before answering.
# Implication for evals: you can score the *reasoning chain* itself,
# not just the final answer (see CoT-faithfulness evals in module 02).
```

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

## 0.4 Your First Eval (Hands-On)

Let's build a simple evaluation system using OpenAI and LangChain.

### Setup

```bash
pip install openai langchain langchain-openai python-dotenv
```

```python
# .env file
OPENAI_API_KEY=sk-your-key-here
```

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
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0  # Deterministic for evals
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
    """Use LLM to judge quality"""
    eval_prompt = f"""Rate this AI response on a scale of 1-5 for:
1. Accuracy: Is the information correct?
2. Helpfulness: Does it answer the question?
3. Clarity: Is it easy to understand?

User Question: {input_text}
AI Response: {response}

Return JSON: {{"accuracy": X, "helpfulness": X, "clarity": X, "reasoning": "..."}}
"""
    
    result = client.chat.completions.create(
        model="gpt-4o",  # Use stronger model for judging
        messages=[{"role": "user", "content": eval_prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(result.choices[0].message.content)

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
    
    # Aggregate results
    avg_accuracy = sum(r["quality"]["accuracy"] for r in results) / len(results)
    avg_helpfulness = sum(r["quality"]["helpfulness"] for r in results) / len(results)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"Average Accuracy: {avg_accuracy}/5")
    print(f"Average Helpfulness: {avg_helpfulness}/5")
    
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
llm = ChatOpenAI(model="gpt-4o", temperature=0)

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
    model="gpt-4o",
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

### Pattern 3: LLM-as-Judge
```python
def eval_llm_judge(question: str, response: str) -> float:
    prompt = f"Rate this response 1-5: Q: {question}, A: {response}"
    score = llm.invoke(prompt)
    return float(score) / 5.0
```

### Pattern 4: Pairwise Comparison
```python
def eval_pairwise(question: str, response_a: str, response_b: str) -> str:
    prompt = f"Which is better? A: {response_a}, B: {response_b}"
    return llm.invoke(prompt)  # Returns "A" or "B"
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
| **LLM** | Large Language Model (GPT-4, Claude, etc.) |
| **Prompt** | Instructions/context given to the model |
| **Completion** | The model's output/response |
| **Token** | Unit of text (~4 characters) |
| **Temperature** | Randomness setting (0 = deterministic) |
| **Hallucination** | Model inventing false information |
| **RAG** | Retrieval-Augmented Generation |
| **Grounding** | Basing responses on provided context |
| **Eval** | Evaluation - measuring model quality |
| **Golden Dataset** | Curated test set with verified answers |

