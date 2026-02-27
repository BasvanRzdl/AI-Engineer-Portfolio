---
date: 2025-02-27
type: technology
topic: "Evaluation Frameworks: Ragas, Promptfoo, DeepEval"
project: "Phase 1 - Prompt Engineering Laboratory"
status: complete
---

# Technology: Evaluation Frameworks

## In My Own Words

Rather than building every evaluation metric from scratch, existing frameworks provide battle-tested implementations. The three main options — **Ragas**, **Promptfoo**, and **DeepEval** — each have different strengths. Understanding them helps decide what to build ourselves vs. what to adopt.

## Why This Matters

- Saves development time by reusing proven metrics
- Provides industry-standard evaluation approaches
- Helps benchmark our custom evaluation against established tools
- May integrate directly into our framework

---

## Framework Comparison

| Feature | Ragas | Promptfoo | DeepEval |
|---------|-------|-----------|----------|
| **Language** | Python | TypeScript/Node | Python |
| **Focus** | RAG evaluation, metrics | Prompt testing/comparison | LLM evaluation |
| **Approach** | Library with metric classes | CLI + config-driven | Library with test runner |
| **LLM-as-Judge** | ✅ Built-in | ✅ Built-in | ✅ Built-in |
| **CI/CD** | Via pytest | Native CLI | Via pytest |
| **Red-teaming** | ❌ | ✅ Strong | ✅ Basic |
| **Stars** | ~7K+ | ~10.7K | ~4K+ |
| **License** | Apache 2.0 | MIT | Apache 2.0 |

---

## Ragas (Retrieval Augmented Generation Assessment)

### Overview

Ragas is a Python framework focused on evaluating RAG pipelines and LLM applications. It provides a rich set of metrics that can be used independently.

### Available Metrics

#### RAG-Specific Metrics

| Metric | Measures | Inputs Required |
|--------|----------|----------------|
| **Faithfulness** | Is the answer grounded in the context? | question, answer, contexts |
| **Answer Relevancy** | Does the answer address the question? | question, answer |
| **Context Precision** | Are relevant contexts ranked higher? | question, contexts, ground_truth |
| **Context Recall** | Are all relevant facts captured? | answer, contexts, ground_truth |
| **Context Utilization** | How well are contexts used? | question, answer, contexts |

#### NLP Comparison Metrics

| Metric | Measures | Type |
|--------|----------|------|
| **BLEU Score** | N-gram overlap (precision) | Reference-based |
| **ROUGE Score** | N-gram overlap (recall) | Reference-based |
| **String Similarity** | Levenshtein-like distance | Reference-based |
| **Semantic Similarity** | Embedding cosine similarity | Reference-based |
| **Factual Correctness** | Factual alignment with reference | Reference-based |
| **Non-LLM String Similarity** | Various string distance metrics | Reference-based |

#### General Purpose Metrics (LLM-as-Judge)

| Metric | Measures | How |
|--------|----------|-----|
| **Aspect Critic** | Custom aspect evaluation | LLM judges a specific aspect |
| **Simple Criteria Score** | Single criterion scoring | LLM scores on one dimension |
| **Rubrics Score** | Multi-level rubric evaluation | LLM scores against detailed rubric |
| **Instance-Specific Rubrics** | Per-example rubrics | Custom rubric per test case |
| **Domain-Specific Evaluation** | Domain-tailored metrics | LLM with domain knowledge |

#### Agent / Tool Use Metrics

| Metric | Measures |
|--------|----------|
| **Tool Call Accuracy** | Were the right tools called? |
| **Agent Goal Accuracy** | Did the agent achieve its goal? |
| **Topic Adherence** | Did the agent stay on topic? |

### Usage Pattern

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset

# Prepare evaluation dataset
eval_data = {
    "question": ["What is the capital of France?"],
    "answer": ["The capital of France is Paris."],
    "contexts": [["France is a country in Europe. Its capital is Paris."]],
    "ground_truth": ["Paris is the capital of France."]
}
dataset = Dataset.from_dict(eval_data)

# Run evaluation
result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
)
print(result)
```

### Using Individual Metrics

```python
from ragas.metrics import AspectCritic

# Custom aspect evaluation
accuracy_critic = AspectCritic(
    name="accuracy",
    definition="Is the response factually accurate based on the given context?",
    llm=your_llm_wrapper,  # Ragas LLM wrapper
)

# Score a single example
score = await accuracy_critic.single_turn_ascore(
    sample=SingleTurnSample(
        user_input="What is X?",
        response="X is Y.",
        reference="X is Y and Z."
    )
)
```

### Key Takeaway for Our Project

Ragas provides excellent **metric implementations** — especially for reference-based evaluation and LLM-as-judge patterns. We can either:
- Use Ragas directly as a dependency
- Study their metric implementations and build our own inspired versions
- Use Ragas for validation and our custom metrics for production

---

## Promptfoo

### Overview

Promptfoo is a CLI tool and library for evaluating and red-teaming LLM applications. It's config-driven with YAML, making it easy to set up comparison tests.

### Architecture

```
promptfooconfig.yaml
├── prompts:        # Prompt templates to test
├── providers:      # LLM providers to use
├── tests:          # Test cases with assertions
└── defaultTest:    # Shared test configuration
```

### Configuration Example

```yaml
# promptfooconfig.yaml
description: "Summarization prompt comparison"

prompts:
  - "Summarize the following text in 3 bullet points:\n\n{{text}}"
  - "You are a professional summarizer. Create a concise 3-point summary:\n\n{{text}}"

providers:
  - openai:gpt-4o
  - openai:gpt-4o-mini

tests:
  - vars:
      text: "The quick brown fox..."
    assert:
      - type: contains
        value: "fox"
      - type: llm-rubric
        value: "The summary captures the main points accurately"
      - type: cost
        threshold: 0.01
      - type: latency
        threshold: 5000  # ms
  
  - vars:
      text: file://test_documents/doc1.txt
    assert:
      - type: similar
        value: "Expected summary text"
        threshold: 0.8
```

### Assertion Types

| Type | Description | Example |
|------|-------------|---------|
| `contains` | Output contains string | `value: "keyword"` |
| `not-contains` | Output doesn't contain string | `value: "forbidden"` |
| `equals` | Exact match | `value: "expected output"` |
| `starts-with` | Output starts with string | `value: "Sure,"` |
| `regex` | Regular expression match | `value: "\\d{4}"` |
| `is-json` | Valid JSON output | — |
| `contains-json` | Contains JSON object | — |
| `javascript` | Custom JS assertion | `value: "output.length < 500"` |
| `python` | Custom Python assertion | `value: "file://eval.py"` |
| `similar` | Semantic similarity | `threshold: 0.8` |
| `llm-rubric` | LLM judges quality | `value: "Is accurate"` |
| `model-graded-closedqa` | LLM fact-checking | `value: "reference answer"` |
| `cost` | Max cost per call | `threshold: 0.01` |
| `latency` | Max latency | `threshold: 5000` |

### CLI Usage

```bash
# Run evaluation
npx promptfoo eval

# View results in web UI
npx promptfoo view

# Compare two configs
npx promptfoo eval --config config_a.yaml --config config_b.yaml

# Red-teaming
npx promptfoo redteam init
npx promptfoo redteam run
```

### Red-Teaming

Promptfoo has strong red-teaming capabilities:

```yaml
# redteam.yaml
redteam:
  purpose: "Financial document summarizer"
  plugins:
    - prompt-injection
    - jailbreak
    - harmful
    - pii
    - overreliance
  strategies:
    - basic
    - jailbreak
    - prompt-injection
```

This automatically generates adversarial test cases and checks model vulnerabilities.

### Key Takeaway for Our Project

Promptfoo excels at **comparison testing** and **red-teaming**. It's ideal for:
- Quick A/B comparisons between prompt versions
- CI/CD integration for prompt regression tests
- Security testing with red-teaming
- However, it's Node.js-based, so integration with our Python framework requires subprocess calls or API

---

## DeepEval

### Overview

DeepEval is a Python evaluation framework that integrates with pytest and provides LLM-as-judge metrics.

### Usage Pattern

```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    GEval,
)

# Create test case
test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    retrieval_context=["France is a European country with Paris as its capital."],
    expected_output="Paris"
)

# Use built-in metrics
relevancy = AnswerRelevancyMetric(threshold=0.7)
faithfulness = FaithfulnessMetric(threshold=0.7)
hallucination = HallucinationMetric(threshold=0.5)

# Or custom G-Eval metrics
accuracy = GEval(
    name="Accuracy",
    criteria="Determine whether the response is factually accurate",
    evaluation_steps=[
        "Check if the main claim is correct",
        "Check for any factual errors",
        "Verify consistency with the context"
    ],
    evaluation_params=["input", "actual_output"],
    threshold=0.7
)

# Use with pytest
assert_test(test_case, [relevancy, faithfulness, hallucination, accuracy])
```

### Pytest Integration

```python
# test_prompts.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

@pytest.mark.parametrize("test_case", [
    LLMTestCase(input="Q1", actual_output="A1", expected_output="E1"),
    LLMTestCase(input="Q2", actual_output="A2", expected_output="E2"),
])
def test_prompt_quality(test_case):
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.7)])
```

### Key Takeaway for Our Project

DeepEval provides **Pythonic evaluation** with pytest integration. Good for:
- Running evaluations as part of the test suite
- G-Eval custom metrics (flexible LLM-as-judge)
- Dashboard for tracking evaluation results over time

---

## Decision Matrix for Our Project

| Need | Recommended Tool | Reasoning |
|------|-----------------|-----------|
| **Core evaluation metrics** | Custom (inspired by Ragas) | Full control, no heavy deps |
| **LLM-as-judge** | Custom implementation | Tailored to our prompt types |
| **Reference-based metrics** | Ragas or custom | BLEU/ROUGE/similarity |
| **Quick A/B testing** | Promptfoo (external tool) | Best-in-class comparison UI |
| **Red-teaming** | Promptfoo (external tool) | Strong adversarial testing |
| **CI integration** | Custom + pytest | Full control |
| **Heuristic checks** | Custom | Specific to our use cases |

### Recommended Approach

**Build custom, integrate selectively:**

1. Build our own evaluation core (metrics, test runners, reporting)
2. Use Ragas metrics as reference/inspiration
3. Use Promptfoo as an external tool for red-teaming and quick comparisons
4. Keep dependencies minimal — don't couple tightly to any framework

---

## Best Practices

- ✅ **Start simple** — heuristic + LLM-as-judge covers 90% of needs
- ✅ **Use established metrics** (BLEU, ROUGE, semantic sim) for reference-based eval
- ✅ **Combine frameworks** — use Promptfoo for red-teaming even if core is custom
- ✅ **Benchmark your custom metrics** against established framework results
- ❌ Don't adopt a heavyweight framework if you only need 2-3 metrics
- ❌ Don't ignore red-teaming — security evaluation is not optional

---

## Resources for Deeper Learning

- [Ragas Documentation](https://docs.ragas.io/) — Full metric reference and guides
- [Promptfoo Documentation](https://promptfoo.dev/docs/intro) — CLI tool guide
- [Promptfoo GitHub](https://github.com/promptfoo/promptfoo) — Source code and examples
- [DeepEval Documentation](https://docs.confident-ai.com/) — Python evaluation framework
- [G-Eval Paper](https://arxiv.org/abs/2303.16634) — The research behind G-Eval methodology
