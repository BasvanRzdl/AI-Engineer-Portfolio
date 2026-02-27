---
date: 2025-02-27
type: concept
topic: "LLM Evaluation Methodologies"
project: "Phase 1 - Prompt Engineering Laboratory"
status: complete
---

# Learning: LLM Evaluation Methodologies

## In My Own Words

LLM evaluation is the systematic process of measuring whether model outputs are "good enough" for your use case. It's harder than traditional software testing because there's rarely one "correct" answer — outputs exist on a spectrum of quality. Evaluation combines automated metrics, LLM-as-judge approaches, and heuristic checks to create a comprehensive quality picture.

## Why This Matters

- Without evaluation, you're guessing whether your prompts work
- Evaluation enables **data-driven** prompt improvement
- It catches regressions when you change prompts or models
- It builds confidence that a system is production-ready
- It's the foundation for A/B testing and prompt optimization

---

## The Evaluation Challenge

### Why LLM Evaluation is Hard

1. **No single correct answer**: "Summarize this document" has many valid summaries
2. **Quality is multidimensional**: An output can be accurate but poorly formatted, or fluent but hallucinated
3. **Context-dependent**: "Good" depends on the use case, audience, and requirements
4. **Scale**: Human evaluation doesn't scale; automated metrics have limitations
5. **Moving target**: Model behavior changes with updates, and evaluation criteria evolve

### The Evaluation Pyramid

```
            ┌──────────────┐
            │  Human Eval  │  ← Gold standard, expensive
            │  (Expert)    │
            ├──────────────┤
            │  LLM-as-     │  ← Scalable, decent proxy
            │  Judge       │
            ├──────────────┤
            │  Heuristic   │  ← Fast, cheap, partial
            │  Metrics     │
            ├──────────────┤
            │  Format /    │  ← Basic sanity checks
            │  Schema      │
            └──────────────┘
```

---

## Quality Dimensions

Every LLM output can be evaluated across multiple dimensions:

### Core Dimensions

| Dimension | What It Measures | Example Metric |
|-----------|-----------------|----------------|
| **Correctness** | Is the content factually accurate? | Factual accuracy score |
| **Relevance** | Does the output address the input? | Response relevancy |
| **Completeness** | Are all required elements present? | Checklist coverage |
| **Faithfulness** | Is the output grounded in provided context? | Hallucination rate |
| **Coherence** | Is the output well-organized and logical? | Coherence score |
| **Conciseness** | Is the output appropriately brief? | Length compliance |
| **Format Compliance** | Does it match the required format? | Schema validation pass/fail |
| **Safety** | Is the output free of harmful content? | Safety filter pass/fail |

### Operational Dimensions

| Dimension | What It Measures | How to Track |
|-----------|-----------------|--------------|
| **Latency** | Response time (seconds) | API timing |
| **Token Usage** | Input + output tokens | API response metadata |
| **Cost** | Dollar cost per call | Tokens × price per token |
| **Consistency** | Same input → similar outputs? | Variance across N runs |
| **Error Rate** | How often does parsing/validation fail? | Exception tracking |

---

## Evaluation Methods

### 1. Heuristic / Rule-Based Metrics

Fast, cheap, deterministic checks:

```python
class HeuristicEvaluator:
    """Rule-based evaluation — fast and cheap."""
    
    def evaluate_length(self, output: str, min_words: int, max_words: int) -> float:
        """Check if output is within expected length range."""
        word_count = len(output.split())
        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            return word_count / min_words
        else:
            return max_words / word_count
    
    def evaluate_format(self, output: str, expected_format: str) -> bool:
        """Check if output matches expected format."""
        if expected_format == "json":
            try:
                json.loads(output)
                return True
            except json.JSONDecodeError:
                return False
        elif expected_format == "bullet_list":
            return output.strip().startswith(("- ", "• ", "* "))
        return True
    
    def evaluate_contains_required(self, output: str, required_terms: list[str]) -> float:
        """Check if output contains required terms."""
        output_lower = output.lower()
        found = sum(1 for term in required_terms if term.lower() in output_lower)
        return found / len(required_terms) if required_terms else 1.0
    
    def evaluate_no_hallucination_signals(self, output: str) -> float:
        """Basic check for common hallucination patterns."""
        red_flags = [
            "I think", "I believe", "probably", "I'm not sure",
            "as an AI", "I don't have access"
        ]
        flags_found = sum(1 for flag in red_flags if flag.lower() in output.lower())
        return max(0.0, 1.0 - (flags_found * 0.2))
```

**Good for**: Length checks, format validation, presence of required elements, basic sanity
**Limitations**: Can't assess semantic quality, correctness, or coherence

### 2. Traditional NLP Metrics

Established metrics from the NLP field:

| Metric | What It Measures | Best For |
|--------|-----------------|----------|
| **BLEU** | N-gram overlap with reference | Translation |
| **ROUGE** | Recall-oriented overlap with reference | Summarization |
| **Exact Match** | Binary — does output match reference exactly? | Classification, extraction |
| **F1 Score** | Precision/Recall balance | Entity extraction |
| **Semantic Similarity** | Embedding cosine similarity to reference | General comparison |

```python
# Example: Semantic similarity using embeddings
import numpy as np

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

async def evaluate_semantic_similarity(
    client, output: str, reference: str, model: str = "text-embedding-3-small"
) -> float:
    """Compute semantic similarity between output and reference."""
    response = await client.embeddings.create(
        model=model,
        input=[output, reference]
    )
    vec_output = response.data[0].embedding
    vec_reference = response.data[1].embedding
    return cosine_similarity(vec_output, vec_reference)
```

**Good for**: When you have reference outputs to compare against
**Limitations**: BLEU/ROUGE miss semantic equivalence; embeddings miss fine-grained differences

### 3. LLM-as-Judge

Use another LLM to evaluate the output:

#### Single-Point Scoring

```python
JUDGE_PROMPT = """
You are an expert evaluator. Rate the following output on a scale of 1-5.

## Evaluation Criteria
- **Accuracy** (1-5): Is the information correct?
- **Relevance** (1-5): Does it address the question?
- **Completeness** (1-5): Are all important aspects covered?
- **Clarity** (1-5): Is it well-written and easy to understand?

## Input
{input_text}

## Output to Evaluate
{output_text}

## Reference Answer (if available)
{reference_text}

Rate each criterion and provide brief justification.
Return as JSON:
{{
    "accuracy": {{"score": 1-5, "reasoning": "..."}},
    "relevance": {{"score": 1-5, "reasoning": "..."}},
    "completeness": {{"score": 1-5, "reasoning": "..."}},
    "clarity": {{"score": 1-5, "reasoning": "..."}}
}}
"""
```

#### Pairwise Comparison

```python
COMPARISON_PROMPT = """
Compare two outputs for the same input. Which is better?

## Input
{input_text}

## Output A
{output_a}

## Output B
{output_b}

Which output is better and why? Consider accuracy, relevance, 
completeness, and clarity.

Return as JSON:
{{
    "winner": "A" or "B" or "tie",
    "reasoning": "explanation",
    "confidence": 0.0-1.0
}}
"""
```

#### Aspect-Based Critique

```python
CRITIQUE_PROMPT = """
Analyze the following output for quality issues.

## Task Description
{task_description}

## Output
{output_text}

Identify any issues in these categories:
1. Factual errors or hallucinations
2. Missing information
3. Format/style issues
4. Logical inconsistencies
5. Unnecessary or irrelevant content

Return as JSON:
{{
    "issues": [
        {{"category": "...", "description": "...", "severity": "high|medium|low"}}
    ],
    "overall_quality": "excellent|good|acceptable|poor",
    "improvement_suggestions": ["..."]
}}
"""
```

**Best Practices for LLM-as-Judge**:
- ✅ Use a **stronger model** as judge (e.g., GPT-4o to judge GPT-4o-mini outputs)
- ✅ Provide **clear rubrics** with examples of each score level
- ✅ Use **structured output** for judge responses
- ✅ Run judge **multiple times** and average (reduces noise)
- ❌ Don't use the **same model** to judge its own outputs (bias)
- ❌ Don't use vague criteria like "is it good?" — be specific

### 4. Human Evaluation

The gold standard, but expensive and slow:

- **Expert evaluation**: Domain experts rate outputs
- **A/B testing**: Show users two outputs, ask which is better
- **Annotation**: Labelers mark specific quality attributes

**When to use**: Final validation before production, establishing ground truth, calibrating automated metrics.

---

## Building Test Cases

### What Makes a Good Test Case?

```python
from pydantic import BaseModel
from typing import Optional

class TestCase(BaseModel):
    """A single evaluation test case."""
    id: str
    name: str
    description: str
    
    # Input
    input_text: str
    input_variables: dict = {}  # For template variable injection
    
    # Expected output (optional — not all evaluations need this)
    reference_output: Optional[str] = None
    
    # Evaluation criteria
    required_elements: list[str] = []  # Must be present in output
    forbidden_elements: list[str] = []  # Must NOT be present
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    expected_format: Optional[str] = None  # "json", "bullet_list", etc.
    
    # Metadata
    difficulty: str = "normal"  # easy, normal, hard, edge_case
    category: str = ""
    tags: list[str] = []
```

### Test Case Categories

| Category | Purpose | Examples |
|----------|---------|---------|
| **Happy path** | Normal, expected inputs | Typical documents, standard questions |
| **Edge cases** | Boundary conditions | Empty input, very long input, special characters |
| **Adversarial** | Try to break the system | Prompt injection, conflicting instructions |
| **Ambiguous** | Multiple valid interpretations | Vague questions, unclear context |
| **Multi-language** | Non-English inputs | Common enterprise languages |
| **Format stress** | Test output format compliance | Complex JSON schemas, nested structures |

### Test Set Design Principles

1. **Representative**: Cover the full range of real inputs
2. **Balanced**: Include easy, medium, hard, and edge cases
3. **Versioned**: Test sets evolve alongside prompts
4. **Independent**: Test cases don't depend on each other
5. **Documented**: Each test case has clear pass/fail criteria

---

## Consistency Evaluation

Running the same input multiple times reveals how stable the model is:

```python
async def evaluate_consistency(
    client, prompt: str, n_runs: int = 5, temperature: float = 0.0
) -> dict:
    """Evaluate output consistency across multiple runs."""
    outputs = []
    for _ in range(n_runs):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        outputs.append(response.choices[0].message.content)
    
    # Compute pairwise similarity
    similarities = []
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            sim = await evaluate_semantic_similarity(client, outputs[i], outputs[j])
            similarities.append(sim)
    
    return {
        "mean_similarity": np.mean(similarities),
        "min_similarity": np.min(similarities),
        "std_similarity": np.std(similarities),
        "n_unique_outputs": len(set(outputs)),
        "outputs": outputs
    }
```

---

## Cost Tracking

```python
# Token pricing (as of 2025 — verify current prices)
PRICING = {
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gpt-4.1": {"input": 2.00 / 1_000_000, "output": 8.00 / 1_000_000},
    "gpt-4.1-mini": {"input": 0.40 / 1_000_000, "output": 1.60 / 1_000_000},
    "gpt-4.1-nano": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate API cost for a single call."""
    prices = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * prices["input"]) + (output_tokens * prices["output"])
```

---

## Evaluation Pipeline Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Test Cases  │────▶│  Run Prompts │────▶│  Raw Outputs  │
│  (Dataset)   │     │  (LLM API)   │     │  + Metadata   │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                    ┌─────────────────────────────┤
                    ▼                              ▼
            ┌──────────────┐              ┌──────────────┐
            │  Heuristic   │              │  LLM-as-     │
            │  Evaluators  │              │  Judge       │
            └──────┬───────┘              └──────┬───────┘
                   │                              │
                   ▼                              ▼
            ┌─────────────────────────────────────────┐
            │           Aggregated Scores              │
            │  (per test case, per metric, per run)    │
            └─────────────────┬───────────────────────┘
                              │
                              ▼
            ┌─────────────────────────────────────────┐
            │           Evaluation Report              │
            │  - Summary statistics                    │
            │  - Pass/fail by category                 │
            │  - Cost breakdown                        │
            │  - Comparison with previous versions     │
            └─────────────────────────────────────────┘
```

---

## Best Practices

- ✅ **Layer evaluations**: Use cheap heuristics first, then LLM-as-judge for deeper analysis
- ✅ **Version your test sets** alongside your prompts
- ✅ **Track costs per evaluation run** — judge calls cost money too
- ✅ **Use multiple dimensions** — no single metric captures "quality"
- ✅ **Establish baselines** before making changes
- ✅ **Automate evaluation** — it should run on every prompt change
- ❌ Don't rely only on LLM-as-judge — it has blind spots
- ❌ Don't optimize for a single metric — Goodhart's Law applies
- ❌ Don't skip edge cases — they reveal the most about robustness

---

## Application to My Project

### How I'll Use This

1. **Multi-layer evaluator**: Heuristics → NLP metrics → LLM-as-judge
2. **Test case management**: YAML/JSON files with structured test cases
3. **Automated reporting**: Per-run reports with cost tracking
4. **A/B comparison**: Side-by-side evaluation of prompt versions

### Decisions to Make

- [ ] Which LLM to use as judge? (GPT-4o is reliable but expensive)
- [ ] How many runs for consistency evaluation? (3-5 is typical)
- [ ] Score aggregation strategy (mean, weighted, pass/fail threshold?)
- [ ] How to handle the cost of evaluation itself

---

## Resources for Deeper Learning

- [Ragas Documentation](https://docs.ragas.io/) — Comprehensive evaluation framework with many metrics
- [DeepEval](https://docs.confident-ai.com/) — Python evaluation framework with LLM-as-judge
- [Promptfoo](https://promptfoo.dev/) — CLI tool for prompt evaluation and red-teaming
- [OpenAI Evals](https://github.com/openai/evals) — OpenAI's evaluation framework
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) — Research on LLM comparison methodology
