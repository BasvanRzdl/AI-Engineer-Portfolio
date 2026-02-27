---
date: 2026-02-27
type: concept
topic: "Quality Assurance & Evaluation for Multi-Agent Systems"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Learning: Quality Assurance & Evaluation for Multi-Agent Systems

## In My Own Words

Quality evaluation in multi-agent systems answers the question: **"Is the output actually good?"** This is harder than it sounds because:

1. "Good" is subjective and context-dependent
2. Multiple agents contribute to the final output — which one made it good or bad?
3. LLM outputs are non-deterministic — quality varies between runs
4. You need both automated evaluation (scalable) and human evaluation (ground truth)

The key insight: **build evaluation into the system itself** (the Critic agent) AND **build evaluation around the system** (test suites, benchmarks, regression testing). The Critic is your in-loop quality gate; external evaluation is your out-of-loop quality assurance.

## Why This Matters

- Without evaluation, you can't know if changes improve or degrade quality
- The Critic agent needs well-defined criteria to be useful
- Your README requires quality evaluation as a core feature
- Evaluation drives iterative improvement of prompts and agent designs
- Cost-quality tradeoffs can only be made when you can measure quality

## Types of Evaluation

### 1. In-Loop Evaluation (The Critic Agent)

The Critic agent runs as part of the workflow, evaluating outputs and deciding whether to approve or request revision. This is real-time quality control.

**How it works in the Strategy Research Team:**

```
Writer produces draft → Critic evaluates → APPROVE or REVISE
                                              ↓
                                    Specific feedback to Writer
                                              ↓
                                    Writer revises → Critic re-evaluates
                                              ↓
                                    ... (max 3 iterations)
```

**Critic Evaluation Rubric:**

| Criterion | Weight | 1 (Poor) | 5 (Average) | 10 (Excellent) |
|-----------|--------|----------|-------------|-----------------|
| **Accuracy** | 25% | Contains factual errors | Mostly accurate, minor issues | All claims supported by evidence |
| **Completeness** | 20% | Missing major topics | Covers main points, some gaps | Comprehensive coverage |
| **Clarity** | 20% | Confusing, poorly structured | Readable but could be clearer | Crystal clear, well-organized |
| **Actionability** | 20% | Vague recommendations | Some actionable items | Specific, prioritized recommendations |
| **Depth** | 15% | Surface-level only | Adequate analysis | Deep insights with nuance |

**Approval threshold:** Weighted score ≥ 7.0/10 → APPROVE

### 2. Out-of-Loop Evaluation (Test Suites)

Run after development to validate system quality. Not part of the live workflow.

**Types:**

| Evaluation Type | What It Tests | How |
|-----------------|---------------|-----|
| **Unit tests** | Individual agent outputs | Fixed inputs, check output structure and content |
| **Integration tests** | Full pipeline end-to-end | Run complete workflow, check final output |
| **Regression tests** | Quality doesn't degrade after changes | Compare output quality before/after changes |
| **Benchmark tests** | Performance on known-good tasks | Run on curated set of topics, compare scores |

### 3. LLM-as-Judge

Use an LLM to evaluate the quality of another LLM's output. This is the most practical automated evaluation method for subjective quality.

```python
JUDGE_PROMPT = """
You are an expert evaluator. Rate the following strategy report 
on a scale of 1-10 for each criterion.

Report to evaluate:
{report}

Original research request:
{request}

Research data available:
{research_data}

Rate on these criteria:
1. Accuracy (claims supported by evidence): _/10
2. Completeness (all aspects covered): _/10
3. Clarity (well-written, well-structured): _/10
4. Actionability (specific recommendations): _/10
5. Depth (insightful analysis): _/10

For each score, provide a brief justification.
Output as JSON.
"""
```

**Caution:** LLM-as-Judge has known biases:
- Position bias (prefers first option in comparisons)
- Verbosity bias (prefers longer outputs)
- Self-enhancement bias (prefers its own model's outputs)
- Style over substance (prefers well-formatted but shallow content)

Mitigate by:
- Using a different model as judge than as generator
- Providing clear rubrics with examples
- Running multiple evaluations and averaging
- Validating against human judgments periodically

## Evaluation Metrics

### Quality Metrics

| Metric | What It Measures | How to Calculate |
|--------|-----------------|-----------------|
| **Rubric Score** | Overall quality on defined criteria | Weighted average of criterion scores |
| **Factual Accuracy** | % of claims that are verifiable | Fact-check against source material |
| **Completeness** | % of required topics covered | Checklist comparison |
| **Citation Rate** | % of claims with sources | Count citations vs. claims |
| **Readability** | How easy to read | Flesch-Kincaid or similar |
| **Revision Rate** | How often Critic requests revision | Count iterations before approval |

### Efficiency Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Tokens per quality point** | Cost efficiency | Minimize |
| **Time to approval** | How fast we produce quality output | < 2 min |
| **Iteration efficiency** | Quality improvement per revision | > 1 point/iteration |
| **First-draft quality** | How good the Writer's initial output is | > 5/10 (improves over time) |

### System-Level Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Success rate** | % of tasks that complete successfully | > 95% |
| **Average quality** | Mean rubric score across tasks | > 7/10 |
| **Quality variance** | Consistency of output quality | Low variance (σ < 1.5) |
| **Cost per task** | Average cost to complete one research task | Track and optimize |

## Building the Evaluation Pipeline

### Step 1: Define Gold Standard Examples

Create a small set (5-10) of "perfect" research reports on known topics. These serve as your ground truth.

```python
gold_standards = [
    {
        "topic": "AI adoption in healthcare",
        "expected_sections": ["market size", "key players", "challenges", "trends"],
        "key_facts": ["market expected to reach $X by 2030", "..."],
        "quality_score": 9.5,  # Human-rated
        "reference_report": "path/to/gold_standard.md"
    },
    # ... more examples
]
```

### Step 2: Automated Test Suite

```python
import pytest

class TestStrategyResearchTeam:
    
    def test_researcher_produces_findings(self):
        """Researcher should return structured findings."""
        result = researcher_node({"subtask": "AI adoption trends 2024"})
        assert "findings" in result
        assert len(result["findings"]) >= 3
        assert all("source" in f for f in result["findings"])
    
    def test_analyst_identifies_patterns(self):
        """Analyst should identify patterns from research."""
        result = analyst_node({"research_results": sample_research})
        assert "patterns" in result
        assert "implications" in result
        assert len(result["patterns"]) >= 2
    
    def test_writer_produces_structured_report(self):
        """Writer should produce a well-structured report."""
        result = writer_node({"analysis": sample_analysis})
        assert "executive_summary" in result["draft"].lower()
        assert "recommendations" in result["draft"].lower()
    
    def test_critic_provides_actionable_feedback(self):
        """Critic should score and provide specific feedback."""
        result = critic_node({"draft": sample_draft})
        assert result["decision"] in ["APPROVE", "REVISE"]
        assert "scores" in result
        assert all(1 <= s <= 10 for s in result["scores"].values())
    
    def test_full_pipeline_produces_quality_output(self):
        """End-to-end test: full pipeline produces approved output."""
        result = graph.invoke({"topic": "AI adoption in retail"})
        assert result["status"] == "complete"
        assert result["quality_score"] >= 7.0
        assert result["total_cost"] < 1.0  # Under $1
    
    def test_iteration_improves_quality(self):
        """Quality should improve after each Writer-Critic iteration."""
        # Track quality scores across iterations
        scores = result["iteration_scores"]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i-1] - 0.5  # Allow small regression
```

### Step 3: Regression Testing

After every prompt change, run the benchmark suite and compare:

```python
def regression_test(old_prompts, new_prompts, test_topics):
    old_results = [run_pipeline(topic, old_prompts) for topic in test_topics]
    new_results = [run_pipeline(topic, new_prompts) for topic in test_topics]
    
    old_avg = mean([r["quality_score"] for r in old_results])
    new_avg = mean([r["quality_score"] for r in new_results])
    
    print(f"Old average: {old_avg:.2f}")
    print(f"New average: {new_avg:.2f}")
    print(f"Change: {new_avg - old_avg:+.2f}")
    
    assert new_avg >= old_avg - 0.5, "Quality regression detected!"
```

## The Evaluator-Optimizer Pattern (Deep Dive)

This is the formal name for the Writer ↔ Critic loop. It's one of the most powerful patterns for quality.

### How It Works

```python
# LangGraph implementation
def should_continue(state):
    if state["critic_decision"] == "APPROVE":
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"  # Approve with caveats
    return "writer"   # Another round

graph.add_conditional_edges("critic", should_continue, {
    "writer": "writer",
    "end": END
})
```

### Optimal Iteration Count

Research suggests:
- **Iteration 1 → 2:** Largest quality improvement (~2-3 points)
- **Iteration 2 → 3:** Moderate improvement (~1-2 points)
- **Iteration 3+:** Diminishing returns (<0.5 points)
- **Recommendation:** Set `max_iterations = 3` and track actual usage

### Feedback Quality Matters

The quality of the Critic's feedback directly determines how much the Writer improves. Vague feedback leads to random changes; specific feedback leads to targeted improvements.

**Bad feedback:** "The analysis needs to be better."
**Good feedback:** "Section 3 claims AI adoption grew 40% but provides no source. Add a citation or remove the specific number. The recommendations section lists 5 items but doesn't prioritize them — add priority rankings."

## Application to My Project

### Evaluation Architecture

```
                    [Full Pipeline]
                         |
              ┌──── In-Loop ────┐
              |                  |
          [Writer] ←→ [Critic]  |    ← Real-time quality gate
              |                  |
              └──────────────────┘
                         |
              ┌──── Out-of-Loop ─┐
              |                   |
          [LLM Judge]  [Test Suite] ← Offline validation
              |                   |
          [Benchmarks]  [Regression] ← Continuous improvement
              └───────────────────┘
```

### Implementation Priority

1. **Phase 1:** Critic agent with rubric-based evaluation (in-loop)
2. **Phase 2:** Unit tests for each agent (out-of-loop)
3. **Phase 3:** LLM-as-Judge for automated quality scoring
4. **Phase 4:** Regression testing for prompt changes
5. **Phase 5:** Quality dashboard and tracking over time

### Critic Agent Configuration

```python
QUALITY_THRESHOLDS = {
    "approval_score": 7.0,      # Minimum weighted score to approve
    "max_iterations": 3,         # Maximum revision cycles
    "max_issues_per_review": 5,  # Focus on top issues
    "escalation_threshold": 4.0, # Below this, escalate to human
}
```

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Rubber-stamp Critic | Critic approves everything | Make approval criteria specific and measurable |
| Overcritical Critic | Critic never approves | Set reasonable thresholds, cap iterations |
| Vague feedback | Critic says "needs improvement" | Require specific, located, actionable feedback |
| Infinite loops | No iteration cap | Always set max_iterations |
| Evaluating style over substance | LLM biases toward well-formatted text | Include substance checks in rubric |
| No baseline | Can't tell if quality improved | Establish gold standards before optimizing |

## Resources for Deeper Learning

- [LangSmith Evaluation](https://docs.smith.langchain.com/evaluation) — LLM evaluation framework
- [Ragas](https://docs.ragas.io/) — Evaluation framework for RAG systems (useful patterns)
- [Anthropic: Evaluations](https://docs.anthropic.com/en/docs/build-with-claude/develop-tests) — Guide to building LLM evaluations
- [Google ADK Evaluation](https://google.github.io/adk-docs/evaluate/) — ADK's built-in evaluation framework
- [DeepEval](https://docs.confident-ai.com/) — Open-source LLM evaluation framework

## Questions Remaining

- [ ] How to calibrate the Critic so it's neither too strict nor too lenient?
- [ ] What's the best way to collect human feedback to validate LLM-as-Judge?
- [ ] How to evaluate the *process* quality (not just final output)?
