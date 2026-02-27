---
date: 2026-02-27
type: concept
topic: "Observability & Debugging Multi-Agent Systems"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Learning: Observability & Debugging Multi-Agent Systems

## In My Own Words

Observability is the ability to understand what's happening inside your multi-agent system from the outside — by looking at logs, traces, metrics, and state snapshots. Debugging is using that observability to find and fix problems.

Multi-agent systems are inherently harder to debug than single-agent systems because:
- Multiple agents interact in complex ways
- The flow is often non-deterministic (LLM decisions vary)
- Problems cascade — one bad agent output corrupts downstream agents
- Context windows are large and hard to inspect manually

Without good observability, your system is a black box. When something goes wrong (and it will), you need to answer: **Which agent? What input did it get? What did it produce? Why was it wrong?**

## Why This Matters

- Multi-agent systems have many failure modes that are hard to reproduce
- LLM outputs are non-deterministic — the same input might produce different results
- Token costs can spiral without visibility into what's consuming them
- Debugging by re-running the whole pipeline is expensive and slow
- Your Strategy Research Team needs cost tracking and failure handling (per the README)

## The Three Pillars of Observability

### 1. Logging (What Happened)

Record events as they occur. The foundation of observability.

**What to log for multi-agent systems:**

```python
# Per-agent call
{
    "timestamp": "2025-01-15T10:30:00Z",
    "agent": "researcher",
    "node": "research_node",
    "action": "llm_call",
    "input_tokens": 1500,
    "output_tokens": 800,
    "model": "gpt-4o-mini",
    "latency_ms": 2340,
    "status": "success",
    "input_summary": "Research AI adoption in healthcare",
    "output_summary": "Found 5 key trends...",
    "cost_usd": 0.0023
}
```

**Logging levels for different needs:**

| Level | What | When |
|-------|------|------|
| ERROR | Agent failures, API errors, unexpected exceptions | Always |
| WARN | Retries, fallbacks, quality below threshold | Always |
| INFO | Agent starts/completes, routing decisions, key state changes | Production |
| DEBUG | Full prompts, complete responses, state snapshots | Development |
| TRACE | Token-by-token streaming, internal framework events | Deep debugging |

### 2. Tracing (How It Happened)

Traces show the end-to-end flow of a request through the system. Each agent invocation is a "span" within the trace.

```
Trace: "Research AI Adoption" (total: 45s, $0.15)
├── Orchestrator (3s, $0.02)
│   └── Planned 3 research subtasks
├── Researcher[1] (8s, $0.01) ─── parallel
├── Researcher[2] (12s, $0.01) ── parallel  
├── Researcher[3] (10s, $0.01) ── parallel
├── Synthesizer (5s, $0.02)
├── Analyst (6s, $0.03)
├── Writer (8s, $0.03)
├── Critic (3s, $0.01) → REVISE
├── Writer (7s, $0.03) ← revision
├── Critic (2s, $0.01) → APPROVE
└── Total: 45s, $0.15, 12,000 tokens
```

**Framework support:**

- **LangGraph:** Native integration with LangSmith for tracing. Every node execution becomes a span.
- **AutoGen:** Built-in event logging with `ConsoleLogHandler`, `FileLogHandler`. Structured events for agent messages, tool calls, and termination.
- **Google ADK:** Callbacks system for event observation. Built-in evaluation framework.

### 3. Metrics (How Well It's Working)

Aggregate measurements over time to track system health and performance.

**Key metrics for multi-agent systems:**

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| Total cost per task | Token usage × model pricing | < $0.50 per research report |
| End-to-end latency | Wall clock time from request to output | < 2 minutes |
| Agent utilization | % of time each agent is active | Balanced across agents |
| Iteration count | How many Writer ↔ Critic loops | Average < 3 |
| Quality score | Critic's rating of final output | Average > 7/10 |
| Error rate | % of tasks that fail | < 5% |
| Token efficiency | Useful output tokens / total tokens | > 30% |

## Debugging Strategies

### 1. State Inspection

The most powerful debugging technique: look at the state at each step.

```python
# LangGraph: Print state after each node
for event in graph.stream(input, stream_mode="debug"):
    print(f"Node: {event['node']}")
    print(f"State: {json.dumps(event['state'], indent=2)}")
```

### 2. Replay / Time Travel

With checkpointing enabled, you can replay from any point in the workflow.

```python
# LangGraph: Resume from a specific checkpoint
config = {"configurable": {"thread_id": "abc123"}}

# Get all checkpoints
checkpoints = list(checkpointer.list(config))

# Resume from a specific state
graph.invoke(None, config={"configurable": {
    "thread_id": "abc123",
    "checkpoint_id": checkpoints[2].checkpoint_id
}})
```

**Why this is powerful:**
- Fix a bug, then replay from just before the bug — don't re-run everything
- Compare outputs at the same checkpoint with different prompts
- Step through the workflow one node at a time

### 3. Prompt Debugging

When an agent produces bad output, the prompt is usually the culprit.

**Debugging checklist:**
1. ✅ What was the full system prompt? (Log it at DEBUG level)
2. ✅ What was the actual input (user message + state)?
3. ✅ What did the LLM actually return?
4. ✅ Was the output parsed correctly?
5. ✅ Was the context too long? (Check token counts)

### 4. Deterministic Testing

Make non-deterministic systems testable by controlling randomness.

```python
# Force deterministic outputs for testing
test_config = {
    "model": "gpt-4o",
    "temperature": 0,        # Deterministic output
    "seed": 42,              # Reproducible (OpenAI)
}

# Or mock the LLM entirely
class MockLLM:
    def invoke(self, prompt):
        return "Mocked research findings about AI adoption..."
```

### 5. Isolated Agent Testing

Test each agent independently before testing the full pipeline.

```python
# Test the Analyst agent in isolation
def test_analyst():
    test_input = {
        "research_results": [
            "Finding 1: AI adoption grew 40% in 2024",
            "Finding 2: Healthcare leads in AI implementation",
        ]
    }
    result = analyst_node(test_input)
    
    assert "patterns" in result
    assert "implications" in result
    assert len(result["patterns"]) > 0
```

## Cost Tracking Implementation

### Token Counting

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Track per-agent costs
class CostTracker:
    PRICING = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    }
    
    def __init__(self):
        self.total_cost = 0
        self.agent_costs = {}
    
    def track(self, agent: str, model: str, input_tokens: int, output_tokens: int):
        cost = (input_tokens * self.PRICING[model]["input"] + 
                output_tokens * self.PRICING[model]["output"])
        self.total_cost += cost
        self.agent_costs[agent] = self.agent_costs.get(agent, 0) + cost
        return cost
```

### Budget Guardrails

```python
def check_budget(state):
    """Abort if we're spending too much."""
    if state["total_cost"] > state["budget_limit"]:
        return "budget_exceeded"  # Route to graceful termination
    return "continue"
```

## Framework-Specific Observability

### LangGraph + LangSmith

LangSmith is LangChain's observability platform. It provides:
- Automatic tracing of all LangGraph node executions
- Token usage and cost tracking per node
- Visual workflow execution graphs
- Run comparison and regression testing
- Feedback collection for evaluation

```python
# Enable LangSmith tracing
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "strategy-research-team"

# Every graph execution is automatically traced
```

### AutoGen Logging

AutoGen has built-in structured logging:

```python
from autogen_agentchat.ui import Console

# Stream all agent messages to console
async for message in team.run_stream(task="Research AI adoption"):
    print(message)

# Or use the Console helper for formatted output
await Console(team.run_stream(task="Research AI adoption"))
```

AutoGen also supports custom log handlers for structured event logging.

### Google ADK Observability

ADK provides callbacks for event observation:
- Before/after agent execution callbacks
- Tool call observation
- Built-in evaluation framework for quality measurement

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| No logging in production | "We'll add it later" | Build logging from day one |
| Logging too much | Full prompts/responses in production | Use log levels, DEBUG for dev only |
| No cost tracking | Forget to count tokens | Implement CostTracker from the start |
| No replay capability | No checkpointing | Enable checkpointing by default |
| Testing only the happy path | Only test when everything works | Test error cases, retries, timeouts |
| Ignoring latency | Focus on quality only | Track and optimize end-to-end time |

## Application to My Project

### Observability Strategy

1. **Development:** Full DEBUG logging + LangSmith tracing + state inspection
2. **Testing:** Deterministic mode (temp=0) + isolated agent tests + replay tests
3. **Production:** INFO logging + cost tracking + quality metrics + error alerting

### Must-Have Monitoring

```python
# Add to your graph's state
class ResearchState(TypedDict):
    # ... existing fields ...
    
    # Observability fields
    trace_id: str                    # Unique ID for this run
    agent_logs: list[dict]           # Log entries per agent
    total_tokens: int                # Running token count
    total_cost: float                # Running cost
    started_at: str                  # Timestamp
    node_timings: dict[str, float]   # Time per node
```

### Quality Dashboard (Aspirational)

Track over time:
- Average quality score per topic type
- Cost distribution across agents
- Iteration count trends (are we getting better at first drafts?)
- Error rates per agent (which agent fails most?)

## Resources for Deeper Learning

- [LangSmith Documentation](https://docs.smith.langchain.com/) — Tracing, evaluation, and monitoring
- [AutoGen Logging Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/logging.html) — Built-in logging in AutoGen
- [OpenTelemetry for LLMs](https://opentelemetry.io/) — Open standard for observability
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents) — Debugging advice

## Questions Remaining

- [ ] Is LangSmith worth the cost for a learning project, or can local logging suffice?
- [ ] How to implement automatic quality regression testing?
- [ ] What's the best way to visualize multi-agent traces locally?
