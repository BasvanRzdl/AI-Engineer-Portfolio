---
date: 2026-02-27
type: concept
topic: "Task Decomposition & Delegation"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Learning: Task Decomposition & Delegation

## In My Own Words

Task decomposition is the process of breaking down a complex, ambiguous goal into smaller, well-defined subtasks that individual agents can handle. Delegation is the process of assigning those subtasks to the right agent.

This is arguably the most critical capability in a multi-agent system. If you decompose poorly, agents get confused tasks. If you delegate poorly, the wrong agent gets the wrong job. Both waste time and tokens.

Think of it like a project manager receiving a vague request: "Research AI trends and write a report." A good PM breaks this into: (1) identify key AI trends, (2) gather data on each trend, (3) analyze implications, (4) draft the report, (5) review and refine. Then they assign each subtask to the right team member.

## Why This Matters

- LLMs work best on focused, well-scoped tasks — not giant ambiguous ones
- Decomposition quality directly determines output quality
- Poor decomposition is the #1 cause of multi-agent system failures
- The orchestrator's decomposition strategy is the "brain" of the whole system
- Your Strategy Research Team needs to decompose "research topic X" into concrete subtasks

## Decomposition Strategies

### 1. Static Decomposition (Pre-defined)

The task breakdown is hardcoded into the workflow. You know the steps in advance.

```
"Write a strategy report on AI adoption" →
  1. Research current AI adoption trends
  2. Research competitor AI strategies  
  3. Analyze market implications
  4. Draft executive summary
  5. Draft detailed findings
  6. Review and refine
```

**When to use:** Tasks follow a predictable structure (reports, analyses, reviews)
**Pros:** Deterministic, fast, no LLM call needed for planning
**Cons:** Rigid, can't adapt to unexpected task shapes

### 2. Dynamic Decomposition (LLM-Planned)

An orchestrator agent uses an LLM to analyze the task and generate a decomposition plan at runtime.

```python
# Orchestrator prompt
"""
You are a research project planner. Given a research topic, 
break it down into specific research subtasks.

Topic: {topic}

Output a JSON plan:
{
  "subtasks": [
    {"id": 1, "description": "...", "agent": "researcher", "dependencies": []},
    {"id": 2, "description": "...", "agent": "researcher", "dependencies": []},
    {"id": 3, "description": "...", "agent": "analyst", "dependencies": [1, 2]},
    ...
  ]
}
"""
```

**When to use:** Tasks vary significantly, can't predict structure ahead of time
**Pros:** Flexible, adapts to any task, can create novel plans
**Cons:** LLM might create poor plans, adds latency and cost, non-deterministic

### 3. Hybrid Decomposition (Templated + Dynamic)

Define a template with fixed phases but let the LLM fill in the specifics.

```
Template (fixed):
  Phase 1: Research (N subtasks - LLM decides what to research)
  Phase 2: Analysis (1 task - always the same structure)
  Phase 3: Writing (1 task - always the same structure)
  Phase 4: Review (iterative - always the same structure)

Dynamic parts:
  - How many research subtasks?
  - What specific topics to research?
  - What angle for analysis?
```

**When to use:** You have a known process but variable inputs
**Pros:** Best of both worlds — structure + flexibility
**Cons:** Template might not fit all cases

**This is the recommended approach for your Strategy Research Team.**

## Delegation Strategies

### 1. Skill-Based Routing

Match subtasks to agents based on their declared capabilities.

```python
agents = {
    "researcher": {
        "skills": ["web_search", "document_analysis", "data_gathering"],
        "description": "Expert at finding and gathering information"
    },
    "analyst": {
        "skills": ["data_analysis", "trend_identification", "swot_analysis"],
        "description": "Expert at analyzing data and identifying patterns"
    },
    "writer": {
        "skills": ["report_writing", "summarization", "formatting"],
        "description": "Expert at producing clear, structured written content"
    },
    "critic": {
        "skills": ["quality_review", "fact_checking", "feedback"],
        "description": "Expert at evaluating and improving written content"
    }
}
```

### 2. LLM-Based Selection (AutoGen SelectorGroupChat)

An LLM reads agent descriptions and the current context, then picks the best agent for the next step.

```python
# AutoGen SelectorGroupChat internally does:
"""
Given these agents and their descriptions:
- {agent1.name}: {agent1.description}
- {agent2.name}: {agent2.description}
- ...

And the current conversation:
{messages}

Which agent should speak next? Select from: {agent_names}
"""
```

### 3. Conditional Routing (LangGraph)

Use conditional edges in the graph to route based on state.

```python
def route_after_research(state):
    if state["research_results"]:
        return "analyst"
    elif state["research_retries"] < 3:
        return "researcher"  # retry
    else:
        return "error_handler"

graph.add_conditional_edges("researcher", route_after_research)
```

### 4. Self-Delegation (Swarm/Handoff)

Agents decide for themselves who should go next.

```python
# Agent's tools include handoff functions
@agent.tool
def transfer_to_analyst():
    """Transfer to the analyst when research is complete."""
    return HandoffMessage(target="analyst")
```

## Dependency Management

Subtasks often have dependencies — Task 3 can't start until Tasks 1 and 2 are done.

### Dependency Graph

```
Task 1 (Research: Market Trends) ──┐
                                    ├── Task 3 (Analyze) → Task 4 (Write) → Task 5 (Review)
Task 2 (Research: Competitors) ────┘
```

### Implementation Approaches

| Approach | How | Framework |
|----------|-----|-----------|
| **Graph topology** | Define edges so dependent nodes run after predecessors | LangGraph edges |
| **State conditions** | Check if prerequisite data exists in state | LangGraph conditional edges |
| **Explicit ordering** | Define task order upfront | AutoGen RoundRobinGroupChat |
| **Dynamic planning** | Orchestrator tracks dependency graph at runtime | Custom orchestrator logic |

### LangGraph Implementation with Send API

The `Send` API enables dynamic fan-out — the orchestrator decides at runtime how many parallel tasks to create:

```python
from langgraph.types import Send

def orchestrator(state):
    """Decompose research topic into subtasks and fan out."""
    plan = llm.invoke(f"Break down research on '{state['topic']}' into 3 subtasks")
    
    # Dynamically create parallel research tasks
    return [
        Send("researcher", {"subtask": task, "parent_topic": state["topic"]})
        for task in plan.subtasks
    ]
```

## Error Handling in Decomposition

What happens when things go wrong?

### Retry Strategies

```python
def route_with_retry(state):
    if state["error"] and state["retries"] < 3:
        return "retry_same_agent"
    elif state["error"]:
        return "fallback_agent"
    else:
        return "next_agent"
```

### Fallback Strategies

| Strategy | When | How |
|----------|------|-----|
| **Retry same agent** | Transient error (API timeout) | Re-run with same input |
| **Retry with modification** | Bad output | Re-run with additional instructions |
| **Skip and continue** | Non-critical subtask failed | Mark as failed, proceed with what we have |
| **Escalate to human** | Critical failure | Interrupt and ask human to intervene |
| **Graceful degradation** | Multiple failures | Produce partial output with disclaimers |

## Application to My Project

### Task Decomposition Flow

```
User Input: "Research AI adoption strategies for enterprise"
                    |
            [Orchestrator]
                    |
        ┌── Decompose into research subtasks ──┐
        |                                       |
  [1] "Current AI adoption     [2] "Enterprise AI    [3] "Risk factors
       rates and trends"            case studies"          and challenges"
        |                           |                      |
        └───── [Research Agents - Parallel] ──────────────┘
                        |
                [Synthesize Results]
                        |
                [Analyze Patterns & Implications]
                        |
                [Draft Strategy Report]
                        |
                [Review & Refine]
                        |
                [Final Output]
```

### Orchestrator Design

The orchestrator should:
1. Parse the user's request to understand the topic and scope
2. Generate 2-4 focused research subtasks (not too many, not too few)
3. Assign each subtask to a Research agent via `Send` (parallel)
4. After all research completes, route to Analysis
5. Track progress and handle failures

### Key Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Decomposition type | Hybrid (template + dynamic) | Fixed phases, dynamic research subtasks |
| Number of research subtasks | 2-4 per topic | Balance depth vs cost |
| Delegation method | Graph topology + conditional edges | LangGraph's native strength |
| Error handling | Retry once, then skip with note | Don't block on one failed subtask |
| Max parallel tasks | 3-5 | API rate limits + cost control |

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Over-decomposition | Too many tiny subtasks | Aim for 3-5 meaningful subtasks, not 15 micro-tasks |
| Vague subtask descriptions | Orchestrator prompt too generic | Include examples in the orchestrator prompt |
| Missing dependencies | Didn't model task dependencies | Draw the dependency graph before coding |
| Circular dependencies | Task A needs B, B needs A | Detect cycles in your dependency graph |
| Ignoring failures | No retry or fallback logic | Always handle the error case for each subtask |
| Token waste | Each subtask includes full original context | Only pass relevant context to each subtask |

## Resources for Deeper Learning

- [LangGraph Send API](https://docs.langchain.com/oss/python/langgraph/concepts/low_level/#send) — Dynamic task fan-out
- [LangGraph Orchestrator-Worker Pattern](https://docs.langchain.com/oss/python/langgraph/workflows-agents#orchestrator-worker) — Complete pattern with code
- [AutoGen SelectorGroupChat](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/selector-group-chat.html) — LLM-based task delegation
- [Google ADK Hierarchical Task Decomposition](https://google.github.io/adk-docs/agents/multi-agents/#4-hierarchical-task-decomposition) — ADK pattern

## Questions Remaining

- [ ] What's the ideal prompt structure for the orchestrator's decomposition step?
- [ ] How to validate that a decomposition plan is good before executing it?
- [ ] Should the orchestrator track a task board or is graph structure sufficient?
