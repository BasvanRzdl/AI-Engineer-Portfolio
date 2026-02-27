---
date: 2026-02-27
type: concept
topic: "Agent Communication & Coordination"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Learning: Agent Communication & Coordination

## In My Own Words

If multi-agent architecture patterns define the *structure* of your team, communication and coordination define *how the team actually talks and works together*. This is the glue that makes multi-agent systems functional.

There are fundamentally two challenges:
1. **Communication** — How do agents exchange information? (messages, shared state, events)
2. **Coordination** — How do agents decide who does what and when? (turn-taking, handoffs, arbitration)

Getting this wrong leads to agents talking past each other, doing duplicate work, or entering deadlocks. Getting it right means smooth, efficient collaboration.

## Why This Matters

- Agents are only as good as the information they receive
- Poor coordination leads to wasted LLM calls (= wasted money)
- The communication protocol directly impacts system reliability
- Different frameworks handle this very differently — you need to pick the right approach
- Your Strategy Research Team needs clear handoffs between Research → Analysis → Writing → Review

## Communication Mechanisms

### 1. Message Passing (Direct)

**How it works:** Agents send messages directly to other agents. Each message contains content, sender info, and sometimes metadata. This is the most explicit form of communication.

**Framework implementations:**
- **AutoGen Core:** Event-driven message passing with typed messages via publish/subscribe
- **AutoGen AgentChat:** `ChatMessage`, `TextMessage`, `HandoffMessage` passed through team
- **LangGraph:** Messages stored in `MessagesState` — agents read and append to message list

**Pros:** Explicit, traceable, easy to log and debug
**Cons:** Can create coupling between agents, verbose for simple data sharing

### 2. Shared State

**How it works:** All agents read from and write to a shared state object. Instead of sending messages to each other, they update state that others can read. Like a shared whiteboard.

**Framework implementations:**
- **LangGraph:** The core mechanism. A `TypedDict` or Pydantic model is shared across all nodes. Agents read state, process, and return updates. Reducers handle concurrent updates.
- **Google ADK:** `session.state` is the shared state object. Agents read/write using `output_key` to store results. State keys can be scoped (app-level, user-level, session-level).

**Pros:** Simple, decoupled — agents don't need to know about each other
**Cons:** Race conditions with parallel writes, state can grow large, harder to trace who changed what

### 3. Event-Driven (Publish/Subscribe)

**How it works:** Agents publish events to topics. Other agents subscribe to topics they care about. Loose coupling — publishers don't know who's listening, subscribers don't know who's publishing.

**Framework implementations:**
- **AutoGen Core:** The primary model. Agents subscribe to message types (topics). The runtime routes messages. Supports both local and distributed runtimes.

**Pros:** Very scalable, loose coupling, natural for distributed systems
**Cons:** Harder to reason about flow, ordering can be tricky, debugging requires event tracing

### 4. Tool-Based Invocation

**How it works:** One agent calls another agent as if it were a tool. The calling agent sends a request, the called agent processes it, and returns a result. Like a function call between agents.

**Framework implementations:**
- **Google ADK:** `AgentTool` wraps an agent so it can be called as a tool by another agent. The caller explicitly invokes the sub-agent and gets a response.
- **LangGraph:** Subgraphs can be invoked as nodes within a parent graph, acting like tool calls.

**Pros:** Explicit control flow, easy to understand, natural for hierarchical patterns
**Cons:** Tight coupling, synchronous (blocks until the sub-agent responds)

### 5. LLM-Driven Delegation

**How it works:** The LLM itself decides which agent to delegate to, typically by calling a special function like `transfer_to_agent("agent_name")`. The framework intercepts this and routes accordingly.

**Framework implementations:**
- **Google ADK:** Agents can delegate using `transfer_to_agent` — the LLM decides based on sub-agent descriptions
- **AutoGen Swarm:** Agents use `HandoffMessage` to transfer control based on their judgment

**Pros:** Flexible, can handle unexpected situations, leverages LLM reasoning
**Cons:** Non-deterministic, LLM might route incorrectly, harder to guarantee behavior

## Coordination Strategies

### Turn-Taking

Agents take turns in a defined or dynamic order. Only one agent "speaks" at a time.

| Strategy | How It Works | Framework |
|----------|-------------|-----------|
| **Round-Robin** | Fixed cyclic order: A → B → C → A → ... | AutoGen `RoundRobinGroupChat` |
| **Model-Selected** | An LLM picks the next speaker based on context | AutoGen `SelectorGroupChat` |
| **Condition-Based** | Routing edges determine next agent based on state | LangGraph conditional edges |

### Handoff Protocols

One agent explicitly passes control to another.

- **AutoGen Swarm:** Agent calls a `HandoffMessage(target="agent_name")` to transfer control
- **Google ADK:** Agent calls `transfer_to_agent("agent_name")` 
- **LangGraph:** `Command(goto="node_name")` routes to the next agent

### Termination Conditions

How does the team know when to stop?

| Condition | Description | Example |
|-----------|-------------|---------|
| **Max Turns** | Stop after N conversation turns | `MaxMessageTermination(10)` in AutoGen |
| **Token Limit** | Stop when token budget is exhausted | Custom condition |
| **Quality Gate** | Stop when output meets criteria | Critic says "APPROVED" |
| **Text Match** | Stop when specific text appears | `TextMentionTermination("TERMINATE")` |
| **Consensus** | Stop when agents agree | All agents output "DONE" |
| **Timeout** | Stop after time limit | Process-level timeout |

For the Strategy Research Team, combine **quality gate** (Critic approval) with **max turns** (safety net):
```
termination = quality_check | MaxMessageTermination(15)
```

## State Management

### What Goes in State?

For a multi-agent system, the shared state typically includes:

```
State = {
    messages: list[Message]         # Conversation history
    current_task: str               # What we're working on
    research_results: dict          # Output from research phase
    analysis: str                   # Output from analysis phase  
    draft: str                      # Current writing draft
    feedback: list[str]             # Critic's feedback history
    iteration_count: int            # How many refinement loops
    status: str                     # Current phase/status
    cost_tracker: dict              # Token/cost tracking
}
```

### State Reducers (LangGraph Concept)

When multiple agents update the same state key, reducers define how to merge the updates:

- **Replace:** Latest value wins (default) — good for `current_task`, `status`
- **Append:** Add to list — good for `messages`, `feedback`
- **Custom:** Your own logic — good for `cost_tracker` (sum up costs)

### Checkpointing

Saving state at key points so you can:
- Resume after failures
- Enable human-in-the-loop (pause, inspect, modify, resume)
- Audit the decision trail

LangGraph has built-in checkpointing with `MemorySaver` (in-memory), `SqliteSaver`, and `PostgresSaver`. AutoGen supports serialization via `save_state()` / `load_state()`.

## Human-in-the-Loop (HITL)

Humans need to participate in multi-agent workflows for:
- Approving high-stakes decisions
- Providing input the agents can't generate
- Correcting agent mistakes
- Steering the direction of research

### HITL Patterns

| Pattern | How It Works | Framework Support |
|---------|-------------|-------------------|
| **Approval Gate** | Agent pauses, human approves/rejects before continuing | LangGraph `interrupt()`, AutoGen `UserProxyAgent` |
| **Input Injection** | Human provides data mid-workflow | LangGraph `Command(resume=value)`, ADK callbacks |
| **Edit-in-Place** | Human modifies agent output before passing to next agent | LangGraph state edit + resume |
| **Observation Only** | Human monitors but doesn't intervene | All frameworks via logging |

### Implementation in LangGraph

```python
from langgraph.types import interrupt, Command

def critic_node(state):
    feedback = evaluate(state["draft"])
    if feedback["needs_human_review"]:
        # Pause execution, wait for human input
        human_decision = interrupt({
            "draft": state["draft"],
            "feedback": feedback,
            "question": "Should we continue refining or publish as-is?"
        })
        return {"status": human_decision}
    return {"feedback": feedback}
```

## Approaches Compared

| Mechanism | Coupling | Scalability | Debugging | Best For |
|-----------|---------|-------------|-----------|----------|
| Message Passing | Medium | Medium | Easy | Structured conversations |
| Shared State | Low | Medium | Medium | Pipeline workflows |
| Pub/Sub Events | Very Low | High | Hard | Distributed systems |
| Tool Invocation | High | Low | Easy | Hierarchical delegation |
| LLM Delegation | Low | Medium | Hard | Dynamic routing |

## Application to My Project

### Recommended Approach: Shared State + Conditional Routing

For the Strategy Research Team:

1. **Primary mechanism:** Shared state (LangGraph `TypedDict`) — agents read/write to a common state
2. **Routing:** Conditional edges based on state — orchestrator routes to the right agent
3. **Iteration:** Writer → Critic loop using conditional edges (if feedback says "revise" → Writer, else → END)
4. **Human-in-the-loop:** `interrupt()` after Critic review for high-stakes outputs
5. **Termination:** Quality gate (Critic approves) + max iterations (3-5 cycles)

### State Schema Design

```python
from typing import TypedDict, Annotated
from operator import add

class ResearchState(TypedDict):
    topic: str
    research_queries: list[str]
    research_results: Annotated[list[str], add]  # append reducer
    analysis: str
    draft: str
    feedback: Annotated[list[str], add]           # append reducer
    iteration: int
    max_iterations: int
    status: str  # "researching" | "analyzing" | "writing" | "reviewing" | "complete"
    total_tokens: int
    total_cost: float
```

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Context overflow | Passing entire conversation history to every agent | Summarize or filter state between agents |
| Lost state | No checkpointing, crash loses everything | Enable persistence (SqliteSaver) |
| Deadlocks | Agent A waits for Agent B who waits for Agent A | Design acyclic workflows or set timeouts |
| Stale state | Reading outdated info after parallel execution | Use reducers/locks for concurrent access |
| Over-communication | Agents share too much irrelevant context | Only pass relevant state keys to each agent |

## Resources for Deeper Learning

- [LangGraph State Management](https://docs.langchain.com/oss/python/langgraph/concepts/low_level/#state) — State, reducers, and persistence
- [AutoGen AgentChat Models](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/) — Message types and team coordination
- [Google ADK Session & State](https://google.github.io/adk-docs/sessions/) — Session state management in ADK

## Questions Remaining

- [ ] How to efficiently summarize long conversation histories to keep context manageable?
- [ ] What's the performance impact of different checkpoint storage backends?
- [ ] How to handle state schema migrations as the system evolves?
