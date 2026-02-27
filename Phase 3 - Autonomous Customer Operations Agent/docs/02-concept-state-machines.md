# State Machines for Conversational Agents

> **Type:** Concept Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## In My Own Words

A **state machine** models an agent's behavior as a set of **states** connected by **transitions**. At any point, the agent is in exactly one state (e.g., "gathering info," "processing refund," "waiting for approval"). Transitions between states are triggered by events — user messages, tool results, or internal decisions.

For conversational agents, a state machine provides **structure and predictability** to what would otherwise be an unpredictable LLM conversation. It tells the agent: "Given where you are and what just happened, here's what you can do next."

---

## Why This Matters

Without a state machine, an agent is just an LLM in a loop — it can do anything at any time, which sounds flexible but is actually dangerous:
- It might process a refund before verifying the customer's identity
- It might skip confirmation steps for destructive actions  
- It might loop forever without making progress
- It's hard to audit what happened and why

A state machine gives you **guardrails without rigidity**: the LLM still makes decisions within each state, but the overall flow is controlled.

---

## Core Principles

### 1. State = Current Situation

The **state** captures everything the system needs to know right now:
- Where are we in the conversation flow?
- What information have we collected?
- What actions have been taken?
- What's the current decision context?

In LangGraph, state is represented as a `TypedDict` or Pydantic model:

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import MessagesState

class AgentState(MessagesState):
    """Extends MessagesState with custom fields."""
    customer_id: str | None
    order_id: str | None  
    action_type: str | None        # "refund", "shipping", "inquiry", etc.
    action_confirmed: bool
    escalation_reason: str | None
    total_refund_amount: float
```

### 2. Nodes = Actions / Processing

**Nodes** are the functions that do the actual work within each state. A node:
- Reads the current state
- Performs some computation (LLM call, tool call, logic)
- Returns an update to the state

```python
def classify_request(state: AgentState):
    """Classify what the customer wants."""
    result = llm.invoke([
        SystemMessage(content="Classify the customer request..."),
        *state["messages"]
    ])
    return {"action_type": result.content}
```

### 3. Edges = Transitions

**Edges** define how the agent moves between nodes. They can be:
- **Fixed**: Always go from A to B
- **Conditional**: Choose the next node based on the current state

```python
# Fixed edge: after greeting, always classify
graph.add_edge("greet", "classify_request")

# Conditional edge: after classification, route to the right handler
def route_by_type(state: AgentState):
    if state["action_type"] == "refund":
        return "handle_refund"
    elif state["action_type"] == "shipping":
        return "handle_shipping"
    else:
        return "general_inquiry"

graph.add_conditional_edges("classify_request", route_by_type)
```

### 4. The Graph IS Your Architecture

In LangGraph, the graph structure literally encodes your architectural decisions:
- Which operations are possible?
- What order must they happen in?
- Where are the decision points?
- Where does human oversight fit?

This is a fundamental insight: **designing the graph is designing the agent's behavior**.

---

## State Machine Patterns for Agents

### Pattern 1: Linear Pipeline

The simplest pattern — each step happens in order:

```
START → Greet → Classify → Handle → Respond → END
```

**Use when:** Tasks are straightforward and always follow the same steps.

### Pattern 2: Branching Router

Classify the input and route to specialized handlers:

```
START → Classify → [Refund Handler | Shipping Handler | General Handler] → Respond → END
```

**Use when:** Different request types need fundamentally different handling.

### Pattern 3: ReAct Loop

The agent loops between thinking and acting until done:

```
START → Agent (LLM + Tools) → Should Continue? → [Agent Loop | END]
```

**Use when:** Tasks are open-ended and require multiple tool calls.

### Pattern 4: Approval Gate

Critical actions require confirmation before execution:

```
... → Propose Action → [Human Approve? | Auto-approve?] → Execute → ...
                                ↓ (rejected)
                           Revise or Escalate
```

**Use when:** Destructive or high-value operations need oversight.

### Pattern 5: Escalation Path

Any node can escalate to a human when the agent is uncertain:

```
Any Node → Confidence Check → [Continue | Escalate to Human]
```

**Use when:** The agent encounters ambiguity or exceeds its authority.

---

## Decision Boundaries: Act vs. Ask vs. Escalate

A critical design decision for any customer operations agent is defining **when to act autonomously, when to ask the customer, and when to escalate to a human**.

### Act Autonomously When:
- The action is low-risk and reversible (e.g., looking up order status)
- The customer's intent is clear and unambiguous
- The action is within policy limits (e.g., refund under $100)
- No sensitive information is being modified

### Ask the Customer When:
- The request is ambiguous ("I have a problem with my order" — which order?)
- Multiple valid interpretations exist
- You need confirmation of identity or intent
- The customer needs to make a choice

### Escalate to Human When:
- The request exceeds the agent's authority (refund > $100)
- The customer is frustrated or asks for a human
- The agent has tried multiple times without resolution
- Legal, compliance, or safety concerns arise
- The request is outside the agent's tool capabilities

### Encoding This in the State Machine:

```python
def decide_next_step(state: AgentState):
    """Determine whether to act, ask, or escalate."""
    
    # Escalation triggers
    if state.get("escalation_reason"):
        return "escalate_to_human"
    if state.get("customer_frustrated"):
        return "escalate_to_human"
    
    # Need more info from customer
    if state.get("needs_clarification"):
        return "ask_customer"
    
    # Approval required for high-value actions
    if state.get("total_refund_amount", 0) > 100:
        return "request_approval"
    
    # Safe to act autonomously
    return "execute_action"
```

---

## State Design Best Practices

### 1. Keep State Flat and Simple
Avoid deeply nested objects. Flat state is easier to update, serialize, and debug.

```python
# ✅ Good: flat, clear keys
class AgentState(TypedDict):
    customer_id: str
    order_id: str
    refund_amount: float

# ❌ Bad: nested, harder to work with
class AgentState(TypedDict):
    context: dict  # {"customer": {"id": "...", "order": {"id": "..."}}}
```

### 2. Use Reducers for Accumulating Data
When multiple nodes write to the same key, use reducers to define how updates combine:

```python
from typing import Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Messages accumulate
    actions_taken: Annotated[list[str], add]  # Actions accumulate
    current_step: str  # Overwrites (no reducer)
```

### 3. Track Metadata for Debugging
Include fields that help you understand what happened:

```python
class AgentState(TypedDict):
    # ... operational fields ...
    steps_taken: int
    last_tool_called: str
    confidence_score: float
    reasoning_trace: list[str]
```

### 4. Define Clear Terminal States
Know when the conversation is done:

```python
def is_conversation_complete(state: AgentState):
    if state.get("resolution_reached"):
        return END
    if state.get("escalated"):
        return END
    if state["steps_taken"] > 10:
        return "escalate_to_human"  # Prevent infinite loops
    return "continue"
```

---

## Application to Our Project

### Proposed State Schema

```python
class CustomerAgentState(MessagesState):
    # Customer context
    customer_id: str | None
    customer_name: str | None
    
    # Current request
    request_type: str | None          # refund, shipping, inquiry, complaint
    order_id: str | None
    
    # Action tracking  
    proposed_action: str | None       # What the agent wants to do
    action_requires_approval: bool    # Does this need human approval?
    action_confirmed: bool            # Has it been confirmed?
    
    # Financial
    refund_amount: float              # Total refund amount proposed
    
    # Flow control
    escalation_reason: str | None
    resolution_summary: str | None
    steps_taken: int
```

### Proposed Graph Structure

```
START
  ↓
[Receive Message]
  ↓
[Classify Request] ──→ refund ──→ [Lookup Order] → [Check Eligibility] → [Process/Escalate]
  ↓                 ──→ shipping → [Lookup Order] → [Update Shipping]
  ↓                 ──→ inquiry ─→ [Search KB] → [Respond]
  ↓                 ──→ escalate → [Prepare Context] → [Human Handoff]
  ↓
[Respond to Customer]
  ↓
END (or loop for follow-up)
```

---

## Resources

- [LangGraph Graph API Concepts](https://docs.langchain.com/oss/python/langgraph/graph-api) — State, nodes, edges, reducers
- [LangGraph Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Pattern examples
- [Finite State Machines in AI Agents](https://blog.langchain.dev/langgraph/) — LangChain blog
