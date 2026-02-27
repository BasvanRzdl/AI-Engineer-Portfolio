# Customer Operations Agent — Architecture Design

> **Type:** Architecture Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## Purpose

This document synthesizes all research (docs 01–10) into a concrete architecture design for the Customer Operations Agent. It serves as the **blueprint** for implementation.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Customer Operations Agent                     │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │  FastAPI  │───▶│  LangGraph   │───▶│  Tools & Services  │    │
│  │ Endpoint  │◀───│  StateGraph  │◀───│  (Mock Database)   │    │
│  └──────────┘    └──────┬───────┘    └────────────────────┘    │
│                         │                                       │
│               ┌─────────┴─────────┐                             │
│               │                   │                             │
│        ┌──────┴──────┐    ┌───────┴──────┐                     │
│        │ Checkpointer│    │    Store     │                     │
│        │ (Short-term)│    │ (Long-term) │                     │
│        └─────────────┘    └──────────────┘                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Architecture Pattern: Hybrid ReAct + Router

Based on research from docs 01 and 02, we'll use a **hybrid architecture**:

- **Router** at the top level to classify requests and route to specialized subflows
- **ReAct loop** within subflows for tool-using tasks (order lookup, refund processing)
- **Guardrail nodes** at critical points for safety checks
- **Human-in-the-loop** via `interrupt()` for high-risk actions

### Why This Combination?

| Concern | Solution |
|---------|----------|
| Different request types need different handling | Router pattern |
| Agent needs to gather info dynamically | ReAct loop |
| High-risk actions need oversight | HITL interrupts |
| Must be debuggable and auditable | State machine with tracing |

---

## Graph Architecture

```
                        START
                          │
                    ┌─────▼──────┐
                    │   greet /   │
                    │ load_memory │
                    └─────┬──────┘
                          │
                    ┌─────▼──────┐
                    │  classify   │
                    │  request    │
                    └─────┬──────┘
                          │
             ┌────────────┼────────────┬──────────────┐
             ▼            ▼            ▼              ▼
      ┌──────────┐ ┌──────────┐ ┌──────────┐  ┌──────────┐
      │  order   │ │  refund  │ │ shipping │  │ escalate │
      │  inquiry │ │  flow    │ │  inquiry │  │ to human │
      └────┬─────┘ └────┬─────┘ └────┬─────┘  └────┬─────┘
           │             │            │              │
           │        ┌────▼─────┐     │              │
           │        │  check   │     │              │
           │        │ guardrail│     │              │
           │        └────┬─────┘     │              │
           │             │           │              │
           │        ┌────▼─────┐     │              │
           │        │  approve │     │              │
           │        │ (HITL?)  │     │              │
           │        └────┬─────┘     │              │
           │             │           │              │
           ▼             ▼           ▼              ▼
      ┌──────────────────────────────────────────────┐
      │              respond / wrap_up                │
      └──────────────────┬───────────────────────────┘
                         │
                   ┌─────▼──────┐
                   │save_memory │
                   └─────┬──────┘
                         │
                        END
```

---

## State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class CustomerAgentState(TypedDict):
    # === Conversation ===
    messages: Annotated[list, add_messages]
    
    # === Customer Context ===
    customer_id: str | None          # Identified customer
    customer_name: str | None        # For personalization
    customer_preferences: dict       # From long-term memory
    
    # === Request Classification ===
    request_type: str | None         # "order_inquiry", "refund", "shipping", "general", "escalate"
    confidence: float                # Classification confidence
    
    # === Current Operation ===
    order_info: dict | None          # Looked-up order details
    refund_amount: float             # Refund amount if applicable
    refund_eligible: bool            # Whether refund is allowed
    
    # === Workflow Control ===
    needs_escalation: bool           # Should escalate to human agent
    escalation_reason: str | None    # Why escalation is needed
    approved: bool                   # Whether HITL approved action
    
    # === Tracking ===
    actions_taken: list[str]         # Log of actions performed
    resolution: str | None           # How the issue was resolved
```

---

## Node Definitions

### Node 1: Greet and Load Memory

```python
def greet_and_load_memory(state: CustomerAgentState) -> dict:
    """Initialize conversation with customer context from long-term memory."""
    store = get_store()
    customer_id = state.get("customer_id")
    
    preferences = {}
    if customer_id:
        items = store.search(("customers", customer_id, "preferences"))
        if items:
            preferences = items[0].value
    
    return {
        "customer_preferences": preferences,
        "actions_taken": [],
        "needs_escalation": False,
        "approved": False,
        "refund_amount": 0.0,
        "refund_eligible": False,
    }
```

### Node 2: Classify Request

```python
def classify_request(state: CustomerAgentState) -> dict:
    """Classify the customer's intent."""
    classification_prompt = """Classify the customer's request into exactly one category:
    - order_inquiry: Questions about order status, details, delivery
    - refund: Requests for refunds, returns, money back
    - shipping: Shipping status, tracking, address changes
    - general: General questions, greetings, account info
    - escalate: Customer explicitly asks for a human
    
    Respond with JSON: {"type": "<category>", "confidence": <0.0-1.0>}"""
    
    response = model.invoke([
        SystemMessage(content=classification_prompt),
        *state["messages"]
    ])
    
    result = json.loads(response.content)
    return {
        "request_type": result["type"],
        "confidence": result["confidence"],
        "messages": [response]
    }
```

### Node 3: Order Inquiry (ReAct)

```python
def handle_order_inquiry(state: CustomerAgentState) -> dict:
    """Handle order-related questions using tools."""
    order_tools = [order_lookup, check_shipping_status]
    model_with_tools = model.bind_tools(order_tools)
    
    system = SystemMessage(content="""You are handling an order inquiry.
    Use the available tools to look up order information.
    Always look up the order before answering questions about it.
    If you can't find the order, ask the customer to verify the order ID.""")
    
    response = model_with_tools.invoke([system, *state["messages"]])
    return {"messages": [response]}
```

### Node 4: Refund Flow

```python
def handle_refund(state: CustomerAgentState) -> dict:
    """Begin the refund process."""
    refund_tools = [order_lookup, check_return_eligibility]
    model_with_tools = model.bind_tools(refund_tools)
    
    system = SystemMessage(content="""You are handling a refund request.
    Steps:
    1. Look up the order if not already done
    2. Check return eligibility
    3. Confirm the refund details with the customer
    
    Do NOT process the refund yet — just gather information.""")
    
    response = model_with_tools.invoke([system, *state["messages"]])
    return {"messages": [response]}
```

### Node 5: Guardrail Check

```python
def check_guardrails(state: CustomerAgentState) -> dict:
    """Apply safety guardrails before processing refund."""
    amount = state.get("refund_amount", 0)
    
    if amount > 500:
        return {
            "needs_escalation": True,
            "escalation_reason": f"Refund amount ${amount} exceeds maximum auto-approve limit"
        }
    
    if not state.get("refund_eligible"):
        return {
            "needs_escalation": True,
            "escalation_reason": "Order not eligible for refund per policy"
        }
    
    return {"needs_escalation": False}
```

### Node 6: Human Approval

```python
from langgraph.types import interrupt

def request_approval(state: CustomerAgentState) -> dict:
    """Request human approval for high-value actions."""
    amount = state.get("refund_amount", 0)
    
    if amount > 100:
        decision = interrupt({
            "type": "refund_approval",
            "customer_id": state.get("customer_id"),
            "order_id": state.get("order_info", {}).get("order_id"),
            "amount": amount,
            "reason": state.get("messages")[-1].content if state["messages"] else "Unknown",
            "options": ["approve", "reject", "modify"]
        })
        
        if decision == "reject":
            return {
                "approved": False,
                "messages": [AIMessage(content="I'm sorry, the refund request was not approved. Let me connect you with a supervisor who can help.")]
            }
        
        return {"approved": True}
    
    # Auto-approve small amounts
    return {"approved": True}
```

### Node 7: Process Refund

```python
def process_refund_action(state: CustomerAgentState) -> dict:
    """Actually process the approved refund."""
    if not state.get("approved"):
        return {"resolution": "refund_rejected"}
    
    # Call the refund tool
    result = process_refund_tool.invoke({
        "order_id": state["order_info"]["order_id"],
        "amount": state["refund_amount"],
        "reason": "Customer request"
    })
    
    return {
        "messages": [AIMessage(content=f"Your refund of ${state['refund_amount']} has been processed. "
                               f"Reference: {result['refund_id']}. "
                               f"You should see it in {result['estimated_days']} business days.")],
        "resolution": "refund_processed",
        "actions_taken": ["process_refund"]
    }
```

### Node 8: Escalate

```python
def escalate_to_human(state: CustomerAgentState) -> dict:
    """Escalate conversation to a human agent."""
    reason = state.get("escalation_reason", "Customer request")
    
    # Create summary for human agent
    summary = model.invoke([
        SystemMessage(content="Summarize this conversation in 2-3 sentences for a human agent."),
        *state["messages"][-6:]
    ])
    
    result = escalation_tool.invoke({
        "reason": reason,
        "priority": "high" if state.get("refund_amount", 0) > 200 else "normal",
        "context_summary": summary.content
    })
    
    return {
        "messages": [AIMessage(content=f"I'm connecting you with a specialist. "
                               f"Reference: {result['escalation_id']}. "
                               f"Estimated wait: {result['estimated_wait']} minutes.")],
        "resolution": "escalated",
        "actions_taken": ["escalate"]
    }
```

### Node 9: Save Memory

```python
def save_conversation_memory(state: CustomerAgentState) -> dict:
    """Save interaction summary to long-term memory."""
    store = get_store()
    customer_id = state.get("customer_id")
    
    if customer_id and state.get("resolution"):
        store.put(
            namespace=("customers", customer_id, "history"),
            key=str(uuid.uuid4()),
            value={
                "date": datetime.now().isoformat(),
                "request_type": state.get("request_type"),
                "resolution": state.get("resolution"),
                "actions": state.get("actions_taken", []),
                "refund_amount": state.get("refund_amount", 0),
            }
        )
    
    return {}
```

---

## Edge Definitions

```python
def route_by_request_type(state: CustomerAgentState) -> str:
    """Route to appropriate handler based on classification."""
    if state.get("needs_escalation"):
        return "escalate"
    
    request_type = state.get("request_type", "general")
    confidence = state.get("confidence", 0)
    
    # Low confidence → escalate
    if confidence < 0.6:
        return "escalate"
    
    routes = {
        "order_inquiry": "handle_order",
        "refund": "handle_refund",
        "shipping": "handle_shipping",
        "escalate": "escalate",
        "general": "handle_general",
    }
    
    return routes.get(request_type, "handle_general")

def route_after_guardrail(state: CustomerAgentState) -> str:
    """Route based on guardrail check results."""
    if state.get("needs_escalation"):
        return "escalate"
    return "request_approval"

def should_continue_react(state: CustomerAgentState) -> str:
    """Check if ReAct loop should continue (tool calls pending)."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "respond"
```

---

## Full Graph Assembly

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.prebuilt import ToolNode

# Tools
all_tools = [order_lookup, check_shipping_status, check_return_eligibility, 
             process_refund_tool, escalation_tool, customer_lookup]

# Build graph
builder = StateGraph(CustomerAgentState)

# Add nodes
builder.add_node("greet", greet_and_load_memory)
builder.add_node("classify", classify_request)
builder.add_node("handle_order", handle_order_inquiry)
builder.add_node("handle_refund", handle_refund)
builder.add_node("handle_shipping", handle_shipping_inquiry)
builder.add_node("handle_general", handle_general_inquiry)
builder.add_node("tools", ToolNode(all_tools))
builder.add_node("check_guardrails", check_guardrails)
builder.add_node("request_approval", request_approval)
builder.add_node("process_refund", process_refund_action)
builder.add_node("escalate", escalate_to_human)
builder.add_node("respond", generate_response)
builder.add_node("save_memory", save_conversation_memory)

# Edges: main flow
builder.add_edge(START, "greet")
builder.add_edge("greet", "classify")

# Routing after classification
builder.add_conditional_edges("classify", route_by_request_type, {
    "handle_order": "handle_order",
    "handle_refund": "handle_refund",
    "handle_shipping": "handle_shipping",
    "handle_general": "handle_general",
    "escalate": "escalate",
})

# ReAct loops for tool-using nodes
for node in ["handle_order", "handle_refund", "handle_shipping"]:
    builder.add_conditional_edges(node, should_continue_react, {
        "tools": "tools",
        "respond": "check_guardrails" if node == "handle_refund" else "respond",
    })
    builder.add_edge("tools", node)  # Loop back after tool execution

# Refund-specific flow
builder.add_conditional_edges("check_guardrails", route_after_guardrail, {
    "request_approval": "request_approval",
    "escalate": "escalate",
})
builder.add_edge("request_approval", "process_refund")
builder.add_edge("process_refund", "respond")

# General and escalation flows
builder.add_edge("handle_general", "respond")
builder.add_edge("escalate", "save_memory")

# Wrap up
builder.add_edge("respond", "save_memory")
builder.add_edge("save_memory", END)

# Compile
checkpointer = InMemorySaver()
store = InMemoryStore()

graph = builder.compile(
    checkpointer=checkpointer,
    store=store
)
```

---

## Tool Ecosystem

| Tool | Input | Output | Risk | Scoping |
|------|-------|--------|------|---------|
| `customer_lookup` | customer_id or email | Customer profile | Low | All flows |
| `order_lookup` | order_id | Order details | Low | Order/Refund flows |
| `check_shipping_status` | order_id or tracking# | Shipping info | Low | Order/Shipping flows |
| `check_return_eligibility` | order_id | Eligibility result | Low | Refund flow |
| `process_refund` | order_id, amount, reason | Refund confirmation | **High** | Refund flow only |
| `update_shipping_address` | order_id, new_address | Confirmation | Medium | Shipping flow |
| `escalate_to_human` | reason, priority, context | Ticket reference | Low | All flows |

---

## Guardrail Strategy

Based on research from doc 05:

| Layer | Implementation |
|-------|---------------|
| **Input validation** | Validate order IDs, amounts, customer IDs before tool calls |
| **Action limits** | Refunds: auto < $100, HITL $100-$500, escalate > $500 |
| **Rate limiting** | Max 3 refunds per customer per day |
| **PII protection** | Mask email/phone in logs, don't store raw PII in memory |
| **Output filtering** | Don't leak system prompts, internal IDs, or tool schemas |
| **Escalation triggers** | Customer frustration, repeated failures, explicit request |

---

## Memory Strategy

Based on research from docs 04 and 10:

| Type | Storage | Scope | Lifetime |
|------|---------|-------|----------|
| **Conversation** | Checkpointer | Per thread | Session |
| **Customer prefs** | Store | Per customer | Persistent |
| **Interaction history** | Store | Per customer | 90 days |
| **Business policies** | Store | Global | Persistent |

---

## Technology Stack

| Component | Technology | Reason |
|-----------|-----------|--------|
| **Agent framework** | LangGraph | State management, HITL, persistence |
| **LLM** | GPT-4o (via Azure OpenAI or OpenAI) | Best tool-calling, reasoning |
| **API layer** | FastAPI | Async, WebSocket support for streaming |
| **Checkpointer** | InMemorySaver → PostgresSaver | Dev → Production |
| **Store** | InMemoryStore → PostgresStore | Dev → Production |
| **Database** | Mock (dict-based) → SQLite → PostgreSQL | Progressive complexity |
| **Observability** | LangSmith + structured logging | Tracing + audit trail |
| **Testing** | pytest + golden dataset | Unit + integration + adversarial |

---

## Development Phases

### Phase A: Foundation (Week 1)
- Set up project structure
- Define state schema
- Implement mock database with sample data
- Build basic tools (order_lookup, customer_lookup)
- Create minimal graph (classify → respond)

### Phase B: Core Flows (Week 2)
- Add routing logic
- Implement order inquiry flow with ReAct
- Implement shipping inquiry flow
- Add conversation memory (checkpointer)

### Phase C: Refund Flow (Week 3)
- Implement refund tools (eligibility, processing)
- Add guardrail checks
- Implement human-in-the-loop approval
- Add escalation path

### Phase D: Memory & Polish (Week 4)
- Implement long-term memory (Store)
- Add personalized greetings
- Customer interaction history
- Conversation summarization

### Phase E: Evaluation & Hardening (Week 5)
- Build golden test dataset
- Implement evaluation pipeline
- Adversarial testing (prompt injection, edge cases)
- Performance optimization

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Hybrid ReAct + Router | Best of both: structured routing + flexible tool use |
| State management | TypedDict | Simpler than Pydantic for initial build |
| Refund threshold | $100 HITL / $500 escalate | Balance automation with risk management |
| Memory model | Checkpointer + Store | Short-term + long-term covered |
| Tool scoping | Per-flow | Reduce errors, enforce workflow ordering |
| Evaluation | Golden dataset + LLM-as-judge | Automated regression + nuanced quality |

---

## Resources

All prior research documents in this series:
- [01 — Agent Architectures](01-concept-agent-architectures.md)
- [02 — State Machines](02-concept-state-machines.md)
- [03 — Tool Use & Function Calling](03-concept-tool-use-and-function-calling.md)
- [04 — Memory Systems](04-concept-memory-systems.md)
- [05 — Guardrails & Safety](05-concept-guardrails-and-safety.md)
- [06 — Evaluation & Observability](06-concept-evaluation-and-observability.md)
- [07 — LangGraph Deep Dive](07-technology-langgraph-deep-dive.md)
- [08 — Tool Definition Patterns](08-technology-tool-definition-patterns.md)
- [09 — Human-in-the-Loop](09-technology-human-in-the-loop.md)
- [10 — Conversation Memory](10-technology-conversation-memory.md)
