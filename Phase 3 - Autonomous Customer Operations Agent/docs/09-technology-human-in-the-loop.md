# Human-in-the-Loop Implementations

> **Type:** Technology Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## In My Own Words

Human-in-the-loop (HITL) means the agent can **pause execution**, present information to a human, **wait for input**, and then **resume** based on the human's decision. It's the bridge between "fully automated" and "fully manual."

For our Customer Operations Agent, this is critical: the agent should handle routine tasks automatically but pause for human approval on high-risk actions (large refunds, account changes, edge cases).

---

## The LangGraph HITL Model

LangGraph uses a simple but powerful pattern:

1. **`interrupt()`** — A function that pauses graph execution and returns a value to the caller
2. **`Command(resume=value)`** — The way to resume execution with the human's input
3. **Checkpointer** — Required to persist state while the graph is paused

```
Agent running → hits interrupt() → execution pauses → state saved
                                                          ↓
Human reviews ← information displayed ← interrupt value returned
      ↓
Human decides → Command(resume=decision) → execution resumes
```

---

## Basic Pattern: Approve or Reject

```python
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

def process_refund(state: MessagesState):
    """Process a refund, pausing for approval if needed."""
    refund_amount = state.get("refund_amount", 0)
    
    if refund_amount > 100:
        # Pause for human approval
        human_response = interrupt({
            "question": f"Approve refund of ${refund_amount}?",
            "order_id": state.get("order_id"),
            "customer_id": state.get("customer_id"),
            "options": ["approve", "reject", "modify"]
        })
        
        # Execution resumes here after human responds
        if human_response == "reject":
            return {
                "messages": [AIMessage(content="Refund has been declined by a supervisor.")]
            }
        elif human_response == "approve":
            # Process the refund
            return {
                "messages": [AIMessage(content=f"Refund of ${refund_amount} approved and processed.")]
            }
    else:
        # Auto-approve small refunds
        return {
            "messages": [AIMessage(content=f"Refund of ${refund_amount} processed automatically.")]
        }

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("process_refund", process_refund)
builder.add_edge(START, "process_refund")
builder.add_edge("process_refund", END)

# MUST have a checkpointer for HITL
graph = builder.compile(checkpointer=InMemorySaver())
```

### Running the HITL Flow

```python
config = {"configurable": {"thread_id": "refund-session-1"}}

# Step 1: Invoke the graph (will pause at interrupt)
result = graph.invoke(
    {"messages": [HumanMessage(content="I want a refund")], "refund_amount": 150.00},
    config=config
)
# Result shows the interrupt information

# Step 2: Check what the graph is waiting for
state = graph.get_state(config)
print(state.tasks)  # Shows pending interrupt with the question

# Step 3: Resume with human decision
result = graph.invoke(
    Command(resume="approve"),
    config=config
)
# Graph continues from where it paused
```

---

## Pattern: Review and Edit

The human can modify the agent's proposed action:

```python
def draft_response(state: AgentState):
    """Draft a response and let human review it."""
    # Agent drafts a response
    draft = model.invoke([
        SystemMessage(content="Draft a customer response for this situation."),
        *state["messages"]
    ])
    
    # Let human review and potentially edit
    human_review = interrupt({
        "type": "review_draft",
        "draft": draft.content,
        "instruction": "Review the draft. Reply with 'approve' or provide an edited version."
    })
    
    if human_review == "approve":
        return {"messages": [draft]}
    else:
        # Human provided an edited response
        return {"messages": [AIMessage(content=human_review)]}
```

---

## Pattern: Multi-Step Approval

For complex workflows with multiple approval points:

```python
def process_complex_refund(state: AgentState):
    """Multi-step refund with multiple checkpoints."""
    
    # Step 1: Confirm refund details
    details_approval = interrupt({
        "step": 1,
        "type": "confirm_details",
        "message": f"Refund details: Order {state['order_id']}, Amount: ${state['refund_amount']}",
        "options": ["confirm", "cancel"]
    })
    
    if details_approval == "cancel":
        return {"messages": [AIMessage(content="Refund cancelled.")]}
    
    # Step 2: Verify method
    method_approval = interrupt({
        "step": 2,
        "type": "confirm_method",
        "message": "Refund to original payment method?",
        "options": ["original_method", "store_credit", "cancel"]
    })
    
    if method_approval == "cancel":
        return {"messages": [AIMessage(content="Refund cancelled.")]}
    
    # Process with approved details
    return {
        "messages": [AIMessage(
            content=f"Refund of ${state['refund_amount']} processed via {method_approval}."
        )],
        "refund_method": method_approval
    }
```

**Important:** Each `interrupt()` is resumed separately. The graph pauses at the first interrupt, resumes, continues to the second interrupt, pauses again, and so on.

```python
config = {"configurable": {"thread_id": "complex-refund-1"}}

# First invoke → pauses at step 1
graph.invoke(input, config)

# Resume step 1 → pauses at step 2
graph.invoke(Command(resume="confirm"), config)

# Resume step 2 → completes
graph.invoke(Command(resume="original_method"), config)
```

---

## Pattern: Interrupt in Tools

You can use `interrupt()` inside tool functions:

```python
@tool
def process_refund(order_id: str, amount: float) -> dict:
    """Process a refund for an order."""
    
    if amount > 100:
        approval = interrupt({
            "type": "refund_approval",
            "order_id": order_id,
            "amount": amount,
            "question": f"Approve refund of ${amount} for order {order_id}?"
        })
        
        if approval != "approve":
            return {"status": "rejected", "reason": "Human reviewer declined the refund"}
    
    # Process the refund
    refund = payment_service.process(order_id, amount)
    return {"status": "processed", "refund_id": refund.id}
```

When this tool is called by the agent and the amount exceeds $100, the entire graph pauses. When resumed, the tool continues from the `interrupt()` call.

---

## Critical Rules for Interrupts

### Rule 1: Never Wrap in Try/Except

```python
# ❌ WRONG: interrupt() raises a special exception — don't catch it
def my_node(state):
    try:
        result = interrupt("approve?")
    except:
        pass  # This will break HITL!

# ✅ CORRECT: Let interrupt propagate
def my_node(state):
    result = interrupt("approve?")
    # Code continues here after resume
```

### Rule 2: Don't Reorder Interrupts

Multiple interrupts in a node must always execute in the same order:

```python
# ✅ CORRECT: Interrupts always in the same order
def my_node(state):
    approval_1 = interrupt("First question?")
    approval_2 = interrupt("Second question?")
    return {"approved": approval_1 == "yes" and approval_2 == "yes"}

# ❌ WRONG: Conditional interrupts that change order
def my_node(state):
    if some_condition:
        a = interrupt("Question A?")
        b = interrupt("Question B?")
    else:
        b = interrupt("Question B?")  # Different order!
        a = interrupt("Question A?")
```

### Rule 3: Side Effects Must Be Idempotent

Code before an `interrupt()` may run again on resume:

```python
# ❌ DANGEROUS: Side effect runs twice
def my_node(state):
    send_email(state["customer_email"])  # Runs on first call AND on resume!
    result = interrupt("Continue?")
    return result

# ✅ SAFE: Guard against re-execution
def my_node(state):
    if not state.get("email_sent"):
        send_email(state["customer_email"])
    result = interrupt("Continue?")
    return {"email_sent": True, ...}
```

### Rule 4: Checkpointer Is Required

```python
# ❌ Won't work: No checkpointer
graph = builder.compile()  # interrupt() will raise an error

# ✅ Works: Checkpointer provided
graph = builder.compile(checkpointer=InMemorySaver())
```

---

## Alternative: interrupt_before / interrupt_after

Instead of using `interrupt()` inside nodes, you can configure the graph to automatically pause before or after specific nodes:

```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["process_refund"],  # Pause BEFORE this node runs
    interrupt_after=["draft_response"]     # Pause AFTER this node runs
)
```

Then resume with:
```python
# Check state
state = graph.get_state(config)
print(f"Paused before: {state.next}")  # ['process_refund']

# Resume (optionally update state first)
graph.update_state(config, {"approved": True})
graph.invoke(None, config)  # Continue execution
```

**When to use which:**
- `interrupt()` in nodes: More control, can pass/receive structured data
- `interrupt_before/after`: Simpler, good for "pause-and-inspect" patterns

---

## Practical HITL Architecture for Our Agent

### Decision: When to Interrupt

| Scenario | Action | Threshold |
|----------|--------|-----------|
| Refund amount | Require approval | > $100 |
| Account modification | Require approval | Always |
| Unclear customer intent | Ask for guidance | Confidence < 0.7 |
| Multiple possible actions | Let human choose | > 2 reasonable actions |
| Customer requests human | Escalate | Always |
| Repeated failed attempts | Escalate | > 3 failures |

### Integration with the UI

```python
# API endpoint that handles the conversation
async def handle_message(thread_id: str, message: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config
    )
    
    # Check if graph is paused (waiting for human)
    state = graph.get_state(config)
    
    if state.tasks:
        # Graph is paused — return interrupt info to UI
        interrupt_info = state.tasks[0].interrupts[0].value
        return {
            "type": "approval_needed",
            "data": interrupt_info,
            "thread_id": thread_id
        }
    else:
        # Graph completed — return agent response
        return {
            "type": "response",
            "message": result["messages"][-1].content,
            "thread_id": thread_id
        }

async def handle_approval(thread_id: str, decision: str):
    config = {"configurable": {"thread_id": thread_id}}
    
    result = graph.invoke(
        Command(resume=decision),
        config=config
    )
    
    return {
        "type": "response",
        "message": result["messages"][-1].content,
        "thread_id": thread_id
    }
```

---

## Testing HITL Flows

```python
def test_refund_approval_flow():
    """Test that high-value refunds require approval."""
    config = {"configurable": {"thread_id": "test-hitl-1"}}
    
    # Trigger refund flow
    graph.invoke(
        {"messages": [HumanMessage(content="Refund order ORD-123")], "refund_amount": 200},
        config=config
    )
    
    # Verify graph is paused
    state = graph.get_state(config)
    assert state.tasks  # Should have pending tasks
    assert "approve" in str(state.tasks[0].interrupts[0].value)
    
    # Resume with approval
    result = graph.invoke(Command(resume="approve"), config=config)
    
    # Verify refund was processed
    assert "processed" in result["messages"][-1].content.lower()

def test_refund_rejection_flow():
    """Test that rejected refunds are handled gracefully."""
    config = {"configurable": {"thread_id": "test-hitl-2"}}
    
    graph.invoke(
        {"messages": [HumanMessage(content="Refund order ORD-123")], "refund_amount": 200},
        config=config
    )
    
    # Reject
    result = graph.invoke(Command(resume="reject"), config=config)
    assert "declined" in result["messages"][-1].content.lower()
```

---

## Resources

- [LangGraph Human-in-the-Loop](https://docs.langchain.com/oss/python/langgraph/how-tos/human-in-the-loop) — Official HITL guide
- [LangGraph Interrupts Conceptual Guide](https://docs.langchain.com/oss/python/langgraph/concepts/interrupts) — How interrupts work
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/concepts/persistence) — Checkpointers and state management
