# LangGraph Deep Dive

> **Type:** Technology Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## What Is LangGraph?

LangGraph is a framework for building **stateful, multi-step AI applications** using a graph-based architecture. It's built by the LangChain team but is a separate library focused specifically on **orchestrating agent workflows** as directed graphs.

### Why LangGraph Over Raw LangChain?

| Concern | Raw LangChain | LangGraph |
|---------|--------------|-----------|
| **State management** | Manual, ad-hoc | Built-in with TypedDict/Pydantic |
| **Control flow** | Linear chains or basic routing | Full graph with conditional edges |
| **Persistence** | Not built in | Checkpointers (memory, Postgres, SQLite) |
| **Human-in-the-loop** | Difficult to implement | Native `interrupt()` + `Command(resume=)` |
| **Streaming** | Basic token streaming | Token, node, state update, and custom event streaming |
| **Long-running workflows** | Hard to manage | Built-in with persistence + thread resumption |
| **Debugging** | Limited visibility | Full state inspection at every step |

**Key insight:** LangChain provides the **components** (models, tools, prompts). LangGraph provides the **orchestration** (how components work together).

---

## Core Concepts

### 1. StateGraph

The central object. You define a graph that operates on a shared **state** object.

```python
from langgraph.graph import StateGraph, START, END

# Define your state
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # Chat history
    customer_id: str | None                   # Current customer
    action_type: str | None                   # What the agent is doing

# Create the graph
graph_builder = StateGraph(AgentState)
```

### 2. State (TypedDict or Pydantic)

State is the **data container** that flows through the graph. Every node reads from and writes to this state.

```python
# Option A: TypedDict (simple, recommended for starting)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    customer_id: str | None
    order_info: dict | None
    refund_amount: float

# Option B: Pydantic BaseModel (with validation)
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    messages: Annotated[list, add_messages]
    customer_id: str | None = None
    order_info: dict | None = None
    refund_amount: float = Field(default=0.0, ge=0.0)
```

**Important: Reducers**

When a node returns state, how does it combine with existing state? Reducers define this:

```python
from operator import add

class AgentState(TypedDict):
    # add_messages: Appends new messages to the list
    messages: Annotated[list, add_messages]
    
    # add: Accumulates values (e.g., for counting)
    steps_taken: Annotated[int, add]
    
    # No annotation = REPLACE (last write wins)
    customer_id: str | None
```

Default behavior (no annotation): the value is **replaced** entirely.

### 3. Nodes

Nodes are **Python functions** that take state as input and return a partial state update:

```python
def classify_request(state: AgentState) -> dict:
    """Classify the customer's request."""
    messages = state["messages"]
    
    # Call LLM to classify
    response = model.invoke([
        SystemMessage(content="Classify the customer request as: order_inquiry, refund, shipping, escalation"),
        *messages
    ])
    
    # Return partial state update
    return {
        "messages": [response],
        "action_type": response.content.strip().lower()
    }

# Add node to graph
graph_builder.add_node("classify", classify_request)
```

**Rules for nodes:**
- Take `state` as first parameter
- Return a `dict` with the state keys you want to update
- Only include keys you want to change (partial update)
- Can be sync or async

### 4. Edges

Edges connect nodes and define the flow:

```python
# Fixed edge: Always goes from A to B
graph_builder.add_edge("classify", "route")

# Conditional edge: Choose next node based on state
def route_request(state: AgentState) -> str:
    """Return the name of the next node."""
    action = state.get("action_type")
    if action == "refund":
        return "handle_refund"
    elif action == "order_inquiry":
        return "handle_order_inquiry"
    elif action == "shipping":
        return "handle_shipping"
    else:
        return "escalate"

graph_builder.add_conditional_edges(
    "classify",  # Source node
    route_request,  # Routing function
    {
        "handle_refund": "handle_refund",
        "handle_order_inquiry": "handle_order_inquiry", 
        "handle_shipping": "handle_shipping",
        "escalate": "escalate"
    }
)

# Start and End
graph_builder.add_edge(START, "classify")
graph_builder.add_edge("handle_refund", END)
```

### 5. MessagesState (Convenience)

For chat-based agents, LangGraph provides a pre-built state:

```python
from langgraph.graph import MessagesState

# MessagesState is equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]

# You can extend it:
class AgentState(MessagesState):
    customer_id: str | None
    action_type: str | None
```

### 6. Compilation

After defining nodes and edges, compile the graph:

```python
# Simple compilation
graph = graph_builder.compile()

# With persistence (enables memory, HITL, time-travel)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# With interrupt points
graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["process_refund"]  # Pause before this node
)
```

---

## Graph Execution Model

### How Execution Works

1. You call `graph.invoke(initial_state, config)` or `graph.stream(...)`
2. LangGraph starts at the `START` node
3. It follows edges to the next node(s)
4. Each node runs and returns state updates
5. State is updated (using reducers)
6. Next node(s) are determined via edges
7. Repeat until `END` is reached or an interrupt occurs

### The Config Object

Every invocation takes a `config` that identifies the conversation:

```python
config = {
    "configurable": {
        "thread_id": "conversation-123"  # Required for persistence
    }
}

# Invoke
result = graph.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config=config
)

# Continue the same conversation
result = graph.invoke(
    {"messages": [HumanMessage(content="What's my order status?")]},
    config=config  # Same thread_id = same conversation
)
```

### Streaming

LangGraph supports multiple streaming modes:

```python
# Stream state updates (see state after each node)
for event in graph.stream(input, config, stream_mode="updates"):
    print(event)

# Stream values (see full state after each node)  
for event in graph.stream(input, config, stream_mode="values"):
    print(event)

# Stream LLM tokens as they're generated
async for event in graph.astream_events(input, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

---

## Persistence and Checkpointers

### What Checkpointers Do

Checkpointers save the state of the graph at every step. This enables:
- **Conversation memory**: Resume conversations across requests
- **Human-in-the-loop**: Pause and resume execution
- **Time travel**: Go back to any previous state
- **Fault tolerance**: Recover from crashes

### Available Checkpointers

```python
# In-memory (development only, lost on restart)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# SQLite (single-server, persists to disk)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# PostgreSQL (production, multi-server)
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:pass@localhost:5432/mydb"
)
```

### State Inspection

With a checkpointer, you can inspect the graph's state:

```python
# Get current state
state = graph.get_state(config)
print(state.values)  # Current state values
print(state.next)    # Next node(s) to execute

# Get state history (all checkpoints)
for state in graph.get_state_history(config):
    print(f"Step: {state.metadata['step']}, Node: {state.metadata.get('source')}")
    
# Update state manually (e.g., after human review)
graph.update_state(config, {"action_type": "approved"})
```

---

## The Command Object

`Command` combines a state update with a navigation instruction:

```python
from langgraph.types import Command

def review_node(state: AgentState) -> Command:
    """Review and decide next step."""
    if state["refund_amount"] > 100:
        # Update state AND navigate to escalation
        return Command(
            update={"escalation_reason": "High value refund"},
            goto="escalate"
        )
    else:
        return Command(
            update={"approved": True},
            goto="process_refund"
        )
```

**When to use Command vs conditional edges:**
- **Conditional edges**: Routing logic is separate from node logic (cleaner separation)
- **Command**: Routing is inherently tied to the node's computation (avoids redundant computation)

---

## Subgraphs

For complex agents, break the graph into subgraphs:

```python
# Define a subgraph for refund handling
refund_builder = StateGraph(RefundState)
refund_builder.add_node("check_eligibility", check_eligibility)
refund_builder.add_node("calculate_amount", calculate_amount)
refund_builder.add_node("process", process_refund)
# ... add edges
refund_graph = refund_builder.compile()

# Use it as a node in the main graph
main_builder = StateGraph(AgentState)
main_builder.add_node("classify", classify_request)
main_builder.add_node("handle_refund", refund_graph)  # Subgraph as a node!
main_builder.add_node("respond", generate_response)
```

**State mapping**: If subgraph state differs from parent state, LangGraph handles overlapping keys automatically. For non-overlapping keys, you can write wrapper functions.

---

## Recursion Limit

To prevent infinite loops, LangGraph has a recursion limit:

```python
# Default is 25 steps
result = graph.invoke(
    input, 
    config={
        "configurable": {"thread_id": "t1"},
        "recursion_limit": 50  # Override if needed
    }
)
```

If the limit is hit, a `GraphRecursionError` is raised.

---

## Common Patterns in LangGraph

### Pattern 1: ReAct Agent

```python
from langgraph.prebuilt import create_react_agent

# Simplest way to create an agent
tools = [order_lookup, process_refund, check_shipping]
agent = create_react_agent(model, tools, checkpointer=checkpointer)

# Or build it manually for more control:
def agent_node(state: AgentState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState):
    # Execute tool calls from the last message
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        result = tool_map[tool_call["name"]].invoke(tool_call["args"])
        results.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": results}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
graph = builder.compile()
```

### Pattern 2: Router

```python
def router(state: AgentState) -> str:
    """Route based on intent classification."""
    last_message = state["messages"][-1].content
    
    classification = model.invoke([
        SystemMessage(content="Classify as: refund, order_status, shipping, other"),
        HumanMessage(content=last_message)
    ])
    
    return classification.content.strip().lower()

builder = StateGraph(AgentState)
builder.add_node("greet", greet_customer)
builder.add_node("refund_flow", refund_subgraph)
builder.add_node("order_flow", order_subgraph)
builder.add_node("shipping_flow", shipping_subgraph)
builder.add_node("fallback", human_escalation)

builder.add_edge(START, "greet")
builder.add_conditional_edges("greet", router, {
    "refund": "refund_flow",
    "order_status": "order_flow",
    "shipping": "shipping_flow",
    "other": "fallback"
})
```

### Pattern 3: Multi-Turn with Memory

```python
# Using checkpointer for multi-turn conversations
graph = builder.compile(checkpointer=InMemorySaver())

# Turn 1
result = graph.invoke(
    {"messages": [HumanMessage(content="Hi, I need help")]},
    config={"configurable": {"thread_id": "session-1"}}
)

# Turn 2 - agent remembers turn 1
result = graph.invoke(
    {"messages": [HumanMessage(content="What's the status of my order?")]},
    config={"configurable": {"thread_id": "session-1"}}  # Same thread
)
```

---

## Async Support

LangGraph fully supports async operations:

```python
async def agent_node(state: AgentState):
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}

# Async invocation
result = await graph.ainvoke(input, config)

# Async streaming
async for event in graph.astream(input, config, stream_mode="updates"):
    print(event)
```

---

## Error Handling

```python
def safe_tool_node(state: AgentState):
    """Tool node with error handling."""
    try:
        # Execute tools
        results = execute_tools(state)
        return {"messages": results}
    except Exception as e:
        # Return error as a message so the agent can handle it
        error_msg = ToolMessage(
            content=f"Error: {str(e)}. Please try a different approach.",
            tool_call_id=state["messages"][-1].tool_calls[0]["id"]
        )
        return {"messages": [error_msg]}
```

---

## Application to Our Project

### Why LangGraph Is the Right Choice

1. **State management**: We need to track customer ID, order info, action type, approval status — TypedDict state handles this
2. **Conditional routing**: Different flows for refunds vs. order status vs. shipping — conditional edges handle this
3. **Persistence**: Multi-turn conversations across HTTP requests — checkpointers handle this
4. **Human-in-the-loop**: Approval for high-value refunds — `interrupt()` handles this
5. **Observability**: Need to see every decision — built-in tracing handles this

### Implementation Approach

```
START → classify_request → [conditional routing]
                              ├── handle_order_inquiry → respond → END
                              ├── handle_shipping → respond → END  
                              ├── handle_refund → [check amount]
                              │                     ├── auto_approve → process → respond → END
                              │                     └── request_approval → [interrupt] → process → respond → END
                              └── escalate → END
```

---

## Resources

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/) — Official docs
- [LangGraph Graph API Reference](https://docs.langchain.com/oss/python/langgraph/graph-api) — StateGraph, nodes, edges
- [LangGraph Tutorials](https://docs.langchain.com/oss/python/langgraph/tutorials) — Guided walkthroughs
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph) — Source code and examples
