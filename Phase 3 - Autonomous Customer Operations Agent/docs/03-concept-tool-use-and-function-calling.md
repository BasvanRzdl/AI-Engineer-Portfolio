# Tool Use and Function Calling in LLMs

> **Type:** Concept Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## In My Own Words

**Tool use** (also called **function calling**) is the mechanism by which an LLM can interact with the outside world. Instead of just generating text, the model can output a structured request to call a specific function with specific arguments. The application then executes the function and feeds the result back to the model.

This is what turns a chatbot into an **agent**: the ability to *do things* rather than just *say things*.

---

## Why This Matters

For the Customer Operations Agent, tool use is the foundation of everything operational:
- Looking up orders in a database
- Processing refunds through a payment system
- Updating shipping addresses
- Searching a knowledge base
- Escalating to a human operator

Without tools, the agent can only generate text about what it *would* do. With tools, it actually does it.

---

## How Function Calling Works

### The Protocol

1. **Define tools**: You describe available tools to the LLM as part of the system prompt or API call (name, description, parameters with types)
2. **LLM decides**: Given a user message, the LLM decides whether to call a tool (and which one) or respond directly
3. **Structured output**: If calling a tool, the LLM outputs the tool name and arguments as structured data (JSON)
4. **Execute**: Your application executes the actual function with the provided arguments
5. **Feed back**: The result is added to the conversation as a "tool message" and the LLM generates its next response

### The Flow

```
User: "What's the status of order #12345?"
  ↓
LLM thinks: "I need to look up this order"
LLM outputs: tool_call(name="order_lookup", args={"order_id": "12345"})
  ↓
Application executes: order_lookup("12345")  
  → Returns: {"status": "shipped", "tracking": "1Z999AA1..."}
  ↓
Tool result is fed back to LLM as a ToolMessage
  ↓
LLM responds: "Your order #12345 has shipped! Tracking number: 1Z999AA1..."
```

### In Code (LangChain/LangGraph)

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# 1. Define the tool
@tool
def order_lookup(order_id: str) -> dict:
    """Look up an order by its ID. Returns order details including status and tracking."""
    # In production, this would query a real database
    return db.get_order(order_id)

# 2. Bind tools to the LLM
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools([order_lookup])

# 3. LLM decides whether to use a tool
response = llm_with_tools.invoke("What's the status of order #12345?")

# 4. Check if the LLM wants to call a tool
if response.tool_calls:
    tool_call = response.tool_calls[0]
    # tool_call = {"name": "order_lookup", "args": {"order_id": "12345"}, "id": "call_abc123"}
    
    # 5. Execute the tool
    result = order_lookup.invoke(tool_call["args"])
    
    # 6. Feed result back as ToolMessage
    tool_message = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
```

---

## Tool Design Principles

### 1. Clear, Descriptive Names and Docstrings

The LLM uses the tool's **name** and **description** to decide when to use it. These must be clear and unambiguous.

```python
# ✅ Good: clear name, detailed description
@tool
def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """Process a refund for a customer order.
    
    Use this when a customer requests a refund. The refund will be applied
    to the original payment method. Refunds over $100 require manager approval.
    
    Args:
        order_id: The order ID (e.g., "ORD-12345")
        amount: The refund amount in USD
        reason: The reason for the refund (e.g., "defective item", "wrong size")
    
    Returns:
        Dictionary with refund_id, status, and estimated_processing_time
    """
    ...

# ❌ Bad: vague name, no description
@tool
def do_thing(data: dict) -> dict:
    """Process data."""
    ...
```

### 2. Typed Parameters with Descriptions

Use Pydantic models or typed arguments so the LLM knows exactly what to provide:

```python
from pydantic import BaseModel, Field

class RefundRequest(BaseModel):
    order_id: str = Field(description="The order ID to refund")
    amount: float = Field(description="Refund amount in USD", gt=0)
    reason: str = Field(description="Customer's reason for the refund")

@tool(args_schema=RefundRequest)
def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """Process a refund for a customer order."""
    ...
```

### 3. Return Structured, Useful Results

Tool results should give the LLM enough information to continue the conversation:

```python
# ✅ Good: structured, informative result
return {
    "status": "success",
    "refund_id": "REF-789",
    "amount": 49.99,
    "estimated_days": 5,
    "message": "Refund of $49.99 initiated for order ORD-12345"
}

# ❌ Bad: ambiguous result
return "done"
```

### 4. Handle Errors Gracefully

Tools should return error information rather than raising exceptions that crash the agent:

```python
@tool
def order_lookup(order_id: str) -> dict:
    """Look up an order by its ID."""
    try:
        order = db.get_order(order_id)
        if not order:
            return {"error": f"No order found with ID {order_id}"}
        return order
    except DatabaseError as e:
        return {"error": f"Unable to look up order: {str(e)}"}
```

### 5. One Tool, One Purpose

Each tool should do one thing well. Avoid "Swiss army knife" tools:

```python
# ✅ Good: focused tools
@tool
def lookup_order(order_id: str) -> dict: ...

@tool  
def process_refund(order_id: str, amount: float) -> dict: ...

@tool
def update_shipping_address(order_id: str, new_address: str) -> dict: ...

# ❌ Bad: one tool does everything
@tool
def manage_order(action: str, order_id: str, **kwargs) -> dict:
    """Do stuff with orders. Action can be 'lookup', 'refund', 'update'..."""
    ...
```

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| **LLM calls wrong tool** | Tool descriptions are ambiguous or overlapping | Make descriptions specific, test with edge cases |
| **Wrong arguments** | Parameter names/types unclear | Use typed parameters with descriptions, use Pydantic |
| **Infinite tool loops** | Agent keeps calling tools without making progress | Set max iterations, add "done" detection logic |
| **Tool errors crash agent** | Exceptions propagate up | Return error dicts instead of raising exceptions |
| **Missing context** | LLM doesn't have enough info to pick the right tool | Include examples in tool descriptions |
| **Too many tools** | LLM gets confused with 20+ tools | Group related tools, use routing to present subsets |

---

## Tool Calling in LangGraph

In LangGraph, the tool-calling pattern typically follows this structure:

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain.messages import ToolMessage

# Define tools
tools = [order_lookup, process_refund, update_shipping]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)

# Node: LLM decides what to do
def agent_node(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Node: Execute tool calls
def tool_node(state: MessagesState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": results}

# Edge: Should we continue or respond?
def should_continue(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tool_node", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["tool_node", END])
graph.add_edge("tool_node", "agent")  # After tool execution, back to agent
```

---

## Multiple Tool Calls

Modern LLMs can request **multiple tool calls in a single turn**. For example:

```
User: "Check my order #123 and also my order #456"

LLM outputs two tool calls:
  1. order_lookup(order_id="123")
  2. order_lookup(order_id="456")
```

The tool node should handle all pending calls:

```python
def tool_node(state: MessagesState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": results}
```

---

## Application to Our Project

### Tools to Build

| Tool | Purpose | Risk Level |
|------|---------|------------|
| `order_lookup` | Retrieve order details | Low (read-only) |
| `customer_lookup` | Get customer profile | Low (read-only) |
| `process_refund` | Initiate a refund | **High** (financial) |
| `update_shipping_address` | Change delivery address | Medium (modifying) |
| `search_knowledge_base` | Search FAQ/policy docs | Low (read-only) |
| `escalate_to_human` | Transfer to human agent | Low (routing) |
| `check_refund_eligibility` | Verify refund policy | Low (read-only) |

### Error Handling Strategy

```python
@tool
def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """Process a refund for a customer order."""
    # Validate inputs
    if amount <= 0:
        return {"error": "Refund amount must be positive"}
    if amount > 1000:
        return {"error": "Refunds over $1000 require manual processing", 
                "action": "escalate"}
    
    # Check order exists
    order = db.get_order(order_id)
    if not order:
        return {"error": f"Order {order_id} not found"}
    
    # Check eligibility
    if not order.is_refundable():
        return {"error": "This order is not eligible for refund",
                "reason": order.refund_ineligibility_reason}
    
    # Process
    try:
        refund = payment_service.refund(order_id, amount)
        return {
            "status": "success",
            "refund_id": refund.id,
            "amount": amount,
            "estimated_days": 5
        }
    except PaymentError as e:
        return {"error": f"Payment processing failed: {e}",
                "action": "retry_or_escalate"}
```

---

## Resources

- [LangChain Tool Calling Guide](https://docs.langchain.com/oss/python/langchain/tools) — How to define and use tools
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling) — The API-level protocol
- [LangGraph Agent Tutorial](https://docs.langchain.com/oss/python/langgraph/workflows-agents#agents) — Agent with tools in LangGraph
