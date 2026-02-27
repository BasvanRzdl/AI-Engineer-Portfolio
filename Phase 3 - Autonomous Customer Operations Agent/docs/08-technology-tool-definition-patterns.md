# Tool Definition Patterns

> **Type:** Technology Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## In My Own Words

Tools are how the agent **does things in the world** — look up orders, process refunds, check shipping status. The LLM decides *which* tool to call and *with what arguments*, but the tool implementation is regular Python code.

The quality of your tool definitions directly affects agent performance. Poorly defined tools → wrong tool calls → bad outcomes.

---

## How Tools Work in LangGraph

### The Flow

```
1. LLM receives messages + tool definitions (schemas)
2. LLM decides to call a tool → outputs a tool_call in its response
3. Your code executes the tool with the provided arguments
4. Result is added as a ToolMessage to the conversation
5. LLM receives the result and decides what to do next
```

The LLM never executes code — it only **requests** tool calls. Your application is the one that actually runs them.

---

## Defining Tools

### Method 1: The @tool Decorator (Recommended)

```python
from langchain_core.tools import tool

@tool
def order_lookup(order_id: str) -> dict:
    """Look up an order by its order ID.
    
    Use this tool when a customer asks about an order's status,
    details, or wants to take an action on a specific order.
    
    Args:
        order_id: The order identifier (e.g., "ORD-12345")
    
    Returns:
        A dictionary containing order details including status,
        items, total amount, and shipping information.
    """
    # Implementation
    order = database.get_order(order_id)
    if not order:
        return {"error": f"Order {order_id} not found"}
    return {
        "order_id": order.id,
        "status": order.status,
        "items": [item.to_dict() for item in order.items],
        "total": order.total,
        "shipping_status": order.shipping_status,
        "created_at": order.created_at.isoformat()
    }
```

**Critical:** The docstring becomes the tool's description that the LLM reads. Write it for the LLM, not for developers.

### Method 2: Pydantic Schema (For Complex Inputs)

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class RefundRequest(BaseModel):
    """Schema for processing a refund."""
    order_id: str = Field(description="The order ID to refund (e.g., 'ORD-12345')")
    reason: str = Field(description="The customer's reason for the refund")
    amount: float | None = Field(
        default=None, 
        description="Specific refund amount. If not provided, full order amount is used."
    )

@tool(args_schema=RefundRequest)
def process_refund(order_id: str, reason: str, amount: float | None = None) -> dict:
    """Process a refund for a customer's order.
    
    Use this tool ONLY after:
    1. Looking up the order to confirm it exists
    2. Verifying the customer's identity
    3. Confirming the refund with the customer
    
    Returns the refund confirmation with a reference number.
    """
    # Implementation
    order = database.get_order(order_id)
    if not order:
        return {"error": f"Order {order_id} not found"}
    
    refund_amount = amount or order.total
    refund = payment_service.process_refund(order.id, refund_amount, reason)
    
    return {
        "refund_id": refund.id,
        "amount": refund.amount,
        "status": "processed",
        "estimated_days": 5
    }
```

### Method 3: StructuredTool (For Dynamic Tool Creation)

```python
from langchain_core.tools import StructuredTool

def _shipping_lookup(tracking_number: str) -> dict:
    """Internal implementation."""
    return shipping_api.track(tracking_number)

shipping_tool = StructuredTool.from_function(
    func=_shipping_lookup,
    name="check_shipping_status",
    description="Check the shipping status of an order using its tracking number.",
)
```

---

## Tool Design Principles

### 1. Clear, Descriptive Names

```python
# ❌ Bad: Vague or too technical
@tool
def query(q: str) -> dict:
    """Run a query."""

# ✅ Good: Specific and action-oriented
@tool
def order_lookup(order_id: str) -> dict:
    """Look up an order by its order ID to get status, items, and shipping info."""
```

### 2. Excellent Descriptions

The LLM uses descriptions to decide when to call a tool. Include:
- **What** the tool does
- **When** to use it
- **What** it returns
- **Prerequisites** if any

```python
@tool
def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund for a customer's order.
    
    When to use: Customer explicitly requests a refund and you have
    confirmed the order exists and is eligible for a refund.
    
    Do NOT use this tool if:
    - You haven't looked up the order yet (use order_lookup first)
    - The refund amount would exceed $100 (escalate instead)
    - The order is still being shipped (offer alternatives first)
    
    Returns: Refund confirmation with reference number and estimated days.
    """
```

### 3. Typed Parameters with Descriptions

```python
class OrderLookupInput(BaseModel):
    order_id: str = Field(
        description="The order identifier, always starts with 'ORD-' followed by digits"
    )

@tool(args_schema=OrderLookupInput)
def order_lookup(order_id: str) -> dict:
    """Look up order details by order ID."""
```

### 4. Structured Return Values

```python
# ❌ Bad: Unstructured string
@tool
def order_lookup(order_id: str) -> str:
    order = db.get(order_id)
    return f"Order {order_id} is {order.status}, total ${order.total}"

# ✅ Good: Structured dict (LLM can extract specific fields)
@tool
def order_lookup(order_id: str) -> dict:
    """Look up order details."""
    order = db.get(order_id)
    return {
        "order_id": order_id,
        "status": order.status,
        "total": order.total,
        "items": order.items,
        "eligible_for_refund": order.is_refundable(),
        "shipping_tracking": order.tracking_number
    }
```

### 5. Error Handling

Never let tools crash — return error information the LLM can work with:

```python
@tool
def order_lookup(order_id: str) -> dict:
    """Look up order details."""
    # Validate input
    if not order_id.startswith("ORD-"):
        return {
            "error": "Invalid order ID format. Order IDs start with 'ORD-' followed by digits.",
            "example": "ORD-12345"
        }
    
    try:
        order = database.get_order(order_id)
    except DatabaseConnectionError:
        return {
            "error": "Unable to access the order system. Please try again shortly.",
            "retry": True
        }
    
    if not order:
        return {
            "error": f"No order found with ID {order_id}.",
            "suggestion": "Please ask the customer to verify the order ID."
        }
    
    return {
        "order_id": order.id,
        "status": order.status,
        # ...
    }
```

### 6. Single Responsibility

Each tool should do **one thing** well:

```python
# ❌ Bad: Tool does too much
@tool
def handle_order(action: str, order_id: str, amount: float = 0) -> dict:
    """Handle any order action: lookup, refund, cancel, modify."""

# ✅ Good: Separate tools for each action
@tool
def order_lookup(order_id: str) -> dict:
    """Look up order details."""

@tool
def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund for an order."""

@tool
def cancel_order(order_id: str, reason: str) -> dict:
    """Cancel a pending order."""
```

---

## Binding Tools to Models

```python
from langchain_openai import ChatOpenAI

# Create model
model = ChatOpenAI(model="gpt-4o")

# Define tools
tools = [order_lookup, process_refund, check_shipping, escalate_to_human]

# Bind tools to model
model_with_tools = model.bind_tools(tools)

# Now when you invoke the model, it can choose to call tools
response = model_with_tools.invoke([
    SystemMessage(content="You are a customer service agent."),
    HumanMessage(content="What's the status of order ORD-12345?")
])

# Check if model wants to call a tool
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call['name']}, Args: {call['args']}")
```

---

## Tool Execution in LangGraph

### Using the Prebuilt ToolNode

```python
from langgraph.prebuilt import ToolNode

# Create tool node that handles execution
tool_node = ToolNode(tools)

# Add to graph
builder.add_node("tools", tool_node)
```

The `ToolNode` automatically:
- Reads tool calls from the last AIMessage
- Executes each tool
- Returns ToolMessages with results

### Custom Tool Execution (For More Control)

```python
def execute_tools(state: AgentState) -> dict:
    """Custom tool execution with logging and validation."""
    last_message = state["messages"][-1]
    results = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        # Log the tool call
        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
        
        # Validate before execution
        if tool_name == "process_refund" and tool_args.get("amount", 0) > 100:
            results.append(ToolMessage(
                content="Error: Refunds over $100 require human approval. Please escalate.",
                tool_call_id=tool_call["id"]
            ))
            continue
        
        # Execute
        try:
            tool = tool_map[tool_name]
            result = tool.invoke(tool_args)
            results.append(ToolMessage(
                content=json.dumps(result) if isinstance(result, dict) else str(result),
                tool_call_id=tool_call["id"]
            ))
        except Exception as e:
            results.append(ToolMessage(
                content=f"Tool error: {str(e)}",
                tool_call_id=tool_call["id"]
            ))
    
    return {"messages": results}
```

---

## Tool Scoping

Not every tool should be available at every point in the conversation. Scope tools by context:

```python
# Different tools for different stages
greeting_tools = [customer_lookup]
investigation_tools = [order_lookup, check_shipping, check_account_history]
action_tools = [process_refund, cancel_order, update_shipping_address]
escalation_tools = [escalate_to_human, create_ticket]

# In your graph nodes, bind appropriate tools:
def investigation_node(state: AgentState):
    model_with_investigation_tools = model.bind_tools(investigation_tools)
    response = model_with_investigation_tools.invoke(state["messages"])
    return {"messages": [response]}

def action_node(state: AgentState):
    model_with_action_tools = model.bind_tools(action_tools)
    response = model_with_action_tools.invoke(state["messages"])
    return {"messages": [response]}
```

**Why scope tools?**
- Fewer tools → better tool selection accuracy
- Prevents the agent from taking actions before gathering info
- Enforces workflow ordering (look up order BEFORE processing refund)

---

## Tools for the Customer Operations Agent

### Planned Tool Ecosystem

| Tool | Purpose | Risk Level | Scoping |
|------|---------|------------|---------|
| `customer_lookup` | Find customer by ID or email | Low | All stages |
| `order_lookup` | Get order details | Low | Investigation+ |
| `check_shipping_status` | Track shipment | Low | Investigation+ |
| `check_return_eligibility` | Check if order can be returned | Low | Investigation+ |
| `process_refund` | Issue a refund | **High** | Action only |
| `update_shipping_address` | Change delivery address | Medium | Action only |
| `escalate_to_human` | Transfer to human agent | Low | All stages |

### Example: Complete Tool Implementation

```python
@tool
def customer_lookup(customer_id: str | None = None, email: str | None = None) -> dict:
    """Look up a customer's account information.
    
    Use this tool to find a customer by their ID or email address.
    You need at least one of: customer_id or email.
    
    Returns customer details including name, account status,
    and recent order history.
    """
    if not customer_id and not email:
        return {"error": "Provide either customer_id or email"}
    
    customer = db.find_customer(id=customer_id, email=email)
    if not customer:
        return {"error": "Customer not found", "suggestion": "Verify the ID or email"}
    
    return {
        "customer_id": customer.id,
        "name": customer.name,
        "email": customer.email,  # Masked: j***@example.com
        "account_status": customer.status,
        "member_since": customer.created_at.isoformat(),
        "recent_orders": [
            {"order_id": o.id, "date": o.date.isoformat(), "total": o.total}
            for o in customer.recent_orders(limit=5)
        ]
    }

@tool
def check_return_eligibility(order_id: str) -> dict:
    """Check if an order is eligible for return or refund.
    
    Use this BEFORE attempting to process a refund. This checks:
    - Whether the order is within the return window
    - Whether the items are eligible for return
    - What refund amount would apply
    
    Returns eligibility status and details.
    """
    order = db.get_order(order_id)
    if not order:
        return {"error": f"Order {order_id} not found"}
    
    eligibility = returns_service.check_eligibility(order)
    return {
        "eligible": eligibility.is_eligible,
        "reason": eligibility.reason,
        "refund_amount": eligibility.refund_amount,
        "return_window_days_remaining": eligibility.days_remaining,
        "refund_method": eligibility.refund_method,
        "restrictions": eligibility.restrictions
    }

@tool
def escalate_to_human(
    reason: str,
    priority: str = "normal",
    context_summary: str = ""
) -> dict:
    """Escalate the conversation to a human agent.
    
    Use this when:
    - The customer explicitly asks for a human
    - The issue is too complex for automated handling
    - A high-value action needs approval
    - The customer is frustrated or upset
    
    Args:
        reason: Why escalation is needed
        priority: "low", "normal", "high", or "urgent"
        context_summary: Brief summary of the conversation for the human agent
    """
    ticket = support_service.create_escalation(
        reason=reason,
        priority=priority,
        context=context_summary
    )
    return {
        "escalation_id": ticket.id,
        "estimated_wait": ticket.estimated_wait_minutes,
        "message": f"I've escalated this to our team (Reference: {ticket.id}). "
                   f"Expected wait time: {ticket.estimated_wait_minutes} minutes."
    }
```

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Too many tools | LLM gets confused, picks wrong tool | Scope tools to relevant graph nodes |
| Vague descriptions | LLM calls tool at wrong time | Write descriptions from LLM's perspective |
| String returns | LLM can't extract specific info | Return structured dicts |
| No error handling | Tool crashes → graph crashes | Return error info, never raise |
| Side effects in tools | Retries cause duplicate actions | Make tools idempotent or add confirmation |
| Sensitive args logged | PII in logs | Redact sensitive fields before logging |

---

## Resources

- [LangChain Tool Documentation](https://docs.langchain.com/oss/python/langchain/tools) — Tool definition reference
- [LangGraph ToolNode](https://docs.langchain.com/oss/python/langgraph/prebuilt#toolnode) — Prebuilt tool execution
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI's function calling reference
