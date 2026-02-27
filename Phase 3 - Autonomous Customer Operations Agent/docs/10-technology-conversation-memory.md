# Conversation Memory Implementations

> **Type:** Technology Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## In My Own Words

Document 04 covered memory *concepts*. This document covers *implementation* — how to actually build memory into our LangGraph agent using checkpointers and the Store API.

Two levels of memory:
1. **Short-term (thread memory)** — The conversation history within a single session. Handled by **checkpointers**.
2. **Long-term (cross-thread memory)** — Customer preferences, past interactions, learned patterns. Handled by **Store**.

---

## Short-Term Memory: Checkpointers

### How It Works

Every time you call `graph.invoke()` with a `thread_id`, LangGraph saves the state. Next call with the same `thread_id` picks up where it left off.

```python
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver

# Build a simple chat graph
def chat(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# Compile with checkpointer
graph = builder.compile(checkpointer=InMemorySaver())

# Conversation turn 1
config = {"configurable": {"thread_id": "session-abc"}}
graph.invoke({"messages": [HumanMessage(content="Hi, I'm Alice")]}, config)

# Conversation turn 2 — agent remembers turn 1
graph.invoke({"messages": [HumanMessage(content="What's my name?")]}, config)
# Agent responds: "Your name is Alice!"

# Different thread = different conversation
config2 = {"configurable": {"thread_id": "session-xyz"}}
graph.invoke({"messages": [HumanMessage(content="What's my name?")]}, config2)
# Agent responds: "I don't know your name yet."
```

### Production Checkpointers

```python
# Development: In-memory (lost on restart)
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# Production: PostgreSQL (persistent, scalable)
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

conn_string = "postgresql://user:pass@localhost:5432/agent_db"
with psycopg.connect(conn_string) as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # Creates required tables

# Alternative: SQLite (persistent, single server)
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("agent_checkpoints.db")
```

### Managing Conversation Length

The chat history grows with every turn. Strategies to manage this:

#### Strategy 1: Sliding Window

Keep only the last N messages:

```python
from langchain_core.messages import trim_messages

def chat_with_trim(state: MessagesState):
    # Keep only last 20 messages (plus system message)
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,  # Or use token count
        strategy="last",
        token_counter=model,  # Uses model's tokenizer
        include_system=True,
        allow_partial=False,
    )
    
    response = model.invoke(trimmed)
    return {"messages": [response]}
```

#### Strategy 2: Summarization

Periodically summarize older messages:

```python
from langchain_core.messages import RemoveMessage

def maybe_summarize(state: AgentState):
    """Summarize if conversation gets too long."""
    messages = state["messages"]
    
    if len(messages) <= 10:
        return {}  # No need to summarize yet
    
    # Summarize older messages
    summary_prompt = f"""Summarize the key points from this conversation so far:
    {format_messages(messages[:-4])}
    
    Include: customer identity, their issue, actions taken, current status."""
    
    summary = model.invoke([HumanMessage(content=summary_prompt)])
    
    # Delete old messages and replace with summary
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-4]]
    
    return {
        "messages": delete_messages + [
            SystemMessage(content=f"Conversation summary: {summary.content}")
        ] + messages[-4:]  # Keep last 4 messages
    }
```

#### Strategy 3: Relevant Context Injection

Instead of keeping all messages, inject only relevant context:

```python
def chat_with_context(state: AgentState):
    """Inject relevant context instead of full history."""
    system = SystemMessage(content=f"""You are a customer service agent.
    
Customer: {state.get('customer_name', 'Unknown')}
Issue: {state.get('issue_summary', 'Not yet identified')}
Actions taken: {state.get('actions_taken', 'None')}
Current status: {state.get('status', 'investigating')}""")
    
    # Only include last few messages for immediate context
    recent = state["messages"][-6:]
    
    response = model.invoke([system] + recent)
    return {"messages": [response]}
```

---

## Long-Term Memory: The Store API

### What the Store Does

The Store is a key-value database that persists across conversations. It uses **namespaces** to organize data.

```python
from langgraph.store.memory import InMemoryStore

# Create store
store = InMemoryStore()

# Compile graph with both checkpointer AND store
graph = builder.compile(
    checkpointer=InMemorySaver(),
    store=store
)
```

### Namespace Design

Namespaces organize data hierarchically:

```python
# Pattern: (scope, entity_id, data_type)
("customers", "cust-123", "preferences")    # Customer preferences
("customers", "cust-123", "history")         # Interaction history
("customers", "cust-123", "notes")           # Agent notes about customer
("agents", "global", "policies")             # Business policies
("agents", "global", "common_issues")        # Known issue patterns
```

### Reading from Store in Nodes

```python
from langgraph.config import get_store

def personalized_greeting(state: AgentState):
    """Greet customer using their preferences from long-term memory."""
    store = get_store()  # Access the store from within a node
    customer_id = state.get("customer_id")
    
    if customer_id:
        # Read customer preferences
        namespace = ("customers", customer_id, "preferences")
        items = store.search(namespace)
        
        preferences = {}
        for item in items:
            preferences.update(item.value)
        
        greeting = f"Welcome back! "
        if preferences.get("preferred_name"):
            greeting += f"Good to see you, {preferences['preferred_name']}. "
        if preferences.get("last_issue"):
            greeting += f"Is this about your previous {preferences['last_issue']} issue, or something new?"
    else:
        greeting = "Hello! How can I help you today?"
    
    return {"messages": [AIMessage(content=greeting)]}
```

### Writing to Store in Nodes

```python
from langgraph.config import get_store
import uuid

def save_interaction_summary(state: AgentState):
    """Save a summary of this interaction to long-term memory."""
    store = get_store()
    customer_id = state.get("customer_id")
    
    if customer_id:
        # Save interaction summary
        store.put(
            namespace=("customers", customer_id, "history"),
            key=str(uuid.uuid4()),
            value={
                "date": datetime.now().isoformat(),
                "issue_type": state.get("action_type"),
                "resolution": state.get("resolution"),
                "satisfaction": state.get("satisfaction_score"),
                "summary": state.get("conversation_summary")
            }
        )
        
        # Update preferences if we learned something new
        if state.get("learned_preference"):
            store.put(
                namespace=("customers", customer_id, "preferences"),
                key="main",
                value={
                    "preferred_name": state.get("preferred_name"),
                    "communication_style": state.get("communication_style"),
                    "last_issue": state.get("action_type"),
                    "last_interaction": datetime.now().isoformat()
                }
            )
    
    return {}  # No state changes needed
```

### Semantic Search in Store

Find relevant memories using vector similarity:

```python
from langchain_openai import OpenAIEmbeddings

# Create store with embedding support
store = InMemoryStore(
    index={
        "embed": OpenAIEmbeddings(model="text-embedding-3-small"),
        "dims": 1536,
        "fields": ["summary", "issue_type"]  # Fields to embed
    }
)

# Later, search for relevant past interactions
def recall_similar_issues(state: AgentState):
    """Find similar past issues for this customer."""
    store = get_store()
    customer_id = state["customer_id"]
    current_issue = state["messages"][-1].content
    
    # Semantic search across customer's history
    similar = store.search(
        namespace=("customers", customer_id, "history"),
        query=current_issue,
        limit=3
    )
    
    if similar:
        context = "Previous similar interactions:\n"
        for item in similar:
            context += f"- {item.value['date']}: {item.value['summary']} (resolved: {item.value['resolution']})\n"
        
        return {"context": context}
    
    return {}
```

---

## Memory Architecture for Our Agent

### Proposed Design

```
┌─────────────────────────────────────────────┐
│              Memory Architecture             │
├─────────────────────────────────────────────┤
│                                             │
│  SHORT-TERM (Checkpointer)                  │
│  ├── Current conversation messages          │
│  ├── Current order context                  │
│  ├── In-progress action state               │
│  └── Thread: per customer session           │
│                                             │
│  LONG-TERM (Store)                          │
│  ├── Customer Preferences                   │
│  │   └── (customers, {id}, preferences)     │
│  ├── Interaction History                    │
│  │   └── (customers, {id}, history)         │
│  ├── Agent Notes                            │
│  │   └── (customers, {id}, notes)           │
│  └── Business Knowledge                     │
│      └── (agents, global, policies)         │
│                                             │
└─────────────────────────────────────────────┘
```

### State Schema with Memory

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Short-term: conversation
    messages: Annotated[list, add_messages]
    
    # Short-term: current context
    customer_id: str | None
    order_info: dict | None
    action_type: str | None
    
    # Injected from long-term memory
    customer_preferences: dict | None
    past_interactions: list[dict] | None
    
    # Workflow tracking
    steps_taken: int
    needs_escalation: bool
```

### Memory Loading Pattern

```python
def load_customer_memory(state: AgentState):
    """Load relevant long-term memory at the start of a conversation."""
    store = get_store()
    customer_id = state.get("customer_id")
    
    if not customer_id:
        return {}
    
    # Load preferences
    prefs_items = store.search(("customers", customer_id, "preferences"))
    preferences = prefs_items[0].value if prefs_items else {}
    
    # Load recent history
    history_items = store.search(
        ("customers", customer_id, "history"),
        limit=5
    )
    past_interactions = [item.value for item in history_items]
    
    return {
        "customer_preferences": preferences,
        "past_interactions": past_interactions
    }
```

### Memory Saving Pattern

```python
def save_conversation_memory(state: AgentState):
    """Save conversation insights to long-term memory at the end."""
    store = get_store()
    customer_id = state.get("customer_id")
    
    if not customer_id:
        return {}
    
    # Summarize the conversation
    summary = model.invoke([
        SystemMessage(content="Summarize this customer interaction in 2-3 sentences. "
                     "Include: the issue, resolution, and any customer preferences noted."),
        *state["messages"][-10:]  # Last 10 messages for context
    ])
    
    # Save to history
    store.put(
        namespace=("customers", customer_id, "history"),
        key=str(uuid.uuid4()),
        value={
            "date": datetime.now().isoformat(),
            "issue_type": state.get("action_type", "unknown"),
            "resolution": state.get("resolution", "unknown"),
            "summary": summary.content,
            "turns": state.get("steps_taken", 0)
        }
    )
    
    return {}
```

---

## Production Considerations

### Store Backend Options

```python
# Development
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

# Production: PostgreSQL
from langgraph.store.postgres import PostgresStore
store = PostgresStore.from_conn_string(
    "postgresql://user:pass@localhost:5432/agent_db"
)
```

### Memory Cleanup

Over time, memory accumulates. Implement cleanup strategies:

```python
def cleanup_old_memories(customer_id: str, max_history: int = 50):
    """Keep only the most recent interactions."""
    store = get_store()
    namespace = ("customers", customer_id, "history")
    
    all_items = store.search(namespace, limit=1000)
    
    if len(all_items) > max_history:
        # Sort by date, keep newest
        sorted_items = sorted(all_items, key=lambda x: x.value["date"], reverse=True)
        to_delete = sorted_items[max_history:]
        
        for item in to_delete:
            store.delete(namespace, item.key)
```

### Privacy and PII

```python
def sanitize_for_storage(data: dict) -> dict:
    """Remove PII before storing in long-term memory."""
    sanitized = data.copy()
    
    # Remove sensitive fields
    sensitive_fields = ["email", "phone", "address", "payment_method", "ssn"]
    for field in sensitive_fields:
        sanitized.pop(field, None)
    
    return sanitized
```

---

## Testing Memory

```python
def test_short_term_memory():
    """Verify conversation context persists within a thread."""
    config = {"configurable": {"thread_id": "test-memory-1"}}
    
    # Turn 1: Introduce
    graph.invoke(
        {"messages": [HumanMessage(content="My name is Alice and my order is ORD-123")]},
        config
    )
    
    # Turn 2: Reference previous context
    result = graph.invoke(
        {"messages": [HumanMessage(content="What order did I mention?")]},
        config
    )
    
    assert "ORD-123" in result["messages"][-1].content

def test_long_term_memory():
    """Verify customer preferences persist across sessions."""
    # Session 1: Learn preference
    config1 = {"configurable": {"thread_id": "session-1"}}
    graph.invoke(
        {"messages": [HumanMessage(content="I prefer to be called Bob")], 
         "customer_id": "cust-test"},
        config1
    )
    
    # Session 2: Different thread, same customer
    config2 = {"configurable": {"thread_id": "session-2"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="Hi")],
         "customer_id": "cust-test"},
        config2
    )
    
    # Should reference stored preference
    assert "Bob" in result["messages"][-1].content
```

---

## Resources

- [LangGraph Persistence Guide](https://docs.langchain.com/oss/python/langgraph/concepts/persistence) — Checkpointers and state management
- [LangGraph Memory Concepts](https://docs.langchain.com/oss/python/langgraph/concepts/memory) — Short and long-term memory
- [LangGraph Store API](https://docs.langchain.com/oss/python/langgraph/reference/store) — Store reference
- [Message Trimming](https://docs.langchain.com/oss/python/langchain/how-tos/trim-messages) — Managing conversation length
