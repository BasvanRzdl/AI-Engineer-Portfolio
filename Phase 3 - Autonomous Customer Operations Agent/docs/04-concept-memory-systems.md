# Memory Systems for Agents

> **Type:** Concept Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## In My Own Words

**Memory** is what allows an agent to be coherent over time. Without memory, every turn of a conversation is a blank slate — the agent has no idea what was said before, what actions were taken, or who the customer is. 

Memory in AI agents parallels human memory types:
- **Short-term memory**: What's happening right now in this conversation
- **Long-term memory**: What we know across all conversations (user preferences, past interactions)

The challenge isn't *whether* to have memory — it's *how much*, *what kind*, and *how to manage it* given the finite context window of LLMs.

---

## Why This Matters

For the Customer Operations Agent:
- The agent must remember what the customer said earlier in the conversation
- It should know about past interactions ("You called about this order last week")
- It needs to track what actions it has already taken
- Context windows are limited — we can't just dump everything in

---

## Memory Types (Mapped from Psychology to AI)

### 1. Short-Term Memory (Within a Session)

**What it is:** The conversation history and accumulated state within a single interaction session (a "thread").

**In practice:** This is the list of messages — human messages, AI responses, tool calls, and tool results — that grows as the conversation progresses.

**Managed by:** LangGraph's state persistence (checkpointers). Each thread has its own state that is saved at every super-step.

**Challenge:** Conversations can get long. A 50-turn support conversation with tool calls can easily exceed context limits.

### 2. Semantic Memory (Facts About the World)

**What it is:** Stored facts and knowledge — things the agent "knows" about users, products, policies, etc.

**In practice:** Customer profiles, preference data, facts learned from past interactions.

**Examples for our agent:**
- "Customer prefers email communication"
- "Customer has a VIP loyalty status"
- "Customer had a bad experience with shipping last month"

**Managed by:** LangGraph's `Store` (cross-thread memory). Stored as key-value documents in namespaced collections.

### 3. Episodic Memory (Past Experiences)

**What it is:** Memories of specific past events — what happened in previous conversations.

**In practice:** Summaries of past support interactions, previous complaints, resolution history.

**Examples for our agent:**
- "On Jan 15, customer requested a refund for order #789 which was processed successfully"
- "Customer previously escalated a late delivery issue"

**Managed by:** Either the `Store` or a separate database of interaction summaries.

### 4. Procedural Memory (How to Do Things)

**What it is:** Knowledge of how to perform tasks — the agent's "skills."

**In practice:** This is essentially the agent's prompt, code, and graph structure. It's rarely modified at runtime, but the concept matters:
- System prompts define behavior
- The graph structure encodes procedures
- Tool definitions encode capabilities

---

## Short-Term Memory: Strategies for Managing Conversation History

Since LLM context windows are finite, you need strategies to manage growing conversations:

### Strategy 1: Full History (Simple but Limited)

Keep the entire conversation in the `messages` list.

```python
class State(MessagesState):
    pass  # messages accumulate via add_messages reducer
```

**Pros:** Complete context, no information loss  
**Cons:** Hits context limits on long conversations, higher cost

**Use when:** Conversations are short (< 20 turns)

### Strategy 2: Sliding Window (Truncation)

Keep only the last N messages, discarding older ones.

```python
def trim_messages(state: State):
    """Keep only the last 20 messages."""
    messages = state["messages"]
    if len(messages) > 20:
        # Keep system message + last 19 messages
        trimmed = [messages[0]] + messages[-19:]
        return {"messages": trimmed}
    return {}
```

**Pros:** Bounded context size, predictable costs  
**Cons:** Loses important earlier context

**Use when:** Most relevant info is recent, older context is less important

### Strategy 3: Summarization

Periodically summarize older messages into a condensed form.

```python
def summarize_conversation(state: State):
    """Summarize older messages to save context space."""
    messages = state["messages"]
    if len(messages) > 30:
        # Summarize everything except the last 10 messages
        old_messages = messages[:-10]
        summary = llm.invoke(
            f"Summarize this conversation so far:\n{format_messages(old_messages)}"
        )
        # Replace old messages with summary + keep recent
        return {
            "messages": [
                SystemMessage(content=f"Conversation summary: {summary.content}"),
                *messages[-10:]
            ]
        }
    return {}
```

**Pros:** Retains key context, bounded size  
**Cons:** Summary may lose details, extra LLM call

**Use when:** Conversations are long but earlier context still matters

### Strategy 4: Relevant Context Injection

Instead of keeping everything, pull in only what's relevant to the current question.

```python
def inject_relevant_context(state: State):
    """Pull relevant past context based on current message."""
    current_question = state["messages"][-1].content
    
    # Search past memories for relevant context
    relevant = memory_store.search(
        namespace=("customer", state["customer_id"]),
        query=current_question,
        limit=5
    )
    
    context = "\n".join([m.value["content"] for m in relevant])
    return {
        "messages": [
            SystemMessage(content=f"Relevant context from past interactions:\n{context}")
        ]
    }
```

**Pros:** Always relevant, bounded size  
**Cons:** May miss context not semantically related to current query

**Use when:** Long history across sessions, need cross-session recall

---

## Long-Term Memory: Persisting Across Sessions

### LangGraph Store

LangGraph provides a `Store` interface for cross-thread memory:

```python
from langgraph.store.memory import InMemoryStore

# Create a store (use PostgresStore in production)
store = InMemoryStore()

# Store a memory
store.put(
    namespace=("customer", "cust-123", "preferences"),
    key="communication",
    value={"preference": "email", "language": "english"}
)

# Retrieve memories
memories = store.search(
    namespace=("customer", "cust-123", "preferences")
)

# Semantic search (requires embedding configuration)
relevant = store.search(
    namespace=("customer", "cust-123", "interactions"),
    query="refund experience",
    limit=3
)
```

### Using Store in LangGraph Nodes

```python
from langgraph.runtime import Runtime
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    """Model node that uses long-term memory."""
    user_id = runtime.context.user_id
    namespace = (user_id, "memories")
    
    # Search for relevant memories
    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=5
    )
    
    memory_context = "\n".join([m.value["memory"] for m in memories])
    
    messages = [
        SystemMessage(content=f"Known about this customer:\n{memory_context}"),
        *state["messages"]
    ]
    
    response = await llm.ainvoke(messages)
    return {"messages": [response]}
```

### Memory Writing Strategies

#### In the Hot Path (During Conversation)

Write memories as part of the agent's regular processing. The agent decides what's worth remembering.

```python
async def update_memory(state: MessagesState, runtime: Runtime[Context]):
    """Extract and save important facts from the conversation."""
    user_id = runtime.context.user_id
    
    # Ask the LLM to extract memorable facts
    extraction_prompt = """Analyze this conversation and extract important facts 
    about the customer (preferences, complaints, key decisions). 
    Return as a JSON list of facts."""
    
    facts = await llm.ainvoke([
        SystemMessage(content=extraction_prompt),
        *state["messages"][-10:]  # Recent messages
    ])
    
    # Save each fact
    for fact in parse_facts(facts.content):
        await runtime.store.aput(
            (user_id, "memories"),
            str(uuid4()),
            {"memory": fact, "timestamp": datetime.now().isoformat()}
        )
```

**Pros:** Real-time, immediate availability  
**Cons:** Adds latency, agent multitasks between memory and conversation

#### In the Background (After Conversation)

Process the conversation after it ends to extract memories.

**Pros:** No latency impact, more thorough extraction  
**Cons:** Memories not available until processing completes

---

## Memory Approaches Compared

| Approach | Scope | Persistence | Best For |
|----------|-------|-------------|----------|
| **Message history** | Single thread | Via checkpointer | Current conversation |
| **Sliding window** | Single thread | N/A | Long conversations with recent focus |
| **Summarization** | Single thread | Via state | Long conversations needing full context |
| **Store (profile)** | Cross-thread | Via Store | User preferences, facts |
| **Store (collection)** | Cross-thread | Via Store | Interaction history, episodic memory |
| **Vector search** | Cross-thread | Via Store + embeddings | Finding relevant past context |

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| **Context overflow** | Too many messages in state | Implement trimming or summarization |
| **Stale memories** | Old facts never updated | Include timestamps, implement memory refresh |
| **Memory conflicts** | Different memories contradict | Use upsert patterns, resolve conflicts explicitly |
| **Over-memorizing** | Storing everything increases noise | Be selective — only store actionable facts |
| **Under-memorizing** | Not saving enough context | Define clear criteria for what to remember |

---

## Application to Our Project

### Memory Architecture

```
┌─────────────────────────────────────────────────┐
│  Short-Term (Thread State)                       │
│  ├── Conversation messages (via checkpointer)   │
│  ├── Current request context                     │
│  └── Actions taken this session                  │
├─────────────────────────────────────────────────┤
│  Long-Term (Store)                               │
│  ├── Customer profile (preferences, status)      │
│  ├── Interaction history (past conversations)    │
│  └── Resolution patterns (what worked before)    │
└─────────────────────────────────────────────────┘
```

### Decisions to Make
- [ ] Which memory management strategy for short-term? (Start with full history, add summarization if needed)
- [ ] What facts to store in long-term memory?
- [ ] When to write memories? (hot path for critical facts, background for summaries)
- [ ] Use semantic search for memory retrieval?

---

## Resources

- [LangGraph Memory Concepts](https://docs.langchain.com/oss/python/langgraph/memory) — Official conceptual guide
- [LangGraph Persistence](https://docs.langchain.com/oss/python/langgraph/persistence) — Checkpointers and stores
- [CoALA Paper](https://arxiv.org/pdf/2309.02427) — Academic framework for agent memory types
- [Memory Agent Template](https://github.com/langchain-ai/memory-agent) — Reference implementation
