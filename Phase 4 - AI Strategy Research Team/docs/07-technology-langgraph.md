---
date: 2026-02-27
type: technology
topic: "LangGraph — Deep Dive"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Technology: LangGraph

## What It Is

LangGraph is a low-level orchestration framework for building stateful, multi-agent applications as graphs. Built by the LangChain team, it lets you define agents and workflows as **nodes** (functions) connected by **edges** (routing logic), with shared **state** flowing through the graph.

It's the **required framework** for your Phase 4 project.

**Key philosophy:** LangGraph gives you fine-grained control. It doesn't abstract away complexity — it gives you primitives to build exactly the workflow you need. Think of it as "React for agents" rather than "Wix for agents."

## Core Concepts

### StateGraph

The fundamental building block. A graph where:
- **State** flows through the graph and is updated by nodes
- **Nodes** are Python functions that read state and return updates
- **Edges** define the routing between nodes

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class MyState(TypedDict):
    topic: str
    research: str
    analysis: str
    draft: str

graph = StateGraph(MyState)
graph.add_node("researcher", researcher_fn)
graph.add_node("analyst", analyst_fn)
graph.add_node("writer", writer_fn)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "analyst")
graph.add_edge("analyst", "writer")
graph.add_edge("writer", END)

app = graph.compile()
result = app.invoke({"topic": "AI adoption strategies"})
```

### State

State is a shared data structure (TypedDict or Pydantic model) that all nodes can read from and write to. It's the "memory" of the workflow.

```python
from typing import TypedDict, Annotated
from operator import add

class ResearchState(TypedDict):
    topic: str                                    # Replace: latest value wins
    messages: Annotated[list[str], add]           # Append: new items added to list
    research_results: Annotated[list[dict], add]  # Append
    draft: str                                    # Replace
    iteration: int                                # Replace
    status: str                                   # Replace
```

**Reducers** define how state updates are merged:
- **Default (replace):** New value overwrites old value
- **Annotated with `add`:** New values are appended to the list
- **Custom reducer:** Your own merge function

### Nodes

Nodes are Python functions that:
1. Receive the current state
2. Do work (call an LLM, run a tool, process data)
3. Return a partial state update (only the keys that changed)

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

def researcher_node(state: ResearchState) -> dict:
    """Research the given topic."""
    response = llm.invoke(f"Research the following topic: {state['topic']}")
    return {
        "research_results": [{"content": response.content, "source": "llm"}],
        "status": "researched"
    }

def analyst_node(state: ResearchState) -> dict:
    """Analyze research results."""
    research = "\n".join([r["content"] for r in state["research_results"]])
    response = llm.invoke(f"Analyze these findings:\n{research}")
    return {
        "analysis": response.content,
        "status": "analyzed"
    }
```

**Key:** Nodes return **partial updates**, not the full state. LangGraph merges updates using reducers.

### Edges

#### Normal Edges

Fixed routing — always go from A to B:
```python
graph.add_edge("researcher", "analyst")
```

#### Conditional Edges

Dynamic routing based on state:
```python
def route_after_critic(state: ResearchState) -> str:
    if state["critic_decision"] == "APPROVE":
        return "end"
    if state["iteration"] >= 3:
        return "end"
    return "writer"

graph.add_conditional_edges("critic", route_after_critic, {
    "writer": "writer",
    "end": END
})
```

#### START and END

Special nodes:
- `START` — entry point of the graph
- `END` — exit point (graph returns final state)

### Command

`Command` lets a node update state AND route to the next node in one return:

```python
from langgraph.types import Command

def orchestrator(state):
    plan = create_plan(state["topic"])
    return Command(
        update={"plan": plan, "status": "planned"},
        goto="researcher"  # Explicit routing
    )
```

### Send (Dynamic Fan-Out)

`Send` creates parallel tasks dynamically at runtime. This is how you implement the orchestrator-worker pattern:

```python
from langgraph.types import Send

def orchestrator(state):
    """Create dynamic parallel research tasks."""
    subtasks = decompose_topic(state["topic"])
    
    # Each Send creates a parallel execution of the target node
    return [
        Send("researcher", {"subtask": task, "parent_topic": state["topic"]})
        for task in subtasks
    ]

# The researcher node runs once per Send
graph.add_conditional_edges("orchestrator", orchestrator)
```

**How results are collected:** Each researcher node's output updates the shared state. With an append reducer on `research_results`, all results accumulate automatically.

### Checkpointing (Persistence)

Checkpointing saves the graph's state after each node execution. This enables:
- **Resuming after failures** — pick up where you left off
- **Human-in-the-loop** — pause, let human inspect/modify, resume
- **Time travel** — go back to any previous state

```python
from langgraph.checkpoint.memory import MemorySaver
# For production: from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# Each thread has its own state history
config = {"configurable": {"thread_id": "research-001"}}
result = app.invoke({"topic": "AI adoption"}, config)

# Get state at any point
state = app.get_state(config)

# List all checkpoints
for cp in app.get_state_history(config):
    print(cp.created_at, cp.next)  # timestamp + next node to execute
```

### Human-in-the-Loop with `interrupt()`

Pause execution and wait for human input:

```python
from langgraph.types import interrupt, Command

def review_node(state):
    """Pause for human review."""
    human_input = interrupt({
        "draft": state["draft"],
        "question": "Approve this draft? (yes/no/edit)"
    })
    return {"human_feedback": human_input}

# In your application:
# 1. Graph pauses at review_node
# 2. Get interrupted state: app.get_state(config)
# 3. Human makes decision
# 4. Resume: app.invoke(Command(resume="yes"), config)
```

### MessagesState

A convenience state schema for chat-based agents:

```python
from langgraph.graph import MessagesState

# Equivalent to:
# class MessagesState(TypedDict):
#     messages: Annotated[list[BaseMessage], add_messages]

# The add_messages reducer handles message deduplication,
# updates by ID, and proper message ordering
```

### Subgraphs

Break complex graphs into reusable sub-components:

```python
# Define a subgraph for the research phase
research_graph = StateGraph(ResearchSubState)
research_graph.add_node("search", search_fn)
research_graph.add_node("summarize", summarize_fn)
research_graph.add_edge(START, "search")
research_graph.add_edge("search", "summarize")
research_graph.add_edge("summarize", END)
compiled_research = research_graph.compile()

# Use it as a node in the parent graph
main_graph = StateGraph(MainState)
main_graph.add_node("research", compiled_research)
main_graph.add_node("analysis", analyst_fn)
main_graph.add_edge(START, "research")
main_graph.add_edge("research", "analysis")
```

## Workflow Patterns (With Code)

### Pattern 1: Sequential Pipeline

```python
graph = StateGraph(State)
graph.add_node("research", research_fn)
graph.add_node("analyze", analyze_fn)
graph.add_node("write", write_fn)

graph.add_edge(START, "research")
graph.add_edge("research", "analyze")
graph.add_edge("analyze", "write")
graph.add_edge("write", END)
```

### Pattern 2: Routing

```python
def route_by_topic(state):
    if "technical" in state["topic"].lower():
        return "technical_researcher"
    elif "market" in state["topic"].lower():
        return "market_researcher"
    else:
        return "general_researcher"

graph.add_conditional_edges(START, route_by_topic)
```

### Pattern 3: Parallelization

```python
# Static parallelism: Multiple nodes from same source
graph.add_edge(START, "researcher_a")
graph.add_edge(START, "researcher_b")
graph.add_edge(START, "researcher_c")
graph.add_edge(["researcher_a", "researcher_b", "researcher_c"], "synthesizer")

# Dynamic parallelism: Use Send
def fan_out(state):
    return [Send("researcher", {"query": q}) for q in state["queries"]]
graph.add_conditional_edges("planner", fan_out)
```

### Pattern 4: Evaluator-Optimizer Loop

```python
def should_revise(state):
    if state["approved"]:
        return "end"
    if state["iteration"] >= 3:
        return "end"
    return "revise"

graph.add_node("writer", writer_fn)
graph.add_node("critic", critic_fn)

graph.add_edge(START, "writer")
graph.add_edge("writer", "critic")
graph.add_conditional_edges("critic", should_revise, {
    "revise": "writer",
    "end": END
})
```

### Pattern 5: Orchestrator-Worker (Your Main Pattern)

```python
from langgraph.types import Send

def orchestrator(state):
    """Plan research and dispatch to workers."""
    plan = llm.invoke(f"Create 3 research subtasks for: {state['topic']}")
    subtasks = parse_plan(plan)
    return [Send("researcher", {"subtask": t}) for t in subtasks]

def researcher(state):
    """Execute a single research subtask."""
    result = llm.invoke(f"Research: {state['subtask']}")
    return {"research_results": [result.content]}

def synthesizer(state):
    """Combine all research results."""
    combined = "\n\n".join(state["research_results"])
    summary = llm.invoke(f"Synthesize these findings:\n{combined}")
    return {"synthesis": summary.content}

graph = StateGraph(ResearchState)
graph.add_node("orchestrator", orchestrator)
graph.add_node("researcher", researcher)
graph.add_node("synthesizer", synthesizer)

graph.add_conditional_edges("orchestrator", orchestrator)  # Send to researchers
graph.add_edge("researcher", "synthesizer")
graph.add_edge(START, "orchestrator")
graph.add_edge("synthesizer", END)
```

## Configuration & Runtime

### Runtime Configuration

Pass configuration that isn't part of the state:

```python
from langchain_core.runnables import RunnableConfig

def researcher(state, config: RunnableConfig):
    model_name = config["configurable"].get("model", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name)
    # ...

# Pass at runtime
result = app.invoke(
    {"topic": "AI adoption"},
    config={"configurable": {"model": "gpt-4o", "thread_id": "123"}}
)
```

### Recursion Limit

Safety net to prevent infinite loops:

```python
# Default is 25 steps
result = app.invoke(input, config={"recursion_limit": 50})
```

### Streaming

Stream outputs as they're produced:

```python
# Stream state updates
for event in app.stream({"topic": "AI"}, stream_mode="updates"):
    print(event)

# Stream values (full state after each step)
for state in app.stream({"topic": "AI"}, stream_mode="values"):
    print(state)

# Stream debug info
for event in app.stream({"topic": "AI"}, stream_mode="debug"):
    print(event)
```

### Visualization

```python
# Generate a Mermaid diagram of your graph
print(app.get_graph().draw_mermaid())

# Or render as PNG (requires graphviz)
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

## Quick Start Template

Here's a minimal but complete LangGraph multi-agent setup:

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# 1. Define state
class State(TypedDict):
    topic: str
    research: Annotated[list[str], add]
    analysis: str
    draft: str
    feedback: str
    approved: bool
    iteration: int

# 2. Define agent functions
llm = ChatOpenAI(model="gpt-4o")

def research(state):
    result = llm.invoke(f"Research: {state['topic']}")
    return {"research": [result.content]}

def analyze(state):
    data = "\n".join(state["research"])
    result = llm.invoke(f"Analyze:\n{data}")
    return {"analysis": result.content}

def write(state):
    result = llm.invoke(f"Write report based on:\n{state['analysis']}\nFeedback: {state.get('feedback', 'None')}")
    return {"draft": result.content, "iteration": state.get("iteration", 0) + 1}

def critique(state):
    result = llm.invoke(f"Evaluate this draft (respond APPROVE or REVISE with feedback):\n{state['draft']}")
    approved = "APPROVE" in result.content.upper()
    return {"feedback": result.content, "approved": approved}

# 3. Define routing
def should_revise(state):
    if state["approved"] or state["iteration"] >= 3:
        return "end"
    return "revise"

# 4. Build graph
graph = StateGraph(State)
graph.add_node("research", research)
graph.add_node("analyze", analyze)
graph.add_node("write", write)
graph.add_node("critique", critique)

graph.add_edge(START, "research")
graph.add_edge("research", "analyze")
graph.add_edge("analyze", "write")
graph.add_edge("write", "critique")
graph.add_conditional_edges("critique", should_revise, {
    "revise": "write",
    "end": END
})

# 5. Compile and run
from langgraph.checkpoint.memory import MemorySaver
app = graph.compile(checkpointer=MemorySaver())

result = app.invoke(
    {"topic": "AI adoption strategies for enterprise"},
    config={"configurable": {"thread_id": "001"}}
)
print(result["draft"])
```

## Key Dependencies

```
pip install langgraph langchain-openai langchain-core
# Optional:
pip install langgraph-checkpoint-sqlite  # Persistent checkpointing
pip install langsmith                     # Tracing & evaluation
```

## Strengths & Limitations

### Strengths
- ✅ Fine-grained control over every aspect of the workflow
- ✅ Built-in checkpointing and human-in-the-loop support
- ✅ Dynamic parallelism with Send API
- ✅ First-class streaming support
- ✅ LangSmith integration for observability
- ✅ Graph visualization
- ✅ Battle-tested in production

### Limitations
- ❌ Steeper learning curve than higher-level frameworks
- ❌ More boilerplate code for simple workflows
- ❌ Tightly coupled with LangChain ecosystem
- ❌ State schema changes require careful migration
- ❌ Limited built-in agent templates (you build from scratch)

## Resources

- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/) — Official docs
- [LangGraph Tutorials](https://docs.langchain.com/oss/python/langgraph/tutorials/) — Step-by-step guides
- [LangGraph Workflows & Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Pattern reference
- [LangGraph API Reference](https://docs.langchain.com/oss/python/langgraph/reference/) — Complete API docs
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph) — Source code

## Questions Remaining

- [ ] How does LangGraph handle errors inside individual nodes?
- [ ] What's the performance overhead of checkpointing after every node?
- [ ] How to implement proper retry logic for LLM API failures?
- [ ] Best practices for testing LangGraph applications?
