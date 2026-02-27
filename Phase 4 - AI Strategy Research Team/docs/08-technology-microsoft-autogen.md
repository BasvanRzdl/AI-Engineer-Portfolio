---
date: 2026-02-27
type: technology
topic: "Microsoft AutoGen — Deep Dive"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Technology: Microsoft AutoGen

## What It Is

AutoGen is Microsoft's open-source framework for building multi-agent AI systems. It provides two layers:

1. **AgentChat** (high-level) — Pre-built agent types and team patterns. Quick to set up, opinionated.
2. **Core** (low-level) — Event-driven runtime with typed messages. Full control, build from scratch.

As of v0.7.x (stable), AutoGen focuses on **conversational multi-agent systems** where agents collaborate through structured conversations managed by team patterns.

**Key philosophy:** Multi-agent collaboration through conversation. Agents are participants in a group chat, and the team pattern determines who speaks when.

## Architecture: Two Layers

### AgentChat Layer (Recommended Starting Point)

Pre-built abstractions for common patterns:

```
AgentChat
├── Agents
│   ├── AssistantAgent    — LLM-powered agent with tools
│   ├── UserProxyAgent    — Human participant in the conversation
│   └── CodeExecutorAgent — Runs code safely
├── Teams
│   ├── RoundRobinGroupChat    — Agents take turns in fixed order
│   ├── SelectorGroupChat      — LLM picks next speaker dynamically
│   ├── Swarm                  — Agents hand off to each other
│   └── GraphFlow              — DAG-based workflow execution
└── Termination
    ├── MaxMessageTermination
    ├── TextMentionTermination
    ├── TokenUsageTermination
    └── Custom conditions
```

### Core Layer (Advanced)

Low-level event-driven system built on an Actor model:

```
Core
├── Agents — Custom agent implementations
├── Runtime — Message routing (SingleThreaded, Distributed)
├── Messages — Typed, serializable messages
├── Topics — Publish/subscribe for events
└── Tools — Function tools for agents
```

## Core Concepts

### AssistantAgent

The primary agent type. An LLM-powered agent that can use tools.

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model = OpenAIChatCompletionClient(model="gpt-4o")

researcher = AssistantAgent(
    name="Researcher",
    description="Expert at finding and gathering information from multiple sources.",
    system_message="""You are a research specialist. Your job is to find 
    relevant information on the given topic. Always cite sources.""",
    model_client=model,
    tools=[web_search_tool, document_reader_tool]
)
```

### Teams

Teams manage how agents collaborate. The team pattern is the core abstraction.

#### RoundRobinGroupChat

Agents speak in fixed rotation: A → B → C → A → B → C → ...

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

team = RoundRobinGroupChat(
    participants=[researcher, analyst, writer],
    termination_condition=MaxMessageTermination(10)
)

result = await team.run(task="Research AI adoption strategies")
```

**Best for:** Simple sequential workflows, predictable turn-taking.

#### SelectorGroupChat

An LLM dynamically selects which agent speaks next based on the conversation context.

```python
from autogen_agentchat.teams import SelectorGroupChat

team = SelectorGroupChat(
    participants=[planner, researcher, analyst, writer],
    model_client=model,  # LLM that selects the next speaker
    termination_condition=MaxMessageTermination(15)
)
```

**How selection works:**
1. The selector LLM receives all agent names + descriptions + conversation history
2. It picks the most appropriate next speaker
3. That agent generates its response
4. The process repeats

**Customizing selection:**

```python
# Custom selector function (override LLM selection)
def custom_selector(messages):
    """Always start with planner, then let LLM decide."""
    if len(messages) == 0:
        return "Planner"
    return None  # Fall back to LLM selection

team = SelectorGroupChat(
    participants=[planner, researcher, analyst, writer],
    model_client=model,
    selector_func=custom_selector
)
```

```python
# Custom candidate function (restrict which agents can speak)
def custom_candidates(messages):
    """After researcher, only analyst or writer can speak."""
    last = messages[-1].source if messages else None
    if last == "Researcher":
        return ["Analyst", "Writer"]
    return None  # All candidates

team = SelectorGroupChat(
    participants=[planner, researcher, analyst, writer],
    model_client=model,
    candidate_func=custom_candidates
)
```

**Best for:** Dynamic workflows, intelligent routing, complex multi-step tasks.

#### Swarm

Agents hand off to each other using `HandoffMessage`. No central selector — each agent decides who goes next.

```python
from autogen_agentchat.teams import Swarm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination

# Define agents with handoff capabilities
travel_agent = AssistantAgent(
    name="TravelAgent",
    description="Handles travel-related queries",
    system_message="Help with travel. If the customer wants a refund, hand off to RefundAgent.",
    model_client=model,
    handoffs=["RefundAgent"]  # Can hand off to these agents
)

refund_agent = AssistantAgent(
    name="RefundAgent", 
    description="Handles refund requests",
    system_message="Process refund requests. When done, hand off to TravelAgent.",
    model_client=model,
    handoffs=["TravelAgent"]
)

team = Swarm(
    participants=[travel_agent, refund_agent],
    termination_condition=MaxMessageTermination(10)
)
```

**Best for:** Customer support, conversational routing, decentralized workflows.

#### GraphFlow (DAG-based)

Define exact workflow as a directed acyclic graph:

```python
from autogen_agentchat.teams import GraphFlow, DiGraph

# Define the workflow graph
graph = DiGraph()
graph.add_edge("Researcher", "Analyst")
graph.add_edge("Analyst", "Writer")
graph.add_edge("Writer", "Critic")

team = GraphFlow(
    participants=[researcher, analyst, writer, critic],
    graph=graph,
    termination_condition=MaxMessageTermination(10)
)
```

**Best for:** When you need deterministic execution order with explicit dependencies.

### Termination Conditions

How the team knows when to stop:

```python
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
    TokenUsageTermination
)

# Stop after 10 messages
stop_at_10 = MaxMessageTermination(10)

# Stop when "APPROVED" appears in a message
stop_on_approval = TextMentionTermination("APPROVED")

# Combine conditions (OR logic)
termination = stop_on_approval | MaxMessageTermination(15)
```

### UserProxyAgent (Human-in-the-Loop)

Insert a human into the conversation:

```python
from autogen_agentchat.agents import UserProxyAgent

human = UserProxyAgent(
    name="HumanReviewer",
    description="A human reviewer who approves final outputs."
)

team = SelectorGroupChat(
    participants=[researcher, analyst, writer, critic, human],
    model_client=model,
    termination_condition=MaxMessageTermination(20)
)
```

When the selector picks `HumanReviewer`, execution pauses and waits for human input.

### Memory

AutoGen supports persistent memory for agents:

```python
from autogen_agentchat.memory import ChromaMemory

memory = ChromaMemory(name="research_memory")

# Agent with memory
researcher = AssistantAgent(
    name="Researcher",
    model_client=model,
    memory=[memory]  # Agent remembers across sessions
)
```

### Streaming Output

```python
from autogen_agentchat.ui import Console

# Stream all messages to console with formatted output
await Console(team.run_stream(task="Research AI adoption"))
```

## Complete Example: Strategy Research Team

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

model = OpenAIChatCompletionClient(model="gpt-4o")
fast_model = OpenAIChatCompletionClient(model="gpt-4o-mini")

# Define agents
planner = AssistantAgent(
    name="Planner",
    description="Breaks down research tasks into subtasks and coordinates the team.",
    system_message="""You are the team lead. When given a research topic:
    1. Break it into 2-3 specific research subtasks
    2. Assign each to the Researcher
    3. After research, ask the Analyst to analyze
    4. After analysis, ask the Writer to draft
    5. After writing, ask the Critic to review
    Always start by creating a plan.""",
    model_client=model
)

researcher = AssistantAgent(
    name="Researcher",
    description="Finds and gathers information on specific topics.",
    system_message="""You are a research specialist. Find relevant information 
    on the assigned topic. Cite sources. Report findings clearly.""",
    model_client=fast_model,
    tools=[web_search]
)

analyst = AssistantAgent(
    name="Analyst",
    description="Analyzes research data to identify patterns and implications.",
    system_message="""You are a strategic analyst. Analyze the research findings.
    Identify patterns, trends, risks, and opportunities. Be evidence-based.""",
    model_client=model
)

writer = AssistantAgent(
    name="Writer",
    description="Writes professional strategy reports from analysis.",
    system_message="""You are a strategy writer. Transform the analysis into
    a clear, professional report with executive summary, findings, and 
    recommendations. If given feedback, revise accordingly.""",
    model_client=model
)

critic = AssistantAgent(
    name="Critic",
    description="Evaluates report quality and provides feedback.",
    system_message="""You are a quality reviewer. Evaluate the report for:
    accuracy, completeness, clarity, and actionability. Score 1-10.
    If score >= 7, say APPROVED. Otherwise, provide specific feedback.""",
    model_client=model
)

# Create team
termination = TextMentionTermination("APPROVED") | MaxMessageTermination(20)

team = SelectorGroupChat(
    participants=[planner, researcher, analyst, writer, critic],
    model_client=model,
    termination_condition=termination
)

# Run
async def main():
    result = await team.run(task="Research AI adoption strategies for enterprise")
    for msg in result.messages:
        print(f"\n{'='*60}")
        print(f"[{msg.source}]:")
        print(msg.content[:500])

asyncio.run(main())
```

## Serialization & State Management

```python
# Save team state
state = await team.save_state()
# Save to file: json.dump(state, open("team_state.json", "w"))

# Load team state
await team.load_state(state)

# Reset team for a new task
await team.reset()
```

## Logging & Observability

```python
import logging
from autogen_agentchat.ui import Console

# Structured logging
logging.basicConfig(level=logging.INFO)

# Console output with all messages
await Console(team.run_stream(task="..."))

# Or collect messages programmatically
result = await team.run(task="...")
for message in result.messages:
    print(f"Agent: {message.source}, Tokens: {message.models_usage}")
```

## Key Dependencies

```
pip install autogen-agentchat autogen-ext[openai]
# Optional:
pip install autogen-ext[chromadb]  # For memory
pip install autogen-ext[docker]    # For code execution
```

## Strengths & Limitations

### Strengths
- ✅ High-level team abstractions (quick to prototype)
- ✅ Multiple team patterns out of the box
- ✅ Excellent SelectorGroupChat for dynamic routing
- ✅ Built-in human-in-the-loop with UserProxyAgent
- ✅ Async-first architecture
- ✅ Good documentation and examples
- ✅ Microsoft backing and active development

### Limitations
- ❌ Less fine-grained control than LangGraph
- ❌ Conversation-centric — harder for non-conversational workflows
- ❌ Team patterns can be opaque (hard to debug selector decisions)
- ❌ Breaking changes between major versions (v0.2 → v0.4 was a full rewrite)
- ❌ Async-only (no synchronous API)
- ❌ Less mature ecosystem compared to LangChain

## AutoGen vs LangGraph (Quick Comparison)

| Aspect | AutoGen | LangGraph |
|--------|---------|-----------|
| **Abstraction level** | High (teams, agents) | Low (nodes, edges, state) |
| **Paradigm** | Conversational group chat | State machine / graph |
| **Control** | Team patterns manage flow | You define every edge |
| **Parallelism** | Limited (within team patterns) | Native (Send API) |
| **Checkpointing** | save_state/load_state | Built-in with multiple backends |
| **Human-in-loop** | UserProxyAgent | interrupt() / Command(resume=) |
| **Best for** | Rapid prototyping, conversation-heavy tasks | Complex workflows, production systems |

## Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/stable/) — Official docs
- [AgentChat User Guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/) — High-level guide
- [Core User Guide](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/) — Low-level guide
- [AutoGen GitHub](https://github.com/microsoft/autogen) — Source code
- [AutoGen Examples](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/examples/) — Gallery of examples

## Questions Remaining

- [ ] How does SelectorGroupChat perform with 5+ agents?
- [ ] Can GraphFlow handle cycles (e.g., Writer → Critic → Writer)?
- [ ] How to implement cost tracking across a team run?
- [ ] What's the migration path from AutoGen to LangGraph if needed?
