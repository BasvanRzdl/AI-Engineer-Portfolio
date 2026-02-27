---
date: 2026-02-27
type: technology
topic: "Google Agent Development Kit (ADK) — Deep Dive"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Technology: Google Agent Development Kit (ADK)

## What It Is

The Agent Development Kit (ADK) is Google's open-source, model-agnostic framework for building AI agents and multi-agent systems. Despite being by Google, it works with any LLM provider (OpenAI, Anthropic, etc.), not just Gemini.

ADK organizes agents into three categories:
1. **LlmAgent** — LLM-powered agents with instructions and tools
2. **Workflow Agents** — Deterministic orchestrators (Sequential, Parallel, Loop)
3. **Custom Agents** — Extend `BaseAgent` for fully custom behavior

**Key philosophy:** Composable agent hierarchy. Agents are organized in parent-child trees where workflow agents orchestrate LLM agents. This cleanly separates deterministic coordination from LLM-driven reasoning.

## Architecture

```
ADK
├── Agents
│   ├── LlmAgent          — LLM-powered with instructions, tools, sub-agents
│   ├── SequentialAgent    — Runs sub-agents in order
│   ├── ParallelAgent      — Runs sub-agents concurrently
│   ├── LoopAgent          — Runs sub-agent repeatedly until condition
│   └── BaseAgent          — Override for custom behavior
├── Communication
│   ├── Shared State       — session.state (key-value)
│   ├── LLM Delegation     — transfer_to_agent()
│   └── AgentTool          — Invoke agent as tool
├── Sessions & Memory
│   ├── Session Service    — In-memory or database-backed
│   └── Memory Service     — Long-term memory with search
├── Callbacks
│   ├── before_agent_callback
│   ├── after_agent_callback
│   ├── before_tool_callback
│   └── after_tool_callback
└── Deployment
    ├── Agent Engine (Google Cloud)
    ├── Cloud Run
    └── Local development server
```

## Core Concepts

### LlmAgent

The primary agent type. An LLM-powered agent with instructions, tools, and optional sub-agents.

```python
from google.adk.agents import LlmAgent

researcher = LlmAgent(
    name="researcher",
    model="gemini-2.0-flash",  # Or any supported model
    instruction="""You are a research specialist. Your job is to find 
    relevant information on the given topic. Always cite sources.
    When you're done, store your findings using the output_key.""",
    description="Expert at finding and gathering information from multiple sources.",
    tools=[web_search, document_reader],
    output_key="research_results"  # Stores output in session.state["research_results"]
)
```

**Key properties:**
- `name` — Unique identifier
- `model` — LLM model to use
- `instruction` — System prompt (supports templates with `{variable}` referencing state)
- `description` — Used by parent agents for delegation decisions
- `tools` — List of callable tools
- `sub_agents` — Child agents for delegation
- `output_key` — State key to store agent's final output
- `input_schema` / `output_schema` — Pydantic models for structured I/O

### Dynamic Instructions with State

Instructions can reference session state using template variables:

```python
analyst = LlmAgent(
    name="analyst",
    model="gemini-2.0-flash",
    instruction="""You are a strategic analyst. 
    Analyze the following research results: {research_results}
    The original topic was: {topic}
    Identify patterns, trends, and strategic implications.""",
    output_key="analysis"
)
```

When the agent runs, `{research_results}` and `{topic}` are replaced with values from `session.state`.

### Workflow Agents

Deterministic orchestrators that coordinate sub-agents without using an LLM.

#### SequentialAgent

Runs sub-agents one after another:

```python
from google.adk.agents import SequentialAgent

pipeline = SequentialAgent(
    name="research_pipeline",
    sub_agents=[researcher, analyst, writer],
    description="Executes the full research pipeline in order."
)
# Execution: researcher → analyst → writer
```

#### ParallelAgent

Runs sub-agents concurrently:

```python
from google.adk.agents import ParallelAgent

parallel_research = ParallelAgent(
    name="parallel_research",
    sub_agents=[market_researcher, tech_researcher, competitor_researcher],
    description="Researches multiple aspects simultaneously."
)
# All three run at the same time; results stored via output_key
```

#### LoopAgent

Runs sub-agent(s) repeatedly until a condition is met:

```python
from google.adk.agents import LoopAgent

refinement_loop = LoopAgent(
    name="refinement_loop",
    sub_agents=[writer, critic],  # Writer then Critic, repeatedly
    max_iterations=3,
    description="Iteratively refines the report until approved."
)
```

**Exit condition:** The sub-agent sets `escalate=True` in its response, or `max_iterations` is reached.

### Combining Workflow Agents (Composable)

The real power: nest workflow agents to build complex pipelines.

```python
# Full pipeline: research in parallel → analyze → write/review loop
full_pipeline = SequentialAgent(
    name="strategy_pipeline",
    sub_agents=[
        ParallelAgent(
            name="research_phase",
            sub_agents=[market_researcher, tech_researcher, competitor_researcher]
        ),
        analyst,
        LoopAgent(
            name="writing_phase",
            sub_agents=[writer, critic],
            max_iterations=3
        )
    ]
)
```

This gives you:
```
research_phase (parallel)
├── market_researcher
├── tech_researcher
└── competitor_researcher
    ↓
analyst (sequential)
    ↓
writing_phase (loop)
├── writer
└── critic → (loop until approved or max 3)
```

## Communication Mechanisms

### 1. Shared Session State

The primary communication mechanism. Agents read/write to `session.state`.

```python
# Researcher writes to state
researcher = LlmAgent(
    name="researcher",
    output_key="research_results",  # Output stored here
    ...
)

# Analyst reads from state
analyst = LlmAgent(
    name="analyst",
    instruction="Analyze: {research_results}",  # Reads from state
    output_key="analysis",
    ...
)
```

State keys can be scoped:
- **Session-level:** Default. Scoped to current conversation.
- **User-level:** Persists across sessions for the same user.
- **App-level:** Shared across all users and sessions.

### 2. LLM-Driven Delegation (transfer_to_agent)

An LLM agent can delegate to sub-agents based on their descriptions:

```python
orchestrator = LlmAgent(
    name="orchestrator",
    model="gemini-2.0-flash",
    instruction="You coordinate the research team. Delegate tasks to the right specialist.",
    sub_agents=[researcher, analyst, writer, critic],
    # The LLM uses sub-agent descriptions to decide who to call
)
```

When the LLM decides to delegate, it calls `transfer_to_agent("researcher")` internally. The framework handles routing.

### 3. AgentTool (Agent as Tool)

Wrap an agent so it can be explicitly invoked as a tool:

```python
from google.adk.tools import AgentTool

research_tool = AgentTool(agent=researcher)

analyst = LlmAgent(
    name="analyst",
    tools=[research_tool],  # Can call researcher as a tool
    instruction="If you need more data, use the researcher tool."
)
```

**Difference from transfer_to_agent:**
- `transfer_to_agent` — transfers control; the sub-agent takes over
- `AgentTool` — calls the agent like a function and gets the result back

## Sessions & Memory

### Session Service

Manages conversation state:

```python
from google.adk.sessions import InMemorySessionService

session_service = InMemorySessionService()

# Create a session
session = await session_service.create_session(
    app_name="strategy_research",
    user_id="user_123"
)

# Session contains state and conversation history
session.state["topic"] = "AI adoption strategies"
```

### Memory Service

Long-term memory that persists across sessions:

```python
from google.adk.memory import InMemoryMemoryService

memory_service = InMemoryMemoryService()

# Memory is automatically searchable
# Agents can recall information from past sessions
```

## Callbacks (Observability)

ADK provides hooks for monitoring agent execution:

```python
from google.adk.agents import LlmAgent

def log_before(callback_context):
    print(f"Agent {callback_context.agent_name} starting...")
    # Can modify state or cancel execution

def log_after(callback_context):
    print(f"Agent {callback_context.agent_name} finished.")
    print(f"Output: {callback_context.response}")

researcher = LlmAgent(
    name="researcher",
    before_agent_callback=log_before,
    after_agent_callback=log_after,
    before_tool_callback=log_before_tool,
    after_tool_callback=log_after_tool,
    ...
)
```

Callbacks can:
- Log events for observability
- Modify state before/after agent execution
- Cancel or redirect agent execution
- Implement guardrails and safety checks

## Built-in Evaluation

ADK has a built-in evaluation framework:

```python
# Define test cases
test_cases = [
    {
        "input": "Research AI adoption in healthcare",
        "expected_output_contains": ["market size", "key players"],
        "expected_tools_used": ["web_search"],
        "quality_threshold": 7.0
    }
]

# Run evaluation
# ADK provides evaluation metrics and scoring
```

## Complete Example: Strategy Research Team

```python
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# Define specialized agents
market_researcher = LlmAgent(
    name="market_researcher",
    model="gemini-2.0-flash",
    instruction="Research market trends and data for: {topic}",
    description="Researches market size, growth, and trends.",
    tools=[web_search],
    output_key="market_research"
)

tech_researcher = LlmAgent(
    name="tech_researcher",
    model="gemini-2.0-flash",
    instruction="Research technology landscape for: {topic}",
    description="Researches technology trends, tools, and capabilities.",
    tools=[web_search],
    output_key="tech_research"
)

analyst = LlmAgent(
    name="analyst",
    model="gemini-2.0-flash",
    instruction="""Analyze these research findings:
    Market: {market_research}
    Technology: {tech_research}
    Identify patterns, risks, and opportunities.""",
    description="Analyzes research data for strategic insights.",
    output_key="analysis"
)

writer = LlmAgent(
    name="writer",
    model="gemini-2.0-flash",
    instruction="""Write a strategy report based on: {analysis}
    Previous feedback (if any): {feedback}
    Include: executive summary, findings, recommendations.""",
    description="Writes professional strategy reports.",
    output_key="draft"
)

critic = LlmAgent(
    name="critic",
    model="gemini-2.0-flash",
    instruction="""Evaluate this report: {draft}
    Score 1-10 on: accuracy, completeness, clarity, actionability.
    If average score >= 7, respond with 'APPROVED'.
    Otherwise, provide specific feedback for improvement.""",
    description="Reviews and evaluates report quality.",
    output_key="feedback"
)

# Compose the pipeline
strategy_team = SequentialAgent(
    name="strategy_team",
    sub_agents=[
        ParallelAgent(
            name="research_phase",
            sub_agents=[market_researcher, tech_researcher]
        ),
        analyst,
        LoopAgent(
            name="writing_phase",
            sub_agents=[writer, critic],
            max_iterations=3
        )
    ]
)

# Run
session_service = InMemorySessionService()
runner = Runner(
    agent=strategy_team,
    app_name="strategy_research",
    session_service=session_service
)

async def main():
    session = await session_service.create_session(
        app_name="strategy_research",
        user_id="user_1"
    )
    session.state["topic"] = "AI adoption strategies for enterprise"
    
    async for event in runner.run_async(
        user_id="user_1",
        session_id=session.id,
        new_message="Start the research"
    ):
        print(event)

import asyncio
asyncio.run(main())
```

## Multi-Agent Patterns in ADK

### Coordinator/Dispatcher Pattern

```python
coordinator = LlmAgent(
    name="coordinator",
    instruction="Route tasks to the right specialist based on the request.",
    sub_agents=[researcher, analyst, writer, critic]
    # LLM decides who to delegate to based on descriptions
)
```

### Sequential Pipeline

```python
pipeline = SequentialAgent(
    name="pipeline",
    sub_agents=[step1, step2, step3]
)
```

### Parallel Fan-Out / Fan-In

```python
research = ParallelAgent(
    name="research",
    sub_agents=[researcher_a, researcher_b, researcher_c]
)
# Each researcher writes to a different output_key
# Synthesizer reads all keys
```

### Iterative Refinement

```python
refinement = LoopAgent(
    name="refinement",
    sub_agents=[generator, evaluator],
    max_iterations=3
)
```

### Human-in-the-Loop

```python
def human_review_callback(callback_context):
    """Pause and ask human for input."""
    draft = callback_context.state.get("draft", "")
    print(f"Draft for review:\n{draft}")
    decision = input("Approve? (yes/no): ")
    if decision == "yes":
        callback_context.state["approved"] = True
    else:
        feedback = input("Feedback: ")
        callback_context.state["human_feedback"] = feedback

critic = LlmAgent(
    name="critic",
    after_agent_callback=human_review_callback,
    ...
)
```

## Key Dependencies

```
pip install google-adk
# For specific model providers:
pip install google-adk[google]   # Gemini models
pip install google-adk[openai]   # OpenAI models
pip install google-adk[anthropic] # Anthropic models
```

## Strengths & Limitations

### Strengths
- ✅ Clean agent hierarchy with composable workflow agents
- ✅ Model-agnostic — works with any LLM provider
- ✅ Clear separation of deterministic orchestration and LLM reasoning
- ✅ Built-in evaluation framework
- ✅ Callback system for observability
- ✅ Multi-language support (Python, TypeScript, Go, Java)
- ✅ Google Cloud deployment options (Agent Engine)
- ✅ A2A (Agent-to-Agent) protocol for inter-service communication

### Limitations
- ❌ Newer framework — smaller community and fewer examples
- ❌ Less mature checkpointing than LangGraph
- ❌ LoopAgent exit conditions can be tricky
- ❌ Documentation still evolving
- ❌ Less ecosystem tooling (no equivalent of LangSmith)
- ❌ Shared state can become complex with many agents

## ADK vs LangGraph (Quick Comparison)

| Aspect | Google ADK | LangGraph |
|--------|-----------|-----------|
| **Abstraction level** | Medium (agent hierarchy) | Low (nodes, edges) |
| **Orchestration** | Workflow agents (Sequential, Parallel, Loop) | Graph edges + conditional routing |
| **Parallelism** | ParallelAgent | Send API |
| **State management** | session.state (key-value) | TypedDict with reducers |
| **Checkpointing** | Session service | Built-in with multiple backends |
| **Observability** | Callbacks | LangSmith integration |
| **Model support** | Any (model-agnostic) | Any (via LangChain) |
| **Maturity** | Newer | More established |
| **Best for** | Clean hierarchical agent systems | Complex custom workflows |

## Resources

- [Google ADK Documentation](https://google.github.io/adk-docs/) — Official docs
- [ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/) — Multi-agent patterns
- [ADK Workflow Agents](https://google.github.io/adk-docs/agents/workflow-agents/) — Sequential, Parallel, Loop
- [ADK GitHub](https://github.com/google/adk-python) — Source code
- [ADK Quickstart](https://google.github.io/adk-docs/get-started/quickstart/) — Getting started

## Questions Remaining

- [ ] How does ADK handle state conflicts in parallel agents writing to the same key?
- [ ] Can LoopAgent sub-agents access the iteration count?
- [ ] How to implement cost tracking across the agent hierarchy?
- [ ] What's the production deployment experience like with Agent Engine?
