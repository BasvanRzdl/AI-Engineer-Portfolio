---
date: 2026-02-27
type: technology
topic: "Framework Comparison — LangGraph vs AutoGen vs Google ADK"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Technology: Framework Comparison

## Overview

This document provides a structured comparison of the three multi-agent frameworks researched for the Phase 4 project, culminating in a recommendation for the Strategy Research Team.

| | LangGraph | Microsoft AutoGen | Google ADK |
|---|-----------|-------------------|-----------|
| **By** | LangChain | Microsoft | Google |
| **Version** | Stable | v0.7.x (stable) | Early (evolving) |
| **License** | MIT | MIT | Apache 2.0 |
| **Language** | Python, JS/TS | Python, .NET | Python, TS, Go, Java |
| **GitHub Stars** | ~10k+ | ~40k+ | ~10k+ |

## Comparison Matrix

### Architecture & Philosophy

| Aspect | LangGraph | AutoGen | Google ADK |
|--------|-----------|---------|-----------|
| **Core metaphor** | State machine / directed graph | Group conversation | Agent hierarchy / tree |
| **Abstraction level** | Low-level primitives | High-level team patterns | Medium (composable agents) |
| **Control paradigm** | You define every edge and condition | Framework manages turns | Workflow agents orchestrate |
| **Primary unit** | Node (function) | Agent (class) | Agent (class) |
| **Coordination** | Graph edges + conditional routing | Team patterns (Selector, Swarm, etc.) | Workflow agents + LLM delegation |

### State Management

| Aspect | LangGraph | AutoGen | Google ADK |
|--------|-----------|---------|-----------|
| **State model** | TypedDict / Pydantic with reducers | Conversation messages | session.state (key-value dict) |
| **State updates** | Partial updates merged by reducers | Messages appended to conversation | Direct key-value writes |
| **Concurrent writes** | Reducers handle merging | Team manages turn-taking | Potential conflicts in parallel |
| **Checkpointing** | Built-in (Memory, SQLite, Postgres) | save_state/load_state (manual) | Session service (in-memory, DB) |
| **Persistence** | First-class with multiple backends | Basic serialization | Session-level persistence |

### Multi-Agent Patterns

| Pattern | LangGraph | AutoGen | Google ADK |
|---------|-----------|---------|-----------|
| **Sequential** | Edges: A → B → C | RoundRobinGroupChat | SequentialAgent |
| **Parallel** | Send API (dynamic fan-out) | Limited (within teams) | ParallelAgent |
| **Routing** | Conditional edges | SelectorGroupChat | LLM delegation / transfer_to_agent |
| **Swarm/Handoff** | Command(goto=) | Swarm team + HandoffMessage | transfer_to_agent |
| **Iterative** | Conditional edge loop | TextMentionTermination | LoopAgent |
| **Hierarchical** | Subgraphs | Nested teams | Agent hierarchy (sub_agents) |
| **Map-Reduce** | Send → node → aggregator | Not built-in | ParallelAgent → SequentialAgent |

### Human-in-the-Loop

| Aspect | LangGraph | AutoGen | Google ADK |
|--------|-----------|---------|-----------|
| **Mechanism** | `interrupt()` + `Command(resume=)` | UserProxyAgent | Callbacks (before/after) |
| **Granularity** | Any node can interrupt | Agent-level (human as agent) | Callback-level |
| **State editing** | Edit state before resuming | Human types response | Modify state in callback |
| **Blocking** | Yes — graph pauses | Yes — waits for input | Yes — callback blocks |

### Observability & Debugging

| Aspect | LangGraph | AutoGen | Google ADK |
|--------|-----------|---------|-----------|
| **Tracing** | LangSmith (full tracing platform) | Console logging, event handlers | Callbacks |
| **Visualization** | Graph diagram (Mermaid/PNG) | Message log | Agent tree visualization |
| **Cost tracking** | Via LangSmith or custom | Token usage in message metadata | Custom via callbacks |
| **Debugging** | State inspection, replay from checkpoint | Message inspection | State inspection, callbacks |
| **Production monitoring** | LangSmith dashboard | Custom logging | Custom via callbacks |

### Developer Experience

| Aspect | LangGraph | AutoGen | Google ADK |
|--------|-----------|---------|-----------|
| **Learning curve** | Steep (need to understand graph concepts) | Moderate (team patterns are intuitive) | Moderate (clean hierarchy) |
| **Boilerplate** | More code for simple workflows | Less code for common patterns | Moderate |
| **Documentation** | Excellent (tutorials, concepts, API) | Good (user guides, examples) | Growing (newer) |
| **Community** | Large (LangChain ecosystem) | Large (Microsoft + research) | Growing |
| **Testing** | Custom + LangSmith eval | Custom | Built-in evaluation framework |
| **Sync/Async** | Both | Async-only | Both |

### Production Readiness

| Aspect | LangGraph | AutoGen | Google ADK |
|--------|-----------|---------|-----------|
| **Maturity** | Production-ready | Stable (post-rewrite) | Newer, evolving |
| **Deployment** | LangGraph Cloud, self-hosted | Self-hosted | Agent Engine, Cloud Run, self-hosted |
| **Scalability** | Checkpointing + async execution | Distributed runtime (Core) | Agent Engine (managed) |
| **Error handling** | Conditional edges for error routing | Try/catch in agents | Callbacks for error handling |
| **Versioning** | Stable API | Breaking changes between majors | Early, expect changes |

## Scoring (For Strategy Research Team)

Rate each framework 1-5 on criteria important for the Phase 4 project:

| Criterion | Weight | LangGraph | AutoGen | Google ADK |
|-----------|--------|-----------|---------|-----------|
| **Fine-grained control** | 20% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐ (3) | ⭐⭐⭐⭐ (4) |
| **Multi-agent patterns** | 20% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐ (4) | ⭐⭐⭐⭐⭐ (5) |
| **Checkpointing/HITL** | 15% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐ (3) | ⭐⭐⭐ (3) |
| **Observability** | 15% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐ (3) | ⭐⭐⭐ (3) |
| **Documentation** | 10% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐ (4) | ⭐⭐⭐ (3) |
| **Ease of use** | 10% | ⭐⭐⭐ (3) | ⭐⭐⭐⭐ (4) | ⭐⭐⭐⭐ (4) |
| **Community/ecosystem** | 10% | ⭐⭐⭐⭐⭐ (5) | ⭐⭐⭐⭐ (4) | ⭐⭐⭐ (3) |
| **Weighted Total** | | **4.70** | **3.50** | **3.80** |

## Decision Matrix: Which Framework to Use?

### Use LangGraph When...
- ✅ You need fine-grained control over every routing decision
- ✅ Complex workflows with conditional branching, loops, and parallelism
- ✅ Checkpointing and human-in-the-loop are critical
- ✅ You want production-grade observability (LangSmith)
- ✅ You're building a custom, non-standard workflow
- ✅ You're already in the LangChain ecosystem

### Use AutoGen When...
- ✅ Rapid prototyping of conversational multi-agent systems
- ✅ The problem fits a "group chat" metaphor naturally
- ✅ You want SelectorGroupChat's intelligent speaker selection
- ✅ Customer support or conversational workflows
- ✅ You want the least code to get something working
- ✅ You're in the Microsoft ecosystem

### Use Google ADK When...
- ✅ Clean separation of deterministic orchestration and LLM reasoning
- ✅ You want composable agent hierarchies (nest Sequential/Parallel/Loop)
- ✅ Model-agnostic is important (not tied to one provider)
- ✅ You plan to deploy on Google Cloud (Agent Engine)
- ✅ You want built-in evaluation
- ✅ Multi-language team (Python + TypeScript + Go + Java)

## Recommendation for Phase 4

### Primary: LangGraph ✅

LangGraph is the required framework per the project README, and it's the best fit for the Strategy Research Team because:

1. **Orchestrator-Worker pattern** is a first-class citizen (Send API)
2. **Evaluator-Optimizer loop** maps directly to conditional edges
3. **Checkpointing** is built-in and production-ready
4. **Human-in-the-loop** via `interrupt()` is elegant
5. **Cost tracking** via state + LangSmith
6. **Visualization** helps explain the system architecture

### Secondary Framework: Google ADK 🥈 (Recommended)

For the "compare with one additional framework" requirement, Google ADK is recommended over AutoGen because:

1. **Composable workflow agents** (SequentialAgent + ParallelAgent + LoopAgent) map perfectly to your research pipeline
2. **Clean architectural contrast** with LangGraph — graph-based vs. hierarchy-based
3. **Model-agnostic** — interesting comparison point
4. **Built-in evaluation** — demonstrates a different approach to quality
5. **Growing ecosystem** — valuable to learn an emerging framework

### Alternative: AutoGen 🥉

AutoGen is a strong alternative if you prefer:
- Less boilerplate for the initial prototype
- SelectorGroupChat's intelligent routing
- The Microsoft ecosystem
- Conversational style of agent interaction

### Suggested Comparison Approach

Build the same Strategy Research Team in both LangGraph and ADK, then compare:

| Compare | LangGraph Implementation | ADK Implementation |
|---------|------------------------|--------------------|
| **Architecture** | StateGraph with nodes + edges | SequentialAgent + ParallelAgent + LoopAgent |
| **State** | TypedDict with reducers | session.state with output_key |
| **Parallelism** | Send API | ParallelAgent |
| **Iteration** | Conditional edges (critic → writer) | LoopAgent with writer + critic |
| **HITL** | interrupt() | Callbacks |
| **Lines of code** | Count them | Count them |
| **Readability** | Assess | Assess |
| **Performance** | Benchmark | Benchmark |
| **Cost** | Track tokens | Track tokens |

## Implementation Roadmap

### Week 1: LangGraph Implementation
1. Set up basic sequential pipeline (research → analyze → write)
2. Add the Critic + evaluator-optimizer loop
3. Add parallel research with Send API
4. Add checkpointing and human-in-the-loop
5. Add cost tracking and observability

### Week 2: ADK Implementation
1. Build the same pipeline with SequentialAgent
2. Add ParallelAgent for research phase
3. Add LoopAgent for writer-critic iteration
4. Add callbacks for observability
5. Compare with LangGraph implementation

### Week 3: Analysis & Report
1. Benchmark both implementations
2. Compare code complexity, readability, flexibility
3. Analyze cost and performance differences
4. Write the comparison report
5. Make final recommendation

## Resources

- [LangGraph Docs](https://docs.langchain.com/oss/python/langgraph/)
- [AutoGen Docs](https://microsoft.github.io/autogen/stable/)
- [Google ADK Docs](https://google.github.io/adk-docs/)
- [LangGraph vs AutoGen (Community)](https://github.com/langchain-ai/langgraph/discussions)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
