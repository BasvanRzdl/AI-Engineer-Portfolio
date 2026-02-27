---
date: 2026-02-27
type: concept
topic: "Multi-Agent Architecture Patterns"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Learning: Multi-Agent Architecture Patterns

## In My Own Words

A multi-agent system (MAS) is a collection of AI agents that work together to accomplish tasks too complex or diverse for a single agent. Instead of building one monolithic "super-agent," you decompose the problem into specialized roles and let multiple agents collaborate—each bringing a focused capability to the table.

Think of it like a consulting team: you don't have one person who does the research, the analysis, the writing, AND the quality review. You have specialists who are great at their part, and a coordination mechanism that ensures they work together effectively.

The key insight is that **the architecture pattern you choose determines how agents interact**, and this has enormous impact on the system's reliability, flexibility, and cost.

## Why This Matters

- Single agents hit capability ceilings on complex, multi-step tasks
- Specialized agents produce higher-quality outputs in their domain
- Different problems demand different coordination strategies
- The wrong architecture pattern leads to wasted tokens, confused agents, and poor results
- Understanding patterns helps you pick the right approach for your Strategy Research Team

## Core Patterns

### 1. Sequential Pipeline (Chain)

```
[Research] → [Analysis] → [Writing] → [Review]
```

**How it works:** Agents execute in a fixed order. Each agent receives the output of the previous one and adds its contribution. Like an assembly line.

**When to use:**
- Tasks have a natural step-by-step progression
- Each step depends on the previous step's output
- Order matters and is predictable

**Pros:**
- ✅ Simple to reason about and debug
- ✅ Clear data flow — easy to trace what happened
- ✅ Each agent has focused context
- ✅ Deterministic execution order

**Cons:**
- ❌ Slow — no parallelism, total time = sum of all steps
- ❌ Rigid — can't adapt if the task doesn't fit the pipeline
- ❌ Single point of failure — one bad agent output cascades downstream
- ❌ No iteration — can't go back to fix earlier stages

**Application to Strategy Research Team:**
Your primary workflow (research → analyze → write → review) naturally fits a sequential pipeline. This should be your base pattern.

---

### 2. Hierarchical (Orchestrator-Worker)

```
         [Orchestrator]
        /      |       \
  [Worker1] [Worker2] [Worker3]
```

**How it works:** A central orchestrator agent receives tasks, decomposes them, delegates subtasks to specialized worker agents, and synthesizes results. The orchestrator makes all routing decisions.

**When to use:**
- Complex tasks that need dynamic decomposition
- When you can't predict the exact workflow ahead of time
- When subtasks are independent and can run in parallel
- When you need centralized control and monitoring

**Pros:**
- ✅ Flexible — orchestrator adapts to different task types
- ✅ Enables parallelism through dynamic delegation
- ✅ Central point of control for monitoring and cost tracking
- ✅ Can handle failures by reassigning work

**Cons:**
- ❌ Orchestrator is a bottleneck and single point of failure
- ❌ Orchestrator needs high-quality LLM (expensive)
- ❌ Can be over-engineered for simple workflows
- ❌ Debugging is harder — need to trace orchestrator decisions

**Application to Strategy Research Team:**
This is ideal for your system. The orchestrator breaks down "research topic X" into subtasks and delegates to Research, Analysis, Writer, and Critic agents.

---

### 3. Swarm (Decentralized/Handoff)

```
[Agent A] ←→ [Agent B] ←→ [Agent C]
   ↕                          ↕
[Agent D] ←→ [Agent E] ←→ [Agent F]
```

**How it works:** There is no central orchestrator. Each agent makes local decisions about who to hand off to next, based on the current context. Agents signal handoffs through their tool calls or messages. Popularized by OpenAI's Swarm framework.

**When to use:**
- Agents have clear, non-overlapping responsibilities
- The routing logic is simple enough for each agent to decide locally
- You want resilience — no single point of failure
- Customer support or conversational flows with clear routing rules

**Pros:**
- ✅ No bottleneck — distributed decision-making
- ✅ Resilient — no single point of failure
- ✅ Simple local logic per agent
- ✅ Scales well as you add more agents

**Cons:**
- ❌ Hard to debug — no central view of what's happening
- ❌ Possible infinite loops if handoff logic isn't careful
- ❌ No global optimization — agents make local decisions
- ❌ Emergent behavior can be unpredictable

**Application to Strategy Research Team:**
Less suitable for your use case because strategy research needs a structured workflow with clear phases. However, elements of this pattern (like the Critic handing back to the Writer) can be combined with other patterns.

---

### 4. Parallel Fan-Out / Fan-In (Map-Reduce)

```
            [Coordinator]
           /      |      \
     [Worker1] [Worker2] [Worker3]   ← Fan-out (parallel)
           \      |      /
            [Synthesizer]            ← Fan-in (aggregate)
```

**How it works:** A coordinator sends the same task (or different subtasks) to multiple agents simultaneously. After all agents complete, a synthesizer combines results. Like MapReduce for AI agents.

**When to use:**
- Independent subtasks that can run concurrently
- Need multiple perspectives on the same problem
- Speed is important — parallelize to reduce total time
- Gathering information from multiple sources

**Pros:**
- ✅ Fast — parallel execution reduces wall-clock time
- ✅ Multiple perspectives improve quality
- ✅ Natural fit for research and information gathering
- ✅ Easy to add/remove workers

**Cons:**
- ❌ Requires a good synthesis step
- ❌ All workers must complete before synthesis (bottleneck on slowest)
- ❌ Higher cost — running N agents in parallel means N× tokens
- ❌ Workers may produce conflicting information

**Application to Strategy Research Team:**
Excellent for the research phase. You can have multiple Research agents investigating different aspects of a topic in parallel, then synthesize their findings.

---

### 5. Evaluator-Optimizer (Iterative Refinement)

```
[Generator] → [Evaluator] → (if not good enough) → [Generator] → [Evaluator] → ... → [Done]
```

**How it works:** One agent generates output, another evaluates it against criteria. If the output doesn't meet the bar, feedback loops back to the generator for improvement. The cycle continues until quality thresholds are met or iteration limits are reached.

**When to use:**
- Quality standards are well-defined
- Output can be measurably improved through iteration
- You prefer quality over speed
- Writing, code generation, translation tasks

**Pros:**
- ✅ Produces higher-quality output through iteration
- ✅ Clear separation of generation and evaluation concerns
- ✅ Self-improving — each iteration gets feedback
- ✅ Natural quality gate

**Cons:**
- ❌ Can be expensive — multiple iterations = multiple LLM calls
- ❌ Risk of infinite loops without proper termination conditions
- ❌ Evaluator quality limits the whole system
- ❌ Diminishing returns after a few iterations

**Application to Strategy Research Team:**
This is exactly your Writer → Critic → Writer loop. The Critic evaluates the draft and sends specific feedback until the output meets quality standards.

---

### 6. Democratic / Voting

```
[Agent A] ──┐
[Agent B] ──┼── [Aggregator / Voter] → [Final Decision]
[Agent C] ──┘
```

**How it works:** Multiple agents independently work on the same problem. A voting or aggregation mechanism determines the final output based on consensus, majority, or quality scoring.

**When to use:**
- High-stakes decisions where you need confidence
- Tasks where multiple approaches might yield different valid answers
- Ensemble methods for improved reliability
- Fact-checking or verification scenarios

**Pros:**
- ✅ Higher confidence through consensus
- ✅ Reduces individual agent errors
- ✅ Can detect outliers or hallucinations

**Cons:**
- ❌ Expensive — N agents × same task
- ❌ Majority isn't always right
- ❌ Complex aggregation logic needed

---

### 7. Market-Based / Bidding

**How it works:** Tasks are broadcast to all agents. Agents "bid" on tasks they're best suited for (based on capability, cost, or confidence). The system assigns tasks to the winning bidder.

**When to use:**
- Large, heterogeneous agent pools
- Dynamic task allocation
- When you want self-organizing behavior

This is more theoretical and less commonly implemented in current frameworks. Good to know conceptually but unlikely to use in your Phase 4 project.

## Approaches Compared

| Pattern | Complexity | Flexibility | Speed | Cost | Quality | Debug |
|---------|-----------|-------------|-------|------|---------|-------|
| Sequential | Low | Low | Slow | Low | Medium | Easy |
| Hierarchical | Medium | High | Medium | Medium | High | Medium |
| Swarm | Medium | High | Medium | Medium | Medium | Hard |
| Fan-Out/In | Medium | Medium | Fast | High | High | Medium |
| Evaluator-Optimizer | Low | Low | Slow | High | Very High | Easy |
| Democratic | Low | Low | Medium | Very High | Very High | Easy |

## Best Practices

- ✅ **Start simple** — begin with a sequential pipeline, add complexity only when needed
- ✅ **Combine patterns** — use hierarchical for orchestration + iterative refinement for quality
- ✅ **Set iteration limits** — always cap loops to prevent runaway costs
- ✅ **Design for failure** — what happens when an agent produces garbage? Have fallbacks.
- ✅ **Keep the orchestrator smart but lean** — use a capable model but with focused instructions
- ❌ **Don't over-engineer** — 4-5 well-designed agents beat 10 superficial ones (as your README says)
- ❌ **Don't let every agent talk to every other** — constrain communication to what's needed
- ❌ **Don't skip the synthesis step** — parallel results without aggregation are useless

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Infinite loops | No termination condition on iterative patterns | Always set max_iterations and quality thresholds |
| Orchestrator confusion | Too many agents or unclear descriptions | Give each agent a clear, distinct name and description |
| Token explosion | Passing full context through every agent | Filter and summarize context between agents |
| Conflicting outputs | Parallel agents disagree on facts | Add a reconciliation/verification step |
| Over-specialization | Too many narrow agents | Start with 4-5 agents, split only when needed |

## Application to My Project

### Recommended Architecture: Hybrid Pattern

For the AI Strategy Research Team, combine:

1. **Hierarchical Orchestration** — Central orchestrator decomposes the research task
2. **Parallel Fan-Out** — Multiple research subtasks run in parallel
3. **Sequential Pipeline** — Results flow through Analysis → Writing
4. **Iterative Refinement** — Writer ↔ Critic loop until quality bar is met

```
                    [Orchestrator]
                    /     |      \
            [Research1] [Research2] [Research3]   (parallel)
                    \     |      /
                    [Synthesizer]                  (fan-in)
                        |
                    [Analyst]                      (sequential)
                        |
                    [Writer] ←→ [Critic]           (iterative)
                        |
                    [Final Output]
```

### Decisions to Make
- [ ] How many research subtasks to fan out in parallel?
- [ ] What quality criteria trigger the Critic → Writer loop?
- [ ] Maximum iterations for the refinement loop?
- [ ] How does the orchestrator decide the research plan?

## Resources for Deeper Learning

- [LangGraph Workflows & Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Complete patterns with code
- [AutoGen Design Patterns](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/intro.html) — Multi-agent design patterns from Microsoft
- [Google ADK Multi-Agent Systems](https://google.github.io/adk-docs/agents/multi-agents/) — Patterns implemented in ADK
- [Building Effective Agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents) — Practical guide from Anthropic
- [OpenAI Swarm](https://github.com/openai/swarm) — Reference implementation for swarm pattern

## Questions Remaining

- [ ] How to handle conflicting information between parallel research agents?
- [ ] What's the optimal number of refinement iterations before diminishing returns?
- [ ] How to balance cost vs quality in the orchestrator's model choice?
