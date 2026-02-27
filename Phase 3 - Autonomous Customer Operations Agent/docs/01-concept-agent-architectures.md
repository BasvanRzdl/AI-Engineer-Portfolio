# Agent Architectures

> **Type:** Concept Research  
> **Project:** Phase 3 — Autonomous Customer Operations Agent  
> **Date:** 2025-02-27

---

## In My Own Words

An **agent** is an LLM-powered system that doesn't just generate text — it *reasons* about what to do, *acts* by calling tools or APIs, and *observes* the results to decide its next step. Unlike a static prompt-in/response-out pipeline, agents operate in a **loop**: they repeatedly think, act, and adjust until a task is complete.

The key architectural decision is **how the agent decides what to do next**. Different architectures make different trade-offs between autonomy, reliability, and interpretability.

---

## Why This Matters

For the Customer Operations Agent, we need an architecture that:
- Can handle multi-step tasks (look up order → check policy → process refund)
- Makes decisions about when to act, when to ask the user, and when to escalate
- Produces interpretable reasoning traces (for audit and debugging)
- Is robust to unexpected inputs and tool failures

---

## Core Architectures

### 1. ReAct (Reasoning + Acting)

**What it is:** The ReAct pattern (Yao et al., 2022) interleaves **reasoning traces** and **actions** in a single LLM generation loop. The model generates a thought (reasoning), then an action (tool call), observes the result, and repeats.

**How it works:**

```
Loop:
  1. Thought: "I need to look up order #12345 to check its status"
  2. Action: call order_lookup(order_id="12345")
  3. Observation: {"status": "shipped", "tracking": "1Z999..."}
  4. Thought: "The order is shipped. I should provide the tracking number."
  5. Action: respond_to_customer("Your order has shipped! Tracking: 1Z999...")
  6. DONE
```

**Key insight:** By making the LLM verbalize its reasoning *before* acting, you get:
- **Better grounding**: The model checks external sources instead of hallucinating
- **Interpretability**: You can see *why* the agent made each decision
- **Error recovery**: Reasoning traces help the model catch and correct mistakes

**Trade-offs:**

| Pros | Cons |
|------|------|
| Highly interpretable — reasoning is visible | More tokens per step (reasoning + action) |
| Good at multi-step tasks | Can get stuck in loops |
| Naturally handles exceptions | Relies on LLM quality for reasoning |
| Easy to debug via traces | Single LLM handles everything |

**Best for:** Tasks where you need to see the reasoning, where multi-step tool use is required, and where the task space is somewhat open-ended.

---

### 2. Plan-and-Execute

**What it is:** A two-phase approach where one LLM (or call) creates a **plan** (a list of steps), and then a separate execution phase carries out each step. The plan can be revised if steps fail or produce unexpected results.

**How it works:**

```
Phase 1 — Planning:
  Input: "Customer wants a refund for order #12345"
  Plan:
    1. Look up order #12345
    2. Check refund eligibility
    3. If eligible and under $100, process automatically
    4. If over $100, request human approval
    5. Notify customer of outcome

Phase 2 — Execution:
  Execute step 1 → observe result → execute step 2 → ...
  (Can re-plan if something unexpected happens)
```

**Key insight:** Separating planning from execution lets you:
- Review the plan before execution starts
- Use different models for planning vs. execution (e.g., stronger model for planning)
- Re-plan when the situation changes

**Trade-offs:**

| Pros | Cons |
|------|------|
| Clear structure and predictability | More complex to implement |
| Plan can be reviewed before execution | Plan may not survive contact with reality |
| Good for well-defined multi-step processes | Overhead of planning step |
| Natural human oversight point | Re-planning adds latency |

**Best for:** Tasks with clear step-by-step procedures, where you want human review of the plan, and where tasks are well-defined but multi-step.

---

### 3. Reflection / Self-Critique

**What it is:** The agent generates an output, then **evaluates its own output** and iterates until quality criteria are met. This is less of a standalone architecture and more of a pattern that can be layered on top of ReAct or Plan-and-Execute.

**How it works:**

```
Loop:
  1. Generate: Produce a response or action plan
  2. Reflect: "Is this correct? Did I miss anything? Could this cause harm?"
  3. If issues found → Revise and go to step 1
  4. If satisfied → Execute / Respond
```

**Key insight:** Self-critique catches errors before they reach the user or trigger irreversible actions. This is especially valuable for:
- High-stakes operations (refunds, account changes)
- Complex reasoning where the first answer is often wrong
- Quality control without human involvement

**Trade-offs:**

| Pros | Cons |
|------|------|
| Catches errors before execution | Higher latency (multiple LLM calls) |
| Improves output quality | Higher cost (more tokens) |
| Can be selective (only reflect on critical actions) | LLM may not catch its own errors |
| Natural quality gate | Can over-iterate without converging |

**Best for:** High-stakes operations, complex outputs that benefit from review, and as a safety layer before irreversible actions.

---

### 4. Routing / Orchestrator-Worker

**What it is:** A central orchestrator classifies the input and routes it to specialized sub-agents or workflows. Each sub-agent handles a specific type of task.

**How it works:**

```
Orchestrator receives: "I want to return my order"
  → Classify: This is a "returns" request
  → Route to: Returns Agent
  → Returns Agent handles the full returns workflow
  → Returns Agent reports result back to orchestrator
```

**Key insight:** Specialization improves reliability. Instead of one agent that does everything (and can get confused), you have focused agents that each do one thing well.

**Trade-offs:**

| Pros | Cons |
|------|------|
| Each agent can be optimized for its task | More complex system architecture |
| Easier to test and debug individual agents | Routing errors send to wrong agent |
| Can use different models/prompts per agent | More code to maintain |
| Natural separation of concerns | Handoffs between agents need care |

**Best for:** Systems with clearly distinct task types, when you want to scale team development across agents, and when different tasks need different capabilities.

---

## Comparing Architectures

| Aspect | ReAct | Plan-and-Execute | Reflection | Routing |
|--------|-------|-------------------|------------|---------|
| **Complexity** | Low | Medium | Low (add-on) | High |
| **Interpretability** | High | High | Medium | Depends |
| **Latency** | Medium | Higher | Higher | Variable |
| **Reliability** | Good | Good | Better | Best (per-task) |
| **Flexibility** | Very flexible | Structured | Flexible | Per-route |
| **Human oversight** | Via traces | At plan stage | At reflection | At routing |

---

## Workflow Patterns (from LangGraph docs)

Beyond agent architectures, LangGraph also supports several **workflow patterns** that are useful building blocks:

### Prompt Chaining
Each step processes the output of the previous step. Useful for well-defined sequential tasks with verification gates between steps.

### Parallelization
Run multiple tasks simultaneously and aggregate results. Useful for independent subtasks (e.g., check order status AND pull customer history in parallel).

### Evaluator-Optimizer
One LLM generates, another evaluates, and the loop continues until quality criteria are met. This is the Reflection pattern formalized as a workflow.

### Orchestrator-Worker
A planner breaks work into subtasks, workers execute in parallel, and a synthesizer combines results. Useful when the number of subtasks isn't known in advance.

---

## Application to Our Project

### Recommended Architecture: ReAct with Routing Elements

For the Customer Operations Agent, I recommend a **hybrid approach**:

1. **Primary loop: ReAct** — The agent reasons about what to do, calls tools, and observes results. This gives us interpretability and flexibility for the wide variety of customer requests.

2. **Routing at the edges** — Use conditional edges in the LangGraph state machine to route to specialized handling for different operation types (refunds, shipping updates, escalations).

3. **Reflection for critical actions** — Before executing high-value operations (refunds > $100), add a reflection/confirmation step.

### Why this works for our project:
- **ReAct** handles the open-ended nature of customer conversations
- **Routing** lets us create focused tool sets for different operations
- **Reflection** provides safety for destructive/expensive actions
- **LangGraph** natively supports all of these as graph patterns

### Decisions to Make
- [ ] How many specialized routes vs. one general agent?
- [ ] Where exactly to place reflection/confirmation nodes?
- [ ] What's the maximum number of reasoning steps before escalation?

---

## Resources for Deeper Learning

- [ReAct Paper (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) — Original paper introducing ReAct
- [LangGraph Workflows and Agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — Official guide to workflow patterns
- [Building Effective Agents (Anthropic)](https://www.anthropic.com/research/building-effective-agents) — Practical patterns for agent design
- [Plan-and-Execute Agents](https://blog.langchain.dev/planning-agents/) — LangChain blog on planning patterns
