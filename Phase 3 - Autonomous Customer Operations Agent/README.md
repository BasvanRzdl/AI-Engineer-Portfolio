# Phase 3: Autonomous Customer Operations Agent

> **Duration:** Week 5-6 | **Hours Budget:** ~40 hours  
> **Outcome:** Tool use, state management, LangGraph fundamentals

---

## Business Context

An e-commerce company processes thousands of customer inquiries daily. They want an AI agent that can actually *do things* — not just answer questions, but look up orders, process refunds, update shipping addresses, and escalate to humans when needed. This isn't a chatbot; it's an operational agent.

---

## Your Mission

Build an **autonomous agent** using **LangGraph** that can handle customer operations end-to-end, with appropriate guardrails and human oversight.

---

## Deliverables

1. **Agent core:**
   - LangGraph state machine implementation
   - ReAct-style reasoning loop
   - Clear decision boundaries (when to act vs. ask vs. escalate)

2. **Tool ecosystem:**
   - Order lookup tool (mock database)
   - Refund processing tool (with approval thresholds)
   - Shipping update tool
   - Knowledge base search tool (connect to Project 2 if possible)
   - Human escalation tool

3. **Memory and context:**
   - Conversation memory (within session)
   - Customer context awareness (past interactions, preferences)
   - Long-term memory patterns (summarization, retrieval)

4. **Guardrails and safety:**
   - Action confirmation for destructive operations
   - Spending limits and approval workflows
   - PII handling considerations
   - Injection attack resistance

5. **Observability:**
   - Full trace logging of agent reasoning
   - Decision audit trail
   - Performance metrics (resolution rate, escalation rate, etc.)

---

## Technical Requirements

- Build with **LangGraph** (primary framework for this project)
- Implement proper state management
- Include checkpoint/resume capability
- Create a conversation interface (CLI or simple web UI)

---

## Constraints

- Agent must never take irreversible actions without confirmation above $100
- Must gracefully handle ambiguous requests
- Escalation to human should include full context summary

---

## Learning Objectives

- Deep understanding of agent architectures
- LangGraph state machines and graph patterns
- Tool design and implementation
- Memory patterns for agents
- Guardrails and safety in agentic systems

---

## Concepts to Explore

- Agent architectures (ReAct, Plan-and-Execute, Reflection)
- LangGraph concepts (nodes, edges, state, checkpointing)
- Tool calling patterns and error handling
- Memory types (buffer, summary, vector-backed)
- Agent evaluation strategies

---

## Hints

- Start simple: get one tool working perfectly before adding more
- Think about failure modes: what if a tool fails mid-operation?
- The graph structure IS your architecture decision
- Human escalation is a feature, not a failure
