# Phase 4: AI Strategy Research Team

> **Duration:** Week 7-8 | **Hours Budget:** ~40 hours  
> **Outcome:** Multi-agent orchestration, specialized agents, coordination

---

## Business Context

Strategy consulting requires deep research, synthesis of multiple sources, structured analysis, and polished deliverables. A single AI agent struggles with such complex, multi-faceted work. You're asked to build a *team* of specialized AI agents that collaborate like a real consulting team.

---

## Your Mission

Build a **multi-agent system** that can produce strategy research deliverables. The system should demonstrate meaningful agent specialization and collaboration patterns.

---

## Deliverables

1. **Specialized agents (minimum 4):**
   - **Research Agent**: Web research, source gathering, fact extraction
   - **Analysis Agent**: Pattern identification, SWOT/framework application
   - **Writer Agent**: Structured document creation, coherent narrative
   - **Critic Agent**: Quality review, fact-checking, improvement suggestions

2. **Orchestration layer:**
   - Task decomposition and assignment
   - Agent communication protocols
   - Parallel vs. sequential execution control
   - Conflict resolution when agents disagree

3. **Workflow patterns (implement at least 2):**
   - Sequential pipeline (research → analyze → write → review)
   - Iterative refinement (write → critique → revise loop)
   - Parallel research with synthesis
   - Human-in-the-loop checkpoints

4. **Output artifacts:**
   - Structured research report with citations
   - Evidence trail showing agent contributions
   - Quality metrics and confidence scores

---

## Technical Requirements

- Use **LangGraph** for orchestration
- Implement with at least one additional framework (**Microsoft Agent Framework** or **Google ADK**) to compare approaches
- Include configurable workflow templates
- Build visualization of agent interactions

---

## Constraints

- Total cost per research project should be trackable and bounded
- Long-running tasks should be resumable (checkpoint system)
- Must handle agent failures gracefully (retry, fallback, escalate)

---

## Learning Objectives

- Multi-agent design patterns
- Orchestration and coordination strategies
- Framework comparison (LangGraph vs. alternatives)
- Complex workflow management
- Cost management in multi-agent systems

---

## Concepts to Explore

- Multi-agent patterns (hierarchical, democratic, market-based)
- Agent communication (shared state, message passing, blackboard)
- Specialization vs. generalization trade-offs
- Emergent behavior in multi-agent systems
- Debugging and observability for multi-agent systems

---

## Hints

- Don't over-engineer agent count; 4-5 well-designed agents beat 10 superficial ones
- The orchestrator might be the hardest part
- Consider: how do you evaluate multi-agent system quality?
- Look at AutoGen and CrewAI for inspiration, even if not using them
