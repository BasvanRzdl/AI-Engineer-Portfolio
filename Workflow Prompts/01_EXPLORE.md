# EXPLORE Phase Prompt

> **Purpose:** Understand the codebase, identify relevant patterns, and gather context before planning.  
> **Mode:** Read-only — no code changes during this phase.  
> **Time Allocation:** ~10% of project time  
> **Copilot Mode:** Chat with @workspace, Agent for deep exploration

---

## When to Use This Prompt

Use this prompt at the start of any substantial task:
- Beginning a new project phase
- Working in an unfamiliar part of the codebase
- Before making architectural decisions
- When you need to understand how existing systems work
- After completing RESEARCH to apply concepts to your specific project

### GitHub Copilot Tips for Exploration

**Quick exploration (Copilot Chat):**
- Use `@workspace` to ask about your entire codebase
- Use `@workspace /explain` to understand specific files or patterns
- Reference files directly: "Explain how #file:retriever.py handles errors"

**Deep exploration (Copilot Agent):**
- Ask Copilot to trace through data flows
- Request investigation of specific patterns across multiple files
- Use sub-agents for parallel exploration of different subsystems

---

## The Explore Prompt

Copy and adapt this prompt for your AI assistant:

```
# EXPLORE: [Task/Feature Name]

I'm starting work on [brief description of what you're building].

## Context
- Project: [Project name from learning path]
- Phase/Deliverable: [Specific deliverable you're working on]
- Relevant documentation: [Link to any specs, requirements, or prior research]

## What I Need to Understand

Before I start planning or implementing, I need to explore:

1. **Existing Patterns:**
   - How does the codebase currently handle [related functionality]?
   - What conventions are already established?
   - Are there similar implementations I can learn from?

2. **Architecture:**
   - What are the key components involved?
   - How do they interact?
   - What are the data flows?

3. **Dependencies:**
   - What external libraries/services are relevant?
   - What internal modules will I need to integrate with?
   - Are there version constraints or compatibility concerns?

4. **Constraints:**
   - What are the technical limitations?
   - What are the business/project constraints?
   - What should I explicitly NOT do?

## Research Tasks

Please help me explore by:
1. Finding relevant files and patterns in the codebase
2. Identifying conventions I should follow
3. Highlighting potential complexities or edge cases
4. Noting any assumptions that need verification

## Output Format

Provide your findings as:
- **Key Files:** List of relevant files with brief descriptions
- **Patterns to Follow:** Existing conventions and patterns
- **Integration Points:** Where my work will connect to existing code
- **Risks/Unknowns:** Things that might be harder than they appear
- **Questions for Clarification:** Things you couldn't determine from code alone
```

---

## Exploration Checklist

Before moving to the Plan phase, verify:

### Understanding (For Learning)
- [ ] I can explain what I found in my own words
- [ ] I understand WHY the code is structured this way
- [ ] I connected what I learned in RESEARCH to actual implementation
- [ ] I noted patterns I want to remember for future projects

### Context (For Implementation)
- [ ] I understand the existing architecture relevant to my task
- [ ] I've identified files I'll need to read or modify
- [ ] I know which patterns and conventions to follow
- [ ] I've noted potential risks and edge cases
- [ ] I have a list of questions/unknowns to resolve
- [ ] I understand how my work integrates with existing systems

---

## Tips for Effective Exploration with GitHub Copilot

### Be Thorough
- Read files completely, not just snippets
- Trace data flows end-to-end
- Look at tests to understand expected behavior
- Ask Copilot "why" questions, not just "what" questions

### Use Copilot's Workspace Context
```
@workspace How does error handling work in the retrieval system?
@workspace Show me all files that implement BaseRetriever
@workspace What patterns are used for async operations?
```

### Use Sub-agents for Parallel Exploration
```
Use a subagent to investigate how [specific system] works.
Focus on: [specific files or directories]
Report back: key patterns, integration points, constraints
```

### Ask Targeted Questions
Instead of: "How does this work?"
Ask: "In #file:retriever.py, how does the DenseRetriever handle empty query results? Show me the code path and explain the design decision."

### Verbalize Your Understanding
Tell Copilot what you think you understand:
```
I think the retrieval flow is: query → embed → search → rerank → return.
Is this correct? What am I missing?
```
This helps Copilot correct misconceptions and fills gaps in your mental model.

### Document Your Findings
Keep notes of what you discover — this becomes input for the Plan phase.
Ask Copilot to help summarize: "Summarize my key findings from this exploration session."

---

## Example: Explore Prompt for RAG System

```
# EXPLORE: Document Retrieval Pipeline

I'm starting work on the retrieval system for Project 2 (Enterprise Document Intelligence Platform).

## Context
- Project: Phase 2 - Enterprise Document Intelligence Platform
- Phase/Deliverable: Retrieval system with hybrid search
- Relevant documentation: LEARNING_PLAN.md Phase 2 section

## What I Need to Understand

1. **Existing Patterns:**
   - How do LangChain retrievers work?
   - What hybrid search patterns exist (dense + sparse)?
   - How do similar projects handle re-ranking?

2. **Architecture:**
   - How will the retriever connect to the vector database?
   - What's the flow from query → retrieval → re-ranking → output?
   - How should I structure the retriever classes?

3. **Dependencies:**
   - LangChain retriever abstractions
   - Vector database client (Qdrant/Weaviate)
   - Embedding model integration

4. **Constraints:**
   - Must handle context window limits
   - Must track retrieval metrics
   - Cost per query should be measurable

## Research Tasks

Please help me explore:
1. LangChain retriever patterns and best practices
2. Hybrid search implementation approaches
3. Re-ranking strategies (cross-encoders vs LLM re-ranking)
4. How to structure this for evaluation

## Output Format

Provide findings organized by:
- Key abstractions I should use
- Patterns from LangChain docs/examples
- Integration points with my document ingestion pipeline
- Potential challenges and how others solved them
```

---

## Common Explore Scenarios

### Exploring a Framework
```
I need to understand how [Framework] handles [capability].
Show me:
- Core abstractions and classes
- Configuration patterns
- Common usage examples
- Error handling conventions
```

### Exploring Existing Code
```
I need to understand the [component] in our codebase.
Trace through:
- Entry points and public interfaces
- Internal data flow
- Error handling
- Test coverage and edge cases
```

### Exploring Integration Points
```
I need to connect my [new feature] to the existing [system].
Identify:
- Where the integration should happen
- What interfaces I need to implement
- What data format is expected
- How errors are propagated
```

---

*This prompt is part of the Explore → Plan → Implement → Verify workflow.*
*After completing exploration, proceed to [02_PLAN.md](02_PLAN.md).*
