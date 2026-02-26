# PLAN Phase Prompt

> **Purpose:** Create a detailed, actionable implementation plan before writing code.  
> **Mode:** Design and specification — capture decisions, not code.  
> **Time Allocation:** ~20% of project time  
> **Copilot Mode:** Agent mode for interactive planning, Chat for quick decisions

---

## When to Use This Prompt

Use this prompt after completing exploration:
- You understand the codebase and constraints
- You've identified patterns to follow
- You're ready to decide HOW to build, not just WHAT to build

### GitHub Copilot Tips for Planning

**Interactive planning (Agent mode):**
- Use Copilot Agent for back-and-forth design discussions
- Ask it to challenge your assumptions and find edge cases
- Request alternative approaches before committing

**Quick decisions (Chat):**
- Use `@workspace` to reference existing patterns
- Ask "What are the trade-offs between X and Y for my use case?"
- Validate decisions: "Given my constraints, is this approach reasonable?"

**Learning through planning:**
- Ask Copilot to explain WHY certain approaches are better
- Request it to connect your plan to concepts from your RESEARCH
- Have it identify what you'll learn by implementing each phase

---

## The Plan Prompt

Copy and adapt this prompt for your AI assistant:

```
# CREATE PLAN: [Feature/Task Name]

## Context from Exploration

Based on my exploration, I've learned:
- [Key finding 1 with file references]
- [Key finding 2 with file references]
- [Pattern to follow]
- [Constraint to work within]

## Task Requirements

I need to implement: [Clear description of what you're building]

**Functional Requirements:**
1. [Requirement 1]
2. [Requirement 2]
3. [Requirement 3]

**Non-Functional Requirements:**
- Performance: [constraints]
- Cost: [constraints]
- Security: [considerations]

## Questions to Resolve

Before creating the plan, help me decide:
1. [Design decision 1 - e.g., "Should I use approach A or B?"]
2. [Design decision 2]
3. [Technical uncertainty]

## Plan Request

Create a detailed implementation plan that includes:

### 1. Overview
- What we're building and why
- High-level approach

### 2. What We're NOT Doing
- Explicit out-of-scope items to prevent scope creep

### 3. Implementation Phases
For each phase, include:
- **Overview:** What this phase accomplishes
- **Changes Required:** Specific files and modifications
- **Success Criteria:**
  - Automated verification (tests, linting, type checks)
  - Manual verification (what to check by hand)

### 4. Risk Mitigation
- Edge cases to handle
- Failure modes and fallbacks
- Dependencies on external systems

## Interaction Style

Work back and forth with me:
1. Share your initial understanding and open questions
2. Propose the phase structure before writing details
3. Let me confirm before you write the full plan
4. Be skeptical — challenge vague requirements
```

---

## Plan Template

Use this structure for your implementation plans:

```markdown
# [Feature Name] Implementation Plan

## Overview

[Brief description of what we're implementing and why]

## Current State Analysis

[What exists now, what's missing, key constraints discovered during exploration]

## Desired End State

[Specification of the desired end state and how to verify it]

### Key Discoveries from Exploration:
- [Finding 1 with file:line reference]
- [Finding 2]
- [Pattern to follow]

## What We're NOT Doing

- [Out of scope item 1]
- [Out of scope item 2]
- [Explicit limitation]

## Implementation Approach

[High-level strategy and reasoning for the chosen approach]

---

## Phase 1: [Descriptive Name]

### Overview
[What this phase accomplishes]

### Changes Required

#### 1. [Component/File Group]
**File:** `path/to/file.py`
**Changes:** [Summary of changes]

```python
# Key code structure (pseudocode or skeleton)
```

### Success Criteria

#### Automated Verification:
- [ ] Tests pass: `pytest tests/test_component.py`
- [ ] Type checking: `mypy src/`
- [ ] Linting: `ruff check src/`

#### Manual Verification:
- [ ] [Specific behavior to test manually]
- [ ] [Edge case to verify]

---

## Phase 2: [Next Phase Name]

[Same structure as Phase 1]

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [Risk 1] | Medium | High | [How to handle] |
| [Risk 2] | Low | Medium | [How to handle] |

## Open Questions

[Any unresolved questions to address during implementation]
```

---

## Planning Checklist

Before moving to Implement phase, verify:

- [ ] The plan covers all requirements
- [ ] Each phase is independently testable
- [ ] Success criteria are specific and measurable
- [ ] Out-of-scope items are explicitly listed
- [ ] Risks have been identified with mitigations
- [ ] The plan references specific files and patterns from exploration
- [ ] I could hand this plan to someone else and they could implement it

---

## Tips for Effective Planning with GitHub Copilot

### Work Iteratively with Copilot
Don't ask for the full plan in one shot:
1. Share your exploration findings first
2. Ask Copilot to propose a phase structure
3. Discuss and refine each phase
4. Request that Copilot challenge weak points

```
Based on my exploration, I think we need 3 phases:
1. [Phase 1 idea]
2. [Phase 2 idea]
3. [Phase 3 idea]

Does this make sense? What am I missing? What could go wrong?
```

### Have Copilot Be Your Skeptic
```
Review this plan critically:
- What edge cases haven't I considered?
- Where might this approach fail?
- What assumptions am I making that might be wrong?
- What would a senior engineer push back on?
```

### Connect Planning to Learning
```
For each phase, tell me:
1. What I'll learn by implementing it
2. What concepts from my research this applies
3. What skills this will build
```

### Be Specific
- Include file paths and line numbers
- Reference patterns from exploration
- Make success criteria measurable
- Include specific commands to run

### Keep Phases Small
- Each phase should be completable in 1-2 hours
- Each phase should be independently verifiable
- If a phase is too big, ask Copilot to split it

### Save Your Plan
Ask Copilot to create a PLAN.md file in your phase directory:
```
Create a PLAN.md file with this implementation plan in the Phase X directory.
```

---

## Example: Plan Prompt for RAG Retrieval

```
# CREATE PLAN: Hybrid Search Retrieval System

## Context from Exploration

Based on my exploration, I've learned:
- LangChain's BaseRetriever provides the core abstraction (langchain_core/retrievers.py)
- Qdrant client supports both dense and sparse search
- Re-ranking with cross-encoders is common pattern (sentence-transformers library)
- Need to implement custom retriever inheriting from BaseRetriever

## Task Requirements

I need to implement: A hybrid retrieval system combining dense vector search with BM25 sparse search, with re-ranking.

**Functional Requirements:**
1. Query vector database with semantic search (dense)
2. Query with BM25 keyword search (sparse)
3. Combine and re-rank results
4. Return top-k documents with relevance scores

**Non-Functional Requirements:**
- Performance: <500ms p95 latency
- Cost: Track embedding API calls
- Observability: Log retrieval metrics

## Questions to Resolve

1. Should I store BM25 index in Qdrant or use separate index (e.g., Elasticsearch)?
2. What fusion strategy: reciprocal rank fusion or weighted combination?
3. Re-rank all results or just top-N before re-ranking?

## Plan Request

Create a detailed implementation plan with:
- Phase 1: Dense retrieval implementation
- Phase 2: Sparse retrieval integration
- Phase 3: Fusion and re-ranking
- Phase 4: Evaluation and metrics

Work back and forth with me — share open questions and phase outline before writing details.
```

---

## Common Planning Scenarios

### Planning a New Module
```
Create a plan for implementing [module name].
Consider:
- How it integrates with existing code
- What interfaces it needs to expose
- How it will be tested
- What could go wrong
```

### Planning a Refactor
```
Create a plan for refactoring [component].
Current state: [description]
Target state: [description]
Constraints: 
- Must maintain backwards compatibility
- Must not break existing tests
- Must be done incrementally
```

### Planning an Integration
```
Create a plan for integrating [external service/library].
Consider:
- Authentication and configuration
- Error handling and fallbacks
- Rate limiting and cost control
- Testing strategy (mocks vs real)
```

---

## Plan Review Checklist

Ask your AI assistant to review the plan:

```
Review this plan for:
- Completeness: Are all requirements covered?
- Testability: Can each phase be verified independently?
- Risk: What could go wrong that we haven't addressed?
- Scope creep: Is anything unnecessary included?
- Clarity: Could someone else implement this?
```

---

*This prompt is part of the Explore → Plan → Implement → Verify workflow.*
*After completing planning, proceed to [03_IMPLEMENT.md](03_IMPLEMENT.md).*
