# IMPLEMENT Phase Prompt

> **Purpose:** Execute the implementation plan systematically, with verification at each step.  
> **Mode:** Active coding — write, test, commit incrementally.  
> **Time Allocation:** ~50% of project time  
> **Copilot Mode:** Agent mode for guided implementation, Inline suggestions for flow

---

## When to Use This Prompt

Use this prompt when:
- You have a reviewed and approved implementation plan
- You understand what needs to be built
- You're ready to write production code

### GitHub Copilot Modes for Implementation

**Agent Mode (Recommended for learning):**
- Copilot reads your plan and implements with you
- You see the reasoning and can ask questions
- Good for understanding WHY code is written a certain way
- Use: "Implement Phase 1 from my plan. Explain each decision."

**Inline Suggestions (For flow state):**
- Tab-complete while coding
- Good when you know what you want
- Fast iteration on familiar patterns

**Chat (For quick help):**
- Debug specific issues
- Ask about syntax or patterns
- Get quick code snippets

### Learning While Implementing

Don't just let Copilot write code — engage with it:
- Ask WHY it chose a particular approach
- Request explanations of patterns you don't recognize
- Have it teach you as it implements

---

## The Implement Prompt

Copy and adapt this prompt for your AI assistant:

```
# IMPLEMENT PLAN: [Feature/Task Name]

## Plan Reference
Implementation plan: [path to your plan file or paste the plan]

## Getting Started

1. Read the plan completely
2. Check for any existing progress (checkmarks)
3. Read all files mentioned in the plan
4. Create a todo list to track progress

## Implementation Guidelines

**Work incrementally:**
- Implement one phase at a time
- Verify success criteria before moving to next phase
- Commit after each meaningful change
- Use Copilot's todo list to track progress

**Quality standards:**
- Follow existing code patterns identified in exploration
- Include appropriate error handling
- Add logging for observability
- Write tests as you go (or immediately after)

**Learning standards:**
- Understand every line Copilot generates before accepting
- Ask for explanations of unfamiliar patterns
- Note new techniques you learn for future reference

**Communication with Copilot:**
- Tell Copilot to show you what it's implementing before writing
- Ask it to explain any deviations from the plan
- Request it stop and ask if something doesn't match the plan

```
Follow my implementation plan. For each change:
1. Tell me what you're about to implement
2. Explain why you're doing it this way
3. Wait for my confirmation before proceeding
4. After implementing, show me what to verify
```

## Current Phase

Starting with: Phase [N]: [Phase Name]

## Constraints

- [Project-specific constraints]
- [Coding standards to follow]
- [Testing requirements]
```

---

## Implementation Workflow

### Step 1: Read and Prepare

```
Read the implementation plan and all referenced files.
Create a todo list for tracking progress.
Confirm understanding before starting.
```

### Step 2: Implement One Phase

```
Implementing Phase [N]: [Phase Name]

I will:
1. [First change]
2. [Second change]
3. [Third change]

Proceeding with implementation...
```

### Step 3: Verify Phase

```
Phase [N] complete. Running verification:

Automated checks:
- [ ] pytest tests/test_component.py
- [ ] mypy src/
- [ ] ruff check src/

Manual verification needed:
- [ ] [Specific behavior to test]

Results: [pass/fail with details]
```

### Step 4: Commit and Continue

```
Committing Phase [N] changes:
git add [files]
git commit -m "[descriptive message]"

Moving to Phase [N+1]...
```

---

## Implementation Checklist

For each phase:

- [ ] Read the phase requirements completely
- [ ] Understand how it connects to previous phases
- [ ] Implement the changes
- [ ] Run automated verification (tests, types, linting)
- [ ] Perform manual verification
- [ ] Commit with a clear message
- [ ] Update plan with checkmarks if tracking in file

---

## Tips for Effective Implementation with GitHub Copilot

### Stay Focused
- Follow the plan, don't let Copilot add unplanned features
- If Copilot suggests scope expansion, note it for later
- Ask Copilot to stay within the current phase

```
Stay focused on Phase [N] only. Don't implement features from later phases.
If you see opportunities for improvement, note them but don't implement.
```

### Understand Before Accepting
**Don't blindly accept Copilot suggestions:**
- Read generated code carefully
- Ask "Why did you use X instead of Y?"
- Request simpler alternatives if code is too complex
- Have Copilot explain unfamiliar patterns

```
Explain this code you just generated:
- What pattern is this?
- Why is it appropriate here?
- What are the alternatives?
```

### Handle Deviations Gracefully
If the plan doesn't match reality, ask Copilot:
```
The plan says [X] but I'm seeing [Y] in the code.
What are my options? Recommend an approach and explain why.
```

### Test As You Go
- Ask Copilot to write tests alongside implementation
- Run tests in the terminal after each change
- Have Copilot fix failures with explanations

```
Write a test for the function you just created.
Explain what edge cases you're testing and why.
```

### Commit Atomically
Ask Copilot to help with commits:
```
Generate a commit message for the changes I just made.
Follow conventional commits format.
```

---

## Example: Implement Prompt for RAG Retrieval

```
# IMPLEMENT PLAN: Hybrid Search Retrieval System

## Plan Reference
Plan: Phase 2 - Enterprise Document Intelligence Platform
Retrieval System Implementation Plan (created during planning phase)

## Getting Started

Read the plan and all referenced files:
- src/retrieval/base_retriever.py
- src/retrieval/vector_store.py  
- tests/test_retrieval.py

## Current Phase

Starting with: Phase 1 - Dense Retrieval Implementation

Expected changes:
1. Create DenseRetriever class inheriting from BaseRetriever
2. Implement _get_relevant_documents method
3. Add embedding query logic
4. Write unit tests

## Constraints

- Use async patterns throughout
- Track API costs per query
- Log retrieval metrics at INFO level
- Follow existing error handling patterns from src/utils/errors.py
```

---

## Handling Common Situations

### When Implementation Gets Complex

```
This is more complex than the plan anticipated.

Current situation: [what you're encountering]
Plan assumption: [what the plan assumed]
Actual complexity: [what you're discovering]

Options:
1. Simplify approach: [description]
2. Extend the phase: [description]
3. Split into sub-phases: [description]

Which approach should we take?
```

### When Tests Fail

```
Tests are failing after Phase [N] implementation.

Failing test: [test name]
Error: [error message]
Root cause hypothesis: [what you think is wrong]

I've tried:
1. [Attempt 1]
2. [Attempt 2]

Need help with: [specific question]
```

### When You Need to Deviate

```
I need to deviate from the plan:

Reason: [why deviation is necessary]
Original approach: [what plan said]
Proposed change: [new approach]
Impact: [how this affects other phases]

Approval to proceed?
```

### Resuming After a Break

```
Resuming implementation of [feature].

Progress so far:
- [x] Phase 1: Complete
- [x] Phase 2: Complete  
- [ ] Phase 3: In progress (stopped at [point])
- [ ] Phase 4: Not started

Picking up from: [where you stopped]
```

---

## Code Quality Standards

### Always Include

```python
# Error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise

# Type hints
def process_document(doc: Document, options: ProcessOptions) -> ProcessResult:
    ...

# Docstrings
def complex_function(param: str) -> dict:
    """
    Brief description.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param is invalid
    """
    ...

# Logging
logger.info(f"Processing document: {doc.id}")
logger.debug(f"Full document content: {doc.content[:100]}...")
```

### Testing Pattern

```python
# Test file structure
class TestComponentName:
    """Tests for ComponentName functionality."""
    
    def test_happy_path(self):
        """Test normal operation."""
        ...
    
    def test_edge_case_empty_input(self):
        """Test handling of empty input."""
        ...
    
    def test_error_handling(self):
        """Test proper error propagation."""
        ...
```

---

## Progress Tracking

Update your plan or progress file after each phase:

```markdown
## Phase 1: Dense Retrieval ✅

Completed: 2026-02-25
Changes:
- Created src/retrieval/dense_retriever.py
- Added tests in tests/test_dense_retriever.py
- Updated src/retrieval/__init__.py

Verification:
- [x] All tests pass
- [x] Type checking clean
- [x] Manual test: retrieves relevant docs

Notes:
- Chose cosine similarity over euclidean (better for normalized embeddings)
- Added retry logic for embedding API calls
```

---

## When to Stop and Ask

Stop implementation and ask for guidance when:

- ❓ The plan doesn't match what you see in the code
- ❓ You discover a significant new constraint
- ❓ Tests are failing and you can't figure out why
- ❓ You need to make a decision not covered by the plan
- ❓ Implementation would take significantly longer than estimated

Don't struggle alone — asking questions is part of learning.

---

*This prompt is part of the Explore → Plan → Implement → Verify workflow.*
*After completing implementation, proceed to [04_VERIFY.md](04_VERIFY.md).*
