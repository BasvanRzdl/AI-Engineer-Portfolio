# VERIFY Phase Prompt

> **Purpose:** Validate that implementation meets requirements, verify quality, and document completion.  
> **Mode:** Review and testing — critical evaluation of the work done.  
> **Time Allocation:** ~15% of project time (remaining 5% for reflection)  
> **Copilot Mode:** Agent for comprehensive review, Chat for specific checks

---

## When to Use This Prompt

Use this prompt when:
- Implementation is complete (all phases done)
- You need to verify the work before calling it "done"
- You want a systematic review of quality and completeness

### GitHub Copilot Tips for Verification

**Comprehensive review (Agent mode):**
```
Review my implementation against the plan. Check:
1. All requirements are met
2. Code quality issues
3. Missing error handling
4. Test coverage gaps
5. Security concerns

Be critical — I want to learn from mistakes.
```

**Quick checks (Chat):**
- `@workspace /tests` to check test status
- Ask about specific code: "Are there any issues in #file:retriever.py?"
- "What edge cases might I have missed?"

**Learning through verification:**
- Ask Copilot to explain any issues it finds
- Request it to teach you better patterns
- Have it suggest what you should study next based on gaps

---

## The Verify Prompt

Copy and adapt this prompt for your AI assistant:

```
# VALIDATE IMPLEMENTATION: [Feature/Task Name]

## Implementation Summary

What was implemented: [Brief description]
Plan reference: [Path to implementation plan]
Phases completed: [List of phases]

## Verification Request

Please help me validate this implementation:

### 1. Automated Verification
Run all automated checks:
- Unit tests: [command]
- Integration tests: [command]
- Type checking: [command]
- Linting: [command]
- Security scan: [command if applicable]

### 2. Plan Compliance
For each phase in the plan:
- Verify completion status matches actual code
- Check that success criteria are met
- Identify any deviations from plan

### 3. Code Quality Review
Review the implementation for:
- Security vulnerabilities
- Performance issues
- Error handling gaps
- Missing edge cases
- Code style violations
- Documentation completeness

### 4. Manual Verification
Guide me through manual testing:
- [Test scenario 1]
- [Test scenario 2]
- [Edge case to verify]

### 5. Issues Found
Document:
- 🔴 Must Fix: Issues that block completion
- 🟡 Should Improve: Issues for production readiness
- 🟢 Consider: Suggestions for excellence

## Output Format

Provide a verification report with:
- Overall status (Pass/Fail/Needs Work)
- Detailed findings by category
- Recommended next steps
```

---

## Verification Report Template

```markdown
# Verification Report: [Feature Name]

**Date:** YYYY-MM-DD
**Implementation Plan:** [path to plan]
**Status:** ✅ PASSED | ⚠️ NEEDS WORK | ❌ FAILED

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: [Name] | ✅ Complete | [Any notes] |
| Phase 2: [Name] | ✅ Complete | [Any notes] |
| Phase 3: [Name] | ⚠️ Partial | [What's missing] |

---

## Automated Verification Results

### Tests
```
pytest tests/ -v
================================
X passed, Y failed, Z skipped
```
Status: ✅ Pass | ❌ Fail

### Type Checking
```
mypy src/
================================
Success: no issues found
```
Status: ✅ Pass | ❌ Fail

### Linting
```
ruff check src/
================================
All checks passed
```
Status: ✅ Pass | ❌ Fail

---

## Code Quality Findings

### 🔴 Must Fix (Blocking)

1. **[Issue Title]**
   - Location: `file.py:line`
   - Problem: [Description]
   - Fix: [How to fix]

### 🟡 Should Improve (For Production)

1. **[Issue Title]**
   - Location: `file.py:line`
   - Problem: [Description]
   - Recommendation: [How to improve]

### 🟢 Consider (Excellence)

1. **[Suggestion Title]**
   - Location: `file.py:line`
   - Opportunity: [Description]
   - Benefit: [Why this would help]

---

## Manual Verification Checklist

- [x] [Scenario 1]: [Result]
- [x] [Scenario 2]: [Result]
- [ ] [Scenario 3]: [Not yet tested]

---

## Security Review

- [ ] No hardcoded credentials
- [ ] Input validation present
- [ ] Error messages don't leak sensitive info
- [ ] Dependencies are up to date
- [ ] [Project-specific security checks]

---

## Performance Considerations

- [ ] No obvious N+1 queries
- [ ] Appropriate caching in place
- [ ] Async operations used where beneficial
- [ ] Resource cleanup (connections, files) handled

---

## Documentation Status

- [ ] Code has appropriate docstrings
- [ ] README updated if needed
- [ ] API documentation complete
- [ ] Architecture decisions documented

---

## Recommendations

### Before Merging
1. [Action item 1]
2. [Action item 2]

### Future Improvements
1. [Improvement for later]
2. [Nice to have]

---

## Verification Sign-off

**Verified by:** [Your name]
**Date:** YYYY-MM-DD
**Decision:** Ready to merge | Needs fixes | Major rework needed
```

---

## Verification Checklist

### Functional Completeness
- [ ] All requirements from the plan are implemented
- [ ] All success criteria are met
- [ ] Edge cases are handled
- [ ] Error scenarios work correctly

### Code Quality
- [ ] Code follows project conventions
- [ ] No copy-pasted code without understanding
- [ ] Appropriate abstractions used
- [ ] No obvious performance issues

### Testing
- [ ] Unit tests cover main functionality
- [ ] Edge cases have test coverage
- [ ] Tests are readable and maintainable
- [ ] All tests pass

### Documentation
- [ ] Public functions have docstrings
- [ ] Complex logic is commented
- [ ] README updated if user-facing changes
- [ ] API docs complete if applicable

### Security
- [ ] No credentials in code
- [ ] Input is validated
- [ ] Errors are handled safely
- [ ] Dependencies are vetted

### Observability
- [ ] Appropriate logging in place
- [ ] Metrics tracked (if applicable)
- [ ] Errors are traceable

### Learning Verification
- [ ] I can explain how this implementation works without looking at the code
- [ ] I understand the design decisions and could defend them
- [ ] I learned something new and documented it
- [ ] I know what I would do differently next time

---

## Common Verification Scenarios

### Verifying a RAG System

```
Verify the RAG implementation:

Automated:
- Run retrieval quality tests
- Check embedding costs are tracked
- Verify latency meets requirements

Manual:
- Query with known-good questions
- Query with edge cases (empty, very long)
- Check source attribution accuracy
- Verify "I don't know" responses

Quality:
- Chunking produces sensible segments
- Retrieval returns relevant documents
- Generation is grounded in sources
```

### Verifying an Agent

```
Verify the agent implementation:

Automated:
- Run tool execution tests
- Verify state machine transitions
- Check error handling paths

Manual:
- Walk through happy path conversation
- Test tool failures and recovery
- Verify human escalation works
- Check guardrails are enforced

Quality:
- Agent reasoning is traceable
- Tool calls are appropriate
- Memory works correctly
- Cost tracking is accurate
```

### Verifying an API

```
Verify the API implementation:

Automated:
- All endpoints return correct responses
- Authentication works
- Rate limiting enforced
- Input validation rejects bad data

Manual:
- Test from client perspective
- Verify error messages are helpful
- Check response times
- Test concurrent requests

Quality:
- API follows REST conventions
- Responses are consistent
- Documentation matches behavior
- Versioning is handled
```

---

## What to Do With Findings

### 🔴 Must Fix Issues

These block completion. Fix them before moving on:

```
Found must-fix issue: [description]

Fixing now:
1. [Change 1]
2. [Change 2]

Re-running verification...
```

### 🟡 Should Improve Issues

Track these for follow-up:

```markdown
## Technical Debt Log

### [Date]: [Feature Name]

Should improve (from verification):
1. [Issue] - [planned fix approach]
2. [Issue] - [planned fix approach]

Will address: [when - e.g., before Phase 3 or during polish]
```

### 🟢 Consider Items

Note for future reference:

```markdown
## Future Improvements

Ideas from verification:
- [Suggestion]: [potential benefit]
- [Suggestion]: [potential benefit]
```

---

## Copilot Review Prompts

Use these prompts to get thorough reviews from Copilot:

### Code Quality Review
```
Review this implementation critically. Act as a senior engineer doing a code review.
Point out:
- Code smells or anti-patterns
- Missing error handling
- Performance concerns
- Places where I could have used better patterns
- Anything that would fail in production

Be specific and reference line numbers.
```

### Security Review
```
Do a security review of my implementation:
- Check for injection vulnerabilities
- Look for hardcoded secrets
- Verify input validation
- Check error messages for info leakage
- Review any authentication/authorization code
```

### Learning Review
```
Based on the code I wrote, assess my understanding:
- Where do I seem confident vs uncertain?
- What concepts should I study more deeply?
- What patterns did I use well?
- What patterns did I misuse or could use better?
- Grade my implementation (A-F) with explanation.
```

---

## Verification Anti-Patterns

### ❌ Don't Do This

| Anti-Pattern | Problem | Instead |
|--------------|---------|----------|
| Skipping manual tests | Automation doesn't catch everything | Do both automated and manual |
| Verifying only happy path | Edge cases break in production | Test errors and boundaries |
| Self-rubber-stamping | Missing your own blind spots | Ask Copilot to be critical |
| Accepting without understanding | You don't learn | Ask Copilot to explain issues |
| Ignoring warnings | Small issues become big problems | Address or document all findings |
| Rushing to "done" | Technical debt accumulates | Take time to verify properly |

---

## Post-Verification Reflection

After verification, have a learning conversation with Copilot:

```
I just completed [project/phase]. Help me reflect:

1. What patterns did I use that I should remember?
2. What mistakes did I make that I can learn from?
3. What concepts from my learning path did this project reinforce?
4. What should I study next to improve?
5. How could I have approached this more efficiently?
```

Then update PROGRESS.md:

```markdown
## Project [N] Verification Complete

**Date:** YYYY-MM-DD
**Verification Status:** Passed

### What Went Well
- [Positive finding]
- [Positive finding]

### Issues Found and Fixed
- [Issue] → [Fix]
- [Issue] → [Fix]

### Deferred Improvements
- [Item for later]
- [Item for later]

### Learnings
- [New concept or pattern I learned]
- [Framework/library knowledge gained]
- [Design decision I now understand better]

### What I'd Do Differently
- [Improvement for next time]
- [Different approach I'd try]

### Study Topics for Follow-up
- [Topic to go deeper on]
- [Gap in knowledge to fill]
```

---

## Final Checklist Before "Done"

- [ ] All automated checks pass
- [ ] All manual verification complete
- [ ] Must-fix issues resolved
- [ ] Should-improve issues logged
- [ ] Code committed with clean history
- [ ] Documentation updated
- [ ] PROGRESS.md updated
- [ ] Ready for next phase or project

---

*This prompt is part of the Explore → Plan → Implement → Verify workflow.*
*After verification, complete your project reflection in PROGRESS.md.*
