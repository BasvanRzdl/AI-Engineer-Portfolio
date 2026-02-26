# AI-Assisted Coding: Best Practices & Personal Way of Working

> **Purpose:** Establish a personal methodology for AI-assisted development that maximizes productivity, learning, and code quality.  
> **Tools Covered:** GitHub Copilot (VS Code), Claude Code, and general principles applicable to all AI coding assistants.  
> **Last Updated:** February 25, 2026

---

## Philosophy: Human-AI Collaboration

AI coding assistants are not replacements for engineering judgment. They are **force multipliers** that work best when you understand their strengths, limitations, and how to communicate effectively with them.

### The Fundamental Mindset

| Traditional Coding | AI-Assisted Coding |
|-------------------|---------------------|
| You write every line | You describe intent, AI proposes, you verify |
| Manual research | AI can search, synthesize, and suggest |
| Linear workflow | Iterative conversation |
| Limited to what you know | Access to patterns you haven't seen |
| Slow but controlled | Fast but requires validation |

**Key Principle:** You are always in control. The AI is a capable colleague, not an authority. Every suggestion must pass your engineering judgment.

---

## Part 1: Context Is Everything

The single most important factor in AI-assisted coding is **context**. AI assistants generate responses based on the context you provide. Better context = better outputs.

### 1.1 Provide Rich, Relevant Context

#### Open Relevant Files
- Keep files open that are relevant to your current task
- Close unrelated files when switching contexts
- AI assistants use open files to understand patterns, types, and conventions

#### Reference Files Explicitly
```
# In GitHub Copilot Chat
@file:src/auth/session.ts explain the session handling

# In Claude Code
Look at @src/auth/ and understand how we handle sessions
```

#### Use Top-Level Comments
When starting a new file or major feature, add a descriptive comment at the top:

```python
"""
User Authentication Module

This module handles OAuth2 authentication with multiple providers.
It uses JWT tokens stored in HTTP-only cookies.
Key dependencies: authlib, pyjwt
Key patterns: Factory pattern for providers, Strategy pattern for token handling

Author: [You]
Date: 2026-02-25
"""
```

#### Set Imports Explicitly
Don't let AI guess your dependencies. Specify them:

```python
# I want to use these specific libraries
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
```

### 1.2 Use Meaningful Names

AI assistants infer intent from names. Quality of names directly affects quality of suggestions.

| Poor | Better | Best |
|------|--------|------|
| `data` | `user_data` | `authenticated_user_profile` |
| `process()` | `process_order()` | `validate_and_submit_order()` |
| `temp` | `buffer` | `message_buffer_for_retry` |

### 1.3 Manage Context Windows

All AI assistants have limited context windows. As conversations grow, older context gets lost or compressed.

**Signs of context degradation:**
- AI starts repeating earlier mistakes
- AI "forgets" instructions you gave earlier
- Responses become less specific to your codebase

**Mitigation strategies:**

| Strategy | When to Use |
|----------|-------------|
| Clear/reset context | Between unrelated tasks |
| Start new conversation | When debugging a specific issue |
| Summarize and restart | After long exploration sessions |
| Use persistent instructions | For rules that always apply (CLAUDE.md, custom instructions) |

---

## Part 2: Prompt Engineering for Code

### 2.1 The SCOPE Framework

When asking AI to write or modify code, structure your request using **SCOPE**:

| Element | Description | Example |
|---------|-------------|---------|
| **S**ituation | What exists, what's the context | "We have a FastAPI app with SQLAlchemy models" |
| **C**onstraints | Limitations, requirements, non-negotiables | "Must be async, must not break existing tests" |
| **O**bjective | What you want to achieve | "Add rate limiting to the /api/users endpoint" |
| **P**atterns | Reference existing patterns or examples | "Follow the pattern in auth_middleware.py" |
| **E**xpectations | How you'll verify success | "It should pass these test cases: [list]" |

**Example prompt using SCOPE:**

```
SITUATION: We have a FastAPI application with existing middleware for auth.
The rate limiting should integrate with our Redis cache.

CONSTRAINTS:
- Must be async
- Must not affect existing auth middleware
- Rate limits should be configurable per endpoint
- Must handle Redis connection failures gracefully

OBJECTIVE: Add rate limiting middleware that limits requests per user per minute.

PATTERNS: Look at src/middleware/auth.py for how we structure middleware.
Use the existing Redis connection from src/cache/redis_client.py.

EXPECTATIONS: 
- 429 response when limit exceeded
- X-RateLimit-Remaining header in responses
- Logs rate limit events at INFO level
```

### 2.2 Start General, Then Get Specific

Begin with the big picture, then add requirements:

```
# First prompt
Write a Python function that validates email addresses.

# Follow-up with specifics
The function should:
- Return True/False
- Handle edge cases: empty string, None input
- Support international domains
- Include these test cases: user@example.com (true), invalid (false), user@.com (false)
```

### 2.3 Break Complex Tasks into Steps

Don't ask for everything at once. Use progressive refinement:

```
# Step 1: Architecture
"I need to build a document processing pipeline. 
What components do I need? Create a high-level design."

# Step 2: First component
"Let's implement the document loader first. 
It should handle PDF, DOCX, and plain text."

# Step 3: Integration
"Now connect the document loader to the chunking component."

# Step 4: Testing
"Write tests for the document loader."
```

### 2.4 Provide Examples

Examples are powerful context. Include:

- Sample input data
- Expected output format
- Similar implementations to follow

```
Write a function that parses log entries.

Example input:
"2026-02-25T14:30:00Z INFO [auth] User login successful user_id=123"
"2026-02-25T14:30:01Z ERROR [db] Connection timeout after 30s"

Expected output:
{
    "timestamp": "2026-02-25T14:30:00Z",
    "level": "INFO",
    "component": "auth",
    "message": "User login successful",
    "metadata": {"user_id": "123"}
}
```

---

## Part 3: Workflow Patterns

### 3.1 The Explore → Plan → Implement → Verify Cycle

This is the most effective workflow for substantial features:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. EXPLORE (Read-only)                                         │
│     - Understand the codebase                                   │
│     - Identify relevant files and patterns                      │
│     - Ask questions, don't make changes                         │
├─────────────────────────────────────────────────────────────────┤
│  2. PLAN (Design before coding)                                 │
│     - Create detailed implementation plan                       │
│     - Identify edge cases and risks                             │
│     - Get AI to propose architecture                            │
├─────────────────────────────────────────────────────────────────┤
│  3. IMPLEMENT (Write code)                                      │
│     - Follow the plan                                           │
│     - Implement in small, testable increments                   │
│     - Commit frequently                                         │
├─────────────────────────────────────────────────────────────────┤
│  4. VERIFY (Test and validate)                                  │
│     - Run tests                                                 │
│     - Check for edge cases                                      │
│     - Review code quality                                       │
└─────────────────────────────────────────────────────────────────┘
```

**When to use:** Complex features, unfamiliar codebases, anything touching multiple files.

**When to skip:** Simple fixes, obvious changes, well-understood modifications.

### 3.2 Test-Driven Development with AI

AI excels at TDD workflows:

```
# Step 1: Define behavior through tests
"Write unit tests for a UserAuthenticator class that:
- Validates credentials against a user store
- Returns a JWT token on success
- Raises AuthenticationError on failure
- Handles rate limiting"

# Step 2: Implement to pass tests
"Now implement UserAuthenticator to pass these tests"

# Step 3: Refactor with confidence
"Refactor this implementation to use async/await while keeping tests passing"
```

### 3.3 Debugging Workflow

AI assistants are excellent debugging partners:

```
# Effective debugging prompt
I'm seeing an error when I run npm test:

[paste error message]

Here's the relevant code:
[paste or reference file]

What I've already tried:
- Checked that dependencies are installed
- Verified the environment variables

What I think might be wrong:
- The mock might not be set up correctly
- The async handling looks suspicious

Can you help me identify the root cause?
```

**Key elements:**
1. Include the actual error message
2. Reference relevant code
3. List what you've already tried
4. Share your hypothesis

### 3.4 The Interview Pattern

For complex or ambiguous features, let AI interview you:

```
I want to build a notification system for our app.
Interview me to understand the requirements.
Ask about:
- Technical implementation details
- Edge cases I might not have considered
- Integration points with existing systems
- Performance requirements

Keep asking until you have enough to write a complete spec.
```

This surfaces requirements you might not have considered.

---

## Part 4: Code Quality & Verification

### 4.1 Always Verify AI Output

**Never accept code you don't understand.**

Verification checklist:
- [ ] I understand what this code does
- [ ] I've read through the logic, not just skimmed
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] It follows project conventions
- [ ] Tests exist or have been added
- [ ] Security considerations addressed

### 4.2 Give AI Verification Criteria

The best way to get quality output is to give AI a way to check itself:

```
Implement a validateEmail function.

Test cases that must pass:
- "user@example.com" → true
- "invalid" → false  
- "user@.com" → false
- "" → false
- null → false (should not throw)
- "user+tag@example.com" → true

After implementing, run the tests and fix any failures.
```

### 4.3 Use AI for Code Review

After implementation, use AI as a reviewer:

```
Review this code for:
- Security vulnerabilities
- Performance issues
- Edge cases not handled
- Code style violations
- Opportunities to simplify

Be critical. I want to catch issues before merging.
```

### 4.4 Automated Quality Gates

Set up automated checks that run on AI-generated code:

- **Linting:** ESLint, Pylint, Ruff
- **Type checking:** TypeScript, mypy, Pyright
- **Security scanning:** Bandit, npm audit
- **Test suite:** Ensure all tests pass
- **Code coverage:** No decrease in coverage

---

## Part 5: Tool-Specific Best Practices

### 5.1 GitHub Copilot in VS Code

#### Inline Completions
- Use Tab to accept, Esc to dismiss
- Use `Ctrl+Enter` to see multiple suggestions
- Write descriptive comments before functions to guide completion

#### Copilot Chat
- Use `@workspace` to give context about your entire project
- Use `#file:path/to/file.py` to reference specific files
- Use slash commands: `/explain`, `/fix`, `/tests`, `/doc`

#### Chat Participants (VS Code)
| Participant | Purpose |
|-------------|---------|
| `@workspace` | Questions about your entire codebase |
| `@vscode` | VS Code features and settings |
| `@terminal` | Help with terminal commands |

#### Best Practices
1. Keep relevant files open
2. Close irrelevant files when context-switching
3. Use threads for different topics
4. Delete unhelpful responses to clean context
5. Highlight code before asking questions about it

### 5.2 Claude Code

#### Core Commands
| Command | Purpose |
|---------|---------|
| `/clear` | Reset context between tasks |
| `/compact` | Summarize context to free space |
| `/resume` | Continue previous session |
| `/model` | Switch between models |
| `Shift+Tab` | Toggle permission modes |
| `Esc` | Interrupt current action |
| `Esc+Esc` | Open rewind menu |

#### CLAUDE.md Configuration
Create a `CLAUDE.md` file in your project root with persistent instructions:

```markdown
# Project Context

## Build Commands
- `npm run dev` - Start development server
- `npm test` - Run tests
- `npm run lint` - Run linter

## Code Style
- Use TypeScript strict mode
- Prefer functional components in React
- Use async/await over .then()
- Error messages should be user-friendly

## Architecture
- API routes in /src/api/
- Components in /src/components/
- Business logic in /src/services/
- Types in /src/types/

## Important Constraints
- Never commit .env files
- All API calls must use the apiClient wrapper
- User data must be sanitized before logging
```

#### Context Management
- Use `/clear` frequently between unrelated tasks
- Use subagents for investigation to preserve main context
- Name sessions with `/rename` for easy resumption
- Use Plan Mode (`--permission-mode plan`) for exploration

#### Subagents
Delegate focused work to subagents to keep main context clean:

```
Use a subagent to investigate how our authentication system works.
Report back with a summary of the key files and patterns.
```

### 5.3 General Principles (All Tools)

| Principle | Implementation |
|-----------|----------------|
| Context is king | Provide relevant files, clear descriptions, examples |
| Iterate, don't restart | Build on previous responses when possible |
| Verify everything | Never accept code you don't understand |
| Clear frequently | Fresh context often beats accumulated context |
| Be specific | Vague requests get vague responses |
| Course-correct early | Interrupt and redirect as soon as you notice drift |

---

## Part 6: Anti-Patterns to Avoid

### 6.1 Common Mistakes

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| **The Kitchen Sink Session** | Mixing unrelated tasks in one conversation | Use `/clear` between tasks, start new threads |
| **Correction Spiral** | Repeatedly correcting without progress | After 2-3 failed corrections, clear and restart with better prompt |
| **Blind Acceptance** | Accepting code without understanding | Always read and understand before accepting |
| **Vague Prompts** | "Fix the bug" without context | Provide error messages, relevant code, what you've tried |
| **Overloaded CLAUDE.md** | Too many instructions that get ignored | Keep it short, only include what AI wouldn't know |
| **Skipping Verification** | Not testing AI-generated code | Always run tests, check edge cases |
| **Context Hoarding** | Never clearing context | Context degrades; clear it regularly |

### 6.2 When AI Struggles

AI assistants tend to struggle with:

- **Novel algorithms:** They pattern-match from training data
- **Highly specific business logic:** They don't know your domain
- **Complex state management:** Hard to track across files
- **Security-critical code:** May miss subtle vulnerabilities
- **Performance optimization:** May not understand your constraints
- **Up-to-date APIs:** Training cutoffs mean outdated knowledge

**When AI struggles:** Take over, write the code yourself, or provide more examples and constraints.

---

## Part 7: Learning While Using AI

Since your goal is to learn, not just produce code, adopt these practices:

### 7.1 Explain-Before-Accept Rule

Before accepting any suggestion, explain to yourself (or rubber duck) what it does. If you can't explain it, don't accept it.

### 7.2 Ask "Why" Regularly

```
You suggested using a WeakMap here instead of a regular Map.
Explain why that's the better choice in this context.
```

### 7.3 Request Alternative Approaches

```
Show me 2-3 different ways to implement this.
Explain the tradeoffs between them.
```

### 7.4 Build Mental Models

After completing a feature with AI assistance:

1. Summarize what you learned
2. Identify patterns you can reuse
3. Note any concepts you need to study more

### 7.5 Deliberate Practice

- Sometimes implement without AI to reinforce learning
- Use AI to generate challenges or exercises
- Review AI suggestions critically, even when correct

---

## Part 8: Enterprise Considerations

When working in enterprise environments (as you will be consulting for large companies), additional considerations apply:

### 8.1 Security

- **Never paste secrets** into AI prompts
- **Be cautious with proprietary code** — understand your organization's AI policy
- **Review for security** — AI may generate vulnerable patterns
- **Sanitize examples** — Remove PII, credentials, internal URLs from prompts

### 8.2 Consistency

- **Establish team standards** for AI usage
- **Document AI-assisted code** so others know to review carefully
- **Use shared instructions** (CLAUDE.md, workspace settings) for consistency

### 8.3 Compliance

- Know what data can be sent to AI services
- Understand data retention policies of AI tools
- Use enterprise versions with appropriate data handling

---

## Part 9: My Personal AI Workflow

Based on everything above, here's the workflow I'll follow for this learning journey:

### Starting a New Feature/Task

1. **Clear context** from previous work
2. **Explore** the relevant code (Plan Mode / read-only)
3. **Ask questions** until I understand
4. **Plan** the implementation approach
5. **Implement** in small, testable steps
6. **Verify** with tests and manual review
7. **Reflect** on what I learned

### Daily Practice

1. Open relevant files before starting
2. Close irrelevant files when switching context
3. Use explicit file references (`@file`, `#file`)
4. Clear context between different tasks
5. Name sessions for easy resumption
6. Always understand before accepting

### For Learning Projects

1. Ask for explanations, not just code
2. Request alternative approaches
3. Implement some parts manually to reinforce learning
4. Document learnings in project reflections
5. Build a personal library of patterns

---

## Quick Reference Card

### Prompting Checklist
- [ ] Provided relevant context (files, imports, examples)
- [ ] Stated constraints clearly
- [ ] Gave verification criteria (test cases, expected output)
- [ ] Referenced existing patterns when applicable
- [ ] Broke complex tasks into steps

### Before Accepting Code
- [ ] I understand what it does
- [ ] Edge cases are handled
- [ ] Error handling is appropriate
- [ ] It follows project conventions
- [ ] Tests exist or I'll add them

### Context Management
- [ ] Cleared context from previous unrelated task
- [ ] Relevant files are open
- [ ] Irrelevant files are closed
- [ ] Using fresh thread/session if context is cluttered

### After Completing a Feature
- [ ] All tests pass
- [ ] Code reviewed for quality
- [ ] Security considerations checked
- [ ] Documented what I learned
- [ ] Committed with clear message

---

## Resources

### Official Documentation
- [GitHub Copilot Best Practices](https://docs.github.com/en/copilot/using-github-copilot/best-practices-for-using-github-copilot)
- [Claude Code Best Practices](https://code.claude.com/docs/en/best-practices)
- [Claude Code Common Workflows](https://code.claude.com/docs/en/common-workflows)
- [Prompt Engineering for Copilot](https://docs.github.com/en/copilot/using-github-copilot/copilot-chat/prompt-engineering-for-copilot-chat)

### Concepts to Explore
- Agentic AI patterns
- Context window management
- Prompt engineering taxonomies
- AI-assisted TDD
- Human-in-the-loop workflows

---

*This document is a living guide. Update it as you develop your personal AI-assisted coding style.*

*Created: February 25, 2026*
