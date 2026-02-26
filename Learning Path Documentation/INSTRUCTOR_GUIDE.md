# Instructor Guide: How This Learning System Works

> This document explains how I (your AI instructor) will interact with you throughout this program.  
> It also contains guidance for yourself on how to approach the work, ask for help, and track progress.

---

## My Role as Your Instructor

I am not here to write your code. I am here to:

1. **Clarify** — When requirements are ambiguous, I'll help you understand the intent
2. **Unblock** — When you're stuck, I'll give you enough to move forward without solving it for you
3. **Review** — When you've built something, I'll give you honest feedback
4. **Challenge** — When your solution is superficial, I'll push you deeper
5. **Connect** — I'll help you see how concepts relate to enterprise reality

---

## How to Work Through Each Project

### Phase 1: Understanding (10% of project time)

Before writing any code:

1. **Read the project brief completely** — twice
2. **Identify what you don't know** — make a list of concepts/technologies to explore
3. **Sketch the architecture** — boxes and arrows, not code
4. **Define success criteria** — what does "done" look like?
5. **Ask clarifying questions** — I'm here to help

**What to bring to me:**
- "Here's my understanding of the problem: [summary]. Am I missing anything?"
- "I'm planning to use [technology] because [reasoning]. Does this make sense?"
- "I don't understand what [concept] means in this context. Can you explain?"

### Phase 2: Research & Design (20% of project time)

1. **Explore the concepts** — read documentation, papers, blog posts
2. **Design your approach** — write it down, even if informally
3. **Identify risks** — what might be harder than it looks?
4. **Make technology choices** — with reasoning

**What to bring to me:**
- "Here's my architecture. What am I not thinking about?"
- "I'm choosing between [A] and [B]. Here are the trade-offs I see: [list]. Which would you recommend?"
- "This part seems risky because [reason]. How should I approach it?"

### Phase 3: Implementation (50% of project time)

1. **Start with the core** — get the essential flow working
2. **Build incrementally** — don't try to build everything at once
3. **Test as you go** — don't save all testing for the end
4. **Commit frequently** — small, logical commits with good messages
5. **Document decisions** — write down why you did what you did

**What to bring to me:**
- "I'm stuck on [specific problem]. Here's what I've tried: [list]. Here's what I think is happening: [hypothesis]."
- "My code is working, but I think there's a better way. Here's my approach: [explanation]. What do you think?"
- "I made a trade-off here: [description]. Is this reasonable for production?"

### Phase 4: Refinement (15% of project time)

1. **Review your own work** — would you approve this PR?
2. **Add observability** — logging, metrics, tracing
3. **Handle edge cases** — what happens when things fail?
4. **Optimize if needed** — measure first, optimize second

**What to bring to me:**
- "Here's my completed project. Can you review it?"
- "I'm not happy with [part]. Here's why: [reasoning]. How can I improve it?"

### Phase 5: Reflection (5% of project time)

1. **Write a project retrospective** — what worked, what didn't, what you learned
2. **Update your skills checklist** — be honest about your level
3. **Connect to enterprise context** — how would this be used in a real engagement?

**What to bring to me:**
- "I finished the project. Here's my reflection: [summary]. What else should I take away?"
- "In a real enterprise, I think this would need [additions]. Am I thinking about this correctly?"

---

## How I Will Give Feedback

### Code Reviews

When you show me code, I will evaluate:

1. **Correctness** — Does it work? Does it handle edge cases?
2. **Clarity** — Is it readable? Would a colleague understand it?
3. **Architecture** — Is the structure appropriate? Is it maintainable?
4. **Enterprise-readiness** — Is it production-quality? What's missing?
5. **Learning demonstration** — Does it show you understood the concepts?

I will give feedback in three categories:

- 🔴 **Must Fix** — Issues that would fail a code review
- 🟡 **Should Improve** — Issues that matter for production
- 🟢 **Consider** — Suggestions for excellence

### Architecture Reviews

When you show me designs, I will evaluate:

1. **Completeness** — Does it address all requirements?
2. **Trade-offs** — Have you considered alternatives?
3. **Scalability** — Will it handle enterprise scale?
4. **Operability** — Can it be monitored, debugged, maintained?
5. **Security** — Are there obvious vulnerabilities?

### Progress Check-ins

At the end of each project, we should discuss:

1. What you built and why you made key decisions
2. What was harder than expected and how you solved it
3. What you would do differently with more time
4. How confident you feel about the skills practiced
5. Any lingering questions or confusions

---

## How to Ask for Help Effectively

### Good questions include:

1. **Context** — What are you trying to do?
2. **Attempt** — What have you already tried?
3. **Observation** — What happened vs. what you expected?
4. **Hypothesis** — What do you think is wrong?

**Example of a good question:**
> "I'm implementing the retrieval component for Project 2. I'm using LangChain's retriever with Qdrant, but the results aren't relevant. I've tried adjusting the chunk size and the number of results, but it's still poor. The embeddings seem fine when I inspect them manually. I think the issue might be with how I'm formatting the query, but I'm not sure how to debug this."

### Less effective questions:

- "It doesn't work" (too vague)
- "What should I do next?" (too open-ended)
- "Is this right?" without showing your reasoning

---

## Progress Tracking System

Use **PROGRESS.md** to track your journey. Update it at least:

1. **Daily** — Brief notes on what you worked on
2. **Per milestone** — When you complete a significant part
3. **Per project** — Full reflection when you finish

The progress file is structured to capture:

- **Completion status** — What's done
- **Time tracking** — How long things actually took
- **Challenges** — What was hard
- **Learnings** — Key insights
- **Self-assessment** — Honest evaluation of your skills

---

## When to Move On

You should move to the next project when:

1. ✅ All deliverables are complete
2. ✅ The system works end-to-end
3. ✅ You can explain your architectural decisions
4. ✅ You've reflected on learnings
5. ✅ You feel reasonably confident (not perfect, but competent)

You should NOT move on if:

- ❌ Core features are broken
- ❌ You copied code you don't understand
- ❌ You skipped the "hard parts"
- ❌ You haven't tested edge cases
- ❌ You couldn't explain it to a colleague

---

## Enterprise Mindset Reminders

Keep these principles in mind throughout:

### Think Like a Consultant

- You're not building for yourself; you're building for a client
- Documentation matters as much as code
- "It works on my machine" isn't acceptable
- You need to explain and defend your decisions

### Think About Production

- Happy path isn't enough
- What happens at 3 AM when this fails?
- Who will maintain this code?
- What will this cost at scale?

### Think About Security

- Never trust user input
- LLMs can be manipulated
- Sensitive data needs protection
- Audit trails matter

### Think About Cost

- Every API call has a price
- Cheaper isn't always better; ROI matters
- Enterprise clients care about unit economics
- Can you estimate cost per query?

---

## Framework Learning Priority

Since you want to showcase specific frameworks, here's where each appears:

| Framework | Primary Project | Supporting Projects |
|-----------|-----------------|---------------------|
| **LangChain** | Project 2 (RAG) | Projects 1, 3, 5 |
| **LangGraph** | Projects 3, 4 | Project 6 |
| **Agent Framework** | Project 4 | Project 6 |
| **Google ADK** | Project 4 | (comparison) |
| **Azure AI Foundry** | Project 5 | Project 6 |

You'll gain deep experience with LangChain/LangGraph and working knowledge of the others.

---

## Communication Patterns

### Starting a Project
```
"I'm starting Project [X]. I've read the brief. Here's my initial understanding: [summary].
Before I dive in, I want to clarify: [questions]"
```

### Daily Check-in (optional)
```
"Today I worked on [component]. I accomplished [x]. I'm stuck on [y].
Tomorrow I plan to [z]."
```

### Asking for Help
```
"I'm working on [feature]. The problem is [description].
I've tried [attempts]. I think the issue is [hypothesis].
Can you help me [specific request]?"
```

### Requesting Review
```
"I've completed [deliverable]. Here's how it works: [explanation].
Key decisions I made: [list with reasoning].
I'm not sure about: [uncertainties].
Can you review?"
```

### Project Completion
```
"I've finished Project [X]. 
Summary: [what I built]
Challenges: [what was hard]
Learnings: [key insights]
Self-assessment: [honest evaluation]
Questions: [anything still unclear]"
```

---

## Quality Standards

### Code Quality

- Follows PEP 8 (or configured linter)
- Type hints used consistently
- Functions are focused (single responsibility)
- Error handling is thoughtful
- Logging is informative but not excessive
- Comments explain "why", not "what"
- No commented-out code
- No hardcoded secrets

### Documentation Quality

- README explains what and why
- Setup instructions actually work
- Architecture is diagrammed
- API endpoints are documented
- Environment variables are listed
- Known limitations are noted

### Testing Quality

- Happy path is tested
- Edge cases are tested
- Error conditions are tested
- Tests are readable and maintainable
- Tests can run in CI

---

## Reflection Prompts

Use these to guide your thinking after each project:

### Technical Reflection
- What was the hardest technical problem I solved?
- What would I do differently if I started over?
- What patterns or techniques will I reuse?
- Where did I cut corners, and what are the consequences?

### Learning Reflection
- What concepts moved from "heard of" to "understand"?
- What concepts moved from "understand" to "can apply"?
- What surprised me about this project?
- What do I still not fully understand?

### Enterprise Reflection
- How would this work in a real enterprise context?
- What production concerns did I address? Which did I skip?
- How would I present this to a client?
- What questions would a skeptical architect ask?

---

## Final Advice

1. **Embrace struggle** — Being stuck is where learning happens
2. **Be honest** — With yourself about what you know and don't know
3. **Build ugly first** — Working code beats beautiful plans
4. **Refactor often** — Your first approach is rarely best
5. **Document as you go** — Not at the end when you've forgotten
6. **Ask questions** — There are no stupid questions, only wasted time
7. **Connect the dots** — Each project builds on the last
8. **Think enterprise** — Always imagine this running in production

---

*This guide is your companion throughout the program. Refer back to it when you're unsure how to proceed.*

*Created: February 25, 2026*
