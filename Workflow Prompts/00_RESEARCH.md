# RESEARCH Phase Prompt

> **Purpose:** Build foundational understanding through comprehensive research before implementation.  
> **Dual Goals:** (1) Enable deep learning of concepts and technologies. (2) Gather sufficient context for project development.  
> **Mode:** Documentation and synthesis — gather, organize, and internalize knowledge.  
> **Time Allocation:** Variable — use at project start and when entering unfamiliar territory.

---

## Why Research First?

Research serves two critical purposes in your learning journey:

### 1. Enable Learning
- Understand **concepts** before applying them
- Learn **why** certain approaches work, not just **how**
- Build mental models that transfer across projects
- Identify knowledge gaps before they block you

### 2. Build Project Foundation
- Gather context for informed architectural decisions
- Identify technologies and libraries to use
- Understand best practices and common pitfalls
- Create reference material for implementation

---

## When to Use This Prompt

| Scenario | Research Focus |
|----------|----------------|
| **Starting a new project phase** | Learning objectives, core concepts, technology landscape |
| **Learning a framework/library** | Architecture, patterns, APIs, best practices |
| **Making technology choices** | Compare options, trade-offs, enterprise fit |
| **Understanding a concept** | Theory, applications, implementation patterns |
| **Before architectural decisions** | Patterns, approaches, what others have done |

This differs from **Explore** in scope:
- **Research** = Broad understanding (concepts, technologies, patterns, landscape)
- **Explore** = Task-specific investigation for a known feature in existing code

---

## Research Types

### Type 1: Concept & Learning Research
*"I need to understand a concept deeply before applying it"*

```
# RESEARCH: [Concept/Topic]

## Learning Objective

I need to understand: [Clear description of what you want to learn]

## Context

- Project Phase: [Phase from learning path]
- Why I need this: [How this knowledge supports my project]
- Current understanding: [What I already know, what's fuzzy]

## Research Questions

1. **Fundamentals**
   - What is [concept] and why does it exist?
   - What problem does it solve?
   - What are the core principles?

2. **How It Works**
   - What are the key components/steps?
   - How do the pieces fit together?
   - What's the typical workflow/lifecycle?

3. **Variations & Approaches**
   - What different approaches exist?
   - What are the trade-offs between them?
   - When should I use each approach?

4. **Practical Application**
   - How is this used in production systems?
   - What are common pitfalls to avoid?
   - What are best practices?

5. **Connection to My Project**
   - How does this apply to what I'm building?
   - What decisions do I need to make?
   - What should I implement vs. use off-the-shelf?

## Output Format

Create a learning document with:
- **Summary**: Core concept in my own words
- **Key Principles**: The fundamentals I must understand
- **How It Works**: Step-by-step explanation
- **Approaches Compared**: Options and trade-offs
- **Best Practices**: What experts recommend
- **Application to My Project**: Specific guidance
- **Resources**: Links for deeper learning
```

### Type 2: Technology Research
*"I need to evaluate and learn a technology for my project"*

```
# RESEARCH: [Technology/Framework/Library]

## Research Objective

I need to evaluate and understand: [Technology name]
For use in: [Project/Phase]
To accomplish: [What capability I need]

## Research Questions

1. **What Is It?**
   - What does this technology do?
   - What problem does it solve?
   - Who maintains it and how mature is it?

2. **Core Concepts**
   - What are the key abstractions?
   - What terminology do I need to know?
   - What's the mental model for using it?

3. **Architecture & Patterns**
   - How is it structured?
   - What patterns does it use/encourage?
   - How does data flow through it?

4. **Getting Started**
   - What's the minimal setup?
   - What does a "hello world" look like?
   - What are the essential APIs?

5. **Advanced Features**
   - What capabilities exist beyond basics?
   - What extensibility points exist?
   - What are the performance characteristics?

6. **Ecosystem & Integration**
   - What other tools does it work with?
   - What's the community like?
   - What resources exist for learning?

7. **Trade-offs & Alternatives**
   - What are the pros and cons?
   - What alternatives exist?
   - When should I NOT use this?

8. **Enterprise Considerations**
   - How does it handle scale?
   - What about security?
   - What's the cost model?

## Output Format

Create a technology brief with:
- **Overview**: What it is and why it matters
- **Core Concepts**: Key abstractions explained
- **Quick Start**: Minimal code to get running
- **Key APIs**: Essential functions/classes
- **Patterns**: How to use it effectively
- **Pitfalls**: Common mistakes to avoid
- **Comparison**: vs alternatives (if relevant)
- **Decision**: Should I use this? Why/why not?
```

### Type 3: Codebase/Implementation Research
*"I need to understand existing code or implementation patterns"*

```
# RESEARCH CODEBASE: [Topic/Area]

## Research Question

I need to understand: [What you want to learn about the codebase]

## Context

- Project: [Project name]
- Codebase: [Repository or project being researched]
- Goal: [What implementation work this supports]

## Research Scope

1. **Architecture Overview**
   - How is the system organized?
   - What are the main components?
   - How do they interact?

2. **Key Patterns**
   - What patterns are used consistently?
   - What conventions should I follow?
   - How is configuration handled?

3. **Data Flow**
   - How does data move through the system?
   - What are the key interfaces?
   - Where are the integration points?

4. **File Organization**
   - Directory structure and purpose
   - Important files to know
   - Test organization

5. **Dependencies**
   - External libraries and their roles
   - Version constraints
   - Integration points

## Guidelines

- **Document what IS, not what SHOULD BE**
- Include specific file paths and line numbers
- Note patterns for me to follow
- Highlight potential challenges

## Output Format

- Summary of findings
- Architecture diagram if helpful
- Code references with file:line
- Patterns to follow
- Open questions
```

---

## Research Document Templates

### Template 1: Concept Learning Document

Save in your project folder as `research/YYYY-MM-DD-concept-name.md`:

```markdown
---
date: YYYY-MM-DD
type: concept
topic: "[Concept Name]"
project: "[Phase/Project Name]"
status: complete
---

# Learning: [Concept Name]

## In My Own Words

[Explain the concept as if teaching someone else — this forces understanding]

## Why This Matters

[Why is this concept important? What problems does it solve?]

## Core Principles

1. **[Principle 1]**: [Explanation]
2. **[Principle 2]**: [Explanation]
3. **[Principle 3]**: [Explanation]

## How It Works

### The Big Picture
[High-level explanation with diagram if helpful]

### Step by Step
1. [Step 1]: [What happens and why]
2. [Step 2]: [What happens and why]
3. [Step 3]: [What happens and why]

## Approaches & Trade-offs

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| [Approach 1] | [Scenarios] | [Benefits] | [Drawbacks] |
| [Approach 2] | [Scenarios] | [Benefits] | [Drawbacks] |

## Best Practices

- ✅ [Do this because...]
- ✅ [Do this because...]
- ❌ [Avoid this because...]
- ❌ [Avoid this because...]

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| [Mistake 1] | [Cause] | [Prevention] |
| [Mistake 2] | [Cause] | [Prevention] |

## Application to My Project

### How I'll Use This
[Specific application in your current project]

### Decisions to Make
- [ ] [Decision 1]
- [ ] [Decision 2]

### Implementation Notes
[Any specific guidance for your implementation]

## Resources for Deeper Learning

- [Resource 1](link) — [Why it's useful]
- [Resource 2](link) — [Why it's useful]

## Questions Remaining

- [ ] [Open question for further research]
- [ ] [Concept to revisit later]
```

### Template 2: Technology Brief

Save as `research/YYYY-MM-DD-technology-name.md`:

```markdown
---
date: YYYY-MM-DD
type: technology
topic: "[Technology Name]"
project: "[Phase/Project Name]"
status: complete
decision: [use / evaluate further / skip]
---

# Technology Brief: [Technology Name]

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | [One-line description] |
| **For** | [What problems it solves] |
| **Maturity** | [Stable/Growing/Experimental] |
| **License** | [License type] |
| **Decision** | [Use / Evaluate Further / Skip] |

## Core Concepts

### [Concept 1]
[Explanation with code example if helpful]

### [Concept 2]
[Explanation with code example if helpful]

## Quick Start

```python
# Minimal working example
[Code that gets you from zero to something working]
```

## Key APIs

| API/Function | Purpose | Example |
|--------------|---------|---------|
| `function_1()` | [What it does] | [Brief usage] |
| `function_2()` | [What it does] | [Brief usage] |

## Patterns to Follow

### Pattern 1: [Name]
```python
# How to do [common task]
[Code example]
```

### Pattern 2: [Name]
```python
# How to do [another common task]
[Code example]
```

## Integration Points

- **Works with**: [Other technologies it integrates well with]
- **Configuration**: [How to configure for your use case]
- **Dependencies**: [What it requires]

## Trade-offs

### Pros
- ✅ [Advantage 1]
- ✅ [Advantage 2]

### Cons
- ❌ [Limitation 1]
- ❌ [Limitation 2]

## Comparison with Alternatives

| Feature | [This Tech] | [Alternative 1] | [Alternative 2] |
|---------|-------------|-----------------|-----------------|
| [Feature 1] | [Rating/Notes] | [Rating/Notes] | [Rating/Notes] |
| [Feature 2] | [Rating/Notes] | [Rating/Notes] | [Rating/Notes] |

## Enterprise Considerations

- **Scale**: [How it handles scale]
- **Security**: [Security considerations]
- **Cost**: [Pricing model/considerations]
- **Support**: [Community/commercial support]

## Decision

**Recommendation**: [Use / Don't Use / Evaluate Further]

**Reasoning**: [Why this decision makes sense for your project]

**If using, next steps**:
1. [Action 1]
2. [Action 2]
```

### Template 3: Implementation Research Document

Save as `research/YYYY-MM-DD-implementation-topic.md`:

```markdown
---
date: YYYY-MM-DD
type: implementation
topic: "[Research Topic]"
project: "[Phase/Project Name]"
status: complete
---

# Research: [Topic]

## Research Question

[Original question driving this research]

## Summary

[3-5 sentences answering the research question]

## Detailed Findings

### [Component/Area 1]

**Purpose:** [What this does]

**Key Files:**
- `path/to/file.py` — [Description]
- `path/to/file.py:45-60` — [Specific section]

**How It Works:**
[Explanation]

**Patterns:**
- [Pattern 1]: [How it's used]
- [Pattern 2]: [How it's used]

### [Component/Area 2]

[Same structure]

## Architecture

```
[ASCII diagram or description]
```

## Patterns to Follow

| Pattern | Where Used | Example |
|---------|------------|---------|
| [Pattern 1] | [Location] | [Brief] |

## Code References

- `file.py:123` — [What's there]
- `another.py:45` — [What's there]

## Open Questions

- [ ] [Question needing follow-up]
```

---

## Research by Project Phase

### Phase 1: Prompt Engineering Laboratory

**Concepts to Research:**
```
Research topics:
- What is prompt engineering? Why systematic vs ad-hoc?
- Prompt patterns: few-shot, chain-of-thought, self-consistency
- Output parsing and structured generation
- LLM evaluation: what makes output "good"?
- Prompt versioning and testing methodologies
```

**Technologies to Research:**
```
Research topics:
- OpenAI API / Azure OpenAI: core concepts, authentication, pricing
- Pydantic for structured outputs
- LLM evaluation frameworks (ragas, deepeval, or custom)
- Prompt templating approaches (Jinja2, LangChain prompts)
```

### Phase 2: Enterprise Document Intelligence

**Concepts to Research:**
```
Research topics:
- RAG architecture: why retrieval + generation?
- Chunking strategies: fixed vs semantic vs structural
- Embedding models: how they work, what to consider
- Hybrid search: dense vs sparse, when to combine
- Hallucination and grounding in RAG systems
```

**Technologies to Research:**
```
Research topics:
- Vector databases: Qdrant vs Weaviate vs Azure AI Search
- Document loaders: unstructured, pypdf, docx libraries
- Embedding models: OpenAI vs Cohere vs open-source
- LangChain RAG abstractions
```

### Phase 3: Autonomous Customer Operations Agent

**Concepts to Research:**
```
Research topics:
- Agent architectures: ReAct, Plan-and-Execute, Reflection
- State machines for conversational agents
- Tool use and function calling in LLMs
- Memory systems: buffer, summary, vector-backed
- Guardrails and safety in agentic systems
```

**Technologies to Research:**
```
Research topics:
- LangGraph: state, nodes, edges, checkpointing
- Tool definition patterns
- Human-in-the-loop implementations
- Conversation memory implementations
```

### Phase 4: AI Strategy Research Team

**Concepts to Research:**
```
Research topics:
- Multi-agent patterns: hierarchical, democratic, swarm
- Agent communication: shared state vs message passing
- Task decomposition and delegation
- Coordination and conflict resolution
- Emergent behavior in multi-agent systems
```

**Technologies to Research:**
```
Research topics:
- LangGraph for multi-agent orchestration
- Microsoft Agent Framework: architecture and patterns
- Google ADK: what it offers
- AutoGen/CrewAI for inspiration
```

### Phase 5: Multi-Modal Enterprise Assistant

**Concepts to Research:**
```
Research topics:
- Vision-language models: capabilities and limitations
- Multi-modal reasoning: how models handle image+text
- Fine-tuning concepts: full, LoRA, QLoRA
- When to fine-tune vs prompt engineer
- Audio processing pipelines
```

**Technologies to Research:**
```
Research topics:
- GPT-4 Vision / Azure AI Vision APIs
- Whisper / Azure Speech Services
- Fine-tuning with Azure ML or HuggingFace
- Azure AI Foundry for model management
```

### Phase 6: Enterprise AI Platform (Capstone)

**Concepts to Research:**
```
Research topics:
- API gateway patterns for AI services
- Observability for LLM systems
- Security patterns for LLM applications (OWASP LLM Top 10)
- Cost allocation and tracking
- Deployment strategies (blue-green, canary)
```

**Technologies to Research:**
```
Research topics:
- FastAPI patterns for AI services
- Docker containerization essentials
- Kubernetes basics for AI deployment
- Observability: Prometheus, Grafana, or cloud equivalents
- Azure deployment options
```

---

## Research Strategies

### Strategy 1: Learning-First Research

When your primary goal is understanding:

```
# LEARNING RESEARCH: [Topic]

## What I Want to Learn
[Specific concept or skill]

## Why I Need This
[How it connects to my project/goals]

## My Current Understanding
[What I already know - forces you to identify gaps]

## Research Approach

1. **Start with "Why"**
   - What problem does this solve?
   - Why was this approach developed?
   - What alternatives exist?

2. **Build Mental Models**
   - What are the core abstractions?
   - How do the pieces fit together?
   - Can I draw a diagram of this?

3. **See It In Action**
   - Find working examples
   - Trace through execution
   - Identify common patterns

4. **Understand Trade-offs**
   - What are the strengths?
   - What are the limitations?
   - When should I NOT use this?

5. **Connect to My Work**
   - How will I apply this?
   - What decisions does this inform?

## Output: Teach It Back
Write an explanation as if teaching someone else.
If you can't explain it simply, you don't understand it well enough.
```

### Strategy 2: Technology Evaluation Research

When choosing between options:

```
# TECHNOLOGY EVALUATION: [Category]

## What I Need
[Capability or problem to solve]

## Candidates
1. [Option 1]
2. [Option 2]
3. [Option 3]

## Evaluation Criteria
- Learning curve
- Feature completeness
- Enterprise readiness
- Community/support
- Integration with my stack
- Cost considerations

## For Each Candidate, Research:
1. Core value proposition
2. Basic usage patterns
3. Limitations and gotchas
4. Real-world adoption
5. Learning resources available

## Output: Decision Matrix
Create a comparison table and make a recommendation.
```

### Strategy 3: Implementation Context Research

When preparing to build:

```
# IMPLEMENTATION RESEARCH: [Feature/Component]

## What I'm Building
[Clear description]

## What I Need to Know Before Starting

1. **Patterns Used by Others**
   - How do tutorials/docs approach this?
   - What patterns appear repeatedly?
   - What's considered best practice?

2. **Technology Specifics**
   - Key APIs I'll use
   - Configuration requirements
   - Common pitfalls

3. **Integration Points**
   - How will this connect to existing code?
   - What interfaces need to match?
   - What data formats are expected?

4. **Testing Approaches**
   - How do others test this?
   - What mocking/fixtures are common?
   - What edge cases matter?

## Output: Implementation Readiness
- Key decisions made
- Patterns to follow
- Risks identified
- Ready to write Explore prompt for specific task
```

---

## Research Checklist

Before moving to Explore/Plan phases, verify:

### For Learning Goals ✅
- [ ] I can explain the concept in my own words
- [ ] I understand WHY, not just HOW
- [ ] I know the trade-offs between approaches
- [ ] I've identified best practices to follow
- [ ] I know common pitfalls to avoid
- [ ] I have resources for deeper learning if needed

### For Project Foundation ✅
- [ ] I know which technologies to use
- [ ] I understand the key patterns
- [ ] I have enough context to make architectural decisions
- [ ] I've identified integration points
- [ ] I know what "good" looks like
- [ ] Open questions are logged for follow-up

---

## Parallel Research with Sub-agents

For complex research, delegate to focused sub-agents:

```
I need to research [broad topic]. Please spawn parallel research tasks:

## Sub-task 1: Concept Deep-Dive
Research the theoretical foundations of [concept].
Return: Core principles, why it works, mental models

## Sub-task 2: Technology Survey  
Find and compare tools/libraries for [capability].
Return: Options, trade-offs, recommendations

## Sub-task 3: Pattern Finding
Find implementation examples and best practices.
Return: Code patterns, common approaches, pitfalls

## Sub-task 4: Enterprise Considerations
Research production/enterprise aspects.
Return: Scale, security, cost, observability concerns

After all sub-tasks complete, synthesize into a unified research document.
```

---

## Research Workflow

```
┌────────────────────────────────────────────────────────────────────┐
│  1. IDENTIFY GAPS                                                  │
│     - What do I need to know?                                      │
│     - What's blocking my understanding?                            │
│     - What decisions do I need to make?                            │
├────────────────────────────────────────────────────────────────────┤
│  2. CHOOSE RESEARCH TYPE                                           │
│     - Concept learning?                                            │
│     - Technology evaluation?                                       │
│     - Implementation context?                                      │
├────────────────────────────────────────────────────────────────────┤
│  3. GATHER INFORMATION                                             │
│     - Official documentation                                       │
│     - Tutorials and examples                                       │
│     - Source code when relevant                                    │
│     - Community discussions                                        │
├────────────────────────────────────────────────────────────────────┤
│  4. SYNTHESIZE                                                     │
│     - Write in your own words                                      │
│     - Create diagrams if helpful                                   │
│     - Connect to your project                                      │
├────────────────────────────────────────────────────────────────────┤
│  5. DOCUMENT                                                       │
│     - Save research document                                       │
│     - Note open questions                                          │
│     - Identify next steps                                          │
└────────────────────────────────────────────────────────────────────┘
```

---

## Tips for Effective Research

### Optimize for Learning

**Write to Understand**
- Summarize in your own words — copying doesn't build understanding
- If you can't explain it simply, research more
- Teaching (even to yourself) reveals gaps

**Build Mental Models**
- Draw diagrams, even rough ones
- Find analogies to things you already know
- Ask "why is it this way?" not just "how does it work?"

**Connect to Practice**
- Always ask: "How will I use this?"
- Find working examples you can modify
- Identify decisions this research informs

### Optimize for Context Gathering

**Be Thorough but Time-boxed**
- Set a limit (2-4 hours for initial research)
- Note open questions rather than rabbit-holing
- You can always do follow-up research

**Document As You Go**
- Create the research document incrementally
- Include links and references immediately
- Don't rely on memory

**Focus on What Matters**
- Prioritize decisions you need to make now
- Depth where it matters, breadth otherwise
- Enterprise considerations if production-relevant

### Ask Quality Questions

Instead of: "How does RAG work?"
Ask: "What are the different chunking strategies for RAG, what are the trade-offs between them, and how do I choose for my document types?"

Instead of: "What is LangGraph?"
Ask: "How does LangGraph model agent state, what patterns does it enable that raw LangChain doesn't, and when should I choose it?"

---

## Research vs Explore vs Plan

| Phase | Purpose | Scope | Output |
|-------|---------|-------|--------|
| **Research** | Build understanding | Broad — concepts, tech, patterns | Research documents |
| **Explore** | Investigate for task | Narrow — specific feature | Mental model, notes |
| **Plan** | Decide how to build | Specific — one implementation | Action plan |

**The flow:**
1. **Research** concepts and technologies broadly (once per phase/area)
2. **Explore** the codebase for specific features (uses research as foundation)
3. **Plan** the implementation (uses explore findings)
4. **Implement** the plan
5. **Verify** the implementation

---

*This prompt is part of the Research → Explore → Plan → Implement → Verify workflow.*  
*After completing research, use [01_EXPLORE.md](01_EXPLORE.md) for task-specific investigation.*
