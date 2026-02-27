---
date: 2026-02-27
type: concept
topic: "Agent Specialization & Role Design"
project: "Phase 4 - AI Strategy Research Team"
status: complete
---

# Learning: Agent Specialization & Role Design

## In My Own Words

Agent specialization is the practice of designing each agent in a multi-agent system with a focused, well-defined role — complete with its own system prompt, tools, and behavioral constraints. Instead of creating general-purpose agents that try to do everything, you create specialists that excel at one thing.

Role design is the art of defining *what makes a good specialist*: their persona, capabilities, limitations, and interfaces. It's like writing a job description — but for an AI agent.

The key insight: **a well-designed specialist with a focused prompt will outperform a generalist every time**, even when using the same underlying LLM. The system prompt is the primary lever for making agents behave differently.

## Why This Matters

- Focused system prompts produce dramatically better outputs than vague ones
- Each role needs different tools, context windows, and quality criteria
- Role boundaries prevent agents from stepping on each other's toes
- Well-designed roles make debugging easier — you know exactly which agent to fix
- Your Strategy Research Team has 4+ roles that need crisp definitions

## Principles of Agent Role Design

### 1. Single Responsibility

Each agent should do ONE thing well. Don't combine research + analysis in one agent.

**Bad:** "You are a research assistant. Find information, analyze it, write a report, and review it."
**Good:** "You are a research specialist. Your only job is to find and gather relevant information."

### 2. Clear Boundaries

Define what the agent DOES and DOES NOT do. Explicit exclusions prevent scope creep.

```
You are the Analyst agent.

YOU DO:
- Identify patterns and trends in research data
- Perform SWOT analysis
- Draw strategic implications
- Highlight risks and opportunities

YOU DO NOT:
- Conduct additional research (ask the Researcher)
- Write full reports (pass to the Writer)
- Make final quality judgments (that's the Critic's job)
```

### 3. Appropriate Tools

Each role gets only the tools it needs. Don't give every agent every tool.

| Agent | Tools | Why |
|-------|-------|-----|
| Researcher | web_search, document_reader, summarizer | Needs to find and read information |
| Analyst | calculator, chart_generator, template_library | Needs to crunch data and structure analysis |
| Writer | formatting_tools, style_checker | Needs to produce polished text |
| Critic | quality_rubric, fact_checker | Needs to evaluate against standards |
| Orchestrator | task_planner, agent_router | Needs to manage the workflow |

### 4. Consistent Output Format

Define the expected output structure for each agent. This makes downstream consumption reliable.

```python
# Researcher output format
{
    "findings": [
        {
            "source": "url or name",
            "key_points": ["point1", "point2"],
            "relevance": "high/medium/low",
            "summary": "2-3 sentence summary"
        }
    ],
    "gaps": ["things I couldn't find"],
    "suggested_follow_ups": ["additional research needed"]
}
```

### 5. Model Selection Per Role

Not every agent needs GPT-4o. Match the model to the role's complexity.

| Agent | Recommended Model | Rationale |
|-------|-------------------|-----------|
| Orchestrator | GPT-4o / Claude Sonnet | Needs strong reasoning for planning |
| Researcher | GPT-4o-mini / Claude Haiku | Mostly following instructions, tool use |
| Analyst | GPT-4o / Claude Sonnet | Needs analytical reasoning |
| Writer | GPT-4o / Claude Sonnet | Needs writing quality |
| Critic | GPT-4o / Claude Sonnet | Needs judgment and nuance |

This strategy can significantly reduce costs while maintaining quality.

## Designing the Strategy Research Team Roles

### Role 1: Orchestrator (Project Manager)

**Persona:** Senior project manager who breaks down complex research requests into actionable plans.

```python
ORCHESTRATOR_PROMPT = """
You are the Orchestrator of an AI Strategy Research Team. Your role is 
to manage the research workflow.

RESPONSIBILITIES:
1. Receive a research topic from the user
2. Break it down into 2-4 specific research subtasks
3. Assign subtasks to Research agents
4. Monitor progress and handle failures
5. Ensure the final output meets quality standards

RULES:
- Create focused, specific research subtasks (not vague)
- Each subtask should be answerable with available tools
- Include clear success criteria for each subtask
- Track total token usage and cost

OUTPUT: A structured research plan in JSON format.
"""
```

### Role 2: Researcher (Information Gatherer)

**Persona:** Skilled research analyst who systematically gathers and organizes information.

```python
RESEARCHER_PROMPT = """
You are a Research Specialist on the AI Strategy Research Team. Your job 
is to find, gather, and organize information on specific topics.

RESPONSIBILITIES:
1. Search for relevant, authoritative information
2. Distinguish facts from opinions
3. Identify primary sources over secondary
4. Organize findings in a structured format
5. Flag information gaps and uncertainties

RULES:
- Always cite your sources
- Indicate confidence level (high/medium/low) for each finding
- Don't analyze or draw conclusions — just gather facts
- If you can't find information, say so clearly
- Prefer recent sources (last 2 years) over older ones

OUTPUT FORMAT:
- List of findings with sources
- Confidence assessment
- Information gaps identified
- Suggested follow-up research
"""
```

### Role 3: Analyst (Pattern Finder)

**Persona:** Strategic analyst who identifies patterns, implications, and actionable insights.

```python
ANALYST_PROMPT = """
You are a Strategic Analyst on the AI Strategy Research Team. Your job 
is to analyze research findings and extract strategic insights.

RESPONSIBILITIES:
1. Identify patterns and trends across research findings
2. Perform structured analysis (SWOT, competitive, trend analysis)
3. Draw strategic implications
4. Identify risks, opportunities, and trade-offs
5. Prioritize insights by business impact

RULES:
- Base every conclusion on evidence from the research
- Clearly distinguish facts, inferences, and opinions
- Consider multiple perspectives (optimistic, pessimistic, realistic)
- Don't add new research — work with what the Researcher provided
- If research data is insufficient, flag it

OUTPUT FORMAT:
- Key patterns identified
- Strategic implications
- SWOT analysis (if applicable)
- Risk assessment
- Recommended actions (prioritized)
"""
```

### Role 4: Writer (Report Producer)

**Persona:** Professional business writer who produces clear, well-structured strategy reports.

```python
WRITER_PROMPT = """
You are a Strategy Writer on the AI Strategy Research Team. Your job 
is to transform analysis into polished, professional strategy documents.

RESPONSIBILITIES:
1. Structure content for the target audience (executives, technical leads)
2. Write clearly and concisely — no jargon without explanation
3. Include executive summary, key findings, analysis, recommendations
4. Use data visualizations where appropriate (describe charts/tables)
5. Ensure logical flow from findings to conclusions to recommendations

RULES:
- Write for busy executives — lead with the insight, then support it
- Every claim must trace back to the research/analysis
- Use clear headers and bullet points for scannability
- Include a "So What?" section — why should the reader care?
- Keep the report under the specified length

OUTPUT FORMAT:
- Executive Summary (1 paragraph)
- Key Findings (bulleted)
- Detailed Analysis (structured sections)
- Recommendations (prioritized, actionable)
- Appendix (sources, methodology)
"""
```

### Role 5: Critic (Quality Gate)

**Persona:** Senior editor and quality reviewer with high standards.

```python
CRITIC_PROMPT = """
You are the Quality Critic on the AI Strategy Research Team. Your job 
is to evaluate outputs and provide specific, actionable feedback.

RESPONSIBILITIES:
1. Evaluate report quality against defined criteria
2. Check for factual accuracy and logical consistency
3. Identify gaps, weak arguments, or unsupported claims
4. Provide specific, actionable feedback (not vague "make it better")
5. Decide: APPROVE (ready to publish) or REVISE (needs work)

EVALUATION CRITERIA:
- Accuracy: Are claims supported by evidence?
- Completeness: Are all key aspects covered?
- Clarity: Is the writing clear and well-structured?
- Actionability: Are recommendations specific and implementable?
- Quality: Does it meet professional standards?

RULES:
- Be specific: "Paragraph 3 claims X but no source supports this" (good)
- Not vague: "The analysis needs work" (bad)
- Limit feedback to 3-5 most important issues per review
- Don't rewrite — provide feedback for the Writer to act on
- After 3 revision cycles, approve with noted caveats

OUTPUT FORMAT:
- Overall assessment: APPROVE or REVISE
- Score: 1-10 for each criterion
- Top issues (max 5), each with:
  - What's wrong
  - Where it is
  - How to fix it
"""
```

## Agent Description Best Practices

How you describe agents matters a lot for model-based routing (AutoGen SelectorGroupChat, Google ADK delegation):

| ❌ Bad Description | ✅ Good Description |
|---|---|
| "Research agent" | "Expert at finding and gathering information from multiple sources. Call when you need facts, data, or evidence about a topic." |
| "Analyst" | "Strategic analyst who identifies patterns, trends, and implications from research data. Call after research is complete." |
| "Writer" | "Professional business writer who transforms analysis into polished strategy reports. Call when analysis is complete and you need a written deliverable." |

### Key Elements of Good Agent Descriptions

1. **What they do** — their core capability
2. **When to use them** — trigger conditions
3. **What they produce** — expected output type
4. **What they need** — input requirements

## Managing Agent Context

### Context Window Strategy

Each agent should receive only the context it needs:

```
Orchestrator: user request + agent descriptions + task history
Researcher:   assigned subtask + search tools
Analyst:      research results (not the full conversation)
Writer:       analysis output + report template + style guide
Critic:       draft + evaluation rubric + research data (for fact-checking)
```

### Prompt Engineering for Roles

The system prompt is the primary tool for specialization. Key techniques:

1. **Role framing:** "You are a [role] with expertise in [domain]"
2. **Behavioral rules:** Explicit DO and DON'T lists
3. **Output format:** Structured output requirements
4. **Examples:** Few-shot examples of good output
5. **Guardrails:** What to do when unsure or when tasks are out of scope

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Role overlap | Two agents have similar descriptions | Make responsibilities mutually exclusive |
| Role too broad | Agent tries to do too much | Split into focused sub-roles |
| Role too narrow | Agent has nothing meaningful to do | Merge with a related role |
| Missing tools | Agent can't perform its duties | Audit tool access per role |
| Identity confusion | Agent doesn't stay in character | Reinforce role in system prompt + message prefix |
| Output mismatch | Agent output doesn't match next agent's expected input | Define explicit interfaces between agents |

## Application to My Project

### Agent Roster (Minimum Viable)

| # | Agent | Model | Tools | Input | Output |
|---|-------|-------|-------|-------|--------|
| 1 | Orchestrator | GPT-4o | task_planner | User request | Research plan (JSON) |
| 2 | Researcher | GPT-4o-mini | web_search, reader | Subtask description | Findings list |
| 3 | Analyst | GPT-4o | none (reasoning only) | Research results | Strategic analysis |
| 4 | Writer | GPT-4o | formatting | Analysis output | Draft report |
| 5 | Critic | GPT-4o | rubric_evaluator | Draft + criteria | APPROVE/REVISE + feedback |

### Design Decisions

- Start with 5 agents (Orchestrator + 4 specialists) — the minimum in the README
- Use different model tiers for cost optimization
- Each agent gets a structured output format for reliable handoffs
- The Critic has the most constrained prompt to ensure consistent evaluation

## Resources for Deeper Learning

- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — Practical role design guidance
- [AutoGen Agent Configuration](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html) — Configuring agent roles in AutoGen
- [Google ADK LlmAgent](https://google.github.io/adk-docs/agents/llm-agents/) — Agent instructions and descriptions in ADK
- [LangGraph Agent Tutorial](https://docs.langchain.com/oss/python/langgraph/tutorials/) — Building agents in LangGraph

## Questions Remaining

- [ ] Should each agent have a "fallback" behavior when it receives unexpected input?
- [ ] How much of the agent's persona affects output quality vs. just the instructions?
- [ ] Is there a benefit to giving agents names and personalities beyond just functional roles?
