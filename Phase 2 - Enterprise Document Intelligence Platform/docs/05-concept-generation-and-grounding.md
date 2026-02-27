---
date: 2026-02-27
type: concept
topic: "Generation, Grounding, and Hallucination Mitigation"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
---

# Learning: Generation, Grounding, and Hallucination Mitigation

## In My Own Words

The generation stage is where the LLM takes the retrieved context and the user's question and produces an answer. But this is also where things go wrong in the most dangerous ways: the LLM might **hallucinate** (fabricate information), **ignore context** (answer from parametric memory instead), or **fail to attribute** (provide correct info but you can't verify where it came from).

Grounding means forcing the LLM to base its answer **exclusively** on the retrieved evidence. Source attribution means every claim in the answer can be traced back to a specific document. Together, these turn a RAG system from "pretty good chatbot" into "trustworthy enterprise knowledge system."

## Why This Matters

- **Trust**: Enterprise users won't adopt a system that makes things up
- **Liability**: Wrong answers based on hallucinated facts have real consequences
- **Verifiability**: Users must be able to check the sources behind any answer
- **Confidence**: Knowing when the system is uncertain prevents wrong decisions
- **Compliance**: Regulated industries require audit trails for AI-generated content

In our consulting firm scenario: a consultant relying on fabricated methodology details could damage client relationships. **Every answer must be verifiable.**

---

## Core Principles

### 1. The LLM Is Not the Knowledge Source

In a RAG system, the LLM's job is **reasoning and synthesis**, not **knowledge recall**:

```
❌ WRONG mental model:
   LLM knows things → sometimes checks documents → answers

✅ RIGHT mental model:
   Documents contain knowledge → retrieval selects relevant parts →
   LLM synthesizes an answer FROM those parts → cites sources
```

The prompt must enforce this boundary.

### 2. Hallucination Is the Default, Not the Exception

LLMs are trained to generate fluent, confident text. When they don't have enough context, they **fill in the gaps** with plausible-sounding fabrications. This isn't a bug — it's how they work.

Types of hallucination in RAG:

| Type | Description | Example |
|------|-------------|---------|
| **Intrinsic** | Contradicts the retrieved context | Context says "revenue grew 5%", LLM says "revenue grew 15%" |
| **Extrinsic** | Adds information not in context | Context describes methodology, LLM adds statistics not mentioned |
| **Fabricated citations** | Invents source references | "According to our 2024 Strategy Report..." when no such report was retrieved |
| **Parametric leakage** | Uses training data instead of context | Answers with general knowledge instead of company-specific info |

### 3. "I Don't Know" Is a Feature, Not a Bug

A system that says "I don't have enough information to answer this" is **more valuable** than one that confidently hallucinates an answer.

```python
# The system SHOULD do this:
{
    "answer": "I don't have sufficient information in the knowledge base to answer 
               this question. The closest related documents discuss [X] and [Y], 
               but they don't directly address your question about [Z].",
    "confidence": 0.2,
    "retrieved_sources": [...],
    "suggestion": "You might want to check with the Strategy team directly."
}
```

---

## How Grounded Generation Works

### The Generation Prompt

The prompt is the primary lever for controlling generation quality:

```python
GENERATION_PROMPT = """You are an AI assistant for a consulting firm's knowledge base.
Answer the user's question based ONLY on the provided context documents.

RULES:
1. ONLY use information from the provided context documents
2. For every claim, cite the source using [Source: document_name, page X]
3. If the context doesn't contain enough information, say "I don't have sufficient 
   information in the knowledge base to answer this completely"
4. If documents contradict each other, note the contradiction and cite both sources
5. Never make up facts, statistics, dates, or names not in the context
6. If you're unsure, express uncertainty explicitly

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {question}

ANSWER (with source citations):"""
```

### Key Prompt Engineering Techniques for Grounding

#### Technique 1: Explicit Grounding Instructions

Be very specific about what the LLM should and shouldn't do:

```python
# Strong grounding instructions
"""
- Base your answer EXCLUSIVELY on the provided documents
- Do NOT use your general knowledge to supplement the answer
- If information is missing from the documents, acknowledge the gap
- Cite specific documents for each claim using [Doc: name]
"""
```

#### Technique 2: Structured Output for Attribution

Force the LLM to structure its response for easy verification:

```python
from pydantic import BaseModel
from typing import List, Optional

class SourceCitation(BaseModel):
    document_name: str
    page_or_section: str
    relevant_quote: str  # Exact quote from the source

class AnswerWithSources(BaseModel):
    answer: str
    citations: List[SourceCitation]
    confidence: float  # 0.0 to 1.0
    information_gaps: Optional[List[str]]  # What couldn't be answered
    caveats: Optional[List[str]]  # Limitations of the answer
```

Using structured output (from Phase 1 learnings) ensures consistent attribution.

#### Technique 3: Chain-of-Thought for Complex Questions

For multi-document synthesis, ask the LLM to think step by step:

```python
"""
Think through this step by step:
1. First, identify which context documents are relevant to each part of the question
2. Extract the key information from each relevant document
3. Synthesize the information into a coherent answer
4. Cite each source used
5. Note any contradictions or gaps
"""
```

#### Technique 4: Self-Verification

Ask the LLM to check its own answer:

```python
"""
After writing your answer, verify:
- Is every factual claim supported by a cited document?
- Did I use any information not present in the context?
- Are there any claims I'm not confident about?

If you find unsupported claims, either remove them or flag them as uncertain.
"""
```

---

## Context Window Management

### The Problem

You have limited space in the context window, and what you put there matters:

```
┌─────────────────────────────────────────────────────┐
│                 Context Window (128K tokens)          │
│                                                       │
│  System Prompt:        ~500 tokens                    │
│  Grounding Instructions: ~300 tokens                  │
│  Retrieved Context:    ~4000-8000 tokens  ← VARIABLE │
│  User Question:        ~50-200 tokens                 │
│  Generation Space:     ~500-2000 tokens               │
│                                                       │
│  Remaining: unused (but you pay for generation tokens)│
└─────────────────────────────────────────────────────┘
```

### Strategies for Context Management

#### Strategy 1: Stuff (Simple)

Put all retrieved chunks into the prompt directly.

```python
context = "\n\n---\n\n".join([
    f"[Source: {doc.metadata['source']}, Page {doc.metadata['page']}]\n{doc.page_content}"
    for doc in retrieved_docs
])
```

**When to use**: ≤5 chunks, total context fits comfortably in window.
**Limitation**: Doesn't scale to many documents.

#### Strategy 2: Map-Reduce

Process each document separately, then combine the individual answers.

```
Chunk 1 → LLM → Summary 1 ─┐
Chunk 2 → LLM → Summary 2 ──┤→ LLM → Final Answer
Chunk 3 → LLM → Summary 3 ──┤
Chunk 4 → LLM → Summary 4 ─┘
```

```python
# Map step: extract relevant info from each chunk
summaries = []
for doc in retrieved_docs:
    summary = llm.generate(f"Extract information relevant to '{question}' from:\n{doc}")
    summaries.append(summary)

# Reduce step: combine summaries into final answer
final = llm.generate(f"Based on these extracts, answer: {question}\n\n{''.join(summaries)}")
```

**When to use**: Many documents, need comprehensive coverage.
**Limitation**: Multiple LLM calls (higher cost/latency), may lose nuance.

#### Strategy 3: Refine (Iterative)

Process documents one at a time, refining the answer with each new piece of context.

```
Chunk 1 → LLM → Answer v1
Chunk 2 + Answer v1 → LLM → Answer v2
Chunk 3 + Answer v2 → LLM → Answer v3 (final)
```

**When to use**: When documents build on each other, chronological information.
**Limitation**: Serial processing (slow), later documents can override earlier ones.

#### Strategy 4: Compression Before Generation

Use a lightweight model to compress/extract relevant info from each chunk, then stuff the compressed versions.

```python
compressed_context = []
for doc in retrieved_docs:
    relevant_parts = llm.generate(
        f"Extract ONLY the sentences relevant to '{question}' from:\n{doc.page_content}"
    )
    if relevant_parts != "NOT_RELEVANT":
        compressed_context.append(relevant_parts)

# Now the context is much smaller → stuff approach works
```

### The "Lost in the Middle" Problem

Research shows LLMs pay more attention to the **beginning and end** of the context, and less to the **middle**:

```
Attention distribution:
[HIGH] ██████████░░░░░░░░██████████ [HIGH]
       ↑ Start        Middle ↑        ↑ End

Most relevant docs should go HERE → [START] or [END]
```

**Mitigation strategies:**
- Put the most relevant chunks at the **start** of the context
- Put the second-most relevant at the **end**
- Limit total context to reduce the "middle" problem
- Use structured formatting (headers, numbered sources) to help attention

---

## Confidence Scoring

### Why Confidence Matters

Not all answers are equally reliable. Confidence scoring helps users calibrate trust:

```python
# HIGH confidence (0.9): Multiple sources agree, direct match
"Our M&A methodology consists of 4 phases: Due Diligence, Planning, 
Execution, and Review. [Source: M&A Methodology Guide v3, p.5]
[Source: Case Study - TechCorp, p.2]"

# MEDIUM confidence (0.5): Single source, indirect match
"Based on one case study, the typical integration timeline is 6-12 months,
though this may vary by deal size. [Source: Case Study - TechCorp, p.8]"

# LOW confidence (0.2): Weak evidence, extrapolation
"I found limited information about this. The closest reference suggests...
I recommend consulting the Strategy team for a definitive answer."
```

### Approaches to Confidence Scoring

#### Approach 1: Retrieval-Based Confidence

Use retrieval scores as a proxy:

```python
def retrieval_confidence(retrieval_scores: list[float]) -> float:
    """Confidence based on how well documents matched."""
    if not retrieval_scores:
        return 0.0
    
    top_score = retrieval_scores[0]
    avg_score = sum(retrieval_scores[:3]) / min(3, len(retrieval_scores))
    
    # High confidence if top results are very similar to query
    if top_score > 0.85 and avg_score > 0.75:
        return 0.9
    elif top_score > 0.7:
        return 0.6
    elif top_score > 0.5:
        return 0.3
    else:
        return 0.1
```

#### Approach 2: LLM Self-Assessment

Ask the LLM to rate its own confidence:

```python
"""
After your answer, rate your confidence:
- HIGH: Multiple sources directly support the answer
- MEDIUM: Some sources support it, but information is incomplete
- LOW: Sources are tangentially related; answer involves extrapolation
- NONE: Cannot answer from the provided context

Confidence: [HIGH/MEDIUM/LOW/NONE]
Reasoning: [Why this confidence level]
"""
```

#### Approach 3: Multi-Signal Confidence

Combine multiple signals:

```python
def composite_confidence(
    retrieval_score: float,      # How well docs matched query
    source_count: int,           # How many sources support the answer
    llm_self_assessment: str,    # LLM's own confidence rating
    answer_length: int           # Very short answers may indicate uncertainty
) -> float:
    score = 0.0
    
    # Retrieval quality (40% weight)
    score += 0.4 * min(retrieval_score / 0.9, 1.0)
    
    # Source agreement (30% weight)
    source_factor = min(source_count / 3, 1.0)  # 3+ sources = max
    score += 0.3 * source_factor
    
    # LLM self-assessment (20% weight)
    llm_scores = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3, "NONE": 0.0}
    score += 0.2 * llm_scores.get(llm_self_assessment, 0.5)
    
    # Answer substance (10% weight)
    score += 0.1 * min(answer_length / 200, 1.0)
    
    return round(score, 2)
```

---

## Handling Special Cases

### Contradictory Documents

When documents disagree, the system should surface the contradiction:

```python
CONTRADICTION_INSTRUCTIONS = """
If context documents provide conflicting information:
1. Present both perspectives with their sources
2. Note the dates of each document (newer may be more current)
3. Do NOT pick a side — present the contradiction transparently
4. Suggest the user verify which is current

Example:
"There is a discrepancy in the available documents:
- [Source: Strategy Guide 2023, p.15] states the process has 4 phases
- [Source: Updated Process Doc 2024, p.3] describes 5 phases, adding a 
  'Pre-Assessment' phase
The 2024 document is more recent and may reflect current practice.
I recommend confirming with the Strategy team."
"""
```

### Multi-Document Synthesis

When the answer requires combining information from multiple sources:

```python
SYNTHESIS_INSTRUCTIONS = """
When synthesizing across multiple documents:
1. Clearly attribute each piece of information to its source
2. Use phrases like "According to [Source]..." or "Based on [Source]..."
3. When combining, make clear what comes from where
4. Note if documents provide complementary vs overlapping information

Example:
"Our cloud migration approach combines elements from several methodology guides:
- The assessment framework comes from [Source: Cloud Playbook, p.8]
- The risk evaluation matrix is detailed in [Source: Risk Management Guide, p.22]
- The timeline estimates are based on [Source: Case Study - BankCo Cloud, p.5]"
"""
```

### When Evidence Is Insufficient

```python
INSUFFICIENT_EVIDENCE_INSTRUCTIONS = """
When the context doesn't contain enough information to answer:
1. Explicitly state what information is missing
2. Share what related information WAS found
3. Suggest where the user might find the answer
4. Do NOT fill gaps with general knowledge

Example:
"I couldn't find specific information about our pricing model for cloud services 
in the knowledge base. However, I found:
- [Source: Cloud Playbook, p.3] describes our service offerings (but not pricing)
- [Source: Sales Guide, p.12] mentions 'contact the pricing team for current rates'

I suggest reaching out to the Cloud Services pricing team for current information."
"""
```

---

## Generation Patterns for Different Query Types

| Query Type | Generation Strategy | Notes |
|-----------|-------------------|-------|
| **Factual** ("What is X?") | Direct answer + citation | Single source usually sufficient |
| **Procedural** ("How do we do X?") | Step-by-step with citations per step | Often from methodology docs |
| **Comparative** ("Compare X and Y") | Side-by-side with separate citations | Need sources for both X and Y |
| **Analytical** ("Why did X happen?") | Multi-source synthesis with reasoning | Chain-of-thought helpful |
| **Aggregative** ("List all X") | Comprehensive list with sources | Map-reduce may be needed |
| **Yes/No** ("Do we have X?") | Clear yes/no + evidence + caveats | Must cite supporting evidence |

---

## Best Practices

- ✅ **Enforce grounding in the system prompt** — clear instructions to use ONLY retrieved context
- ✅ **Require inline citations** — `[Source: doc_name, p.X]` for every factual claim
- ✅ **Implement "I don't know"** — low-confidence threshold below which system refuses to answer
- ✅ **Surface contradictions** — don't hide conflicts, present them transparently
- ✅ **Use structured output** — Pydantic models for answer + citations + confidence
- ✅ **Order context by relevance** — most relevant at start and end (lost-in-middle mitigation)
- ✅ **Track generation faithfulness** — measure how often the LLM goes beyond the context
- ❌ **Don't let the LLM use general knowledge** — explicitly prohibit it in the prompt
- ❌ **Don't show confidence as exact numbers to users** — use HIGH/MEDIUM/LOW labels
- ❌ **Don't skip source verification** — validate that cited sources actually exist
- ❌ **Don't use map-reduce when stuff works** — simpler is better when context fits

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| LLM ignores grounding instructions | Prompt too permissive, no enforcement | Stronger system prompts, structured output |
| Fabricated source citations | LLM predicts plausible source names | Post-process: verify all citations exist in retrieved docs |
| Over-confident answers | LLM defaults to confident tone | Require explicit confidence assessment |
| Generic answers | LLM falls back to training knowledge | Include "Use ONLY the provided context" in prompt |
| Missing attribution | Answer is correct but no sources cited | Enforce citation format in structured output |

---

## Application to Our Project

### Generation Pipeline Design

```
Retrieved Chunks (5-8 with metadata)
    │
    ▼
┌──────────────────────┐
│  Context Formatting   │  Format chunks with source labels
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│  Prompt Construction  │  System prompt + grounding rules + 
│                       │  context + question
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│  LLM Generation       │  Structured output with citations
│  (GPT-4 / GPT-4o)    │
└──────────┬───────────┘
           │
    ▼
┌──────────────────────┐
│  Post-Processing      │  Verify citations, calculate confidence,
│                       │  format response
└──────────┬───────────┘
           │
    ▼
Response {answer, citations[], confidence, gaps[]}
```

### Decisions to Make

- [ ] Which LLM for generation (GPT-4o for quality vs GPT-4o-mini for cost)
- [ ] Confidence threshold for "I don't know" responses
- [ ] How to format citations in the user-facing response
- [ ] Whether to implement map-reduce for multi-document synthesis or keep stuff-only initially
- [ ] How to handle the lost-in-middle problem (context ordering strategy)

### Implementation Notes

- Start with the "stuff" approach (simplest), switch to map-reduce only if needed
- Use Pydantic for structured generation output from day one
- Build a citation verification post-processor (check that cited docs are real)
- Set initial "I don't know" threshold at retrieval score < 0.5
- Log all generated answers with their context for evaluation

---

## Resources for Deeper Learning

- [OpenAI best practices for RAG](https://platform.openai.com/docs/guides/prompt-engineering) — Prompt engineering for grounded generation
- [ARES: Automated RAG Evaluation](https://arxiv.org/abs/2311.09476) — Academic work on evaluating RAG faithfulness
- [Vectara Hallucination Leaderboard](https://github.com/vectara/hallucination-leaderboard) — Which LLMs hallucinate least
- [Lost in the Middle paper](https://arxiv.org/abs/2307.03172) — Research on context position effects
- [Self-RAG paper](https://arxiv.org/abs/2310.11511) — Self-reflective RAG approach

---

## Questions Remaining

- [ ] What's the faithfulness rate of GPT-4o vs GPT-4o-mini for grounded generation?
- [ ] How to handle queries that need real-time information (not in the document store)?
- [ ] Should we implement Self-RAG (model decides when to retrieve) or keep it simpler?
- [ ] How to handle multi-turn conversations where context accumulates across turns?
