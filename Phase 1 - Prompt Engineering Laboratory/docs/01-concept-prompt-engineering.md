---
date: 2025-02-27
type: concept
topic: "Prompt Engineering Fundamentals"
project: "Phase 1 - Prompt Engineering Laboratory"
status: complete
---

# Learning: Prompt Engineering Fundamentals

## In My Own Words

Prompt engineering is the discipline of designing, structuring, and optimizing inputs (prompts) to Large Language Models to get reliable, high-quality outputs. Instead of ad-hoc "just ask the AI," it treats prompts as **first-class software artifacts** — versioned, tested, and evaluated like code.

At its core, LLMs simply predict "the most likely next tokens" given an input sequence. A well-engineered prompt constrains and guides that prediction toward the exact output you need.

## Why This Matters

- **Consistency**: Different team members get the same quality outputs
- **Reproducibility**: Results can be measured, compared, and improved
- **Cost efficiency**: Better prompts = fewer retries = lower API costs
- **Safety**: Systematic approach enables guardrails and testing
- **Scalability**: Reusable prompt templates across the organization

---

## Core Principles

### 1. Prompts Have Structure

Every prompt is composed of one or more of these components:

| Component | Purpose | Example |
|-----------|---------|---------|
| **Instructions** | Tell the model what to do | "Summarize the following text in 3 bullet points" |
| **Primary Content** | The data being processed | The actual document to summarize |
| **Examples (Few-shot)** | Show desired input/output patterns | "Input: X → Output: Y" |
| **Cues** | Jumpstart or prime the output format | "Key Points:\n- " |
| **Supporting Content** | Context that influences output | "The audience is C-level executives" |

### 2. Specificity Beats Cleverness

The more specific and constrained your prompt, the better the output:
- ❌ "Write something about the meeting"
- ✅ "Write a 150-word summary of the all-hands meeting, thanking the team for Q3 work, and signed by the SLT"

### 3. Order Matters (Recency Bias)

LLMs are susceptible to **recency bias** — information at the end of the prompt may have more influence than information at the beginning. Best practices:
- Put instructions **before** content (most common)
- **Repeat** critical instructions at the end for emphasis
- Use clear **delimiters** (---,  ```, XML tags) to separate sections

### 4. Show, Don't Just Tell (Few-Shot Learning)

Providing examples is one of the most powerful techniques:

| Type | Description | When to Use |
|------|-------------|-------------|
| **Zero-shot** | No examples, just instructions | Simple, well-defined tasks |
| **One-shot** | Single example | When format/style needs demonstration |
| **Few-shot** | Multiple examples (2-5+) | Complex tasks, classification, extraction |

```
# Few-shot example for classification
Headline: "Twins' Correa to use opt-out, test free agency"
Topic: Baseball

Headline: "Qatar World Cup to have zones for sobering up"  
Topic: Soccer

Headline: "Coach confident injury won't derail Warriors"
Topic: [model predicts: Basketball]
```

### 5. Break Down Complex Tasks

Instead of one massive prompt, decompose into steps:
- **Chain-of-thought**: Ask the model to "think step by step" before answering
- **Multi-step pipelines**: Extract facts first, then reason over them
- **Affordances**: Let the model generate function calls, then execute them

---

## Key Techniques

### Chain-of-Thought (CoT) Prompting

Force the model to show its reasoning before giving a final answer:

```
Who was the most decorated individual athlete at the Sydney Olympics?

Take a step-by-step approach in your response, cite sources and 
give reasoning before sharing final answer in the below format:
ANSWER is: <name>
```

**Why it works**: By requiring intermediate steps, you reduce the chance of reasoning errors. The model can't just guess — it must build toward the answer logically.

**Important note**: Chain-of-thought is for non-reasoning models. Newer reasoning models (o-series, GPT-5+) handle this internally and CoT prompting may be redundant or even harmful.

### Self-Consistency

Run the same prompt multiple times and take the majority answer:
1. Generate N responses (e.g., N=5)
2. Extract the final answer from each
3. Return the most common answer

**Trade-off**: Higher quality but N× the cost and latency.

### Role / Persona Prompting

Assign a role to shape the model's behavior:

```
You are a senior financial analyst with 20 years of experience in 
Fortune 500 companies. You communicate precisely using industry 
terminology and always cite data to support your conclusions.
```

### Output Priming

Start the output for the model to constrain format:

```
Summarize the above email message:
Key Points:
•
```

The model continues from "•" and naturally produces a bulleted list.

### Clear Syntax & Delimiters

Use Markdown, XML, or clear separators to structure prompts:

```
---INSTRUCTIONS---
Analyze the following customer feedback.

---INPUT---
[customer text here]

---OUTPUT FORMAT---
{
  "sentiment": "positive|negative|neutral",
  "topics": ["topic1", "topic2"],
  "urgency": "low|medium|high"
}
```

---

## Parameters That Matter

### Temperature (0.0 - 2.0)

Controls randomness/creativity:

| Value | Behavior | Use Case |
|-------|----------|----------|
| 0.0 | Deterministic, focused | Classification, extraction, factual Q&A |
| 0.3-0.5 | Slightly varied but focused | Summarization, analysis |
| 0.7-1.0 | Creative, diverse | Brainstorming, creative writing |
| 1.0+ | Very random | Rarely useful in production |

### Top-p (Nucleus Sampling)

Alternative to temperature — limits token selection to the top p% cumulative probability:
- `top_p=0.1`: Only considers tokens in the top 10% probability mass → very focused
- `top_p=0.9`: Considers top 90% → more diverse

**Rule**: Adjust **one** (temperature OR top_p), not both simultaneously.

### Max Tokens

Controls the maximum length of the response. Set this intentionally:
- Too low → truncated outputs
- Too high → unnecessary cost
- For structured outputs (JSON), ensure enough room for the full structure

### Frequency / Presence Penalty

- **Frequency penalty** (0-2): Reduces repetition of already-used tokens
- **Presence penalty** (0-2): Encourages the model to talk about new topics

---

## Enterprise Prompt Patterns

### Pattern 1: Structured Extraction

```
Extract the following information from the document below.
Return your answer as JSON matching this exact schema:

{
  "company_name": "string",
  "revenue": "number or null",
  "key_risks": ["string"],
  "sentiment": "positive | negative | neutral"
}

If a field cannot be determined from the document, use null.
Do not invent or hallucinate information.

---DOCUMENT---
[document text]
```

### Pattern 2: Multi-Label Classification

```
Classify the following customer message into one or more categories.
Return confidence scores (0.0 to 1.0) for each category.

Categories: billing, technical_support, feature_request, complaint, praise

Output format:
{
  "categories": [
    {"label": "category_name", "confidence": 0.95}
  ],
  "reasoning": "brief explanation"
}

---MESSAGE---
[customer message]
```

### Pattern 3: Grounded Summarization

```
Summarize the following document in exactly 3 bullet points.

Rules:
- Only use information present in the document
- If the document doesn't contain enough information, say so
- Do not add external knowledge or assumptions
- Keep each bullet point under 30 words

---DOCUMENT---
[document text]

Summary:
•
```

### Pattern 4: Chain-of-Thought Reasoning

```
Analyze the following business scenario and provide a recommendation.

Think through this step by step:
1. Identify the key factors
2. Consider pros and cons of each option
3. Assess risks
4. Make a recommendation with reasoning

---SCENARIO---
[business scenario]

---ANALYSIS---
Step 1: Key Factors
```

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Vague instructions | Assuming the model "knows what you mean" | Be explicit about format, length, style |
| No output format | Model picks its own structure | Always specify JSON, bullets, table, etc. |
| Conflicting instructions | "Be comprehensive" + "Be brief" | Prioritize one; give explicit constraints |
| Missing edge case handling | Not telling model what to do when unsure | Add "If you can't determine X, respond with Y" |
| Prompt injection vulnerability | User input manipulates model behavior | Separate instructions from user data with delimiters |
| Over-reliance on temperature | Thinking temperature fixes bad prompts | Fix the prompt first, then tune temperature |
| Too many examples | Eating up context window | 3-5 high-quality examples > 20 mediocre ones |

---

## Application to My Project

### How I'll Use This

1. **Prompt template system**: Build templates with variable injection for each pattern above
2. **Parameter configuration**: Each template should specify recommended temperature, max_tokens
3. **Few-shot management**: Store examples separately, inject dynamically
4. **Evaluation suite**: Test each technique against the others for the same task

### Decisions to Make

- [ ] Template format: Jinja2 vs custom variable syntax?
- [ ] Storage: YAML files vs Python objects vs database?
- [ ] How to handle prompt versioning (semantic versioning?)
- [ ] How to inject few-shot examples dynamically?

---

## Resources for Deeper Learning

- [Azure Prompt Engineering Guide](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering) — Comprehensive with examples
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — Official best practices
- [Prompt Engineering for Developers (DeepLearning.AI)](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/) — Free course
- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/claude/docs/prompt-engineering) — Excellent techniques documentation

## Questions Remaining

- [ ] How do reasoning models (o-series) change prompt engineering best practices?
- [ ] What is the optimal number of few-shot examples for different task types?
- [ ] How to systematically measure "prompt quality"?
