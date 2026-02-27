---
date: 2026-02-27
type: concept
topic: "RAG Architecture Fundamentals"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
---

# Learning: RAG Architecture Fundamentals

## In My Own Words

Retrieval-Augmented Generation (RAG) is an architecture pattern that makes LLMs useful for **your specific data** without retraining them. Instead of hoping the model "knows" the answer from its training data, you **retrieve** relevant documents from your own knowledge base and **inject** them into the prompt as context, then let the LLM **generate** an answer grounded in that evidence.

Think of it like an open-book exam: the LLM is the student, your document store is the textbook, and the retrieval system is the index that helps the student find the right pages before answering.

## Why This Matters

- **LLMs have knowledge cutoffs** — they don't know about your internal documents, recent events, or proprietary data
- **Fine-tuning is expensive and brittle** — retraining for every document update is impractical
- **Hallucination is dangerous** — without grounding, LLMs confidently fabricate answers
- **Enterprise data is private** — you can't (and shouldn't) put proprietary documents into model training
- **RAG enables attribution** — you can cite exactly which documents informed each answer

For our consulting firm scenario: decades of proposals, case studies, and methodology guides can be made searchable and synthesizable **without exposing them to model providers** and **without retraining any model**.

---

## Core Principles

### 1. Separation of Knowledge and Reasoning

RAG cleanly separates two concerns:

| Concern | Who Handles It | How |
|---------|---------------|-----|
| **Knowledge** (facts, documents, data) | Retrieval system (vector DB + search) | Index, store, and retrieve relevant chunks |
| **Reasoning** (understanding, synthesis, language) | LLM (GPT-4, Claude, etc.) | Read retrieved context, generate coherent answer |

This means you can update knowledge (add/remove documents) **without touching the model**, and upgrade the model **without re-indexing documents**.

### 2. Garbage In, Garbage Out — At Every Stage

RAG has multiple stages, and quality degrades through the pipeline:

```
Document Quality → Chunking Quality → Embedding Quality → Retrieval Quality → Generation Quality
```

If your chunking is bad, even perfect retrieval can't save you. If retrieval returns irrelevant documents, even GPT-4 will produce a poor answer. **Every stage matters.**

### 3. Retrieval Is the Bottleneck

In most RAG systems, the **retrieval step** is what limits quality — not the LLM. Common failure modes:

- Retrieved chunks are too short (missing context)
- Retrieved chunks are too long (diluting the answer with noise)
- The right document exists but wasn't retrieved (recall failure)
- Wrong documents rank higher than right ones (precision failure)

### 4. Context Window ≠ Knowledge Base

Even with 128K+ context windows, you can't just dump all your documents into the prompt:

- **Cost**: More tokens = more money per query
- **Latency**: More tokens = slower responses
- **Lost in the Middle**: LLMs struggle with information buried in the middle of long contexts
- **Diminishing returns**: After a point, more context hurts quality

RAG selects the **most relevant** context, keeping prompts focused and efficient.

---

## How It Works

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    OFFLINE: INGESTION PIPELINE               │
│                                                              │
│  Documents → Parse → Chunk → Embed → Store in Vector DB     │
│  (PDF,DOCX)  (Extract)  (Split)  (Vectorize)  (Index)      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    ONLINE: QUERY PIPELINE                     │
│                                                              │
│  User Query → Transform → Embed → Retrieve → Rerank →       │
│                                                              │
│  → Build Prompt (query + retrieved context) → LLM →          │
│                                                              │
│  → Generate Answer (with sources) → Return to User           │
└─────────────────────────────────────────────────────────────┘
```

### Step by Step

#### Phase A: Ingestion (Offline — done once per document)

1. **Load Documents**: Read PDFs, DOCX, PPTX, HTML, Markdown files
2. **Parse & Extract**: Convert to plain text, extract metadata (title, author, date, section headers)
3. **Chunk**: Split documents into meaningful pieces (not too big, not too small)
4. **Enrich**: Add metadata to each chunk (source file, page number, section, document type)
5. **Embed**: Convert each chunk into a vector (a list of numbers representing meaning)
6. **Store**: Save vectors + text + metadata in a vector database

#### Phase B: Query (Online — every user question)

1. **Receive Query**: User asks a question in natural language
2. **Transform Query**: Optionally rewrite/expand the query for better retrieval
3. **Embed Query**: Convert the question into a vector using the same embedding model
4. **Retrieve**: Find the most similar document chunks by vector similarity
5. **Rerank**: (Optional) Re-score results with a more accurate model
6. **Build Prompt**: Combine the query + retrieved chunks into a prompt with instructions
7. **Generate**: Send prompt to LLM, get a grounded answer
8. **Post-process**: Extract source citations, add confidence scores, format response

---

## RAG vs Alternatives

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **RAG** | Large/changing knowledge base, need attribution, enterprise data | No retraining, updatable, attributable, cost-effective | Retrieval quality limits output, requires infrastructure |
| **Fine-tuning** | Need to change model behavior/style, small specialized domain | Deep model customization, faster inference | Expensive, needs retraining on updates, no attribution |
| **Long-context stuffing** | Small corpus (<50 pages), simple use case | Simple, no retrieval infrastructure | Expensive per query, "lost in middle" problem, doesn't scale |
| **RAG + Fine-tuning** | Highest quality needs, domain-specific language | Best of both worlds | Complex, expensive, harder to maintain |
| **Knowledge graphs + RAG** | Structured relationships matter, multi-hop reasoning | Better for complex queries, structured knowledge | Complex to build and maintain |

**For our project**: RAG is the clear choice — the consulting firm has a large, evolving document collection where attribution and accuracy are essential.

---

## RAG Taxonomy: Naive vs Advanced

### Naive RAG (baseline)

```
Query → Embed → Top-K Retrieval → Stuff into prompt → Generate
```

Problems with naive RAG:
- No query optimization
- Single retrieval pass
- No reranking
- Fixed chunk size
- No metadata filtering

### Advanced RAG (what we're building)

Improvements at every stage:

| Stage | Naive | Advanced |
|-------|-------|----------|
| **Pre-retrieval** | Raw query | Query transformation, HyDE, multi-query |
| **Chunking** | Fixed-size | Semantic, structural, parent-child |
| **Retrieval** | Single dense search | Hybrid (dense + sparse), multi-stage |
| **Post-retrieval** | None | Reranking, compression, filtering |
| **Generation** | Simple stuffing | Iterative refinement, chain-of-thought, source tracking |
| **Evaluation** | Vibes | Systematic metrics (faithfulness, relevance, recall) |

### Modular RAG (the mental model)

Think of RAG as a modular pipeline where you can swap components:

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  Query   │ → │ Retrieval│ → │ Reranking│ → │  Prompt  │ → │Generation│
│Transform │   │  Engine  │   │  Module  │   │ Builder  │   │  Engine  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
     ↑              ↑              ↑              ↑              ↑
  [HyDE,         [Dense,       [Cross-       [Stuff,        [GPT-4,
   Multi-Q,       Sparse,       encoder,      MapReduce,     Claude,
   Step-back]     Hybrid]       LLM-based]    Refine]        Llama]
```

Each module can be independently tested, swapped, and improved.

---

## Failure Modes of RAG

Understanding how RAG fails is as important as understanding how it works:

| Failure Mode | What Happens | Root Cause | Mitigation |
|-------------|-------------|------------|------------|
| **Retrieval miss** | Right answer exists in docs but isn't retrieved | Poor embeddings, bad chunking, query mismatch | Hybrid search, query expansion, better chunking |
| **Retrieval noise** | Retrieved docs are irrelevant | Semantic similarity ≠ relevance | Reranking, metadata filtering, threshold tuning |
| **Lost in the middle** | LLM ignores relevant context in the middle of prompt | Attention pattern of LLMs | Put key info at start/end, reduce context size |
| **Hallucination despite context** | LLM fabricates beyond what's in retrieved docs | Model tendency, ambiguous prompts | Strict grounding instructions, confidence scoring |
| **Stale data** | Answer based on outdated information | Ingestion pipeline not run recently | Incremental updates, timestamp filtering |
| **Contradictory sources** | Documents disagree, LLM picks one arbitrarily | Multiple versions, different authors | Surface contradictions explicitly, date-based priority |
| **Chunk boundary problem** | Answer spans two chunks, only one retrieved | Fixed chunking breaks semantic units | Overlapping chunks, parent-child retrieval |

---

## Best Practices

- ✅ **Build evaluation first** — You can't improve what you can't measure. Create test questions before optimizing
- ✅ **Start simple, then iterate** — Begin with naive RAG, measure, then add complexity where it helps
- ✅ **Track cost per query** — Token usage adds up fast; know your economics
- ✅ **Use metadata aggressively** — Filter by document type, date, department before vector search
- ✅ **Test with adversarial queries** — "What's our policy on X?" when no such policy exists
- ✅ **Log everything** — Queries, retrieved chunks, generated answers, user feedback
- ❌ **Don't use RAG when a database query suffices** — "How many employees?" doesn't need RAG
- ❌ **Don't stuff the entire context window** — More context ≠ better answers
- ❌ **Don't ignore chunk boundaries** — A chunk that starts mid-sentence is useless
- ❌ **Don't skip reranking** — Initial retrieval is fast but imprecise; reranking is crucial

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Evaluating only generation quality | Easier to check final answer | Evaluate retrieval AND generation separately |
| Using tiny chunks for everything | Tutorials default to 500 tokens | Experiment with sizes; some docs need bigger chunks |
| Ignoring document structure | Treating a PDF as flat text | Use headers, sections, tables as chunking boundaries |
| Single embedding model assumption | "OpenAI is best" thinking | Benchmark on YOUR data; domain matters |
| No fallback for low-confidence | System always answers | Implement "I don't know" when retrieval scores are low |

---

## Application to Our Project

### How We'll Use This

The consulting firm RAG system maps directly to this architecture:

| Component | Our Implementation |
|-----------|-------------------|
| **Documents** | Proposals, case studies, methodology guides, research reports (PDF, DOCX, PPTX, HTML, MD) |
| **Ingestion** | Multi-format pipeline with intelligent chunking and metadata extraction |
| **Vector DB** | Qdrant / Weaviate / Azure AI Search (to be decided in technology research) |
| **Retrieval** | Hybrid search with reranking |
| **Generation** | Source-attributed answers with confidence scoring |
| **Evaluation** | Retrieval metrics + generation quality + end-to-end testing |

### Key Decisions to Make

- [ ] Which vector database to use
- [ ] Which embedding model(s) to use
- [ ] Chunking strategy per document type
- [ ] How to handle document updates (incremental vs full re-index)
- [ ] Context window budget (how many chunks per query)
- [ ] How to surface contradictions between documents

### Implementation Notes

- Start with a small, curated test corpus (10-20 diverse documents)
- Build the evaluation suite before optimizing any component
- Instrument every stage for latency and cost tracking
- Use LangChain's abstractions but understand what's happening underneath

---

## Resources for Deeper Learning

- [RAG paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — The original RAG paper that started it all
- [LangChain RAG documentation](https://python.langchain.com/docs/tutorials/rag/) — Practical implementation guide
- [Pinecone RAG guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) — Excellent visual explanations
- [Jerry Liu (LlamaIndex) talks on advanced RAG](https://www.youtube.com/results?search_query=jerry+liu+advanced+rag) — Deep dives on retrieval patterns
- [RAGAS documentation](https://docs.ragas.io/) — Evaluation framework for RAG

---

## Questions Remaining

- [ ] How to handle documents that are updated frequently (versioning in vector DB)?
- [ ] What's the optimal chunk overlap for enterprise documents?
- [ ] How to handle multi-language documents in the same corpus?
- [ ] What's the cost-performance sweet spot for embedding models?
