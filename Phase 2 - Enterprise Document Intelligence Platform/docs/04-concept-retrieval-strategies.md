---
date: 2026-02-27
type: concept
topic: "Advanced Retrieval Strategies"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
---

# Learning: Advanced Retrieval Strategies

## In My Own Words

Retrieval is the **make-or-break stage** of RAG. The LLM can only reason over what you give it — if the right documents aren't retrieved, even the most powerful model will produce a wrong or incomplete answer. Advanced retrieval goes beyond naive "embed query → find similar chunks" to use multiple strategies for finding the right information.

Think of it as the difference between a library that only has one card catalog (naive) versus one that has keyword search, topic browsing, librarian recommendations, and cross-references (advanced).

## Why This Matters

- **Naive retrieval works for ~60-70% of queries** — the remaining 30-40% need more sophisticated approaches
- **Different query types need different retrieval strategies** — factual lookups vs analytical questions vs multi-hop reasoning
- **Retrieval directly bounds generation quality** — no amount of prompting can compensate for missing context
- **Enterprise queries are complex** — "Compare our M&A methodology from 2024 with the approach we used for the TechCorp deal" requires multi-step retrieval

---

## Core Principles

### 1. Retrieval Is a Multi-Stage Pipeline

Don't think of retrieval as one step — it's a pipeline:

```
User Query
    │
    ▼
┌──────────────────┐
│  Pre-Retrieval   │  Transform/expand the query
│  (Query Side)    │
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Retrieval       │  Find candidate documents (fast, broad)
│  (Search Stage)  │
└────────┬─────────┘
         │
    ▼
┌──────────────────┐
│  Post-Retrieval  │  Refine, rerank, compress (slower, precise)
│  (Refinement)    │
└────────┬─────────┘
         │
    ▼
Final Context for LLM
```

### 2. Recall First, Then Precision

In the retrieval stage, cast a wide net (high recall). In the post-retrieval stage, narrow down to the most relevant pieces (high precision).

```
Retrieval: Get 50 potentially relevant chunks (high recall)
     ↓
Reranking: Score and keep top 10 (high precision)
     ↓
Prompt:    Include top 5-8 in LLM context
```

### 3. The Query Is Often the Problem

Users don't always ask questions the way documents are written. Query transformation bridges this gap:

```
User asks:    "What's our approach to digital transformation?"
Documents say: "The Digital Accelerator Methodology (DAM) provides a structured
               framework for enterprise-wide digital modernization..."
```

Direct embedding similarity might miss this. Query transformation can help.

---

## Pre-Retrieval Strategies

### Strategy 1: Query Rewriting

**Problem**: User queries are often vague, ambiguous, or use different terminology than the documents.

**Solution**: Use an LLM to rewrite the query before retrieval.

```python
rewrite_prompt = """Given the user's question, rewrite it to be more specific 
and better suited for searching a corporate knowledge base.

Original question: {query}
Rewritten question:"""

# Example:
# Input:  "What about the cloud stuff?"
# Output: "What is our cloud migration methodology and what services 
#          do we offer for cloud transformation?"
```

### Strategy 2: HyDE (Hypothetical Document Embeddings)

**Problem**: Short queries embed differently than long documents — the embedding spaces don't perfectly overlap.

**Solution**: Ask the LLM to generate a **hypothetical answer**, then embed THAT instead of the query. The hypothetical answer is more document-like and closer in embedding space.

```python
hyde_prompt = """Answer the following question as if you were a consulting 
firm's knowledge base. Write a detailed paragraph.

Question: {query}
Answer:"""

# 1. LLM generates a hypothetical answer (may have hallucinations — that's OK)
hypothetical_doc = llm.generate(hyde_prompt.format(query=user_query))

# 2. Embed the hypothetical document (closer to real docs in embedding space)
hyde_embedding = embed(hypothetical_doc)

# 3. Search with this embedding
results = vector_db.similarity_search(hyde_embedding, k=20)
```

**Why it works**: The hypothetical document uses vocabulary and phrasing similar to actual documents, producing a better embedding for retrieval.

**When to use**: Complex questions, when direct query embedding underperforms.
**Risk**: Adds an LLM call (cost + latency). The hypothetical doc may bias retrieval.

### Strategy 3: Multi-Query Retrieval

**Problem**: A single query may not capture all aspects of the user's information need.

**Solution**: Generate multiple query variations and retrieve for each, then merge results.

```python
multi_query_prompt = """Generate 3 different versions of the following question
that capture different aspects of the information need. Each version should 
approach the topic from a different angle.

Original: {query}

Version 1:
Version 2:
Version 3:"""

# Example:
# Original: "How do we handle data privacy in cloud projects?"
# V1: "What are our data privacy policies for cloud migration engagements?"
# V2: "What compliance frameworks do we follow for cloud data protection?"
# V3: "What technical safeguards do we implement for data privacy in cloud?"

# Retrieve for each version, combine with RRF
all_results = []
for query_version in [original, v1, v2, v3]:
    results = retrieve(query_version, k=10)
    all_results.append(results)

final = reciprocal_rank_fusion(all_results)
```

**When to use**: Broad questions, exploratory queries, when you want comprehensive coverage.

### Strategy 4: Step-Back Prompting

**Problem**: Specific questions sometimes need broader context to answer well.

**Solution**: Generate a more general "step-back" question, retrieve for both.

```python
# Original: "What was the revenue impact of the TechCorp M&A deal in Q3 2024?"
# Step-back: "What are the financial outcomes and revenue impacts of our M&A deals?"

# Retrieve for both questions → combine results
# The step-back may find the M&A methodology doc that provides framework
# The original may find the specific TechCorp case study
```

### Strategy 5: Query Decomposition

**Problem**: Complex questions require information from multiple documents.

**Solution**: Break the question into sub-questions, retrieve for each.

```python
decompose_prompt = """Break this complex question into simpler sub-questions
that can each be answered independently.

Complex question: {query}
Sub-questions:"""

# Example:
# Complex: "Compare our 2024 M&A methodology with what we used for TechCorp"
# Sub-Q1: "What is our current M&A methodology as of 2024?"
# Sub-Q2: "What approach did we use for the TechCorp acquisition deal?"
# Sub-Q3: "What are the key differences between methodology versions?"
```

---

## Retrieval Strategies

### Strategy 1: Naive Vector Search (Baseline)

```python
# Simple top-K retrieval
results = vector_db.similarity_search(query_embedding, k=5)
```

Fast, simple, works for many cases. Use as baseline.

### Strategy 2: Hybrid Search (Dense + Sparse)

Covered in the embeddings doc — combine semantic and keyword search:

```python
# Dense (semantic) results
dense_results = vector_db.similarity_search(query_embedding, k=20)

# Sparse (BM25/keyword) results
sparse_results = bm25_index.search(query_text, k=20)

# Combine with Reciprocal Rank Fusion
final_results = rrf_merge(dense_results, sparse_results, k=60)
```

**This should be your default** — almost always outperforms either method alone.

### Strategy 3: Metadata Filtering + Vector Search

Use document metadata to narrow the search space BEFORE vector search:

```python
# Filter first, then search within filtered set
results = vector_db.similarity_search(
    query_embedding,
    k=10,
    filter={
        "document_type": "case_study",
        "year": {"$gte": 2023},
        "department": "strategy"
    }
)
```

**When to use**: When the query implies a specific scope (time period, document type, department).

### Strategy 4: Sentence Window Retrieval

**Problem**: Small chunks are great for precise retrieval but may lack surrounding context.

**Solution**: Retrieve the matching chunk, but expand to include surrounding sentences.

```
Document: [S1] [S2] [S3] [S4] [S5] [S6] [S7] [S8] [S9] [S10]

Chunk matched: [S5]
Window (±2):   [S3] [S4] [S5] [S6] [S7]   ← This gets sent to LLM
```

```python
# At indexing time: embed individual sentences
# At retrieval time: retrieve sentence, expand to window

class SentenceWindowRetriever:
    def retrieve(self, query, k=5, window_size=2):
        # Find matching sentences
        matches = self.vector_db.search(query, k=k)
        
        # Expand each match to include surrounding sentences
        expanded = []
        for match in matches:
            doc_id = match.metadata["doc_id"]
            sent_idx = match.metadata["sentence_index"]
            
            # Get surrounding sentences
            start = max(0, sent_idx - window_size)
            end = sent_idx + window_size + 1
            window = self.get_sentences(doc_id, start, end)
            expanded.append(window)
        
        return expanded
```

**Trade-off**: Better context than small chunks, more precise than large chunks.

### Strategy 5: Parent-Child (Auto-Merging) Retrieval

**Problem**: Need precise matching (small chunks) AND rich context (large chunks).

**Solution**: Index small chunks, but when multiple children from the same parent are retrieved, merge up to the parent.

```
Parent: "Chapter 3: Post-Merger Integration"  (2000 tokens)
├── Child A: "Phase 1: Due Diligence..."       (300 tokens) ← matched
├── Child B: "Phase 2: Integration Planning..." (300 tokens) ← matched
├── Child C: "Phase 3: Execution..."            (300 tokens)
└── Child D: "Phase 4: Review..."               (300 tokens) ← matched

3 out of 4 children matched → auto-merge to parent chunk
→ Send full "Chapter 3" to LLM instead of 3 fragments
```

```python
class AutoMergingRetriever:
    def retrieve(self, query, k=10, merge_threshold=0.5):
        # Retrieve child chunks
        children = self.vector_db.search(query, k=k)
        
        # Group by parent
        parent_groups = defaultdict(list)
        for child in children:
            parent_id = child.metadata["parent_id"]
            parent_groups[parent_id].append(child)
        
        results = []
        for parent_id, matched_children in parent_groups.items():
            parent = self.get_parent(parent_id)
            total_children = parent.metadata["num_children"]
            match_ratio = len(matched_children) / total_children
            
            if match_ratio >= merge_threshold:
                # Many children matched → use parent for richer context
                results.append(parent)
            else:
                # Few children matched → use children for precision
                results.extend(matched_children)
        
        return results
```

**When to use**: Documents with clear hierarchical structure (reports, manuals, guides).

### Strategy 6: Multi-Index Retrieval

**Problem**: Different document types or collections might benefit from separate indexes.

**Solution**: Maintain multiple indexes, query relevant ones based on the question.

```python
indexes = {
    "case_studies": vector_db_case_studies,
    "methodologies": vector_db_methodologies,
    "proposals": vector_db_proposals,
    "research": vector_db_research
}

# Route query to relevant indexes
relevant_indexes = classify_query(query)  # LLM or rule-based routing

results = []
for idx_name in relevant_indexes:
    idx_results = indexes[idx_name].search(query, k=5)
    results.extend(idx_results)
```

---

## Post-Retrieval Strategies

### Reranking

The most impactful post-retrieval strategy. Initial retrieval (embedding similarity) is fast but imprecise. Reranking uses a more powerful model to re-score results.

#### Cross-Encoder Reranking

A cross-encoder processes **query + document together** (not separately like embedding models), enabling much more accurate relevance scoring.

```python
from sentence_transformers import CrossEncoder

# Cross-encoder sees query AND document together
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Score all retrieved documents
pairs = [(query, doc.page_content) for doc in retrieved_docs]
scores = reranker.predict(pairs)

# Sort by reranker score
reranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
top_results = [doc for doc, score in reranked[:5]]
```

**Why it's better than embedding similarity**: Embedding models encode query and document independently — they can't model the fine-grained interaction between them. Cross-encoders see both together and can reason about specific matches.

**Trade-off**: Much slower than embedding similarity (can't be pre-computed). Use on top-20-50 candidates, not the full index.

#### LLM-Based Reranking

Use an LLM to judge relevance:

```python
rerank_prompt = """Given the question and document, rate the document's 
relevance on a scale of 0-10. Only consider whether the document contains 
information that helps answer the question.

Question: {query}
Document: {document}

Relevance score (0-10):"""

# More expensive but can understand nuance better than cross-encoders
```

**When to use**: When cross-encoder reranking isn't sufficient, or when you need explainable relevance judgments.

#### Cohere Rerank API

A hosted reranking service — easy to integrate:

```python
import cohere

co = cohere.Client("your-api-key")

results = co.rerank(
    model="rerank-english-v3.0",
    query="What is our M&A methodology?",
    documents=[doc.page_content for doc in retrieved_docs],
    top_n=5
)
```

### Context Compression

**Problem**: Retrieved chunks may contain relevant AND irrelevant information.

**Solution**: Use an LLM to extract only the relevant parts before stuffing the prompt.

```python
compress_prompt = """Given the question and context, extract ONLY the parts 
of the context that are relevant to answering the question. If nothing is 
relevant, say "NOT_RELEVANT".

Question: {query}
Context: {chunk}

Relevant extract:"""
```

This reduces tokens sent to the final LLM, saving cost and reducing noise.

### Diversity Filtering

**Problem**: Top results are all from the same document/section (redundant).

**Solution**: Apply Maximum Marginal Relevance (MMR) to balance relevance and diversity.

$$\text{MMR} = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot \text{sim}(d, q) - (1-\lambda) \cdot \max_{d' \in S} \text{sim}(d, d') \right]$$

In plain English: pick documents that are relevant to the query BUT also different from documents already selected.

```python
# LangChain supports MMR out of the box
results = vector_db.max_marginal_relevance_search(
    query,
    k=5,
    fetch_k=20,     # Fetch 20 candidates
    lambda_mult=0.7  # 0.7 relevance, 0.3 diversity
)
```

---

## Retrieval Strategy Selection Guide

| Query Type | Recommended Strategy | Why |
|-----------|---------------------|-----|
| **Simple factual** ("What is X?") | Hybrid search + reranking | Direct match, precision matters |
| **Conceptual** ("How does X work?") | Multi-query + hybrid + reranking | Need multiple perspectives |
| **Comparative** ("Compare X and Y") | Query decomposition + multi-retrieval | Need info about both X and Y |
| **Temporal** ("What changed since 2024?") | Metadata filtering + hybrid | Time filter critical |
| **Exploratory** ("Tell me about X") | Multi-query + MMR diversity | Need breadth, not just depth |
| **Multi-hop** ("What did X do for Y's Z?") | Query decomposition + iterative retrieval | Need to chain information |

---

## Best Practices

- ✅ **Always use hybrid search** — dense + sparse almost always beats either alone
- ✅ **Always add reranking** — it's the highest-ROI improvement after hybrid search
- ✅ **Use metadata filters** — reducing the search space improves precision dramatically
- ✅ **Retrieve more than you need, then filter** — get top-20, rerank to top-5
- ✅ **Log retrieval results** — you need to see what's retrieved to debug quality issues
- ✅ **Test with diverse query types** — factual, conceptual, comparative, temporal
- ❌ **Don't just increase K** — more chunks ≠ better answers (noise increases)
- ❌ **Don't skip reranking to save cost** — it's cheap compared to the quality gain
- ❌ **Don't use one retrieval strategy for all queries** — route based on query type if possible
- ❌ **Don't ignore retrieval latency** — query expansion adds LLM calls; budget accordingly

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Retrieving too many chunks | "More context = better answer" thinking | Measure quality vs K; usually 3-8 chunks is optimal |
| Ignoring chunk overlap in results | Same info retrieved from overlapping chunks | Deduplicate results before sending to LLM |
| Not evaluating retrieval separately | Only measuring final answer quality | Track recall@k and MRR independently |
| Over-engineering query expansion | Adding complexity without measuring benefit | A/B test each strategy against baseline |
| Reranking the wrong candidates | Reranker can only reorder what's retrieved | Ensure initial retrieval has good recall (retrieve broadly) |

---

## Application to Our Project

### Recommended Retrieval Pipeline

```
User Query
    │
    ├─→ [Query Rewriting] (LLM cleanup for vague queries)
    │
    ├─→ [Multi-Query] (3 variations for complex questions)
    │
    ▼
┌─────────────────────┐
│   Hybrid Search     │  Dense (embedding) + Sparse (BM25)
│   k=20 per method   │  Combined with RRF
└────────┬────────────┘
         │
    ▼
┌─────────────────────┐
│   Metadata Filter   │  Filter by doc type, date, relevance
└────────┬────────────┘
         │
    ▼
┌─────────────────────┐
│   Cross-Encoder     │  Rerank top ~30 results
│   Reranking         │  Keep top 5-8
└────────┬────────────┘
         │
    ▼
┌─────────────────────┐
│   MMR Diversity     │  Ensure results aren't redundant
└────────┬────────────┘
         │
    ▼
Context for LLM (5-8 chunks with metadata)
```

### Decisions to Make

- [ ] Which cross-encoder model for reranking (open-source vs Cohere API)
- [ ] Whether to implement HyDE (adds latency but may help for complex queries)
- [ ] Query routing: should we classify queries and route to different strategies?
- [ ] How many chunks to include in final LLM context (start with 5, measure)
- [ ] Whether to implement auto-merging retrieval (depends on document structure)

### Implementation Notes

- **Phase 1**: Hybrid search + cross-encoder reranking (biggest bang for buck)
- **Phase 2**: Add multi-query for complex questions
- **Phase 3**: Add metadata filtering and query routing
- Measure recall@10 at each phase to track improvement
- Log every retrieval so you can debug quality issues

---

## Resources for Deeper Learning

- [LangChain Retrievers documentation](https://python.langchain.com/docs/how_to/#retrievers) — All built-in retrieval strategies
- [LlamaIndex Retriever guide](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/) — Alternative perspective on retrieval
- [Cohere Rerank documentation](https://docs.cohere.com/docs/rerank-2) — Production reranking API
- [HyDE paper](https://arxiv.org/abs/2212.10496) — Original hypothetical document embeddings paper
- [sentence-transformers Cross-Encoders](https://www.sbert.net/docs/cross_encoder/pretrained_models.html) — Open-source rerankers

---

## Questions Remaining

- [ ] What's the latency budget for retrieval? Can we afford multi-query + reranking?
- [ ] Should we implement query routing (classify query → route to strategy) or keep it simple?
- [ ] How to handle queries that require information from 10+ documents (beyond context window)?
- [ ] Is Cohere Rerank worth the cost vs open-source cross-encoders for our use case?
