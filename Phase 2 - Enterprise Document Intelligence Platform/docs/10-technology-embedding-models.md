---
date: 2026-02-27
type: technology
topic: "Embedding Models Comparison"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
decision: start with OpenAI text-embedding-3-small, benchmark against Cohere embed-v3
---

# Technology Brief: Embedding Models Comparison

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Models that convert text into dense vector representations for semantic similarity |
| **For** | Encoding document chunks and queries for vector search in our RAG pipeline |
| **Candidates** | OpenAI text-embedding-3, Cohere embed-v3, open-source (BGE, E5, GTE) |
| **Decision** | Start with OpenAI text-embedding-3-small; benchmark Cohere embed-v3 as alternative |

---

## Candidate Models

### OpenAI text-embedding-3 Family

The current generation of OpenAI embeddings, with two variants:

#### text-embedding-3-small

| Property | Value |
|----------|-------|
| **Dimensions** | 1536 (default), reducible via Matryoshka |
| **Max tokens** | 8,191 |
| **Price** | $0.02 per 1M tokens |
| **MTEB score** | ~62.3% |
| **Best for** | Cost-effective general use |

#### text-embedding-3-large

| Property | Value |
|----------|-------|
| **Dimensions** | 3072 (default), reducible via Matryoshka |
| **Max tokens** | 8,191 |
| **Price** | $0.13 per 1M tokens |
| **MTEB score** | ~64.6% |
| **Best for** | Maximum quality when cost isn't the primary concern |

```python
from langchain_openai import OpenAIEmbeddings

# Small model (recommended starting point)
embeddings_small = OpenAIEmbeddings(model="text-embedding-3-small")

# Large model
embeddings_large = OpenAIEmbeddings(model="text-embedding-3-large")

# With reduced dimensions (Matryoshka trick)
embeddings_compact = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512  # Reduce from 1536 to 512 — ~60% cost reduction in vector DB
)
```

**Matryoshka Representations**: These models are trained so that the first N dimensions are useful on their own. You can truncate from 1536 → 1024 → 512 → 256 dimensions with graceful quality degradation. This is excellent for trading quality vs. storage/speed.

| Dimensions | Relative Quality | Storage Savings |
|-----------|-----------------|-----------------|
| 1536 (full) | 100% | Baseline |
| 1024 | ~98% | 33% less |
| 512 | ~95% | 67% less |
| 256 | ~90% | 83% less |

### Cohere embed-v3

| Property | Value |
|----------|-------|
| **Dimensions** | 1024 (default), 384, 512, 768, 1024 |
| **Max tokens** | 512 |
| **Price** | ~$0.10 per 1M tokens |
| **MTEB score** | ~64.5% |
| **Best for** | Multilingual, input-type-aware embeddings |

```python
import cohere

co = cohere.Client("your-api-key")

# Document embedding
doc_embeddings = co.embed(
    texts=["Our M&A methodology consists of..."],
    model="embed-english-v3.0",
    input_type="search_document",  # Critical: tells model this is a document
    embedding_types=["float"]
).embeddings.float

# Query embedding
query_embedding = co.embed(
    texts=["What is our M&A methodology?"],
    model="embed-english-v3.0",
    input_type="search_query",    # Critical: tells model this is a query
    embedding_types=["float"]
).embeddings.float
```

**Key feature**: `input_type` parameter — Cohere trains different behavior for queries vs documents, which can improve retrieval quality.

| Pros | Cons |
|------|------|
| ✅ Input-type-aware (query vs document) | ❌ Lower max token limit (512) |
| ✅ Excellent multilingual support | ❌ Requires API key (no local option) |
| ✅ Built-in compression types | ❌ Less LangChain ecosystem support than OpenAI |
| ✅ Competitive quality | ❌ Slightly more expensive than OpenAI small |
| ✅ Cohere Rerank pairs well | |

### Open-Source Models

#### BGE (BAAI General Embedding)

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},  # or "cuda" for GPU
    encode_kwargs={"normalize_embeddings": True},
    query_instruction="Represent this sentence for searching relevant passages: "
)
```

| Property | BGE-large-en-v1.5 | BGE-M3 (multilingual) |
|----------|-------------------|----------------------|
| **Dimensions** | 1024 | 1024 |
| **Max tokens** | 512 | 8,192 |
| **Price** | Free (self-hosted) | Free (self-hosted) |
| **MTEB score** | ~63.5% | ~68.2% |
| **Hardware** | ~2GB RAM | ~4GB RAM |

#### E5 (Microsoft)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")

# E5 requires specific prefixes
query_embedding = model.encode("query: What is our M&A methodology?")
doc_embedding = model.encode("passage: Our M&A methodology consists of...")
```

#### GTE (Alibaba)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("thenlper/gte-large")
embeddings = model.encode(["Our M&A methodology..."])
```

### Open-Source Summary

| Model | Dims | Max Tokens | MTEB | GPU Required? | Notes |
|-------|------|-----------|------|--------------|-------|
| BGE-large-en-v1.5 | 1024 | 512 | 63.5% | Recommended | Query prefix needed |
| BGE-M3 | 1024 | 8192 | 68.2% | Yes | Multilingual, long context |
| E5-large-v2 | 1024 | 512 | 62.0% | Recommended | Query/passage prefixes |
| GTE-large | 1024 | 512 | 63.1% | Recommended | No prefix needed |
| all-MiniLM-L6-v2 | 384 | 256 | 56.3% | No (CPU OK) | Lightweight, fast |
| nomic-embed-text | 768 | 8192 | 62.4% | Recommended | Long context, open-source |

---

## Decision Matrix

| Criterion | OpenAI small | OpenAI large | Cohere v3 | BGE-large | all-MiniLM |
|-----------|-------------|-------------|-----------|-----------|------------|
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Cost** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (free) | ⭐⭐⭐⭐⭐ (free) |
| **Max tokens** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Ease of use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Latency** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (local) | ⭐⭐⭐⭐⭐ (local) |
| **Privacy** | ⭐⭐ (API) | ⭐⭐ (API) | ⭐⭐ (API) | ⭐⭐⭐⭐⭐ (local) | ⭐⭐⭐⭐⭐ (local) |
| **LangChain support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Cost Analysis

For our consulting firm project, let's estimate costs:

### Assumptions
- 1,000 documents averaging 5,000 tokens each → 5M tokens to embed
- Each document splits into ~10 chunks of 500 tokens → 10,000 chunks
- 100 queries per day, each ~50 tokens

### Ingestion Cost (One-Time)

| Model | Cost to Embed 5M tokens |
|-------|------------------------|
| OpenAI text-embedding-3-small | $0.10 |
| OpenAI text-embedding-3-large | $0.65 |
| Cohere embed-v3 | $0.50 |
| Open-source (self-hosted) | $0 (compute costs only) |

### Query Cost (Per Day, 100 queries)

| Model | Cost per Day | Cost per Month |
|-------|-------------|----------------|
| OpenAI small | $0.0001 | $0.003 |
| OpenAI large | $0.00065 | $0.02 |
| Cohere v3 | $0.0005 | $0.015 |
| Open-source | $0 | $0 |

**Conclusion**: Embedding costs are negligible compared to LLM generation costs. Choose based on quality, not price.

---

## Important Considerations

### Token Limits and Truncation

If your chunk exceeds the model's max token limit, it gets **silently truncated**:

```python
# OpenAI: 8,191 max tokens — generous, rarely an issue
# Cohere: 512 max tokens — WILL truncate chunks > 512 tokens!
# BGE/E5: 512 max tokens — same issue

# Always check:
import tiktoken
enc = tiktoken.encoding_for_model("text-embedding-3-small")
token_count = len(enc.encode(chunk_text))
if token_count > 8191:
    print(f"Warning: chunk has {token_count} tokens, will be truncated!")
```

**For our project**: OpenAI's 8,191 token limit is very generous — most chunks will be well under this. If using Cohere or open-source, ensure chunks are < 512 tokens.

### Prefix Requirements

Some models require specific prefixes — forgetting these **dramatically hurts quality**:

| Model | Query Prefix | Document Prefix |
|-------|-------------|-----------------|
| OpenAI | None needed | None needed |
| Cohere | `input_type="search_query"` | `input_type="search_document"` |
| BGE | `"Represent this sentence..."` | None |
| E5 | `"query: "` | `"passage: "` |
| GTE | None needed | None needed |

### Batch Embedding

Always embed in batches for efficiency:

```python
# OpenAI handles batching internally
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=1000  # Batch size for API calls
)

# For open-source models
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
vectors = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
```

---

## Benchmarking on Your Data

MTEB scores are useful but **your domain matters more**. Here's how to benchmark:

```python
def benchmark_embedding_model(model, test_queries, relevant_docs, all_chunks):
    """Benchmark an embedding model on your test set."""
    
    # Embed all chunks
    chunk_embeddings = model.embed_documents([c.page_content for c in all_chunks])
    
    recall_scores = []
    mrr_scores = []
    
    for query, expected_docs in zip(test_queries, relevant_docs):
        # Embed query
        query_embedding = model.embed_query(query)
        
        # Find top-K by cosine similarity
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        top_k_indices = similarities.argsort()[-10:][::-1]
        
        retrieved_sources = [all_chunks[i].metadata["source"] for i in top_k_indices]
        
        # Calculate metrics
        recall_scores.append(recall_at_k(retrieved_sources, expected_docs, k=10))
        mrr_scores.append(reciprocal_rank(retrieved_sources, expected_docs))
    
    return {
        "model": model.model,
        "recall@10": sum(recall_scores) / len(recall_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores)
    }
```

---

## Recommendation

### Start with: OpenAI text-embedding-3-small

**Why:**
1. **Lowest friction** — already using OpenAI for LLM, same API key
2. **Best cost-effectiveness** — $0.02/1M tokens, cheapest quality option
3. **8K token limit** — won't truncate any reasonable chunk
4. **Matryoshka support** — can reduce dimensions later for speed
5. **Excellent LangChain support** — first-class integration
6. **No prefix shenanigans** — just embed, no special formatting

### Benchmark against: Cohere embed-v3

**Why:**
- Input-type awareness may improve retrieval quality
- Pairs naturally with Cohere Rerank
- Strong multilingual support if documents include other languages

### Consider later: Open-source (BGE-M3)

**Why:**
- Data privacy (no API calls)
- Cost elimination at scale
- BGE-M3 supports very long context (8,192 tokens)
- But: requires GPU for reasonable performance

---

## Best Practices

- ✅ **Use the same model for queries and documents** — never mix models
- ✅ **Track which model version was used** — store in metadata for migration
- ✅ **Benchmark on YOUR data** — MTEB scores are directional, not definitive
- ✅ **Consider Matryoshka dimensions** — start with 1536, reduce if speed matters
- ✅ **Batch your embedding calls** — both for cost and latency
- ✅ **Cache embeddings** — don't re-embed unchanged documents
- ❌ **Don't forget model-specific prefixes** — BGE, E5, Cohere all need them
- ❌ **Don't exceed token limits without knowing** — silent truncation loses information
- ❌ **Don't optimize embeddings before evaluating** — measure first, then optimize

---

## Resources for Deeper Learning

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Benchmark rankings
- [OpenAI Embeddings docs](https://platform.openai.com/docs/guides/embeddings) — Official guide
- [Cohere Embed docs](https://docs.cohere.com/docs/embed-2) — Cohere embedding guide
- [Sentence-Transformers](https://www.sbert.net/) — Open-source embedding framework
- [Matryoshka Embeddings paper](https://arxiv.org/abs/2205.13147) — How dimension reduction works

---

## Questions Remaining

- [ ] How much quality do we lose going from 1536 → 512 dimensions for our specific documents?
- [ ] Does Cohere's input_type awareness actually improve retrieval for enterprise docs?
- [ ] What's the latency of local BGE-M3 vs OpenAI API for batch embedding?
- [ ] Should we store both full and reduced-dimension embeddings for different use cases?
