---
date: 2026-02-27
type: concept
topic: "Embeddings and Vector Search"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
---

# Learning: Embeddings and Vector Search

## In My Own Words

Embeddings are the bridge between **human language** and **machine-searchable math**. An embedding model converts text into a dense vector (a list of numbers, typically 256-3072 dimensions) where texts with similar **meaning** end up close together in vector space, and dissimilar texts end up far apart.

This is what makes RAG possible: instead of matching keywords (like traditional search), you match **concepts**. A query about "employee turnover" will match documents about "staff attrition" and "workforce retention" — even though they share zero keywords.

Vector search is the engine that finds the closest vectors to a query vector. A vector database is the infrastructure that makes this fast at scale.

## Why This Matters

- **Semantic understanding**: Goes beyond keyword matching to meaning-based retrieval
- **Cross-lingual potential**: Similar concepts in different languages can map to nearby vectors
- **Multi-modal foundation**: Same approach extends to images, audio, code
- **Scale**: Modern vector databases search millions of vectors in milliseconds
- **Foundation of RAG**: Without good embeddings, retrieval fails, and the whole pipeline collapses

---

## Core Principles

### 1. Embeddings Capture Meaning, Not Keywords

```
Traditional keyword search:
  Query: "How to reduce employee turnover?"
  ❌ Misses: "Staff retention strategies and reducing attrition rates"
  ✅ Matches: "Employee turnover rate calculation formula" (wrong intent!)

Semantic (embedding) search:
  Query: "How to reduce employee turnover?"
  ✅ Matches: "Staff retention strategies and reducing attrition rates"
  ✅ Matches: "Best practices for keeping your workforce engaged"
  ❌ Correctly deprioritizes: "Employee turnover rate calculation formula"
```

### 2. The Embedding Space Is Geometric

Vectors live in high-dimensional space where distance = dissimilarity:

```
In a simplified 2D visualization:

    "dog training tips"  •
                            •  "puppy obedience classes"
    
    "machine learning"  •
                           •  "neural network training"
    
    "Italian cooking"  •
                          •  "pasta recipes"
```

Texts about similar topics cluster together. The embedding model learns these relationships from massive training data.

### 3. Same Model for Queries and Documents

This is critical: the **same embedding model** must be used to embed both your documents (at ingestion time) and user queries (at search time). If you use different models, the vectors live in different spaces and distances are meaningless.

```python
# Ingestion time
doc_embedding = embedding_model.embed("Our merger integration methodology...")

# Query time — SAME MODEL
query_embedding = embedding_model.embed("How do we handle post-merger integration?")

# Now we can compare these meaningfully
similarity = cosine_similarity(query_embedding, doc_embedding)  # → 0.87 (high!)
```

### 4. Dimensionality Is a Trade-off

| Dimensions | Accuracy | Speed | Storage | Example Models |
|-----------|----------|-------|---------|----------------|
| 256-384 | Good | Fast | Small | MiniLM, small sentence-transformers |
| 768-1024 | Very good | Medium | Medium | BGE, E5, Cohere embed-v3 |
| 1536 | Excellent | Slower | Larger | OpenAI text-embedding-ada-002 |
| 3072 | Best | Slowest | Largest | OpenAI text-embedding-3-large |

More dimensions capture more nuance, but cost more to store and search.

---

## How Embeddings Work

### The Transformer Pipeline

```
Input Text: "RAG systems retrieve relevant documents"
     │
     ▼
┌─────────────┐
│  Tokenizer  │  → Splits text into tokens (subwords)
└─────────────┘
     │
     ▼
┌─────────────┐
│ Transformer │  → Processes tokens with attention mechanism
│   Layers    │     (understands relationships between words)
└─────────────┘
     │
     ▼
┌─────────────┐
│   Pooling   │  → Combines token representations into one vector
│   Layer     │     (mean pooling, CLS token, etc.)
└─────────────┘
     │
     ▼
Output: [0.023, -0.156, 0.891, ..., 0.044]  (1536 dimensions)
```

### Training Objective: Contrastive Learning

Embedding models are trained to:
- Pull **similar** pairs **closer** in vector space (positive pairs)
- Push **dissimilar** pairs **farther apart** (negative pairs)

Training data looks like:
```
Positive pair (should be close):
  "What is machine learning?" ↔ "ML is a subset of AI that learns from data"

Negative pair (should be far):
  "What is machine learning?" ↔ "The best Italian restaurants in Rome"
```

This is why the training data quality of the embedding model matters — models trained on diverse, high-quality pairs produce better embeddings.

---

## Similarity Metrics

How do we measure "closeness" between two vectors?

### Cosine Similarity (most common)

Measures the **angle** between two vectors, ignoring magnitude:

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

- Range: -1 (opposite) to 1 (identical)
- **Recommended for text** — length of text shouldn't affect similarity
- Most embedding models are optimized for cosine similarity

### Dot Product

$$\text{dot\_product}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} a_i \cdot b_i$$

- Range: -∞ to +∞
- Sensitive to vector magnitude (longer vectors = higher scores)
- Faster to compute than cosine (no normalization)
- **If vectors are normalized (unit length), dot product = cosine similarity**

### Euclidean Distance (L2)

$$\text{euclidean}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

- Range: 0 (identical) to +∞
- Lower = more similar (opposite of similarity)
- Less commonly used for text embeddings

**For our project**: Use **cosine similarity** — it's the standard for text embeddings, and all major vector databases support it.

---

## Dense vs Sparse Retrieval

### Dense Retrieval (Embedding-based)

What we've been discussing — convert text to dense vectors:

```python
# Dense vector: every dimension has a value
[0.023, -0.156, 0.891, 0.044, -0.312, ...]  # 1536 dimensions, all non-zero
```

**Strengths**: Semantic understanding, handles synonyms, cross-lingual
**Weaknesses**: Can miss exact keyword matches, opaque (hard to debug), requires good model

### Sparse Retrieval (Keyword-based)

Traditional approach — represent text by word frequencies (TF-IDF, BM25):

```python
# Sparse vector: most dimensions are zero
# Each dimension = one word in vocabulary
{"merger": 0.8, "integration": 0.6, "methodology": 0.4, ...}
# 99%+ of the 50,000+ vocabulary dimensions are 0
```

**BM25** is the gold standard for sparse retrieval:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}$$

Don't memorize the formula — just know that BM25 scores based on:
- **Term frequency**: How often the query word appears in the document
- **Inverse document frequency**: Rare words matter more than common ones
- **Document length normalization**: Longer documents don't automatically score higher

**Strengths**: Exact keyword matching, fast, well-understood, interpretable
**Weaknesses**: No semantic understanding, synonyms are missed, word order ignored

### Why Both: Hybrid Search

Dense and sparse retrieval have **complementary strengths**:

| Query Type | Dense (Semantic) | Sparse (BM25) |
|-----------|-----------------|----------------|
| "employee turnover reduction" | ✅ Finds "staff attrition" docs | ❌ Misses synonyms |
| "AX-2024-B compliance form" | ❌ Might not match exact code | ✅ Exact keyword match |
| "how to handle M&A integration" | ✅ Understands concepts | ⚠️ Partial keyword overlap |
| "Capgemini Q3 2025 revenue" | ⚠️ May match wrong quarter | ✅ Matches exact terms |

**Hybrid search** combines both and is considered best practice:

```python
# Conceptual hybrid search
dense_results = vector_db.similarity_search(query_embedding, k=20)
sparse_results = bm25_search(query_text, k=20)

# Combine with Reciprocal Rank Fusion (RRF)
final_results = reciprocal_rank_fusion(dense_results, sparse_results)
```

### Reciprocal Rank Fusion (RRF)

The standard way to combine results from different retrieval methods:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where $r(d)$ is the rank of document $d$ in result list $r$, and $k$ is a constant (typically 60).

In plain English: for each document, sum up $\frac{1}{60 + \text{rank}}$ across all result lists. Documents that appear high in multiple lists get the highest combined score.

---

## Approximate Nearest Neighbor (ANN) Search

Finding the exact nearest neighbors among millions of vectors is slow (requires comparing to every vector). Vector databases use **approximate** methods that sacrifice a tiny bit of accuracy for massive speed gains.

### HNSW (Hierarchical Navigable Small World)

The most popular ANN algorithm — used by Qdrant, Weaviate, pgvector, and most modern vector DBs.

**How it works** (simplified):
1. Build a multi-layer graph where nodes are vectors
2. Top layers have few connections (express highways)
3. Bottom layers have many connections (local roads)
4. Search starts at the top and navigates down, getting more precise

```
Layer 3:    A -------- E (few connections, long jumps)
Layer 2:    A --- C --- E --- G
Layer 1:    A - B - C - D - E - F - G - H (many connections, precise)
```

**Key parameters:**
- `ef_construction`: How many neighbors to consider when building the graph (higher = better quality, slower build)
- `M`: Max connections per node (higher = more memory, better recall)
- `ef_search`: How many candidates to explore at query time (higher = better recall, slower search)

### IVF (Inverted File Index)

Clusters vectors into groups, searches only nearby clusters:

1. Divide vectors into N clusters (using k-means)
2. At query time, find the closest clusters
3. Only search vectors in those clusters

**Faster but less accurate than HNSW.** Used when memory is constrained.

### Product Quantization (PQ)

Compresses vectors to use less memory:
- Splits each vector into sub-vectors
- Approximates each sub-vector with a codebook entry
- Trades accuracy for 10-100x memory reduction

**For our project**: HNSW is the default choice. Only consider IVF/PQ if scaling to tens of millions of documents.

---

## Embedding Quality Factors

What determines if your embeddings will work well for retrieval?

| Factor | Impact | How to Optimize |
|--------|--------|-----------------|
| **Model choice** | Huge — the foundation of quality | Benchmark on your domain data |
| **Input quality** | Garbage text → garbage embeddings | Clean documents before embedding |
| **Chunk size** | Too small = no context, too large = diluted | Match to your query patterns |
| **Query-document asymmetry** | Queries are short, docs are long | Consider asymmetric models or query expansion |
| **Domain specificity** | General models may miss domain terminology | Test on domain data; consider fine-tuned models |
| **Instruction prefixes** | Some models use "query:" vs "document:" prefixes | Follow model-specific instructions |

### Important: Instruction Prefixes

Many modern embedding models expect different prefixes for queries vs documents:

```python
# E5 model family
query_text = "query: How to handle post-merger integration?"
document_text = "passage: The merger integration methodology involves..."

# Cohere Embed v3
# Uses input_type parameter: "search_query" vs "search_document"

# BGE models
query_text = "Represent this sentence for searching relevant passages: How to..."
```

**Forgetting these prefixes can dramatically hurt retrieval quality.** Always check the model card.

---

## Best Practices

- ✅ **Use the same embedding model for queries and documents** — mixing models breaks similarity
- ✅ **Benchmark on YOUR data** — MTEB leaderboard rankings don't predict domain performance
- ✅ **Implement hybrid search** (dense + sparse) — it almost always outperforms either alone
- ✅ **Follow model-specific instructions** — prefixes, max token lengths, normalization
- ✅ **Normalize vectors before storing** — ensures cosine similarity works correctly
- ✅ **Monitor embedding quality** — run periodic retrieval evaluations on test queries
- ❌ **Don't assume bigger dimensions = better** — test smaller, faster models first
- ❌ **Don't embed garbage** — clean text, remove boilerplate, fix encoding issues
- ❌ **Don't ignore the "lost in the middle" problem** — even with great retrieval, placement in prompt matters
- ❌ **Don't forget to re-embed when changing models** — old vectors are incompatible with new models

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Different embed models for query/doc | Model migration without re-indexing | Track model version, re-embed on model change |
| Truncated embeddings | Document chunk exceeds model's max tokens | Check model's max tokens, split before embedding |
| Missing instruction prefixes | Didn't read the model card | Always check model-specific documentation |
| Over-relying on cosine similarity | High similarity ≠ relevant answer | Add reranking stage after initial retrieval |
| Not considering cost | Embedding API calls add up | Calculate cost per chunk and per query at scale |

---

## Application to Our Project

### How We'll Use This

```
                   ┌──────────────┐
                   │  Documents   │
                   └──────┬───────┘
                          │
                   ┌──────▼───────┐
                   │   Chunking   │
                   └──────┬───────┘
                          │
              ┌───────────▼───────────┐
              │   Embedding Model     │  ← Same model for docs & queries
              │  (e.g., text-embed-3) │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │    Vector Database    │  ← HNSW index, cosine similarity
              │   (+ BM25 for sparse) │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │    Hybrid Search      │  ← Dense + Sparse + RRF
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │      Reranking        │  ← Cross-encoder refinement
              └───────────────────────┘
```

### Decisions to Make

- [ ] Embedding model: OpenAI text-embedding-3-small/large vs Cohere vs open-source
- [ ] Dimensions: Full dimensionality vs reduced (cost/speed trade-off)
- [ ] Similarity metric: Cosine (default) vs dot product
- [ ] Hybrid search weights: How much weight for dense vs sparse
- [ ] ANN parameters: HNSW ef and M values (start with defaults, tune later)

### Implementation Notes

- Start with OpenAI `text-embedding-3-small` (1536 dims, cheap, good quality)
- Implement BM25 alongside vector search from the start (hybrid search is worth it)
- Cache embeddings — re-embedding unchanged documents wastes money
- Track embedding model version in document metadata for future migration
- Set up a retrieval evaluation before comparing embedding models

---

## Resources for Deeper Learning

- [OpenAI Embeddings guide](https://platform.openai.com/docs/guides/embeddings) — Practical guide with best practices
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Massive Text Embedding Benchmark rankings
- [Sentence-Transformers docs](https://www.sbert.net/) — Open-source embedding models
- [Pinecone: What are Vector Embeddings?](https://www.pinecone.io/learn/vector-embeddings/) — Visual explanation
- [James Briggs on Hybrid Search](https://www.youtube.com/watch?v=lYxGYXjfrNI) — Practical hybrid search tutorial
- [HNSW explained](https://www.pinecone.io/learn/series/faiss/hnsw/) — Deep dive on the algorithm

---

## Questions Remaining

- [ ] How does text-embedding-3's Matryoshka representation work? Can we use lower dims for faster search and higher dims for reranking?
- [ ] What's the practical latency difference between 1536 and 3072 dimensions at our scale (10K-100K chunks)?
- [ ] Should we pre-compute both sparse (BM25) and dense representations, or compute BM25 at query time?
- [ ] How do multilingual embedding models handle mixed-language enterprise documents?
