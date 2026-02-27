---
date: 2026-02-27
type: technology
topic: "Vector Databases"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
decision: evaluate Qdrant (primary) with Azure AI Search as enterprise alternative
---

# Technology Brief: Vector Databases

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Specialized databases for storing, indexing, and querying high-dimensional vectors |
| **For** | Fast similarity search over document embeddings in RAG systems |
| **Candidates** | Qdrant, Weaviate, Azure AI Search, Chroma, pgvector |
| **Decision** | Qdrant (primary — best DX, performance, open-source) |

## Why Vector Databases?

You could store embeddings in a regular database (PostgreSQL, SQLite) and compute similarity on every query, but this doesn't scale:

- **100K vectors × 1536 dimensions = 600MB** of raw data
- Brute-force search: compare query against ALL vectors → seconds per query
- Vector DBs use ANN indexes (HNSW, IVF) → milliseconds per query

A vector database provides:
1. **Efficient ANN indexing** (HNSW, IVF, etc.)
2. **Metadata filtering** (filter by document type, date, etc.)
3. **CRUD operations** (add, update, delete documents)
4. **Horizontal scaling** (handle millions of vectors)
5. **Persistence** (data survives restarts)

---

## Candidate Comparison

### Qdrant

**Overview**: Purpose-built vector database written in Rust. Excellent performance, rich filtering, great developer experience.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Connect
client = QdrantClient(url="http://localhost:6333")

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,              # Embedding dimensions
        distance=Distance.COSINE
    )
)

# Insert documents
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],  # 1536-dim embedding
            payload={
                "text": "Our M&A methodology...",
                "source": "methodology-guide.pdf",
                "page": 12,
                "document_type": "methodology",
                "date": "2024-06"
            }
        )
    ]
)

# Search with metadata filtering
results = client.search(
    collection_name="documents",
    query_vector=[0.15, 0.25, ...],  # Query embedding
    query_filter={
        "must": [
            {"key": "document_type", "match": {"value": "methodology"}},
            {"key": "date", "range": {"gte": "2023-01"}}
        ]
    },
    limit=10
)
```

| Pros | Cons |
|------|------|
| ✅ Written in Rust — very fast | ❌ Smaller ecosystem than some alternatives |
| ✅ Rich filtering with payload indexes | ❌ Managed cloud can be pricey at scale |
| ✅ Excellent Python client | ❌ Less enterprise sales/support than Azure |
| ✅ Built-in hybrid search (sparse + dense) | |
| ✅ Easy Docker deployment | |
| ✅ Quantization support (reduce memory) | |
| ✅ Named vectors (multiple embeddings per point) | |
| ✅ Snapshot and backup support | |

**Docker setup:**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

### Weaviate

**Overview**: Open-source vector database with built-in vectorization modules. Can embed documents internally.

```python
import weaviate

client = weaviate.Client("http://localhost:8080")

# Create schema
client.schema.create_class({
    "class": "Document",
    "vectorizer": "text2vec-openai",  # Built-in embedding
    "properties": [
        {"name": "text", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]},
        {"name": "document_type", "dataType": ["string"]},
    ]
})

# Insert (Weaviate can auto-embed!)
client.data_object.create(
    class_name="Document",
    data_object={
        "text": "Our M&A methodology...",
        "source": "methodology-guide.pdf",
        "document_type": "methodology"
    }
)

# Hybrid search (built-in)
results = (
    client.query
    .get("Document", ["text", "source"])
    .with_hybrid(query="merger integration approach", alpha=0.75)
    .with_limit(10)
    .do()
)
```

| Pros | Cons |
|------|------|
| ✅ Built-in vectorization (can embed for you) | ❌ Heavier resource requirements |
| ✅ Native hybrid search (BM25 + vector) | ❌ More complex setup than Qdrant |
| ✅ GraphQL-like query language | ❌ API can be verbose |
| ✅ Multi-tenancy support | ❌ Learning curve for schema design |
| ✅ Active community and development | |
| ✅ Good documentation | |

**Docker setup:**
```bash
docker compose up -d  # Using their docker-compose.yml
```

### Azure AI Search

**Overview**: Microsoft's fully managed search service with vector search capabilities. Enterprise-grade with Azure integration.

```python
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

# Connect
client = SearchClient(
    endpoint="https://my-search.search.windows.net",
    index_name="documents",
    credential=AzureKeyCredential("your-key")
)

# Search (after index is configured in Azure portal)
vector_query = VectorizedQuery(
    vector=[0.1, 0.2, ...],
    k_nearest_neighbors=10,
    fields="embedding"
)

results = client.search(
    search_text="merger integration approach",  # Keyword search
    vector_queries=[vector_query],               # Vector search
    select=["text", "source", "document_type"],
    filter="document_type eq 'methodology'",
    top=10
)
```

| Pros | Cons |
|------|------|
| ✅ Fully managed — no infrastructure to maintain | ❌ Azure lock-in |
| ✅ Built-in hybrid search (BM25 + vector) | ❌ More expensive than self-hosted |
| ✅ Enterprise features (RBAC, compliance, SLA) | ❌ Less control over indexing |
| ✅ Semantic ranking built-in | ❌ Slower iteration during development |
| ✅ Azure ecosystem integration | ❌ Pricing can be complex |
| ✅ Managed scaling | |

### Chroma

**Overview**: Lightweight, embedded vector database. Great for prototyping and small projects.

```python
import chromadb

client = chromadb.Client()  # In-memory
# or: client = chromadb.PersistentClient(path="./chroma_data")

collection = client.create_collection("documents")

collection.add(
    documents=["Our M&A methodology..."],
    metadatas=[{"source": "methodology-guide.pdf"}],
    ids=["doc1"]
)

results = collection.query(
    query_texts=["merger integration approach"],
    n_results=10,
    where={"source": "methodology-guide.pdf"}
)
```

| Pros | Cons |
|------|------|
| ✅ Simplest to get started | ❌ Not designed for production scale |
| ✅ In-memory or persistent | ❌ Limited filtering capabilities |
| ✅ Zero-config setup | ❌ No built-in hybrid search |
| ✅ Great for prototyping | ❌ No distributed deployment |

### pgvector (PostgreSQL Extension)

**Overview**: Vector search as a PostgreSQL extension. Good if you already use PostgreSQL.

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    text TEXT,
    source VARCHAR(255),
    embedding vector(1536)
);

-- Create index
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);

-- Search
SELECT text, source, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

| Pros | Cons |
|------|------|
| ✅ Leverage existing PostgreSQL knowledge | ❌ Not as optimized as purpose-built vector DBs |
| ✅ ACID transactions with vectors | ❌ Scaling is harder |
| ✅ SQL interface | ❌ Filtering less flexible |
| ✅ No new infrastructure | ❌ No native hybrid search |

---

## Decision Matrix

| Feature | Qdrant | Weaviate | Azure AI Search | Chroma | pgvector |
|---------|--------|----------|-----------------|--------|----------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Ease of setup** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Filtering** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Hybrid search** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| **Scalability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Docker support** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | N/A (managed) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **LangChain integration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Enterprise readiness** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Cost (development)** | Free (Docker) | Free (Docker) | ~$75+/month | Free | Free |
| **Learning value** | High | High | Medium | Low | Medium |

---

## Recommendation: Qdrant

**Primary choice: Qdrant** for the following reasons:

1. **Best developer experience** — simple Python client, intuitive API
2. **Excellent performance** — Rust-based, HNSW indexing, fast filtering
3. **Built-in sparse vectors** — enables hybrid search without a separate BM25 index
4. **Easy Docker deployment** — one command to start, perfect for our first containerization experience
5. **Rich filtering** — payload indexing for metadata filtering (document type, date, etc.)
6. **Great LangChain integration** — well-supported in the ecosystem
7. **Production-ready** — used in production by many companies, not just a toy
8. **Learning value** — understanding a purpose-built vector DB is valuable knowledge

**Alternative: Azure AI Search** if we want enterprise features (managed service, RBAC, compliance). Good to know for future Azure-focused deployments.

### Getting Started with Qdrant

```bash
# 1. Start Qdrant with Docker
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# 2. Install Python client
pip install qdrant-client

# 3. Access dashboard
# http://localhost:6333/dashboard
```

```python
# Minimal working example
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

# Create collection for document chunks
client.create_collection(
    collection_name="consulting_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

# Insert a chunk
client.upsert(
    collection_name="consulting_docs",
    points=[PointStruct(
        id=1,
        vector=embedding,  # From your embedding model
        payload={"text": chunk_text, "source": "guide.pdf", "page": 5}
    )]
)

# Search
hits = client.search(
    collection_name="consulting_docs",
    query_vector=query_embedding,
    limit=5
)

for hit in hits:
    print(f"Score: {hit.score:.3f} | {hit.payload['source']} p.{hit.payload['page']}")
    print(f"  {hit.payload['text'][:100]}...")
```

---

## Qdrant Key Features for Our Project

### Payload Filtering

```python
# Filter by document type AND date range
results = client.search(
    collection_name="consulting_docs",
    query_vector=query_embedding,
    query_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="document_type",
                match=models.MatchValue(value="case_study")
            ),
            models.FieldCondition(
                key="year",
                range=models.Range(gte=2023)
            )
        ]
    ),
    limit=10
)
```

### Sparse Vectors (for Hybrid Search)

```python
from qdrant_client.models import SparseVector, NamedSparseVector

# Create collection with both dense and sparse vectors
client.create_collection(
    collection_name="consulting_docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    sparse_vectors_config={
        "bm25": models.SparseVectorParams()
    }
)

# Insert with both vector types
client.upsert(
    collection_name="consulting_docs",
    points=[PointStruct(
        id=1,
        vector=dense_embedding,
        payload={"text": chunk_text, "source": "guide.pdf"},
    )],
    # Sparse vector in same point
)

# Hybrid search combines dense + sparse automatically
```

### Quantization (Memory Optimization)

```python
# Reduce memory by ~4x with scalar quantization
client.update_collection(
    collection_name="consulting_docs",
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True
        )
    )
)
```

---

## Enterprise Considerations

| Consideration | Qdrant | Azure AI Search |
|--------------|--------|-----------------|
| **Hosting** | Self-hosted (Docker) or Qdrant Cloud | Fully managed by Azure |
| **Security** | API key, TLS, RBAC (cloud) | Azure AD, RBAC, encryption at rest |
| **Compliance** | Depends on your hosting | SOC 2, HIPAA, ISO 27001 |
| **SLA** | Self-managed or 99.9% (cloud) | 99.9% or 99.95% |
| **Backup** | Snapshots (manual or automated) | Automated by Azure |
| **Cost at 100K docs** | ~$0 (Docker) or ~$25/month (cloud) | ~$75-250/month |
| **Cost at 1M docs** | ~$50-100/month (cloud) | ~$250-750/month |

---

## Resources for Deeper Learning

- [Qdrant documentation](https://qdrant.tech/documentation/) — Comprehensive docs with examples
- [Qdrant + LangChain guide](https://python.langchain.com/docs/integrations/vectorstores/qdrant/) — Integration guide
- [Weaviate documentation](https://weaviate.io/developers/weaviate) — Alternative vector DB
- [Azure AI Search vector docs](https://learn.microsoft.com/en-us/azure/search/vector-search-overview) — Enterprise option
- [Vector Database comparison (2024)](https://benchmark.vectorview.ai/) — Performance benchmarks

---

## Questions Remaining

- [ ] What's the performance difference between Qdrant and Weaviate for our document scale?
- [ ] How to handle Qdrant backups and data persistence in Docker?
- [ ] Should we index metadata fields for faster filtering? (Yes — but which fields?)
- [ ] What HNSW parameters to start with for our collection size?
