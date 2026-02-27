---
date: 2026-02-27
type: architecture
topic: "End-to-End RAG Pipeline Architecture"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
---

# Architecture: End-to-End RAG Pipeline

## Overview

This document defines the **complete architecture** for the Enterprise Document Intelligence Platform, connecting all the concepts and technologies from the research docs into a cohesive, buildable system.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                              │
│                                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────────┐   │
│  │  Streamlit   │  │  REST API    │  │  (Future: Slack/Teams bot)   │   │
│  │  Web UI      │  │  (FastAPI)   │  │                              │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬───────────────┘   │
│         └─────────────────┼──────────────────────────┘                   │
│                           │                                              │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────┐
│                        APPLICATION LAYER                                 │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     RAG Query Pipeline                            │   │
│  │                                                                    │   │
│  │  Query → Transform → Embed → Retrieve → Rerank → Generate →      │   │
│  │                                                    Respond         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   Document Ingestion Pipeline                     │   │
│  │                                                                    │   │
│  │  Upload → Detect → Parse → Clean → Chunk → Embed → Store         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    Evaluation Pipeline                             │   │
│  │                                                                    │   │
│  │  Test Set → Run Queries → Measure Retrieval → Measure Generation  │   │
│  │          → Track Metrics → Compare Experiments                     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────────────────┐
│                        DATA & SERVICES LAYER                             │
│                                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────┐    │
│  │   Qdrant    │  │  OpenAI    │  │  Reranker   │  │  File Storage  │    │
│  │ Vector DB   │  │  API       │  │  (Cross-    │  │  (Documents)   │    │
│  │             │  │ (Embed +   │  │   encoder)  │  │                │    │
│  │ - Dense     │  │  Generate) │  │             │  │                │    │
│  │ - Sparse    │  │            │  │             │  │                │    │
│  │ - Metadata  │  │            │  │             │  │                │    │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
phase2-document-intelligence/
│
├── docker-compose.yml              # Qdrant + app orchestration
├── Dockerfile                       # Application container
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment variables template
├── README.md                        # Project documentation
│
├── config/
│   ├── __init__.py
│   └── settings.py                  # Configuration management (Pydantic settings)
│
├── src/
│   ├── __init__.py
│   │
│   ├── ingestion/                   # Document ingestion pipeline
│   │   ├── __init__.py
│   │   ├── loader.py                # Multi-format document loader
│   │   ├── parser.py                # Document parsing and cleaning
│   │   ├── chunker.py               # Chunking strategies
│   │   ├── embedder.py              # Embedding generation
│   │   └── pipeline.py              # Orchestrates the full ingestion
│   │
│   ├── retrieval/                   # Retrieval pipeline
│   │   ├── __init__.py
│   │   ├── query_transform.py       # Query rewriting, multi-query
│   │   ├── search.py                # Hybrid search (dense + sparse)
│   │   ├── reranker.py              # Cross-encoder reranking
│   │   └── pipeline.py              # Orchestrates retrieval
│   │
│   ├── generation/                  # Generation pipeline
│   │   ├── __init__.py
│   │   ├── prompts.py               # Prompt templates
│   │   ├── generator.py             # LLM generation with grounding
│   │   ├── models.py                # Pydantic response models
│   │   └── pipeline.py              # Orchestrates generation
│   │
│   ├── evaluation/                  # Evaluation suite
│   │   ├── __init__.py
│   │   ├── test_set.py              # Test set management
│   │   ├── retrieval_metrics.py     # Recall, MRR, NDCG, Precision
│   │   ├── generation_metrics.py    # Faithfulness, relevancy, correctness
│   │   ├── system_metrics.py        # Latency, cost tracking
│   │   └── evaluator.py             # Main evaluation harness
│   │
│   ├── api/                         # API layer
│   │   ├── __init__.py
│   │   ├── main.py                  # FastAPI application
│   │   ├── routes.py                # API endpoints
│   │   └── schemas.py               # Request/response schemas
│   │
│   └── rag.py                       # Main RAG pipeline (composes everything)
│
├── ui/
│   └── app.py                       # Streamlit web interface
│
├── data/
│   ├── documents/                   # Source documents (PDF, DOCX, etc.)
│   ├── test_set/                    # Evaluation test questions
│   └── results/                     # Evaluation results
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── test_api.py
│
└── docs/                            # Research & architecture docs (these files)
```

---

## Component Design

### 1. Configuration Management

```python
# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "consulting_docs"
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval
    retrieval_k: int = 10
    rerank_top_k: int = 5
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    
    # Evaluation
    confidence_threshold: float = 0.5
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 2. Document Ingestion Pipeline

```python
# src/ingestion/pipeline.py
from pathlib import Path
from typing import Optional

class IngestionPipeline:
    """Orchestrates document ingestion: load → parse → chunk → embed → store."""
    
    def __init__(self, config: Settings):
        self.loader = DocumentLoader()
        self.chunker = SmartChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedder = EmbeddingService(model=config.embedding_model)
        self.store = VectorStoreService(
            url=config.qdrant_url,
            collection=config.collection_name,
            embeddings=self.embedder
        )
    
    def ingest_file(self, filepath: Path, metadata: Optional[dict] = None) -> dict:
        """Ingest a single file into the vector store."""
        # 1. Load and parse
        document = self.loader.load(filepath)
        
        # 2. Assess quality
        quality = assess_document_quality(document.text, str(filepath))
        if not quality["is_clean"]:
            logger.warning(f"Quality issues in {filepath}: {quality['issues']}")
        
        # 3. Chunk with appropriate strategy
        chunks = self.chunker.chunk(document)
        
        # 4. Enrich metadata
        for chunk in chunks:
            chunk.metadata.update(metadata or {})
            chunk.metadata["ingested_at"] = datetime.utcnow().isoformat()
        
        # 5. Embed and store
        self.store.add_documents(chunks)
        
        return {
            "file": str(filepath),
            "chunks_created": len(chunks),
            "quality": quality
        }
    
    def ingest_directory(self, directory: Path) -> list[dict]:
        """Ingest all supported files in a directory."""
        results = []
        for filepath in directory.rglob("*"):
            if filepath.suffix in self.loader.SUPPORTED_FORMATS:
                result = self.ingest_file(filepath)
                results.append(result)
        return results
```

### 3. RAG Query Pipeline

```python
# src/rag.py
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class RAGPipeline:
    """Main RAG pipeline: query → retrieve → generate → respond."""
    
    def __init__(self, config: Settings):
        self.config = config
        self.retriever = self._build_retriever()
        self.chain = self._build_chain()
    
    def _build_retriever(self):
        """Build hybrid retriever with reranking."""
        # Vector retriever
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )
        
        # BM25 retriever (sparse)
        bm25_retriever = BM25Retriever.from_documents(
            self.all_chunks, k=self.config.retrieval_k
        )
        
        # Combine
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[self.config.bm25_weight, self.config.vector_weight]
        )
        
        return ensemble
    
    def _build_chain(self):
        """Build the LCEL chain."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE)
        ])
        
        llm = ChatOpenAI(
            model=self.config.llm_model,
            temperature=self.config.llm_temperature
        )
        
        chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm.with_structured_output(RAGResponse)
        )
        
        return chain
    
    def query(self, question: str) -> RAGResponse:
        """Execute a RAG query and return structured response."""
        response = self.chain.invoke(question)
        return response
    
    def _format_docs(self, docs):
        """Format retrieved documents for the prompt."""
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}, "
            f"Page {d.metadata.get('page', '?')}]\n{d.page_content}"
            for d in docs
        )
```

### 4. API Layer

```python
# src/api/main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(
    title="Document Intelligence API",
    description="Enterprise RAG system for consulting knowledge base",
    version="0.1.0"
)

class QueryRequest(BaseModel):
    question: str
    filters: dict | None = None
    max_sources: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]
    confidence: str
    latency_ms: float
    sources_used: int

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """Query the knowledge base with a natural language question."""
    start = time.time()
    
    response = rag_pipeline.query(request.question)
    
    return QueryResponse(
        answer=response.answer,
        citations=[c.dict() for c in response.citations],
        confidence=response.confidence,
        latency_ms=(time.time() - start) * 1000,
        sources_used=len(response.citations)
    )

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Upload and ingest a document into the knowledge base."""
    # Save uploaded file
    filepath = save_upload(file)
    
    # Ingest
    result = ingestion_pipeline.ingest_file(filepath)
    
    return {"status": "success", "details": result}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}
```

### 5. Streamlit UI

```python
# ui/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Document Intelligence", page_icon="📚")
st.title("📚 Enterprise Document Intelligence")

# Sidebar for document upload
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "pptx", "html", "md"]
    )
    if uploaded_file and st.button("Ingest Document"):
        response = requests.post(
            "http://localhost:8000/ingest",
            files={"file": uploaded_file}
        )
        if response.ok:
            st.success(f"Document ingested: {response.json()['details']['chunks_created']} chunks")

# Main query interface
question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Searching and generating answer..."):
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question}
        )
    
    if response.ok:
        data = response.json()
        
        # Display answer
        st.markdown(f"### Answer")
        st.markdown(data["answer"])
        
        # Display confidence
        confidence_colors = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}
        st.markdown(f"**Confidence**: :{confidence_colors.get(data['confidence'], 'gray')}[{data['confidence']}]")
        
        # Display sources
        with st.expander(f"📖 Sources ({data['sources_used']} documents)"):
            for citation in data["citations"]:
                st.markdown(f"- **{citation['source']}** (p. {citation.get('page', '?')})")
                st.markdown(f"  > {citation.get('quote', '')}")
        
        # Display latency
        st.caption(f"Response time: {data['latency_ms']:.0f}ms")
```

---

## Docker Setup

### docker-compose.yml

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant
    volumes:
      - ./data/documents:/app/data/documents

  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    ports:
      - "8501:8501"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api

volumes:
  qdrant_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY config/ config/
COPY src/ src/

# Expose API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Data Flow Diagrams

### Ingestion Flow

```
User uploads document
         │
         ▼
    ┌─────────┐     ┌──────────┐     ┌──────────┐
    │  Detect  │ ──▶ │  Parse   │ ──▶ │  Clean   │
    │  Format  │     │  Extract │     │  Text    │
    └─────────┘     └──────────┘     └──────────┘
                                           │
                                           ▼
    ┌─────────┐     ┌──────────┐     ┌──────────┐
    │  Store   │ ◀── │  Embed   │ ◀── │  Chunk   │
    │  Qdrant  │     │  OpenAI  │     │  Smart   │
    └─────────┘     └──────────┘     └──────────┘
```

### Query Flow

```
User asks question
         │
         ▼
    ┌──────────────┐
    │  Query        │  (Optional: rewrite, multi-query)
    │  Transform    │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Embed Query  │  Same embedding model as ingestion
    └──────┬───────┘
           │
    ┌──────▼────────────────┐
    │  Hybrid Search         │  Dense (Qdrant) + Sparse (BM25)
    │  k=20                  │  Combined with RRF
    └──────┬────────────────┘
           │
    ┌──────▼───────┐
    │  Rerank       │  Cross-encoder, top 5-8
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Build Prompt │  System instructions + context + question
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  LLM Generate │  Structured output (answer + citations + confidence)
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Post-Process │  Verify citations, format response
    └──────┬───────┘
           │
           ▼
    Response to user
```

---

## Technology Stack Summary

| Layer | Technology | Why |
|-------|-----------|-----|
| **Vector DB** | Qdrant (Docker) | Fast, great filtering, hybrid search, easy setup |
| **Embeddings** | OpenAI text-embedding-3-small | Cost-effective, 8K tokens, Matryoshka support |
| **LLM** | GPT-4o (generation), GPT-4o-mini (evaluation) | Quality for generation, cost-effective for eval |
| **Framework** | LangChain (LCEL) | Standard RAG abstractions, composable chains |
| **Doc Processing** | unstructured + specialized parsers | Multi-format support, structure detection |
| **Reranking** | Cross-encoder (sentence-transformers) | Open-source, runs locally, high quality |
| **API** | FastAPI | Modern Python API, auto-docs, async support |
| **UI** | Streamlit | Quick to build, good enough for demo/MVP |
| **Containerization** | Docker + docker-compose | First DevOps experience, reproducible setup |
| **Evaluation** | RAGAS + custom metrics | Standard framework + our specific metrics |

---

## Implementation Phases

### Phase A: Foundation (Days 1-3)

**Goal**: Working end-to-end pipeline with the simplest possible implementation.

1. Set up project structure and Docker (Qdrant)
2. Implement basic document loader (PDF + Markdown)
3. Implement recursive character chunking
4. Set up embedding + Qdrant storage
5. Build naive RAG chain (embed → retrieve → generate)
6. Create 20 test questions
7. **Milestone**: Can ask a question and get an answer from a small set of documents

### Phase B: Evaluation (Days 4-5)

**Goal**: Measurement framework to guide all further improvements.

1. Build evaluation harness with retrieval + generation metrics
2. Create comprehensive test set (50+ questions)
3. Measure baseline metrics
4. Set up experiment tracking
5. **Milestone**: Can quantitatively measure any change to the pipeline

### Phase C: Quality (Days 6-8)

**Goal**: Improve retrieval and generation quality systematically.

1. Implement hybrid search (dense + BM25)
2. Add cross-encoder reranking
3. Implement structural chunking for DOCX/Markdown
4. Add metadata filtering
5. Improve generation prompt (grounding, citations, confidence)
6. Implement "I don't know" capability
7. **Milestone**: Significant metric improvements over baseline

### Phase D: Multi-Format + Polish (Days 9-11)

**Goal**: Handle all document types, add API and UI.

1. Add DOCX, PPTX, HTML loaders
2. Implement document type-specific chunking strategies
3. Build FastAPI endpoints
4. Build Streamlit UI
5. Add document upload capability
6. **Milestone**: Full multi-format ingestion with web interface

### Phase E: Containerization + Final Eval (Days 12-14)

**Goal**: Docker deployment and final evaluation.

1. Create Dockerfile and docker-compose.yml
2. Containerize the full application
3. Run comprehensive evaluation
4. Document final architecture and results
5. Create demo with real consulting-style documents
6. **Milestone**: Fully containerized, evaluated, documented system

---

## Key Dependencies

```
# requirements.txt
langchain>=0.3.0
langchain-core>=0.3.0
langchain-openai>=0.2.0
langchain-qdrant>=0.2.0
langchain-text-splitters>=0.3.0
langchain-community>=0.3.0

# Document processing
unstructured>=0.15.0
pypdf>=4.0
python-docx>=1.0
python-pptx>=0.6
beautifulsoup4>=4.12

# Vector store
qdrant-client>=1.10

# Embeddings & Reranking
sentence-transformers>=3.0

# API
fastapi>=0.110
uvicorn>=0.27

# UI
streamlit>=1.35

# Evaluation
ragas>=0.2.0
datasets>=2.18

# Utilities
pydantic>=2.7
pydantic-settings>=2.2
python-dotenv>=1.0
tiktoken>=0.7

# BM25
rank-bm25>=0.2
```

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| PDF extraction quality is poor | HIGH | MEDIUM | Multiple parser fallbacks, quality assessment |
| Embedding costs exceed budget | LOW | LOW | text-embedding-3-small is very cheap; monitor usage |
| LLM hallucination despite grounding | HIGH | MEDIUM | Strong prompts, structured output, confidence scoring |
| Docker setup complexity | MEDIUM | LOW | Start with simple docker-compose, add complexity later |
| Qdrant data loss | HIGH | LOW | Volume mounts, snapshot backups |
| LangChain version breaking changes | MEDIUM | MEDIUM | Pin versions, test before upgrading |
| Evaluation test set bias | MEDIUM | MEDIUM | Diverse question types, periodic human review |

---

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Retrieval Recall@10 | ≥ 0.85 | Evaluation suite on test set |
| MRR | ≥ 0.70 | Evaluation suite on test set |
| Faithfulness | ≥ 0.95 | RAGAS faithfulness metric |
| Answer Relevancy | ≥ 0.80 | RAGAS answer relevancy |
| "I don't know" accuracy | ≥ 0.90 | Correctly refuse unanswerable questions |
| Latency (p95) | < 5 seconds | System metrics |
| Cost per query | < $0.05 | Token tracking |
| Document formats supported | ≥ 5 | PDF, DOCX, PPTX, HTML, Markdown |
| Containerized | Yes | docker-compose up works |

---

## Resources

All research documents for this architecture:

1. [01-concept-rag-architecture.md](01-concept-rag-architecture.md) — RAG fundamentals
2. [02-concept-chunking-strategies.md](02-concept-chunking-strategies.md) — Chunking approaches
3. [03-concept-embeddings-and-vector-search.md](03-concept-embeddings-and-vector-search.md) — Embeddings and search
4. [04-concept-retrieval-strategies.md](04-concept-retrieval-strategies.md) — Advanced retrieval
5. [05-concept-generation-and-grounding.md](05-concept-generation-and-grounding.md) — Grounded generation
6. [06-concept-rag-evaluation.md](06-concept-rag-evaluation.md) — Evaluation framework
7. [07-technology-vector-databases.md](07-technology-vector-databases.md) — Vector DB comparison
8. [08-technology-document-processing.md](08-technology-document-processing.md) — Document processing
9. [09-technology-langchain-rag.md](09-technology-langchain-rag.md) — LangChain patterns
10. [10-technology-embedding-models.md](10-technology-embedding-models.md) — Embedding model comparison
