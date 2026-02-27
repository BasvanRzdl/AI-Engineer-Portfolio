---
date: 2026-02-27
type: technology
topic: "LangChain for RAG"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
decision: use LangChain as primary framework with LCEL patterns
---

# Technology Brief: LangChain for RAG

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Python framework for building LLM-powered applications |
| **For** | Composing document loaders, splitters, embeddings, retrievers, and chains into RAG pipelines |
| **Version** | langchain 0.3.x+ (with langchain-core, langchain-community, partner packages) |
| **Decision** | Use as primary framework — industry standard with excellent RAG abstractions |

## Why LangChain?

LangChain provides **pre-built abstractions** for every component of a RAG pipeline. Without it, you'd need to manually:
- Write document loading code for each format
- Implement text splitting algorithms
- Build embedding and retrieval wrappers
- Compose prompt templates and LLM calls
- Handle streaming, error handling, and retries

LangChain does all this and lets you swap components (different LLMs, vector stores, splitters) without rewriting your pipeline.

---

## Package Architecture (Important!)

LangChain has been modularized. Understanding the package structure prevents confusion:

```
langchain-core          ← Base abstractions (LCEL, Runnables, prompts)
langchain               ← Main chains, agents, high-level APIs
langchain-community     ← Community integrations (various vector stores, LLMs)
langchain-openai        ← OpenAI-specific (ChatOpenAI, OpenAIEmbeddings)
langchain-qdrant        ← Qdrant integration (if using Qdrant)
langchain-text-splitters ← Text splitting utilities
```

```bash
# Install what we need
pip install langchain langchain-core langchain-openai langchain-qdrant langchain-text-splitters
```

---

## Core Concepts

### 1. Documents

The fundamental data unit in LangChain:

```python
from langchain_core.documents import Document

doc = Document(
    page_content="Our M&A methodology consists of 4 phases...",
    metadata={
        "source": "methodology-guide.pdf",
        "page": 12,
        "section": "Chapter 3",
        "document_type": "methodology"
    }
)
```

Everything in LangChain works with `Document` objects — loaders produce them, splitters split them, retrievers return them.

### 2. Document Loaders

Load files into `Document` objects:

```python
# PDF
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("report.pdf")
docs = loader.load()  # Returns list[Document], one per page

# DOCX
from langchain_community.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("proposal.docx")
docs = loader.load()

# Directory of mixed files
from langchain_community.document_loaders import DirectoryLoader
loader = DirectoryLoader(
    "documents/",
    glob="**/*.*",
    show_progress=True
)
docs = loader.load()

# Unstructured (multi-format)
from langchain_community.document_loaders import UnstructuredFileLoader
loader = UnstructuredFileLoader("any_document.pdf", mode="elements")
docs = loader.load()
```

### 3. Text Splitters

Split documents into chunks:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)

chunks = splitter.split_documents(docs)  # Preserves metadata!
print(f"Split {len(docs)} documents into {len(chunks)} chunks")
```

Specialized splitters:
```python
# Markdown
from langchain_text_splitters import MarkdownHeaderTextSplitter

# HTML
from langchain_text_splitters import HTMLHeaderTextSplitter

# Code
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=100
)

# Token-based (more accurate for LLM context budgeting)
from langchain_text_splitters import TokenTextSplitter
splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
```

### 4. Embeddings

Convert text to vectors:

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # dimensions=1024  # Optional: reduce dimensions for cost savings
)

# Single text
vector = embeddings.embed_query("What is our M&A methodology?")
print(f"Vector dimensions: {len(vector)}")  # 1536

# Batch documents
doc_vectors = embeddings.embed_documents([
    "Our M&A methodology has 4 phases...",
    "Phase 1 involves due diligence...",
])
```

### 5. Vector Stores

Store and search embeddings:

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Create vector store from documents
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="consulting_docs",
)

# Or connect to existing collection
client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(
    client=client,
    collection_name="consulting_docs",
    embedding=embeddings,
)

# Search
results = vector_store.similarity_search(
    "What is our M&A methodology?",
    k=5
)
for doc in results:
    print(f"Source: {doc.metadata['source']} | {doc.page_content[:100]}...")

# Search with scores
results_with_scores = vector_store.similarity_search_with_score(
    "What is our M&A methodology?",
    k=5
)
for doc, score in results_with_scores:
    print(f"Score: {score:.3f} | {doc.page_content[:100]}...")

# MMR search (diversity)
results = vector_store.max_marginal_relevance_search(
    "What is our M&A methodology?",
    k=5,
    fetch_k=20
)
```

### 6. Retrievers

The abstraction for getting documents — wraps vector stores and more:

```python
# Basic retriever from vector store
retriever = vector_store.as_retriever(
    search_type="similarity",  # or "mmr" or "similarity_score_threshold"
    search_kwargs={"k": 5}
)

# Retrieve documents
docs = retriever.invoke("What is our M&A methodology?")

# With score threshold
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 10}
)

# With metadata filtering (Qdrant-specific)
from qdrant_client.models import Filter, FieldCondition, MatchValue

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": Filter(
            must=[FieldCondition(key="document_type", match=MatchValue(value="methodology"))]
        )
    }
)
```

Advanced retrievers:
```python
# Multi-query retriever (generates query variations)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# Contextual compression (extract relevant parts only)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# Ensemble retriever (combine multiple retrievers)
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

bm25_retriever = BM25Retriever.from_documents(chunks, k=10)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 30% BM25, 70% semantic
)
```

### 7. LCEL (LangChain Expression Language)

The modern way to compose LangChain components — pipes and chains using the `|` operator:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Define the prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer (cite your sources):
""")

# Define the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Helper to format retrieved documents
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

# Compose the RAG chain with LCEL
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Use it
answer = rag_chain.invoke("What is our M&A methodology?")
print(answer)
```

**How LCEL works:**
- `|` connects components in a pipeline (output of left → input of right)
- `RunnablePassthrough()` passes input through unchanged
- `{}` creates a parallel execution (dict of runnables)
- Every component is a `Runnable` with `.invoke()`, `.stream()`, `.batch()`

### 8. Streaming

Stream the generated answer token by token:

```python
# Stream the RAG chain
for chunk in rag_chain.stream("What is our M&A methodology?"):
    print(chunk, end="", flush=True)
```

### 9. Structured Output

Combine with Pydantic for structured responses:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Citation(BaseModel):
    source: str = Field(description="The document source name")
    page: Optional[int] = Field(description="Page number if available")
    quote: str = Field(description="The relevant quote from the source")

class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question")
    citations: List[Citation] = Field(description="Sources cited in the answer")
    confidence: str = Field(description="HIGH, MEDIUM, or LOW")
    
# Use with_structured_output
structured_llm = llm.with_structured_output(RAGResponse)

rag_chain_structured = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | structured_llm
)

response: RAGResponse = rag_chain_structured.invoke("What is our M&A methodology?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
for citation in response.citations:
    print(f"  - {citation.source} p.{citation.page}: {citation.quote}")
```

---

## Complete RAG Pipeline Example

Putting it all together:

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# ===== INGESTION =====

# 1. Load documents
loader = DirectoryLoader("documents/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Create embeddings and store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="consulting_docs",
)

# ===== RETRIEVAL =====

# 4. Set up hybrid retrieval
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})
bm25_retriever = BM25Retriever.from_documents(chunks, k=10)

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

# ===== GENERATION =====

# 5. Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for a consulting firm's knowledge base.
Answer questions based ONLY on the provided context.
For every claim, cite the source using [Source: filename, Page X].
If you can't answer from the context, say "I don't have enough information."
"""),
    ("human", """Context:
{context}

Question: {question}

Answer with citations:""")
])

# 6. Define LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 7. Format docs helper
def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )

# 8. Compose the chain
rag_chain = (
    {
        "context": ensemble_retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ===== USE IT =====
answer = rag_chain.invoke("What is our approach to digital transformation?")
print(answer)
```

---

## Key Patterns to Follow

### Pattern 1: Configuration-Driven Pipeline

```python
from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Embedding
    embedding_model: str = "text-embedding-3-small"
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Retrieval
    retrieval_k: int = 10
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    
    # Generation
    llm_model: str = "gpt-4o"
    temperature: float = 0.0
    
    # Vector store
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "consulting_docs"

config = RAGConfig()
# Now use config.xxx everywhere instead of hardcoding
```

### Pattern 2: Retrieval with Metadata Logging

```python
from langchain_core.runnables import RunnableLambda

def retrieve_and_log(query: str) -> list:
    """Retrieve documents and log the retrieval for debugging."""
    docs = ensemble_retriever.invoke(query)
    
    # Log retrieval for debugging/evaluation
    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        print(f"  {i+1}. {source}: {doc.page_content[:80]}...")
    
    return docs

# Use in chain
rag_chain = (
    {
        "context": RunnableLambda(retrieve_and_log) | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)
```

### Pattern 3: Error Handling

```python
from langchain_core.runnables import RunnableConfig

def safe_rag_query(query: str) -> dict:
    """RAG query with error handling."""
    try:
        # Retrieve
        docs = ensemble_retriever.invoke(query)
        
        if not docs:
            return {
                "answer": "I couldn't find any relevant documents for your query.",
                "sources": [],
                "error": None
            }
        
        # Generate
        answer = rag_chain.invoke(query)
        
        return {
            "answer": answer,
            "sources": [d.metadata.get("source") for d in docs],
            "error": None
        }
    
    except Exception as e:
        return {
            "answer": "An error occurred while processing your query.",
            "sources": [],
            "error": str(e)
        }
```

---

## LangChain Gotchas

| Gotcha | Description | Solution |
|--------|-------------|----------|
| **Import confusion** | Packages moved between `langchain`, `langchain-community`, and partner packages | Check docs for current import path |
| **Version breaking changes** | LangChain evolves fast | Pin versions in requirements.txt |
| **Deprecated chains** | Old `RetrievalQA`, `ConversationalRetrievalChain` are deprecated | Use LCEL chains instead |
| **Hidden API calls** | Some methods make API calls you don't expect | Check token usage with callbacks |
| **Memory usage with BM25** | `BM25Retriever` loads all docs in memory | For large collections, use a proper search engine |

---

## Best Practices

- ✅ **Use LCEL** (not legacy chains) — it's the current and future pattern
- ✅ **Pin your LangChain version** — breaking changes are common
- ✅ **Use partner packages** (langchain-openai, langchain-qdrant) — better maintained than community
- ✅ **Add callbacks for monitoring** — track token usage, latency, errors
- ✅ **Understand what's happening underneath** — LangChain is convenient but don't use it as a black box
- ❌ **Don't use deprecated chains** — `RetrievalQA` is replaced by LCEL patterns
- ❌ **Don't over-abstract** — if a LangChain abstraction doesn't fit, write custom code
- ❌ **Don't ignore token counting** — use `tiktoken` or callbacks to track costs

---

## Application to Our Project

### LangChain Components We'll Use

| Component | LangChain Class | Purpose |
|-----------|----------------|---------|
| Document loading | `DirectoryLoader`, `PyPDFLoader`, `UnstructuredFileLoader` | Multi-format ingestion |
| Text splitting | `RecursiveCharacterTextSplitter`, `MarkdownHeaderTextSplitter` | Intelligent chunking |
| Embeddings | `OpenAIEmbeddings` | Vector generation |
| Vector store | `QdrantVectorStore` | Storage and search |
| Retrievers | `EnsembleRetriever`, `MultiQueryRetriever` | Hybrid retrieval |
| Prompts | `ChatPromptTemplate` | Grounded generation prompts |
| LLM | `ChatOpenAI` | Text generation |
| Output parsing | `StrOutputParser`, `with_structured_output` | Response formatting |

---

## Resources for Deeper Learning

- [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) — Official step-by-step guide
- [LangChain LCEL docs](https://python.langchain.com/docs/concepts/lcel/) — Expression language guide
- [LangChain Qdrant integration](https://python.langchain.com/docs/integrations/vectorstores/qdrant/) — Our vector store
- [LangChain retrievers how-to](https://python.langchain.com/docs/how_to/#retrievers) — All retriever patterns
- [LangSmith](https://smith.langchain.com/) — Tracing and debugging for LangChain apps

---

## Questions Remaining

- [ ] How to add custom reranking to a LangChain LCEL chain?
- [ ] Best approach for hybrid search: EnsembleRetriever vs Qdrant's native sparse vectors?
- [ ] How to track per-query costs with LangChain callbacks?
- [ ] Should we use LangSmith for tracing in development?
