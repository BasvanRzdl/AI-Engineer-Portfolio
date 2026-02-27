---
date: 2026-02-27
type: concept
topic: "Chunking Strategies for RAG"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
---

# Learning: Chunking Strategies for RAG

## In My Own Words

Chunking is the process of splitting documents into smaller pieces that can be independently embedded and retrieved. It's arguably the **single most impactful design decision** in a RAG pipeline — more important than which embedding model or LLM you choose. Bad chunking means the right information either can't be found (too fragmented) or drowns in noise (too large).

The art of chunking is finding the "Goldilocks zone" for each document type: pieces large enough to be self-contained and meaningful, but small enough to be specific and retrievable.

## Why This Matters

- **Embedding models have token limits** — you can't embed a 50-page document as one vector
- **Retrieval precision depends on chunk granularity** — a 10-page chunk matching a query is mostly noise
- **LLM context windows are limited and costly** — smaller, precise chunks = better answers at lower cost
- **Different documents have different natural units** — a legal contract chunks differently than a research report
- **Chunk boundaries determine what can be found** — if an answer spans two chunks, it might never be retrieved

---

## Core Principles

### 1. Semantic Coherence > Character Count

A chunk should represent a **complete thought or concept**. Cutting mid-paragraph or mid-sentence destroys meaning:

```
❌ BAD CHUNK (cuts mid-thought):
"The company's revenue grew by 15% in Q3, driven primarily by
expansion into the European market. Key factors included the"

✅ GOOD CHUNK (complete thought):
"The company's revenue grew by 15% in Q3, driven primarily by
expansion into the European market. Key factors included the new
partnership with Deutsche Telekom and the Frankfurt office opening."
```

### 2. Context Preservation

Each chunk should contain enough context to be understood independently:

```
❌ BAD: "It increased by 15% compared to the previous year."
(What is "it"? No context.)

✅ GOOD: "Capgemini's cloud services revenue increased by 15%
compared to the previous year, reaching €2.1B in FY2025."
```

### 3. Metadata Is Part of the Chunk

A chunk without metadata is like a quote without attribution — useful but limited:

```python
# Good chunk structure
{
    "text": "The merger integration methodology involves...",
    "metadata": {
        "source": "methodology-guide-v3.pdf",
        "page": 12,
        "section": "Chapter 3: Post-Merger Integration",
        "document_type": "methodology",
        "date": "2025-06",
        "author": "Strategy Team"
    }
}
```

---

## Chunking Strategies Explained

### Strategy 1: Fixed-Size Chunking

**How it works**: Split text into chunks of N characters/tokens with optional overlap.

```python
from langchain.text_splitter import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=1000,       # Characters per chunk
    chunk_overlap=200,     # Overlap between consecutive chunks
    separator="\n\n"       # Prefer splitting at paragraph boundaries
)
chunks = splitter.split_text(document_text)
```

**When to use**: Quick prototyping, homogeneous text (blog posts, articles), baseline to compare against.

| Pros | Cons |
|------|------|
| Simple to implement | Ignores document structure |
| Predictable chunk sizes | Cuts mid-sentence/thought |
| Easy to reason about | No semantic awareness |
| Good baseline | Poor for structured documents |

### Strategy 2: Recursive Character Splitting

**How it works**: Tries a hierarchy of separators, falling back to smaller ones if chunks are too big.

Default separator hierarchy: `["\n\n", "\n", " ", ""]`

This means: first try to split on double newlines (paragraphs), if chunks are still too big split on single newlines, then spaces, then characters.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document_text)
```

**When to use**: General-purpose default, most text-heavy documents.

| Pros | Cons |
|------|------|
| Respects paragraph boundaries | Still size-based at core |
| Better than pure fixed-size | Doesn't understand document semantics |
| Good default choice | May still split related content |
| Configurable separators | Overlap is positional, not semantic |

### Strategy 3: Semantic Chunking

**How it works**: Uses embeddings to find natural breakpoints. Computes similarity between consecutive sentences; when similarity drops significantly, that's a chunk boundary.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

splitter = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=75
)
chunks = splitter.create_documents([document_text])
```

**How the algorithm works**:
1. Split document into sentences
2. Embed each sentence
3. Compute cosine similarity between consecutive sentence embeddings
4. Where similarity drops below a threshold → chunk boundary
5. Group sentences between boundaries into chunks

**When to use**: Narrative text, research papers, reports where topics flow into each other.

| Pros | Cons |
|------|------|
| Semantically coherent chunks | Requires embedding API calls (cost) |
| Adaptive chunk sizes | Slower than rule-based methods |
| Finds natural topic boundaries | Variable chunk sizes (harder to predict) |
| Great for narrative text | Doesn't use document structure |

### Strategy 4: Document-Aware / Structural Chunking

**How it works**: Uses the document's own structure (headings, sections, tables, lists) as chunking boundaries.

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
chunks = splitter.split_text(markdown_text)
# Each chunk includes its header hierarchy as metadata
```

For HTML:
```python
from langchain.text_splitter import HTMLHeaderTextSplitter

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(html_text)
```

**When to use**: Well-structured documents (manuals, reports with headers, HTML pages, Markdown).

| Pros | Cons |
|------|------|
| Respects author's structure | Requires structured input |
| Natural metadata from headers | Section sizes vary wildly |
| Each chunk has clear context | Not all documents have structure |
| Best for enterprise documents | Large sections still need sub-splitting |

### Strategy 5: Parent-Child (Hierarchical) Chunking

**How it works**: Create two layers of chunks — large "parent" chunks for context, small "child" chunks for precise retrieval. Retrieve child chunks, but pass the parent chunk to the LLM.

```
Document
├── Parent Chunk 1 (e.g., full section, ~2000 tokens)
│   ├── Child Chunk 1a (e.g., paragraph, ~200 tokens)
│   ├── Child Chunk 1b
│   └── Child Chunk 1c
├── Parent Chunk 2
│   ├── Child Chunk 2a
│   └── Child Chunk 2b
```

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Parent splitter: larger chunks
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)

# Child splitter: smaller chunks
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=0)

parent_chunks = parent_splitter.split_text(document_text)
for parent in parent_chunks:
    children = child_splitter.split_text(parent.page_content)
    for child in children:
        child.metadata["parent_id"] = parent.id  # Link child → parent
```

**Retrieval flow**: Search against child embeddings → get matching children → retrieve their parent chunks → send parents to LLM.

**When to use**: When you need precision in retrieval but context in generation. Best for detailed documents where a paragraph matches the query but the full section is needed for a good answer.

| Pros | Cons |
|------|------|
| Precise retrieval + rich context | More complex indexing |
| Solves the chunk size dilemma | Requires parent-child linking |
| Better answers from more context | Doubles storage requirements |
| Widely used in production RAG | More moving parts |

---

## Chunk Size Trade-offs

This is the most common tuning parameter. Here's how to think about it:

| Chunk Size | Retrieval Impact | Generation Impact | Cost Impact |
|------------|-----------------|-------------------|-------------|
| **Small** (100-300 tokens) | High precision, lower recall | May lack context for good answers | Cheaper retrieval, more chunks needed |
| **Medium** (300-800 tokens) | Balanced precision/recall | Usually sufficient context | Moderate cost |
| **Large** (800-2000 tokens) | Higher recall, lower precision | More context but more noise | More expensive per chunk |
| **Very large** (2000+ tokens) | Retrieves whole sections | Often too much noise | Expensive, fewer fit in context window |

**Rules of thumb:**
- For **factual Q&A**: smaller chunks (200-500 tokens) → precise retrieval
- For **summarization**: larger chunks (500-1500 tokens) → more context
- For **analysis/comparison**: medium chunks (400-800 tokens) + parent retrieval

### The Overlap Question

Overlap prevents information loss at chunk boundaries:

```
Chunk 1: [==========|overlap]
Chunk 2:           [overlap|==========|overlap]
Chunk 3:                    [overlap|==========]
```

**Typical overlap**: 10-20% of chunk size (e.g., 200 overlap for 1000-sized chunks)

- **Too little overlap**: Information at boundaries is lost
- **Too much overlap**: Duplicate content, increased storage and cost, retrieval noise
- **Zero overlap with parent-child**: Often viable if using hierarchical chunking

---

## Chunking Per Document Type

For our consulting firm project, different document types need different strategies:

| Document Type | Recommended Strategy | Chunk Size | Notes |
|--------------|---------------------|------------|-------|
| **PDF Reports** | Structural + recursive fallback | 500-1000 tokens | Use section headers; fall back to recursive for unstructured sections |
| **DOCX Proposals** | Structural (heading-based) | 500-800 tokens | Split by heading hierarchy; preserve proposal structure |
| **PPTX Presentations** | Slide-based | 1 slide per chunk | Each slide is a natural unit; include slide title as context |
| **HTML Pages** | HTML header splitting | 400-800 tokens | Use HTML structure; strip navigation/boilerplate |
| **Markdown Docs** | Header-based splitting | 400-1000 tokens | Natural structure; include header hierarchy in metadata |
| **Meeting Notes** | Semantic chunking | 300-600 tokens | Topic-based splitting; often unstructured |
| **Legal/Contracts** | Clause-based (structural) | Per clause | Each clause is a retrievable unit; maintain clause numbering |

---

## Best Practices

- ✅ **Always include metadata with chunks** — source, page, section, date, document type
- ✅ **Prepend section headers to chunk text** — gives each chunk context even when retrieved alone
- ✅ **Use different strategies for different document types** — one size does NOT fit all
- ✅ **Test chunk quality manually** — read 20 random chunks; do they make sense standalone?
- ✅ **Measure retrieval performance with different chunk sizes** — let the metrics decide
- ✅ **Include document title/section in every chunk** — "From 'Q3 Revenue Report', Section: European Market..."
- ❌ **Don't use fixed-size chunking for structured documents** — you're throwing away free structure
- ❌ **Don't chunk without overlap unless using parent-child** — boundary information loss is real
- ❌ **Don't assume one chunk size works for all queries** — Q&A and summarization have different needs
- ❌ **Don't ignore tables** — tables need special handling (keep together or convert to text)

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Tables split across chunks | Table rows treated as regular text | Detect tables and keep them whole, or convert to structured text |
| Headers separated from content | Fixed-size splitting cuts after header | Use structural splitting; always prepend header to chunk |
| Code blocks broken apart | Language-agnostic splitting | Detect code blocks and keep intact |
| Bullet lists split mid-item | Character-based boundary | Use list-aware splitting |
| Cross-reference chunks useless | "As mentioned in Section 3.2" without context | Resolve cross-references or include referenced content |
| Chunk too generic | Large chunks average over multiple topics | Smaller chunks or semantic splitting |

---

## Application to Our Project

### How We'll Use This

For the consulting firm's document collection, we need a **multi-strategy chunking pipeline**:

```python
def chunk_document(document, doc_type):
    """Route to appropriate chunking strategy based on document type."""
    if doc_type == "markdown":
        return markdown_header_chunking(document)
    elif doc_type == "html":
        return html_structural_chunking(document)
    elif doc_type in ("pdf", "docx") and has_headers(document):
        return structural_chunking_with_recursive_fallback(document)
    elif doc_type == "pptx":
        return slide_based_chunking(document)
    else:
        return recursive_character_chunking(document)
```

### Decisions to Make

- [ ] Default chunk size per document type (start with 500, measure, adjust)
- [ ] Overlap strategy (200 token overlap vs parent-child approach)
- [ ] How to handle tables (keep whole? convert to text? separate table index?)
- [ ] Whether to implement parent-child chunking from the start or add later
- [ ] How to detect and handle document structure in PDFs (which often lose structure)

### Implementation Notes

- Start with `RecursiveCharacterTextSplitter` as baseline
- Add structural splitting for Markdown and HTML first (easiest)
- Invest in PDF structure detection — most enterprise docs are PDFs
- Always prepend `{document_title} > {section_header}:` to each chunk
- Build a chunk quality viewer early (show chunks, their metadata, and boundaries)

---

## Resources for Deeper Learning

- [LangChain Text Splitters docs](https://python.langchain.com/docs/how_to/#text-splitters) — All built-in splitters with examples
- [Pinecone Chunking guide](https://www.pinecone.io/learn/chunking-strategies/) — Visual guide to strategies
- [Unstructured.io](https://unstructured.io/) — Library for intelligent document parsing
- [Greg Kamradt's chunking experiments](https://www.youtube.com/watch?v=8OJC21T2SL4) — Excellent video comparing strategies with real data
- [LlamaIndex on Node Parsing](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/) — Alternative take on chunking

---

## Questions Remaining

- [ ] What's the best way to handle tables in PDFs (especially complex multi-column tables)?
- [ ] How does the `unstructured` library compare to custom parsing for our document types?
- [ ] Should we store multiple chunk sizes of the same document for different query types?
- [ ] What's the practical performance difference between semantic and recursive chunking on enterprise docs?
