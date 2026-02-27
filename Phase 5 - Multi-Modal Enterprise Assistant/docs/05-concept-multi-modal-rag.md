---
date: 2026-02-27
type: concept
topic: "Multi-Modal RAG Patterns"
project: "Phase 5 - Multi-Modal Enterprise Assistant"
status: complete
---

# Learning: Multi-Modal RAG Patterns

## In My Own Words

Multi-Modal RAG extends the classic Retrieval-Augmented Generation pattern beyond text. Instead of only retrieving text documents to ground LLM responses, **multi-modal RAG retrieves and reasons over images, tables, charts, audio transcripts, and other media types** alongside text. The goal is the same — reduce hallucinations and ground responses in actual data — but the data now spans multiple modalities.

The fundamental challenge: how do you store, search, and retrieve non-text content in a way that a language model can use for generation?

## Why This Matters

Enterprise knowledge lives in many formats:
- **PDFs** with text, tables, charts, and images
- **Slide decks** that are primarily visual
- **Technical diagrams** (architecture, flowcharts, ER diagrams)
- **Photos** (product images, damage assessments, site inspections)
- **Audio/video** recordings of meetings and presentations

A text-only RAG system misses most of this information. Multi-modal RAG unlocks the full breadth of enterprise knowledge for AI-assisted retrieval and reasoning.

## Core Principles

### 1. The Multi-Modal RAG Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        INDEXING PHASE                            │
│                                                                   │
│  Documents ──▶ [Multi-Modal Parser] ──▶ [Chunking per modality] │
│                    │                         │                    │
│                    ├── Text chunks           ├── Text embeddings  │
│                    ├── Image descriptions     ├── Image embeddings │
│                    ├── Table extractions      ├── Table embeddings │
│                    └── Audio transcripts      └── Audio embeddings │
│                                                    │              │
│                                              [Vector Store]       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL PHASE                            │
│                                                                   │
│  Query ──▶ [Embed query] ──▶ [Search vector store] ──▶ Results  │
│                                    │                              │
│                                    ├── Text chunks                │
│                                    ├── Image descriptions         │
│                                    ├── Original images            │
│                                    └── Table data                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      GENERATION PHASE                            │
│                                                                   │
│  [Retrieved context + Query] ──▶ [VLM / LLM] ──▶ Grounded Answer│
│       (text + images)                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Strategies for Handling Non-Text Content

There are three main approaches to making non-text content retrievable:

#### Strategy A: Convert Everything to Text (Text-Centric)

```
Image  ──▶ [VLM: "Describe this image"] ──▶ Text Description ──▶ Embed as text
Chart  ──▶ [VLM: "Extract data from chart"] ──▶ Text/JSON    ──▶ Embed as text
Audio  ──▶ [Whisper: Transcribe] ──▶ Transcript               ──▶ Embed as text
Table  ──▶ [Parse to markdown/JSON] ──▶ Structured text       ──▶ Embed as text
```

| Pros | Cons |
|------|------|
| Uses standard text embeddings and search | Information loss during conversion |
| Works with any text-based vector store | VLM description may miss details |
| Simpler pipeline | Expensive indexing (VLM calls per image) |
| Proven and reliable | Can't show original images to the LLM |

#### Strategy B: Use Multi-Modal Embeddings (Embedding-Centric)

```
Image  ──▶ [CLIP/Multi-modal encoder] ──▶ Image embedding  ──▶ Store in vector DB
Text   ──▶ [CLIP/Multi-modal encoder] ──▶ Text embedding   ──▶ Store in vector DB
Query  ──▶ [CLIP/Multi-modal encoder] ──▶ Query embedding  ──▶ Search both
```

Multi-modal embedding models (like CLIP, OpenCLIP, or Azure AI Vision multimodal embeddings) place images and text in the **same embedding space**, enabling cross-modal search.

| Pros | Cons |
|------|------|
| True cross-modal retrieval | Embedding quality varies |
| Can retrieve images directly for VLM | Requires multi-modal embedding model |
| No information loss from description | More complex pipeline |
| Fast retrieval | Limited to modalities the embedding model supports |

#### Strategy C: Hybrid (Text Description + Original Storage)

```
Image ──▶ [VLM: Describe] ──▶ Description ──▶ Embed (for search)
      └──▶ Store original image (for generation)

At retrieval:
Query ──▶ Search text descriptions ──▶ Retrieve matching images ──▶ Send to VLM with query
```

This is the **recommended approach** for most enterprise use cases:
1. Generate text descriptions for search/retrieval
2. Store original media for generation
3. Retrieved originals are passed to VLM for accurate reasoning

| Pros | Cons |
|------|------|
| Best of both worlds | More storage required |
| Accurate retrieval via text | More complex indexing pipeline |
| Rich generation with originals | Indexing cost (VLM for descriptions) |
| Can fall back to text-only | Need to manage media storage |

### 3. Chunking Strategies for Multi-Modal Documents

Multi-modal documents (like PDFs with charts) need special chunking:

```
PDF Page
├── Text Region 1 ──▶ Text chunk (standard chunking)
├── Image/Chart ──▶ Image chunk (extracted + described)
├── Table ──▶ Table chunk (parsed to structured format)
└── Text Region 2 ──▶ Text chunk (standard chunking)
```

**Key considerations**:
- **Keep context together**: A chart and its caption should be in the same chunk
- **Parent-child relationships**: Link image descriptions back to their source page/document
- **Metadata preservation**: Store page number, position, document source with every chunk
- **Deduplication**: The same chart may appear in multiple documents

### 4. Multi-Modal Index Design

```
Vector Store Schema:
┌──────────────────────────────────────────┐
│ id: string                                │
│ content_type: "text" | "image" | "table" │
│ text_content: string (description/text)   │
│ embedding: vector[1536]                   │
│ media_url: string (link to original)      │
│ metadata: {                               │
│   source_document: string                 │
│   page_number: int                        │
│   position: {x, y, width, height}        │
│   parent_chunk_id: string                 │
│   modality: string                        │
│ }                                         │
└──────────────────────────────────────────┘
```

## Approaches & Trade-offs

| Approach | Retrieval Quality | Generation Quality | Complexity | Cost |
|----------|------------------|-------------------|------------|------|
| **Text-only RAG** | Good for text | Good for text | Low | Low |
| **Text-centric multi-modal** | Good | Medium (loses visual detail) | Medium | Medium |
| **Multi-modal embeddings** | Great for cross-modal | Great (with VLM) | High | Medium |
| **Hybrid (recommended)** | Great | Great | High | Medium-High |

## Enterprise Document Processing Pipeline

For a real enterprise system, the document ingestion pipeline looks like:

```
Document Input (PDF, DOCX, PPTX, images, audio)
        │
        ▼
┌─────────────────────────────────┐
│  1. Document Parsing             │
│  ├── PDF → text + images + tables│
│  ├── DOCX → text + images       │
│  ├── PPTX → slides as images    │
│  ├── Images → stored as-is      │
│  └── Audio → transcribed text    │
├─────────────────────────────────┤
│  2. Content Extraction           │
│  ├── OCR for scanned documents   │
│  ├── Table parsing (structure)   │
│  ├── Chart description (VLM)     │
│  └── Image description (VLM)    │
├─────────────────────────────────┤
│  3. Chunking                     │
│  ├── Text: semantic chunking     │
│  ├── Images: description chunks  │
│  ├── Tables: row/section chunks  │
│  └── Audio: segment chunks       │
├─────────────────────────────────┤
│  4. Embedding                    │
│  ├── Text embeddings (ada-002)   │
│  └── Optional: CLIP embeddings   │
├─────────────────────────────────┤
│  5. Storage                      │
│  ├── Vectors → Vector DB         │
│  ├── Original media → Blob store │
│  └── Metadata → Document DB      │
└─────────────────────────────────┘
```

## Best Practices

- ✅ **Use the hybrid approach** — text descriptions for search, originals for generation
- ✅ **Preserve parent-child relationships** — link images to their source pages and documents
- ✅ **Store rich metadata** — page numbers, positions, document source, content type
- ✅ **Generate multiple descriptions per image** — a chart might need a data-focused and a trend-focused description
- ✅ **Test retrieval before generation** — poor retrieval = poor generation regardless of LLM quality
- ✅ **Handle tables as structured data** — markdown or JSON tables preserve structure better than prose descriptions
- ❌ **Don't embed images as raw pixels** — use proper visual embeddings (CLIP) or text descriptions
- ❌ **Don't ignore document structure** — headings, sections, and layout carry meaning
- ❌ **Don't mix embedding models** — queries and documents must use the same embedding model

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| **Charts retrieved but not understood** | Text description doesn't capture data | Use VLM to extract data as JSON, not just describe visually |
| **Wrong image retrieved** | Description doesn't differentiate similar images | Include contextual info (document title, section, caption) in description |
| **Context too large for VLM** | Too many retrieved images | Limit to top-K most relevant; summarize if needed |
| **Duplicate content** | Same image in multiple documents | Deduplication by content hash |
| **Stale descriptions** | Source document updated, descriptions not | Implement update pipeline with content hashing |

## Application to My Project

### How I'll Use This

The Multi-Modal Enterprise Assistant should support multi-modal RAG for:
1. **Visual document Q&A**: "What was the revenue in the Q3 report chart?"
2. **Cross-modal search**: Find images related to text queries and vice versa
3. **Meeting context**: Retrieve relevant slides when asked about meeting topics

### Implementation Priority

1. Start with **text-centric approach** (Strategy A) — simplest to implement
2. Add **hybrid approach** (Strategy C) for documents with important visuals
3. Consider **multi-modal embeddings** (Strategy B) if cross-modal search is needed

### Decisions to Make

- [ ] Which vector database to use (reuse from Phase 2 or evaluate alternatives?)
- [ ] Image description strategy: per-image VLM call vs batch processing
- [ ] How to handle PDF parsing (Azure Document Intelligence vs open-source)
- [ ] Media storage approach (Azure Blob Storage vs local filesystem)
- [ ] Embedding model: text-only vs multi-modal

## Resources for Deeper Learning

- [Azure AI Search + Vector Search](https://learn.microsoft.com/en-us/azure/search/vector-search-overview) — Built-in vector search with Azure
- [CLIP Paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) — Multi-modal embeddings
- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449) — State-of-the-art multi-modal document retrieval
- [Unstructured.io](https://unstructured.io/) — Document parsing library for multi-modal content

## Questions Remaining

- [ ] How does multi-modal RAG accuracy compare to text-only RAG on enterprise documents?
- [ ] What's the optimal number of images to include in a VLM generation context?
- [ ] How to handle version control of multi-modal indexes?
- [ ] Cost model for multi-modal indexing at enterprise scale?
