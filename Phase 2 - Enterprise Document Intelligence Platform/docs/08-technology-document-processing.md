---
date: 2026-02-27
type: technology
topic: "Document Processing and Ingestion"
project: "Phase 2 - Enterprise Document Intelligence Platform"
status: complete
decision: use unstructured as primary parser, with specialized libraries as fallback
---

# Technology Brief: Document Processing and Ingestion

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Libraries and techniques for extracting text, structure, and metadata from enterprise documents |
| **For** | Converting PDF, DOCX, PPTX, HTML, Markdown into clean, chunked text for RAG |
| **Key Libraries** | unstructured, pypdf/pymupdf, python-docx, python-pptx, beautifulsoup4 |
| **Decision** | unstructured as primary; specialized libraries as fallbacks |

## Why Document Processing Is Hard

Enterprise documents aren't clean text files. They're:

- **PDFs**: Might be text-based, scanned images, or a mix. Tables, columns, headers/footers, page numbers
- **DOCX**: Styled with headings, tables, images, track changes, comments, embedded objects
- **PPTX**: Slides with text boxes, images, speaker notes, animations, master layouts
- **HTML**: Navigation, ads, scripts, CSS — the actual content is a fraction of the file
- **Scanned documents**: No text at all — requires OCR

The quality of your document processing **directly determines** the quality of your chunks, which determines the quality of everything downstream.

---

## Document Processing Pipeline

```
Raw File (PDF, DOCX, PPTX, HTML, MD)
     │
     ▼
┌──────────────┐
│  Detection    │  Determine file type, check if scanned
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Extraction   │  Extract raw text + structural elements
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Cleaning     │  Remove noise, fix encoding, normalize whitespace
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Structure    │  Identify sections, headers, tables, lists
│  Detection    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Metadata     │  Extract title, author, date, document type
│  Extraction   │
└──────┬───────┘
       │
       ▼
Clean text with structure + metadata → Ready for chunking
```

---

## Libraries by Document Type

### PDF Processing

PDFs are the most common and most challenging format.

#### PyPDF (pypdf)

Basic PDF text extraction:

```python
from pypdf import PdfReader

reader = PdfReader("document.pdf")

for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    print(f"--- Page {page_num + 1} ---")
    print(text)

# Extract metadata
metadata = reader.metadata
print(f"Title: {metadata.title}")
print(f"Author: {metadata.author}")
print(f"Pages: {len(reader.pages)}")
```

| Pros | Cons |
|------|------|
| ✅ Simple, lightweight | ❌ Poor table extraction |
| ✅ Fast for text-based PDFs | ❌ No structure detection |
| ✅ Good metadata extraction | ❌ Can't handle scanned PDFs |
| ✅ Zero external dependencies | ❌ Multi-column text can jumble |

#### PyMuPDF (fitz)

More powerful PDF processing:

```python
import fitz  # PyMuPDF

doc = fitz.open("document.pdf")

for page_num, page in enumerate(doc):
    # Text with position information
    blocks = page.get_text("blocks")  # Returns positioned text blocks
    
    # Or structured extraction
    text_dict = page.get_text("dict")  # Full structure: blocks, lines, spans
    
    # Or as HTML (preserves some structure)
    html = page.get_text("html")

# Extract images
for page in doc:
    images = page.get_images()
    for img in images:
        xref = img[0]
        pix = fitz.Pixmap(doc, xref)
        pix.save(f"image_{xref}.png")
```

| Pros | Cons |
|------|------|
| ✅ Fast (C-based) | ❌ Complex API |
| ✅ Block-level extraction (position-aware) | ❌ GPL license (consider for commercial) |
| ✅ Can extract images | ❌ Table extraction still imperfect |
| ✅ OCR integration possible | |

#### pdfplumber

Specialized in table extraction from PDFs:

```python
import pdfplumber

with pdfplumber.open("document.pdf") as pdf:
    for page in pdf.pages:
        # Regular text
        text = page.extract_text()
        
        # Table extraction (its specialty)
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                print(row)  # ['Header1', 'Header2', 'Header3']
```

| Pros | Cons |
|------|------|
| ✅ Best table extraction | ❌ Slower than PyMuPDF |
| ✅ Visual debugging tools | ❌ Doesn't handle scanned docs |
| ✅ Character-level precision | |

### DOCX Processing

#### python-docx

```python
from docx import Document

doc = Document("proposal.docx")

# Extract paragraphs with style information
for para in doc.paragraphs:
    print(f"Style: {para.style.name} | Text: {para.text}")
    # Style names like: 'Heading 1', 'Heading 2', 'Normal', 'List Bullet'

# Extract tables
for table in doc.tables:
    for row in table.rows:
        for cell in row.cells:
            print(cell.text, end=" | ")
        print()

# Extract metadata
core_props = doc.core_properties
print(f"Title: {core_props.title}")
print(f"Author: {core_props.author}")
print(f"Created: {core_props.created}")
print(f"Modified: {core_props.modified}")
```

**Building a structured extractor:**

```python
def extract_docx_structured(filepath: str) -> dict:
    """Extract DOCX content preserving document structure."""
    doc = Document(filepath)
    
    sections = []
    current_section = {"heading": "Introduction", "level": 0, "content": []}
    
    for para in doc.paragraphs:
        if para.style.name.startswith("Heading"):
            # Start new section
            if current_section["content"]:
                sections.append(current_section)
            
            level = int(para.style.name.split()[-1]) if para.style.name[-1].isdigit() else 1
            current_section = {
                "heading": para.text,
                "level": level,
                "content": []
            }
        else:
            if para.text.strip():
                current_section["content"].append(para.text)
    
    if current_section["content"]:
        sections.append(current_section)
    
    return {
        "metadata": {
            "title": doc.core_properties.title or filepath,
            "author": doc.core_properties.author,
            "created": str(doc.core_properties.created),
        },
        "sections": sections
    }
```

### PPTX Processing

#### python-pptx

```python
from pptx import Presentation

prs = Presentation("presentation.pptx")

for slide_num, slide in enumerate(prs.slides, 1):
    print(f"--- Slide {slide_num} ---")
    
    # Slide title (usually first placeholder)
    if slide.shapes.title:
        print(f"Title: {slide.shapes.title.text}")
    
    # All text from all shapes
    for shape in slide.shapes:
        if shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                print(paragraph.text)
    
    # Speaker notes
    if slide.notes_slide:
        notes = slide.notes_slide.notes_text_frame.text
        print(f"Notes: {notes}")
```

**Strategy for PPTX**: Each slide becomes one chunk, with the slide title as context:

```python
def extract_pptx_as_chunks(filepath: str) -> list[dict]:
    """Extract each slide as a separate chunk."""
    prs = Presentation(filepath)
    chunks = []
    
    for slide_num, slide in enumerate(prs.slides, 1):
        title = slide.shapes.title.text if slide.shapes.title else f"Slide {slide_num}"
        
        # Collect all text from the slide
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        texts.append(para.text.strip())
        
        # Include speaker notes
        notes = ""
        if slide.notes_slide:
            notes = slide.notes_slide.notes_text_frame.text
        
        chunks.append({
            "text": f"[Slide: {title}]\n" + "\n".join(texts),
            "metadata": {
                "source": filepath,
                "slide_number": slide_num,
                "slide_title": title,
                "has_notes": bool(notes),
                "notes": notes
            }
        })
    
    return chunks
```

### HTML Processing

#### BeautifulSoup4

```python
from bs4 import BeautifulSoup

with open("page.html", "r") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# Remove script, style, and nav elements
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()

# Extract text with structure
for heading in soup.find_all(["h1", "h2", "h3"]):
    section_text = []
    for sibling in heading.find_next_siblings():
        if sibling.name in ["h1", "h2", "h3"]:
            break
        section_text.append(sibling.get_text(strip=True))
    
    print(f"## {heading.get_text(strip=True)}")
    print("\n".join(section_text))
```

### Markdown Processing

Markdown is the easiest — it's already semi-structured text:

```python
import re

def extract_markdown_sections(filepath: str) -> list[dict]:
    """Split markdown by headers into sections."""
    with open(filepath, "r") as f:
        content = f.read()
    
    # Split by headers
    sections = re.split(r'^(#{1,6}\s+.+)$', content, flags=re.MULTILINE)
    
    chunks = []
    current_header = "Introduction"
    
    for i, section in enumerate(sections):
        if section.startswith("#"):
            current_header = section.strip("# \n")
        elif section.strip():
            chunks.append({
                "text": f"## {current_header}\n{section.strip()}",
                "metadata": {
                    "source": filepath,
                    "section": current_header
                }
            })
    
    return chunks
```

---

## The Unstructured Library

### Why Unstructured?

`unstructured` is a library that provides a **unified interface** for parsing multiple document formats with intelligent structure detection.

```python
from unstructured.partition.auto import partition

# Auto-detect format and parse
elements = partition(filename="document.pdf")

for element in elements:
    print(f"Type: {type(element).__name__}")
    print(f"Text: {element.text[:100]}")
    print(f"Metadata: {element.metadata}")
    print("---")
```

**Element types returned:**
- `Title` — headings and titles
- `NarrativeText` — body paragraphs
- `ListItem` — bullet/numbered list items
- `Table` — table content
- `Image` — image descriptions (with OCR)
- `Header` / `Footer` — page headers/footers
- `PageBreak` — page boundaries

### Unstructured for Different Formats

```python
# PDF
from unstructured.partition.pdf import partition_pdf
elements = partition_pdf("document.pdf", strategy="hi_res")  # hi_res uses layout detection

# DOCX
from unstructured.partition.docx import partition_docx
elements = partition_docx("proposal.docx")

# HTML
from unstructured.partition.html import partition_html
elements = partition_html("page.html")

# PPTX
from unstructured.partition.pptx import partition_pptx
elements = partition_pptx("presentation.pptx")

# Auto-detect
from unstructured.partition.auto import partition
elements = partition("any_file.xyz")  # Figures out the format
```

### Unstructured + Metadata

```python
elements = partition(filename="report.pdf")

for element in elements:
    meta = element.metadata
    print(f"Text: {element.text[:80]}...")
    print(f"  Type: {type(element).__name__}")
    print(f"  Page: {meta.page_number}")
    print(f"  Section: {meta.section}")
    print(f"  Filename: {meta.filename}")
```

### Unstructured Chunking

Unstructured has its own chunking that respects document structure:

```python
from unstructured.chunking.title import chunk_by_title

elements = partition(filename="report.pdf")

chunks = chunk_by_title(
    elements,
    max_characters=1500,
    new_after_n_chars=1000,
    combine_text_under_n_chars=200
)

for chunk in chunks:
    print(f"Chunk ({len(chunk.text)} chars): {chunk.text[:100]}...")
    print(f"  Metadata: {chunk.metadata}")
```

| Pros | Cons |
|------|------|
| ✅ Unified API for all formats | ❌ Heavier dependency (many sub-packages) |
| ✅ Intelligent structure detection | ❌ `hi_res` strategy requires extra models |
| ✅ Built-in OCR support | ❌ Can be slow for large documents |
| ✅ Document element types | ❌ Table extraction quality varies |
| ✅ Active development | ❌ Breaking changes between versions |
| ✅ Works with LangChain loaders | |

---

## Handling Messy Documents

### OCR for Scanned Documents

When PDFs are scanned images (no text layer):

```python
# Using unstructured with OCR
elements = partition_pdf(
    "scanned_document.pdf",
    strategy="hi_res",       # Use layout model
    ocr_languages=["eng"],   # OCR language
    infer_table_structure=True
)

# Or using pytesseract directly
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

doc = fitz.open("scanned.pdf")
for page in doc:
    pix = page.get_pixmap(dpi=300)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img)
```

### Cleaning Extracted Text

```python
import re

def clean_extracted_text(text: str) -> str:
    """Clean up common document extraction artifacts."""
    
    # Fix common OCR artifacts
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    
    # Remove page numbers (standalone numbers on their own line)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove headers/footers (repeated text across pages)
    # This is document-specific — might need customization
    
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)       # Multiple spaces → single
    
    # Fix broken words (hyphena-\ntion → hyphenation)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Remove null bytes and control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    
    return text.strip()
```

### Handling Tables

Tables are one of the hardest challenges in document processing:

```python
def table_to_text(table_data: list[list[str]]) -> str:
    """Convert a table to readable text format."""
    if not table_data or not table_data[0]:
        return ""
    
    headers = table_data[0]
    rows = table_data[1:]
    
    text_parts = []
    for row in rows:
        row_description = "; ".join(
            f"{headers[i]}: {cell}" 
            for i, cell in enumerate(row) 
            if cell and i < len(headers)
        )
        text_parts.append(row_description)
    
    return "Table data:\n" + "\n".join(text_parts)

# Example:
# Input: [["Year", "Revenue", "Growth"], ["2023", "$5.2B", "12%"], ["2024", "$6.1B", "17%"]]
# Output:
# Table data:
# Year: 2023; Revenue: $5.2B; Growth: 12%
# Year: 2024; Revenue: $6.1B; Growth: 17%
```

### Detecting and Handling Document Quality Issues

```python
def assess_document_quality(text: str, source: str) -> dict:
    """Assess the quality of extracted text."""
    issues = []
    
    # Check for OCR artifacts
    garbled_ratio = len(re.findall(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', text)) / max(len(text), 1)
    if garbled_ratio > 0.05:
        issues.append("Possible OCR artifacts (>5% unusual characters)")
    
    # Check for very short extraction
    if len(text.split()) < 50:
        issues.append("Very little text extracted — may be image-based")
    
    # Check for missing whitespace (common in bad PDF extraction)
    words = text.split()
    long_words = [w for w in words if len(w) > 30]
    if len(long_words) / max(len(words), 1) > 0.1:
        issues.append("Missing word boundaries — text may be concatenated")
    
    # Check for encoding issues
    if "â€" in text or "Ã©" in text or "â€™" in text:
        issues.append("Encoding issues detected (UTF-8/Latin-1 mismatch)")
    
    return {
        "source": source,
        "word_count": len(words),
        "character_count": len(text),
        "quality_issues": issues,
        "is_clean": len(issues) == 0
    }
```

---

## Building the Ingestion Pipeline

### Unified Document Loader

```python
from pathlib import Path
from typing import Union

class DocumentLoader:
    """Unified document loader supporting multiple formats."""
    
    SUPPORTED_FORMATS = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt"}
    
    def load(self, filepath: Union[str, Path]) -> dict:
        """Load a document and return structured content + metadata."""
        filepath = Path(filepath)
        
        if filepath.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {filepath.suffix}")
        
        # Route to appropriate loader
        loader_map = {
            ".pdf": self._load_pdf,
            ".docx": self._load_docx,
            ".pptx": self._load_pptx,
            ".html": self._load_html,
            ".htm": self._load_html,
            ".md": self._load_markdown,
            ".txt": self._load_text,
        }
        
        loader = loader_map[filepath.suffix]
        result = loader(filepath)
        
        # Post-processing
        result["text"] = clean_extracted_text(result["text"])
        result["quality"] = assess_document_quality(result["text"], str(filepath))
        
        return result
    
    def _load_pdf(self, filepath: Path) -> dict:
        """Load PDF using unstructured with fallback to pypdf."""
        try:
            from unstructured.partition.pdf import partition_pdf
            elements = partition_pdf(str(filepath), strategy="fast")
            text = "\n\n".join([el.text for el in elements if el.text])
            return {"text": text, "metadata": {"format": "pdf"}, "elements": elements}
        except Exception:
            from pypdf import PdfReader
            reader = PdfReader(str(filepath))
            text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            return {"text": text, "metadata": {"format": "pdf", "pages": len(reader.pages)}}
    
    # ... similar methods for other formats
```

---

## Best Practices

- ✅ **Test extraction on real documents** — synthetic test files don't reveal real-world issues
- ✅ **Build a quality assessment step** — flag documents with extraction issues
- ✅ **Preserve document structure** — headings, sections, tables are valuable signals
- ✅ **Extract rich metadata** — title, author, date, section, page number
- ✅ **Handle failures gracefully** — some documents will fail; log and continue
- ✅ **Use unstructured for the 80% case** — add specialized parsers for the 20%
- ❌ **Don't assume all PDFs are the same** — text-based vs scanned vs mixed
- ❌ **Don't discard tables** — convert to text format or keep as structured data
- ❌ **Don't ignore encoding** — UTF-8, Latin-1, Windows-1252 all exist in enterprise docs
- ❌ **Don't process in-memory for large files** — stream or process page by page

---

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Garbled text from PDFs | Scanned PDFs or font encoding issues | Detect and route to OCR |
| Lost table structure | Tables treated as flowing text | Use table-aware extraction (pdfplumber) |
| Headers/footers in every chunk | Repeated content per page | Detect and strip repeated text |
| Missing images/diagrams | Text extraction ignores visual content | Log warnings for pages with images |
| Encoding errors | Mixed encodings in document set | Detect and normalize encoding |

---

## Application to Our Project

### Implementation Priority

1. **Start with Markdown and text** — easiest, test pipeline end-to-end
2. **Add DOCX** — common format, good structure extraction
3. **Add PDF (text-based)** — most common but harder
4. **Add PPTX** — slide-per-chunk strategy
5. **Add HTML** — boilerplate removal is the main challenge
6. **Add OCR for scanned PDFs** — most complex, do last

### Test Corpus Plan

Create a test corpus of 10-20 documents covering:
- 5 well-structured PDFs (with headings, tables)
- 3 DOCX proposals (with heading hierarchy)
- 2 PPTX presentations
- 2 HTML pages
- 2 Markdown documents
- 1 scanned PDF (OCR test)
- 1 poorly formatted document (stress test)

---

## Resources for Deeper Learning

- [Unstructured documentation](https://docs.unstructured.io/) — Main library docs
- [PyMuPDF documentation](https://pymupdf.readthedocs.io/) — Powerful PDF processing
- [pdfplumber documentation](https://github.com/jsvine/pdfplumber) — Table extraction
- [LangChain document loaders](https://python.langchain.com/docs/how_to/#document-loaders) — Built-in loaders
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — OCR engine

---

## Questions Remaining

- [ ] How well does unstructured handle our specific document types? (Need to test)
- [ ] Is the `hi_res` strategy worth the performance cost for our PDFs?
- [ ] How to handle documents with mixed content (text + diagrams + tables on same page)?
- [ ] Should we pre-process and cache extracted text, or extract on-demand?
