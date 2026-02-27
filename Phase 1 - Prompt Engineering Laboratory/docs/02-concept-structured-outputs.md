---
date: 2025-02-27
type: concept
topic: "Structured Outputs & Output Parsing"
project: "Phase 1 - Prompt Engineering Laboratory"
status: complete
---

# Learning: Structured Outputs & Output Parsing

## In My Own Words

Structured output is about getting LLMs to return data in a predictable, machine-parseable format (usually JSON) rather than free-form text. This is critical for building reliable systems because downstream code needs to **programmatically process** model outputs. There are three main approaches: prompt-based JSON, JSON Mode, and Structured Outputs (schema-enforced).

## Why This Matters

- **Reliability**: Unstructured text is fragile to parse
- **Integration**: APIs, databases, and UIs need structured data
- **Validation**: You can check if outputs conform to expected schemas
- **Automation**: Structured outputs enable pipeline architectures

---

## Core Approaches

### 1. Prompt-Based JSON (Basic)

Simply ask the model to return JSON in your prompt:

```python
prompt = """
Extract entities from the text below. Return valid JSON:
{
  "persons": ["name1", "name2"],
  "organizations": ["org1"],
  "dates": ["date1"]
}

Text: {input_text}
"""
```

**Pros**: Works with any model, no special API features needed
**Cons**: Model may not always return valid JSON, may add markdown fences, may hallucinate fields

### 2. JSON Mode (API-Level)

OpenAI and Azure OpenAI offer a `response_format` parameter:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"}
)
```

**Pros**: Guarantees valid JSON output
**Cons**: Doesn't enforce a specific schema — model chooses the structure

**Important**: You MUST mention "JSON" in your prompt when using JSON mode, or the API may error or produce unexpected results.

### 3. Structured Outputs (Schema-Enforced)

The most robust approach — provide a JSON Schema that the model must follow:

```python
from pydantic import BaseModel

class ExtractedEntities(BaseModel):
    persons: list[str]
    organizations: list[str]  
    dates: list[str]
    confidence: float

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "entity_extraction",
            "schema": ExtractedEntities.model_json_schema()
        }
    }
)
```

**Pros**: Guarantees both valid JSON AND correct schema
**Cons**: Slightly higher latency, some schema limitations (no arbitrary additional properties)

### 4. Function Calling / Tool Use

Originally designed for tool integration, but widely used for structured extraction:

```python
tools = [{
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": "Extract named entities from text",
        "parameters": {
            "type": "object",
            "properties": {
                "persons": {"type": "array", "items": {"type": "string"}},
                "organizations": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"}
            },
            "required": ["persons", "organizations", "confidence"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "extract_entities"}}
)
```

**Pros**: Well-structured, strongly typed, parallel tool calls supported
**Cons**: Adds complexity, originally designed for tool use not just extraction

---

## Pydantic for Structured Outputs

### Why Pydantic?

Pydantic is the natural bridge between LLM outputs and Python code:

1. **Define schemas as Python classes** → readable, type-safe
2. **Generate JSON Schema automatically** → feed to OpenAI Structured Outputs
3. **Validate responses** → catch malformed outputs
4. **Serialize/deserialize** → convert between JSON and Python objects

### Core Patterns

#### Basic Model Definition

```python
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class DocumentSummary(BaseModel):
    """Structured summary of a document."""
    title: str = Field(description="Title or subject of the document")
    summary: str = Field(description="2-3 sentence summary")
    key_points: list[str] = Field(description="Main takeaways", min_length=1, max_length=5)
    sentiment: Sentiment = Field(description="Overall sentiment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    categories: list[str] = Field(default_factory=list)
    word_count: Optional[int] = None
```

#### Nested Models

```python
class Entity(BaseModel):
    name: str
    entity_type: str  # PERSON, ORG, DATE, etc.
    confidence: float = Field(ge=0.0, le=1.0)

class Relationship(BaseModel):
    source: str
    target: str
    relationship_type: str

class ExtractionResult(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]
    raw_text_length: int
```

#### Validation

```python
from pydantic import BaseModel, field_validator

class ClassificationResult(BaseModel):
    label: str
    confidence: float
    reasoning: str
    
    @field_validator('confidence')
    @classmethod
    def confidence_must_be_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v
    
    @field_validator('label')
    @classmethod
    def label_must_be_known(cls, v):
        valid_labels = {'billing', 'technical', 'feature_request', 'complaint'}
        if v not in valid_labels:
            raise ValueError(f'Label must be one of {valid_labels}')
        return v
```

#### Generating JSON Schema for OpenAI

```python
# Pydantic models produce JSON Schema that OpenAI understands
schema = DocumentSummary.model_json_schema()
# Returns a dict that can be passed to response_format

# Parsing the response back
import json
response_text = response.choices[0].message.content
result = DocumentSummary.model_validate_json(response_text)
# Now result is a fully typed Python object
print(result.summary)
print(result.key_points[0])
```

### Key Pydantic Features for LLM Work

| Feature | Usage | Example |
|---------|-------|---------|
| `Field(description=...)` | Helps the LLM understand what to generate | `name: str = Field(description="Full legal name")` |
| `Field(ge=, le=)` | Numeric constraints | `score: float = Field(ge=0, le=1)` |
| `Optional[T]` | Fields that might be missing | `email: Optional[str] = None` |
| `Literal["a", "b"]` | Constrained string values | `status: Literal["active", "inactive"]` |
| `Enum` | Enumerated types | `class Color(str, Enum): ...` |
| `model_json_schema()` | Export to JSON Schema | For OpenAI structured outputs |
| `model_validate_json()` | Parse JSON string to model | For response parsing |
| `model_dump()` | Convert to dictionary | For serialization |

---

## Error Handling Strategy

### Parsing Pipeline

```python
import json
from pydantic import ValidationError

def parse_llm_response(response_text: str, model_class: type[BaseModel]):
    """Robust parsing of LLM responses."""
    
    # Step 1: Try direct JSON parse
    try:
        return model_class.model_validate_json(response_text)
    except (json.JSONDecodeError, ValidationError):
        pass
    
    # Step 2: Try to extract JSON from markdown fences
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
        try:
            return model_class.model_validate_json(json_str)
        except (json.JSONDecodeError, ValidationError):
            pass
    
    # Step 3: Try to find JSON object in text
    import re
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        try:
            return model_class.model_validate_json(json_match.group())
        except (json.JSONDecodeError, ValidationError):
            pass
    
    # Step 4: Retry with the LLM (or raise)
    raise ValueError(f"Could not parse response as {model_class.__name__}")
```

### Retry Strategy

```python
async def get_structured_output(
    client, prompt: str, model_class: type[BaseModel], max_retries: int = 3
):
    """Get structured output with retries."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": model_class.__name__,
                        "schema": model_class.model_json_schema()
                    }
                }
            )
            return model_class.model_validate_json(
                response.choices[0].message.content
            )
        except ValidationError as e:
            if attempt == max_retries - 1:
                raise
            # Optionally: include error in retry prompt
```

---

## Comparison of Approaches

| Feature | Prompt-Based | JSON Mode | Structured Outputs | Function Calling |
|---------|-------------|-----------|-------------------|-----------------|
| Valid JSON guaranteed | ❌ | ✅ | ✅ | ✅ |
| Schema enforced | ❌ | ❌ | ✅ | ✅ |
| Works with any model | ✅ | ❌ (OpenAI) | ❌ (OpenAI) | ❌ (OpenAI) |
| Latency overhead | None | Minimal | Small | Small |
| Complexity | Low | Low | Medium | Medium |
| Best for | Quick prototypes | Simple JSON | Production systems | Tool integration |

---

## Best Practices

- ✅ Use Pydantic models for all structured outputs — they serve as documentation AND validation
- ✅ Include `description` in Field definitions — the LLM uses these as guidance
- ✅ Use `Literal` types and `Enum` for constrained values — prevents hallucinated categories
- ✅ Always handle parsing errors gracefully — have a retry/fallback strategy
- ✅ Prefer Structured Outputs over prompt-based JSON in production
- ❌ Don't rely on the model always returning valid JSON without enforcement
- ❌ Don't use overly complex nested schemas — flatten when possible
- ❌ Don't forget to validate that the content is correct (schema compliance ≠ factual accuracy)

---

## Application to My Project

### How I'll Use This

1. **Every prompt template** will have an associated Pydantic model defining its expected output
2. **The framework** will automatically generate JSON Schema from Pydantic models
3. **Response parsing** will use the robust pipeline (try structured → JSON mode → prompt-based)
4. **Multi-provider support**: Use Structured Outputs for OpenAI, prompt-based for others

### Decisions to Make

- [ ] Minimum Pydantic version to require (v2+ recommended)
- [ ] Whether to use `response_format` (structured outputs) or function calling for extraction
- [ ] How to handle providers that don't support structured outputs natively
- [ ] Strategy for schema evolution (adding fields to existing models)

---

## Resources for Deeper Learning

- [Pydantic Documentation](https://docs.pydantic.dev/latest/) — Complete reference
- [OpenAI Structured Outputs Guide](https://platform.openai.com/docs/guides/structured-outputs) — Official guide
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) — Tool use patterns
- [Instructor Library](https://github.com/jxnl/instructor) — Pydantic + LLM structured outputs made easy
