---
date: 2025-02-27
type: technology
topic: "OpenAI & Azure OpenAI API"
project: "Phase 1 - Prompt Engineering Laboratory"
status: complete
---

# Technology: OpenAI & Azure OpenAI API

## In My Own Words

OpenAI provides the models, Azure OpenAI provides an enterprise-grade hosting wrapper. Both expose the same models through similar (but not identical) APIs. For enterprise work, Azure OpenAI adds data residency, SLA guarantees, content filtering, and managed deployments. Our framework should abstract over both providers.

## Why This Matters

- These are the primary LLM providers for the project
- Understanding the API surface is essential for building the framework
- Cost management requires understanding pricing and token usage
- Multi-provider support requires understanding differences between providers

---

## Available Models (2025)

### GPT-5 Series (Latest Generation)

| Model | Context | Max Output | Best For |
|-------|---------|------------|----------|
| **gpt-5** | 1M tokens | 64K | Highest intelligence, complex reasoning |
| **gpt-5-mini** | 1M tokens | 64K | Balance of intelligence and speed |

### GPT-4.1 Series

| Model | Context | Max Output | Best For |
|-------|---------|------------|----------|
| **gpt-4.1** | 1M tokens | 32K | Complex coding, instruction following |
| **gpt-4.1-mini** | 1M tokens | 32K | Fast, cost-effective, good for evals |
| **gpt-4.1-nano** | 1M tokens | 32K | Fastest, cheapest, classification/extraction |

### GPT-4o Series

| Model | Context | Max Output | Best For |
|-------|---------|------------|----------|
| **gpt-4o** | 128K | 16K | Multimodal, strong all-around |
| **gpt-4o-mini** | 128K | 16K | Lightweight, cost-effective |

### Reasoning Models (o-series)

| Model | Context | Best For |
|-------|---------|----------|
| **o3** | 200K | Complex reasoning, analysis |
| **o4-mini** | 200K | Efficient reasoning tasks |

### Embedding Models

| Model | Dimensions | Best For |
|-------|-----------|----------|
| **text-embedding-3-large** | 3072 | Highest quality embeddings |
| **text-embedding-3-small** | 1536 | Cost-effective embeddings |

---

## API Patterns

### OpenAI Client Setup

```python
from openai import AsyncOpenAI

# Direct OpenAI
client = AsyncOpenAI(api_key="sk-...")

# Azure OpenAI
from openai import AsyncAzureOpenAI

client = AsyncAzureOpenAI(
    api_key="...",
    api_version="2024-12-01-preview",
    azure_endpoint="https://your-resource.openai.azure.com"
)
```

### Chat Completions (Core API)

```python
async def chat_completion(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    max_tokens: int | None = None,
    response_format: dict | None = None,
) -> dict:
    """Make a chat completion call with full metadata."""
    import time
    
    start = time.monotonic()
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    elapsed = time.monotonic() - start
    
    return {
        "content": response.choices[0].message.content,
        "finish_reason": response.choices[0].finish_reason,
        "model": response.model,
        "usage": {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "latency_seconds": elapsed,
    }
```

### Streaming

```python
async def stream_completion(client, messages, model="gpt-4o"):
    """Stream a completion — useful for long outputs."""
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )
    
    full_content = ""
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            yield content
```

### Batch API

For running evaluations at scale (50% cost reduction):

```python
# Create a batch of requests in JSONL format
import json

def create_batch_file(requests: list[dict], output_path: str):
    """Create a JSONL batch file for OpenAI Batch API."""
    with open(output_path, "w") as f:
        for i, req in enumerate(requests):
            batch_request = {
                "custom_id": f"request-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": req["model"],
                    "messages": req["messages"],
                    "temperature": req.get("temperature", 0.0),
                }
            }
            f.write(json.dumps(batch_request) + "\n")

# Submit batch
batch_file = await client.files.create(
    file=open("batch_input.jsonl", "rb"),
    purpose="batch"
)
batch = await client.batches.create(
    input_file_id=batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)
```

---

## Key API Differences: OpenAI vs Azure OpenAI

| Feature | OpenAI | Azure OpenAI |
|---------|--------|--------------|
| **Authentication** | API key (`sk-...`) | API key + Azure AD/Entra ID |
| **Endpoint** | `api.openai.com` | `{resource}.openai.azure.com` |
| **Model reference** | Model name (`gpt-4o`) | Deployment name (custom) |
| **API version** | Built into SDK | Explicit (`api_version` param) |
| **Rate limits** | Per-org, per-model | Per-deployment, configurable |
| **Content filter** | Basic | Configurable severity levels |
| **Data residency** | US | Choose region |
| **SLA** | Best-effort | 99.9% uptime SLA |
| **Batch API** | ✅ | ✅ (Global Batch) |
| **Structured Outputs** | ✅ | ✅ (API version dependent) |

### Multi-Provider Abstraction

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Abstract base for LLM providers."""
    
    @abstractmethod
    async def complete(
        self, 
        messages: list[dict], 
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """Send a completion request."""
        ...
    
    @abstractmethod
    def get_pricing(self, model: str) -> dict:
        """Get pricing for a model."""
        ...


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
    
    async def complete(self, messages, model="gpt-4o", **kwargs):
        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return self._format_response(response)


class AzureOpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, endpoint: str, api_version: str):
        self.client = AsyncAzureOpenAI(
            api_key=api_key, 
            azure_endpoint=endpoint,
            api_version=api_version
        )
        self.deployment_map = {}  # model_name -> deployment_name
    
    async def complete(self, messages, model="gpt-4o", **kwargs):
        deployment = self.deployment_map.get(model, model)
        response = await self.client.chat.completions.create(
            model=deployment, messages=messages, **kwargs
        )
        return self._format_response(response)
```

---

## Pricing & Cost Management (2025 Estimates)

### Token Pricing (per 1M tokens)

| Model | Input | Output | Cached Input |
|-------|-------|--------|--------------|
| gpt-4o | $2.50 | $10.00 | $1.25 |
| gpt-4o-mini | $0.15 | $0.60 | $0.075 |
| gpt-4.1 | $2.00 | $8.00 | $0.50 |
| gpt-4.1-mini | $0.40 | $1.60 | $0.10 |
| gpt-4.1-nano | $0.10 | $0.40 | $0.025 |
| o3 | $10.00 | $40.00 | $2.50 |
| o4-mini | $1.10 | $4.40 | $0.275 |

### Cost Tracking Implementation

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class CostTracker:
    """Track API costs across all calls."""
    
    calls: list[dict] = field(default_factory=list)
    
    PRICING = {
        "gpt-4o": {"input": 2.50e-6, "output": 10.0e-6},
        "gpt-4o-mini": {"input": 0.15e-6, "output": 0.60e-6},
        "gpt-4.1": {"input": 2.0e-6, "output": 8.0e-6},
        "gpt-4.1-mini": {"input": 0.40e-6, "output": 1.60e-6},
        "gpt-4.1-nano": {"input": 0.10e-6, "output": 0.40e-6},
    }
    
    def record(self, model: str, input_tokens: int, output_tokens: int, 
               latency: float, purpose: str = ""):
        pricing = self.PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        
        self.calls.append({
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "latency": latency,
            "purpose": purpose,
        })
        return cost
    
    @property
    def total_cost(self) -> float:
        return sum(c["cost"] for c in self.calls)
    
    @property
    def total_tokens(self) -> int:
        return sum(c["input_tokens"] + c["output_tokens"] for c in self.calls)
    
    def summary(self) -> dict:
        return {
            "total_calls": len(self.calls),
            "total_cost": f"${self.total_cost:.4f}",
            "total_tokens": self.total_tokens,
            "by_model": self._group_by_model(),
            "by_purpose": self._group_by_purpose(),
        }
    
    def _group_by_model(self) -> dict:
        groups = {}
        for call in self.calls:
            model = call["model"]
            if model not in groups:
                groups[model] = {"calls": 0, "cost": 0, "tokens": 0}
            groups[model]["calls"] += 1
            groups[model]["cost"] += call["cost"]
            groups[model]["tokens"] += call["input_tokens"] + call["output_tokens"]
        return groups
    
    def _group_by_purpose(self) -> dict:
        groups = {}
        for call in self.calls:
            purpose = call["purpose"] or "unknown"
            if purpose not in groups:
                groups[purpose] = {"calls": 0, "cost": 0}
            groups[purpose]["calls"] += 1
            groups[purpose]["cost"] += call["cost"]
        return groups
```

### Budget Enforcement

```python
class BudgetManager:
    """Enforce cost budgets per session/day/project."""
    
    def __init__(self, daily_budget: float = 10.0, session_budget: float = 5.0):
        self.daily_budget = daily_budget
        self.session_budget = session_budget
        self.tracker = CostTracker()
    
    def can_afford(self, model: str, estimated_tokens: int) -> bool:
        """Check if we can afford an API call."""
        pricing = self.tracker.PRICING.get(model, {})
        estimated_cost = estimated_tokens * max(
            pricing.get("input", 0), pricing.get("output", 0)
        )
        return (self.tracker.total_cost + estimated_cost) < self.session_budget
    
    def enforce(self, model: str, estimated_tokens: int):
        """Raise if budget would be exceeded."""
        if not self.can_afford(model, estimated_tokens):
            raise BudgetExceededError(
                f"Budget exceeded. Current: ${self.tracker.total_cost:.4f}, "
                f"Budget: ${self.session_budget:.2f}"
            )
```

---

## Rate Limiting & Retry Logic

```python
import asyncio
from openai import RateLimitError, APITimeoutError

async def completion_with_retry(
    client, messages, model="gpt-4o", 
    max_retries=3, initial_delay=1.0, **kwargs
):
    """Completion with exponential backoff retry."""
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except RateLimitError:
            if attempt == max_retries:
                raise
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
        except APITimeoutError:
            if attempt == max_retries:
                raise
            await asyncio.sleep(delay)
            delay *= 1.5
```

---

## Async Patterns for Evaluation

Evaluations often need to run many API calls in parallel:

```python
import asyncio

async def run_parallel_evaluations(
    client, test_cases: list[dict], model: str, max_concurrent: int = 10
):
    """Run evaluations with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_one(test_case):
        async with semaphore:
            return await chat_completion(
                client, test_case["messages"], model=model
            )
    
    tasks = [evaluate_one(tc) for tc in test_cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return [
        r if not isinstance(r, Exception) else {"error": str(r)}
        for r in results
    ]
```

---

## Content Safety (Azure OpenAI)

Azure OpenAI adds configurable content filtering:

### Safety System Message Template

```python
SAFETY_SYSTEM_MESSAGE = """
## Safety Guidelines

You must follow these rules:
- Do not generate content that could be harmful, hateful, or violent
- Do not generate content about protected material (copyrighted text)
- If you don't know something, say so — do not make up information
- Ground your responses in the provided context only
- If asked to do something outside your scope, politely decline

## Grounding Instructions

- Only use information from the provided documents
- If the answer cannot be found in the provided documents, say 
  "I cannot find this information in the provided documents"
- Do not speculate or infer beyond what is explicitly stated
"""
```

### Content Filter Configuration

Azure OpenAI content filters check four categories:
- **Hate and Fairness** — Severity levels: safe, low, medium, high
- **Sexual** — Severity levels: safe, low, medium, high
- **Violence** — Severity levels: safe, low, medium, high
- **Self-Harm** — Severity levels: safe, low, medium, high

Each can be configured independently for both input and output.

---

## Best Practices

- ✅ **Use async** for all API calls — essential for evaluation parallelism
- ✅ **Track every call** — tokens, cost, latency, purpose
- ✅ **Implement retry logic** with exponential backoff
- ✅ **Set budgets** and enforce them before calls
- ✅ **Abstract over providers** — don't hardcode OpenAI-specific patterns
- ✅ **Use the Batch API** for large evaluation runs (50% savings)
- ❌ Don't hardcode model names — use configuration
- ❌ Don't ignore rate limits — implement proper backoff
- ❌ Don't forget to close async clients properly

---

## Application to My Project

### How I'll Use This

1. **Provider abstraction layer** — `LLMProvider` base class with OpenAI and Azure implementations
2. **Built-in cost tracking** — Every call recorded with purpose tagging
3. **Budget enforcement** — Configurable per-session and daily limits
4. **Async-first** — All API interactions are async with controlled concurrency
5. **Retry with backoff** — Transparent retry logic for all calls

### Decisions to Make

- [ ] Which models to support initially? (gpt-4o + gpt-4o-mini as minimum)
- [ ] Azure OpenAI or direct OpenAI for development?
- [ ] How to handle API key management? (env vars, config file, keyring?)
- [ ] Max concurrency for evaluation runs?
- [ ] Should cost tracking persist to disk or just in-memory?

---

## Resources for Deeper Learning

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/) — Complete Azure reference
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — API endpoints and parameters
- [OpenAI Cookbook](https://cookbook.openai.com/) — Practical examples and recipes
- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — Current pricing
