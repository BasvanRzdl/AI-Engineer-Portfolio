---
date: 2026-02-27
type: technology
topic: "Azure AI Foundry"
project: "Phase 5 - Multi-Modal Enterprise Assistant"
status: complete
---

# Technology Brief: Azure AI Foundry

## Overview

**Azure AI Foundry** (formerly Azure AI Studio) is Microsoft's unified platform for building, evaluating, and deploying AI applications. It provides a single portal to access models from multiple providers, manage AI projects, and operationalize AI solutions.

For the Phase 5 Multi-Modal Enterprise Assistant, Foundry serves as the **central hub** for model discovery, deployment, and management.

## Key Concepts

### Azure AI Foundry Hub

A **Hub** is a top-level Azure resource that provides:
- Centralized security and governance
- Shared resources (compute, storage, key vault)
- Connections to external services (Azure OpenAI, Azure AI Search, etc.)
- One hub can host multiple projects

### Azure AI Foundry Project

A **Project** is a workspace within a hub for building a specific AI application:
- Has its own models, deployments, evaluations
- Scoped permissions (who can access what)
- Connected to parent hub's shared resources
- Equivalent to a "workspace" for your AI application

```
Azure AI Foundry Hub
├── Shared Resources
│   ├── Azure OpenAI connection
│   ├── Azure AI Search connection
│   ├── Azure Storage
│   └── Key Vault
├── Project: Multi-Modal Assistant
│   ├── GPT-4o deployment
│   ├── Whisper deployment
│   ├── Evaluation runs
│   └── Prompt flows
└── Project: Document Intelligence
    ├── GPT-4o-mini deployment
    └── Custom model deployment
```

## Model Catalog

The Model Catalog contains **1,900+ models** from multiple providers:

### Model Categories

| Category | Examples | Deployment Options |
|----------|---------|-------------------|
| **Azure OpenAI** | GPT-4o, GPT-4o-mini, Whisper, DALL-E | Managed compute, Global Standard |
| **Meta** | LLaMA 3.1, LLaMA 3.2, Code LLaMA | Serverless API, Managed compute |
| **Mistral** | Mistral Large, Mistral Small, Mixtral | Serverless API |
| **Microsoft** | Phi-3, Phi-3.5, Phi-4, MAI-1 | Serverless API, Managed compute |
| **Cohere** | Command R+, Embed v3 | Serverless API |
| **Hugging Face** | Thousands of open-source models | Managed compute |
| **NVIDIA** | NIM models | Serverless API |

### Deployment Options

| Option | Description | Cost Model | Best For |
|--------|-------------|-----------|----------|
| **Global Standard** | Azure-managed, auto-scaling | Per token | Azure OpenAI models, production |
| **Standard** | Region-specific deployment | Per token | Data residency requirements |
| **Provisioned Throughput** | Reserved capacity | Per PTU/month | Predictable high-volume workloads |
| **Serverless API (MaaS)** | Pay-per-call, no deployment | Per token | Testing, low-volume, partner models |
| **Managed Compute (MaaP)** | VM with model | Per VM/hour | Fine-tuned models, custom hosting |

### Model Deployment Flow

```
1. Browse Model Catalog
       │
       ▼
2. Select Model (e.g., GPT-4o)
       │
       ▼
3. Choose Deployment Type
   ├── Global Standard (recommended for Azure OpenAI)
   ├── Standard (region-specific)
   ├── Serverless API (for partner models)
   └── Managed Compute (for open-source)
       │
       ▼
4. Configure
   ├── Deployment name
   ├── Token-per-minute limit
   ├── Content filter
   └── Region
       │
       ▼
5. Deploy → Get endpoint + key
       │
       ▼
6. Use via API (same OpenAI SDK)
```

## Using Foundry Models via SDK

### Azure OpenAI Models

```python
from openai import AzureOpenAI

# Same SDK you've been using - Foundry deployments work identically
client = AzureOpenAI(
    azure_endpoint="https://<foundry-resource>.openai.azure.com/",
    api_key="<key>",
    api_version="2024-10-21"
)

response = client.chat.completions.create(
    model="gpt-4o",  # your deployment name
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Serverless API Models (Partner Models)

```python
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# For serverless deployments of Mistral, LLaMA, etc.
client = ChatCompletionsClient(
    endpoint="https://<model-name>.<region>.models.ai.azure.com",
    credential=AzureKeyCredential("<key>")
)

response = client.complete(
    messages=[{"role": "user", "content": "Hello"}],
    model="mistral-large-latest"
)
```

### Azure AI Inference SDK (Unified)

The **Azure AI Inference SDK** provides a unified interface across deployment types:

```python
from azure.ai.inference import ChatCompletionsClient
from azure.identity import DefaultAzureCredential

# Works with any Foundry deployment type
client = ChatCompletionsClient(
    endpoint="https://<project>.services.ai.azure.com/models",
    credential=DefaultAzureCredential()
)

# Same interface regardless of whether it's GPT-4o, LLaMA, or Mistral
response = client.complete(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers."}
    ],
    model="gpt-4o"  # or "meta-llama-3-1-70b" or "mistral-large"
)
```

## Evaluation in Foundry

Foundry provides built-in evaluation for AI applications:

### Available Evaluators

| Evaluator | Measures | Requires Ground Truth |
|-----------|---------|----------------------|
| **Groundedness** | Is the response grounded in the context? | No |
| **Relevance** | Is the response relevant to the question? | No |
| **Coherence** | Is the response well-structured? | No |
| **Fluency** | Is the language natural? | No |
| **Similarity** | How similar to the ground truth? | Yes |
| **F1 Score** | Token overlap with ground truth | Yes |

### Running Evaluations

```python
from azure.ai.evaluation import evaluate, GroundednessEvaluator, RelevanceEvaluator

# Create evaluators
groundedness = GroundednessEvaluator(model_config={
    "azure_endpoint": "...",
    "azure_deployment": "gpt-4o",
    "api_key": "..."
})

# Evaluate
results = evaluate(
    data="test_data.jsonl",  # query, response, context columns
    evaluators={
        "groundedness": groundedness,
        "relevance": RelevanceEvaluator(model_config=model_config)
    }
)

print(results.metrics)
# {'groundedness.score': 4.2, 'relevance.score': 4.5}
```

## Prompt Flow

Prompt Flow is Foundry's visual tool for building LLM application pipelines:

```
┌──────────────────────────────────────────────┐
│              Prompt Flow                      │
│                                               │
│  Input ──▶ [Preprocess] ──▶ [LLM Call]        │
│                               │               │
│                          ──▶ [Post-process]   │
│                               │               │
│                          ──▶ [Output]         │
│                                               │
│  Supports:                                    │
│  ├── LLM nodes (any deployed model)           │
│  ├── Python script nodes                      │
│  ├── Tool nodes (search, API calls)           │
│  ├── Conditional branching                    │
│  └── Parallel execution                       │
└──────────────────────────────────────────────┘
```

**When to use Prompt Flow**:
- Prototyping multi-step LLM pipelines
- A/B testing different prompts or models
- Building evaluation pipelines
- Creating deployable flows as REST APIs

**When NOT to use Prompt Flow**:
- Simple single-call LLM applications (overkill)
- When you need full code control (use Python SDK directly)
- Complex orchestration better suited for code (LangChain, Semantic Kernel)

## Content Safety

Foundry integrates Azure AI Content Safety for filtering:

| Filter Category | What It Detects |
|----------------|-----------------|
| **Hate** | Hate speech, discrimination |
| **Sexual** | Sexual content |
| **Violence** | Violent content |
| **Self-harm** | Self-harm content |
| **Jailbreak** | Prompt injection attacks |
| **Protected material** | Copyrighted content |

Content filters are applied at the deployment level and can be customized (strictness levels: low, medium, high, or custom).

## Foundry for Multi-Modal Applications

### Relevant Features for Phase 5

| Feature | How It Helps |
|---------|-------------|
| **Model Catalog** | Discover and deploy GPT-4o (vision), Whisper (audio), embedding models |
| **Multiple deployments** | Run different models for different modalities |
| **Evaluation** | Test multi-modal pipeline quality |
| **Prompt Flow** | Prototype modality routing and fusion pipelines |
| **Content Safety** | Filter inappropriate content across modalities |
| **Connections** | Connect to Azure AI Search, Storage, Speech Service |

### Example: Multi-Modal Project Setup

```
Foundry Hub: "enterprise-ai"
└── Project: "multi-modal-assistant"
    ├── Deployments:
    │   ├── gpt-4o (Standard) ──── Vision + text reasoning
    │   ├── gpt-4o-mini (Global Standard) ──── Cost-optimized text
    │   ├── whisper (Standard) ──── Audio transcription
    │   └── text-embedding-3-large ──── Vector embeddings
    ├── Connections:
    │   ├── Azure AI Search ──── Vector store for RAG
    │   ├── Azure Blob Storage ──── Media file storage
    │   └── Azure Speech Service ──── Diarization
    └── Evaluations:
        ├── Vision QA accuracy
        ├── Transcription quality (WER)
        └── End-to-end response quality
```

## Pricing Model

Foundry itself is free — you pay for the underlying resources:

| Component | Billing |
|-----------|---------|
| **Hub/Project** | Free |
| **Azure OpenAI deployments** | Per token (varies by model and tier) |
| **Serverless API (partner models)** | Per token (varies by model) |
| **Managed Compute** | Per VM-hour |
| **Storage** | Per GB/month |
| **Evaluation** | LLM evaluators cost model tokens |

## Getting Started Checklist

1. **Create a Foundry Hub** in Azure Portal or via CLI
2. **Create a Project** within the hub
3. **Deploy models**:
   - GPT-4o (for vision + text reasoning)
   - GPT-4o-mini (for cost-optimized tasks)
   - Whisper (for audio transcription)
   - text-embedding-3-large (for embeddings)
4. **Set up connections** to Azure AI Search, Storage
5. **Configure content safety** filters
6. **Test via Playground** in the Foundry portal
7. **Build pipeline** using SDK or Prompt Flow
8. **Evaluate** using built-in evaluators

## Best Practices

- ✅ **Use Foundry as your control plane** — single place for model management and evaluation
- ✅ **Start with the Playground** for interactive testing before writing code
- ✅ **Use Global Standard deployments** for Azure OpenAI models unless you need data residency
- ✅ **Set up content safety** from the beginning — not as an afterthought
- ✅ **Use the Azure AI Inference SDK** for a unified interface across model providers
- ✅ **Track all experiments** — Foundry logs model calls and evaluations
- ❌ **Don't deploy models you're not using** — deployments cost money even when idle (provisioned)
- ❌ **Don't skip evaluation** — Foundry makes it easy, use it
- ❌ **Don't hardcode endpoints** — use Foundry connections for service discovery

## Resources

- [Azure AI Foundry Documentation](https://learn.microsoft.com/en-us/azure/ai-studio/) — Official docs
- [Azure AI Foundry Portal](https://ai.azure.com/) — Web interface
- [Model Catalog](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/model-catalog-overview) — Browse available models
- [Azure AI Inference SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/inference-sdk-overview) — Unified SDK
- [Prompt Flow](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/prompt-flow) — Visual pipeline builder
- [AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-studio/how-to/evaluate-sdk) — Built-in evaluators
