---
date: 2026-02-27
type: concept
topic: "Microservices Architecture for AI"
project: "Phase 6 - Enterprise AI Platform"
status: complete
---

# Microservices Architecture for AI

## In My Own Words

Microservices architecture means breaking your application into small, independently deployable services that each do one thing well. For an AI platform, this means each AI capability (knowledge search, customer agent, research team, multi-modal assistant) runs as its own service with its own resources, can be scaled independently, and can fail without taking down the entire platform.

The key insight for AI specifically: different AI workloads have wildly different resource profiles. A knowledge search query might take 200ms and use minimal GPU. A research team orchestration might take 5 minutes and make 50 LLM calls. You can't treat these the same — microservices let you give each the resources and scaling rules it needs.

## Why This Matters

Phase 6 requires integrating four previous projects into a unified platform. Each project was built independently. Microservices is the natural architecture because:

1. Each project already exists as a separate codebase
2. Each has different scaling requirements
3. Each may use different models/resources
4. Teams (in a real enterprise) could own different services
5. Failure in one service shouldn't crash the whole platform

## Core Principles

1. **Single Responsibility**: Each service does one thing. The knowledge service handles document search. The agent service handles customer operations. No mixing.

2. **Independent Deployability**: You can update the agent service without touching the knowledge service. This enables faster iteration and reduces deployment risk.

3. **Decentralized Data Management**: Each service owns its data. The knowledge service owns the vector database. The agent service owns conversation state. No shared databases between services.

4. **Design for Failure**: Services will fail. Design so that failures are isolated, detected quickly, and handled gracefully (circuit breakers, retries, fallbacks).

5. **Smart Endpoints, Dumb Pipes**: Services contain the business logic. Communication between them is simple (HTTP/REST, gRPC, or message queues) — not a smart enterprise service bus.

## How It Works

### The Big Picture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY (FastAPI)                          │
│         Auth │ Rate Limiting │ Routing │ Cost Tracking                │
└──────┬───────────┬───────────────┬──────────────┬────────────────────┘
       │           │               │              │
┌──────▼─────┐ ┌───▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│ Knowledge  │ │ Customer │ │ Research   │ │ Multi-Modal│
│ Base       │ │ Agent    │ │ Team       │ │ Assistant  │
│ Service    │ │ Service  │ │ Service    │ │ Service    │
├────────────┤ ├──────────┤ ├────────────┤ ├────────────┤
│ Vector DB  │ │ State    │ │ Task Queue │ │ File Store │
│ (Qdrant)   │ │ Store    │ │ (Redis)    │ │ (Blob)     │
└────────────┘ └──────────┘ └────────────┘ └────────────┘
       │           │               │              │
       └───────────┴───────┬───────┴──────────────┘
                           │
                    ┌──────▼──────┐
                    │ Shared Infra│
                    │ - LLM Pool  │
                    │ - Logging   │
                    │ - Metrics   │
                    │ - Config    │
                    └─────────────┘
```

### Step by Step: Request Flow

1. **Client → Gateway**: Client sends `POST /api/v1/research/start` with a research question
2. **Gateway → Service**: After auth/rate-limit, gateway routes to the Research Team Service
3. **Service processes**: Research service decomposes the task, spawns agents, makes LLM calls
4. **Inter-service calls**: Research service might call Knowledge Base Service to retrieve documents (service-to-service communication)
5. **Service → Gateway**: Research service returns a task ID (async pattern — research takes time)
6. **Client polls**: Client calls `GET /api/v1/research/status/{task_id}` to check progress

### Communication Patterns Between Services

```
┌───────────────────────────────────────────────────────────────┐
│                  COMMUNICATION PATTERNS                        │
├───────────────────┬───────────────────────────────────────────┤
│ Synchronous       │ HTTP/REST or gRPC                         │
│ (request-reply)   │ Used when: caller needs immediate answer  │
│                   │ Example: Knowledge search during research │
├───────────────────┼───────────────────────────────────────────┤
│ Asynchronous      │ Message queue (Redis, RabbitMQ, etc.)     │
│ (fire-and-forget) │ Used when: long-running tasks             │
│                   │ Example: Start research, get task ID back │
├───────────────────┼───────────────────────────────────────────┤
│ Event-driven      │ Pub/Sub (Redis Pub/Sub, Azure Event Grid) │
│ (broadcast)       │ Used when: multiple services care         │
│                   │ Example: "New document ingested" event    │
└───────────────────┴───────────────────────────────────────────┘
```

## Key Patterns for AI Microservices

### Pattern 1: Service Per AI Capability

Each AI capability = one service. Clean, simple, obvious boundaries.

```
knowledge-service/          # Phase 2 - Document Intelligence
├── app/
│   ├── main.py             # FastAPI app
│   ├── routes/
│   │   ├── search.py       # Search endpoints
│   │   └── ingest.py       # Document ingestion
│   ├── services/
│   │   ├── retriever.py    # RAG retrieval logic
│   │   └── embedder.py     # Embedding generation
│   └── models/
│       └── schemas.py      # Pydantic models
├── Dockerfile
└── requirements.txt
```

### Pattern 2: Shared LLM Client Pool

Don't let each service manage its own LLM connections. Use a shared pattern:

```python
# shared/llm_client.py — used by all services
from openai import AsyncAzureOpenAI

class LLMClientPool:
    """Centralized LLM client with connection pooling and retry logic."""
    
    def __init__(self, config: LLMConfig):
        self.client = AsyncAzureOpenAI(
            azure_endpoint=config.endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
        )
        self.default_model = config.default_model
    
    async def complete(
        self, 
        messages: list[dict],
        model: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """Unified completion with automatic cost tracking."""
        model = model or self.default_model
        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        # Track tokens and cost
        return LLMResponse(
            content=response.choices[0].message.content,
            tokens_used=response.usage.total_tokens,
            cost=self._calculate_cost(model, response.usage),
        )
```

### Pattern 3: Sidecar for Cross-Cutting Concerns

Instead of every service implementing logging, metrics, and tracing — use a sidecar pattern where a small helper process runs alongside each service to handle these concerns.

In Kubernetes, this is naturally supported via sidecar containers in a pod.

### Pattern 4: Strangler Fig for Integration

Since we're integrating existing projects (Phase 2-5), use the Strangler Fig pattern:

1. Start with the gateway routing to your existing projects (even if they're monolithic)
2. Gradually refactor each project into a proper microservice behind the gateway
3. The gateway abstracts the transition — clients don't notice

This avoids a "big bang" rewrite.

### Pattern 5: Async Task Pattern for Long-Running AI Operations

Some AI operations (research, multi-step agent tasks) take minutes. Don't hold HTTP connections open.

```
Client                    Gateway           Research Service           Task Queue
  │                         │                     │                       │
  │  POST /research/start   │                     │                       │
  │────────────────────────►│                     │                       │
  │                         │  Forward request    │                       │
  │                         │────────────────────►│                       │
  │                         │                     │  Enqueue task          │
  │                         │                     │──────────────────────►│
  │                         │  Return task_id     │                       │
  │                         │◄────────────────────│                       │
  │  202 Accepted           │                     │                       │
  │  { task_id: "abc123" }  │                     │                       │
  │◄────────────────────────│                     │                       │
  │                         │                     │                       │
  │  GET /research/abc123   │                     │                       │
  │────────────────────────►│                     │                       │
  │                         │  Check status       │                       │
  │  200 { status: "running", progress: 60% }     │                       │
  │◄────────────────────────│                     │                       │
```

## Approaches & Trade-offs

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Microservices (per AI capability)** | Multiple distinct AI services, different scaling needs | Independent scaling/deployment, fault isolation, clear ownership | Operational complexity, network latency, distributed debugging |
| **Modular Monolith** | Small team, tightly coupled services, early stage | Simpler deployment, no network overhead, easier debugging | Hard to scale individually, can become spaghetti, deployment couples everything |
| **Service Mesh (Istio/Linkerd)** | Many services, complex traffic rules | Automatic mTLS, observability, traffic splitting | Very complex, resource-heavy, steep learning curve |
| **Serverless Functions** | Event-driven, sporadic workloads | Pay-per-use, auto-scaling, no infra management | Cold starts, execution limits, vendor lock-in |

### Recommendation for Phase 6

**Use microservices (one per AI capability)** with a FastAPI gateway. Start simple:

- No service mesh (too complex for learning)
- Direct HTTP calls between services (not message queues) for synchronous operations
- Redis for async task management
- Docker Compose for local development, Kubernetes for deployment

## Best Practices

- ✅ **Define clear service boundaries**: Each service owns a single AI capability and its data
- ✅ **Use shared libraries for common concerns**: LLM clients, Pydantic schemas, logging config — shared via a Python package
- ✅ **Standardize service structure**: Every service follows the same directory layout, has the same health check endpoint, same logging format
- ✅ **Contract-first API design**: Define OpenAPI specs before implementing — ensures services can integrate cleanly
- ✅ **Implement health checks**: Every service exposes `/health` with dependency status (can reach DB? Can reach LLM?)
- ❌ **Don't share databases**: Each service owns its data. If services need each other's data, they call each other's APIs.
- ❌ **Don't create a distributed monolith**: If every request needs to call 5 services synchronously, you haven't gained anything. Design for independence.
- ❌ **Don't over-decompose**: 4-6 services is right for this platform. Don't split into 20 micro-micro-services.

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Distributed monolith | Services too tightly coupled | Design for independent operation; async where possible |
| Data consistency issues | No shared DB, eventual consistency | Accept eventual consistency for AI systems (it's usually fine) |
| Service sprawl | Over-decomposition | Start with fewer, larger services; split when there's a clear reason |
| Debugging nightmare | Requests span multiple services | Implement distributed tracing (OpenTelemetry) from day 1 |
| Cascading failures | One service down → everything down | Circuit breakers, timeouts, fallback responses |

## Application to My Project

### Service Decomposition

| Service | Source | Data Store | Scaling Profile |
|---------|--------|------------|-----------------|
| `knowledge-service` | Phase 2 | Vector DB (Qdrant) | High read throughput, low latency |
| `agent-service` | Phase 3 | State store (Redis/Postgres) | Medium throughput, stateful |
| `research-service` | Phase 4 | Task queue (Redis) | Low throughput, long-running |
| `assistant-service` | Phase 5 | File store (Azure Blob) | Variable, handles large payloads |
| `gateway-service` | New | Config store | High throughput, stateless |
| `platform-service` | New | Postgres | Low traffic, admin/metrics |

### Shared Infrastructure

- **LLM Client Pool**: Shared library, same Azure OpenAI endpoint
- **Observability**: OpenTelemetry collector, all services emit traces/metrics
- **Config**: Environment variables + shared config service
- **Auth**: JWT validation library shared across services

### Decisions to Make
- [ ] HTTP vs gRPC for inter-service communication (recommend HTTP for simplicity)
- [ ] Redis vs RabbitMQ for async tasks (recommend Redis — simpler, multi-purpose)
- [ ] Shared Python package structure (monorepo with packages, or separate repos?)
- [ ] How much to refactor existing Phase 2-5 projects vs wrapping them

## Resources for Deeper Learning

- [Microsoft: Microservices architecture](https://learn.microsoft.com/en-us/azure/architecture/guide/architecture-styles/microservices) — Comprehensive guide
- [Sam Newman: Building Microservices](https://samnewman.io/books/building_microservices_2nd_edition/) — The definitive book
- [Martin Fowler: Microservices](https://martinfowler.com/articles/microservices.html) — Original article defining the pattern
- [12-Factor App](https://12factor.net/) — Principles for cloud-native services
- [Microsoft: Strangler Fig pattern](https://learn.microsoft.com/en-us/azure/architecture/patterns/strangler-fig) — For integrating existing projects

## Questions Remaining

- [ ] Should the platform use a monorepo (all services in one git repo) or polyrepo?
- [ ] How to handle shared schema evolution (e.g., LLM response format changes)?
- [ ] What's the right level of service autonomy vs. consistency enforcement?
