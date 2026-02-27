---
date: 2026-02-27
type: architecture
topic: "Enterprise AI Platform — Reference Architecture"
project: "Phase 6 - Enterprise AI Platform"
status: complete
---

# Enterprise AI Platform — Reference Architecture

## Purpose

This document synthesizes all research from the Phase 6 concept and technology documents into a unified, actionable architecture for the Enterprise AI Platform. This is the blueprint for implementation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE AI PLATFORM                                │
│                                                                              │
│  ┌─── External Layer ──────────────────────────────────────────────────┐    │
│  │  HTTPS/TLS │ Azure Front Door (optional) │ DNS                      │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                           │
│  ┌─── AKS Cluster ─────────────┼──────────────────────────────────────┐    │
│  │                              │                                      │    │
│  │  ┌─ NGINX Ingress ──────────┘                                      │    │
│  │  │                                                                  │    │
│  │  ├── /api/v1/* ──────────────────────────────────────────────┐     │    │
│  │  │                                                            │     │    │
│  │  │  ┌─────────────────────────────────────────────────────┐  │     │    │
│  │  │  │              API GATEWAY SERVICE                     │  │     │    │
│  │  │  │                                                      │  │     │    │
│  │  │  │  ┌────────────────── Middleware Pipeline ──────────┐│  │     │    │
│  │  │  │  │                                                  ││  │     │    │
│  │  │  │  │  Request ID → Auth → Rate Limit → Prompt       ││  │     │    │
│  │  │  │  │  Injection Check → Cost Tracking → Logging     ││  │     │    │
│  │  │  │  │                                                  ││  │     │    │
│  │  │  │  └──────────────────────────────────────────────────┘│  │     │    │
│  │  │  │                                                      │  │     │    │
│  │  │  │  ┌─ Router ────────────────────────────────────────┐│  │     │    │
│  │  │  │  │ /knowledge/* → Knowledge Service                ││  │     │    │
│  │  │  │  │ /agent/*     → Agent Service                    ││  │     │    │
│  │  │  │  │ /research/*  → Research Service                 ││  │     │    │
│  │  │  │  │ /assistant/* → Assistant Service                ││  │     │    │
│  │  │  │  │ /admin/*     → Platform Admin                   ││  │     │    │
│  │  │  │  └─────────────────────────────────────────────────┘│  │     │    │
│  │  │  └─────────────────────────────────────────────────────┘  │     │    │
│  │  │                                                            │     │    │
│  │  │         ┌──────────┬──────────┬──────────┬──────────┐     │     │    │
│  │  │         │          │          │          │          │     │     │    │
│  │  │    ┌────▼────┐ ┌───▼────┐ ┌──▼──────┐ ┌▼────────┐ │     │     │    │
│  │  │    │Knowledge│ │ Agent  │ │Research │ │Assistant│ │     │     │    │
│  │  │    │Service  │ │Service │ │Service  │ │Service  │ │     │     │    │
│  │  │    │(Phase 2)│ │(Ph. 3) │ │(Phase 4)│ │(Ph. 5)  │ │     │     │    │
│  │  │    └────┬────┘ └───┬────┘ └────┬────┘ └────┬────┘ │     │     │    │
│  │  │         │          │           │           │      │     │     │    │
│  │  └─────────┼──────────┼───────────┼───────────┼──────┘     │     │    │
│  │            │          │           │           │             │     │    │
│  │  ┌─── Infrastructure ─┼───────────┼───────────┼──────────┐ │     │    │
│  │  │    ┌────▼────┐  ┌──▼──┐   ┌───▼──┐   ┌───▼──────┐   │ │     │    │
│  │  │    │ Qdrant  │  │Redis│   │Redis │   │ Blob     │   │ │     │    │
│  │  │    │ Vector  │  │State│   │Tasks │   │ Storage  │   │ │     │    │
│  │  │    │ DB      │  │Store│   │Queue │   │          │   │ │     │    │
│  │  │    └─────────┘  └─────┘   └──────┘   └──────────┘   │ │     │    │
│  │  │                                                       │ │     │    │
│  │  │    ┌──────────┐  ┌──────────┐                        │ │     │    │
│  │  │    │Prometheus│  │ Grafana  │                        │ │     │    │
│  │  │    │(Metrics) │  │(Dashbrd) │                        │ │     │    │
│  │  │    └──────────┘  └──────────┘                        │ │     │    │
│  │  └──────────────────────────────────────────────────────┘ │     │    │
│  └───────────────────────────────────────────────────────────┘     │    │
│                                                                     │    │
│  ┌─── Azure Managed Services ──────────────────────────────────┐   │    │
│  │  ┌──────────────┐  ┌────────────┐  ┌─────────────────────┐ │   │    │
│  │  │ Azure OpenAI │  │ PostgreSQL │  │ Azure Key Vault     │ │   │    │
│  │  │ (GPT-4o,     │  │ Flexible   │  │ (Secrets)           │ │   │    │
│  │  │  embeddings) │  │ Server     │  │                     │ │   │    │
│  │  └──────────────┘  └────────────┘  └─────────────────────┘ │   │    │
│  │  ┌──────────────┐  ┌────────────┐                          │   │    │
│  │  │ Azure Cache  │  │ Azure      │                          │   │    │
│  │  │ for Redis    │  │ Container  │                          │   │    │
│  │  │ (Prod)       │  │ Registry   │                          │   │    │
│  │  └──────────────┘  └────────────┘                          │   │    │
│  └────────────────────────────────────────────────────────────┘   │    │
└─────────────────────────────────────────────────────────────────────┘    │
```

## Service Definitions

### 1. API Gateway Service (New — Phase 6)

| Attribute | Value |
|-----------|-------|
| **Port** | 8000 |
| **Framework** | FastAPI |
| **Replicas** | 2-5 |
| **Responsibilities** | Auth, rate limiting, routing, cost tracking, security, logging |
| **Data store** | None (stateless) — uses Redis for rate limits, PostgreSQL for audit |
| **Endpoints** | Routes to all backend services |

### 2. Knowledge Base Service (Phase 2)

| Attribute | Value |
|-----------|-------|
| **Port** | 8000 |
| **Framework** | FastAPI |
| **Replicas** | 2-4 |
| **Responsibilities** | Document ingestion, embedding, vector search, RAG generation |
| **Data store** | Qdrant (vector DB) |
| **Key endpoints** | `POST /search`, `POST /ingest`, `GET /documents` |

### 3. Customer Agent Service (Phase 3)

| Attribute | Value |
|-----------|-------|
| **Port** | 8000 |
| **Framework** | FastAPI + LangGraph |
| **Replicas** | 2-3 |
| **Responsibilities** | Conversational agent, state management, tool execution |
| **Data store** | Redis (conversation state) |
| **Key endpoints** | `POST /chat`, `GET /sessions/{id}`, `POST /chat/stream` |

### 4. Research Team Service (Phase 4)

| Attribute | Value |
|-----------|-------|
| **Port** | 8000 |
| **Framework** | FastAPI + LangGraph (multi-agent) |
| **Replicas** | 1-2 |
| **Responsibilities** | Multi-agent research orchestration, task decomposition |
| **Data store** | Redis (task queue + state) |
| **Key endpoints** | `POST /start`, `GET /status/{id}`, `GET /results/{id}` |

### 5. Multi-Modal Assistant Service (Phase 5)

| Attribute | Value |
|-----------|-------|
| **Port** | 8000 |
| **Framework** | FastAPI |
| **Replicas** | 1-3 |
| **Responsibilities** | Image analysis, audio processing, multi-modal reasoning |
| **Data store** | Azure Blob Storage (file uploads) |
| **Key endpoints** | `POST /analyze`, `POST /transcribe`, `POST /describe-image` |

### 6. Platform Admin Service (New — Phase 6)

| Attribute | Value |
|-----------|-------|
| **Port** | 8000 |
| **Framework** | FastAPI |
| **Replicas** | 1 |
| **Responsibilities** | User/key management, cost reports, platform configuration |
| **Data store** | PostgreSQL |
| **Key endpoints** | `GET /usage`, `POST /keys`, `GET /reports` |

## API Design

### URL Structure

```
Base URL: https://ai-platform.example.com/api/v1

Knowledge Base:
  POST   /api/v1/knowledge/search           Search documents
  POST   /api/v1/knowledge/ingest           Ingest new document
  GET    /api/v1/knowledge/documents         List documents
  DELETE /api/v1/knowledge/documents/{id}    Delete document

Customer Agent:
  POST   /api/v1/agent/chat                  Send message
  POST   /api/v1/agent/chat/stream           Stream response (SSE)
  GET    /api/v1/agent/sessions              List sessions
  GET    /api/v1/agent/sessions/{id}         Get session
  DELETE /api/v1/agent/sessions/{id}         End session

Research Team:
  POST   /api/v1/research/start              Start research task
  GET    /api/v1/research/tasks              List tasks
  GET    /api/v1/research/tasks/{id}         Get task status
  GET    /api/v1/research/tasks/{id}/results Get results

Multi-Modal Assistant:
  POST   /api/v1/assistant/analyze           Analyze image/document
  POST   /api/v1/assistant/transcribe        Transcribe audio

Platform:
  GET    /api/v1/health                       Platform health
  GET    /api/v1/metrics                      Prometheus metrics

Admin (requires admin role):
  GET    /api/v1/admin/usage                  Usage summary
  GET    /api/v1/admin/usage/{client_id}      Client usage
  POST   /api/v1/admin/keys                   Create API key
  DELETE /api/v1/admin/keys/{key_id}          Revoke key
  GET    /api/v1/admin/reports/cost            Cost report
```

### Standard Response Envelope

```json
{
  "status": "success | error",
  "data": { },
  "error": {
    "code": "error_code",
    "message": "Human-readable message"
  },
  "metadata": {
    "request_id": "uuid",
    "timestamp": "2026-02-27T14:30:00Z",
    "processing_time_ms": 2340,
    "tokens_used": 1550,
    "estimated_cost_usd": 0.0078,
    "model": "gpt-4o",
    "prompt_version": "v2.1"
  }
}
```

## Project Repository Structure

```
ai-platform/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # Test + lint on PR
│       └── deploy.yml                # Build + deploy on merge to main
│
├── gateway/                           # API Gateway Service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app setup
│   │   ├── config.py                 # Pydantic Settings
│   │   ├── middleware/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py               # API key validation
│   │   │   ├── rate_limit.py         # Token + request rate limiting
│   │   │   ├── cost_tracking.py      # Cost calculation per request
│   │   │   ├── security.py           # Prompt injection detection
│   │   │   └── observability.py      # Request ID, logging, tracing
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── knowledge.py          # Proxy to knowledge service
│   │   │   ├── agent.py              # Proxy to agent service
│   │   │   ├── research.py           # Proxy to research service
│   │   │   ├── assistant.py          # Proxy to assistant service
│   │   │   ├── admin.py              # Admin endpoints
│   │   │   └── health.py             # Health checks
│   │   └── models/
│   │       ├── __init__.py
│   │       ├── requests.py           # Request schemas
│   │       └── responses.py          # Response schemas
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_auth.py
│   │   ├── test_rate_limit.py
│   │   ├── test_security.py
│   │   └── test_routes.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── services/
│   ├── knowledge/                     # Knowledge Base Service (Phase 2)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── config.py
│   │   │   ├── routes/
│   │   │   │   ├── search.py
│   │   │   │   └── ingest.py
│   │   │   └── services/
│   │   │       ├── retriever.py
│   │   │       ├── embedder.py
│   │   │       └── generator.py
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── agent/                         # Customer Agent Service (Phase 3)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   │   └── chat.py
│   │   │   └── services/
│   │   │       ├── agent.py          # LangGraph agent
│   │   │       ├── tools.py          # Agent tools
│   │   │       └── state.py          # State management
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── research/                      # Research Team Service (Phase 4)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   │   └── research.py
│   │   │   └── services/
│   │   │       ├── orchestrator.py   # Multi-agent coordinator
│   │   │       ├── agents/           # Individual research agents
│   │   │       └── tasks.py          # Task queue
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── assistant/                     # Multi-Modal Assistant (Phase 5)
│       ├── app/
│       │   ├── main.py
│       │   ├── routes/
│       │   │   └── analyze.py
│       │   └── services/
│       │       ├── vision.py
│       │       ├── audio.py
│       │       └── multimodal.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── shared/                            # Shared Python package
│   ├── __init__.py
│   ├── llm_client.py                 # Unified LLM client with cost tracking
│   ├── schemas.py                    # Shared Pydantic models
│   ├── logging_config.py            # Structured logging setup
│   ├── tracing.py                   # OpenTelemetry setup
│   ├── metrics.py                   # Prometheus metrics definitions
│   ├── cost.py                      # Cost calculation utilities
│   └── security.py                  # Prompt injection, PII redaction
│
├── observability/
│   ├── prometheus.yml                # Prometheus configuration
│   └── grafana/
│       ├── provisioning/
│       │   ├── datasources.yml       # Prometheus as data source
│       │   └── dashboards.yml        # Dashboard provisioning
│       └── dashboards/
│           ├── platform-overview.json
│           ├── cost-tracking.json
│           ├── service-health.json
│           └── ai-quality.json
│
├── kubernetes/
│   ├── namespace.yaml
│   ├── gateway/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── hpa.yaml
│   ├── knowledge/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── agent/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── research/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── assistant/
│   │   ├── deployment.yaml
│   │   └── service.yaml
│   ├── infrastructure/
│   │   ├── redis.yaml
│   │   ├── qdrant.yaml
│   │   └── postgres.yaml
│   ├── observability/
│   │   ├── prometheus.yaml
│   │   └── grafana.yaml
│   ├── ingress.yaml
│   └── secrets-provider.yaml
│
├── infrastructure/                    # IaC (Bicep)
│   ├── main.bicep                    # Core Azure resources
│   ├── aks.bicep                     # AKS cluster
│   ├── acr.bicep                     # Container registry
│   └── keyvault.bicep                # Key Vault
│
├── docs/                             # Documentation (deliverable)
│   ├── architecture-decisions.md     # ADRs
│   ├── api-documentation.md          # API reference
│   ├── deployment-guide.md           # Step-by-step deployment
│   └── runbook.md                    # Operational procedures
│
├── scripts/
│   ├── setup-azure.sh                # Create Azure resources
│   ├── deploy.sh                     # Deploy to AKS
│   └── evaluate-quality.py           # Quality evaluation script
│
├── docker-compose.yml                # Local development
├── docker-compose.override.yml       # Local overrides (dev settings)
├── .env.example                      # Example environment variables
├── .gitignore
├── pyproject.toml                    # Python project config
└── README.md                         # Project overview
```

## Security Architecture

```
Request Flow Through Security Layers:

Client Request
    │
    ▼
┌─ Layer 1: TLS ─────────────────────────────────────┐
│  NGINX Ingress terminates HTTPS                     │
└────────────────────────────┬────────────────────────┘
                             │
┌─ Layer 2: Authentication ──┼────────────────────────┐
│  X-API-Key header → Validate against key store      │
│  Return: client_id, role, rate_limits, budget        │
└────────────────────────────┬────────────────────────┘
                             │
┌─ Layer 3: Rate Limiting ───┼────────────────────────┐
│  Check: requests/min, tokens/day, budget remaining   │
│  Reject if exceeded (429 Too Many Requests)          │
└────────────────────────────┬────────────────────────┘
                             │
┌─ Layer 4: Input Security ──┼────────────────────────┐
│  Prompt injection detection (pattern + heuristic)    │
│  Input length validation                             │
│  Content safety check (optional)                     │
└────────────────────────────┬────────────────────────┘
                             │
┌─ Layer 5: Processing ──────┼────────────────────────┐
│  Backend service processes request                   │
│  Least-privilege tool access                         │
│  Data classification enforcement on retrieval        │
└────────────────────────────┬────────────────────────┘
                             │
┌─ Layer 6: Output Security ─┼────────────────────────┐
│  PII redaction (optional, per client config)         │
│  Output format validation                            │
│  Cost tracking and metadata attachment               │
└────────────────────────────┬────────────────────────┘
                             │
                             ▼
                      Client Response
```

## Observability Architecture

### Metrics Flow

```
Services → Prometheus → Grafana → Alerts → Slack/Email

Key Dashboards:
1. Platform Overview   — Request rate, error rate, latency, cost
2. Cost Tracking       — Per client, per service, per model, trends
3. Service Health      — Per-service metrics, pod status, resource usage
4. AI Quality          — Output quality scores, retrieval relevance
5. Client Usage        — Per-client request volume, quota utilization
```

### Logging Flow

```
Services (structlog/JSON) → stdout → Container Runtime → Log Collector → Storage

Log Levels:
- ERROR: Failures, exceptions, critical issues
- WARNING: Approaching limits, degraded performance
- INFO: Request/response summaries, business events
- DEBUG: Detailed traces (dev/staging only)
```

### Tracing Flow

```
Client → Gateway → Service → LLM Call
   │         │         │         │
   └─────────┴─────────┴─────────┘
         All connected via trace_id
         Visualized in Jaeger/Tempo
```

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API Framework** | FastAPI | All service endpoints |
| **LLM** | Azure OpenAI (GPT-4o, GPT-4o-mini) | AI capabilities |
| **Embeddings** | Azure OpenAI (text-embedding-3-small) | Vector search |
| **Agent Framework** | LangGraph | Agent orchestration (Phase 3, 4) |
| **Vector DB** | Qdrant | Semantic search (Phase 2) |
| **Cache/Queue** | Redis | Rate limiting, state, task queue |
| **Database** | PostgreSQL | Cost tracking, user management |
| **File Storage** | Azure Blob Storage | Document/media storage |
| **Containerization** | Docker | Package services |
| **Orchestration** | Kubernetes (AKS) | Run and scale services |
| **Registry** | Azure Container Registry | Store images |
| **Secrets** | Azure Key Vault | Manage credentials |
| **Metrics** | Prometheus | Collect and store metrics |
| **Dashboards** | Grafana | Visualize metrics |
| **Tracing** | OpenTelemetry + Jaeger | Distributed tracing |
| **Logging** | structlog (JSON) | Structured logging |
| **CI/CD** | GitHub Actions | Build and deploy |
| **IaC** | Bicep | Azure infrastructure |
| **PII** | Microsoft Presidio | PII detection/redaction |
| **HTTP Client** | httpx | Inter-service communication |

## Implementation Order

```
Phase 6 Implementation Roadmap:

Week 1: Foundation
├── Day 1-2: Project setup, shared package, Docker Compose
├── Day 3-4: Gateway service (FastAPI skeleton + middleware)
├── Day 5: Authentication + rate limiting

Week 2: Services + Integration
├── Day 1-2: Wrap Phase 2-5 as microservices
├── Day 3: Inter-service communication
├── Day 4: Cost tracking + budget management
├── Day 5: Observability (Prometheus + Grafana dashboards)

Week 3: Production Readiness
├── Day 1: Security (prompt injection, PII redaction)
├── Day 2: Kubernetes manifests + AKS deployment
├── Day 3: CI/CD pipeline
├── Day 4: Documentation (ADRs, API docs, deployment guide, runbook)
├── Day 5: Testing, polish, final deployment

Buffers: 2-3 days for unexpected issues
```

## Key Architectural Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | API Framework | FastAPI | Required by spec, best Python option for AI services |
| 2 | Service Communication | HTTP/REST (sync) + Redis (async) | Simple, debuggable, sufficient for scale |
| 3 | Gateway Pattern | Custom FastAPI gateway | Full control, learning value, AI-specific logic |
| 4 | Observability | OTel + Prometheus + Grafana | Industry standard, free, highly customizable |
| 5 | Deployment | AKS with rolling + canary | Required by spec, production-grade |
| 6 | Secrets | Azure Key Vault + CSI driver | Azure-native, secure, automatic rotation |
| 7 | Database | PostgreSQL (managed) | Reliable, supports JSONB for flexible schemas |
| 8 | Cache/Queue | Redis (managed) | Multi-purpose: caching, rate limiting, task queue |
| 9 | Repository Structure | Monorepo | Easier cross-service changes, shared code, unified CI |
| 10 | Auth Model | API keys with RBAC | Simple for v1, upgradeable to OAuth2/JWT |

## Constraints Validation

| Constraint | How We Meet It |
|-----------|----------------|
| **Deployable from docs** | Deployment guide with step-by-step instructions, IaC scripts |
| **10 concurrent users** | 2+ replicas per service, async FastAPI, HPA for scaling |
| **Cost trackable** | Per-request cost calculation, Grafana dashboards, admin API |
| **2 agentic frameworks** | LangGraph (Phase 3 agent, Phase 4 research) + Microsoft Agent Framework or custom |
| **Docker** | Every service containerized with Dockerfile |
| **Kubernetes** | Full K8s manifests, AKS deployment |
| **FastAPI** | All services built with FastAPI |

## Open Questions for Implementation

- [ ] How much of Phase 2-5 to refactor vs. wrap as-is?
- [ ] Exact agentic framework #2 (alongside LangGraph): Microsoft Semantic Kernel? AutoGen?
- [ ] Qdrant: run in K8s or use Qdrant Cloud?
- [ ] PostgreSQL/Redis: managed Azure services vs. run in K8s? (Recommend managed for production, K8s for dev)
- [ ] Frontend: any UI, or API-only? (Spec doesn't require a frontend, but Grafana serves as the "dashboard")

## Next Steps

1. **Set up the repository** with the structure above
2. **Build the shared package** (LLM client, schemas, logging, metrics)
3. **Build the gateway** (FastAPI + full middleware pipeline)
4. **Integrate Phase 2-5** as backend services
5. **Add observability** (Prometheus + Grafana)
6. **Deploy to AKS** with full CI/CD
7. **Write documentation** (architecture, API, deployment, runbook)
