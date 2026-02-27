---
date: 2026-02-27
type: concept
topic: "Observability for LLM Systems"
project: "Phase 6 - Enterprise AI Platform"
status: complete
---

# Observability for LLM Systems

## In My Own Words

Observability is the ability to understand what's happening inside your system by examining its external outputs — logs, metrics, and traces. For traditional systems, this means tracking HTTP status codes, response times, and error rates. For LLM systems, observability is fundamentally harder and more important because:

1. **Non-determinism**: The same input can produce different outputs. You need to track *quality*, not just *success*.
2. **Cost**: Every LLM call costs money. You need to track token usage and spend in real-time.
3. **Latency variability**: LLM responses can range from 500ms to 30+ seconds. Traditional p99 thresholds don't work the same way.
4. **Quality drift**: Model behavior changes over time (model updates, prompt drift). You need to monitor output quality continuously.
5. **Complex pipelines**: A single user query might trigger RAG retrieval, multiple agent steps, and several LLM calls. You need to trace the entire chain.

## Why This Matters

Phase 6 requires building an observability stack that covers:
- Centralized logging
- Tracing across service boundaries
- Metrics dashboard (latency, cost, usage)
- Quality monitoring (output quality over time)

Without observability, you're flying blind. You won't know which service is slow, which client is expensive, when quality degrades, or why a particular request failed.

## Core Principles

1. **The Three Pillars**: Logs (what happened), Metrics (how much/how often), Traces (the journey of a request). All three are needed — each answers different questions.

2. **Structured Everything**: Unstructured logs are useless at scale. Every log, metric, and trace should be structured (JSON), tagged with request ID, service name, client ID, and model used.

3. **Correlation**: A single user request must be traceable across all services. This requires a correlation ID (trace ID) propagated in headers.

4. **AI-Specific Dimensions**: Traditional observability tracks latency and errors. LLM observability adds: token usage, cost, model version, prompt template version, output quality score.

5. **Alerting on Quality, Not Just Availability**: A 200 OK response that contains hallucinated garbage is worse than a 500 error. Monitor quality metrics alongside traditional SRE metrics.

## How It Works

### The Three Pillars + AI Extensions

```
┌────────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY FOR AI SYSTEMS                     │
├──────────────────┬──────────────────┬──────────────────────────────┤
│     LOGS         │    METRICS       │     TRACES                   │
│                  │                  │                               │
│ What happened?   │ How much?        │ What was the journey?         │
│                  │ How often?       │                               │
│ • Request/       │ • Latency (p50,  │ • Request flow across         │
│   response body  │   p95, p99)      │   services                   │
│ • Errors and     │ • Token usage    │ • Time in each service        │
│   stack traces   │   per model      │ • LLM call spans             │
│ • LLM prompts    │ • Requests/sec   │ • Retrieval spans            │
│   and completions│ • Cost per hour/ │ • Queue wait time            │
│ • Retrieval      │   client/service │ • Parent-child               │
│   context chunks │ • Error rate     │   relationships              │
│ • Agent decisions│ • Cache hit rate │                               │
│                  │ • Quality scores │                               │
├──────────────────┴──────────────────┴──────────────────────────────┤
│                     AI-SPECIFIC EXTENSIONS                         │
│                                                                    │
│ • Token tracking per request, client, and model                    │
│ • Cost allocation and reporting                                    │
│ • Prompt/completion logging (with PII redaction)                   │
│ • Quality scoring (relevance, groundedness, coherence)             │
│ • Model version tracking                                          │
│ • Hallucination detection signals                                  │
│ • Retrieval quality (recall, precision of chunks)                  │
└────────────────────────────────────────────────────────────────────┘
```

### Structured Log Example

```json
{
  "timestamp": "2026-02-27T14:32:01.456Z",
  "level": "INFO",
  "service": "knowledge-service",
  "trace_id": "abc-123-def-456",
  "span_id": "span-789",
  "request_id": "req-001",
  "client_id": "client-acme",
  "event": "llm_completion",
  "data": {
    "model": "gpt-4o",
    "prompt_template": "rag_search_v2.1",
    "prompt_tokens": 1200,
    "completion_tokens": 350,
    "total_tokens": 1550,
    "latency_ms": 2340,
    "estimated_cost_usd": 0.0078,
    "temperature": 0.1,
    "retrieval_chunks": 5,
    "retrieval_score_avg": 0.87
  }
}
```

### Distributed Trace Example

```
Trace: abc-123-def-456  (Total: 3.2s)
│
├─ [gateway] POST /api/v1/knowledge/search  (3.2s)
│  ├─ [gateway] auth_middleware              (5ms)
│  ├─ [gateway] rate_limit_check             (2ms)
│  │
│  ├─ [knowledge-service] search_handler     (3.1s)
│  │  ├─ [knowledge] generate_embedding      (120ms)
│  │  │  └─ [openai] embedding API call      (115ms)
│  │  │
│  │  ├─ [knowledge] vector_search           (45ms)
│  │  │  └─ [qdrant] search query            (40ms)
│  │  │
│  │  ├─ [knowledge] rerank_results          (200ms)
│  │  │
│  │  └─ [knowledge] generate_answer         (2.7s)
│  │     └─ [openai] chat completion         (2.65s)
│  │
│  └─ [gateway] cost_tracking                (3ms)
```

This trace tells you immediately: the 3.2s response time is dominated by the LLM completion (2.65s). If you want to optimize, focus there (caching, smaller model, streaming).

## Key Metrics for LLM Systems

### Operational Metrics (SRE)

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `request_latency_seconds` | End-to-end request latency | p99 > 30s |
| `request_rate` | Requests per second per service | Varies |
| `error_rate` | Percentage of 4xx/5xx responses | > 5% |
| `service_health` | Health check status | Any service down |
| `queue_depth` | Pending async tasks | > 100 |

### AI-Specific Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `llm_tokens_total` | Tokens used (by model, service, client) | Budget exceeded |
| `llm_cost_usd` | Estimated cost per request/hour/day | Daily budget > 80% |
| `llm_latency_seconds` | LLM API call latency specifically | p95 > 10s |
| `retrieval_relevance_score` | Average relevance of retrieved chunks | < 0.7 |
| `output_quality_score` | LLM output quality (automated eval) | < 0.6 |
| `cache_hit_rate` | Percentage of requests served from cache | < 20% (if caching enabled) |
| `prompt_template_version` | Which prompt version is being used | Unexpected changes |
| `token_budget_remaining` | Remaining tokens in client quota | < 10% |

### Quality Metrics (AI-Specific)

```
Quality Monitoring Pipeline:

Request → LLM Response → Quality Evaluator → Score → Metrics Store
                              │
                              ├── Relevance: Is the answer relevant to the question?
                              ├── Groundedness: Is the answer supported by context?
                              ├── Coherence: Is the answer well-structured?
                              ├── Completeness: Does it address all parts?
                              └── Safety: Any harmful/biased content?
```

For automated quality evaluation, you can:
1. **LLM-as-judge**: Use a separate LLM call to score output quality (expensive but effective)
2. **Heuristic checks**: Length, format compliance, keyword presence
3. **Embedding similarity**: Compare output embedding to expected answer embedding
4. **Sampling**: Evaluate a random subset of responses (e.g., 5%) to control costs

## Approaches & Trade-offs

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **OpenTelemetry (OTel)** | Universal standard, any backend | Vendor-agnostic, rich ecosystem, traces+metrics+logs | Setup complexity, learning curve |
| **Azure Monitor + App Insights** | Azure-native, integrated | Managed, auto-instrumentation, good dashboards | Vendor lock-in, cost at scale, less AI-specific |
| **Prometheus + Grafana** | Self-hosted, full control | Free, powerful queries (PromQL), excellent dashboards | Must host yourself, no built-in tracing |
| **LangSmith/LangFuse** | LLM-specific observability | Purpose-built for LLM apps, prompt tracking, evaluations | Additional tool, cost, may overlap with general observability |
| **Custom logging** | Simple, full control | Exactly what you need | Must build everything, no dashboards out of the box |

### Recommendation for Phase 6

**Layered approach:**

1. **OpenTelemetry** as the instrumentation standard (traces + metrics)
2. **Structured JSON logging** to stdout (collected by container runtime)
3. **Prometheus** for metrics collection + **Grafana** for dashboards
4. **Azure Monitor / Application Insights** as the production backend (since we're on Azure)
5. **Custom AI metrics middleware** in FastAPI for token/cost/quality tracking

## Implementation Patterns

### Pattern 1: OpenTelemetry Instrumentation

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer("ai-platform")

# Initialize metrics
meter = metrics.get_meter("ai-platform")
token_counter = meter.create_counter("llm.tokens.total", description="Total LLM tokens used")
cost_counter = meter.create_counter("llm.cost.usd", description="Estimated LLM cost in USD")
latency_histogram = meter.create_histogram("llm.latency.seconds", description="LLM call latency")
```

### Pattern 2: FastAPI Middleware for AI Observability

```python
from starlette.middleware.base import BaseHTTPMiddleware
import time
import uuid

class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Generate trace context
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        
        # Add to request state for downstream use
        request.state.request_id = request_id
        request.state.token_usage = {"prompt": 0, "completion": 0}
        request.state.cost_usd = 0.0
        
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Emit metrics
        latency_histogram.record(duration, {"service": "gateway", "path": request.url.path})
        token_counter.add(
            request.state.token_usage.get("total", 0),
            {"service": "gateway", "client": request.state.client_id}
        )
        
        # Add observability headers to response
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time-Ms"] = str(int(duration * 1000))
        response.headers["X-Tokens-Used"] = str(request.state.token_usage.get("total", 0))
        
        return response
```

### Pattern 3: Structured Logger

```python
import structlog
import logging

def configure_logging(service_name: str):
    """Configure structured JSON logging for a service."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    )
    
    # Bind service-level context
    structlog.contextvars.bind_contextvars(service=service_name)

# Usage in a request handler
logger = structlog.get_logger()

async def handle_search(query: str, request_id: str):
    logger.info("search_started", request_id=request_id, query_length=len(query))
    
    results = await perform_search(query)
    
    logger.info(
        "search_completed",
        request_id=request_id,
        results_count=len(results),
        tokens_used=results.tokens,
        latency_ms=results.latency_ms,
    )
```

## Best Practices

- ✅ **Instrument from day 1**: Adding observability later is painful. Start with structured logging and basic metrics from the first service.
- ✅ **Use correlation IDs everywhere**: Every log line, metric, and trace span must include the request/trace ID.
- ✅ **Track LLM costs in real-time**: Don't wait for your cloud bill. Calculate cost per request using known token prices.
- ✅ **Log prompts and completions**: Essential for debugging. But redact PII before storing.
- ✅ **Create dashboards early**: A Grafana dashboard showing request rate, latency, cost, and error rate should exist for every service.
- ✅ **Set up alerts for cost thresholds**: Get notified before you blow your budget.
- ❌ **Don't log raw PII**: Implement PII detection/redaction before logging prompts and responses.
- ❌ **Don't ignore quality metrics**: A system that's fast and cheap but produces bad output is useless.
- ❌ **Don't create metric explosion**: Be selective about label cardinality. Don't add per-user labels to every metric — it kills Prometheus.

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Log soup | Unstructured, inconsistent logging | Use structlog with enforced schema |
| Missing trace propagation | Forgot to pass trace context between services | Use OpenTelemetry context propagation (it's automatic with the SDK) |
| Cost surprise | No real-time cost tracking | Calculate and log cost per LLM call |
| Alert fatigue | Too many alerts, too sensitive | Start with few critical alerts, tune thresholds |
| Quality blind spot | Only monitoring SRE metrics, not AI quality | Implement automated quality evaluation (even if sampled) |
| PII in logs | Logging full prompts/responses | PII redaction layer before logging |

## Application to My Project

### Observability Architecture for Phase 6

```
┌────────────────────────────────────────────────────────────────┐
│                     ALL SERVICES                                │
│  (gateway, knowledge, agent, research, assistant)              │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │
│  │ Structured  │  │ OTel SDK    │  │ Prometheus Client   │    │
│  │ Logging     │  │ (Traces)    │  │ (Metrics)           │    │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘    │
└─────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                     │
          ▼                ▼                     ▼
   ┌──────────┐   ┌───────────────┐   ┌──────────────┐
   │ Log      │   │ OTel          │   │ Prometheus   │
   │ Collector│   │ Collector     │   │ Server       │
   └────┬─────┘   └───────┬───────┘   └──────┬───────┘
        │                 │                   │
        ▼                 ▼                   ▼
   ┌──────────────────────────────────────────────┐
   │              GRAFANA DASHBOARDS               │
   │                                               │
   │  ┌─────────┐ ┌─────────┐ ┌───────────────┐  │
   │  │ Service │ │ Cost &  │ │ AI Quality    │  │
   │  │ Health  │ │ Usage   │ │ Monitoring    │  │
   │  └─────────┘ └─────────┘ └───────────────┘  │
   └───────────────────────────────────────────────┘
```

### Key Dashboards to Build

1. **Platform Overview**: Request rate, error rate, latency across all services
2. **Cost Dashboard**: Token usage per service, per client, per model. Daily/weekly/monthly cost trends.
3. **Service Health**: Per-service latency, error rate, health check status
4. **AI Quality**: Output quality scores over time, retrieval relevance, hallucination rate
5. **Client Usage**: Per-client request volume, cost, quota utilization

### Decisions to Make
- [ ] OpenTelemetry vs Azure Application Insights SDK directly
- [ ] Where to store logs (Azure Log Analytics? Loki? Elasticsearch?)
- [ ] How to implement quality monitoring (LLM-as-judge per request, or sample?)
- [ ] Granularity of cost tracking (per-request? per-minute? per-client?)

## Resources for Deeper Learning

- [OpenTelemetry Python docs](https://opentelemetry.io/docs/languages/python/) — Instrumentation setup
- [Prometheus docs](https://prometheus.io/docs/) — Metrics collection
- [Grafana docs](https://grafana.com/docs/) — Dashboard building
- [structlog docs](https://www.structlog.org/) — Structured logging for Python
- [Microsoft: Observability patterns](https://learn.microsoft.com/en-us/azure/architecture/microservices/logging-monitoring) — Azure-specific guidance
- [Hamel Husain: LLM Observability](https://hamel.dev/blog/posts/llmops/) — Practical LLM monitoring insights

## Questions Remaining

- [ ] How to handle log volume from prompt/completion logging without exploding storage costs?
- [ ] What's the right sampling rate for quality evaluation (cost vs. coverage)?
- [ ] How to detect quality drift automatically (statistical methods vs. threshold-based)?
