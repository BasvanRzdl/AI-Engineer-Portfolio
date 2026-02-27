---
date: 2026-02-27
type: technology
topic: "Observability Stack: Prometheus, Grafana, and OpenTelemetry"
project: "Phase 6 - Enterprise AI Platform"
status: complete
decision: use
---

# Technology Brief: Observability Stack

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | A combination of tools for metrics, logging, tracing, and dashboards |
| **For** | Monitoring AI platform health, performance, cost, and output quality |
| **Maturity** | All tools are production-grade and widely adopted |
| **License** | All open-source (Apache 2.0) |
| **Decision** | **Use** — Prometheus + Grafana + OpenTelemetry as the core stack |

## The Stack Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVABILITY STACK                            │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ OpenTelemetry│   │ Prometheus   │   │ Grafana            │  │
│  │              │   │              │   │                    │  │
│  │ Instrument   │──►│ Collect &    │──►│ Visualize &        │  │
│  │ your code    │   │ Store metrics│   │ Alert              │  │
│  │              │   │              │   │                    │  │
│  │ Traces +     │   │ Time-series  │   │ Dashboards +       │  │
│  │ Metrics +    │   │ database     │   │ Alerting           │  │
│  │ Logs         │   │              │   │                    │  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
│                                                                  │
│  Optional additions:                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │ Loki         │   │ Tempo/Jaeger │   │ Azure Monitor      │  │
│  │ Log          │   │ Trace        │   │ Cloud-native       │  │
│  │ aggregation  │   │ storage      │   │ alternative        │  │
│  └──────────────┘   └──────────────┘   └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Component 1: OpenTelemetry (OTel)

### What It Is

OpenTelemetry is a vendor-neutral standard for instrumenting your code to produce telemetry data (traces, metrics, logs). It doesn't store or visualize data — it collects and exports it to backends like Prometheus, Jaeger, or Azure Monitor.

**Think of it as**: The universal adapter. You instrument once with OTel, and it can send data to any backend.

### Core Concepts

- **Trace**: The full journey of a request through your system. Made up of spans.
- **Span**: A single operation within a trace (e.g., "LLM call", "database query"). Has start/end time, attributes.
- **Metric**: A numerical measurement over time (counter, histogram, gauge).
- **Context Propagation**: Automatically passing trace IDs between services so spans link together.

### Setup in FastAPI

```python
# observability/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

def setup_tracing(app, service_name: str):
    """Configure OpenTelemetry tracing for a FastAPI app."""
    
    # Create tracer provider
    provider = TracerProvider(
        resource=Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "dev"),
        })
    )
    
    # Export spans to an OTLP collector (Jaeger, Tempo, etc.)
    exporter = OTLPSpanExporter(
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    
    # Auto-instrument FastAPI (adds spans for every request)
    FastAPIInstrumentor.instrument_app(app)
    
    # Auto-instrument outgoing HTTP calls (httpx)
    HTTPXClientInstrumentor().instrument()

# Usage: custom spans for AI operations
tracer = trace.get_tracer("ai-platform")

async def search_knowledge(query: str):
    with tracer.start_as_current_span("knowledge.search") as span:
        span.set_attribute("query.length", len(query))
        
        # Embedding generation
        with tracer.start_as_current_span("knowledge.embed") as embed_span:
            embedding = await generate_embedding(query)
            embed_span.set_attribute("embedding.dimensions", len(embedding))
        
        # Vector search
        with tracer.start_as_current_span("knowledge.vector_search") as search_span:
            results = await vector_db.search(embedding, limit=5)
            search_span.set_attribute("results.count", len(results))
        
        # LLM generation
        with tracer.start_as_current_span("knowledge.generate") as gen_span:
            answer = await llm.complete(...)
            gen_span.set_attribute("tokens.total", answer.usage.total_tokens)
            gen_span.set_attribute("cost.usd", float(answer.cost))
        
        return answer
```

### Key Python Packages

```
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-instrumentation-fastapi
opentelemetry-instrumentation-httpx
opentelemetry-instrumentation-redis
opentelemetry-instrumentation-sqlalchemy
```

## Component 2: Prometheus

### What It Is

Prometheus is a time-series database designed for monitoring. It **pulls** (scrapes) metrics from your services at regular intervals and stores them for querying.

**Think of it as**: A database that specializes in numbers that change over time. "What was the CPU usage 5 minutes ago? What's the request rate trend?"

### Core Concepts

- **Scraping**: Prometheus periodically hits your `/metrics` endpoint and collects data
- **Time Series**: Data stored as `metric_name{labels} value timestamp`
- **PromQL**: Prometheus Query Language for querying and aggregating metrics
- **Alert Rules**: Define conditions that trigger alerts

### Metric Types

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROMETHEUS METRIC TYPES                        │
│                                                                  │
│  COUNTER: Only goes up. Resets on restart.                      │
│  Example: Total requests served, total tokens used              │
│  ai_platform_requests_total{service="gateway"} 15432            │
│                                                                  │
│  GAUGE: Goes up and down. Current value.                        │
│  Example: Active connections, queue depth                       │
│  ai_platform_active_requests{service="gateway"} 12              │
│                                                                  │
│  HISTOGRAM: Distribution of values. Buckets.                    │
│  Example: Request latency, token count distribution             │
│  ai_platform_request_duration_seconds_bucket{le="0.5"} 2341    │
│  ai_platform_request_duration_seconds_bucket{le="1.0"} 3456    │
│  ai_platform_request_duration_seconds_bucket{le="5.0"} 3890    │
│                                                                  │
│  SUMMARY: Similar to histogram but calculates quantiles.        │
│  Example: p50, p90, p99 latency                                │
└─────────────────────────────────────────────────────────────────┘
```

### Exposing Metrics from FastAPI

```python
# observability/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
REQUEST_COUNT = Counter(
    "ai_platform_requests_total",
    "Total requests",
    ["service", "endpoint", "method", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "ai_platform_request_duration_seconds",
    "Request latency in seconds",
    ["service", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)

LLM_TOKENS = Counter(
    "ai_platform_llm_tokens_total",
    "Total LLM tokens used",
    ["service", "model", "token_type"]  # token_type: prompt/completion
)

LLM_COST = Counter(
    "ai_platform_llm_cost_usd_total",
    "Total LLM cost in USD",
    ["service", "model", "client_id"]
)

LLM_LATENCY = Histogram(
    "ai_platform_llm_call_duration_seconds",
    "LLM API call latency",
    ["service", "model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
)

ACTIVE_REQUESTS = Gauge(
    "ai_platform_active_requests",
    "Currently active requests",
    ["service"]
)

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### Prometheus Configuration

```yaml
# observability/prometheus.yml
global:
  scrape_interval: 15s      # How often to scrape metrics
  evaluation_interval: 15s  # How often to evaluate alert rules

scrape_configs:
  - job_name: "gateway"
    static_configs:
      - targets: ["gateway-service:8000"]
    metrics_path: "/metrics"

  - job_name: "knowledge-service"
    static_configs:
      - targets: ["knowledge-service:8000"]
    metrics_path: "/metrics"

  - job_name: "agent-service"
    static_configs:
      - targets: ["agent-service:8000"]
    metrics_path: "/metrics"

  - job_name: "research-service"
    static_configs:
      - targets: ["research-service:8000"]
    metrics_path: "/metrics"

  - job_name: "assistant-service"
    static_configs:
      - targets: ["assistant-service:8000"]
    metrics_path: "/metrics"

# In Kubernetes, use service discovery instead of static targets:
# - job_name: 'kubernetes-pods'
#   kubernetes_sd_configs:
#     - role: pod
```

### Key PromQL Queries

```promql
# Request rate (requests per second) over last 5 minutes
rate(ai_platform_requests_total[5m])

# Average request latency per service
rate(ai_platform_request_duration_seconds_sum[5m]) 
/ rate(ai_platform_request_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(ai_platform_request_duration_seconds_bucket[5m]))

# Total tokens used per hour by model
increase(ai_platform_llm_tokens_total[1h])

# Total cost per day by client
increase(ai_platform_llm_cost_usd_total[24h])

# Error rate percentage
rate(ai_platform_requests_total{status_code=~"5.."}[5m])
/ rate(ai_platform_requests_total[5m]) * 100
```

## Component 3: Grafana

### What It Is

Grafana is a visualization platform for creating dashboards from data sources like Prometheus, Loki, and others.

**Think of it as**: The beautiful UI where you see all your metrics as graphs, tables, and alerts.

### Dashboard Design for AI Platform

```
┌─────────────────────────────────────────────────────────────────┐
│                 AI PLATFORM OVERVIEW DASHBOARD                    │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ Request Rate    │  │ Error Rate      │  │ Avg Latency    │  │
│  │ 145 req/s       │  │ 0.3%            │  │ 2.1s           │  │
│  │ ▁▂▃▄▅▆▇▆▅▄     │  │ ▁▁▁▁▁▂▁▁▁▁     │  │ ▃▃▃▄▃▃▃▃▄▃   │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ Tokens/Hour     │  │ Cost Today      │  │ Active Users   │  │
│  │ 1.2M tokens     │  │ $34.56          │  │ 8              │  │
│  │ ▂▃▅▆▇▇▆▅▃▂     │  │ ▁▂▃▄▅▆▇ ↑      │  │ ▃▄▅▆▇▆▅▄▃▂   │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
│                                                                  │
│  ┌──────────── Service Health ──────────────────────────────┐   │
│  │ Gateway          ✅ UP   │ 45 req/s │ p99: 120ms       │   │
│  │ Knowledge        ✅ UP   │ 30 req/s │ p99: 3.2s        │   │
│  │ Agent            ✅ UP   │ 12 req/s │ p99: 8.5s        │   │
│  │ Research         ✅ UP   │  3 req/s │ p99: 45s         │   │
│  │ Assistant        ✅ UP   │  5 req/s │ p99: 5.1s        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────── Cost by Model (24h) ─────────────────────────┐  │
│  │ gpt-4o        ████████████████████  $28.50 (82%)         │  │
│  │ gpt-4o-mini   ████                  $4.20  (12%)         │  │
│  │ embeddings    ██                    $1.86   (6%)         │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Dashboard Configuration (JSON Model)

Grafana dashboards are configured as JSON and can be provisioned automatically:

```json
{
  "dashboard": {
    "title": "AI Platform Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(ai_platform_requests_total[5m]))",
            "legendFormat": "req/s"
          }
        ]
      },
      {
        "title": "LLM Cost (Today)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(ai_platform_llm_cost_usd_total[24h]))",
            "legendFormat": "USD"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        }
      },
      {
        "title": "Latency by Service",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(ai_platform_request_duration_seconds_bucket[5m])) by (service, le))",
            "legendFormat": "{{service}} p95"
          }
        ]
      }
    ]
  }
}
```

### Alerting

```yaml
# Grafana alert rules
alerts:
  - name: "High Error Rate"
    condition: rate(ai_platform_requests_total{status_code=~"5.."}[5m]) / rate(ai_platform_requests_total[5m]) > 0.05
    for: 5m
    severity: critical
    message: "Error rate above 5% for 5 minutes"

  - name: "Daily Budget Warning"
    condition: increase(ai_platform_llm_cost_usd_total[24h]) > 80
    severity: warning
    message: "Daily LLM cost approaching $100 budget"

  - name: "High LLM Latency"
    condition: histogram_quantile(0.95, rate(ai_platform_llm_call_duration_seconds_bucket[5m])) > 15
    for: 5m
    severity: warning
    message: "LLM p95 latency above 15 seconds"

  - name: "Service Down"
    condition: up{job=~".*-service"} == 0
    for: 1m
    severity: critical
    message: "Service {{ $labels.job }} is down"
```

## Structured Logging with structlog

While Prometheus handles metrics and OTel handles traces, you still need structured logging:

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
)

logger = structlog.get_logger()

# In request handlers
async def handle_search(query: str):
    logger.info(
        "search_started",
        query_length=len(query),
        service="knowledge",
    )
    
    try:
        results = await perform_search(query)
        logger.info(
            "search_completed",
            results_count=len(results),
            tokens_used=results.total_tokens,
            latency_ms=results.latency_ms,
            cost_usd=float(results.cost),
        )
        return results
    except Exception as e:
        logger.error(
            "search_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        raise
```

Output (JSON, machine-parsable):
```json
{
  "event": "search_completed",
  "level": "info",
  "timestamp": "2026-02-27T14:30:00Z",
  "service": "knowledge",
  "results_count": 5,
  "tokens_used": 1550,
  "latency_ms": 2340,
  "cost_usd": 0.0078,
  "request_id": "abc-123"
}
```

## Putting It All Together

### Docker Compose for Observability Stack

```yaml
# In docker-compose.yml, add these services:
services:
  # ... (application services) ...

  prometheus:
    image: prom/prometheus:v2.51.0
    ports:
      - "9090:9090"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:11.0.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./observability/grafana/provisioning:/etc/grafana/provisioning
      - ./observability/grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus

  # Optional: Jaeger for trace visualization
  jaeger:
    image: jaegertracing/all-in-one:1.57
    ports:
      - "16686:16686"   # Jaeger UI
      - "4317:4317"     # OTLP gRPC
      - "4318:4318"     # OTLP HTTP

volumes:
  prometheus-data:
  grafana-data:
```

## Trade-offs

### Prometheus + Grafana vs. Azure Monitor

| Feature | Prometheus + Grafana | Azure Monitor |
|---------|---------------------|---------------|
| **Cost** | Free (self-hosted) | Pay per GB ingested |
| **Setup** | Manual | Integrated with AKS |
| **Flexibility** | Unlimited custom metrics | Pre-defined + custom |
| **Dashboards** | Highly customizable | Good but less flexible |
| **Alerting** | Alertmanager | Azure Alerts |
| **Retention** | Self-managed | Managed |
| **AI-specific** | Custom metrics needed | Container Insights |
| **Learning value** | High (industry standard) | Medium |

### Recommendation for Phase 6

**Use both:**
- **Prometheus + Grafana** for custom AI metrics (tokens, cost, quality) — more control and learning
- **Azure Monitor** for infrastructure metrics (node health, container restarts) — it's free with AKS
- **OpenTelemetry** as the instrumentation layer — vendor-neutral, sends to any backend

## Decision

**Recommendation**: **Use** — Prometheus + Grafana + OpenTelemetry as the primary stack, with Azure Monitor for infrastructure.

**Next steps**:
1. Add OpenTelemetry instrumentation to all services
2. Define custom Prometheus metrics for AI operations
3. Set up Prometheus and Grafana in Docker Compose
4. Create initial dashboards (overview, cost, per-service)
5. Configure alerting rules

## Resources for Deeper Learning

- [OpenTelemetry Python docs](https://opentelemetry.io/docs/languages/python/) — Instrumentation guide
- [Prometheus docs](https://prometheus.io/docs/) — Metrics collection
- [PromQL tutorial](https://prometheus.io/docs/prometheus/latest/querying/basics/) — Query language
- [Grafana Dashboard best practices](https://grafana.com/docs/grafana/latest/dashboards/build-dashboards/best-practices/) — Design guidance
- [structlog documentation](https://www.structlog.org/) — Structured logging
