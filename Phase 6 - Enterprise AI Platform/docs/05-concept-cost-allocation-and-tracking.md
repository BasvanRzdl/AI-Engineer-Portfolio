---
date: 2026-02-27
type: concept
topic: "Cost Allocation and Tracking for AI Platforms"
project: "Phase 6 - Enterprise AI Platform"
status: complete
---

# Cost Allocation and Tracking for AI Platforms

## In My Own Words

Cost tracking in AI platforms is fundamentally different from traditional applications. In a standard web app, the cost of serving a request is roughly fixed — a few cents of compute. In an AI platform, costs are **variable and significant**: a single GPT-4 request can cost $0.01-$0.10, and a complex research task making 50 LLM calls might cost $2-5. Multiply that by thousands of users and you can blow through budgets in hours.

Cost tracking means: knowing exactly how much every request costs, which client/project is responsible, and being able to enforce budgets in real-time. Cost *allocation* means: attributing those costs to the right business unit, project, or customer so you can bill accurately and make informed decisions about which AI services are worth their cost.

## Why This Matters

Phase 6 requires:
- Cost tracking and allocation per client/project
- Rate limiting and quota management
- Cost must be trackable and reportable

Without this, an enterprise AI platform is a financial black hole. You need to answer questions like:
- "How much did Team X spend on AI this month?"
- "Which AI service is the most expensive?"
- "Are we within budget? If not, who's over?"
- "Is this AI feature worth its cost based on usage?"

## Core Principles

1. **Track at the Atomic Level**: Every LLM call should record its token usage and calculated cost. You can always aggregate up, but you can't decompose a lump sum.

2. **Real-Time Visibility**: Don't wait for the cloud bill. Calculate costs at request time based on known pricing. Enterprise clients need dashboards, not monthly surprises.

3. **Enforce, Don't Just Report**: Tracking without enforcement is just watching money disappear. Implement hard budget limits and automatic cutoffs.

4. **Multi-Dimensional Attribution**: Costs should be attributable by: client/tenant, project, service, model, and time period. Different stakeholders need different views.

5. **Include All Costs**: LLM tokens are the obvious cost, but also track: embedding generation, vector search queries, storage, compute, and egress.

## How It Works

### Cost Flow Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    COST TRACKING FLOW                              │
│                                                                   │
│  Request → Gateway → Service → LLM Call → Response               │
│     │                             │                               │
│     │                     ┌───────▼─────────┐                    │
│     │                     │  Token Counter   │                    │
│     │                     │  - prompt tokens  │                    │
│     │                     │  - completion     │                    │
│     │                     │  - model used     │                    │
│     │                     └───────┬──────────┘                    │
│     │                             │                               │
│     │                     ┌───────▼─────────┐                    │
│     │                     │ Cost Calculator  │                    │
│     │                     │ tokens × price   │                    │
│     │                     └───────┬──────────┘                    │
│     │                             │                               │
│     │              ┌──────────────▼──────────────┐               │
│     │              │     Cost Attribution         │               │
│     │              │  client_id + project_id      │               │
│     │              │  + service + model + time    │               │
│     │              └──────────────┬───────────────┘               │
│     │                             │                               │
│     │    ┌────────────────────────▼──────────────────────┐       │
│     │    │              COST STORE                        │       │
│     │    │  ┌──────────┐  ┌───────────┐  ┌────────────┐ │       │
│     │    │  │ Per-Request│ │ Aggregated│ │  Budget     │ │       │
│     │    │  │ Records   │ │ Summaries │ │  Tracking   │ │       │
│     │    │  └──────────┘  └───────────┘  └────────────┘ │       │
│     │    └───────────────────────────────────────────────┘       │
│     │                                                             │
│     └──── Response includes: cost_usd, tokens_used in metadata   │
└──────────────────────────────────────────────────────────────────┘
```

### Pricing Model Reference (as of early 2026)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Notes |
|-------|----------------------|------------------------|-------|
| GPT-4o | ~$2.50 | ~$10.00 | Primary model for complex tasks |
| GPT-4o-mini | ~$0.15 | ~$0.60 | Cost-effective for simpler tasks |
| text-embedding-3-small | ~$0.02 | N/A | For embeddings |
| text-embedding-3-large | ~$0.13 | N/A | Higher quality embeddings |

> **Important**: Prices change. Store pricing in a configuration file, not hardcoded.

### Cost Calculation

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class ModelPricing:
    """Token pricing for a model."""
    input_price_per_million: Decimal   # USD per 1M input tokens
    output_price_per_million: Decimal  # USD per 1M output tokens

# Configurable pricing table
MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        input_price_per_million=Decimal("2.50"),
        output_price_per_million=Decimal("10.00"),
    ),
    "gpt-4o-mini": ModelPricing(
        input_price_per_million=Decimal("0.15"),
        output_price_per_million=Decimal("0.60"),
    ),
    "text-embedding-3-small": ModelPricing(
        input_price_per_million=Decimal("0.02"),
        output_price_per_million=Decimal("0"),
    ),
}

def calculate_cost(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    """Calculate the cost of an LLM call."""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        raise ValueError(f"Unknown model: {model}")
    
    input_cost = (Decimal(prompt_tokens) / Decimal(1_000_000)) * pricing.input_price_per_million
    output_cost = (Decimal(completion_tokens) / Decimal(1_000_000)) * pricing.output_price_per_million
    
    return input_cost + output_cost
```

## Key Components

### 1. Token Tracking Middleware

Every LLM call goes through a wrapper that captures token usage:

```python
class CostTrackingLLMWrapper:
    """Wraps LLM client to automatically track costs."""
    
    async def complete(self, messages, model, **kwargs):
        response = await self.client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        
        # Extract token usage from response
        usage = response.usage
        cost = calculate_cost(
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        
        # Record cost event
        await self.cost_store.record(CostEvent(
            timestamp=datetime.utcnow(),
            request_id=current_request_id(),
            client_id=current_client_id(),
            project_id=current_project_id(),
            service=current_service_name(),
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
        ))
        
        return response
```

### 2. Budget Management

```python
class BudgetManager:
    """Manage per-client budgets with enforcement."""
    
    async def check_budget(self, client_id: str) -> BudgetStatus:
        """Check if client is within budget."""
        budget = await self.get_client_budget(client_id)
        spent = await self.cost_store.get_period_spend(
            client_id=client_id,
            period_start=budget.period_start,
        )
        
        remaining = budget.limit_usd - spent
        utilization = spent / budget.limit_usd
        
        return BudgetStatus(
            limit_usd=budget.limit_usd,
            spent_usd=spent,
            remaining_usd=remaining,
            utilization_pct=utilization * 100,
            is_exceeded=remaining <= 0,
            is_warning=utilization >= 0.8,  # 80% threshold
        )
    
    async def enforce_budget(self, client_id: str) -> None:
        """Reject requests if budget is exceeded."""
        status = await self.check_budget(client_id)
        
        if status.is_exceeded:
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "budget_exceeded",
                    "message": f"Monthly budget of ${status.limit_usd} exceeded",
                    "spent": float(status.spent_usd),
                    "limit": float(status.limit_usd),
                }
            )
        
        if status.is_warning:
            # Add warning header but allow request
            logger.warning(
                "budget_warning",
                client_id=client_id,
                utilization_pct=status.utilization_pct,
            )
```

### 3. Rate Limiting

```python
from datetime import timedelta

class RateLimiter:
    """Token-aware rate limiting."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        client_id: str,
        limits: RateLimitConfig,
    ) -> RateLimitResult:
        """Check rate limits using sliding window."""
        
        # Check requests per minute
        rpm_key = f"ratelimit:{client_id}:rpm"
        current_rpm = await self.redis.incr(rpm_key)
        if current_rpm == 1:
            await self.redis.expire(rpm_key, 60)
        
        if current_rpm > limits.requests_per_minute:
            return RateLimitResult(
                allowed=False,
                reason="requests_per_minute_exceeded",
                retry_after_seconds=await self.redis.ttl(rpm_key),
            )
        
        # Check tokens per day
        tpd_key = f"ratelimit:{client_id}:tpd:{date.today()}"
        current_tpd = int(await self.redis.get(tpd_key) or 0)
        
        if current_tpd > limits.tokens_per_day:
            return RateLimitResult(
                allowed=False,
                reason="tokens_per_day_exceeded",
                retry_after_seconds=seconds_until_midnight(),
            )
        
        return RateLimitResult(allowed=True)
    
    async def record_token_usage(
        self,
        client_id: str,
        tokens_used: int,
    ):
        """Record token usage for rate limiting."""
        tpd_key = f"ratelimit:{client_id}:tpd:{date.today()}"
        await self.redis.incrby(tpd_key, tokens_used)
        await self.redis.expire(tpd_key, 86400)  # Expire after 24h
```

### 4. Cost Reporting

```python
class CostReporter:
    """Generate cost reports from tracked data."""
    
    async def generate_client_report(
        self,
        client_id: str,
        start_date: date,
        end_date: date,
    ) -> CostReport:
        """Generate a cost report for a client."""
        events = await self.cost_store.query(
            client_id=client_id,
            start=start_date,
            end=end_date,
        )
        
        return CostReport(
            client_id=client_id,
            period=f"{start_date} to {end_date}",
            total_cost_usd=sum(e.cost_usd for e in events),
            total_tokens=sum(e.prompt_tokens + e.completion_tokens for e in events),
            requests_count=len(events),
            breakdown_by_service={
                service: sum(e.cost_usd for e in group)
                for service, group in groupby(events, key=lambda e: e.service)
            },
            breakdown_by_model={
                model: sum(e.cost_usd for e in group)
                for model, group in groupby(events, key=lambda e: e.model)
            },
            daily_trend=[
                {"date": day, "cost": sum(e.cost_usd for e in group)}
                for day, group in groupby(events, key=lambda e: e.timestamp.date())
            ],
        )
```

## Quota Tiers

Design quota tiers for different client types:

| Tier | Requests/min | Tokens/day | Monthly Budget | Services Access |
|------|-------------|------------|----------------|-----------------|
| **Free/Trial** | 10 | 50,000 | $5 | Knowledge search only |
| **Standard** | 60 | 500,000 | $100 | All services |
| **Enterprise** | 300 | 5,000,000 | $1,000 | All services + priority |
| **Unlimited** | 1,000 | No limit | Custom | All services + SLA |

## Approaches & Trade-offs

| Approach | When to Use | Pros | Cons |
|----------|-------------|------|------|
| **Real-time calculation (in-request)** | Need instant cost visibility | Immediate, accurate | Adds latency to every request |
| **Async aggregation (background)** | High throughput, cost reports | No request overhead | Slight delay in budget enforcement |
| **Cloud billing APIs** | Need actual infra costs | Ground truth from provider | Delayed (hours/days), no per-request detail |
| **Hybrid (real-time + async)** | Enterprise platform | Best of both worlds | More complex to build |

### Recommendation for Phase 6

**Hybrid approach:**
- **Real-time**: Calculate estimated cost per request, enforce budget limits
- **Async**: Aggregate into reports, reconcile with actual cloud billing
- **Store**: Use PostgreSQL for cost events, Redis for real-time rate limiting

## Best Practices

- ✅ **Use Decimal, not float, for money**: Floating point arithmetic causes rounding errors. Use Python's `Decimal` type or store as integer cents.
- ✅ **Store pricing in config, not code**: Model prices change. Store them in a config file or database.
- ✅ **Include cost in every response**: Clients should see their spending in real-time via response metadata.
- ✅ **Set up budget alerts at 50%, 80%, 95%**: Don't wait until 100% to notify.
- ✅ **Track embedding and retrieval costs too**: Not just LLM completion costs. Vector search queries and storage have costs.
- ✅ **Plan for multi-model cost tracking**: Different models have different prices. The system must handle this.
- ❌ **Don't hardcode prices**: They will change, and you'll forget to update.
- ❌ **Don't use only request-based rate limits**: Token-based limits are more meaningful for LLM APIs.
- ❌ **Don't skip rate limiting for internal services**: A bug in one service can generate infinite LLM calls.

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Budget exceeded before alert | Only checking budget periodically | Real-time budget checks on every request |
| Inaccurate cost estimates | Using wrong pricing, ignoring overhead | Regularly reconcile with actual bills |
| Rate limit bypass | Limits only at gateway, not inter-service | Rate limit at both gateway and service level |
| Token estimation errors | Estimating tokens before LLM call | Track actual tokens from response, not estimates |
| Missing cost dimensions | Only tracking LLM tokens | Include embeddings, storage, compute in cost model |

## Application to My Project

### Cost Tracking Implementation

1. **Cost Event Store**: PostgreSQL table storing every LLM call with token counts and calculated cost
2. **Budget Manager**: Middleware that checks client budgets before processing requests
3. **Rate Limiter**: Redis-based sliding window for requests/min and tokens/day
4. **Cost API**: Endpoints for clients to check their usage and download reports
5. **Cost Dashboard**: Grafana dashboard showing real-time cost metrics

### API Endpoints for Cost Management

```
GET  /api/v1/usage/current          # Current period usage summary
GET  /api/v1/usage/history          # Historical usage data
GET  /api/v1/usage/breakdown        # Breakdown by service/model
GET  /api/v1/budget/status          # Current budget status
POST /api/v1/admin/budgets          # Set client budgets (admin only)
GET  /api/v1/admin/reports/{period} # Generate cost reports (admin only)
```

### Decisions to Make
- [ ] PostgreSQL vs. TimescaleDB for cost event storage (time-series data is a natural fit)
- [ ] How to reconcile estimated costs with actual Azure billing
- [ ] Billing model: pre-paid budgets or post-paid with invoicing?
- [ ] Granularity of reporting: per-request, per-hour, per-day?

## Resources for Deeper Learning

- [Azure OpenAI Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — Current pricing
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — Understanding token counting
- [tiktoken library](https://github.com/openai/tiktoken) — Token counting in Python
- [Redis Rate Limiting Patterns](https://redis.io/glossary/rate-limiting/) — Sliding window implementation

## Questions Remaining

- [ ] How to handle cost tracking for streaming responses (tokens arrive incrementally)?
- [ ] Should we implement a pre-flight cost estimate endpoint?
- [ ] How to handle cost attribution for multi-service requests (e.g., research → knowledge + LLM)?
