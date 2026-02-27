---
date: 2026-02-27
type: technology
topic: "FastAPI for Enterprise AI Services"
project: "Phase 6 - Enterprise AI Platform"
status: complete
decision: use
---

# Technology Brief: FastAPI for Enterprise AI Services

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Modern, high-performance Python web framework for building APIs |
| **For** | Building the API gateway and all AI service endpoints |
| **Maturity** | Stable, widely adopted in production AI systems |
| **License** | MIT |
| **Decision** | **Use** — Required by Phase 6 and ideal for AI services |

## Why FastAPI for AI Services

FastAPI is the de facto standard for building AI service APIs in Python. Here's why it fits perfectly:

1. **Native async support**: LLM calls are I/O-bound. Async lets you handle many concurrent requests without blocking.
2. **Automatic OpenAPI documentation**: Every endpoint is automatically documented — critical for an enterprise platform.
3. **Pydantic integration**: Type-safe request/response models, validation, serialization — all built in.
4. **Dependency injection**: Clean pattern for auth, database connections, LLM clients — testable and composable.
5. **Streaming support**: Server-Sent Events (SSE) for real-time LLM output streaming.
6. **Performance**: Built on Starlette and uvicorn, one of the fastest Python frameworks.

## Core Concepts

### 1. Async/Await

FastAPI is built on async Python. This is essential for AI services because LLM API calls take 1-30 seconds — you don't want to block the entire server while waiting.

```python
from fastapi import FastAPI

app = FastAPI()

# ❌ Synchronous — blocks the worker while waiting for LLM
@app.post("/search")
def search_sync(query: str):
    result = call_llm(query)  # Blocks for 2+ seconds
    return result

# ✅ Asynchronous — worker handles other requests while waiting
@app.post("/search")
async def search_async(query: str):
    result = await call_llm_async(query)  # Non-blocking
    return result
```

**Key insight**: With async, a single uvicorn worker can handle dozens of concurrent LLM calls. Without it, you need one worker per concurrent request.

### 2. Pydantic Models for Request/Response

Define strict schemas for every endpoint. This ensures type safety, automatic validation, and clean documentation.

```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class SearchRequest(BaseModel):
    """Request to search the knowledge base."""
    query: str = Field(..., min_length=1, max_length=10000, description="Search query")
    max_results: int = Field(default=5, ge=1, le=20, description="Maximum results to return")
    filters: dict[str, str] | None = Field(default=None, description="Optional metadata filters")

class SearchResult(BaseModel):
    """A single search result."""
    content: str
    score: float = Field(ge=0, le=1)
    source: str
    metadata: dict[str, str]

class APIResponse(BaseModel):
    """Standard API response envelope."""
    status: str = "success"
    data: dict | list | None = None
    metadata: ResponseMetadata

class ResponseMetadata(BaseModel):
    """Metadata included in every response."""
    request_id: str
    processing_time_ms: int
    tokens_used: int = 0
    estimated_cost_usd: float = 0.0
    model_version: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

### 3. Dependency Injection

FastAPI's dependency injection system is how you handle cross-cutting concerns cleanly.

```python
from fastapi import Depends, Security

# Dependencies can be chained
async def get_db() -> AsyncSession:
    """Provide a database session."""
    async with async_session() as session:
        yield session

async def get_llm_client() -> LLMClient:
    """Provide an LLM client."""
    return LLMClient(config=get_settings())

async def get_current_client(
    api_key: str = Security(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> ClientInfo:
    """Authenticate and return client info."""
    client = await db.execute(
        select(Client).where(Client.api_key == api_key)
    )
    if not client:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return client

# Use in routes
@app.post("/search")
async def search(
    request: SearchRequest,
    client: ClientInfo = Depends(get_current_client),
    llm: LLMClient = Depends(get_llm_client),
    db: AsyncSession = Depends(get_db),
):
    # client, llm, and db are automatically injected
    ...
```

### 4. Middleware Pipeline

Middleware processes every request before it reaches route handlers. This is where cross-cutting concerns live.

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import time
import uuid

# Order matters: first added = outermost (runs first on request, last on response)
app = FastAPI()

# CORS (outermost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request ID middleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)

# Timing middleware
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start) * 1000)
        response.headers["X-Processing-Time-Ms"] = str(duration_ms)
        return response

app.add_middleware(TimingMiddleware)
```

### 5. Routers for Service Organization

Organize endpoints into routers for clean separation.

```python
from fastapi import APIRouter

# Each service gets its own router
knowledge_router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base"])
agent_router = APIRouter(prefix="/api/v1/agent", tags=["Customer Agent"])
research_router = APIRouter(prefix="/api/v1/research", tags=["Research Team"])
assistant_router = APIRouter(prefix="/api/v1/assistant", tags=["Multi-Modal Assistant"])
admin_router = APIRouter(prefix="/api/v1/admin", tags=["Administration"])

# Define routes on routers
@knowledge_router.post("/search")
async def search_knowledge(request: SearchRequest):
    ...

@knowledge_router.post("/ingest")
async def ingest_document(request: IngestRequest):
    ...

# Mount routers on the app
app = FastAPI(title="Enterprise AI Platform", version="1.0.0")
app.include_router(knowledge_router)
app.include_router(agent_router)
app.include_router(research_router)
app.include_router(assistant_router)
app.include_router(admin_router)
```

### 6. Streaming Responses (SSE)

For real-time LLM output, use Server-Sent Events.

```python
from fastapi.responses import StreamingResponse
from openai import AsyncAzureOpenAI

@agent_router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream LLM response as Server-Sent Events."""
    
    async def event_generator():
        client = AsyncAzureOpenAI(...)
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=request.messages,
            stream=True,
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield f"data: {json.dumps({'content': content})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
```

### 7. Background Tasks

For operations that shouldn't block the response.

```python
from fastapi import BackgroundTasks

@research_router.post("/start", status_code=202)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
):
    """Start a research task asynchronously."""
    task_id = str(uuid.uuid4())
    
    # Return immediately with task ID
    background_tasks.add_task(
        execute_research,
        task_id=task_id,
        query=request.query,
    )
    
    return {"task_id": task_id, "status": "accepted"}

async def execute_research(task_id: str, query: str):
    """Long-running research task executed in background."""
    # This runs after the response is sent
    ...
```

### 8. Error Handling

Centralized exception handling for consistent error responses.

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class AIServiceError(Exception):
    """Base exception for AI service errors."""
    def __init__(self, message: str, status_code: int = 500, error_code: str = "internal_error"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

class BudgetExceededError(AIServiceError):
    def __init__(self, client_id: str, budget: float):
        super().__init__(
            message=f"Monthly budget of ${budget} exceeded",
            status_code=429,
            error_code="budget_exceeded",
        )

class PromptInjectionError(AIServiceError):
    def __init__(self):
        super().__init__(
            message="Request rejected: potential prompt injection detected",
            status_code=400,
            error_code="prompt_injection_detected",
        )

# Register global exception handlers
@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error": {
                "code": exc.error_code,
                "message": exc.message,
            },
            "metadata": {
                "request_id": getattr(request.state, "request_id", "unknown"),
            }
        },
    )
```

### 9. Configuration Management

Use Pydantic Settings for type-safe configuration.

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Service
    service_name: str = "ai-platform-gateway"
    environment: str = "development"
    debug: bool = False
    
    # Azure OpenAI
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str = "2025-01-01"
    default_model: str = "gpt-4o"
    
    # Database
    database_url: str = "postgresql+asyncpg://localhost/ai_platform"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Rate Limiting
    default_rpm: int = 60        # Requests per minute
    default_tpd: int = 500_000   # Tokens per day
    
    # Security
    api_key_header: str = "X-API-Key"
    
    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### 10. Health Checks

Every service needs health endpoints.

```python
@app.get("/health")
async def health_check():
    """Basic liveness check."""
    return {"status": "healthy"}

@app.get("/health/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_db),
    llm: LLMClient = Depends(get_llm_client),
):
    """Detailed readiness check — tests all dependencies."""
    checks = {}
    
    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"
    
    # Check LLM API
    try:
        await llm.complete([{"role": "user", "content": "ping"}], max_tokens=1)
        checks["llm_api"] = "ok"
    except Exception as e:
        checks["llm_api"] = f"error: {str(e)}"
    
    all_ok = all(v == "ok" for v in checks.values())
    
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks}
    )
```

## Project Structure for Phase 6

```
ai-platform/
├── gateway/                        # API Gateway Service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app, middleware, routers
│   │   ├── config.py               # Settings
│   │   ├── middleware/
│   │   │   ├── auth.py             # Authentication
│   │   │   ├── rate_limit.py       # Rate limiting
│   │   │   ├── cost_tracking.py    # Cost tracking
│   │   │   ├── observability.py    # Logging, tracing
│   │   │   └── security.py        # Prompt injection detection
│   │   ├── routes/
│   │   │   ├── knowledge.py        # /api/v1/knowledge/*
│   │   │   ├── agent.py            # /api/v1/agent/*
│   │   │   ├── research.py         # /api/v1/research/*
│   │   │   ├── assistant.py        # /api/v1/assistant/*
│   │   │   └── admin.py            # /api/v1/admin/*
│   │   ├── services/
│   │   │   └── proxy.py            # Service proxy/router
│   │   └── models/
│   │       └── schemas.py          # Pydantic models
│   ├── Dockerfile
│   ├── requirements.txt
│   └── tests/
│       ├── test_auth.py
│       ├── test_routes.py
│       └── conftest.py
├── services/
│   ├── knowledge/                  # Knowledge Base Service
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── routes/
│   │   │   └── services/
│   │   └── Dockerfile
│   ├── agent/                      # Customer Agent Service
│   ├── research/                   # Research Team Service
│   └── assistant/                  # Multi-Modal Assistant Service
├── shared/                         # Shared Python package
│   ├── llm_client.py
│   ├── schemas.py
│   ├── logging_config.py
│   └── cost.py
├── docker-compose.yml
├── kubernetes/
│   ├── gateway.yaml
│   ├── knowledge.yaml
│   └── ...
└── pyproject.toml
```

## Key APIs Reference

| API | Purpose | Example |
|-----|---------|---------|
| `FastAPI()` | Create application | `app = FastAPI(title="AI Platform")` |
| `@app.post()` | Define POST endpoint | `@app.post("/search")` |
| `APIRouter()` | Group related endpoints | `router = APIRouter(prefix="/api/v1")` |
| `Depends()` | Dependency injection | `db: Session = Depends(get_db)` |
| `Security()` | Security dependency | `key: str = Security(api_key_header)` |
| `BackgroundTasks` | Async background work | `tasks.add_task(func, arg)` |
| `StreamingResponse` | SSE/streaming | `StreamingResponse(generator())` |
| `HTTPException` | Raise HTTP errors | `raise HTTPException(404, "Not found")` |
| `BaseModel` | Request/response schema | `class Req(BaseModel): query: str` |
| `BaseSettings` | Configuration | `class Settings(BaseSettings): ...` |

## Running FastAPI

```bash
# Development (auto-reload)
uvicorn app.main:app --reload --port 8000

# Production (multiple workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# With gunicorn (production, process manager)
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## Trade-offs

### Pros
- ✅ Native async — essential for I/O-heavy AI workloads
- ✅ Automatic OpenAPI docs — great for enterprise API documentation
- ✅ Pydantic validation — type safety and clear contracts
- ✅ Dependency injection — clean, testable architecture
- ✅ Streaming support — real-time LLM output
- ✅ Huge ecosystem — middleware, extensions, community

### Cons
- ❌ Python's GIL limits CPU-bound parallelism (mitigated by async for I/O)
- ❌ Not as fast as Go/Rust for pure throughput (but fast enough for AI services)
- ❌ Middleware can become tangled if not carefully organized
- ❌ Background tasks are limited (use Celery/Redis for serious job queues)

## Enterprise Considerations

- **Scale**: Horizontal scaling via multiple pods in Kubernetes. Each pod runs 2-4 uvicorn workers.
- **Security**: Built-in OAuth2/JWT support, API key headers, CORS middleware.
- **Cost**: Free framework. Costs come from compute (Azure) and LLM API usage.
- **Support**: Large open-source community, extensive documentation, Sebastian Ramirez maintains actively.

## Decision

**Recommendation**: **Use** — FastAPI is the right choice for Phase 6.

**Reasoning**: Required by the project spec. Also the best Python framework for AI services: async, Pydantic, OpenAPI docs, streaming support, and a massive ecosystem. No realistic alternative offers the same combination.

**Next steps**:
1. Set up the gateway service skeleton
2. Define Pydantic models for all request/response types
3. Implement middleware pipeline (auth, rate limiting, cost tracking)
4. Add routers for each AI service

## Resources for Deeper Learning

- [FastAPI Official Documentation](https://fastapi.tiangolo.com/) — Comprehensive and well-written
- [FastAPI Best Practices](https://github.com/zhanymkanov/fastapi-best-practices) — Community best practices
- [Pydantic v2 Documentation](https://docs.pydantic.dev/) — Data validation
- [Starlette Documentation](https://www.starlette.io/) — Underlying ASGI framework
- [uvicorn Documentation](https://www.uvicorn.org/) — ASGI server
