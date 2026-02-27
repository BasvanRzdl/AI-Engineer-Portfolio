---
date: 2026-02-27
type: technology
topic: "Docker Containerization for AI Services"
project: "Phase 6 - Enterprise AI Platform"
status: complete
decision: use
---

# Technology Brief: Docker Containerization for AI Services

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Container platform for packaging applications with all dependencies |
| **For** | Consistent deployment across dev, staging, and production |
| **Maturity** | Industry standard, extremely stable |
| **License** | Apache 2.0 (Docker Engine), Proprietary (Docker Desktop) |
| **Decision** | **Use** — Required by Phase 6, essential for K8s deployment |

## Why Docker for AI Services

Docker solves the "works on my machine" problem. For AI services, this is especially important because:

1. **Complex dependencies**: AI services need specific Python versions, CUDA libraries, ML packages — Docker captures all of these.
2. **Reproducibility**: Same container runs identically in dev, CI, staging, and production.
3. **Isolation**: Each service runs in its own container with its own dependencies. No version conflicts.
4. **Kubernetes requirement**: K8s runs containers. If you want K8s, you need Docker (or an OCI-compatible alternative).
5. **Microservices enabler**: Each AI service is a separate container, independently scalable.

## Core Concepts

### 1. Images vs. Containers

- **Image**: A read-only template. Like a class definition. Built from a Dockerfile.
- **Container**: A running instance of an image. Like an object instantiated from a class.
- **Registry**: Where images are stored (Docker Hub, Azure Container Registry).

```
Dockerfile  ──build──►  Image  ──run──►  Container
                          │
                          └──push──►  Registry (ACR)
                                        │
                                        └──pull──►  Container (in K8s)
```

### 2. Dockerfile Anatomy

```dockerfile
# Dockerfile for a FastAPI AI service

# 1. Base image — start from a known Python version
FROM python:3.12-slim AS base

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency files first (layer caching optimization)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY app/ ./app/

# 7. Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 8. Expose port
EXPOSE 8000

# 9. Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 10. Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Multi-Stage Builds

Reduce image size by separating build and runtime stages.

```dockerfile
# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime (smaller image)
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Run as non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Result**: The runtime image doesn't contain build tools (gcc, build-essential), reducing size by 50-70%.

### 4. Docker Compose for Local Development

Docker Compose defines and runs multi-container applications. Essential for developing Phase 6 locally.

```yaml
# docker-compose.yml
version: "3.9"

services:
  gateway:
    build: ./gateway
    ports:
      - "8000:8000"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql+asyncpg://postgres:password@postgres:5432/ai_platform
    depends_on:
      - redis
      - postgres
    volumes:
      - ./gateway/app:/app/app  # Hot reload in development

  knowledge-service:
    build: ./services/knowledge
    ports:
      - "8001:8000"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant

  agent-service:
    build: ./services/agent
    ports:
      - "8002:8000"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  research-service:
    build: ./services/research
    ports:
      - "8003:8000"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

  assistant-service:
    build: ./services/assistant
    ports:
      - "8004:8000"
    environment:
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}

  # Infrastructure services
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:16-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=ai_platform
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres-data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage

  # Observability
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./observability/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana

volumes:
  redis-data:
  postgres-data:
  qdrant-data:
  grafana-data:
```

### 5. .dockerignore

Keep images small and secure by excluding unnecessary files.

```
# .dockerignore
.git
.gitignore
.env
.env.*
__pycache__
*.pyc
*.pyo
.pytest_cache
.coverage
htmlcov/
.venv
venv
node_modules
*.md
LICENSE
docker-compose*.yml
kubernetes/
tests/
docs/
.github/
```

## Docker Image Optimization for AI Services

### Size Optimization

```
Image Size Comparison:
┌──────────────────────────┬──────────┐
│ Approach                 │ Size     │
├──────────────────────────┼──────────┤
│ python:3.12              │ ~1.0 GB  │
│ python:3.12-slim         │ ~150 MB  │
│ python:3.12-alpine       │ ~50 MB   │  ← Compatibility issues
│ python:3.12-slim + multi │ ~200 MB  │  ← Recommended (with deps)
│   stage build            │          │
└──────────────────────────┴──────────┘
```

**Recommendation**: Use `python:3.12-slim` with multi-stage builds. Alpine can cause issues with compiled Python packages (numpy, etc.).

### Layer Caching Strategy

```dockerfile
# ✅ Good: Dependencies change rarely, code changes often
COPY requirements.txt .                # Layer 1: cached until requirements change
RUN pip install -r requirements.txt    # Layer 2: cached until requirements change
COPY app/ ./app/                       # Layer 3: rebuilt on every code change

# ❌ Bad: Entire build reruns on any change
COPY . .                               # Copies everything
RUN pip install -r requirements.txt    # Re-installs on any code change
```

### Security Best Practices

```dockerfile
# 1. Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# 2. Don't store secrets in the image
# Use environment variables or volume-mounted secrets at runtime

# 3. Pin base image versions
FROM python:3.12.3-slim  # Specific version, not just "3.12-slim"

# 4. Scan for vulnerabilities
# In CI: docker scout cve ai-platform:latest

# 5. Use read-only filesystem where possible
# In docker-compose or K8s: read_only: true
```

## Essential Docker Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `docker build` | Build image from Dockerfile | `docker build -t my-service:latest .` |
| `docker run` | Run a container | `docker run -p 8000:8000 my-service` |
| `docker compose up` | Start all services | `docker compose up -d` |
| `docker compose down` | Stop all services | `docker compose down` |
| `docker compose logs` | View service logs | `docker compose logs -f gateway` |
| `docker compose build` | Build all images | `docker compose build` |
| `docker exec` | Run command in container | `docker exec -it gateway bash` |
| `docker ps` | List running containers | `docker ps` |
| `docker images` | List images | `docker images` |
| `docker tag` | Tag image for registry | `docker tag img acr.azurecr.io/img:v1` |
| `docker push` | Push to registry | `docker push acr.azurecr.io/img:v1` |

## Environment Variables and Secrets

```yaml
# docker-compose.yml - development
services:
  gateway:
    environment:
      # Non-sensitive: inline
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
    env_file:
      # Sensitive: in .env file (gitignored)
      - .env
```

```bash
# .env (gitignored, never committed)
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=sk-xxxxxxxxxxxx
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/db
```

For production (Kubernetes), use Kubernetes Secrets or Azure Key Vault — never bake secrets into images.

## Trade-offs

### Pros
- ✅ Reproducible environments across all stages
- ✅ Isolation between services (no dependency conflicts)
- ✅ Standard packaging format for Kubernetes
- ✅ Rich ecosystem (registries, scanning, compose)
- ✅ Docker Compose makes local development easy

### Cons
- ❌ Learning curve for Dockerfile optimization
- ❌ Image size can balloon with ML dependencies
- ❌ Docker Desktop licensing for enterprises (use alternatives like Colima/Podman)
- ❌ Performance overhead vs. bare metal (minimal for AI services)

## Enterprise Considerations

- **Scale**: Images are deployed to Kubernetes across many nodes. Small images = faster deployment.
- **Security**: Use vulnerability scanning (Docker Scout, Trivy). No root. No secrets in images.
- **Cost**: Azure Container Registry costs for storage and pulls. Keep images small.
- **Support**: Docker is the industry standard. Abundant documentation and community support.

## Decision

**Recommendation**: **Use** — Docker is required for the containerization requirement in Phase 6.

**Reasoning**: Industry standard, required for Kubernetes deployment, excellent for microservices architecture. Docker Compose provides a great local development experience.

**Next steps**:
1. Write Dockerfiles for each service (gateway + 4 AI services)
2. Create docker-compose.yml for local development with all services + infrastructure
3. Set up Azure Container Registry for image storage
4. Create .dockerignore and optimize layer caching

## Resources for Deeper Learning

- [Docker Official Documentation](https://docs.docker.com/) — Comprehensive reference
- [Docker Best Practices for Python](https://docs.docker.com/language/python/) — Python-specific guide
- [Dive tool](https://github.com/wagoodman/dive) — Explore Docker image layers to optimize size
- [Docker Compose Documentation](https://docs.docker.com/compose/) — Multi-container development
- [Azure Container Registry docs](https://learn.microsoft.com/en-us/azure/container-registry/) — Push images to Azure
