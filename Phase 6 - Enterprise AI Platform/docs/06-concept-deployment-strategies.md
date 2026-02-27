---
date: 2026-02-27
type: concept
topic: "Deployment Strategies for AI Systems"
project: "Phase 6 - Enterprise AI Platform"
status: complete
---

# Deployment Strategies for AI Systems

## In My Own Words

Deployment strategy is about how you get new code into production safely. For traditional applications, this means updating servers without downtime. For AI systems, it's harder because you're deploying two things: **code** (your application logic) and **models/prompts** (the AI behavior). A code change might be safe, but a prompt change could cause the model to hallucinate differently. You need strategies that let you roll out changes gradually, test them in production, and roll back quickly when things go wrong.

The core tension: you want to ship improvements fast (new models, better prompts, new features) while ensuring that the platform remains reliable and output quality doesn't degrade.

## Why This Matters

Phase 6 requires:
- Model versioning and A/B testing infrastructure
- Deployment strategies (blue-green, canary)
- Must handle 10 concurrent users without degradation
- Must be deployable by someone reading your documentation

In a real enterprise, a bad deployment of an AI service can produce incorrect outputs at scale — misinforming customers, generating wrong analyses, or triggering compliance issues. Safe deployment isn't optional.

## Core Principles

1. **Zero-Downtime Deployments**: Users should never see an error because you're deploying. Keep the old version running while the new version starts up.

2. **Gradual Rollout**: Never switch 100% of traffic to new code at once. Start with 1-5%, monitor, increase gradually.

3. **Quick Rollback**: If something goes wrong, you should be able to revert in under 1 minute. Automate this.

4. **Separate Code from Model/Prompt Versions**: Code deployments and model/prompt changes are different concerns with different rollback needs.

5. **Observability-Driven Deployment**: Deployment decisions (continue rollout vs. rollback) should be based on metrics, not gut feeling.

## Deployment Strategies Explained

### Strategy 1: Blue-Green Deployment

**Concept**: Maintain two identical production environments. One is "live" (Blue), the other is "idle" (Green). Deploy to Green, test it, then switch traffic from Blue to Green.

```
                    BEFORE SWITCH                    AFTER SWITCH
                    
 Users ──► [LB] ──► Blue (v1.0) ✅    Users ──► [LB] ──► Green (v1.1) ✅
                     Green (v1.1) 🔄                       Blue (v1.0) standby
                     
 LB = Load Balancer
 ✅ = Receiving traffic
 🔄 = Being deployed/tested
```

**How it works:**
1. Blue is running v1.0, serving all traffic
2. Deploy v1.1 to Green
3. Run health checks and smoke tests on Green
4. Switch the load balancer from Blue → Green
5. Green now serves all traffic
6. Keep Blue running as an instant rollback target
7. If problems detected → switch back to Blue in seconds

**Pros:**
- Instant rollback (just switch the LB back)
- Full testing in production-like environment before going live
- Zero downtime

**Cons:**
- Requires 2x infrastructure (double cost while both run)
- Database migrations can be tricky (both environments share the DB)
- All-or-nothing switch (no gradual rollout)

**Best for:** Services where you need instant rollback capability and can afford double infrastructure.

### Strategy 2: Canary Deployment

**Concept**: Deploy the new version alongside the old one, but only route a small percentage of traffic to it. Gradually increase the percentage as confidence grows.

```
 Traffic Split Over Time:
 
 Time 0:    [Old v1.0] ████████████████████  100%
            [New v1.1]                         0%
 
 Time 1:    [Old v1.0] ███████████████████   95%
            [New v1.1] █                       5%
 
 Time 2:    [Old v1.0] ████████████████      80%
            [New v1.1] ████                   20%
 
 Time 3:    [Old v1.0] ██████████            50%
            [New v1.1] ██████████             50%
 
 Time 4:    [Old v1.0]                        0%
            [New v1.1] ████████████████████  100%
```

**How it works:**
1. Deploy v1.1 as a "canary" instance alongside v1.0
2. Route 5% of traffic to v1.1
3. Monitor error rates, latency, quality metrics
4. If metrics are good → increase to 20%, then 50%, then 100%
5. If metrics degrade → route 100% back to v1.0, investigate

**Pros:**
- Gradual rollout limits blast radius
- Real-world testing with real traffic
- Can detect subtle issues that testing misses

**Cons:**
- More complex routing logic
- Two versions running simultaneously — must be compatible
- Monitoring must be robust (otherwise you miss degradation)

**Best for:** AI services where quality might vary — canary lets you detect quality regression on a small traffic sample.

### Strategy 3: Rolling Deployment

**Concept**: Gradually replace instances of the old version with the new version, one at a time.

```
 3 Replicas:
 
 Step 1:  [v1.0] [v1.0] [v1.0]    (all old)
 Step 2:  [v1.1] [v1.0] [v1.0]    (1 updated)
 Step 3:  [v1.1] [v1.1] [v1.0]    (2 updated)
 Step 4:  [v1.1] [v1.1] [v1.1]    (all updated)
```

**How it works:**
1. Take one instance out of the load balancer pool
2. Update it to v1.1
3. Add it back to the pool
4. Repeat for each instance
5. Kubernetes does this automatically with Deployment resources

**Pros:**
- Built into Kubernetes (default strategy)
- No extra infrastructure needed
- Gradual transition

**Cons:**
- Two versions running simultaneously during rollout
- Slower rollback than blue-green (must roll forward or backward through instances)
- Not great for detecting quality issues (mixed traffic across versions)

**Best for:** Standard code deployments that don't change AI behavior.

### Strategy 4: A/B Testing (for AI-Specific Changes)

**Concept**: Not really a deployment strategy per se, but critical for AI platforms. Run two different model/prompt versions simultaneously and compare their performance.

```
 A/B Test: "New prompt template vs. old"
 
 Users ──► [Gateway] ──┬── 50% ──► Service (prompt v2.1)  ── Group A
                       └── 50% ──► Service (prompt v2.0)  ── Group B
                       
 Collect metrics for both groups:
 - Quality score
 - Latency
 - User satisfaction
 - Cost per request
```

**How it works:**
1. Define the test: what's changing (model, prompt, parameters)?
2. Configure traffic split (e.g., 50/50 or 90/10)
3. Route consistently per user/session (same user always gets same variant)
4. Collect metrics for both groups
5. Statistical analysis to determine winner
6. Roll out winner to 100%

**AI-specific considerations:**
- Must track quality metrics, not just latency/errors
- Need enough traffic for statistical significance
- Account for prompt variability (same prompt can produce different outputs)
- Consider running for longer periods (AI quality can drift)

## AI-Specific Deployment Challenges

### Challenge 1: Model Version Management

```
Model Versioning in an Enterprise Platform:

model-registry/
├── gpt-4o/
│   ├── v2025-08/          # Specific API version
│   │   ├── config.yaml     # Model parameters
│   │   └── evaluation.json # Quality benchmarks
│   └── v2026-01/
│       ├── config.yaml
│       └── evaluation.json
├── gpt-4o-mini/
│   └── v2025-08/
│       ├── config.yaml
│       └── evaluation.json
└── text-embedding-3-small/
    └── v2024-01/
        └── config.yaml
```

When Azure OpenAI releases a new model version:
1. Register the new version in your model registry
2. Run your evaluation suite against it
3. Compare quality, latency, and cost vs. current version
4. If better → canary deploy, gradually shift traffic
5. If worse → stay on current version, log the evaluation results

### Challenge 2: Prompt Version Management

```python
# Prompt versioning system
class PromptRegistry:
    """Manage prompt versions with A/B testing support."""
    
    prompts = {
        "rag_search": {
            "v2.0": {
                "template": "Given the context: {context}\nAnswer: {question}",
                "model": "gpt-4o",
                "temperature": 0.1,
                "status": "production",
            },
            "v2.1": {
                "template": "You are a helpful assistant...\nContext: {context}\nQuestion: {question}\nProvide a detailed answer with citations.",
                "model": "gpt-4o",
                "temperature": 0.1,
                "status": "canary",    # Only 10% of traffic
                "traffic_pct": 10,
            },
        }
    }
    
    def get_prompt(self, name: str, user_id: str) -> PromptConfig:
        """Get prompt version, considering A/B tests."""
        versions = self.prompts[name]
        
        # Check for active A/B test
        canary = next(
            (v for v in versions.values() if v["status"] == "canary"),
            None
        )
        
        if canary and self._in_test_group(user_id, canary["traffic_pct"]):
            return canary
        
        # Return production version
        return next(v for v in versions.values() if v["status"] == "production")
```

### Challenge 3: Database Migration During Deployment

When deploying new code that changes the database schema:

1. **Forward-compatible migrations**: New code should work with the old schema (add columns as nullable, don't rename)
2. **Expand-contract pattern**:
   - Expand: Add new columns/tables (both versions work)
   - Migrate: Backfill data
   - Contract: Remove old columns (only new version runs)
3. **Never deploy schema changes and code changes simultaneously**

## Deployment Pipeline for Phase 6

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   Code   │    │  Build   │    │  Test    │    │  Stage   │    │  Prod    │
│  Commit  │───►│  Docker  │───►│  Suite   │───►│  Deploy  │───►│  Deploy  │
│          │    │  Image   │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                     │               │               │
                                     ▼               ▼               ▼
                               Unit tests      Smoke tests      Canary (5%)
                               Integration     Quality eval     Monitor
                               Linting         Load test        Gradual rollout
                               Security scan                    Full rollout
```

### GitHub Actions Workflow (Simplified)

```yaml
# .github/workflows/deploy.yml
name: Deploy AI Platform

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run tests
        run: pytest tests/ --cov
      - name: Security scan
        run: bandit -r src/
      - name: Lint
        run: ruff check src/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker image
        run: docker build -t ai-platform:${{ github.sha }} .
      - name: Push to registry
        run: docker push acr.azurecr.io/ai-platform:${{ github.sha }}

  deploy-staging:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to staging
        run: kubectl set image deployment/ai-platform ai-platform=acr.azurecr.io/ai-platform:${{ github.sha }} -n staging
      - name: Run smoke tests
        run: pytest tests/smoke/ --env=staging
      - name: Run quality evaluation
        run: python scripts/evaluate_quality.py --env=staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - name: Canary deploy (5%)
        run: |
          kubectl set image deployment/ai-platform-canary \
            ai-platform=acr.azurecr.io/ai-platform:${{ github.sha }} -n production
      - name: Monitor canary (10 min)
        run: python scripts/monitor_canary.py --duration=600
      - name: Full rollout
        run: |
          kubectl set image deployment/ai-platform \
            ai-platform=acr.azurecr.io/ai-platform:${{ github.sha }} -n production
```

## Approaches & Trade-offs Summary

| Strategy | Downtime | Rollback Speed | Resource Cost | Complexity | Best For |
|----------|----------|----------------|---------------|------------|----------|
| **Blue-Green** | Zero | Instant | 2x during deploy | Medium | Critical services |
| **Canary** | Zero | Fast (route shift) | +10-20% | High | AI quality changes |
| **Rolling** | Zero | Slow (re-roll) | None extra | Low | Standard code changes |
| **A/B Testing** | Zero | N/A (both run) | +50% | High | Prompt/model experiments |

### Recommendation for Phase 6

1. **Rolling deployment** for standard code changes (Kubernetes default)
2. **Canary deployment** for AI-behavior changes (new models, prompt changes)
3. **A/B testing** for prompt optimization experiments
4. **Blue-green** for major platform upgrades (v1 → v2)

## Best Practices

- ✅ **Automate everything**: Manual deployments are error-prone. CI/CD pipeline from day 1.
- ✅ **Run quality evaluations before promoting**: Don't just check "does it start" — check "is the output quality acceptable."
- ✅ **Use feature flags for AI changes**: Toggle new prompts/models without redeploying code.
- ✅ **Health checks that test AI functionality**: A health endpoint that makes a real LLM call (not just "server is up").
- ✅ **Tag Docker images with git SHA**: Always know exactly what code is running.
- ✅ **Keep deployment documentation updated**: Phase 6 requires that someone else can deploy from docs.
- ❌ **Don't deploy on Fridays**: Classic rule. Issues found over weekends are painful.
- ❌ **Don't skip staging**: Even for "small changes." LLM behavior is unpredictable.
- ❌ **Don't couple service deployments**: Deploy each service independently.

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| Big-bang deployment | "Let's deploy everything at once" | Independent service deployments |
| No rollback plan | "What could go wrong?" | Automated rollback triggers on metric degradation |
| Quality regression unnoticed | Only monitoring availability, not output quality | Automated quality eval in deployment pipeline |
| Database migration breaks rollback | Schema changed, old code can't run | Forward-compatible migrations, expand-contract |
| Configuration drift | Staging and production diverge | Infrastructure as Code, same configs everywhere |

## Application to My Project

### Deployment Strategy Plan

1. **Local Development**: Docker Compose for all services
2. **CI/CD**: GitHub Actions for build, test, deploy
3. **Container Registry**: Azure Container Registry
4. **Staging**: AKS namespace `staging` with smoke tests
5. **Production**: AKS namespace `production` with canary deployment
6. **A/B Testing**: Feature flags in gateway for prompt/model experiments

### Decisions to Make
- [ ] GitHub Actions vs. Azure DevOps for CI/CD
- [ ] How long to monitor canary before promoting (10 min? 1 hour?)
- [ ] Feature flag system (LaunchDarkly? Custom? Environment variables?)
- [ ] Rollback criteria: what metrics trigger automatic rollback?

## Resources for Deeper Learning

- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#strategy) — Official docs
- [Martin Fowler: Blue-Green Deployment](https://martinfowler.com/bliki/BlueGreenDeployment.html) — Foundational article
- [Microsoft: Deployment strategies in AKS](https://learn.microsoft.com/en-us/azure/aks/concepts-clusters-workloads) — Azure-specific guidance
- [GitHub Actions documentation](https://docs.github.com/en/actions) — CI/CD setup

## Questions Remaining

- [ ] How to A/B test across multiple services (e.g., new prompt + new retrieval strategy together)?
- [ ] What's a good quality threshold for automated canary promotion?
- [ ] How to handle long-running async tasks during deployment (research tasks mid-execution)?
