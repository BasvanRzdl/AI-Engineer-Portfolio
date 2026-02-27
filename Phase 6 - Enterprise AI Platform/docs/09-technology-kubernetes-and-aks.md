---
date: 2026-02-27
type: technology
topic: "Kubernetes and Azure Kubernetes Service (AKS)"
project: "Phase 6 - Enterprise AI Platform"
status: complete
decision: use
---

# Technology Brief: Kubernetes and AKS for AI Deployment

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Container orchestration platform for automated deployment, scaling, and management |
| **For** | Running the AI platform's containerized services in production |
| **Maturity** | Industry standard, very stable |
| **License** | Apache 2.0 (K8s), Managed service (AKS) |
| **Decision** | **Use** — Required by Phase 6 spec (AKS) |

## Why Kubernetes for AI Platforms

Docker packages your services into containers. Kubernetes (K8s) runs those containers at scale, handling:

1. **Automatic scaling**: Scale services up/down based on CPU, memory, or custom metrics (like request queue depth)
2. **Self-healing**: If a container crashes, K8s automatically restarts it
3. **Service discovery**: Services find each other by name, not IP addresses
4. **Rolling deployments**: Update services with zero downtime
5. **Resource management**: Assign CPU and memory limits per service
6. **Load balancing**: Distribute traffic across multiple instances of a service

Without K8s, you'd have to manually manage each container on each server — which doesn't scale.

## Core Concepts

### The Kubernetes Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│                    KUBERNETES CLUSTER                         │
│                                                              │
│  ┌─────────── Control Plane ──────────┐                     │
│  │  API Server │ Scheduler │ etcd     │                     │
│  │  Controller Manager                │                     │
│  └────────────────────────────────────┘                     │
│                                                              │
│  ┌──── Node 1 ────┐  ┌──── Node 2 ────┐  ┌── Node 3 ──┐  │
│  │ ┌─Pod──────┐   │  │ ┌─Pod──────┐   │  │ ┌─Pod────┐ │  │
│  │ │ gateway  │   │  │ │ gateway  │   │  │ │knowledge│ │  │
│  │ │ container│   │  │ │ container│   │  │ │container│ │  │
│  │ └─────────┘   │  │ └─────────┘   │  │ └────────┘ │  │
│  │ ┌─Pod──────┐   │  │ ┌─Pod──────┐   │  │ ┌─Pod────┐ │  │
│  │ │ agent   │   │  │ │ research │   │  │ │assistant│ │  │
│  │ │ container│   │  │ │ container│   │  │ │container│ │  │
│  │ └─────────┘   │  │ └─────────┘   │  │ └────────┘ │  │
│  └────────────────┘  └────────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Key Resources

| Resource | What It Is | Analogy |
|----------|-----------|---------|
| **Pod** | Smallest deployable unit. Wraps one or more containers. | A running instance of your service |
| **Deployment** | Manages a set of identical pods. Handles scaling and updates. | "I want 3 copies of the gateway running" |
| **Service** | Stable network endpoint for a set of pods. | A load balancer with a fixed name |
| **Ingress** | Routes external HTTP traffic to services. | The front door of your cluster |
| **ConfigMap** | Key-value config data. | Environment variables from a file |
| **Secret** | Sensitive config data (base64 encoded). | API keys, passwords |
| **Namespace** | Logical isolation within a cluster. | Folders for organizing resources |
| **HPA** | Horizontal Pod Autoscaler. Scales pods based on metrics. | "If CPU > 70%, add more pods" |

### How Resources Relate

```
External Traffic
      │
      ▼
┌──────────┐     ┌─────────────────────────────────────────┐
│  Ingress │────►│  Service (gateway-service)               │
│          │     │  Type: ClusterIP                         │
└──────────┘     │  Selector: app=gateway                   │
                 └──────────┬──────────────────────────────┘
                            │
              ┌─────────────┼─────────────────┐
              ▼             ▼                  ▼
         ┌─────────┐  ┌─────────┐        ┌─────────┐
         │ Pod     │  │ Pod     │        │ Pod     │
         │ gateway │  │ gateway │        │ gateway │
         │ (replica│  │ (replica│        │ (replica│
         │  1)     │  │  2)     │        │  3)     │
         └─────────┘  └─────────┘        └─────────┘
              ▲             ▲                  ▲
              └─────────────┴──────────────────┘
                            │
                    ┌───────────────┐
                    │  Deployment   │
                    │  replicas: 3  │
                    └───────────────┘
```

## Kubernetes YAML Manifests for Phase 6

### 1. Gateway Deployment

```yaml
# kubernetes/gateway/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
  namespace: ai-platform
  labels:
    app: gateway
    component: api-gateway
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gateway
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    metadata:
      labels:
        app: gateway
        component: api-gateway
    spec:
      containers:
        - name: gateway
          image: myacr.azurecr.io/ai-platform/gateway:latest
          ports:
            - containerPort: 8000
          env:
            - name: ENVIRONMENT
              value: "production"
            - name: AZURE_OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: azure-openai-secrets
                  key: api-key
            - name: REDIS_URL
              value: "redis://redis-service:6379"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: database-secrets
                  key: connection-string
          resources:
            requests:
              cpu: "250m"      # 0.25 CPU cores
              memory: "256Mi"  # 256 MB RAM
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
```

### 2. Gateway Service

```yaml
# kubernetes/gateway/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: gateway-service
  namespace: ai-platform
spec:
  selector:
    app: gateway
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
```

### 3. Ingress (External Access)

```yaml
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ai-platform-ingress
  namespace: ai-platform
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "120"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - ai-platform.example.com
      secretName: tls-secret
  rules:
    - host: ai-platform.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: gateway-service
                port:
                  number: 80
```

### 4. Knowledge Service (Example Backend)

```yaml
# kubernetes/knowledge/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: knowledge-service
  namespace: ai-platform
spec:
  replicas: 2
  selector:
    matchLabels:
      app: knowledge-service
  template:
    metadata:
      labels:
        app: knowledge-service
    spec:
      containers:
        - name: knowledge
          image: myacr.azurecr.io/ai-platform/knowledge:latest
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: knowledge-service
  namespace: ai-platform
spec:
  selector:
    app: knowledge-service
  ports:
    - port: 80
      targetPort: 8000
```

### 5. Secrets

```yaml
# kubernetes/secrets.yaml (DO NOT commit to git - use sealed secrets or external-secrets)
apiVersion: v1
kind: Secret
metadata:
  name: azure-openai-secrets
  namespace: ai-platform
type: Opaque
data:
  api-key: <base64-encoded-key>
  endpoint: <base64-encoded-endpoint>
```

In practice, use **Azure Key Vault with the CSI driver** instead of raw Kubernetes secrets.

### 6. Horizontal Pod Autoscaler

```yaml
# kubernetes/gateway/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: gateway-hpa
  namespace: ai-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: gateway
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### 7. Namespace

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ai-platform
  labels:
    project: ai-platform
    environment: production
```

## Azure Kubernetes Service (AKS) Specifics

### What AKS Adds on Top of K8s

| Feature | Description |
|---------|-------------|
| **Managed control plane** | Azure manages the K8s master nodes — you don't pay for or manage them |
| **Node pools** | Groups of VMs with specific configurations. Can have GPU nodes for ML. |
| **Azure integration** | Key Vault, ACR, Monitor, Active Directory — all integrated natively |
| **Cluster autoscaler** | Automatically adds/removes nodes based on demand |
| **Azure CNI networking** | Pods get Azure VNet IPs for native networking |

### AKS Cluster Configuration for Phase 6

```
AKS Cluster: ai-platform-cluster
├── System Node Pool
│   ├── VM Size: Standard_D2s_v5 (2 vCPU, 8 GB)
│   ├── Nodes: 2 (min 2, max 4)
│   └── Purpose: K8s system components
│
├── App Node Pool
│   ├── VM Size: Standard_D4s_v5 (4 vCPU, 16 GB)
│   ├── Nodes: 2 (min 1, max 6)
│   └── Purpose: AI service workloads
│
└── Integrations:
    ├── Azure Container Registry (ACR)
    ├── Azure Key Vault (secrets)
    ├── Azure Monitor (logging & metrics)
    └── Azure Managed Identity (auth)
```

### Creating an AKS Cluster

```bash
# Create resource group
az group create --name ai-platform-rg --location westeurope

# Create AKS cluster
az aks create \
  --resource-group ai-platform-rg \
  --name ai-platform-cluster \
  --node-count 2 \
  --node-vm-size Standard_D4s_v5 \
  --enable-managed-identity \
  --attach-acr myacr \
  --network-plugin azure \
  --generate-ssh-keys

# Get credentials
az aks get-credentials \
  --resource-group ai-platform-rg \
  --name ai-platform-cluster

# Verify connection
kubectl get nodes
```

## Essential kubectl Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `kubectl get pods` | List running pods | `kubectl get pods -n ai-platform` |
| `kubectl get services` | List services | `kubectl get svc -n ai-platform` |
| `kubectl get deployments` | List deployments | `kubectl get deploy -n ai-platform` |
| `kubectl describe pod` | Detailed pod info | `kubectl describe pod gateway-xxx` |
| `kubectl logs` | View pod logs | `kubectl logs -f gateway-xxx` |
| `kubectl apply -f` | Apply YAML manifest | `kubectl apply -f kubernetes/` |
| `kubectl delete -f` | Delete resources | `kubectl delete -f kubernetes/gateway.yaml` |
| `kubectl exec` | Run command in pod | `kubectl exec -it gateway-xxx -- bash` |
| `kubectl port-forward` | Forward port locally | `kubectl port-forward svc/gateway 8000:80` |
| `kubectl scale` | Scale deployment | `kubectl scale deploy gateway --replicas=5` |
| `kubectl rollout` | Manage rollouts | `kubectl rollout status deploy/gateway` |
| `kubectl rollout undo` | Rollback | `kubectl rollout undo deploy/gateway` |

## Service Communication in K8s

Inside a Kubernetes cluster, services communicate using DNS names:

```
Gateway (gateway-service) 
    │
    ├── http://knowledge-service/search     ← K8s DNS resolution
    ├── http://agent-service/chat           ← No external networking needed
    ├── http://research-service/start       ← Automatic load balancing
    └── http://assistant-service/analyze    ← Same namespace = simple names
```

In code:

```python
# In the gateway service, calling the knowledge service
import httpx

KNOWLEDGE_SERVICE_URL = os.getenv(
    "KNOWLEDGE_SERVICE_URL", 
    "http://knowledge-service"  # K8s DNS name
)

async def search_knowledge(query: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{KNOWLEDGE_SERVICE_URL}/search",
            json={"query": query},
            timeout=30.0,
        )
        return response.json()
```

## Resource Sizing for AI Services

```
Resource Guidelines for AI Services:
┌─────────────────────┬────────────┬────────────┬──────────┐
│ Service             │ CPU (req)  │ Memory     │ Replicas │
├─────────────────────┼────────────┼────────────┼──────────┤
│ Gateway             │ 250m-500m  │ 256-512Mi  │ 2-5      │
│ Knowledge Service   │ 500m-1000m │ 512Mi-1Gi  │ 2-4      │
│ Agent Service       │ 500m-1000m │ 512Mi-1Gi  │ 2-3      │
│ Research Service    │ 500m-1000m │ 512Mi-1Gi  │ 1-2      │
│ Assistant Service   │ 500m-1000m │ 512Mi-2Gi  │ 1-3      │
│ Redis               │ 100m-250m  │ 128-256Mi  │ 1        │
│ PostgreSQL          │ 250m-500m  │ 256-512Mi  │ 1        │
└─────────────────────┴────────────┴────────────┴──────────┘

m = millicores (1000m = 1 CPU core)
Mi = Mebibytes
```

**Note**: AI services are mostly I/O-bound (waiting for LLM APIs), so they don't need much CPU. Memory is more important for handling concurrent requests.

## Trade-offs

### Pros
- ✅ Automatic scaling, self-healing, rolling updates
- ✅ Service discovery (services find each other by name)
- ✅ Resource management (CPU/memory limits)
- ✅ Industry standard — enormous ecosystem and community
- ✅ AKS is managed — Azure handles the control plane

### Cons
- ❌ Steep learning curve — many concepts and YAML files
- ❌ Operational complexity — even managed K8s requires attention
- ❌ Overkill for small projects (but Phase 6 requires it)
- ❌ Cost: AKS nodes cost money even when idle
- ❌ YAML verbosity — lots of boilerplate

## When NOT to Use Kubernetes

For simpler deployments, consider **Azure Container Apps** — it provides Kubernetes-like scaling without the complexity. But Phase 6 specifically requires K8s/AKS.

## Enterprise Considerations

- **Scale**: Cluster autoscaler adds nodes automatically. HPA scales pods within nodes. Together they handle variable AI workloads.
- **Security**: RBAC, network policies, pod security standards, Azure AD integration, Key Vault CSI driver.
- **Cost**: AKS control plane is free. You pay for node VMs. Use spot instances for non-critical workloads to save ~60%.
- **Support**: Microsoft supports AKS with SLA. Massive K8s community.

## Decision

**Recommendation**: **Use** — AKS is required by Phase 6 specifications.

**Reasoning**: Kubernetes provides the orchestration needed for a multi-service AI platform. AKS removes the burden of managing the control plane. The learning value is significant for enterprise AI engineering.

**Next steps**:
1. Create AKS cluster with Azure CLI
2. Write K8s manifests for all services
3. Set up Azure Container Registry integration
4. Configure Ingress for external access
5. Set up monitoring with Azure Monitor

## Resources for Deeper Learning

- [Kubernetes Official Documentation](https://kubernetes.io/docs/home/) — Comprehensive reference
- [AKS Documentation](https://learn.microsoft.com/en-us/azure/aks/) — Azure-specific guides
- [Kubernetes the Hard Way](https://github.com/kelseyhightower/kubernetes-the-hard-way) — Deep understanding (optional)
- [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/) — Essential commands
- [AKS Best Practices](https://learn.microsoft.com/en-us/azure/aks/best-practices) — Production readiness
