---
date: 2026-02-27
type: technology
topic: "Azure Deployment Options for AI Platforms"
project: "Phase 6 - Enterprise AI Platform"
status: complete
decision: use (AKS as primary)
---

# Technology Brief: Azure Deployment Options for AI Platforms

## Quick Summary

| Aspect | Details |
|--------|---------|
| **What** | Azure services for hosting and managing containerized AI workloads |
| **For** | Deploying the Enterprise AI Platform to the cloud |
| **Maturity** | All options production-grade |
| **Decision** | **AKS** for the platform, with supporting Azure services |

## Azure Compute Options Compared

```
┌────────────────────────────────────────────────────────────────────┐
│              AZURE COMPUTE OPTIONS SPECTRUM                         │
│                                                                    │
│  More Control                                          Less Control │
│  More Complexity                                    Less Complexity │
│                                                                    │
│  ┌──────┐    ┌──────┐    ┌──────────┐    ┌──────────┐  ┌───────┐ │
│  │  VMs │    │ AKS  │    │ Container│    │ App      │  │Azure  │ │
│  │      │    │      │    │ Apps     │    │ Service  │  │Funct. │ │
│  │ IaaS │    │ CaaS │    │ Serverless│   │  PaaS    │  │ FaaS  │ │
│  │      │    │      │    │ Containers│   │          │  │       │ │
│  └──────┘    └──────┘    └──────────┘    └──────────┘  └───────┘ │
│                 ▲                                                   │
│                 │                                                   │
│            Phase 6                                                  │
│            choice                                                   │
└────────────────────────────────────────────────────────────────────┘
```

### Detailed Comparison

| Feature | AKS | Container Apps | App Service | Azure Functions |
|---------|-----|---------------|-------------|-----------------|
| **Control** | Full K8s control | Simplified containers | Managed platform | Serverless |
| **Scaling** | HPA + Cluster autoscaler | KEDA-based auto-scale | Auto-scale rules | Auto (per event) |
| **Networking** | Full VNet, Ingress | VNet integration | VNet integration | VNet (premium) |
| **Cost model** | Pay for VMs | Pay per vCPU/memory/s | Pay per plan | Pay per execution |
| **Min cost** | ~$100/month (2 nodes) | $0 (scale to zero) | ~$50/month (B1) | $0 (consumption) |
| **Complexity** | High | Medium | Low | Low |
| **Multi-container** | Native | Supported | Limited | No |
| **GPU support** | Yes | No | No | No |
| **Best for** | Complex platforms | Microservices, APIs | Web apps | Event processing |

### Why AKS for Phase 6

1. **Required by spec**: The README specifies "basic Kubernetes deployment (can be Azure Kubernetes Service)"
2. **Multi-service platform**: AKS natively supports running many interconnected services
3. **Full control**: Complete control over networking, scaling, deployment strategies
4. **Learning value**: Kubernetes skills are essential for enterprise AI engineering
5. **Observability integration**: Easy to run Prometheus/Grafana alongside your services

## Azure Services Architecture for Phase 6

```
┌────────────────────────────────────────────────────────────────────┐
│                    AZURE ARCHITECTURE                                │
│                                                                     │
│  ┌─── Azure Front Door / Application Gateway ───────────────────┐  │
│  │  SSL termination, WAF, DDoS protection, global load balancing │  │
│  └───────────────────────────┬───────────────────────────────────┘  │
│                              │                                      │
│  ┌─── AKS Cluster ──────────┼──────────────────────────────────┐   │
│  │                           │                                  │   │
│  │  ┌─ Ingress Controller ───┘                                 │   │
│  │  │  (NGINX)                                                  │   │
│  │  └────┬──────┬──────┬──────┬─────┘                          │   │
│  │       │      │      │      │                                 │   │
│  │  ┌────▼┐ ┌───▼──┐ ┌▼────┐ ┌▼─────┐ ┌──────┐               │   │
│  │  │Gate-│ │Know- │ │Agent│ │Resea-│ │Assis-│               │   │
│  │  │way  │ │ledge │ │     │ │rch   │ │tant  │               │   │
│  │  └─────┘ └──────┘ └─────┘ └──────┘ └──────┘               │   │
│  │                                                              │   │
│  │  ┌──────────┐ ┌──────────┐                                  │   │
│  │  │Prometheus│ │ Grafana  │                                  │   │
│  │  └──────────┘ └──────────┘                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─── Azure Services ──────────────────────────────────────────┐   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    │   │
│  │  │ Azure OpenAI │  │ Azure Cache  │  │ Azure Database │    │   │
│  │  │ (LLM API)    │  │ for Redis    │  │ for PostgreSQL │    │   │
│  │  └──────────────┘  └──────────────┘  └────────────────┘    │   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐    │   │
│  │  │ Azure Key    │  │ Azure        │  │ Azure Monitor  │    │   │
│  │  │ Vault        │  │ Container    │  │ / App Insights │    │   │
│  │  │ (Secrets)    │  │ Registry     │  │ (Logging)      │    │   │
│  │  └──────────────┘  └──────────────┘  └────────────────┘    │   │
│  │                                                              │   │
│  │  ┌──────────────┐  ┌──────────────┐                         │   │
│  │  │ Azure Blob   │  │ Azure VNet   │                         │   │
│  │  │ Storage      │  │ (Networking) │                         │   │
│  │  └──────────────┘  └──────────────┘                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

## Azure Services Needed

### Tier 1: Essential (Must Have)

| Service | Purpose | Monthly Cost Estimate |
|---------|---------|----------------------|
| **AKS** | Run containerized services | ~$100-200 (2-3 nodes, D4s_v5) |
| **Azure Container Registry** | Store Docker images | ~$5 (Basic tier) |
| **Azure OpenAI** | LLM API access | Variable (pay per token) |
| **Azure Key Vault** | Secrets management | ~$1 |

### Tier 2: Recommended

| Service | Purpose | Monthly Cost Estimate |
|---------|---------|----------------------|
| **Azure Database for PostgreSQL** | Cost tracking, user management | ~$15 (Flexible, Burstable B1ms) |
| **Azure Cache for Redis** | Rate limiting, caching, task queues | ~$15 (Basic C0) |
| **Azure Blob Storage** | Document storage for assistant | ~$2 |
| **Azure Monitor** | Infrastructure monitoring | Free tier with AKS |

### Tier 3: Nice to Have

| Service | Purpose | Monthly Cost Estimate |
|---------|---------|----------------------|
| **Azure Front Door** | Global load balancing, WAF | ~$35 |
| **Azure API Management** | Developer portal, advanced policies | ~$50 (Developer tier) |
| **Azure Log Analytics** | Centralized log storage | Pay per GB (~$2.76/GB) |

### Estimated Total Monthly Cost

```
Essential only:    ~$110-210/month + LLM token costs
With recommended:  ~$145-245/month + LLM token costs
Full stack:        ~$230-330/month + LLM token costs
```

## Azure Container Registry (ACR)

Store your Docker images close to your AKS cluster.

```bash
# Create ACR
az acr create \
  --resource-group ai-platform-rg \
  --name aiplatformacr \
  --sku Basic

# Attach ACR to AKS (allows AKS to pull images)
az aks update \
  --resource-group ai-platform-rg \
  --name ai-platform-cluster \
  --attach-acr aiplatformacr

# Build and push an image
az acr build \
  --registry aiplatformacr \
  --image ai-platform/gateway:v1.0 \
  ./gateway
```

## Azure Key Vault Integration

Store secrets securely and access them from AKS using the CSI driver.

```bash
# Create Key Vault
az keyvault create \
  --name ai-platform-kv \
  --resource-group ai-platform-rg \
  --location westeurope

# Add secrets
az keyvault secret set \
  --vault-name ai-platform-kv \
  --name "azure-openai-api-key" \
  --value "your-api-key-here"

# Enable Key Vault CSI driver on AKS
az aks enable-addons \
  --resource-group ai-platform-rg \
  --name ai-platform-cluster \
  --addons azure-keyvault-secrets-provider
```

K8s manifest to mount Key Vault secrets:

```yaml
# kubernetes/secrets-provider.yaml
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: azure-kv-secrets
  namespace: ai-platform
spec:
  provider: azure
  parameters:
    keyvaultName: "ai-platform-kv"
    objects: |
      array:
        - |
          objectName: azure-openai-api-key
          objectType: secret
        - |
          objectName: database-connection-string
          objectType: secret
    tenantId: "<your-tenant-id>"
  secretObjects:
    - secretName: azure-openai-secrets
      type: Opaque
      data:
        - objectName: azure-openai-api-key
          key: api-key
```

## Networking

### Virtual Network Setup

```
┌── Azure VNet: 10.0.0.0/16 ──────────────────────────────────────┐
│                                                                   │
│  ┌── AKS Subnet: 10.0.0.0/20 ──────────────────────────────┐   │
│  │  Pods and nodes get IPs from this subnet                  │   │
│  │  Service CIDR: 10.1.0.0/16 (internal K8s services)       │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌── Database Subnet: 10.0.16.0/24 ─────────────────────────┐  │
│  │  PostgreSQL Flexible Server                                │  │
│  │  Private endpoint                                          │  │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌── Redis Subnet: 10.0.17.0/24 ────────────────────────────┐  │
│  │  Azure Cache for Redis                                     │  │
│  │  Private endpoint                                          │  │
│  └───────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

## CI/CD with GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy to AKS

on:
  push:
    branches: [main]

env:
  ACR_NAME: aiplatformacr
  AKS_CLUSTER: ai-platform-cluster
  RESOURCE_GROUP: ai-platform-rg

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [gateway, knowledge, agent, research, assistant]
    steps:
      - uses: actions/checkout@v4
      
      - name: Login to ACR
        uses: azure/docker-login@v2
        with:
          login-server: ${{ env.ACR_NAME }}.azurecr.io
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}
      
      - name: Build and push
        run: |
          docker build -t ${{ env.ACR_NAME }}.azurecr.io/ai-platform/${{ matrix.service }}:${{ github.sha }} \
            ./${{ matrix.service }}
          docker push ${{ env.ACR_NAME }}.azurecr.io/ai-platform/${{ matrix.service }}:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set AKS context
        uses: azure/aks-set-context@v3
        with:
          resource-group: ${{ env.RESOURCE_GROUP }}
          cluster-name: ${{ env.AKS_CLUSTER }}
      
      - name: Deploy to AKS
        run: |
          # Update image tags in manifests
          for service in gateway knowledge agent research assistant; do
            kubectl set image deployment/$service \
              $service=${{ env.ACR_NAME }}.azurecr.io/ai-platform/$service:${{ github.sha }} \
              -n ai-platform
          done
      
      - name: Verify deployment
        run: |
          kubectl rollout status deployment/gateway -n ai-platform --timeout=300s
```

## Infrastructure as Code (IaC)

Use Bicep or Terraform to define Azure infrastructure. Here's a simplified Bicep example:

```bicep
// main.bicep - Core infrastructure
param location string = resourceGroup().location
param clusterName string = 'ai-platform-cluster'
param acrName string = 'aiplatformacr'

// Azure Container Registry
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
}

// AKS Cluster
resource aksCluster 'Microsoft.ContainerService/managedClusters@2024-01-01' = {
  name: clusterName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    dnsPrefix: clusterName
    agentPoolProfiles: [
      {
        name: 'system'
        count: 2
        vmSize: 'Standard_D2s_v5'
        mode: 'System'
      }
      {
        name: 'app'
        count: 2
        vmSize: 'Standard_D4s_v5'
        mode: 'User'
        enableAutoScaling: true
        minCount: 1
        maxCount: 6
      }
    ]
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'ai-platform-kv'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
    enableRbacAuthorization: true
  }
}
```

## Cost Optimization Tips

1. **Use B-series VMs for dev/test**: Burstable, significantly cheaper
2. **Spot instances for non-critical workloads**: Up to 60% savings
3. **Scale to zero**: Use KEDA or cluster autoscaler to minimize idle nodes
4. **Reserved instances**: If running long-term, reserve VMs for 30-60% savings
5. **Azure cost alerts**: Set up budget alerts at $50, $100, $200 thresholds
6. **Right-size everything**: Start small, scale up based on actual usage

```bash
# Set up budget alert
az consumption budget create \
  --amount 200 \
  --budget-name "ai-platform-monthly" \
  --category Cost \
  --resource-group ai-platform-rg \
  --time-grain Monthly \
  --start-date 2026-03-01 \
  --end-date 2027-03-01
```

## Decision

**Recommendation**: **Use AKS** as the primary compute platform, with ACR, Key Vault, PostgreSQL, and Redis as supporting services.

**Reasoning**: AKS is required by the spec, provides full Kubernetes capabilities, integrates well with Azure services, and offers great learning value. Start with a small cluster and scale up.

**Next steps**:
1. Create Azure resource group and core services (ACR, AKS, Key Vault)
2. Configure AKS with node pools and RBAC
3. Set up CI/CD pipeline with GitHub Actions
4. Deploy initial services to staging namespace
5. Set up monitoring and alerts

## Resources for Deeper Learning

- [AKS Documentation](https://learn.microsoft.com/en-us/azure/aks/) — Comprehensive reference
- [AKS Best Practices](https://learn.microsoft.com/en-us/azure/aks/best-practices) — Production readiness
- [Azure Architecture Center](https://learn.microsoft.com/en-us/azure/architecture/) — Reference architectures
- [Azure Pricing Calculator](https://azure.microsoft.com/en-us/pricing/calculator/) — Estimate costs
- [Azure Bicep docs](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/) — Infrastructure as Code
