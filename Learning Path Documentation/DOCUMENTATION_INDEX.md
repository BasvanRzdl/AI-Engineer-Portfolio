# 📚 Documentation Index

> **Quick-reference index of every document in the AI Engineer Learning Path.**
> Use this page to find the right doc fast — browse by phase, or search by topic.

---

## Table of Contents

- [Learning Path Documentation](#learning-path-documentation)
- [Workflow Prompts](#workflow-prompts)
- [Phase 1 — Prompt Engineering Laboratory](#phase-1--prompt-engineering-laboratory)
- [Phase 2 — Enterprise Document Intelligence Platform](#phase-2--enterprise-document-intelligence-platform)
- [Phase 3 — Autonomous Customer Operations Agent](#phase-3--autonomous-customer-operations-agent)
- [Phase 4 — AI Strategy Research Team](#phase-4--ai-strategy-research-team)
- [Phase 5 — Multi-Modal Enterprise Assistant](#phase-5--multi-modal-enterprise-assistant)
- [Phase 6 — Enterprise AI Platform (Capstone)](#phase-6--enterprise-ai-platform-capstone)
- [Cross-Phase Topic Finder](#cross-phase-topic-finder)

---

## Learning Path Documentation

General guides that apply across the entire program.

| Document | Description |
|----------|-------------|
| [LEARNING_PLAN.md](../Learning%20Path%20Documentation/LEARNING_PLAN.md) | The master plan — a 12-week, ~240-hour program designed around solving enterprise problems to build senior-level competency in GenAI/Agentic systems. |
| [PROGRESS.md](../Learning%20Path%20Documentation/PROGRESS.md) | Tracks completion status, hours spent, and dates for each of the 6 project phases. |
| [INSTRUCTOR_GUIDE.md](../Learning%20Path%20Documentation/INSTRUCTOR_GUIDE.md) | Explains the AI instructor's role — to clarify, unblock, review, and challenge — not to write code for the learner. |
| [AI_ASSISTED_CODING.md](../Learning%20Path%20Documentation/AI_ASSISTED_CODING.md) | Personal methodology for AI-assisted development with GitHub Copilot, Claude Code, and other AI coding assistants. |
| [GIT_BEST_PRACTICES.md](../Learning%20Path%20Documentation/GIT_BEST_PRACTICES.md) | Professional Git workflows and practices (branching, commits, PRs) used in enterprise software development. |

---

## Workflow Prompts

Structured prompts that guide each step of the development workflow.

| Document | Description |
|----------|-------------|
| [00_RESEARCH.md](../Workflow%20Prompts/00_RESEARCH.md) | **Research** — Build foundational understanding through comprehensive research before implementation. |
| [01_EXPLORE.md](../Workflow%20Prompts/01_EXPLORE.md) | **Explore** — Understand the codebase, identify relevant patterns, and gather context (read-only, no code changes). |
| [02_PLAN.md](../Workflow%20Prompts/02_PLAN.md) | **Plan** — Create a detailed, actionable implementation plan before writing code. |
| [03_IMPLEMENT.md](../Workflow%20Prompts/03_IMPLEMENT.md) | **Implement** — Execute the plan systematically with verification at each step. |
| [04_VERIFY.md](../Workflow%20Prompts/04_VERIFY.md) | **Verify** — Validate that implementation meets requirements, verify quality, and document completion. |

---

## Phase 1 — Prompt Engineering Laboratory

> *Build a systematic prompt engineering approach for a Fortune 500 financial services company.*

| # | Type | Document | Description |
|---|------|----------|-------------|
| — | 📋 | [README.md](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/README.md) | Phase overview, objectives, and project brief. |
| 01 | 📖 Concept | [Prompt Engineering Fundamentals](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/01-concept-prompt-engineering.md) | Designing, structuring, and optimizing LLM prompts as first-class software artifacts — versioned, tested, and evaluated like code. |
| 02 | 📖 Concept | [Structured Outputs & Output Parsing](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/02-concept-structured-outputs.md) | Getting LLMs to return predictable, machine-parseable formats (JSON) using prompt-based, JSON Mode, and schema-enforced approaches. |
| 03 | 📖 Concept | [LLM Evaluation Methodologies](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/03-concept-llm-evaluation.md) | Systematically measuring model output quality using automated metrics, LLM-as-judge, and heuristic checks. |
| 04 | 📖 Concept | [Prompt Versioning & A/B Testing](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/04-concept-prompt-versioning.md) | Treating prompts as versioned code with semantic versioning and data-driven A/B testing for comparison. |
| 05 | 🔧 Technology | [OpenAI & Azure OpenAI API](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/05-technology-openai-azure-api.md) | Both APIs with Azure's enterprise features (data residency, SLAs, content filtering) and provider abstraction. |
| 06 | 🔧 Technology | [Evaluation Frameworks](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/06-technology-evaluation-frameworks.md) | Comparing Ragas, Promptfoo, and DeepEval — three battle-tested evaluation frameworks. |
| 07 | 🔧 Technology | [Prompt Templating Approaches](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/07-technology-prompt-templating.md) | Prompt template mechanisms for reusable, parameterized artifacts with variable injection and conditional logic. |
| 08 | 🏗️ Architecture | [Prompt Engineering Framework](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/08-architecture-prompt-framework.md) | Concrete architecture defining package structure, core abstractions, and data flow for the project. |

---

## Phase 2 — Enterprise Document Intelligence Platform

> *Build a production-grade RAG system that retrieves from a global consulting firm's knowledge base.*

| # | Type | Document | Description |
|---|------|----------|-------------|
| — | 📋 | [README.md](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/README.md) | Phase overview, objectives, and project brief. |
| 01 | 📖 Concept | [RAG Architecture Fundamentals](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/01-concept-rag-architecture.md) | How Retrieval-Augmented Generation makes LLMs useful for specific data by retrieving documents and injecting them as context. |
| 02 | 📖 Concept | [Chunking Strategies for RAG](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/02-concept-chunking-strategies.md) | Splitting documents into smaller embeddable pieces — the single most impactful design decision in a RAG pipeline. |
| 03 | 📖 Concept | [Embeddings and Vector Search](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/03-concept-embeddings-and-vector-search.md) | How embedding models convert text into dense vectors enabling concept-based semantic search. |
| 04 | 📖 Concept | [Advanced Retrieval Strategies](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/04-concept-retrieval-strategies.md) | Techniques beyond naive vector similarity — including hybrid search, re-ranking, and cross-references. |
| 05 | 📖 Concept | [Generation, Grounding & Hallucination Mitigation](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/05-concept-generation-and-grounding.md) | Forcing LLMs to base answers on retrieved evidence with source attribution to prevent hallucination. |
| 06 | 📖 Concept | [RAG Evaluation](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/06-concept-rag-evaluation.md) | Measuring RAG performance at every pipeline stage — retrieval quality, generation faithfulness, and end-to-end correctness. |
| 07 | 🔧 Technology | [Vector Databases](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/07-technology-vector-databases.md) | Evaluating vector DB options with Qdrant as primary and Azure AI Search as the enterprise alternative. |
| 08 | 🔧 Technology | [Document Processing & Ingestion](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/08-technology-document-processing.md) | Document parsing with Unstructured as primary parser and specialized libraries as fallback. |
| 09 | 🔧 Technology | [LangChain for RAG](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/09-technology-langchain-rag.md) | Using LangChain as the primary RAG framework with LCEL (LangChain Expression Language) patterns. |
| 10 | 🔧 Technology | [Embedding Models Comparison](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/10-technology-embedding-models.md) | Comparing embedding models starting with OpenAI text-embedding-3-small and benchmarking against Cohere embed-v3. |
| 11 | 🏗️ Architecture | [End-to-End RAG Pipeline](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/11-architecture-rag-pipeline.md) | Complete architecture connecting all researched concepts and technologies for the Document Intelligence Platform. |

---

## Phase 3 — Autonomous Customer Operations Agent

> *Build an operational AI agent for e-commerce that handles orders, refunds, address changes, and human escalation.*

| # | Type | Document | Description |
|---|------|----------|-------------|
| — | 📋 | [README.md](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/README.md) | Phase overview, objectives, and project brief. |
| 01 | 📖 Concept | [Agent Architectures](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/01-concept-agent-architectures.md) | How AI agents reason, act via tools, and observe results in a loop — trade-offs between autonomy, reliability, and interpretability. |
| 02 | 📖 Concept | [State Machines for Conversational Agents](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/02-concept-state-machines.md) | Modeling agent behavior as states and transitions for structure and predictability in conversational workflows. |
| 03 | 📖 Concept | [Tool Use and Function Calling](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/03-concept-tool-use-and-function-calling.md) | The mechanism by which LLMs interact with the outside world by outputting structured function call requests. |
| 04 | 📖 Concept | [Memory Systems for Agents](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/04-concept-memory-systems.md) | Short-term and long-term memory that allows agents to be coherent over time across conversations. |
| 05 | 📖 Concept | [Guardrails and Safety](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/05-concept-guardrails-and-safety.md) | Constraints, checks, and safety mechanisms that prevent AI agents with real-world capabilities from causing harm. |
| 06 | 📖 Concept | [Agent Evaluation and Observability](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/06-concept-evaluation-and-observability.md) | Measuring whether an agent is doing a good job and understanding what it's actually doing and why. |
| 07 | 🔧 Technology | [LangGraph Deep Dive](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/07-technology-langgraph-deep-dive.md) | LangGraph as a framework for stateful, multi-step AI applications using graph-based orchestration. |
| 08 | 🔧 Technology | [Tool Definition Patterns](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/08-technology-tool-definition-patterns.md) | Patterns for defining agent tools (Python functions) with high-quality schemas that directly affect performance. |
| 09 | 🔧 Technology | [Human-in-the-Loop Implementations](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/09-technology-human-in-the-loop.md) | How agents pause execution, present info to a human, wait for input, and resume based on the decision. |
| 10 | 🔧 Technology | [Conversation Memory Implementations](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/10-technology-conversation-memory.md) | Practical implementation of short-term (checkpointers) and long-term (Store API) memory in LangGraph. |
| 11 | 🏗️ Architecture | [Customer Operations Agent Design](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/11-architecture-agent-design.md) | Concrete architecture blueprint synthesizing all research for the Customer Operations Agent. |

---

## Phase 4 — AI Strategy Research Team

> *Build a team of specialized AI agents that collaborate like a real consulting team for strategy research.*

| # | Type | Document | Description |
|---|------|----------|-------------|
| — | 📋 | [README.md](../Phase%204%20-%20AI%20Strategy%20Research%20Team/README.md) | Phase overview, objectives, and project brief. |
| 01 | 📖 Concept | [Multi-Agent Architecture Patterns](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/01-concept-multi-agent-architecture-patterns.md) | How multiple specialized AI agents collaborate using patterns like hierarchical teams for complex tasks. |
| 02 | 📖 Concept | [Agent Communication & Coordination](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/02-concept-agent-communication-coordination.md) | How agents in a multi-agent system talk and work together — the coordination "glue" that makes MAS functional. |
| 03 | 📖 Concept | [Task Decomposition & Delegation](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/03-concept-task-decomposition-delegation.md) | Breaking complex goals into well-defined subtasks and assigning them to the right agent — the most critical MAS capability. |
| 04 | 📖 Concept | [Agent Specialization & Role Design](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/04-concept-agent-specialization-role-design.md) | Designing each agent with a focused role — its own system prompt, tools, and behavioral constraints. |
| 05 | 📖 Concept | [Observability & Debugging MAS](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/05-concept-observability-debugging.md) | Understanding what's happening inside multi-agent systems using logs, traces, metrics, and state snapshots. |
| 06 | 📖 Concept | [Quality Assurance & Evaluation for MAS](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/06-concept-quality-evaluation.md) | Evaluating whether multi-agent output is actually good — challenges of subjectivity and pipeline complexity. |
| 07 | 🔧 Technology | [LangGraph for Multi-Agent](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/07-technology-langgraph.md) | LangGraph as the low-level orchestration framework for stateful multi-agent applications as graphs. |
| 08 | 🔧 Technology | [Microsoft AutoGen](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/08-technology-microsoft-autogen.md) | Microsoft's open-source multi-agent framework with high-level AgentChat and low-level core APIs. |
| 09 | 🔧 Technology | [Google Agent Development Kit (ADK)](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/09-technology-google-adk.md) | Google's open-source, model-agnostic framework for building AI agents and multi-agent systems. |
| 10 | 🔧 Technology | [Framework Comparison](../Phase%204%20-%20AI%20Strategy%20Research%20Team/docs/10-technology-framework-comparison.md) | Structured comparison of LangGraph vs. AutoGen vs. Google ADK with recommendation. |

---

## Phase 5 — Multi-Modal Enterprise Assistant

> *Build an AI assistant that works across modalities — vision, audio, and multi-modal reasoning.*

| # | Type | Document | Description |
|---|------|----------|-------------|
| — | 📋 | [README.md](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/README.md) | Phase overview, objectives, and project brief. |
| 01 | 📖 Concept | [Vision-Language Models (VLMs)](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/01-concept-vision-language-models.md) | AI models that simultaneously understand images and text by combining a visual encoder with a language model. |
| 02 | 📖 Concept | [Multi-Modal Reasoning & Architectures](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/02-concept-multi-modal-reasoning.md) | Combining information from different modalities (text, images, audio, video) into unified understanding. |
| 03 | 📖 Concept | [Audio Processing Pipelines](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/03-concept-audio-processing-pipelines.md) | Full pipeline from raw audio to enterprise insights: preprocessing, transcription (Whisper), speaker ID, and summarization. |
| 04 | 📖 Concept | [Fine-Tuning Strategies](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/04-concept-fine-tuning-strategies.md) | Trade-offs of further training pre-trained models on your own data to specialize them for specific tasks. |
| 05 | 📖 Concept | [Multi-Modal RAG Patterns](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/05-concept-multi-modal-rag.md) | Extending RAG beyond text to retrieve and reason over images, tables, charts, and audio alongside documents. |
| 06 | 📖 Concept | [Enterprise Multi-Modal AI Patterns](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/06-concept-enterprise-multi-modal-patterns.md) | Architectural patterns and operational concerns (privacy, cost, reliability, scale) for production multi-modal systems. |
| 07 | 🔧 Technology | [GPT-4 Vision & Azure AI Vision](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/07-technology-gpt4-vision-azure-ai-vision.md) | GPT-4 Vision (via GPT-4o) for open-ended visual reasoning and Azure AI Vision for classical CV tasks (OCR, image analysis). |
| 08 | 🔧 Technology | [Whisper & Azure Speech Services](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/08-technology-whisper-azure-speech.md) | Two main Azure speech-to-text services for converting spoken audio into text for LLM reasoning. |
| 09 | 🔧 Technology | [Fine-Tuning with Azure OpenAI & HuggingFace](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/09-technology-fine-tuning-azure-ml-huggingface.md) | Practical tooling and workflows for fine-tuning models using Azure OpenAI and HuggingFace approaches. |
| 10 | 🔧 Technology | [Azure AI Foundry](../Phase%205%20-%20Multi-Modal%20Enterprise%20Assistant/docs/10-technology-azure-ai-foundry.md) | Microsoft's unified platform (formerly Azure AI Studio) for building, evaluating, and deploying AI applications. |

---

## Phase 6 — Enterprise AI Platform (Capstone)

> *Build a unified enterprise AI platform exposing AI capabilities as services with cost management, security, and observability.*

| # | Type | Document | Description |
|---|------|----------|-------------|
| — | 📋 | [README.md](../Phase%206%20-%20Enterprise%20AI%20Platform/README.md) | Phase overview, objectives, and project brief. |
| 01 | 📖 Concept | [API Gateway Patterns for AI Services](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/01-concept-api-gateway-patterns.md) | Single entry point between clients and internal AI services for auth, cost tracking, rate limiting, and routing. |
| 02 | 📖 Concept | [Microservices Architecture for AI](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/02-concept-microservices-for-ai.md) | Breaking AI platforms into independently deployable services with their own resources and scaling rules. |
| 03 | 📖 Concept | [Observability for LLM Systems](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/03-concept-observability-for-llm-systems.md) | Unique challenges of observing non-deterministic LLM systems — tracking quality, cost, and behavior. |
| 04 | 📖 Concept | [Security Patterns for LLM Applications](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/04-concept-security-patterns-for-llm-applications.md) | Prompt injection, data poisoning, sensitive disclosure, and excessive AI agency — the new LLM security domain. |
| 05 | 📖 Concept | [Cost Allocation and Tracking](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/05-concept-cost-allocation-and-tracking.md) | Tracking per-request AI costs, attributing them to business units/projects, and enforcing budgets in real-time. |
| 06 | 📖 Concept | [Deployment Strategies for AI Systems](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/06-concept-deployment-strategies.md) | Safely deploying code and models/prompts to production with gradual rollouts, testing, and quick rollback. |
| 07 | 🔧 Technology | [FastAPI for Enterprise AI Services](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/07-technology-fastapi-for-ai-services.md) | Using FastAPI as the web framework for building enterprise AI service endpoints. |
| 08 | 🔧 Technology | [Docker Containerization for AI](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/08-technology-docker-containerization.md) | Containerizing AI services with Docker for consistent, reproducible deployments. |
| 09 | 🔧 Technology | [Kubernetes and AKS](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/09-technology-kubernetes-and-aks.md) | Kubernetes and Azure Kubernetes Service for orchestrating containerized AI service deployments at scale. |
| 10 | 🔧 Technology | [Observability Stack](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/10-technology-observability-stack.md) | Prometheus, Grafana, and OpenTelemetry as the observability stack for monitoring AI platform health. |
| 11 | 🔧 Technology | [Azure Deployment Options](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/11-technology-azure-deployment-options.md) | Azure deployment options with AKS as the primary choice for hosting the enterprise AI platform. |
| 12 | 🏗️ Architecture | [Enterprise AI Platform Reference Architecture](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/12-architecture-enterprise-ai-platform.md) | Unified, actionable reference architecture blueprint synthesizing all Phase 6 research. |

---

## Cross-Phase Topic Finder

Looking for a topic that spans multiple phases? Use this quick-reference.

### 🔑 Core AI Concepts
| Topic | Where to Look |
|-------|---------------|
| Prompt Engineering | [Phase 1 — Concepts 01-04](#phase-1--prompt-engineering-laboratory) |
| RAG / Retrieval-Augmented Generation | [Phase 2 — Concepts 01-06](#phase-2--enterprise-document-intelligence-platform) |
| Agents & Agentic Systems | [Phase 3 — Concepts 01-06](#phase-3--autonomous-customer-operations-agent) |
| Multi-Agent Systems | [Phase 4 — Concepts 01-06](#phase-4--ai-strategy-research-team) |
| Multi-Modal AI | [Phase 5 — Concepts 01-06](#phase-5--multi-modal-enterprise-assistant) |
| Enterprise Platform Design | [Phase 6 — Concepts 01-06](#phase-6--enterprise-ai-platform-capstone) |

### 🔧 Recurring Technologies
| Technology | Phases |
|------------|--------|
| LangChain / LCEL | Phase 2 (doc 09) |
| LangGraph | Phase 3 (doc 07), Phase 4 (doc 07) |
| OpenAI / Azure OpenAI API | Phase 1 (doc 05), Phase 5 (doc 07, 09) |
| Evaluation Frameworks | Phase 1 (doc 06), Phase 2 (doc 06), Phase 3 (doc 06), Phase 4 (doc 06) |
| Azure AI Services | Phase 5 (docs 07, 08, 10), Phase 6 (doc 11) |
| Docker & Kubernetes | Phase 6 (docs 08, 09) |
| Observability | Phase 3 (doc 06), Phase 4 (doc 05), Phase 6 (docs 03, 10) |

### 🏗️ Architecture Documents
| Architecture | Phase |
|--------------|-------|
| Prompt Engineering Framework | [Phase 1 — doc 08](../Phase%201%20-%20Prompt%20Engineering%20Laboratory/docs/08-architecture-prompt-framework.md) |
| End-to-End RAG Pipeline | [Phase 2 — doc 11](../Phase%202%20-%20Enterprise%20Document%20Intelligence%20Platform/docs/11-architecture-rag-pipeline.md) |
| Customer Operations Agent Design | [Phase 3 — doc 11](../Phase%203%20-%20Autonomous%20Customer%20Operations%20Agent/docs/11-architecture-agent-design.md) |
| Enterprise AI Platform Reference | [Phase 6 — doc 12](../Phase%206%20-%20Enterprise%20AI%20Platform/docs/12-architecture-enterprise-ai-platform.md) |

---

> **Tip:** Each phase follows the same doc structure: **Concepts** (theory & principles) → **Technologies** (tools & frameworks) → **Architecture** (synthesis into a design). Start with concepts to understand *why*, then technologies to understand *how*, and finish with architecture to see *the plan*.
