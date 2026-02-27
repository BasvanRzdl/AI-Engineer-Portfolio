---
date: 2025-02-27
type: architecture
topic: "Prompt Engineering Framework Architecture"
project: "Phase 1 - Prompt Engineering Laboratory"
status: complete
---

# Architecture: Prompt Engineering Framework

## In My Own Words

This document synthesizes all the research into a concrete architecture for the Prompt Engineering Laboratory. It defines the package structure, core abstractions, data flow, and how all the pieces connect. This is the blueprint for implementation.

## Why This Matters

- Defines the clear scope of what we're building
- Establishes the package structure before writing code
- Maps deliverables to code modules
- Identifies interfaces between components
- Creates a shared vocabulary for the codebase

---

## Project Deliverables → Architecture Mapping

| Deliverable | Module(s) | Description |
|-------------|-----------|-------------|
| Prompt Management System | `templates/`, `registry/` | Template engine, version management, storage |
| Library of Evaluated Prompts | `library/`, `examples/` | Pre-built prompt templates with test cases |
| Evaluation Framework | `evaluation/`, `metrics/` | Evaluators, metrics, test runners |
| Documentation Playbook | `docs/` | Best practices, API reference (this research + generated) |

---

## Package Structure

```
prompt_engineering_lab/
├── __init__.py
├── py.typed                    # PEP 561 type hints marker
│
├── core/                       # Core abstractions
│   ├── __init__.py
│   ├── types.py               # Shared types and enums
│   ├── config.py              # Configuration management
│   └── exceptions.py          # Custom exceptions
│
├── providers/                  # LLM provider abstraction
│   ├── __init__.py
│   ├── base.py                # LLMProvider abstract base
│   ├── openai_provider.py     # OpenAI implementation
│   ├── azure_provider.py      # Azure OpenAI implementation
│   └── mock_provider.py       # Mock for testing
│
├── templates/                  # Prompt template engine
│   ├── __init__.py
│   ├── template.py            # PromptTemplate class
│   ├── renderer.py            # Jinja2-based rendering
│   └── loader.py              # Load templates from files
│
├── registry/                   # Prompt versioning & registry
│   ├── __init__.py
│   ├── version.py             # Semantic versioning
│   ├── registry.py            # PromptRegistry (store, retrieve, list)
│   └── storage.py             # Storage backends (YAML files, in-memory)
│
├── evaluation/                 # Evaluation framework
│   ├── __init__.py
│   ├── evaluator.py           # Main evaluator orchestrator
│   ├── test_case.py           # TestCase and TestSuite models
│   ├── runner.py              # Test runner (parallel execution)
│   └── reporter.py            # Generate evaluation reports
│
├── metrics/                    # Evaluation metrics
│   ├── __init__.py
│   ├── base.py                # BaseMetric abstract class
│   ├── heuristic.py           # Heuristic metrics (length, format, etc.)
│   ├── llm_judge.py           # LLM-as-judge metrics
│   ├── reference.py           # Reference-based (similarity, BLEU, ROUGE)
│   └── composite.py           # Composite/aggregated metrics
│
├── library/                    # Pre-built prompt library
│   ├── __init__.py
│   ├── summarization.py       # Summarization prompts
│   ├── extraction.py          # Entity/data extraction prompts
│   ├── classification.py      # Classification prompts
│   ├── rewriting.py           # Text rewriting prompts
│   └── chain_of_thought.py    # CoT reasoning prompts
│
├── tracking/                   # Cost & usage tracking
│   ├── __init__.py
│   ├── cost_tracker.py        # CostTracker
│   ├── budget.py              # BudgetManager
│   └── usage_logger.py        # Structured logging
│
└── ab_testing/                 # A/B testing framework
    ├── __init__.py
    ├── experiment.py           # ABExperiment definition
    ├── runner.py               # ABTestRunner
    └── analysis.py             # Statistical analysis & reporting
```

### Supporting Files

```
prompt_engineering_lab/
├── prompts/                    # YAML prompt storage (data, not code)
│   ├── summarization/
│   │   ├── v1.0.0.yaml
│   │   └── v1.1.0.yaml
│   ├── classification/
│   │   └── v1.0.0.yaml
│   └── extraction/
│       └── v1.0.0.yaml
│
├── tests/                      # Unit and integration tests
│   ├── __init__.py
│   ├── conftest.py            # Shared fixtures
│   ├── test_templates.py
│   ├── test_registry.py
│   ├── test_evaluation.py
│   ├── test_metrics.py
│   ├── test_providers.py
│   ├── test_tracking.py
│   └── test_ab_testing.py
│
├── examples/                   # Usage examples
│   ├── basic_usage.py
│   ├── evaluation_run.py
│   ├── ab_test_example.py
│   └── custom_metric.py
│
├── docs/                       # Research & documentation
│   ├── 01-concept-prompt-engineering.md
│   ├── ... (this research)
│   └── api_reference.md
│
├── pyproject.toml              # Package configuration
├── README.md                   # Project README
└── .env.example                # Environment variable template
```

---

## Core Abstractions

### 1. PromptTemplate

The central data structure — a versioned, parameterized prompt:

```python
class PromptTemplate(BaseModel):
    """A versioned prompt template."""
    
    # Identity
    name: str                          # "document_summarizer"
    version: str                       # "1.2.0"
    description: str = ""
    
    # Template content
    system_template: str               # Jinja2 template for system message
    user_template: str                 # Jinja2 template for user message
    variables: list[str]               # Required template variables
    
    # Model configuration
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int | None = None
    response_format: str = "text"      # "text", "json", "structured"
    
    # Output specification
    output_schema: type[BaseModel] | None = None
    
    # Few-shot examples
    examples: list[PromptExample] = []
    
    # Metadata
    tags: list[str] = []
    author: str = ""
    changelog: str = ""
    created_at: datetime = datetime.now()
    
    def render(self, **kwargs) -> list[dict]:
        """Render into OpenAI messages format."""
        ...
    
    def validate_variables(self, **kwargs) -> bool:
        """Check all required variables are provided."""
        ...
```

### 2. LLMProvider

Abstract base for model providers:

```python
class LLMProvider(ABC):
    """Abstract LLM provider."""
    
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> CompletionResult:
        """Send a completion request."""
        ...
    
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List supported models."""
        ...


@dataclass
class CompletionResult:
    """Standardized completion result across providers."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float
    finish_reason: str
    raw_response: Any = None
```

### 3. Metric

Abstract base for evaluation metrics:

```python
class BaseMetric(ABC):
    """Abstract evaluation metric."""
    
    name: str
    description: str
    
    @abstractmethod
    async def score(
        self,
        input_text: str,
        output_text: str,
        reference: str | None = None,
        context: dict | None = None,
    ) -> MetricResult:
        """Score a single output."""
        ...


@dataclass
class MetricResult:
    """Result of a metric evaluation."""
    metric_name: str
    score: float           # 0.0 to 1.0
    passed: bool           # Based on threshold
    details: dict = field(default_factory=dict)
    reasoning: str = ""
```

### 4. TestCase / TestSuite

Evaluation test structures:

```python
class TestCase(BaseModel):
    """A single evaluation test case."""
    id: str
    name: str
    input_text: str
    input_variables: dict = {}
    reference_output: str | None = None
    required_elements: list[str] = []
    forbidden_elements: list[str] = []
    category: str = ""
    tags: list[str] = []

class TestSuite(BaseModel):
    """A collection of test cases."""
    name: str
    description: str = ""
    test_cases: list[TestCase]
    metrics: list[str]        # Metric names to apply
    pass_threshold: float = 0.7
```

---

## Data Flow

### Prompt Execution Flow

```
User Request
     │
     ▼
┌─────────────────┐     ┌──────────────────┐
│ Template Loader  │────▶│  PromptTemplate  │
│ (YAML → Model)  │     │  (Pydantic)      │
└─────────────────┘     └────────┬─────────┘
                                 │ .render(**vars)
                                 ▼
                        ┌──────────────────┐
                        │   Messages List  │
                        │   [sys, user]    │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐     ┌──────────────┐
                        │  LLM Provider    │────▶│ Cost Tracker │
                        │  (OpenAI/Azure)  │     │              │
                        └────────┬─────────┘     └──────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ CompletionResult │
                        │ (content, usage) │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Output Parser    │
                        │ (Pydantic model) │
                        └──────────────────┘
```

### Evaluation Flow

```
TestSuite + PromptTemplate
           │
           ▼
    ┌──────────────┐
    │  Test Runner  │─────── Parallel execution (asyncio.Semaphore)
    │              │
    └──────┬───────┘
           │ For each TestCase:
           │   1. Render template with test variables
           │   2. Call LLM Provider
           │   3. Run all Metrics
           │
           ▼
    ┌──────────────┐
    │  Results     │
    │  Collection  │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Reporter    │───── Summary stats, per-case details, cost breakdown
    └──────────────┘
```

### A/B Testing Flow

```
ABExperiment (Template A, Template B, TestSuite)
           │
           ▼
    ┌──────────────┐
    │  AB Runner   │───── Runs both templates on same test suite
    └──────┬───────┘
           │
           ▼
    ┌──────────────────┐
    │  Paired Results  │
    │  A[i] vs B[i]   │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │  Statistical     │───── Paired t-test, effect size, confidence
    │  Analysis        │
    └──────┬───────────┘
           │
           ▼
    ┌──────────────────┐
    │  AB Test Report  │───── Winner, significance, recommendations
    └──────────────────┘
```

---

## Key Design Decisions

### 1. Async-First

All API interactions are async. The framework uses `asyncio` throughout:

```python
# User code
async def main():
    lab = PromptLab(provider=OpenAIProvider(api_key="..."))
    
    template = lab.registry.get("summarizer", version="latest")
    result = await lab.run(template, document_text="...")
    
    # Evaluation
    report = await lab.evaluate(template, test_suite=my_tests)
```

**Rationale**: Evaluation runs need parallel API calls. Starting sync and adding async later is painful.

### 2. Pydantic Models Everywhere

All data structures are Pydantic BaseModel:

**Rationale**: Type safety, validation, JSON Schema generation, serialization — all from one source.

### 3. Provider Abstraction

Never import `openai` directly in business logic:

**Rationale**: Multi-provider support (2+ LLMs required), testability with mocks, future flexibility.

### 4. YAML for Prompt Storage

Templates stored as YAML files alongside code:

**Rationale**: Human-readable, diffable in git, loadable without Python, works with any editor.

### 5. Metric Composition

Metrics are independent, composable units:

```python
# Individual metrics
accuracy = LLMJudgeMetric(name="accuracy", criteria="Is it accurate?")
format_check = FormatMetric(expected_format="json")

# Composite
quality = CompositeMetric(
    name="quality",
    metrics=[accuracy, format_check],
    weights=[0.7, 0.3]
)
```

**Rationale**: Different prompts need different metrics. Composition > inheritance.

### 6. Built-in Cost Tracking

Every API call is recorded:

**Rationale**: Budget awareness is a first-class requirement, not an afterthought.

---

## Interfaces Between Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        PromptLab (Facade)                       │
│  - lab.run(template, **vars) → CompletionResult                 │
│  - lab.evaluate(template, test_suite) → EvalReport              │
│  - lab.ab_test(template_a, template_b, suite) → ABReport        │
│  - lab.registry → PromptRegistry                                │
│  - lab.cost → CostTracker                                       │
└────────┬──────────────┬───────────────┬────────────────┬────────┘
         │              │               │                │
    ┌────▼────┐   ┌─────▼─────┐  ┌──────▼──────┐  ┌─────▼─────┐
    │Templates│   │ Providers │  │ Evaluation  │  │ Tracking  │
    │Registry │   │           │  │ Metrics     │  │ Budget    │
    └─────────┘   └───────────┘  └─────────────┘  └───────────┘
```

The **PromptLab** class is the main facade — users interact primarily with it:

```python
class PromptLab:
    """Main entry point for the Prompt Engineering Laboratory."""
    
    def __init__(
        self,
        provider: LLMProvider,
        prompts_dir: Path = Path("prompts"),
        budget: float | None = None,
    ):
        self.provider = provider
        self.registry = PromptRegistry(storage_dir=prompts_dir)
        self.cost_tracker = CostTracker()
        self.budget = BudgetManager(budget) if budget else None
    
    async def run(
        self, 
        template: PromptTemplate | str,
        **variables
    ) -> CompletionResult:
        """Execute a prompt template."""
        ...
    
    async def evaluate(
        self,
        template: PromptTemplate | str,
        test_suite: TestSuite,
        metrics: list[BaseMetric] | None = None,
    ) -> EvaluationReport:
        """Evaluate a prompt against a test suite."""
        ...
    
    async def ab_test(
        self,
        template_a: PromptTemplate | str,
        template_b: PromptTemplate | str,
        test_suite: TestSuite,
    ) -> ABTestReport:
        """A/B test two prompt versions."""
        ...
```

---

## Dependency Management

### Required Dependencies

```toml
[project]
dependencies = [
    "openai>=1.30.0",        # OpenAI / Azure OpenAI client
    "pydantic>=2.0.0",       # Data models and validation
    "jinja2>=3.1.0",         # Template rendering
    "pyyaml>=6.0",           # YAML file loading
    "numpy>=1.24.0",         # Numerical operations (similarity, stats)
]

[project.optional-dependencies]
eval = [
    "scipy>=1.10.0",         # Statistical tests for A/B testing
]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "ruff>=0.4.0",
    "mypy>=1.0",
]
```

### Dependency Philosophy

- **Minimal core**: Only OpenAI, Pydantic, Jinja2, PyYAML, numpy
- **Optional extras**: scipy for statistics, ragas for advanced metrics
- **No LangChain**: Too heavy, we build our own lightweight abstractions
- **No framework lock-in**: Each component is independently usable

---

## Implementation Order

Based on dependencies between modules:

### Phase 1: Foundation (Days 1-2)
1. `core/` — Types, config, exceptions
2. `providers/` — LLMProvider base + OpenAI + Mock
3. `tracking/` — CostTracker, BudgetManager

### Phase 2: Templates (Days 3-4)
4. `templates/` — PromptTemplate, Jinja2 renderer, YAML loader
5. `registry/` — Version management, storage

### Phase 3: Evaluation (Days 5-7)
6. `metrics/` — BaseMetric + heuristic + LLM-as-judge + reference
7. `evaluation/` — TestCase, TestRunner, Reporter
8. `ab_testing/` — ABExperiment, runner, statistical analysis

### Phase 4: Library (Days 8-9)
9. `library/` — Pre-built prompts (summarization, extraction, classification, rewriting, CoT)
10. YAML prompt files with test cases

### Phase 5: Polish (Day 10)
11. Integration tests
12. Examples
13. README, API documentation
14. Package configuration (pyproject.toml)

---

## Quality Standards

| Standard | Approach |
|----------|----------|
| **Type safety** | Full type hints, mypy strict mode |
| **Testing** | pytest + pytest-asyncio, >80% coverage |
| **Linting** | Ruff for formatting and linting |
| **Documentation** | Docstrings on all public APIs, research docs |
| **Error handling** | Custom exceptions, never bare `except` |
| **Logging** | Structured logging (Python logging module) |
| **Async** | All I/O operations are async |

---

## Open Questions

- [ ] Should we support streaming responses in the evaluation pipeline?
- [ ] How to handle prompt templates that need multi-turn conversations?
- [ ] Should the registry support remote storage (S3, Azure Blob)?
- [ ] How to handle model-specific prompt optimizations (different prompts for different models)?
- [ ] Should we implement a simple CLI for running evaluations?

---

## Resources

- All previous research documents (01-07) feed into this architecture
- [Python Packaging Guide](https://packaging.python.org/) — For pyproject.toml setup
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/) — Async test patterns
- [Ruff](https://docs.astral.sh/ruff/) — Fast Python linting and formatting
