---
date: 2026-02-27
type: concept
topic: "Fine-Tuning Strategies"
project: "Phase 5 - Multi-Modal Enterprise Assistant"
status: complete
---

# Learning: Fine-Tuning Strategies

## In My Own Words

Fine-tuning is the process of **taking a pre-trained model and further training it on your own data** to specialize it for a specific task. Instead of training a model from scratch (which costs millions of dollars and requires massive datasets), you leverage the knowledge already learned by the model and nudge its weights slightly to perform better on your particular use case.

The key insight: fine-tuning is about **trade-offs**. You trade flexibility (a general model can do anything) for performance (a fine-tuned model excels at one thing). The decision of whether to fine-tune vs. prompt engineer depends on your specific constraints around cost, latency, data availability, and required quality.

## Why This Matters

Fine-tuning is a critical tool in the enterprise AI toolkit because:

- **Reduces prompt sizes**: A fine-tuned model doesn't need lengthy instructions — the behavior is "baked in"
- **Lowers per-request cost**: Shorter prompts = fewer input tokens = lower cost at scale
- **Improves consistency**: Fine-tuned models produce more uniform outputs for the same type of input
- **Enables specialization**: Match or exceed large model performance with smaller, cheaper models
- **Protects knowledge**: Domain knowledge is embedded in weights, not exposed in prompts

## Core Principles

### 1. What Fine-Tuning Actually Does

```
Pre-trained Model (general knowledge)
        │
        ▼
   [Training on your data: adjust weights]
        │
        ▼
Fine-tuned Model (general knowledge + your specialization)
```

Technically, fine-tuning adjusts the model's weight matrices to minimize loss on your training examples. The model retains its general capabilities while becoming significantly better at the patterns in your data.

**Azure OpenAI uses LoRA (Low-Rank Adaptation)** for fine-tuning, which is more efficient than full fine-tuning:

```
Original weights (frozen): W₀ ∈ ℝᵈˣᵈ
LoRA update:               ΔW = BA where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ (r << d)
Final weights:             W = W₀ + ΔW
```

Instead of updating all parameters, only two small matrices (A and B) are trained. This dramatically reduces:
- **Trainable parameters**: From billions to millions
- **Training time**: From days to hours
- **GPU memory**: Can fine-tune on consumer hardware
- **Storage**: Each fine-tuned model is a small delta, not a full model copy

### 2. The Fine-Tuning Decision Framework

**When to fine-tune vs. when to prompt engineer:**

```
START: Is my current prompt approach working?
  │
  ├── YES, but too expensive ──▶ Fine-tune smaller model to match large model behavior
  │
  ├── YES, but too slow ──▶ Fine-tune smaller model for lower latency
  │
  ├── NO, outputs inconsistent ──▶ Fine-tune with consistent examples
  │
  ├── NO, prompts too long ──▶ Fine-tune to internalize instructions
  │
  ├── NO, domain knowledge missing ──▶ Consider RAG first, then fine-tune if needed
  │
  └── NO, wrong style/format ──▶ Fine-tune with style examples
```

**Fine-tuning is the RIGHT choice when:**

| Scenario | Why Fine-Tuning Helps |
|----------|----------------------|
| Consistent output format needed | Model learns the format, no need to describe it |
| Many examples in every prompt | Move examples into training, shorten prompts |
| Specific tone/style required | Style becomes the model's default |
| High-volume, same-type queries | Pay training cost once, save per-request |
| Small model needs to match large model | Distillation via fine-tuning |
| Tool calling needs improvement | Fine-tune with tool usage examples |

**Fine-tuning is the WRONG choice when:**

| Scenario | Better Alternative |
|----------|-------------------|
| Need factual accuracy about your data | RAG (retrieval-augmented generation) |
| Knowledge changes frequently | RAG with updated index |
| Very few examples available (<50) | Few-shot prompting |
| Need to experiment rapidly | Prompt engineering |
| Multiple diverse tasks | Prompt engineering with task-specific prompts |
| Model already performs well | Don't fix what isn't broken |

### 3. Fine-Tuning Strategies

#### Full Fine-Tuning
Update **all** model parameters.

```
All weights: W₁, W₂, ..., Wₙ ──▶ [Training] ──▶ W₁', W₂', ..., Wₙ'
```

| Aspect | Details |
|--------|---------|
| **Parameters updated** | All (billions) |
| **Training data needed** | Large (10K+ examples) |
| **Training time** | Long (hours to days) |
| **GPU memory** | Very high (multiple A100s) |
| **Quality** | Highest potential |
| **Risk** | Catastrophic forgetting (model loses general knowledge) |
| **Availability** | Self-hosted models only |

#### LoRA (Low-Rank Adaptation)
Add small trainable matrices to **selected** layers, keep original frozen.

```
Frozen weights: W₀ (unchanged)
Trainable: A ∈ ℝʳˣᵈ, B ∈ ℝᵈˣʳ (r = rank, typically 4-64)
Output: W₀x + BAx
```

| Aspect | Details |
|--------|---------|
| **Parameters updated** | 0.1-1% of total |
| **Training data needed** | Moderate (hundreds to thousands) |
| **Training time** | Hours |
| **GPU memory** | Low (single GPU possible) |
| **Quality** | Comparable to full fine-tuning |
| **Risk** | Low risk of forgetting |
| **Availability** | Azure OpenAI default, HuggingFace PEFT |

**Key LoRA hyperparameters**:
- **r (rank)**: Controls the expressiveness of the update. Higher rank = more capacity but slower training. Typical: 8-64
- **lora_alpha**: Scaling factor. Effective learning rate scales as `lora_alpha/r`. Typical: 16-32
- **target_modules**: Which layers get LoRA adapters. Usually attention layers (q_proj, v_proj)
- **lora_dropout**: Dropout for regularization. Typical: 0.05-0.1

#### QLoRA (Quantized LoRA)
Quantize the base model to 4-bit, then apply LoRA on top.

```
Base model: W₀ ──▶ Quantize to 4-bit ──▶ W₀_q4 (frozen, 4x smaller)
LoRA adapters: A, B (trained in full precision)
```

| Aspect | Details |
|--------|---------|
| **Parameters updated** | Same as LoRA |
| **Memory reduction** | 4x less than LoRA |
| **Training data needed** | Same as LoRA |
| **Quality** | Slightly lower than LoRA |
| **Key benefit** | Fine-tune 65B models on a single 48GB GPU |
| **Availability** | HuggingFace (bitsandbytes + PEFT) |

### 4. Training Data Preparation

**Format** (Azure OpenAI fine-tuning uses JSONL with chat format):

```jsonl
{"messages": [{"role": "system", "content": "You are an invoice classifier."}, {"role": "user", "content": "Invoice #1234 from Acme Corp, $500, office supplies"}, {"role": "assistant", "content": "Category: Office Supplies\nVendor: Acme Corp\nAmount: $500.00"}]}
{"messages": [{"role": "system", "content": "You are an invoice classifier."}, {"role": "user", "content": "Invoice #5678 from CloudHost Inc, $2,400, cloud services"}, {"role": "assistant", "content": "Category: IT Services\nVendor: CloudHost Inc\nAmount: $2,400.00"}]}
```

**Data quality guidelines**:

| Guideline | Why |
|-----------|-----|
| **Minimum 50 examples** | 10 is the floor, but 50+ for any meaningful improvement |
| **Aim for 100-1000** | Sweet spot for most tasks |
| **High quality > quantity** | 100 perfect examples beat 10,000 noisy ones |
| **Diverse examples** | Cover edge cases, not just the happy path |
| **Consistent format** | System message should match what you'll use in production |
| **Use the same system message** | The system message used in training must be used in inference |
| **Include validation set** | 10-20% for evaluation (separate from training) |

### 5. Fine-Tuning Economics

The cost calculation for "Should I fine-tune?":

**Training cost** (one-time):
- Azure OpenAI: ~$0.008/1K training tokens (gpt-4o-mini), varies by model
- Training runs through your data multiple times (epochs)
- Example: 1,000 examples × 500 tokens/example × 3 epochs = 1.5M tokens ≈ $12

**Hosting cost** (ongoing):
- Fine-tuned models deployed as standard or global-standard: same per-token inference pricing
- Provisioned throughput: hourly hosting fee
- **Important**: Inactive deployments (no calls for 15 days) are auto-deleted

**Break-even analysis**:
```
Savings per request = (tokens_saved_per_request × cost_per_token)
Break-even requests = training_cost / savings_per_request

Example:
- Current prompt: 2000 tokens (with examples and instructions)
- Fine-tuned prompt: 200 tokens (just the input)
- Saved: 1800 tokens per request × $0.00015/token = $0.27 saved per request
- Training cost: $50
- Break-even: 50 / 0.27 ≈ 185 requests
```

## Types of Fine-Tuning (Azure OpenAI)

### Supervised Fine-Tuning (SFT)
The standard approach — provide input/output pairs, model learns to map inputs to outputs.

- **Best for**: Classification, extraction, formatting, style transfer
- **Data format**: Chat messages with system/user/assistant roles
- **Training method**: LoRA by default on Azure

### Direct Preference Optimization (DPO)
Provide preferred and non-preferred responses; model learns to favor preferred outputs.

- **Best for**: Alignment, improving response quality, reducing unwanted behaviors
- **Data format**: Same input, two responses (preferred + non-preferred)
- **Available for**: GPT-4o on Azure

### Reinforcement Fine-Tuning (RFT)
Use a grader/reward model to iteratively improve the model through feedback.

- **Best for**: Complex reasoning, problems with many valid solutions
- **Available for**: o4-mini (reasoning models) on Azure
- **Data format**: Problems + grading criteria

### Distillation
Use a large model's outputs to fine-tune a smaller model.

```
[GPT-4o] ──generates high-quality outputs──▶ Training Data ──▶ Fine-tune [GPT-4o-mini]
```

- **Best for**: Cost reduction at scale
- **Approach**: Collect production traffic from expensive model, use to train cheaper model
- **Typical result**: Small model achieves 80-95% of large model performance at 10-20% of cost

## Best Practices

- ✅ **Start with prompt engineering** — fine-tune only when prompting hits its limits
- ✅ **Collect production data first** — log real queries and ideal responses before training
- ✅ **Start with 50-100 high-quality examples** — add more only if needed
- ✅ **Use validation data** — set aside 10-20% for evaluation
- ✅ **Monitor training metrics** — watch for loss convergence and validation accuracy
- ✅ **Keep the same system message** — training and inference system messages must match
- ✅ **Compare fine-tuned vs. prompted** — document the performance difference to justify costs
- ✅ **Version your training data** — track which data produced which model
- ❌ **Don't fine-tune to add knowledge** — use RAG for factual knowledge, fine-tuning for behavior
- ❌ **Don't overtrain** — more epochs isn't always better; watch for diverging validation loss
- ❌ **Don't skip evaluation** — always measure before and after on a held-out test set
- ❌ **Don't forget about hosting costs** — fine-tuned models incur ongoing deployment costs

## Common Pitfalls

| Pitfall | Why It Happens | How to Avoid |
|---------|----------------|--------------|
| **Overfitting** | Too many epochs, too little data | Use validation set, early stopping, fewer epochs |
| **Catastrophic forgetting** | Too aggressive training | Use LoRA (default on Azure), lower learning rate |
| **Bad training data** | Inconsistent or incorrect examples | Curate data carefully, use multiple reviewers |
| **Wrong system message** | Different system message at inference | Keep identical system messages |
| **No baseline comparison** | Can't prove fine-tuning helped | Always evaluate prompted baseline first |
| **Training on edge cases only** | Missing normal cases | Include representative mix of common + edge cases |
| **Forgetting to validate** | Ship model without testing | Automated eval on held-out test set before deployment |

## Application to My Project

### How I'll Use This

Phase 5 requires a **fine-tuning experiment**. My plan:

1. **Task selection**: Fine-tune for document classification or entity extraction (a specific, measurable task)
2. **Data collection**: Create 100-500 examples of the target task
3. **Baseline**: Evaluate GPT-4o with prompt engineering on the same task
4. **Fine-tune**: Train GPT-4o-mini with the collected data
5. **Compare**: Fine-tuned GPT-4o-mini vs. prompted GPT-4o vs. prompted GPT-4o-mini
6. **Document**: Cost, latency, quality, and the overall economics

### Experiment Design

```
Task: Invoice line-item extraction

Baseline measurements:
├── GPT-4o + detailed prompt    → Accuracy: X%, Cost: $Y/request, Latency: Zms
├── GPT-4o-mini + detailed prompt → Accuracy: X%, Cost: $Y/request, Latency: Zms
└── GPT-4o-mini (fine-tuned)    → Accuracy: X%, Cost: $Y/request, Latency: Zms

Report:
├── Training cost
├── Performance comparison table
├── Break-even analysis
└── Recommendation
```

### Decisions to Make

- [ ] Which task to fine-tune for (classification? extraction? formatting?)
- [ ] How many training examples to create
- [ ] Which base model to fine-tune (GPT-4o-mini most likely for cost comparison)
- [ ] Evaluation metrics (accuracy, F1, exact match?)
- [ ] Whether to also attempt LoRA with HuggingFace for comparison

## Resources for Deeper Learning

- [Azure OpenAI Fine-Tuning Guide](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning) — Step-by-step Azure fine-tuning
- [Fine-Tuning Considerations (Azure)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/concepts/fine-tuning-considerations) — When and why to fine-tune
- [LoRA Paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) — Original LoRA research
- [QLoRA Paper (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) — Quantized LoRA for memory-efficient fine-tuning
- [HuggingFace PEFT Library](https://huggingface.co/docs/peft) — Python library for parameter-efficient fine-tuning

## Questions Remaining

- [ ] How does fine-tuned GPT-4o-mini compare to prompted GPT-4o on my specific task?
- [ ] What's the minimum dataset size for meaningful improvement?
- [ ] How often does a fine-tuned model need retraining with new data?
- [ ] Can I stack fine-tuning approaches (SFT → DPO)?
