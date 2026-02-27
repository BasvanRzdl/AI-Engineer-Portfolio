---
date: 2026-02-27
type: technology
topic: "Fine-Tuning with Azure OpenAI & HuggingFace"
project: "Phase 5 - Multi-Modal Enterprise Assistant"
status: complete
---

# Technology Brief: Fine-Tuning with Azure OpenAI & HuggingFace

## Overview

This document covers the **practical tooling and workflows** for fine-tuning models, focusing on two approaches:

| Approach | Platform | Models | Cost | Control |
|----------|----------|--------|------|---------|
| **Azure OpenAI Fine-Tuning** | Managed cloud | GPT-4o, GPT-4o-mini | Per training token + hosting | Low (configuration only) |
| **HuggingFace + PEFT** | Self-managed | Open-source (LLaMA, Mistral, etc.) | Compute cost | Full (code-level) |

## Azure OpenAI Fine-Tuning

### Supported Models

| Model | Fine-Tuning Types | Training Data Format |
|-------|------------------|---------------------|
| **GPT-4o** (2024-08-06) | SFT, DPO | JSONL chat completions |
| **GPT-4o-mini** (2024-07-18) | SFT, DPO | JSONL chat completions |
| **GPT-4.1** | SFT | JSONL chat completions |
| **GPT-4.1-mini** | SFT | JSONL chat completions |
| **GPT-4.1-nano** | SFT | JSONL chat completions |
| **GPT-3.5-turbo** (0125) | SFT | JSONL chat completions |

### Fine-Tuning Types

| Type | Full Name | Purpose |
|------|-----------|---------|
| **SFT** | Supervised Fine-Tuning | Learn from example input/output pairs |
| **DPO** | Direct Preference Optimization | Learn from preferred vs rejected responses |
| **RFT** | Reinforcement Fine-Tuning | Learn with graders/reward functions |

### Training Data Format (SFT)

```jsonl
{"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "Patient presents with acute bronchitis"}, {"role": "assistant", "content": "ICD-10: J20.9 - Acute bronchitis, unspecified"}]}
{"messages": [{"role": "system", "content": "You are a medical coding assistant."}, {"role": "user", "content": "Diagnosis: Type 2 diabetes with neuropathy"}, {"role": "assistant", "content": "ICD-10: E11.40 - Type 2 diabetes mellitus with diabetic neuropathy, unspecified"}]}
```

### Training Data Format (DPO)

```jsonl
{"input": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain machine learning"}], "preferred_output": [{"role": "assistant", "content": "Machine learning is a branch of AI..."}], "non_preferred_output": [{"role": "assistant", "content": "ML is computers learning stuff..."}]}
```

### End-to-End Azure OpenAI Fine-Tuning Workflow

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://<resource>.openai.azure.com/",
    api_key="<key>",
    api_version="2024-10-21"
)

# Step 1: Upload training data
training_file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Step 2: (Optional) Upload validation data
validation_file = client.files.create(
    file=open("validation_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Step 3: Create fine-tuning job
job = client.fine_tuning.jobs.create(
    model="gpt-4o-mini-2024-07-18",  # base model
    training_file=training_file.id,
    validation_file=validation_file.id,
    hyperparameters={
        "n_epochs": 3,               # 1-10, default auto
        "batch_size": 4,             # 1-64, default auto
        "learning_rate_multiplier": 1.0  # 0.01-5.0, default auto
    },
    suffix="my-medical-coder"  # custom model name suffix
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

# Step 4: Monitor progress
import time

while True:
    job = client.fine_tuning.jobs.retrieve(job.id)
    print(f"Status: {job.status}")
    
    if job.status in ("succeeded", "failed", "cancelled"):
        break
    time.sleep(60)

# Step 5: Deploy the fine-tuned model
# Done via Azure Portal or Azure CLI:
# az cognitiveservices account deployment create \
#   --resource-group <rg> \
#   --name <resource> \
#   --deployment-name my-finetuned-model \
#   --model-name <fine-tuned-model-id> \
#   --model-format OpenAI \
#   --sku-name Standard \
#   --sku-capacity 1

# Step 6: Use the fine-tuned model
response = client.chat.completions.create(
    model="my-finetuned-model",  # deployment name
    messages=[
        {"role": "system", "content": "You are a medical coding assistant."},
        {"role": "user", "content": "Patient has acute myocardial infarction"}
    ]
)
```

### Analyzing Training Results

After training, a `results.csv` file is generated with metrics:

```
step,train_loss,train_mean_token_accuracy,valid_loss,valid_mean_token_accuracy
1,3.2155,0.0000,N/A,N/A
2,2.8901,0.1250,N/A,N/A
...
100,0.4523,0.8125,0.5012,0.7890
```

**What to look for**:
- **Training loss declining**: Good — model is learning
- **Validation loss increasing while training loss decreases**: Overfitting — reduce epochs
- **Both losses plateau**: Training is complete — more epochs won't help
- **Training accuracy**: Target depends on task; >80% is typically good

### Azure OpenAI Fine-Tuning Costs

| Model | Training Cost | Hosting Cost |
|-------|--------------|-------------|
| GPT-4o-mini | $0.30 / 1M tokens | Standard deployment rate |
| GPT-4o | $3.00 / 1M tokens | Standard deployment rate |
| GPT-3.5-turbo | $0.80 / 1M tokens | Standard deployment rate |

**Example**: Fine-tuning GPT-4o-mini on 1,000 examples (~500K tokens) × 3 epochs = ~1.5M training tokens = **~$0.45**

### Key Limitations

- Azure uses **LoRA internally** — you don't configure rank or target modules
- No control over LoRA hyperparameters (rank, alpha, target modules)
- Model must be deployed (costs hosting fees) after training
- Limited to chat completions format
- No vision fine-tuning yet (as of early 2026)

## HuggingFace PEFT (Parameter-Efficient Fine-Tuning)

### When to Use HuggingFace Instead

| Criterion | Azure OpenAI | HuggingFace PEFT |
|-----------|-------------|------------------|
| **Control** | Config only | Full code-level control |
| **Models** | GPT-4o, GPT-4o-mini, GPT-3.5 | Any open-source model |
| **LoRA config** | Hidden | Full control (rank, alpha, targets) |
| **Vision fine-tuning** | ❌ | ✅ (LLaVA, Idefics, etc.) |
| **Compute** | Managed | BYO GPU (Azure ML, local, etc.) |
| **Data privacy** | Data goes to Azure | Data stays on your compute |
| **Ease of use** | Very easy | Requires ML engineering |

### PEFT Library Overview

```python
from peft import LoraConfig, get_peft_model, TaskType

# Configure LoRA
lora_config = LoraConfig(
    r=16,                          # Rank (4, 8, 16, 32, 64)
    lora_alpha=32,                 # Scaling factor (typically 2×r)
    target_modules=[               # Which layers to adapt
        "q_proj", "v_proj",        # Attention queries and values
        "k_proj", "o_proj",        # Optionally keys and output
    ],
    lora_dropout=0.05,             # Dropout for regularization
    bias="none",                   # "none", "all", or "lora_only"
    task_type=TaskType.CAUSAL_LM   # Task type
)

# Apply LoRA to a base model
from transformers import AutoModelForCausalLM
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
peft_model = get_peft_model(base_model, lora_config)

# Check trainable parameters
peft_model.print_trainable_parameters()
# Output: trainable params: 3,407,872 || all params: 8,030,261,248 || trainable%: 0.0424
```

### LoRA Hyperparameter Guide

| Parameter | Description | Recommended Range | Effect |
|-----------|-------------|-------------------|--------|
| **r** (rank) | Rank of update matrices | 4-64 | Higher = more capacity, more params |
| **lora_alpha** | Scaling factor | r to 2×r | Higher = stronger LoRA influence |
| **target_modules** | Layers to adapt | q_proj, v_proj minimum | More modules = more expressiveness |
| **lora_dropout** | Regularization | 0.0-0.1 | Higher = less overfitting |

**Rules of thumb**:
- Start with `r=16, alpha=32`
- If underfitting: increase `r` or add more `target_modules`
- If overfitting: decrease `r`, increase `dropout`, or add more data
- For most tasks, `q_proj` + `v_proj` is sufficient

### QLoRA (Quantized LoRA)

QLoRA enables fine-tuning on consumer GPUs by quantizing the base model to 4-bit:

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat4
    bnb_4bit_use_double_quant=True,     # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8b",
    quantization_config=bnb_config,
    device_map="auto"
)

# Apply LoRA on top of quantized model
from peft import LoraConfig, get_peft_model
peft_model = get_peft_model(model, lora_config)

# Now fine-tune with SFTTrainer (see below)
```

**QLoRA memory requirements**:
| Model Size | Full FP16 | QLoRA (4-bit + LoRA) |
|-----------|-----------|---------------------|
| 7B | ~14 GB | ~6 GB |
| 13B | ~26 GB | ~10 GB |
| 70B | ~140 GB | ~40 GB |

### Training with SFTTrainer

```python
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="training_data.jsonl")

# Training config
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,  # or bf16=True for newer GPUs
    report_to="wandb",  # optional: tracking
    max_seq_length=2048
)

# Create trainer
trainer = SFTTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer
)

# Train
trainer.train()

# Save LoRA adapter
peft_model.save_pretrained("./my-lora-adapter")

# Later: Merge LoRA into base model for deployment
from peft import PeftModel, AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b")
merged = PeftModel.from_pretrained(base, "./my-lora-adapter")
merged = merged.merge_and_unload()  # Merge LoRA weights into base
merged.save_pretrained("./merged-model")
```

### Vision Model Fine-Tuning with PEFT

For the Phase 5 multi-modal project, you might fine-tune a vision-language model:

```python
# Example: Fine-tuning LLaVA with LoRA
from transformers import LlavaForConditionalGeneration
from peft import LoraConfig, get_peft_model

model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    load_in_4bit=True  # QLoRA
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj",  # Language model attention
    ],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)

peft_model = get_peft_model(model, lora_config)
# Train with vision-language dataset...
```

## Running Fine-Tuning on Azure ML

For HuggingFace fine-tuning at scale, use Azure ML compute:

```python
from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="<sub-id>",
    resource_group_name="<rg>",
    workspace_name="<workspace>"
)

# Create GPU compute cluster
gpu_compute = AmlCompute(
    name="gpu-cluster",
    type="amlcompute",
    size="Standard_NC24ads_A100_v4",  # A100 GPU
    min_instances=0,
    max_instances=2
)
ml_client.compute.begin_create_or_update(gpu_compute)

# Submit training job
training_job = command(
    code="./training_scripts",
    command="python train.py --model_name meta-llama/Llama-3-8b --epochs 3 --lora_r 16",
    environment="AzureML-pytorch-2.0-cuda11.8:latest",
    compute="gpu-cluster",
    inputs={
        "training_data": Input(type="uri_file", path="azureml://datastores/...")
    }
)

returned_job = ml_client.jobs.create_or_update(training_job)
```

## Decision Framework

```
Do I need fine-tuning?
│
├── Is prompt engineering sufficient? ──▶ YES ──▶ Don't fine-tune
│
├── NO ──▶ What model do I need?
│          │
│          ├── GPT-4o/4o-mini ──▶ Azure OpenAI Fine-Tuning
│          │   ├── Easy setup (API-based)
│          │   ├── SFT or DPO
│          │   └── No GPU management needed
│          │
│          ├── Open-source (LLaMA, Mistral) ──▶ HuggingFace PEFT
│          │   ├── Full control over LoRA config
│          │   ├── QLoRA for smaller GPUs
│          │   └── Can run on Azure ML or local
│          │
│          └── Vision-Language model ──▶ HuggingFace PEFT
│              ├── LLaVA, Idefics, etc.
│              └── Azure OpenAI doesn't support vision fine-tuning yet
```

## Best Practices

- ✅ **Start with prompt engineering** — fine-tune only when prompts aren't enough
- ✅ **Start with Azure OpenAI fine-tuning** for GPT models — it's the simplest path
- ✅ **Use HuggingFace PEFT** when you need control, open-source models, or vision fine-tuning
- ✅ **Always create a validation set** (10-20% of data) — monitor for overfitting
- ✅ **Start small** — 50-100 examples with 1-2 epochs, then scale up if results are promising
- ✅ **Track experiments** — use W&B, MLflow, or Azure ML experiment tracking
- ✅ **Keep base model frozen** — LoRA/PEFT adapters are cheap to store and swap
- ❌ **Don't fine-tune without a clear evaluation metric** — you need to measure improvement
- ❌ **Don't skip data quality** — 100 high-quality examples beats 10,000 noisy ones
- ❌ **Don't forget hosting costs** — a fine-tuned Azure OpenAI model needs a deployment (and that costs money)

## Resources

- [Azure OpenAI Fine-Tuning](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/fine-tuning) — Official guide
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft) — PEFT library docs
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Original QLoRA paper
- [TRL Library](https://huggingface.co/docs/trl) — Transformer Reinforcement Learning (SFTTrainer, DPO)
- [Azure ML Fine-Tuning](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-fine-tune-model-llama) — Fine-tuning on Azure ML
