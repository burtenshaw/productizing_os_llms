# Week 2 of Productizing Open 

## Objectives

1. Train an open source model on a dataset with ORPO
2. Evaluate the model on a test dataset
3. Publish the model to Hugging Face

## Background


## Instructions

### Train an open source model on a dataset with ORPO

1. Clone the Transformers Reinforcement Learning repository:

```bash
git clone https://github.com/huggingface/trl.git
```

2. Install the requirements:

```bash
cd trl
pip install -r requirements.txt
```

3. Train an open source model on a dataset with ORPO. For example, to train GPT-2 on the aligned ORPO dataset:

```bash
python examples/scripts/orpo.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-aligned-orpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns
```

### Evaluate the model on a test dataset

1. Evaluate the model on a test dataset. For example, to evaluate the GPT-2 model trained on the aligned ORPO dataset:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install accelerate

```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=${MODEL_ID},dtype=auto,trust_remote_code=$TRUST_REMOTE_CODE \
    --tasks eq_bench \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path ./evals/${benchmark}.json
```

### Publish the model to Hugging Face

1. Publish the model to Hugging Face. For example, to publish the GPT-2 model trained on the aligned ORPO dataset:

```bash
huggingface-cli login
```

2. Create a new repository and upload the model:

```bash
huggingface-cli repo create <repo_name>
```

3. Upload the model to the repository:

```bash
huggingface-cli repo upload --path <path_to_model> <repo_name>
```