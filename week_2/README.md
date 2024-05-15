# Week 2 of Productizing Open Source Large Language Models

In this week, you will train an open source model on a dataset with ORPO, evaluate the model on a test dataset, and publish the model to Hugging Face. This project is a real world example of how to productize open source large language models. The aim is to share a model on the hugging face model hub that others can use for their own projects.

> You are supplied with functional code snippets to help you get started. You are encouraged to modify and expand upon these code snippets to complete the project. You could adapt the project by using evaluation harnesses, adding additional evaluation metrics, or using different model scales.

## Objectives

1. Train an open source model on a dataset with ORPO
2. Evaluate the model on a test dataset
3. Publish the model to Hugging Face

The outcome of the week is to train an open source model on a dataset with ORPO, evaluate the model on a test dataset, and publish the model to Hugging Face. You will be assessed on the presentation of the model, the evaluation results, and the model publication.

## Background

### Odds Ratio Preference Optimization (ORPO)

Odds Ratio Preference Optimization (ORPO) by Jiwoo Hong, Noah Lee, and James Thorne studies the crucial role of SFT within the context of preference alignment. Using preference data the method posits that a minor penalty for the disfavored generation together with a strong adaption signal to the chosen response via a simple log odds ratio term appended to the NLL loss is sufficient for preference-aligned SFT.

Thus ORPO is a reference model-free preference optimization algorithm eliminating the necessity for an additional preference alignment phase thus saving compute and memory.

The official code can be found [xfactlab/orpo](https://github.com/xfactlab/orpo).

### Transformers Reinforcement Learning

Transformers Reinforcement Learning (TRL) is a library for training and fine-tuning large language models with reinforcement learning. The library is built on top of the Hugging Face Transformers library and PyTorch Lightning. It provides a simple and flexible API for training and fine-tuning large language models with reinforcement learning.

The official code can be found [huggingface/trl](https://github.com/huggingface/trl).

## Instructions

These instructions will guide you through the process of training an open source model on a dataset with ORPO, evaluating the model on a test dataset, and publishing the model to Hugging Face. They supply the basic steps to complete the project. You are encouraged to modify and expand upon these steps to create a unique project.

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

Next, you will evaluate the model on a test dataset. You can use the LM Evaluation Harness to evaluate the model on a benchmark task. You should refer to your work from week 1 to advance this step.

Evaluate the model on a test dataset. For example, to evaluate the GPT-2 model trained on the aligned ORPO dataset:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install accelerate
```

Now you can evaluate the model on a benchmark task. For example, to evaluate the GPT-2 model trained on the aligned ORPO dataset:

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

Finally, you should publish the model to Hugging Face. You should update the model card with the model details and evaluation results. Review the [Hugging Face documentation]() for more information on how to publish a model.

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