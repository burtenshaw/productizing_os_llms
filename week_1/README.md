# Week 2 of Productizing Open 

## Objectives

1. Run an inference script using an open source model from Hugging Face
2. Evaluate the model on a benchmark task using the LM Evaluation Harness
3. Push the inference results to Argilla

## Background

## Instructions

### Run an inference script using an open source model from Hugging Face

### Evaluate the model on a benchmark task using the LM Evaluation Harness

```bash
lm_eval \
    --model hf \
    --model_args pretrained=microsoft/phi-2,trust_remote_code=True \
    --tasks hellaswag,mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/phi-2 \
    --limit 10 \
    --wandb_args project=lm-eval-harness-integration \
    --log_samples
```

### Push the inference results to Argilla


