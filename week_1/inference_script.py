from argparse import ArgumentParser

from distilabel.llms import InferenceEndpointsLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import (
    LoadDataFromDicts,
    TextGenerationToArgilla,
)
from distilabel.steps.tasks import TextGeneration

from datasets import load_dataset

parser = ArgumentParser()
parser.add_argument("--api_url", type=str, required=True)
parser.add_argument("--api_key", type=str, required=True)
parser.add_argument("--dataset_name", type=str, required=True)
parser.add_argument("--dataset_workspace", type=str, default="admin")
parser.add_argument("--num_generations", type=int, default=2)
parser.add_argument("--max_samples", type=int, default=100)

args = parser.parse_args()

dataset = load_dataset("DIBT/10k_prompts_ranked", split="train").filter(
    lambda r: r["avg_rating"] >= 4 and r["num_responses"] >= 2
)
dataset = dataset.to_list()

with Pipeline(
    name="prefs-with-llama-3",
    description="Pipeline for building preference datasets using Llama 3",
) as pipeline:
    load_dataset = LoadDataFromDicts(
        name="load_dataset",
        data=dataset[0:100],
        output_mappings={"prompt": "instruction"},
    )

    text_generation = TextGeneration(
        name="text_generation",
        llm=InferenceEndpointsLLM(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
        ),
    )

    to_argilla = TextGenerationToArgilla(
        dataset_name="my-dataset",
        dataset_workspace="admin",
    )

    load_dataset >> text_generation >> to_argilla


if __name__ == "__main__":
    distiset = pipeline.run(
        parameters={
            "load_dataset": {
                "repo_id": "distilabel-internal-testing/instruction-dataset-mini",
                "split": "test",
            },
            "text_generation": {
                "llm": {
                    "generation_kwargs": {
                        "max_new_tokens": 1024,
                        "temperature": 0.7,
                        "stop_sequences": ["<|eot_id|>", "<|end_of_text|>"],
                    }
                }
            },
            "to_argilla": {
                "api_url": args.api_url,
                "api_key": args.api_key,
                "dataset_name": args.dataset_name,
                "dataset_workspace": args.dataset_workspace,
            },
        }
    )
    distiset.push_to_hub("dvilasuero/distillama3-prompts10k")
