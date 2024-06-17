### Week 1 of Productizing Open Source LLMs

Welcome to Week 1 project of Productizing Open Source LLMs. This a tutorial and project on using large language models (LLMs) to generate and review dataset responses. This tutorial will guide you through creating high-quality datasets with LLMs, evaluating their performance, and publishing the results to the Hugging Face Hub.

In this tutorial, you'll learn to use the LM Evaluation Harness, Argilla, and Distilabel to generate and review dataset responses, evaluate LLM performance, and publish models on the Hugging Face Hub. By the end of this week, you'll understand how to productize open-source language models and gain hands-on experience with the essential tools and frameworks.

## Objectives
- Generate and review responses to a dataset using a Large Language Model (LLM).
- Evaluate LLMs with the LM Evaluation Harness.
- Publish the model and results to the Hugging Face Hub.

## Background

For detailed background on the tools and frameworks used in this project, refer to the written material, lectures, and references provided. Here is a brief reminder of the key concepts.

### Overview of the LM Evaluation Harness

The LM Evaluation Harness is a unified framework for testing generative language models on a large number of different evaluation tasks. It supports over 60 standard academic benchmarks for LLMs, with hundreds of subtasks and variants implemented. The framework is designed to ensure reproducibility and comparability between papers by using publicly available prompts[1].

### Overview of Argilla

Argilla is a tool for creating and managing datasets. It allows users to create datasets by generating instructions based on human-made input. These instructions can then be reviewed using Argilla to keep the best ones. Argilla is used in conjunction with distilabel to generate and label datasets using LLMs[2].

### Overview of Distilabel

Distilabel is an AI Feedback (AIF) framework that can generate and label datasets using LLMs. It is implemented with robustness, efficiency, and scalability in mind, allowing anyone to build synthetic datasets that can be used in many different scenarios. Distilabel is used to create legal preference datasets based on RAG instructions from the European AI Act[3].

### Overview of Hugging Face Inference Endpoints

Hugging Face Inference Endpoints offers a secure production solution to easily deploy any Transformers, Sentence-Transformers, and Diffusion models from the Hub on dedicated and autoscaling infrastructure managed by Hugging Face. It allows users to deploy models directly from the Hugging Face Hub to managed infrastructure on their favorite cloud in just a few clicks. This service simplifies and accelerates HIPAA-compliant Transformer deployments[4].

## Instructions

Let's get started with the project. Follow the steps below to complete the tasks for this week. There are three main parts to this project with subsections for each part numerated accordingly.

### 1. Generate and Review Responses to a Dataset using a Large Language Model (LLM)

In this task we are going to take a dataset from the Hub and generate responses to it using a Large Language Model (LLM). Then we will review the responses using Argilla. The dataset we will use is [DIBT/10k_prompts_ranked]("https://huggingface.co/datasets/DIBT/10k_prompts_ranked"), which contains 10,000 prompts manually ranked by difficulty. This step will use distilabel to generate the responses.

#### 1.1 Install and Setup Dependencies

First, you need to install the necessary dependencies to run the inference script. You can install the required packages using the following commands:

```bash
pip install -qqq huggingface_hub argilla
pip install -qqq --upgrade "distilabel[huggingface]"
```

Before running the inference script, you need to set up Argilla. Follow the instructions provided in the [Argilla documentation](https://docs.argilla.io/en/latest/getting_started/installation/deployments/huggingface-spaces.html) to set up the server.

#### 1.2 Run the Inference Script

To generate responses to a dataset using a Large Language Model (LLM), you will need to run an inference script. This script should include the following parameters:

```bash
python inference_script.py \
    --api_url <API_URL> \
    --api_key <API_KEY> \
    --dataset_name <YOUR_DATASET_NAME> 
```
The script will generate responses to the dataset using the LLM and store the results in Argilla.

#### 1.3 Review the Results in Argilla and on the Hugging Face Hub

After running the inference script, you should review the results in Argilla and on the Hugging Face Hub. This will help you ensure that the model is generating high-quality responses and that the results are accurate and informative.

![Argilla](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/spaces-argilla-embed-space.png)

### Evaluating Large Language Models with the LM Evaluation Harness

Now that you have generated responses to a dataset using a Large Language Model (LLM), it's time to evaluate the model's performance. In this task, you will use the LM Evaluation Harness to evaluate the LLM on a benchmark task. The task we will use is the Hellaswag task, which requires the model to predict the most likely continuation of a sentence based on common sense knowledge.

#### 1. Install the LM Evaluation Harness

To evaluate a Large Language Model (LLM) using the LM Evaluation Harness, you will need to install the framework. This can be done by cloning the repository and installing the necessary dependencies:

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

#### 2. Run the Evaluation Script

To run the evaluation script, you will need to specify the model, tasks, and device. For example:

```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size auto:4 --output hellaswag_test
```

Experiment with different tasks to evaluate the performance of the LLM. You can find an overview of the available tasks in the [LM Evaluation Harness documentation](https://github.com/EleutherAI/lm-evaluation-harness).

### Publishing the Model to the Hugging Face Hub with Results

After evaluating the Large Language Model (LLM) using the LM Evaluation Harness, you can publish the model to the Hugging Face Hub with the evaluation results. This will make the model accessible to others in the machine learning community and allow them to use and evaluate it. 

    This is a learning exercise for the first week. In reality, it wouldn't make sense to publish a copy of a model that is already available on the Hugging Face Hub. It would make more sense to add your evaluation results to the existing model card through a pull request. In the coming weeks, you train your own models and publish them to the Hugging Face Hub.


**Step 1: Create a Repository**

To publish the model to the Hugging Face Hub, you will need to create a repository. This can be done using the `huggingface_hub` library, which provides a Python interface to the Hugging Face Hub. Specifically, you can use the `create_repo` function to create a new repository.

```
python
from huggingface_hub import create_repo

create_repo(repo_id="my_model", repo_type="model")
```

According to the Hugging Face Hub documentation, the `create_repo` function takes two required arguments: `repo_id` and `repo_type`. The `repo_id` is a unique identifier for your repository, and the `repo_type` specifies the type of repository you want to create (in this case, a model repository).

**Step 2: Upload the Model**

Once the repository is created, you can upload the model to the Hub using the `upload_file` function from the `huggingface_hub` library.

```
python
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="/path/to/model/checkpoint",
    path_in_repo="model.ckpt",
    repo_id="my_model"
)
```

As described in the Hugging Face Hub documentation, the `upload_file` function takes three required arguments: `path_or_fileobj`, `path_in_repo`, and `repo_id`. The `path_or_fileobj` argument specifies the local path to the model checkpoint file, the `path_in_repo` argument specifies the path to the file in the repository, and the `repo_id` argument specifies the ID of the repository to upload to.

**Step 3: Add Evaluation Results to the Model Card**

To add evaluation results to the model card, you can include them in the README file of the repository. The model card is a Markdown file that provides a summary of the model, including its description, evaluation results, and usage instructions.

Here is an example of how you can add evaluation results to the model card:
```
markdown
# Model Card

## Model Description

This is a Large Language Model (LLM) trained on a dataset of [dataset description].

## Evaluation Results

### Hellaswag

| Metric | Value |
| --- | --- |
| Accuracy | 0.85 |
| F1 Score | 0.92 |

### Other Evaluation Metrics

| Metric | Value |
| --- | --- |
| Perplexity | 10.2 |
| BLEU Score | 0.78 |

## How to Use

To use this model, simply download the checkpoint and load it into your preferred deep learning framework.
```

As described in the Hugging Face Hub documentation, the model card should include the following sections:

* **Model Description**: a brief description of the model, including its architecture and training dataset.
* **Evaluation Results**: a table or list of evaluation metrics, including their values.
* **How to Use**: instructions on how to use the model, including how to download and load the checkpoint.

By following these steps, you can publish your model to the Hugging Face Hub with evaluation results, making it accessible to others in the machine learning community.

---

## Evaluation Guidelines for Peer Reviewers

Your work will be peer-reviewed based on the published dataset and its presentation. You should ensure that the dataset is well-documented and easy to understand. You should also ensure that the evaluation results are clear and informative.

### Basic Criteria

- The dataset is published to Hugging Face
- The dataset is documented with a README
- The dataset card contains the model details and evaluation results

### Advanced Criteria

- The dataset contains evaluation results from a Judge model
- The dataset contains human feedback from Argilla
- Evaluation results are visualized
- Evaluation results are compared or explained

### Evaluation Tiers

- Submitted: Any of the basic criteria are met
- Basic: All of the basic criteria are met
- Good: All of the basic criteria are met and any of the advanced criteria are met
- Excellent: All of the basic criteria are met and all of the advanced criteria are met

# References

[1] https://distilabel.argilla.io/0.6.0/tutorials/pipeline-notus-instructions-preferences-legal/
[2] https://huggingface.co/docs/inference-endpoints/en/index
[3] https://huggingface.co/blog/inference-endpoints
[4] https://docs.pinecone.io/integrations/hugging-face-inference-endpoints
[5] https://huggingface.co/docs/huggingface_hub/en/guides/inference_endpoints
[6] https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_endpoints
[7] https://huggingface.co/docs/inference-endpoints/en/guides/access
[8] https://huggingface.co/docs/inference-endpoints/en/guides/test_endpoint
[9] https://python.langchain.com/v0.2/docs/integrations/callbacks/argilla/
[10] https://docs.zenml.io/v/docs/stacks-and-components/component-guide/annotators/argilla
[11] https://github.com/argilla-io/argilla/blob/develop/README.md
[12] https://pypi.org/project/distilabel/0.2.1/
[13] https://www.youtube.com/watch?v=0gvpsMfNzVc
[14] https://docs.argilla.io/en/latest/
[15] https://www.reddit.com/r/LocalLLaMA/comments/1c6gzta/distilabel_100_released_a_framework_for_building/
[16] https://www.linkedin.com/posts/argilla-io_dont-miss-this-tutorial-full-of-insights-activity-7074004713402187776-3aGj
[17] https://github.com/argilla-io/distilabel/discussions
[18] https://docs.argilla.io/en/latest/tutorials_and_integrations/tutorials/tutorials.html
[19] https://www.youtube.com/watch?v=8E01Xvc2ybk
[20] https://huggingface.co/docs/inference-endpoints/en/guides/create_endpoint