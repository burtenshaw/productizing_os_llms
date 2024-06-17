# Week 3 of Productizing Open Source Large Language Models

## Objectives

1. Quantize an open-source LLM.
2. Evaluate the quantized model on a benchmark task.
3. Use LlamaCPP server for inference.
4. Utilize the quantized model in a downstream task via the Open Model API.
5. Employ the quantized model with Outlines.
6. Publish one of the models to Hugging Face.

## Background

### Quantization
Quantization involves converting a model from floating-point to fixed-point arithmetic. This reduces the model's memory footprint and computational complexity, making it more efficient for deployment on edge devices. [1]

### LlamaCPP
LlamaCPP is a C++ library for quantizing large language models. It provides tools for converting pre-trained models to fixed-point formats, optimizing them for deployment on resource-constrained devices. [2]

### OpenAI Model API
OpenAI's Python client allows interaction with open-source models served locally. The client provides a user-friendly interface for making requests to the model server and processing the responses. [3]

### Outlines
Outlines is a Python library for generating structured data from natural language prompts. It uses pre-trained language models to parse and extract information from text inputs, facilitating the creation of structured data from unstructured text. [4]

## Steps

### 1. Quantize an Open Source LLM

First, install LlamaCPP and its dependencies to perform quantization to the GGUF format.

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && git pull && make clean && LLAMA_CUBLAS=1 make
pip install -r llama.cpp/requirements.txt
```

Convert the model to the f16 format. We need to specify the input file, output type, and output file. FP16 is a common format to use for quantization because it provides a good balance between precision and performance. FP stands for floating-point, and 16 refers to the number of bits used to represent the floating-point number.

```bash
python llama.cpp/convert.py <input_file> --outtype f16 --outfile <output_file>.fp16.bin
```

Quantize the model to the desired format. Here, we use the Q4_K_M quantization method, which quantizes the model to 4 bits for weights and 8 bits for activations. The k and m parameters control the quantization range and granularity, respectively.

```bash
./llama.cpp/quantize <output_file>.fp16.bin {qtype} q4_k_m
```

Great! You have successfully quantized the model to the desired format.

### 2. Evaluate the Quantized Model on a Benchmark Task

Now that you have quantized the model, evaluate its performance on a benchmark task. You can use the LM Evaluation Harness to evaluate the model on a specific task. This is exactly the same as previous weeks, but with the quantized model. In real world scenarios, you would evaluate the model on a task relevant to your use case to evaluate whether quantization has affected the model's performance.


```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/pythia-160m,revision=step100000,dtype="float" \
    --tasks hellaswag \
    --device cuda:0 \
    --batch_size auto:4 --output hellaswag_test
```

### 3. Publish the Quantized Model to Hugging Face

Publishing your quantized model to Hugging Face allows you to share it with the community and make it accessible to others. To do this, you will need to create a new repository on Hugging Face, upload your quantized model and tokenizer, and create a model card with details about the quantization method and evaluation results.

First, load the quantized model and tokenizer using the `transformers` library:

```python
import transformers

# Load the quantized model
model = transformers.AutoModelForCausalLM.from_pretrained("path/to/quantized/model")
tokenizer = transformers.AutoTokenizer.from_pretrained("path/to/tokenizer")
```

Next, create a new repository on Hugging Face using the `huggingface_hub` library:

```python
import huggingface_hub

# Create a new repository on Hugging Face
repo_name = "my-quantized-model"
repo_description = "A quantized version of the original model."
repo_tags = ["quantization", "LLaMa", "efficient"]

huggingface_hub.create_repo(repo_name, repo_description, repo_tags)
```

Upload the quantized model and tokenizer to the repository:

```python
# Upload the model to the repository
huggingface_hub.upload_model(model, tokenizer, repo_name)
```
Create a model card with details about the quantization method and evaluation results:

```python
# Create a model card with details about the quantization method and evaluation results
model_card = """
# My Quantized Model

This is a quantized version of the original LLaMa model. It uses the GGUF quantization method and has been evaluated on the hellaswag benchmark task.

## Evaluation Results

| Metric | Value |
| --- | --- |
| Perplexity | 10.2 |
| F1 Score | 0.92 |

"""

```

Upload the model card to the repository:

```python
# Upload the model card to the repository
huggingface_hub.upload_model_card(model_card, repo_name)
```

### 4. Use the Quantized Model in a Downstream Task via the Open Model API
To utilize the quantized model in a downstream task, you will need to set up the LlamaCPP server for inference. This can be done using the following command:

python3 -m llama_cpp.server --model <model_path>
This command starts the LlamaCPP server, which allows you to interact with the quantized model using OpenAI's Python client. The --model flag specifies the path to the quantized model file.

Once the server is running, you can interact with the model using OpenAI's Python client. Here's an example of how to do this:

```bash
python3 -m llama_cpp.server --model <model_path>
```

In this example, we create an instance of the OpenAI client, specifying the base URL of the LlamaCPP server and an API key. We then define a prompt, which is a string that specifies the task we want the model to perform. In this case, the prompt is to create a client profile with the fields name, phone_number, and zip_code.

We then use the chat.completions.create method to send the prompt to the model and retrieve the response. The model parameter specifies the path to the quantized model file, and the messages parameter specifies the input prompt. The stream parameter is set to True to enable streaming responses, and the max_tokens parameter specifies the maximum number of tokens to generate.


```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tom",
)

prompt = "Create a client profile with the fields name, phone_number, and zip_code"

response = client.chat.completions.create(
    model="<model_path_to_gguf>",
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
    stream=True,
    max_tokens=1000,
)

```


### 5. Use the Quantized Model with Outlines

Outlines is a Python library that allows you to generate structured data from natural language prompts. To use the quantized model with Outlines, you will need to load the model using the LlamaCPP library and create a generator using the Outlines library.

**Load the Model using LlamaCPP**

To load the model using LlamaCPP, you can use the following code:

```python
from llama_cpp import Llama

llm = Llama("./phi-2.Q4_K_M.gguf")
```
**Create a Generator using Outlines**

To create a generator using Outlines, you can use the following code:

```python
from pydantic import BaseModel
from outlines import models, generate, types

locale = types.locale("us")

class Client(BaseModel):
    name: str
    phone_number: locale.PhoneNumber
    zip_code: locale.ZipCode

generator = generate.json(llm, Client)
```

**Generate Structured Data from Natural Language Prompts**

To generate structured data from natural language prompts, you can use the following code:

```python
result = generator(
    "Create a client profile with the fields name, phone_number, and zip_code"
)

print(result)
# Example output: name='Tommy' phone_number='129-896-5501' zip_code='50766'
```

This code generates structured data from the natural language prompt and prints the result.

---

## Evaluation Guidelines for Peer Reviewers

Your work will be reviewed based on the published model, evaluation results, and model presentation. Ensure the model is well-documented and easy to use, with clear and informative evaluation results.

### Basic Criteria

- Model is published to Hugging Face.
- Model is documented with a model card.
- Model card contains details about the quantization method and evaluation results.

### Advanced Criteria

- Model is evaluated on a benchmark task.
- Evaluation results relate to a use case.
- Evaluation results are visualized.
- Evaluation results are compared to a baseline.
- Evaluation results improve upon the baseline.

### Evaluation Tiers

- **Submitted:** Any of the basic criteria are met.
- **Basic:** All basic criteria are met.
- **Good:** All basic criteria and any advanced criteria are met.
- **Excellent:** All basic and advanced criteria are met.

---

## References

References:

[1] Gholami, A., et al. (2021). A survey of quantization methods for efficient neural network inference. Proceedings of the IEEE, 109(12), 2008-2037.
[2] Ggerganov/llama.cpp: C++ port of Facebook's LLaMa language model. GitHub. Retrieved from https://github.com/ggerganov/llama.cpp
[3] OpenAI API. OpenAI. Retrieved from https://openai.com/api/
[4] Outlines: Structured Data from Natural Language. GitHub. Retrieved from https://github.com/anthropic-research/outlines
* [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub/index)
* [Transformers documentation](https://huggingface.co/docs/transformers/index)
* [Hugging Face Model Card documentation](https://huggingface.co/docs/huggingface_hub/model_cards)
LlamaCPP documentation: https://github.com/ggerganov/llama.cpp/blob/main/docs/server.md
OpenAI Python client documentation: https://github.com/openai/openai-python/blob/main/README.md