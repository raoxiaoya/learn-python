部署Qwen2.5-7b大模型详解



本文参考教程：https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html



### 下载模型

https://modelscope.cn/organization/qwen

搜索 qwen2.5-7b

![image-20240930102342454](D:\dev\php\magook\trunk\server\md\img\image-20240930102342454.png)



可以看到它提供了六个模型，以满足不同的需求，从下载量来看，Qwen2.5-7B-Instruct 远超其他五个。



**1、预训练 (Pre-training) 和基模型 (Base models)：**是基础语言模型，不建议直接使用基础语言模型进行对话任务。您可以在此模型基础上进行后训练，例如SFT、RLHF、持续预训练等。基础模型不带`Instruct`字样，你可以对其进行情景学习，下游微调等。

**2、后训练 (Post-training) 和指令微调模型 (Instruction-tuned models)：**是专门设计用于理解并以对话风格执行特定指令的模型。这些模型经过微调，能准确地解释用户命令，并能以更高的准确性和一致性执行诸如摘要、翻译和问答等任务。与在大量文本语料库上训练的基础模型不同，指令调优模型会使用包含指令示例及其预期结果的数据集进行额外训练，通常涵盖多个回合。这种训练方式使它们非常适合需要特定功能的应用，同时保持生成流畅且连贯文本的能力。

`-Instruct`模型就是之前的`-Chat`模型。

**3、GGUF：**以GGUF格式保存的模型，用于 llama.cpp。

**4、AWQ量化，GPTQ量化：**属于经过量化算法优化后的版本，意在降低部署门槛，后面介绍。



### 显存要求

一般而言，模型加载所需显存可以按参数量乘二计算，例如，7B 模型需要 14GB 显存加载，其原因在于，对于大语言模型，计算所用数据类型为16位浮点数。当然，推理运行时还需要更多显存以记录激活状态。

对于 `transformers` ，推荐加载时使用 `torch_dtype="auto"` ，这样模型将以 `bfloat16` 数据类型加载。否则，默认会以 `float32` 数据类型加载，所需显存将翻倍。也可以显式传入 `torch.bfloat16` 或 `torch.float16` 作为 `torch_dtype` 。



### 关于多卡推理

`transformers` 依赖 `accelerate` 支持多卡推理，其实现为一种简单的模型并行策略：不同的卡计算模型的不同层，分配策略由 `device_map="auto"` 或自定义的 `device_map` 指定。

然而，这种实现方式并不高效，因为对于单一请求而言，同时只有单个 GPU 在进行计算而其他 GPU 则处于等待状态。为了充分利用所有的 GPU ，你需要像流水线一样安排多个处理序列，确保每个 GPU 都有一定的工作负载。但是，这将需要进行并发管理和负载均衡，这些超出了 `transformers` 库的范畴。即便实现了所有这些功能，整体吞吐量可以通过提高并发提高，但每个请求的延迟并不会很理想。

`对于多卡推理，建议使用专门的推理框架，如 vLLM 和 TGI，这些框架支持张量并行。`



### INFERENCE 推理

#### 1、Hugging Face Transformers

https://qwen.readthedocs.io/zh-cn/latest/inference/chat.html

transformers 支持手动和Pipeline两种作业方式。

支持`继续对话，流式输出`



#### 2、ModelScope

ModelScope的用法与Transformers基本一样，就是下载安装快一些，在使用时将包名由transformers改成modelscope即可。

https://github.com/modelscope/modelscope

https://www.modelscope.cn/docs



### RUN LOCALLY 运行模型

#### 1、llama.cpp

https://qwen.readthedocs.io/zh-cn/latest/run_locally/llama.cpp.html

https://github.com/ggerganov/llama.cpp

https://www.cnblogs.com/ghj1976/p/18063411/gguf-mo-xing

它是一个用来运行大模型的工具，其特点是以最小的代价来启动模型进行推理，它允许LLM在CPU上运行和推理。

The main goal of `llama.cpp` is to enable LLM inference with minimal setup and state-of-the-art performance on a wide variety of hardware - locally and in the cloud.

- 纯粹的C/C++实现，没有外部依赖
- 支持广泛的硬件：
  - x86_64 CPU的AVX、AVX2和AVX512支持
  - 通过Metal和Accelerate支持Apple Silicon（CPU和GPU）
  - NVIDIA GPU（通过CUDA）、AMD GPU（通过hipBLAS）、Intel GPU（通过SYCL）、昇腾NPU（通过CANN）和摩尔线程GPU（通过MUSA）
  - GPU的Vulkan后端
- 多种量化方案以加快推理速度并减少内存占用
- CPU+GPU混合推理，以加速超过总VRAM容量的模型



对于不是很精通c++的使用者，主要还是使用`llama-cli `工具



因此要求模型保存为GGUF格式，同时，它支持将一个大模型分割成多个文件保存。

![image-20240930110159197](D:\dev\php\magook\trunk\server\md\img\image-20240930110159197.png)



#### 2、Ollama

https://github.com/ollama/ollama

https://ollama.com/

Go语言开发的，类似于docker一样的工具，用来管理和运行LLM。它利用了 llama.cpp 提供的底层功能，它不仅支持gguf，还支持pt（PyTorch）和 safetensors（Tensorflow）。

它支持以 CLI 和 API 的方式运行LLM。

Ollama并不托管基模型。即便模型标签不带instruct后缀，实际也是instruct模型。

Ollama也可以运行已保存在本地的模型。



需要区分一下：`LLaMa` 是一个Meta公司开源的预训练大型语言模型，`llama.cpp`用于加载和运行GGUF语言模型。`Ollama`是大模型运行框架。



### WEB UI

[Text Generation Web UI](https://github.com/oobabooga/text-generation-webui)（简称TGW)

```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
```

你可以根据你的操作系统直接运行相应的脚本，例如在Linux系统上运行 `start_linux.sh` ，在Windows系统上运行 `start_windows.bat` ，在MacOS系统上运行 `start_macos.sh` ，或者在Windows子系统Linux（WSL）上运行 `start_wsl.bat` 。



访问：http://localhost:7860/?__theme=dark



### DEPLOYMENT 部署

#### 1、vLLM

vLLM是伯克利大学LMSYS组织开源的大语言模型高速推理框架，旨在极大地提升实时场景下的语言模型服务的吞吐与内存使用效率。vLLM是一个快速且易于使用的库，用于 LLM 推理和服务，可以和HuggingFace 无缝集成。vLLM利用了全新的注意力算法「PagedAttention」，有效地管理注意力键和值。吞吐量最多可以达到 huggingface 实现的24倍，文本生成推理（TGI）高出3.5倍，并且不需要对模型结构进行任何的改变。vLLM 库要比 HaggingFace Transformers库的推理速度高出一倍左右。

使用vLLM运行Qwen大模型：https://qwen.readthedocs.io/en/latest/deployment/vllm.html

vLLM 文档：https://docs.vllm.ai/en/stable/

`要部署 Qwen2.5 ，我们建议您使用 vLLM` 。 vLLM 是一个用于 LLM 推理和服务的快速且易于使用的框架。

```bash
# vLLM>=0.4.0

pip install vllm

# 构建一个与 OpenAI 兼容的 API 服务
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct
```

如 `vllm>=0.5.3` ，也可以如下启动：

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct
```

默认情况下，它将在 `http://localhost:8000` 启动服务器。您可以通过 `--host` 和 `--port` 参数来自定义地址。

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'
```

或者您可以按照下面所示的方式，使用 `openai` Python 包中的 Python 客户端：

```bash
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": "Tell me something about large language models."},
    ],
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    extra_body={
        "repetition_penalty": 1.05,
    },
)
print("Chat response:", chat_response)
```



也可以使用 vLLM 包来运行大模型。

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# Prepare your prompts
prompt = "Tell me something about large language models."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# generate outputs
outputs = llm.generate([text], sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```



工具（函数）使用

结构化/JSON输出

多卡分布式部署

上下文支持扩展：YaRN

部署量化模型：只需要指定参数 `quantization`的值



#### 2、TGI

https://qwen.readthedocs.io/en/latest/deployment/tgi.html

Hugging Face 的 Text Generation Inference (TGI) 是一个专为部署大规模语言模型 (Large Language Models, LLMs) 而设计的生产级框架。TGI提供了流畅的部署体验，并稳定支持如下特性：

- [推测解码 (Speculative Decoding)](https://qwen.readthedocs.io/zh-cn/latest/deployment/tgi.html#speculative-decoding) ：提升生成速度。
- 张量并行 ([Tensor Parallelism](https://huggingface.co/docs/text-generation-inference/conceptual/tensor_parallelism)) ：高效多卡部署。
- 流式生成 ([Token Streaming](https://huggingface.co/docs/text-generation-inference/conceptual/streaming)) ：支持持续性生成文本。
- 灵活的硬件支持：与 [AMD](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/deploy-your-model.html#serving-using-hugging-face-tgi) ， [Gaudi](https://github.com/huggingface/tgi-gaudi) 和 [AWS Inferentia](https://aws.amazon.com/blogs/machine-learning/announcing-the-launch-of-new-hugging-face-llm-inference-containers-on-amazon-sagemaker/#:~:text=Get started with TGI on SageMaker Hosting) 无缝衔接。



以docker的方式，通过TGI部署Qwen2.5

```bash
model=Qwen/Qwen2.5-7B-Instruct
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:2.0 --model-id $model
```

访问

```bash
curl http://localhost:8080/generate_stream -H 'Content-Type: application/json' \
        -d '{"inputs":"Tell me something about large language models.","parameters":{"max_new_tokens":512}}'
```



也可使用 OpenAI 风格的 API 访问TGI，注意，JSON 中的 model 字段不会被 TGI 识别，您可传入任意值。

```bash
curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "",
  "messages": [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me something about large language models."}
  ],
  "temperature": 0.7,
  "top_p": 0.8,
  "repetition_penalty": 1.05,
  "max_tokens": 512
}'
```

完整 API 文档，请查阅 [TGI Swagger UI](https://huggingface.github.io/text-generation-inference/#/Text Generation Inference/completions) 。



你也可以使用 Python 访问

```bash
from openai import OpenAI

# initialize the client but point it to TGI
client = OpenAI(
   base_url="http://localhost:8080/v1/",  # replace with your endpoint url
   api_key="",  # this field is not used when running locally
)
chat_completion = client.chat.completions.create(
   model="",  # it is not used by TGI, you can put anything
   messages=[
      {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
      {"role": "user", "content": "Tell me something about large language models."},
   ],
   stream=True,
   temperature=0.7,
   top_p=0.8,
   max_tokens=512,
)

# iterate and print stream
for message in chat_completion:
   print(message.choices[0].delta.content, end="")
```



多卡分布式部署

部署量化模型：只需要指定参数`--quantize gptq`



### QUANTIZATION

#### 0、量化

大语言模型（LLM）在自然语言处理（NLP）任务中取得了显著的进展。然而，LLM 通常具有非常大的模型大小和计算复杂度，这限制了它们在实际应用中的部署。

量化是将浮点数权重转换为低精度整数的过程，可以显著减少模型的大小和计算复杂度。近年来，LLM 量化的研究取得了很大进展，出现了许多新的量化方法。

GPTQ 和 AWQ 是目前最优的 LLM 量化方法之一。GPTQ 是 Google AI 提出的一种基于 group 量化和 OBQ 方法的量化方法。AWQ 是 Facebook AI 提出的一种基于 activation-aware 方法的量化方法。



#### 1、GPTQ

[GPTQ](https://arxiv.org/abs/2210.17323)是一种针对类GPT大型语言模型的量化方法，它基于近似二阶信息进行一次性权重量化。

GPTQ 的工作原理如下：

1. 首先，GPTQ 使用 group 量化将权重分组为多个子矩阵。
2. 然后，GPTQ 使用 OBQ 方法来量化每个子矩阵。
3. 最后，GPTQ 使用动态反量化来恢复权重的原始值。

GPTQ 的改进主要体现在以下几个方面：

- **分组量化**：GPTQ 使用分组量化来将权重分组为多个子矩阵，这可以降低量化精度损失。
- **OBQ 方法**：GPTQ 使用 OBQ 方法来量化权重，该方法可以实现高精度的量化。
- **动态反量化**：GPTQ 使用动态反量化来恢复权重的原始值，这可以提高量化的性能。

GPTQ 在各种 LLM 上进行了实验，结果表明，GPTQ 可以实现 3/4 位量化，在相同精度下，GPTQ 的模型大小比原始模型小 1/4。



vLLM 和 Hugging Face transformers 均已支持了 GPTQ 量化模型，用法和普通的模型是一样的。



使用[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)来对您自己的模型进行量化处理。



#### 2、AWQ

AWQ 的工作原理如下：

1. 首先，AWQ 使用 group 量化将权重分组为多个子矩阵。
2. 然后，AWQ 使用 activation-aware 的方法来量化每个子矩阵。
3. 最后，AWQ 使用无重新排序的在线反量化来提高量化性能。

AWQ 的 activation-aware 方法可以提高量化精度，这是因为激活值在量化后的影响可以通过量化系数进行补偿。具体来说，AWQ 首先计算每个子矩阵的激活分布，然后使用该分布来生成量化系数。

AWQ 的无重新排序的在线反量化可以提高量化性能，这是因为它不需要对权重进行重新排序，可以直接在量化后的权重上进行反量化。

AWQ 在各种 LLM 上进行了实验，结果表明，AWQ 可以实现 3/4 位量化，在相同精度下，AWQ 的模型大小比原始模型小 1/4，推理速度比 GPTQ 快 1.45 倍。



vLLM 和 Hugging Face transformers 均已支持了 AWQ 量化模型，用法和普通的模型是一样的。



如果您希望将自定义的模型量化为[AWQ](https://arxiv.org/abs/2306.00978)量化模型，我们建议您使用[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)。

```bash
git clone https://github.com/casper-hansen/AutoAWQ.git
cd AutoAWQ
pip install -e .
```

假设你已经基于 `Qwen2.5-7B` 模型进行了微调，并将其命名为 `Qwen2.5-7B-finetuned` ，且使用的是你自己的数据集，比如Alpaca。若要构建你自己的AWQ量化模型，你需要使用训练数据进行校准。以下，我们将为你提供一个简单的演示示例以便运行：

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Specify paths and hyperparameters for quantization
model_path = "your_model_path"
quant_path = "your_quantized_model_path"
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

# Load your tokenizer and model with AutoAWQ
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoAWQForCausalLM.from_pretrained(model_path, device_map="auto", safetensors=True)
```

接下来，您需要准备数据以进行校准。您需要做的就是将样本放入一个列表中，其中每个样本都是一段文本。由于我们直接使用微调数据来进行校准，所以我们首先使用ChatML模板对其进行格式化。例如：

```python
data = []
for msg in dataset:
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
    data.append(text.strip())
```

其中每个 `msg` 是一个典型的聊天消息，如下所示：

```bash
[
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": "Tell me who you are."},
    {"role": "assistant", "content": "I am a large language model named Qwen..."}
]
```

然后只需通过一行代码运行校准过程：

```python
model.quantize(tokenizer, quant_config=quant_config, calib_data=data)
```

最后，保存量化模型：

```python
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
tokenizer.save_pretrained(quant_path)
```

然后你就可以得到一个可以用于部署的AWQ量化模型。



#### 3、AWQ与GPTQ比较

| 特征     | AWQ  | GPTQ |
| -------- | ---- | ---- |
| 量化精度 | 优秀 | 良好 |
| 模型大小 | 最小 | 较小 |
| 计算速度 | 最快 | 较快 |
| 实现难度 | 较易 | 较难 |
| 量化成本 | 较高 | 较低 |



AWQ 在量化精度、模型大小和计算速度方面都优于 GPTQ，但在量化成本方面略高。



### 有监督微调：LLaMA-Factory

我们将介绍如何使用 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 微调 Qwen2.5 模型。本脚本包含如下特点：

- 支持单卡和多卡分布式训练
- 支持全参数微调、LoRA、Q-LoRA 和 DoRA 。

https://qwen.readthedocs.io/zh-cn/latest/training/SFT/llama_factory.html



### FRAMEWORK 框架

https://qwen.readthedocs.io/zh-cn/latest/framework/function_call.html

#### 函数调用

为了解决大语言模型的局限性，就需要模型在处理用户需求的过程中可以去调用指定的工具，这里具体就是函数，比如查天气，数学运算等等。



其实现流程是：

1. 定义函数，定义函数描述json schema。
2. 与LLM对话，将你的需求和上面的函数定义传给LLM，它会返回你该调用什么函数。
3. 你自己调用函数。
4. 将计算结果追加到message中再次丢给LLM，让其整合为最终的结果。



以 qwen-agent 框架为例。

1、定义函数，定义函数描述 TOOLS

```python
import json

def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }


def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]
MESSAGES = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow?"},
]
```



2、请求LLM

```python
functions = [tool["function"] for tool in TOOLS]

for responses in llm.chat(
    messages=messages,
    functions=functions,
    extra_generate_cfg=dict(parallel_function_calls=True),
):
    pass

messages.extend(responses)
```

返回信息如下

```python
[
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_temperature', 'arguments': '{"location": "San Francisco, CA, USA", "unit": "celsius"}'}},
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_temperature_date', 'arguments': '{"location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'}},
]
```

3、调用函数

```python
for message in responses:
    if fn_call := message.get("function_call", None):
        fn_name: str = fn_call['name']
        fn_args: dict = json.loads(fn_call["arguments"])

        fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

        messages.append({
            "role": "function",
            "name": fn_name,
            "content": fn_res,
        })
```

此时messages变成了

```python
[
    {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30'},
    {'role': 'user', 'content': "What's the temperature in San Francisco now? How about tomorrow?"},
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_current_temperature', 'arguments': '{"location": "San Francisco, CA, USA", "unit": "celsius"}'}},
    {'role': 'assistant', 'content': '', 'function_call': {'name': 'get_temperature_date', 'arguments': '{"location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'}},
    {'role': 'function', 'name': 'get_current_temperature', 'content': '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}'},
    {'role': 'function', 'name': 'get_temperature_date', 'content': '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}'},
]
```

4、让LLM整合结果

```python
for responses in llm.chat(messages=messages, functions=functions):
    pass
messages.extend(responses)
```

最终响应应如下所示

```python
{'role': 'assistant', 'content': 'Currently, the temperature in San Francisco is approximately 26.1°C. Tomorrow, on 2024-10-01, the temperature is forecasted to be around 25.9°C.'}
```



Hugging Face transformers 框架调用函数

```python
tools = TOOLS
messages = MESSAGES[:]

text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
....
```

剩下的过程与上面一样，具体参看上面的文档。



Ollama / vLLM 框架调用函数的过程与上面基本一致。



![image-20241017152235844](D:\dev\php\magook\trunk\server\md\img\image-20241017152235844.png)



#### Qwen-Agent



#### LlamaIndex

此工具旨在为了实现 Qwen2.5 与外部数据（例如文档、网页等）的连接，以快速部署检索增强生成（RAG）技术。

以下将演示如何从文档或网站构建索引。

```python
import torch
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set prompt template for generation (optional)
from llama_index.core import PromptTemplate

def completion_to_prompt(completion):
   return f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{completion}<|im_end|>\n<|im_start|>assistant\n"

def messages_to_prompt(messages):
    prompt = ""
    for message in messages:
        if message.role == "system":
            prompt += f"<|im_start|>system\n{message.content}<|im_end|>\n"
        elif message.role == "user":
            prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"
        elif message.role == "assistant":
            prompt += f"<|im_start|>assistant\n{message.content}<|im_end|>\n"

    if not prompt.startswith("<|im_start|>system"):
        prompt = "<|im_start|>system\n" + prompt

    prompt = prompt + "<|im_start|>assistant\n"

    return prompt

# Set Qwen2.5 as the language model and set generation config
Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
    context_window=30000,
    max_new_tokens=2000,
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)

# Set embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "BAAI/bge-base-en-v1.5"
)

# Set the size of the text chunk for retrieval
Settings.transformations = [SentenceSplitter(chunk_size=1024)]
```

1、检索文件内容

```python

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# document 为文件夹，里面支持PDF和TXT格式的文件。
documents = SimpleDirectoryReader("./document").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=Settings.transformations
)
```

embed_model：由来向量化数据的模型，您可以使用 `bge-base-en-v1.5` 模型来检索英文文档，下载 `bge-base-zh-v1.5` 模型以检索中文文档。根据您的计算资源，您还可以选择 `bge-large` 或 `bge-small` 作为向量模型，或调整上下文窗口大小或文本块大小。Qwen2.5模型系列支持最大32K上下文窗口大小（7B 、14B 、32B 及 72B可扩展支持 128K 上下文，但需要额外配置）。比如`BAAI/bge-base-en-v1.5`



2、检索网站内容

```python
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["web_address_1","web_address_2",...]
)
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=Settings.embed_model,
    transformations=Settings.transformations
)
```

要保存和加载已构建的索引，您可以使用以下代码示例。

```python
from llama_index.core import StorageContext, load_index_from_storage

# save index
storage_context = StorageContext.from_defaults(persist_dir="save")

# load index
index = load_index_from_storage(storage_context)
```

检索增强RAG的实现

现在您可以输入查询，Qwen2.5 将基于索引文档的内容提供答案。

```python
query_engine = index.as_query_engine()
your_query = "<your query here>"
print(query_engine.query(your_query).response)
```



#### Langchain

它是一个大模型应用框架，对很多应用场景做了封装，下面将使用Langchain与qwen2.5构建一个本地知识库问答系统。其原理为：将文件（PDF或TXT）内容分段并向量化，将其放在prompt最前面，然后连带问题一起发送给LLM，由LLM来匹配答案。

```python
PROMPT_TEMPLATE = """Known information:
    {context_str}
    Based on the above known information, respond to the user's question concisely and professionally. If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,' and do not include fabricated details in the answer. Please respond in English. The question is {question}"""
```

具体代码如下：

```python
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
import numpy as np
from typing import List, Tuple
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import argparse
import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

'''
您可以仅使用您的文档配合langchain来构建一个问答应用。该项目的实现流程包括：
加载文件 -> 阅读文本 -> 文本分段 -> 文本向量化 -> 问题向量化 -> 将最相似的前k个文本向量与问题向量匹配 -> 将匹配的文本作为上下文连同问题一起纳入提示 -> 提交给Qwen2.5-7B-Instruct生成答案。
'''

'''
pip install langchain==0.0.174
pip install faiss-gpu
'''

model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


class Qwen(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history_len: int = 3

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _history_len(self) -> int:
        return self.history_len

    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_token": self.max_token,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "history_len": self.history_len}


class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            text = re.sub(r"\n{3,}", "\n", text)
            text = re.sub('\s', ' ', text)
            text = text.replace("\n\n", "")
        sent_sep_pattern = re.compile(
            '([﹒﹔﹖﹗．。！？]["’”」』]{0,2}|(?=["‘“「『]{1,2}|$))')
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list


def load_file(filepath):
    loader = TextLoader(filepath, autodetect_encoding=True)
    textsplitter = ChineseTextSplitter(pdf=False)
    docs = loader.load_and_split(textsplitter)
    write_check_file(filepath, docs)
    return docs


def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s,len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


def separate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists


class FAISSWrapper(FAISS):
    chunk_size = 250
    chunk_conent = True
    score_threshold = 0

    def similarity_search_with_score_by_vector(
            self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        scores, indices = self.index.search(
            np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                # This happens when not enough docs are returned.
                continue
            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            if not self.chunk_conent:
                if not isinstance(doc, Document):
                    raise ValueError(
                        f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                for l in [i + k, i - k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break_flag = True
                            break
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break
        if not self.chunk_conent:
            return docs
        if len(id_set) == 0 and self.score_threshold > 0:
            return []
        id_list = sorted(list(id_set))
        id_lists = separate_list(id_list)
        for id_seq in id_lists:
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content
            if not isinstance(doc, Document):
                raise ValueError(
                    f"Could not find document for id {_id}, got {doc}")
            doc_score = min([scores[0][id] for id in [
                            indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs.append((doc, doc_score))
        return docs


if __name__ == '__main__':
    # load docs (pdf file or txt file)
    filepath = 'your file path'
    # Embedding model name
    EMBEDDING_MODEL = 'text2vec'
    PROMPT_TEMPLATE = """Known information:
    {context_str}
    Based on the above known information, respond to the user's question concisely and professionally. If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,' and do not include fabricated details in the answer. Please respond in English. The question is {question}"""
    # Embedding running device
    EMBEDDING_DEVICE = "cuda"
    # return top-k text chunk from vector store
    VECTOR_SEARCH_TOP_K = 3
    CHAIN_TYPE = 'stuff'
    embedding_model_dict = {
        "text2vec": "your text2vec model path",
    }
    llm = Qwen()
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_dict[EMBEDDING_MODEL], model_kwargs={'device': EMBEDDING_DEVICE})

    docs = load_file(filepath)

    docsearch = FAISSWrapper.from_documents(docs, embeddings)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context_str", "question"]
    )

    chain_type_kwargs = {"prompt": prompt,
                         "document_variable_name": "context_str"}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=docsearch.as_retriever(
            search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        chain_type_kwargs=chain_type_kwargs)

    query = "Give me a short introduction to large language model."
    print(qa.run(query))

```





### BENCHMARK 基准测试

1、关于GPU内存需求及相应吞吐量的结果

https://qwen.readthedocs.io/en/latest/benchmark/speed_benchmark.html

对于vLLM，由于GPU显存预分配，实际显存使用难以评估。



![image-20240930171437622](D:\dev\php\magook\trunk\server\md\img\image-20240930171437622.png)

![image-20240930171458178](D:\dev\php\magook\trunk\server\md\img\image-20240930171458178.png)





2、对于量化的模型，与原始 bfloat16 模型的基准测试结果

https://qwen.readthedocs.io/en/latest/benchmark/quantization_benchmark.html

- MMLU （准确率）
- C-Eval （准确率）
- IFEval （提示词级的严格准确率，Strict Prompt-Level Accuracy）

![image-20241008171522556](D:\dev\php\magook\trunk\server\md\img\image-20241008171522556.png)



### 其他阅读

https://mp.weixin.qq.com/s/Gn8kV04e_Y_BmXdxSMBiSQ

https://mp.weixin.qq.com/s/Fy-dNFOf-KhAMTJnBwjICQ