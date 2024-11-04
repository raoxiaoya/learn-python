Huggingface transformers 和 modelscope 框架运行原理



以Huggingface上的 `google-bert/bert-base-chinese`模型为例。我们使用Huggingface提供的transformers框架来启动模型

```python
#加载模型
from transformers import AutoModelForMaskedLM
#加载分词器
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
```

虽然填入的模型只有 bert-base-chinese 这一截，实际上完整的是 google-bert/bert-base-chinese，请求达到 huggingface.co 之后，它会搜索 bert-base-chinese，然后取第一个，于是就重定向到了 google-bert/bert-base-chinese

相关的代码如下：`D:\ProgramData\Anaconda3\Lib\site-packages\transformers\utils\hub.py:376`行。

函数是 cached_file()，其中 revision 默认值是 main 分支。
完整路径：https://huggingface.co/google-bert/bert-base-chinese/tree/main

下载的文件会保存到 cache_dir 目录下，默认为 C:\Users\Administrator.DESKTOP-TPJL4TC\.cache\huggingface\hub，即 HOME 目录下。

现在无论是Huggingface还是modelscope平台，大部分模型都不会上传代码，也就是只有几个配置文件和模型权重参数文件，但是，在运行模型的时候是需要构建分词器和构建模型，这就需要根据不同的模型架构使用python代码来实现，在Huggingface和modelscope框架中都内置了很多模型的构建代码，它根据内置的mapping隐射信息找到具体的类，如果本地没有这个代码，它还可以自动从网上下载（这需要在config中配置）

下面是 AutoTokenizer 的过程：
tokenizer 的 from_pretrained 是构建分词器，因此它只会下载三个文件（tokenizer_config.json，vocab.txt，tokenizer.json），不会下载巨大的模型文件。

如果你使用的是 AutoTokenizer，那它是如何寻找到要使用哪个类呢，比如这里应该是 BertTokenizer？
它判断的根据依次是：你传入的参数，模型配置文件（tokenizer_config.json, tokenizer.json, config.json）。
相关字段 tokenizer_class, model_type，此例中根据 model_type=bert，在 CONFIG_MAPPING_NAMES 中找到 bert --> BertConfig，最终会返回 BertTokenizer

但是在构建model的时候，却不能简单的使用AutoModel，因为此处根据不同的模型类别又细分了多个mapping，即同样都是bert，在不同的类别中对应了不同的class，所以此处使用的是AutoModelForMaskedLM，这个在模型的说明里会有说明。

```python
class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING
class AutoModelForKeypointDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_KEYPOINT_DETECTION_MAPPING
class AutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING
class AutoModelForImageToImage(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING
class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING
class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING
class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING
....
```

但是，从Huggingface下载模型是个艰难的事情，所以，建议换到国内的modelscope框架，它是在transformers框架上包装而来，所以用法几乎一样。

在modelscope框架中则需要将模型名称补全，否则会报错，它的cached_dir默认为 C:\Users\Administrator.DESKTOP-TPJL4TC\.cache\modelscope\hub

不同的是 modelscope 只有5个AutoModel

```python
AutoModel = get_wrapped_class(AutoModelHF)
AutoModelForCausalLM = get_wrapped_class(AutoModelForCausalLMHF)
AutoModelForSeq2SeqLM = get_wrapped_class(AutoModelForSeq2SeqLMHF)
AutoModelForSequenceClassification = get_wrapped_class(AutoModelForSequenceClassificationHF)
AutoModelForTokenClassification = get_wrapped_class(
AutoModelForTokenClassificationHF)
```

示例

```python
from modelscope import AutoModelForCausalLM
from modelscope import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiansz/bert-base-chinese")
model = AutoModelForCausalLM.from_pretrained("tiansz/bert-base-chinese")
```



同样都是Huggingface上的模型，有的没有提供构建代码，有的会提供构建代码，这是因为你本地的transformers框架是通过枚举的方式纳入了很多模型的构建代码，而transfomers你又不会时时去更新，所以很多新的模型是没有包含在内的，这个时候transformers给你一个选择从模型的仓库中下载构建代码，当然这些代码必须按照transformers的规定，比如 `THUDM/chatglm2-6b`

![image-20241031094656975](D:\dev\php\magook\trunk\server\md\img\image-20241031094656975.png)

在启动的时候要指定`trust_remote_code=True`

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval() # 切换到评估模式，否则会影响模型的权重参数
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)

```



在保存的时候，我们通常只需要保存模型参数即可，然后将其上传到HF

```python
torch.save(model.state_dict(), 'model_state_dict.pt')
```





延伸阅读：

[扒一扒HuggingFace源码-模型加载篇](https://mp.weixin.qq.com/s/v3AQvA23kgL2Q3mqorExNw)

[如何在HuggingFace上开源你自己的大模型？(小模型模型同样适用)](https://zhuanlan.zhihu.com/p/665201835)