### 1、关于Hugging Face

https://zhuanlan.zhihu.com/p/535100411


Hugging face 起初是一家总部位于纽约的聊天机器人初创服务商，他们本来打算创业做聊天机器人，然后在github上开源了一个Transformers库，虽然聊天机器人业务没搞起来，但是他们的这个库在机器学习社区迅速大火起来。目前已经共享了超100,000个预训练模型，10,000个数据集，变成了机器学习界的github。

HuggingFace 官网 [http://www.huggingface.co](https://link.zhihu.com/?target=http%3A//www.huggingface.co./)

在这里主要有以下大家需要的资源。

1. Datasets：数据集，以及数据集的下载地址
2. Models：各个预训练模型
3. course：免费的nlp课程，可惜都是英文的
4. docs：文档

### 2、名词解释

#### 2.1、pytorch 与 torch

transformers 依赖于 Pytorch ，也就是 Python 版本的 torch，在安装 transformers 的时候会被自动安装上，注意的是，安装是`pip install pytorch`，使用的是`import torch`

官网 https://pytorch.org/

#### 2.2、tensorflow 与 pytorch

它们是用来生产机器学习模型的框架，两者有区别，目前流行的是pytorch。

#### 2.3、transformers

仓库 https://github.com/huggingface/transformers

`pip install transformers`

Transformers是一个框架，以前称为`pytorch-transformers和pytorch-pretrained-bert`，提供用于自然语言理解（NLU）和自然语言生成（NLG）的最先进的模型（BERT、GPT、GPT-2、Transformer-XL、XLNet、XLM、RoBERTa、DistilBERT、CTRL、CamemBERT、ALBERT、T5、XLM-RoBERTa、MMBT、FlauBERT......） ，拥有超过32种预训练模型，支持100多种语言，并且在TensorFlow 2.0和PyTorch之间具有深厚的互操作性。

Transformers 支持三个最热门的深度学习库： [Jax](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/) 以及 [TensorFlow](https://www.tensorflow.org/) — 并与之无缝整合。你可以直接使用一个框架训练你的模型然后用另一个加载和推理。

总结就是，transformers 是一个框架，也是一个平台（连接Hugging Face），此平台预制了一些先进的模型，也有别人上传的模型，你也可以上传自己的模型；它可以很方便的加载平台上和你本地的模型，用来训练，微调，推理；它同时兼容Jax, Pytorch, TensorFlow三种工具生产的模型。

#### 2.4、sentence-transformers

`pip install sentence-transformers`

仓库 https://github.com/UKPLab/sentence-transformers

文档 https://www.sbert.net/

state-of-the-art 最新式的，最先进的。

它是一个python库，是最先进的，用于处理句子，段落，图像的向量化。在句子和段落方面支持100种语言，可以使用 cosine 相似度算法对向量进行计算找出相似的句子。

它依赖于 PyTorch 和 Transformers ，并提供了一大批预训练模型用来处理不同的任务。

依赖要求 Python >= 3.6; PyTorch >= 1.6; Transformers >= 4.6

在 Hugging Face 网站上，sentence-transformers 旗下的模型都是在 sentence-transformers 目录下，即，模型名称是 `sentence-transformers/xxxx`

比如常用的模型 `sentence-transformers/all-MiniLM-L6-v2`，[地址](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，此模型是用来将句子和段落转换成384维度的稠密向量，用来聚类，语义检索，语义相似度。

模型的加载规则为：如果能在本地查找到目录就直接加载。本地没有就去HuggingFace下载，如果是`sentence-transformers`旗下的模型，可以输入`sentence-transformers/b`或者`b`，如果是别人的模型就需要输入`a/b`，模型只会被下载一次，默认存储在`C:\Users\Administrator.DESKTOP-TPJL4TC\.cache\torch`，目录的命名规则为`a_b`，比如`sentence-transformers_multi-qa-MiniLM-L6-cos-v1`，所以也可以手动下载模型文件。

一般会提供基于 tensorflow 和 touch  两个版本的模型，可根据自己得需要选择其一下载，毕竟模型文件都比较大。

如果想把模型下载到其他地方，可以这样

```python
MODEL_CACHE_FOLDER = "D:/dev/php/magook/trunk/server/torch/sentence_transformers"

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1', cache_folder=MODEL_CACHE_FOLDER)
```

模型目录为`D:\dev\php\magook\trunk\server\torch\sentence_transformers\sentence-transformers_multi-qa-MiniLM-L6-cos-v1`



可以阅读`SentenceTransformer`构造函数了解加载规则。

#### 2.5、BERT & SBERT

https://zhuanlan.zhihu.com/p/98855346

BERT 的全称为 `Bidirectional Encoder Representation from Transformers`，是一个预训练的语言表征模型。它强调了不再像以往一样采用传统的单向语言模型或者把两个单向语言模型进行浅层拼接的方法进行预训练，而是采用新的**masked language model（MLM）**，以致能生成**深度的双向**语言表征。BERT论文发表时提及在11个NLP（Natural Language Processing，自然语言处理）任务中获得了新的 state-of-the-art 的结果，令人目瞪口呆。

该模型有以下主要优点：

1. 采用MLM对双向的Transformers进行预训练，以生成深层的双向语言表征。
2. 预训练后，只需要添加一个额外的输出层进行fine-tune，就可以在各种各样的下游任务中取得state-of-the-art的表现。在这过程中并不需要对BERT进行任务特定的结构修改。

市面上很多的模型都是基于BERT来针对特定的数据集做的训练得到的。

**SBERT，即 Sentence-BERT**

随着2018年底Bert的面世，NLP进入了预训练模型的时代。各大预训练模型如GPT-2，Robert，XLNet，Transformer-XL，Albert，T5等等层数不穷。

但是几乎大部分的这些模型均不适合语义相似度搜索，也不适合非监督任务，比如聚类。而解决聚类和语义搜索的一种常见方法是将每个句子映射到一个向量空间，使得语义相似的句子很接近。说到这，可能有的人会尝试将整个句子输入预训练模型中，得到该句的句向量，然后作为句子的句向量表示。

但是这样得到的句向量真的好吗？在论文[《Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks》](https://arxiv.org/abs/1908.10084)就指出了，这样得到的句向量不具有语义信息，也就是说，两个相似的句子，得到的句向量可能会有很大的差别。此外，不仅句向量表示的语义方面存在很大差别，而且，它要求两个句子都被输入到网络中，从而导致巨大开销：从10000个句子集合中找到最相似的sentence-pair需要进行大约5000万个推理计算（约65小时），因此，这种方案不太友好。

面对上述预训练模型在文本语义相似度等句子对的回归任务上的问题，本文提出了Sentence-BERT（SBERT），对预训练的BERT进行修改：使用孪生(Siamese)和三级(triplet)网络结构来获得语义上有意义的句子embedding，以此获得定长的sentence embedding，使用余弦相似度，曼哈顿Manhatten，欧式Euclidean距离等进行比较找到语义相似的句子。通过这样的方法得到的SBERT模型，在文本语义相似度等句子对的回归任务上吊打BERT , RoBERTa。

上面提到的`sentence-transformers`的内部实现就是SBERT，当然上面的论文 [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) 也是这个组织发布的。这也是为什么 `sentence-transformers` 的官方文档是 https://www.sbert.net/

#### 2.6、fine-tune

就是微调的意思，其操作过程是，针对已经训练好的预训练模型，在自己的定制数据集上进行再训练，以期待能够对模型的参数（权重）起到微调的效果，检验的标准就是训练前后的对比，是否更加拟合了自己的预期结果。

`sentence-transformers`提供了对模型的 `fine-tune`功能。并将微调后的模型保存在本地。

### 3、sentence-transformers简单使用

以下例子来自于官网文档 www.sbert.net

```bash
python 3.8.8

> conda list torch
# packages in environment at d:\ProgramData\Anaconda3:
#
# Name                    Version                   Build  Channel
torch                     2.0.1                    pypi_0    pypi
torchvision               0.15.2                   pypi_0    pypi

> conda list transformers
# packages in environment at d:\ProgramData\Anaconda3:
#
# Name                    Version                   Build  Channel
sentence-transformers     2.2.2                    pypi_0    pypi
transformers              4.32.0                   pypi_0    pypi
```



#### 3.1、句子向量化

```bash
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
sentence_embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```

#### 3.2、句子相似度计算

```bash
from sentence_transformers import SentenceTransformers, util

model = SentenceTransformers('all-MiniLM-L6-v2')
emb1 = model.encode('我很生气')
emb2 = model.encode('气死我了')

cos_sim = util.cos_sim(emb1, emb2)
print('Cossim-Similarity:', cos_sim)

# Cosine-Similarity: tensor([[0.7948]])
```

#### 3.3、语义检索

```bash
from sentence_transformers import SentenceTransformers, util

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode([
    'London has 9,787,426 inhabitants at the 2011 census',
    'London is known for its finacial district'
])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))

# Similarity: tensor([[0.5472, 0.6330]])
```

其他示例参考官网文档。

### 4、HuggingFace上好用的模型介绍

文本语义相似度任务STS（Semantic Textual Similarity）

```bash
1、sentence-transformers/distiluse-base-multilingual-cased
    It maps sentences & paragraphs to a 512 dimensional dense vector space and can be used for tasks like clustering or semantic search.
    
2、sentence-transformers/all-MiniLM-L6-v2
	It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

3、sentence-transformers/multi-qa-MiniLM-L6-cos-v1
	It maps sentences & paragraphs to a 384 dimensional dense vector space and was designed for semantic search. It has been trained on 215M (question, answer) pairs from diverse sources. For an introduction to semantic search, have a look at: SBERT.net - Semantic Search
```















