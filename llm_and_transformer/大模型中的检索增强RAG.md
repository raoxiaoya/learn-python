大模型中的检索增强RAG，Embedding，Index，Rerank，BGE，Faiss，chunking



### 一、什么是RAG

检索增强生成 RAG（Retrieval-Augmented Generation），LLM是一个通用的知识库，因此不会有偏向，虽然它也很强大，但是难以做到定制化，特别是深入到具体应该的时候就变得鸡肋，因此RAG就应运而生，它要求LLM以你提供的知识库为参考来生成内容，旨在利用大模型的理解和归纳能力。RAG可以有效解决LLM的三个主要问题：1、数据时效性。2、幻觉。3、数据安全问题。

RAG的具体步骤：

1. 加载文件
2. 阅读文本
3. 文本分段
4. 文本向量化
5. 问题向量化
6. 向量相似度匹配,，找的最相似的topk
7. 如果没有rerank步骤，将这topk条信息作为上下文拼接到prompt中交给LLM，让其概括出答案。
8. 如果存在rerank步骤，对topk条信息进行rerank，找到分数最高的n条，比如k=30，n=10，将这n条信息作为上下文拼接到prompt中交给LLM，让其概括出答案。



这里面影响RAG效果的主要有：

1. 文本分段：将PDF/TXT/WORD等文档进行分割，这个好理解，断句是否准确会影响理解。
2. 相似度匹配：匹配的是否准确也是个问题。
3. 是否有rerank操作，以及rerank模型的选择。



#### 1、FlagOpen/FlagEmbedding 

FlagOpen 是北京人工智能研究院 Beijing Academy of Artificial Intelligence (BAAI) 的开源项目，[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)  是它的子项目。FlagEmbedding专注于检索增强llm领域，它提供了一系列的模型用于RAG。



向量化模型：`BAAI/bge-base-en-v1.5`，中文则替换成`zh`，base 可以替换成 small, large ，分别代表模型大小。

rerank模型：`BAAI/bge-reranker-large`



模型可以在 HuggingFace 下载。



教程 https://github.com/FlagOpen/FlagEmbedding/tree/master/Tutorials



##### Embedding示例

https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/quick_start.ipynb

```python

'''
pip install -U FlagEmbedding
'''

# 提供10个句子
corpus = [
    "Michael Jackson was a legendary pop icon known for his record-breaking music and dance innovations.",
    "Fei-Fei Li is a professor in Stanford University, revolutionized computer vision with the ImageNet project.",
    "Brad Pitt is a versatile actor and producer known for his roles in films like 'Fight Club' and 'Once Upon a Time in Hollywood.'",
    "Geoffrey Hinton, as a foundational figure in AI, received Turing Award for his contribution in deep learning.",
    "Eminem is a renowned rapper and one of the best-selling music artists of all time.",
    "Taylor Swift is a Grammy-winning singer-songwriter known for her narrative-driven music.",
    "Sam Altman leads OpenAI as its CEO, with astonishing works of GPT series and pursuing safe and beneficial AI.",
    "Morgan Freeman is an acclaimed actor famous for his distinctive voice and diverse roles.",
    "Andrew Ng spread AI knowledge globally via public courses on Coursera and Stanford University.",
    "Robert Downey Jr. is an iconic actor best known for playing Iron Man in the Marvel Cinematic Universe.",
]

# 问题
query = "Who could be an expert of neural network?"


# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the query and corpus
# The embedding of each sentence is a vector with length 768
corpus_embeddings = model.encode(corpus)
query_embedding = model.encode(query)

print("shape of the query embedding:  ", query_embedding.shape)
print("shape of the corpus embeddings:", corpus_embeddings.shape)
'''
shape of the query embedding:   (768,)
shape of the corpus embeddings: (10, 768)
'''

# 计算相似度

sim_scores = query_embedding @ corpus_embeddings.T
print(sim_scores)
'''
[0.39290053 0.6031525  0.32672375 0.6082418  0.39446455 0.35350388
 0.4626108  0.40196604 0.5284606  0.36792332]
'''

# 排序

# get the indices in sorted order
sorted_indices = sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True)
print(sorted_indices) # [3, 1, 8, 6, 7, 4, 0, 9, 5, 2]

```

效果评估

即，我们给出问题和答案，看看与模型跑出来的差异多大

```python
# 准备多个问题，知识库还是上面的10个句子
queries = [
    "Who could be an expert of neural network?",
    "Who might had won Grammy?",
    "Won Academy Awards",
    "One of the most famous female singers.",
    "Inventor of AlexNet",
]
# 对应的答案的索引
ground_truth = [
    [1, 3],
    [0, 4, 5],
    [2, 7, 9],
    [5],
    [3],
]

# use bge model to generate embeddings for all the queries
queries_embedding = model.encode(queries)
# compute similarity scores
scores = queries_embedding @ corpus_embeddings.T
# get he final rankings
rankings = [sorted(range(len(sim_scores)), key=lambda k: sim_scores[k], reverse=True) for sim_scores in scores]

print(rankings)

'''
[[3, 1, 8, 6, 7, 4, 0, 9, 5, 2],
 [5, 0, 3, 4, 1, 9, 7, 2, 6, 8],
 [3, 2, 7, 5, 9, 0, 1, 4, 6, 8],
 [5, 0, 4, 7, 1, 9, 2, 3, 6, 8],
 [3, 1, 8, 6, 0, 7, 5, 9, 4, 2]]
'''

```

Mean Reciprocal Rank ([MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)) is a widely used metric in information retrieval to evaluate the effectiveness of a system. Here we use that to have a very rough idea how our system performs.

```python
def MRR(preds, labels, cutoffs):
    mrr = [0 for _ in range(len(cutoffs))]
    for pred, label in zip(preds, labels):
        for i, c in enumerate(cutoffs):
            for j, index in enumerate(pred):
                if j < c and index in label:
                    mrr[i] += 1/(j+1)
                    break
    mrr = [k/len(preds) for k in mrr]
    return mrr
```

We choose to use 1 and 5 as our cutoffs, with the result of 0.8 and 0.9 respectively.

```python
cutoffs = [1, 5]
mrrs = MRR(rankings, ground_truth, cutoffs)
for i, c in enumerate(cutoffs):
    print(f"MRR@{c}: {mrrs[i]}")
    

'''
MRR@1: 0.8
MRR@5: 0.9
'''
```



##### Rerank示例

https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/5_Reranking/reranker.ipynb



Reranker 采用交叉编码器架构设计，精度比向量模型更高但推理效率较低。它同时接收查询和文本，并直接输出它们的相似度得分。它在对查询与文本的相关性进行评分方面更有能力，但代价是速度较慢。因此，完整的检索系统通常在第一阶段包含检索器进行大范围检索，然后由重排器对结果进行更精确的重排。

```python
'''
pip install -U FlagEmbedding faiss-cpu
'''

# 1. Dataset

from datasets import load_dataset
import numpy as np

data = load_dataset("namespace-Pt/msmarco", split="dev")

queries = np.array(data[:100]["query"])
corpus = sum(data[:5000]["positive"], [])

# 2. Embedding

from FlagEmbedding import FlagModel

# get the BGE embedding model
model = FlagModel('BAAI/bge-base-en-v1.5',
                  query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                  use_fp16=True)

# get the embedding of the corpus
corpus_embeddings = model.encode(corpus)

print("shape of the corpus embeddings:", corpus_embeddings.shape)
print("data type of the embeddings: ", corpus_embeddings.dtype)
'''
shape of the corpus embeddings: (5331, 768)
data type of the embeddings:  float32
'''

# 3. Indexing

import faiss

# get the length of our embedding vectors, vectors by bge-base-en-v1.5 have length 768
dim = corpus_embeddings.shape[-1]

# create the faiss index and store the corpus embeddings into the vector space
index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
corpus_embeddings = corpus_embeddings.astype(np.float32)
index.train(corpus_embeddings)
index.add(corpus_embeddings)

print(f"total number of vectors: {index.ntotal}")
# total number of vectors: 5331

```



这里介绍一下 Faiss：是一个由facebook开发以用于高效相似性搜索和密集向量聚类的库。它能够在任意大小的向量集中进行搜索。它还包含用于评估和参数调整的支持代码。Faiss是用C++编写的，带有Python的完整接口。一些最有用的算法是在GPU上实现的。Faiss官方仓库为:[faiss](https://github.com/facebookresearch/faiss)。

所谓相似性搜索是指通过比较多维空间中数据之间的相似性来搜索与输入数据最相似的目标数据。例如人脸识别中，通过比较人脸向量之前的距离来识别当前人脸与哪张人脸相似。因此，该技术被广泛应用于信息检索、计算机视觉、数据分析等领域。如果要检索的数据很多时，那么就需要一个向量检索库来加速检索。Faiss包含多种相似性搜索方法，并提供cpu和gpu版本支持。Faiss的优势在于通过较小的精度损失提高向量相似度的检索速度和减少内存使用量。

在生产环境中就需要使用专业的向量数据库，比如Elasticsearch，Pinecone。

本例就是将Faiss作为向量数据库使用，将数据集的向量存入其中，然后进行检索匹配。

```python
# 4. Retrieval

query_embeddings = model.encode_queries(queries)
ground_truths = [d["positive"] for d in data]
corpus = np.asarray(corpus)

from tqdm import tqdm

res_scores, res_ids, res_text = [], [], []
query_size = len(query_embeddings)
batch_size = 256
# The cutoffs we will use during evaluation, and set k to be the maximum of the cutoffs.
cut_offs = [1, 10]
k = max(cut_offs)

for i in tqdm(range(0, query_size, batch_size), desc="Searching"):
    q_embedding = query_embeddings[i: min(i+batch_size, query_size)].astype(np.float32)
    # search the top k answers for each of the queries
    score, idx = index.search(q_embedding, k=k)
    res_scores += list(score)
    res_ids += list(idx)
    res_text += list(corpus[idx])

# 得到结果为 res_scores, res_ids, res_text，都是List。其中 res_text 为定位到的文本，后面将会放到 reranker 中去排序。
    
# 5. Reranking
```

中间插播一下 reranker 的简单使用示例

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) 
# Setting use_fp16 to True speeds up computation with a slight performance degradation

# use the compute_score() function to calculate scores for each input sentence pair
scores = reranker.compute_score([
    ['what is panda?', 'Today is a sunny day'], 
    ['what is panda?', 'The tiger (Panthera tigris) is a member of the genus Panthera and the largest living cat species native to Asia.'],
    ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
    ])
print(scores)
# [-9.474676132202148, -2.823843240737915, 5.76226806640625]
```

续上上面的案例

```python
new_ids, new_scores, new_text = [], [], []
for i in range(len(queries)):
    # get the new scores of the previously retrieved results
    new_score = reranker.compute_score([[queries[i], text] for text in res_text[i]])
    # sort the lists of ids and scores by the new scores
    new_id = [tup[1] for tup in sorted(list(zip(new_score, res_ids[i])), reverse=True)]
    new_scores.append(sorted(new_score, reverse=True))
    new_ids.append(new_id)
    new_text.append(corpus[new_id])

    
# 6. Evaluate
# 此处省略，可以去查看文档
```



#### 2、Qwen+Langchain 来实现RAG的过程

示例1：https://qwen.readthedocs.io/zh-cn/latest/framework/Langchain.html

示例2：https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/6_RAG/6.2_RAG_LangChain.ipynb



示例2更简洁！



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

```



加载文档与文本分段

```python
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
```

向量化与匹配topk

```python

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

```

发送到LLM

```python

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



#### 3、Qwen+LlamaIndex来实现RAG的过程

示例1：https://qwen.readthedocs.io/zh-cn/latest/framework/LlamaIndex.html

示例2：https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/6_RAG/6.3_RAG_LlamaIndex.ipynb

LlamaIndex工具旨在为了实现 Qwen2.5 与外部数据（例如文档、网页等）的连接，以快速部署检索增强生成（RAG）技术。

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

# set the parser with parameters
Settings.node_parser = SentenceSplitter(
    chunk_size=1000,    # Maximum size of chunks to return
    chunk_overlap=150,  # number of overlap characters between chunks
)

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

**embed_model**：由来向量化数据的模型，您可以使用 `bge-base-en-v1.5` 模型来检索英文文档，下载 `bge-base-zh-v1.5` 模型以检索中文文档。根据您的计算资源，您还可以选择 `bge-large` 或 `bge-small` 作为向量模型，或调整上下文窗口大小或文本块大小。Qwen2.5模型系列支持最大32K上下文窗口大小（7B 、14B 、32B 及 72B可扩展支持 128K 上下文，但需要额外配置）。比如`BAAI/bge-base-en-v1.5`



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



实际上，很多的大模型产品都已经封装好了RAG功能，并且提供了很好的部署和操作界面

1、QAnything：https://mp.weixin.qq.com/s/6NVKXpAqCaY_0RI2V8mf7Q

2、FastGPT：https://mp.weixin.qq.com/s/A5SWr2PFZe6ENCd5L_L9sQ



#### 4、Elasticsearch中的相似度检索算法

ES使用的相似度计算算法为kNN（k-nearest neighbor），同时ES底层使用HNSW构建图数据结构以加速向量检索。

https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html



### 二、Rerank

相似和相关是由本质区别，即向量的相似度很高并不意味着相关度很高，比如这两句话“大连理工大学是个很不错的大学” & “大连医科大学是个很不错的大学”。它们相似度很高但相关度很低。

LLM能接收到的tokens是有限制的，在检索出来的topk中，如果正确的信息排在很后面，有可能会被LLM忽视掉。

在实际部署的时候，一般会将知识库分段向量化后存储到向量数据库中，然后根据用户的问题进行向量匹配得到topk，再次使用用户的问题对topk根据得分排序，称为Rerank，它能大大改善准确性。以下是有道 QAnything 的架构图。

![image-20241018095917395](D:\dev\php\magook\trunk\server\md\img\image-20241018095917395.png)



知识库数据量大的场景下两阶段优势非常明显，如果只用**一阶段embedding**检索， 随着数据量增大会出现检索退化的问题，如下图中绿线所示， 二阶段**rerank重排**后能实现准确率稳定增长，即数据越多，效果越好。

![image-20241017171911148](D:\dev\php\magook\trunk\server\md\img\image-20241017171911148.png)



**Rerank模型**可以采用BERT等预训练模型进行微调。训练数据可以是RAG生成的初步回答和正确答案，通过比较初步回答和正确答案，训练Rerank模型学会如何对初步回答进行排序。针对对话系统，Concat Query的训练数据可以进一步提高模型对连续对话的理解和排序能力。



##### Embedding（Bi-Encoder）

为什么rerank要比向量检索的精度高呢？这与他们的原理有关，向量检索是将文本向量化，然后进行相似度计算，而文本向量化的过程会带来精度的损失。其过程如下

![image-20241020142930210](D:\dev\php\magook\trunk\server\md\img\image-20241020142930210.png)

Embedding 本质上是一个双编码器，两个文本在模型内部没有任何信息交互。只在最后计算两个向量的余弦相似度时才进行唯一一次交互。所以 Embedding 检索只能把最相似的文本片段给你，没有能力来判断候选文本和 query 之间的相关性。但是相似又不等于相关。

如下图所示，从某种程度上，Embedding 其实就是在算两个文本块中相似字符的个数占比，它分不清 query 中的重点是大连医科大学，在它看来每个字符的重要性都是一样的。感兴趣的话可以计算一下下图中红字部分的占比，和最后余弦相似度的得分基本是吻合的。

![image-20241021161524160](D:\dev\php\magook\trunk\server\md\img\image-20241021161524160.png)



##### Rerank（Cross-Encoder）

而rerank模型不会向量化，而是将查询与匹配的单个文档 1 对 1 的计算相关性，没有向量化带来的信息损失，必然会得到更好的效果。

![image-20241020142947160](D:\dev\php\magook\trunk\server\md\img\image-20241020142947160.png)

Rerank 本质是一个 Cross-Encoder 的模型。Cross-Encoder 能让两个文本片段一开始就在 BERT 模型各层中通过 self-attention 进行交互。它能够用 self-attention 判断出来这个 query 中的重点在于大连医科大学，而不是怎么样？。所以，如下图所示，大连医科大学怎么样？这个 query 和大连医科大学创建于 1947 年… 更相关。

![image-20241021161757135](D:\dev\php\magook\trunk\server\md\img\image-20241021161757135.png)





##### Cross-Encoder 这么好，为什么不直接用？

因为速度慢。这里说的速度慢不是 cross-encoder 的模型比 bi-encoder 的模型速度慢。关键在于 bi-encoder 可以离线计算海量文本块的向量化表示，把他们暂存在向量数据库中，在问答检索的时候只需要计算一个 query 的向量化表示就可以了。拿着 query 的向量表示去库里找最相似的文本即可。

但是 cross-encoder 需要实时计算两个文本块的相关度，如果候选文本有几万条，每一条都需要和 query 一起送进 BERT 模型中算一遍，需要实时算几万次。这个成本是非常巨大的。

rerank模型的选择，除了上面提到的BGE系列，还有有道的BCE模型[bce-reranker-base_v1](https://huggingface.co/maidalun1020/bce-reranker-base_v1)，综合下载量和测评结果，bge-reranker-v2-m3 看来是当前最佳选择，bge-reranker-large 和 bce-reranker-base_v1 可以作为备选。



### 三、优化

#### 1、文档分块chunking

从最开始的512固定长度切分，到后面的句切分，再到后面的NLTK和[SpaCy](https://zhida.zhihu.com/search?content_id=238573818&content_type=Article&match_order=1&q=SpaCy&zhida_source=entity)，具体可参见 [《最详细的文本分块(Chunking)方法——可以直接影响基于LLM应用效果》](https://www.luxiangdong.com/2023/09/20/chunk/)

关于tokens如何计算，不同的模型有不同的定义。有的模型一个字符为一个token，有的模型是多个字符。因为LLM对tokens有限制，所以分块的大小也需要仔细斟酌。







参考文章

[技术分享 基于RAG+Rerank调优让大模型更懂你！](https://zhuanlan.zhihu.com/p/718220120)

[改善大模型 RAG 效果：结合检索和重排序模型](https://zhuanlan.zhihu.com/p/748202826) 

[从《红楼梦》的视角看大模型知识库 RAG 服务的 Rerank 调优](https://zhuanlan.zhihu.com/p/699339963)

[Rerank——RAG中百尺竿头更进一步的神器，从原理到解决方案](https://zhuanlan.zhihu.com/p/676996307)

[知识库问答，数据越多效果越好吗？](https://mp.weixin.qq.com/s/jo1TsJGx_cVVNpJEqSbgUQ)