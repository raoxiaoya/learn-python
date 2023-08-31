### 6、使用Elasticsearch向量数据库

参考文章 https://dbaplus.cn/news-160-5359-1.html

向量检索是基于向量之间的距离对已有documents进行相关性排序，和输入document向量距离越小则认为在某个维度上越相似会优先排在前面。

document可以是世界万物，比如图片，一段音视频，一段文字，一张人脸等，对于任何种document， 你都可以根据自己的需求对其进行模型训练，基于训练好的模型抽取特征进完成documents相似性检索。

应用场景
- QA：用户输入一段描述，给出最佳匹配的答案。传统基于关键字搜索问答的局限性之一在于用户必须了解一些特殊的名词，假如关键字没有匹配上则没有返回结果。而在使用词向量之后，直接输入类似的描述性语言可以获得最佳匹配的答案。
- 文章搜索：有时候只记得一篇文章在表达什么意思，而忘记了文章标题和关键字。这时候只需要输入自己记得的大致意思和记得句子，即可根据描述中隐藏的语义信息搜索到最佳匹配的文章。
- 图片搜索：这里的图片搜索有两种含义，一种是讲图片中的特征值进行提取生成向量，实现以图搜图模式的搜索。另一种是基于图片tag的方式，将tag进行向量化，这样可以搜索到语义相近的tag的图片，而不必完全相等。这两种方式在ES的词向量搜索中都可以支持。
- 社交网络：社交网络中的人都是一个单词，而其关注和粉丝都是和其相关的单词，因此可以每一个人的关注和粉丝形成一段“文本”去训练模型。想计算两个人是否相似或者两个的距离，只需要计算两个人的向量即可。

#### 6.1、下载安装 Elasticsearch

由于 Elasticsearch 是使用 Java 构建的，所以你需要确保系统中安装了 Java 开发工具包（JDK）。如果已经安装了 Java，请检查其版本是否为 17 或更高版本。

下载安装 Elasticsearch-8.5.3：https://www.elastic.co/cn/downloads/past-releases/elasticsearch-8-5-3

解压到 D:\elasticsearch-8.5.3

修改文件 `elasticsearch-8.5.3/config/elasticsearch.yml`设置`xpack.security.enabled: false`

启动 Elasticsearch

```bash
> cd elasticsearch-8.5.3/bin
> elasticsearch
```

```bash
> curl localhost:9200

{
  "name" : "WINDOWS10-JACK",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "gcNighT6T6muORs4zxdkJA",
  "version" : {
    "number" : "8.5.3",
    "build_flavor" : "default",
    "build_type" : "zip",
    "build_hash" : "4ed5ee9afac63de92ec98f404ccbed7d3ba9584e",
    "build_date" : "2022-12-05T18:22:22.226119656Z",
    "build_snapshot" : false,
    "lucene_version" : "9.4.2",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

#### 6.2、将文本转换为向量

项目目录 https://github.com/phprao/learn-python/text_vectors_and_elasticsearch  

参考项目 https://github.com/SeaseLtd/vector-search-elastic-tutorial

安装python依赖

```bash
pip install sentence_transformers
```

把 `vector-search-elastic-tutorial\from_text_to_vectors\example_input\documents_10k.tsv`复制到自己项目中。

文件 `text_to_vector.py`

```python
'''
文本向量化
'''

from sentence_transformers import SentenceTransformer
import torch
import sys
from itertools import islice
import time

BATCH_SIZE = 100  # 每批 100个，10k就是100批

INFO_UPDATE_FACTOR = 1

MODEL_NAME = 'all-MiniLM-L6-v2'  # 使用的模型，如果本地没有就会去自动下载

# Load or create a SentenceTransformer model.
model = SentenceTransformer(MODEL_NAME)

# Get device like 'cuda'/'cpu' that should be used for computation.
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

# 本机没有可用的GPU，会使用CPU来运算
print(model.device)


def batch_encode_to_vectors(input_filename, output_filename):
    # Open the file containing text.
    with open(input_filename, 'r') as documents_file:
        # Open the file in which the vectors will be saved.
        with open(output_filename, 'w+') as out:
            processed = 0
            # Processing 100 documents at a time.
            for n_lines in iter(lambda: tuple(islice(documents_file, BATCH_SIZE)), ()):
                processed += 1
                if processed % INFO_UPDATE_FACTOR == 0:
                    print("Processed {} batch of documents".format(processed))

                # Create sentence embedding
                vectors = encode(n_lines)

                # Write each vector into the output file.
                for v in vectors:
                    out.write(','.join([str(i) for i in v]))
                    out.write('\n')


def encode(documents):
    embeddings = model.encode(documents, show_progress_bar=True)
    print('Vector dimension: ' + str(len(embeddings[0])))
    return embeddings


def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    initial_time = time.time()
    batch_encode_to_vectors(input_filename, output_filename)
    finish_time = time.time()
    print('Vectors created in {:f} seconds\n'.format(
        finish_time - initial_time))


if __name__ == "__main__":
    main()

```

运行

```bash
python text_to_vector.py "../files/documents_10k.tsv" "../files/vector_documents_10k.tsv"
```

会先下载模型，切记要关闭VPN，代理等。

然后就会依次执行 100 batch ，打印如下

```bash
Downloading......
cpu
Processed 1 batch of documents
Batches: 100%|███████████████████████████| 4/4 [00:06<00:00,  1.74s/it]]
Vector dimension: 384
Processed 2 batch of documents
Batches: 100%|███████████████████████████| 4/4 [00:03<00:00,  1.03it/s]]
Vector dimension: 384
...
...
Processed 100 batch of documents
Batches: 100%|███████████████████████████| 4/4 [00:04<00:00,  1.02s/it]]
Vector dimension: 384
Vectors created in 481.826802 seconds
```

因为 `documents_10k.tsv`有10000行句子，每一行类似于

```bash
The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.
```

按照程序中的设置为每100行为一个batch，所以会有100 batch，每一个句子都会被计算成一个向量，作为一行数据存储在`vector_documents_10k.tsv`，因此结果文件也有10000行，每一行类似于

```bash
0.03678231,0.07242356,0.047704875,0.03489034,0.06181074,0.0022823475,0.052583564,.......................-0.08142317,0.015440078,-0.035280623,0.040732816,-0.021872856,-0.019021988,0.009163804,-0.037747797,0.03987788,0.038443923,-0.033567403,0.031574957,-0.1442669,0.059140205,0.14816678,0.01050604,-0.032048155,0.005427416
```

在本例程中，我们使用了 all-MiniLM-L6-v2（BERT）模型，它将句子映射到一个 384 维的稠密向量空间。每个维度之间是逗号隔开的，以上省略了很多，本质上是一个csv文件格式。

注意这里的多少维，其实指的是底层神经网络隐藏层的单元数，也可以理解为每一个句子有多少个属性来定义。

#### 6.3、将向量写入到 Elasticsearch

我们可以开始创建一个索引，使用显式映射来精确定义数据的结构。以下是创建 “neural_index” 索引的 API 请求：

```bash
curl http://localhost:9200/neural_index/ -XPUT -H 'Content-Type: application/json' -d '{
"mappings": {
    "properties": {
        "general_text_vector": {
            "type": "dense_vector",
            "dims": 384,
            "index": true,
            "similarity": "cosine"
        },
        "general_text": {
            "type": "text"
        },
        "color": {
            "type": "text"
        }
    }
}}'
```

响应

```bash
{"acknowledged":true,"shards_acknowledged":true,"index":"neural_index"}
```

为了检查索引的创建情况，以下是返回有关索引信息的 API 请求：

```bash
curl -XGET http://localhost:9200/neural_index

{
  "neural_index": {
    "aliases": {},
    "mappings": {
      "properties": {
        "color": {
          "type": "text"
        },
        "general_text": {
          "type": "text"
        },
        "general_text_vector": {
          "type": "dense_vector",
          "dims": 384,
          "index": true,
          "similarity": "cosine"
        }
      }
    },
    "settings": {
      "index": {
        "routing": {
          "allocation": {
            "include": {
              "_tier_preference": "data_content"
            }
          }
        },
        "number_of_shards": "1",
        "provided_name": "neural_index",
        "creation_date": "1693275729052",
        "number_of_replicas": "1",
        "uuid": "VRV1WoE1T6OlRZ6CLhFA5w",
        "version": {
          "created": "8050399"
        }
      }
    }
  }
}
```

根据我们的映射定义，文档包含三个简单的字段：

- **general_text_vector：** 存储由前面部分中的 Python 脚本生成的 Embedding 向量，类型为 dense_vector （稠密向量）。
- **general_text：** source 字段，包含要转换为向量的文本，类型为文本。
- **color：** 一个附加字段，类型为文本，仅用于展示筛选查询的行为（我们将在搜索部分看到）

Elasticsearch 目前通过 dense_vector 字段类型支持存储向量（浮点值），并使用它们计算文档得分。在这种情况下，我们使用以下定义：

- **dims：**（整数）稠密向量的维度，需要与模型的维度相等。在此例中为 384。
- **index：**（布尔值）默认为 false，但你需要将其设置为 `index:true` 以使用 KNN 搜索 API 搜索向量字段。
- **similarity：**（字符串）用于返回最相似的前 K 个向量的向量相似性函数。在此例中，我们选择了余弦相似度（而不是 L2 范数或点积）。仅在 index 为 true 时需要指定。

我们将 index_options 保留为默认值；这一部分配置了与当前算法（HNSW）密切相关的高级参数，它们会影响索引时图形的构建方式。

目前的限制是：

1）已索引向量（`index:true`）的基数限制为 1024，未索引向量的基数限制为 2048。

2）dense_vector 字段不支持以下功能：排序或聚合；多值；在嵌套映射中索引向量



一旦我们创建了向量 Embedding 和索引，就可以开始推送一些文档了。

以下是可以使用的 _bulk 请求 API 将文档推送到你的 neural_index。

```bash
curl http://localhost:9200/neural_index/_bulk -XPOST -H 'Content-Type: application/json' -d '
{"index": {"_id": "0"}}
{"general_text": "The presence of communication amid scientific minds was equally important to the success of the Manhattan Project as scientific intellect was. The only cloud hanging over the impressive achievement of the atomic researchers and engineers is what their success truly meant; hundreds of thousands of innocent lives obliterated.", "general_text_vector": [0.03678231,0.07242356,0.047704875,0.03489034,0.06181074,0.0022823475,0.052583564,0.013747121,-0.00605947,0.020382846,0.022016445,0.017639305,0.026344897,0.0063015413,-0.0465571,0.05615707,0.0068715913,-0.115466185,-0.08517982,-0.032203667,-0.010798692,0.037556708,0.065674365,-0.016283888,0.005361602,0.099470876,0.044416744,-0.053873084,-0.021753304,-0.015601038,0.026994798,0.03152312,0.069841474,-0.011057814,0.040827464,0.062586494,0.06451483,0.009646311,0.064563245,-0.075330645,-0.086552106,-0.072232135,0.04081632,0.03761102,-0.045341577,-0.07036705,0.012725855,-0.08938016,-0.023172807,0.015793981,-0.05936412,0.0023590676,0.04924386,0.025908371,0.04115568,-0.021968503,0.03629561,-0.008033628,-0.040976387,-0.04566905,-0.031725287,-0.067273915,-0.0064599463,-0.022531327,0.10620401,-0.020472264,-0.00088112836,-0.0214439,-0.033033147,0.09433366,0.13675828,0.031479493,0.024484139,-0.023862975,0.0087755155,0.014587189,-0.023618413,0.0044724774,0.016848488,0.09438903,0.06576519,-0.035623156,-0.0740419,0.04581393,-0.03965107,0.015470117,-0.093617335,-0.03699269,-0.03404914,-0.04870865,-0.0032372223,-0.064391114,-0.00031670099,0.018688401,0.029805701,0.0072495053,0.028061153,-0.059177734,6.0052964e-05,0.012951801,0.010668324,0.016489835,-0.07053622,-0.04187402,0.009770811,-0.068676755,0.0140780145,-0.04037583,-0.013411637,0.06492108,-0.020671189,-0.04987132,-0.017610116,0.12054466,0.10405108,-0.02000308,0.05990099,0.11543012,-0.057187766,-0.104558334,0.0074519427,0.015892118,0.0214665,0.06169873,-0.063479476,-0.04144271,0.0005453495,1.283458e-33,0.007954626,0.010941109,-0.019417305,0.09516082,0.029694946,-0.029924773,-0.001929065,-0.07834759,-0.0011461716,-0.032956984,-0.024771098,-0.016988266,0.07273439,-0.0094948355,-0.0064977445,-0.0042426186,-0.029242236,0.04047869,0.016924307,-0.02746305,-0.020931361,-0.039669942,-0.00996878,-0.022671964,-0.01222592,0.0012076936,0.046402536,-0.008073234,-0.08381604,-0.0070763365,-0.01289721,0.1145123,-0.007900905,-0.0007906223,0.034898713,0.0048103803,-0.06152029,-0.07121459,0.015338203,0.029281994,0.020594008,0.07391126,-0.043293785,-0.0011567591,0.08975668,0.09694041,-0.029358536,0.028466417,0.044734888,-0.017081622,-0.049763612,0.049670517,0.010242273,-0.051043995,0.056479186,0.036318563,0.049272105,-0.0043922793,0.010414609,0.058024343,-0.101056986,0.0060702804,-0.031693406,0.020037519,0.017704943,0.054810587,-0.0924704,-0.0012925785,-0.0023938406,0.027001584,-0.058537025,-0.010566472,-0.027242212,-0.057774194,-0.048308816,-0.016389664,-0.042802803,-0.043408,0.077265814,0.028595272,0.031873118,-0.032264166,0.0038445517,-0.07218837,0.053970605,0.000977916,-0.0031416717,-0.030118452,-0.066272914,0.048672095,0.085155144,-0.089326784,0.04592053,-0.07312609,-0.13800502,-3.1553027e-33,-0.1344867,-0.035119805,-0.065308355,0.02835039,-0.006509474,0.008085067,-0.08536709,-0.098481074,-0.046450056,0.09518576,0.025274284,0.039391797,-0.058135014,-0.019020095,0.011532342,0.0038693834,0.050569884,-0.08717485,0.019322127,0.038812637,0.10124947,-0.008012981,-0.06316764,-0.08075084,-0.035215892,0.05520921,0.01416924,-0.028139291,0.026325438,-0.06759833,0.056114625,-0.004118644,-0.047167934,-0.01686735,0.033817887,0.031771943,0.08748101,0.011860394,-0.046322443,-0.051414937,-0.028175032,0.056591175,-0.14519005,-0.016210709,-0.011643618,-0.012496821,-0.017958611,0.056452077,-0.09349239,-0.0019594629,-0.017791625,-0.0029609625,-0.00058943924,-0.060910977,0.020739306,-0.018652255,-0.0012633075,0.06474534,0.051982157,0.018968849,0.012264262,-0.015289237,0.063957416,0.07524408,-0.013623159,-0.030034194,0.027646221,0.1189337,-0.05872378,-0.043435983,0.0374322,0.05568836,-0.009367028,0.030177563,0.021913797,0.0009393531,-0.069067605,-0.05426868,-0.0840546,0.056699127,-0.02801745,0.068634726,-0.032972787,-0.056349322,0.094851084,0.03111054,0.045052815,-0.060022958,-0.061548293,-0.059302438,-0.023715744,-0.018102199,-0.02058774,-0.033178322,-0.05422994,-3.662482e-08,0.020311506,-0.02437668,-0.008285616,-0.0606819,0.025469005,-0.04012086,0.020066328,0.065136194,-0.019260487,0.07312178,-0.026498124,-0.0066478574,-0.047779668,0.115038596,0.07328099,-0.033251584,0.04390376,-0.06340511,-0.045961317,-0.011702473,0.073790565,0.022466892,-0.002455196,-0.0064459695,-0.0051715644,0.06062236,0.023674298,-0.022697013,-0.0530832,-0.0026887509,-0.10601409,-0.08888704,0.011884046,0.041220695,0.052729934,-0.0009493992,0.036308456,-0.056280598,0.03395851,0.015440755,-0.0845933,0.070367664,0.017332021,0.10938247,0.083649196,0.059358895,-0.08142317,0.015440078,-0.035280623,0.040732816,-0.021872856,-0.019021988,0.009163804,-0.037747797,0.03987788,0.038443923,-0.033567403,0.031574957,-0.1442669,0.059140205,0.14816678,0.01050604,-0.032048155,0.005427416], "color": "black"}
'
```

注意参数格式问题，最后一个单引号要单独一行，否则会报错。成功运行后响应

```bash
{
  "took": 298,
  "errors": false,
  "items": [
    {
      "index": {
        "_index": "neural_index",
        "_id": "0",
        "_version": 1,
        "result": "created",
        "_shards": {
          "total": 2,
          "successful": 1,
          "failed": 0
        },
        "_seq_no": 0,
        "_primary_term": 1,
        "status": 201
      }
    }
  ]
}
```

将 _builk 请求封装成脚本，参考文件 `create_body_for_build.py`

好在 Elasticsearch 的官方提供了 Python 客户端 elasticsearch 可以简化这一步骤。

```bash
pip install elasticsearch
```

封装之后的文件为 `indexer_elastic.py`

执行

```bash
python indexer_elastic.py "../files/documents_10k.tsv" "../files/vector_documents_10k.tsv"
```

响应

```bash
Success - 1001 , Failed - 0
Success - 1000 , Failed - 0
Success - 1000 , Failed - 0
Success - 1000 , Failed - 0
Success - 1000 , Failed - 0
Success - 1000 , Failed - 0
Success - 1000 , Failed - 0
Success - 1000 , Failed - 0
Success - 1000 , Failed - 0
Finished
Documents indexed in 26.362656 seconds
```

在这一步之后，已经在 Elasticsearch 中索引了 10000 个文档，并且我们已经准备好根据查询检索这些文档。

要检查文档数量，你可以使用 cat indices API，该 API 可以显示集群中每个索引的高级信息：

```bash
> curl -XGET http://localhost:9200/_cat/count/neural_index?v
epoch      timestamp count
1693279444 03:24:04  10000
```

#### 6.4、使用向量字段进行检索

稠密向量字段可以通过以下方式使用：

- **精确的暴力 KNN：** 使用 script_score 查询
- **近似的 KNN：** 使用搜索 API 中的 knn 选项，找到与查询向量最相似的前 k 个向量

下面示例中的查询是：“what is a bank transit number”。为了将其转换为向量，我们运行了一个自定义的 Python 脚本：

文件 `single_sentence_transformers.py`

```python
from sentence_transformers import SentenceTransformer

# The sentence we like to encode.
sentences = ["what is a bank transit number"]
# Load or create a SentenceTransformer model.
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Compute sentence embeddings.
embeddings = model.encode(sentences)
# Create a list object, comma separated.
vector_embeddings = list(embeddings)
print(vector_embeddings)
```

```bash
python single_sentence_transformers.py
```

```bash
[array([-9.01363976e-03, -7.26634488e-02, -1.73819009e-02, -1.94310341e-02,
       -6.62666187e-02,  9.48029291e-03,  7.01074228e-02, -9.20854602e-03,
        4.02065851e-02, -1.14273168e-01, -6.84478432e-02,  6.08052239e-02,
       -1.05305864e-02, -4.59766723e-02, -1.08843155e-01, -8.85540470e-02,
        9.47617088e-03, -3.43578868e-02,  4.34789918e-02, -3.33821564e-03,
       -2.59357858e-02,  8.54353309e-02, -9.31281447e-02,  5.08734817e-03,
        2.29668748e-02, -1.04451500e-01,  2.15865043e-03, -5.77832479e-03,
       -2.07088925e-02, -1.22487182e-02,  2.35777535e-02,  5.92433400e-02,
       -7.44667724e-02, -5.76894265e-03, -5.05806766e-02, -4.60977666e-02,
        6.09780289e-02,  1.71905551e-02,  2.19825115e-02, -2.99102888e-02,
        3.52623835e-02, -1.01641156e-01,  6.99767917e-02,  1.78133119e-02,
        3.48209552e-02,  5.53497076e-02,  1.61906853e-02,  2.77546663e-02,
       -1.08747398e-02,  7.90182278e-02,  7.05966726e-02,  9.42945108e-03,
        6.12758566e-03,  9.26371813e-02, -4.21451181e-02,  1.51264959e-03,
        2.52709091e-02,  1.93259902e-02,  3.33795761e-05,  1.86467916e-02,
        1.50404628e-02,  5.73550053e-02,  6.84724376e-02, -5.03877401e-02,
        5.24056852e-02,  4.98221852e-02, -9.57602821e-03, -5.62821608e-03,
       -8.93809460e-03,  8.19092058e-03, -1.33250095e-02, -4.78314348e-02,
       -2.89507750e-02,  2.22730767e-02, -2.29331590e-02,  4.32070121e-02,
       -2.11444236e-02,  4.87226211e-02, -1.40435679e-03,  4.06864472e-02,
        8.03506281e-03, -7.95856118e-02, -7.87754357e-03, -4.89976583e-03,
       -9.27677285e-03,  2.28112880e-02,  1.43008525e-04,  2.43994314e-03,
       -4.25456353e-02, -6.06843550e-03,  7.35170990e-02,  9.02277604e-02,
        9.69740227e-02, -6.07660692e-03, -5.91304339e-02,  1.93760414e-02,
       -4.18847762e-02,  2.04413794e-02,  2.30090786e-02,  3.00601739e-02,
        1.53840501e-02,  1.08243063e-01,  2.53835432e-02,  7.99183100e-02,
        3.08602527e-02,  5.56972772e-02,  6.33004382e-02, -4.32077162e-02,
        9.25354380e-03, -7.42618814e-02,  1.95277352e-02, -2.09556967e-02,
        7.11094402e-03, -3.11500076e-02, -3.99018116e-02, -5.31260855e-02,
        8.37208237e-03, -1.76881887e-02,  4.21413332e-02, -2.33083982e-02,
       -9.87822562e-03,  3.05469725e-02, -2.34731380e-02, -3.15721370e-02,
       -2.20579542e-02,  4.88705412e-02, -1.00919204e-02, -8.24773783e-33,
       -3.20992395e-02, -1.57441816e-03,  7.11625740e-02, -4.85454127e-02,
        3.97037417e-02, -6.04918860e-02, -4.52655032e-02,  2.53920406e-02,
        3.27236354e-02,  6.00252934e-02, -8.85453895e-02, -6.22648969e-02,
        9.03632343e-02, -8.46683700e-03,  5.55316824e-03,  4.11313735e-02,
       -1.74637232e-02, -6.06574751e-02,  4.94728237e-02,  3.98653038e-02,
       -3.46470857e-03, -4.00375873e-02, -3.61267813e-02, -2.13603228e-02,
        1.28533810e-01,  7.18061700e-02, -4.24967632e-02, -1.17048537e-02,
        8.01106766e-02, -1.28050835e-03, -2.60691326e-02, -1.98353622e-02,
        1.86456442e-02,  9.15479753e-03, -4.18176576e-02, -5.06665148e-02,
       -1.00052580e-02, -1.96289048e-02,  1.48976874e-02, -2.73947548e-02,
        4.13412182e-03,  1.47435181e-02, -3.39842401e-02,  6.34888634e-02,
        7.19166733e-03,  8.90288427e-02,  9.23396740e-03,  7.65621215e-02,
       -4.75086411e-03,  4.42864448e-02, -4.13844511e-02,  3.38682928e-03,
       -8.83377790e-02, -8.09354633e-02,  1.00699384e-02, -9.29891393e-02,
        3.20385024e-02,  7.15039521e-02, -1.51125863e-02, -7.09700733e-02,
       -2.40881424e-02, -1.99128110e-02, -4.35434580e-02,  3.43998312e-04,
        8.90223980e-02, -1.22431275e-02, -3.60050611e-02,  2.23577563e-02,
        1.98856257e-02, -2.75509357e-02, -6.00061119e-02,  5.66775096e-04,
       -4.25546104e-03,  4.57880609e-02,  1.24186566e-02, -1.62145738e-02,
       -5.71585447e-02, -4.69563417e-02,  1.08305877e-03,  4.55328040e-02,
       -1.49292974e-02,  3.69379520e-02, -3.92488353e-02,  1.10600002e-01,
        8.57273489e-02,  8.37553367e-02, -6.28331024e-03, -7.00851008e-02,
       -5.61820483e-03, -1.29999444e-01,  7.13336542e-02,  1.90303493e-02,
        2.09273249e-02,  5.85933216e-02,  2.55958345e-02,  4.85700895e-33,
        1.17489081e-02, -2.97200661e-02, -2.86355745e-02,  5.12178093e-02,
       -7.61526003e-02, -6.91270083e-02,  6.01409227e-02, -1.99256511e-03,
        2.77269594e-02,  7.75412843e-02,  3.07008550e-02,  4.35537957e-02,
       -4.42620777e-02, -1.79267302e-02,  1.11253604e-01, -5.00905178e-02,
        4.27784882e-02,  1.23660015e-02, -4.01308686e-02, -1.03339506e-02,
        6.50060922e-02,  4.37213928e-02,  6.39234185e-02,  1.32256579e-02,
        6.41422812e-03, -1.04980264e-02,  4.70189117e-02, -3.03180963e-02,
       -1.49100786e-02,  2.68476233e-02, -5.30382842e-02, -3.56379561e-02,
        1.05217516e-01, -1.86393466e-02, -4.95999679e-02, -4.88589555e-02,
        8.52462500e-02,  4.88146059e-02,  1.06171770e-02,  5.19691817e-02,
       -4.59886193e-02,  6.28981460e-03,  1.08347973e-02, -2.06284728e-02,
        1.17662037e-02,  9.52917524e-03,  4.63899933e-02,  3.89787816e-02,
       -3.29159349e-02, -4.61041406e-02,  6.79487810e-02,  4.20599915e-02,
       -1.76527519e-02,  6.32125363e-02,  3.66008542e-02,  1.18185014e-01,
        6.79515898e-02, -5.41924790e-04,  7.02309087e-02, -8.81321914e-03,
        1.34443799e-02,  4.01563831e-02,  7.82888010e-02,  1.87332444e-02,
        1.63495373e-02, -1.94461215e-02, -1.05943546e-01, -6.83335364e-02,
       -2.74906196e-02, -5.96801005e-02, -1.32772364e-02, -4.80948761e-02,
        5.66633977e-03,  1.02899581e-01, -2.31031496e-02, -8.01548511e-02,
        5.10434546e-02, -6.72812399e-04, -3.42263579e-02,  4.07001562e-02,
       -1.33215254e-02,  7.84305297e-03,  2.86809057e-02,  4.60833907e-02,
       -2.07075309e-02, -5.63101396e-02,  4.66892086e-02,  1.34800682e-02,
        1.60413869e-02, -5.18168584e-02, -7.12314434e-03, -1.28765972e-02,
       -1.74010154e-02, -4.15140316e-02, -1.30493641e-01, -1.59426818e-08,
       -9.90548544e-03, -9.77642536e-02, -4.48556244e-02, -6.49941415e-02,
        1.89965125e-02,  3.85337844e-02, -4.46805544e-02,  1.54773742e-02,
       -5.45423888e-02, -2.07887124e-02,  3.45422626e-02,  4.88450341e-02,
       -1.26996070e-01,  4.73383814e-02, -2.72291210e-02, -2.01443955e-02,
        7.93490931e-02, -1.55788198e-01, -5.54018728e-02,  7.49701187e-02,
       -7.76811987e-02, -2.33475287e-02,  1.51783125e-02,  1.44298291e-02,
       -1.50949201e-02, -6.74839839e-02,  5.39948605e-02,  7.90254921e-02,
        1.15107588e-01, -1.16381720e-01, -3.58741656e-02,  5.14598563e-02,
        3.20571917e-03, -6.92821667e-02, -6.87018782e-02,  8.13369676e-02,
        6.05906434e-02,  4.40995917e-02,  4.41604219e-02, -1.78186838e-02,
       -5.11434712e-02, -7.16138929e-02, -2.52058581e-02, -2.79534794e-02,
        1.00278057e-01, -6.76283091e-02, -3.59125920e-02, -4.82271314e-02,
        2.85924673e-02, -9.24515128e-02, -4.24716473e-02, -3.74520570e-02,
        6.77164495e-02,  3.65544148e-02,  9.96216107e-03, -1.42479122e-01,
       -9.75322053e-02, -2.36777198e-02,  1.51587119e-02,  1.12204030e-02,
        7.41675943e-02,  5.11700958e-02,  3.52188647e-02, -1.16323508e-01],
      dtype=float32)]
```

##### 6.4.1、精确的 KNN

```bash
> curl http://localhost:9200/neural_index/_search -XPOST -H 'Content-Type: application/json' -d '{
"query": {
    "script_score": {
        "query" : {
            "match_all": {}
        },
        "script": {
            "source": "cosineSimilarity(params.queryVector, '\''general_text_vector'\'') + 1.0",
            "params": {
                "queryVector": [-9.01364535e-03, -7.26634488e-02, ..., -1.16323479e-01]
            }
        }
    }
}}
'
```


响应

```bash
{
  "took": 349,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 10000,
      "relation": "eq"
    },
    "max_score": 1.3824182,
    "hits": [
      {
        "_index": "neural_index",
        "_id": "7686",
        "_score": 1.3824182,
        "_source": {
          "general_text": "A. A federal tax identification number (also known as an employer identification number or EIN), is a number assigned solely to your business by the IRS. Your tax ID number is used to identify your business to several federal agencies responsible for the regulation of business.\n",
          "general_text_vector": [
            -0.054000974,
            0.045891445,
            ...
            -0.038448807
          ],
          "color": "white"
        }
      },
      {
        "_index": "neural_index",
        "_id": "7691",
        "_score": 1.3680091,
        "_source": {
          "general_text": "A. A federal tax identification number (also known as an employer identification number or EIN), is a number assigned solely to your business by the IRS.\n",
          "general_text_vector": [
            -0.050752804,
            0.046932265,
            ...
            -0.02534396
          ],
          "color": "white"
        }
      },
      {
        "_index": "neural_index",
        "_id": "7692",
        "_score": 1.3574383,
        "_source": {
          "general_text": "Letâ��s start at the beginning. A tax ID number or employer identification number (EIN) is a number you get from the U.S. federal government that gives an identification number to a business, much like a social security number does for a person.\n",
          "general_text_vector": [
            -0.072254635,
            0.05448939,
            ...
            -0.0120156575
          ],
          "color": "white"
        }
      },
      {
        "_index": "neural_index",
        "_id": "4207",
        "_score": 1.350525,
        "_source": {
          "general_text": "A Vehicle Identification Number (VIN) is a set of unique digits that relate to specific information about your vehicle (1981 or newer).\n",
          "general_text_vector": [
            -0.07536415,
            0.045979388,
            ...
            -0.12938495
          ],
          "color": "white"
        }
      },
      {
        "_index": "neural_index",
        "_id": "7685",
        "_score": 1.3432959,
        "_source": {
          "general_text": "A tax ID number or employer identification number (EIN) is a number you get from the U.S. federal government that gives an identification number to a business, much like a social security number does for a person.\n",
          "general_text_vector": [
            -0.0709587,
            0.06297276,
            ...
            -0.011866731
          ],
          "color": "white"
        }
      },
      {
        "_index": "neural_index",
        "_id": "5877",
        "_score": 1.342576,
        "_source": {
          "general_text": "Greater Hudson Bank, National Association: Bardonia Branch at 715 Route 304, branch established on 2008/10/01. Info updated 2009/01/06: Bank assets: $299.2 mil, Deposits: $245.8 mil, headquarters in Middletown, NY, positive income, Commercial Lending Specialization, 4 total offices.\n",
          "general_text_vector": [
            0.11568897,
            -0.116952114,
            ...
            -0.13541293
          ],
          "color": "black"
        }
      },
      {
        "_index": "neural_index",
        "_id": "7684",
        "_score": 1.3402317,
        "_source": {
          "general_text": "An employer identification number (EIN), also called a tax ID number or taxpayer ID, is required for most business entities. As its name implies, this is the number used by the Internal Revenue Service (IRS) to identify businesses with respect to their tax obligations.\n",
          "general_text_vector": [
            -0.061480425,
            0.045844033,
            ...
            -0.035914235
          ],
          "color": "white"
        }
      },
      {
        "_index": "neural_index",
        "_id": "7690",
        "_score": 1.3331752,
        "_source": {
          "general_text": "Download article as a PDF. An employer identification number (EIN), also called a tax ID number or taxpayer ID, is required for most business entities. As its name implies, this is the number used by the Internal Revenue Service (IRS) to identify businesses with respect to their tax obligations.\n",
          "general_text_vector": [
            -0.065706804,
            0.052238673,
            ...
            -0.035678774
          ],
          "color": "green"
        }
      },
      {
        "_index": "neural_index",
        "_id": "4210",
        "_score": 1.3319674,
        "_source": {
          "general_text": "VIN is recorded in Vehicle License of China. A vehicle identification number, commonly abbreviated to VIN, or chassis number, is a unique code including a serial number, used by the automotive industry to identify individual motor vehicles, towed vehicles, motorcycles, scooters and mopeds as defined in ISO 3833. VINs were first used in 1954.\n",
          "general_text_vector": [
            -0.117122486,
            0.05945877,
            ...
            -0.10524516
          ],
          "color": "black"
        }
      },
      {
        "_index": "neural_index",
        "_id": "4206",
        "_score": 1.3285326,
        "_source": {
          "general_text": "All vehicles registered in the UK must have a unique, stamped-in vehicle identification number (.\n",
          "general_text_vector": [
            -0.011287496,
            -0.06897103,
            ...
            -0.022097774
          ],
          "color": "red"
        }
      }
    ]
  }
}
```
在这种情况下，我们使用了 match_all 查询来匹配所有文档，但是除非你处理的是非常小的索引，否则这个查询并不具备可扩展性，可能会显著增加搜索延迟。

如果你想在大型数据集上使用此查询，建议在 script_score 中指定一个过滤查询，以限制传递给向量函数的匹配文档数量。

##### 6.4.2、近似的 KNN

为什么是近似呢？因为 Elasticsearch 使用了一种近似方法来执行 kNN 搜索（即 HNSW），它为了提高搜索速度和减少计算复杂性（尤其是在大型数据集上），牺牲了结果的准确性；因此，搜索结果可能并不总是真正的 k 个最近邻。

```bash
> curl http://localhost:9200/neural_index/_search -XPOST -H 'Content-Type: application/json' -d '{
"knn": {
    "field": "general_text_vector",
    "query_vector": [-9.01363976e-03, ......, -1.73819009e-02],
    "k": 3,
    "num_candidates": 10
},
"_source": [
    "general_text",
    "color"
]}
'
```

通过在请求体中添加 knn 选项来使用。knn 对象具有以下定义属性：

- **field：**（字符串）存储向量嵌入的字段
- **query_vector：**（浮点数数组）表示查询的方括号内的浮点数列表；必须具有与向量字段相同的维度（即 384）
- **k：**（整数）要检索的最近邻数量；必须小于 num_candidates
- **num_candidates：**（整数）在每个分片中要考虑的近似最近邻候选数量（<= 10000）；增加此数字可以提高准确性，但会降低搜索速度

响应

```bash
{
  "took": 44,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 3,
      "relation": "eq"
    },
    "max_score": 0.69120896,
    "hits": [
      {
        "_index": "neural_index",
        "_id": "7686",
        "_score": 0.69120896,
        "_source": {
          "color": "white",
          "general_text": "A. A federal tax identification number (also known as an employer identification number or EIN), is a number assigned solely to your business by the IRS. Your tax ID number is used to identify your business to several federal agencies responsible for the regulation of business.\n"
        }
      },
      {
        "_index": "neural_index",
        "_id": "7691",
        "_score": 0.6840044,
        "_source": {
          "color": "white",
          "general_text": "A. A federal tax identification number (also known as an employer identification number or EIN), is a number assigned solely to your business by the IRS.\n"
        }
      },
      {
        "_index": "neural_index",
        "_id": "7692",
        "_score": 0.6787192,
        "_source": {
          "color": "white",
          "general_text": "Letâ��s start at the beginning. A tax ID number or employer identification number (EIN) is a number you get from the U.S. federal government that gives an identification number to a business, much like a social security number does for a person.\n"
        }
      }
    ]
  }
}
```

设置 topK=3 后，我们获取了对于查询 “what is a bank transit number” 最佳的三个文档。

搜索会计算每个分片中 num_candidates 个向量与查询向量的相似度（确定文档的 _score），从每个分片中选择 k 个最相似的结果，然后合并结果（来自每个分片），返回全局最相似的 k 个邻近文档。

##### 6.4.3、近似 KNN + 预过滤

从 Elasticsearch 8.2 版本开始，支持预过滤功能。

下面的请求执行了一个带有 color 字段过滤器的近似 kNN 搜索：

```bash
> curl http://localhost:9200/neural_index/_search -XPOST -H 'Content-Type: application/json' -d '{
"knn": {
    "field": "general_text_vector",
    "query_vector": [-9.01363976e-03, ......, -1.73819009e-02],
    "k": 3,
    "num_candidates": 10,
    "filter": {
        "term": {
            "color": "white"
        }
    }
},
"fields": ["color"],
"_source": false
}'
```

设置 topK=3 后，我们获取了查询 “what is a bank transit number” 和 color 为 “white” 的最佳三个文档。

这个查询确保返回 k 个匹配的文档，因为过滤器查询是在近似 kNN 搜索期间应用的，而不是之后应用的。

##### 6.4.4、近似 KNN + 其他功能

从 8.4 版本开始，还可以执行混合搜索。在此请求中，我们通过一个 OR 条件将 knn 选项和查询组合起来。

```bash
> curl http://localhost:9200/neural_index/_search -XPOST -H 'Content-Type: application/json' -d '{
"query": {
    "match": {
        "general_text": {
            "query": "federal"
        }
    }
},
"knn": {
    "field": "general_text_vector",
    "query_vector": [-9.01364535e-03, -7.26634488e-02, ..., -1.16323479e-01],
    "k": 3,
    "num_candidates": 10
},
"size": 5,
"_source": [
    "general_text"
]
}'
```

这个搜索操作：

- 从每个分片中收集 knn num_candidates（10）个结果；
- 找到全局的前 k=3 个向量匹配；
- 将它们与 match 查询（=federal）的匹配结果合并；
- 最终返回得分排名前 5 个（size=5）结果（即使总共有 143 个命中）。

得分是通过将 knn 分数和查询分数相加来计算的，还可以指定 boost 值以赋予每个分数不同的权重。

knn 选项还可以与聚合一起使用，即在前 k 个最近的文档上计算聚合，而不是在所有匹配搜索的文档上进行聚合。

