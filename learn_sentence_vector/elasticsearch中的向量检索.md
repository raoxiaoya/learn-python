elasticsearch中的向量检索，语义检索，RRF，kNN，ANN，HNSW



### kNN 算法 （k-Nearest Neighbor）

KNN可以说是最简单的分类算法之一，同时，它也是最常用的分类算法之一。

kNN算法的核心思想是如果一个样本在特征空间中的k个最相邻的样本中的大多数属于某一个类别，则该样本也属于这个类别，并具有这个类别上样本的特性。该方法在确定分类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。 kNN方法在类别决策时，只与极少量的相邻样本有关。由于kNN方法主要靠周围有限的邻近的样本，而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，kNN方法较其他方法更为适合。

KNN的原理就是当预测一个新的值x的类别时候，根据它距离最近的K个点是什么类别来判断x属于哪个类别。

![image-20241022104450267](D:\dev\php\magook\trunk\server\md\img\image-20241022104450267.png)

图中绿色的点就是我们要预测的那个点，假设K=3。那么KNN算法就会找到与它距离最近的三个点（这里用圆圈把它圈起来了），看看哪种类别多一些，比如这个例子中是蓝色三角形多一些，新来的绿色点就归类到蓝三角了。

![image-20241022104528014](D:\dev\php\magook\trunk\server\md\img\image-20241022104528014.png)

但是，当K=5的时候，判定就变成不一样了。这次变成红圆多一些，所以新来的绿点被归类成红圆。从这个例子中，我们就能看得出K的取值是很重要的。

另外一个就是距离计算，我们就使用最常用的欧式距离。所以，kNN算法需要计算x与空间中所有点的距离，然后排序，取得距离最小的k个，然后再看这k个点中哪种类别最多。

回到ES检索中来，使用kNN算法得到了k条文档，其意思就是说这k条文档就能概括你的问题query。

k的取值需要经过实验来获得，随着k值的增加，其错误率的曲线大致如下。

![image-20241022105459024](D:\dev\php\magook\trunk\server\md\img\image-20241022105459024.png)

**KNN算法优点**

1. 简单易用，相比其他算法，KNN算是比较简洁明了的算法。即使没有很高的数学基础也能搞清楚它的原理。
2. 模型训练时间快，上面说到KNN算法是惰性的，这里也就不再过多讲述。
3. 预测效果好。
4. 对异常值不敏感

**KNN算法缺点**

1. 对内存要求较高，因为该算法存储了所有训练数据
2. 预测阶段可能很慢
3. 对不相关的功能和数据规模敏感

至于什么时候应该选择使用KNN算法，sklearn的这张图给了我们一个答案。

![image-20241022105928351](D:\dev\php\magook\trunk\server\md\img\image-20241022105928351.png)

简单得说，当需要使用分类算法，且数据比较大的时候就可以尝试使用KNN算法进行分类了。



### ES中的检索器retriever

检索器retriever概述：https://www.elastic.co/guide/en/elasticsearch/reference/current/retrievers-overview.html



The following retrievers are available:

- [**Standard Retriever**](https://www.elastic.co/guide/en/elasticsearch/reference/current/retriever.html#standard-retriever). Returns top documents from a traditional [query](https://www.elastic.co/guide/en/elasticsearch/reference/master/query-dsl.html). Mimics a traditional query but in the context of a retriever framework. This ensures backward compatibility as existing `_search` requests remain supported. That way you can transition to the new abstraction at your own pace without mixing syntaxes.

- [**kNN Retriever**](https://www.elastic.co/guide/en/elasticsearch/reference/current/retriever.html#knn-retriever). Returns top documents from a [knn search](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-search.html#search-api-knn), in the context of a retriever framework.

- [**RRF Retriever**](https://www.elastic.co/guide/en/elasticsearch/reference/current/retriever.html#rrf-retriever). Combines and ranks multiple first-stage retrievers using the reciprocal rank fusion (RRF) algorithm. Allows you to combine multiple result sets with different relevance indicators into a single result set. An RRF retriever is a **compound retriever**, where its `filter` element is propagated to its sub retrievers.

  Sub retrievers may not use elements that are restricted by having a compound retriever as part of the retriever tree. See the [RRF documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html#rrf-using-multiple-standard-retrievers) for detailed examples and information on how to use the RRF retriever.

- [**Text Similarity Re-ranker Retriever**](https://www.elastic.co/guide/en/elasticsearch/reference/current/retriever.html#text-similarity-reranker-retriever). Used for [semantic reranking](https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-reranking.html). Requires first creating a `rerank` task using the [Elasticsearch Inference API](https://www.elastic.co/guide/en/elasticsearch/reference/current/put-inference-api.html).



检索器retriever具体使用：https://www.elastic.co/guide/en/elasticsearch/reference/current/retriever.html



ES比单纯的向量数据库的优势在于，它支持混合检索，即一个搜索语句即能包含向量字段，也能包含常规的字段。



### ES中的kNN Search

在 ES8.4及之前，kNN search api 是 `_knn_search`，文档地址：https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search-api.html。

```bash
GET my-index/_knn_search
{
  "knn": {
    "field": "image_vector",
    "query_vector": [0.3, 0.1, 1.2],
    "k": 10,
    "num_candidates": 100
  },
  "_source": ["name", "file_type"]
}
```



很明显这个接口只能服务与向量检索，但是现实中更多的场景式混合检索，即一条语句同时包含向量字段也包含常规字段，所以从 8.5 版本之后此 api 被废弃，改成了混合搜索 api，更加强调检索器retriever，所以现在进行kNN搜索的api为`_search`，详细介绍：https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html

```bash
POST image-index/_search
{
  "knn": {
    "field": "image-vector",
    "query_vector": [-5, 9, -12],
    "k": 10,
    "num_candidates": 100
  },
  "fields": [ "title", "file-type" ]
}
```

混合搜索的例子

```bash
POST image-index/_search
{
  "query": {
    "match": {
      "title": {
        "query": "mountain lake",
        "boost": 0.9
      }
    }
  },
  "knn": {
    "field": "image-vector",
    "query_vector": [54, 10, -2],
    "k": 5,
    "num_candidates": 50,
    "boost": 0.1
  },
  "size": 10
}
```

ES提供了两种调用kNN的方法：

- [Approximate kNN](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#approximate-knn) ，近似 kNN，using the `knn` search option or `knn` query
- [Exact, brute-force kNN](https://www.elastic.co/guide/en/elasticsearch/reference/current/knn-search.html#exact-knn) ，精确 kNN，using a `script_score` query with a vector function

由于kNN算法需要逐个计算距离，这显然不太现实，于是实际都是使用 Approximate kNN，即aNN，近似最近邻。可提供较低的延迟，但代价是索引速度较慢且准确度不高。

![image-20241022213335030](D:\dev\php\magook\trunk\server\md\img\image-20241022213335030.png)



精确的强力 kNN 可保证结果准确，但对于大型数据集来说扩展性不佳。



### 多路融合排序 RRF（Reciprocal Rank Fusion）

文档：https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html

参考：https://blog.csdn.net/UbuntuTouch/article/details/131200354

就是一个综合排序算法，正常情况下，你需要给每一个字段设置权重，然后根据权重来计算综合排名，但是如果你不想设置权重，还想有一个排序，那么RRF就是一个凑合的算法，比如字段A按照一个规则排序，字段B按照另一个规则排序，那么每一条记录的综合得分就是这两个字段的排名的倒数相加。即`score = 1 / f1_rank + 1 / f2_rank`。然后再按照 `score` 排序。



### 语义检索

文档：https://www.elastic.co/guide/en/elasticsearch/reference/current/semantic-search-semantic-text.html

我们先要搞清楚相似度和相关性是两个概念，相似度就是纯粹的向量空间的计算，纯数学问题；而相关性则是指内在的语义。比如：湖北工程大学 & 湖北医科大学，这两句话相似度很高，但是相关性很低。

在具体实践上，kNN检索只需要提供算法即可，而语义检索则需要借助于训练好的LLM来完成，因为只有LLM才能理解句子的意思。



### HNSW

虽然ES使用的是kNN（ANN），然而，在高维空间中，过去的实验表明ANN并不比KNN节省多少时间，但是至少要有一个类似于Mysql B+树那样的索引结构以加速检索。ES采用的是HNSW（分层导航小世界算法），HNSW是有能力在几毫秒内从数百万个数据点中找到最近邻的。



下文转自：[像光速一样搜索——HNSW算法介绍 ](https://www.luxiangdong.com/2023/11/06/hnsw/)



**最先进的快速搜索算法**

这些不同的人工神经网络算法是不同的方法来形成**数据结构**，以实现有效的检索。有三种类型的算法：基于图的、基于哈希的和基于树的。

**基于图的算法创建数据的图**表示，其中每个数据点是一个节点，边表示数据点之间的接近性或相似性。最引人注目的是层次导航小世界图(HNSW)。

**基于哈希的算法**使用哈希函数将数据点映射到哈希码或桶。流行的算法包括:位置敏感哈希（LSH）、多索引哈希（MIH）和产品量化

**基于树的算法**将数据集划分为树状结构，以便快速搜索。流行的是kd树、球树和随机投影树（RP树）。对于低维空间（≤10），基于树的算法是非常有效的。

有几个流行的代码库:

1. **Scikit-learn**：它的`NearestNeighbors`类提供了一个简单的接口，可以使用LSH等技术进行精确和近似的最近邻搜索。
2. **Hnswlib**：它是HNSW的Python包装器。
3. **FAISS**：该库支持多种ANN算法，包括HNSW, IVFADC(带ADC量化的倒置文件)和IVFPQ(带产品量化的倒置文件)。
4. **Annoy** (Approximate Nearest Neighbors Oh Yeah)：Annoy是一个c++库，也提供了Python接口。
5. **NMSLIB**(非度量空间库)：它是用c++编写的，并具有Python包装器。它可以执行HNSW、LSH、MIH或随机投影树等算法。

使用上述代码库，您可以超级快速地执行搜索查询。您还需要了解其他库的变体。这里我只提到其中的三个。第一个是[PyNNDescent](https://github.com/lmcinnes/pynndescent)。PyNNDescent是一个Python库，用于基于NN-descent的搜索算法，它是LSH的一个变体。第二个是[NearPy](http://pixelogik.github.io/NearPy/)。它支持多个距离度量和哈希族。第三个是[PyKDTree](https://github.com/storpipfugl/pykdtree)。PyKDTree是一个Python库，用于基于kd树的最近邻（KNN）搜索。虽然kd树更常用于精确搜索，但PyKDTree也可以通过一些启发式优化用于近似搜索。

此外，如果您询问哪些算法和库执行速度最好，您只需要了解[**ANN- benchmark **](https://github.com/erikbern/ann-benchmarks)库，专门为对人工神经网络搜索算法进行基准测试而设计。它提供了一个标准化的框架来评估算法，如[Annoy](https://github.com/spotify/annoy)， [FLANN](http://www.cs.ubc.ca/research/flann/)， [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html) (LSHForest, KDTree, BallTree)， [PANNS](https://github.com/ryanrhymes/panns)， [NearPy](http://pixelogik.github.io/NearPy/)， [KGraph](https://github.com/aaalgo/kgraph)， [NMSLIB(非度量空间库)](https://github.com/nmslib/nmslib)，[hnswlib](https://github.com/nmslib/hnsw)， [RPForest](https://github.com/lyst/rpforest)， [FAISS](https://github.com/facebookresearch/faiss)， [nndescent](https://github.com/brj0/nndescent)， [PyNNDescent](https://github.com/lmcinnes/pynndescent)等等。



HNSW是一种用于在高维空间中进行高效人工神经网络搜索的数据结构和算法。它是**跳表**和**小世界图（SWG）**结构的扩展，可以有效地找到近似的最近邻。如果我们先学习跳表和小世界图，学习HNSW就会很简单。

跳表是一种数据结构，用于维护一组已排序的元素，并允许进行高效的搜索、插入和删除操作。

下图显示了数字[3、6、7、9、12、17、19、21、25、26]的排序链表。假设我们想找到目标19。当值小于目标时，我们向右移动。需要6步才能找到它。

![image-20241023113205305](D:\dev\php\magook\trunk\server\md\img\image-20241023113205305.png)

现在，如果列表的每个其他节点都有一个指向前面节点2的指针，如下图所示，可以将这些指针视为“高速公路”。数学规则是“当数值小于目标时向右移动”。需要4个步骤才能达到19。

![image-20241023113252905](D:\dev\php\magook\trunk\server\md\img\image-20241023113252905.png)

这些高速公路加快了搜索速度。我们可以增加更多。现在，如果列表中每三个其他节点都有一个指向前面第三个节点的指针，如下图所示，那么只需要3步就可以到达19。

你可能会问，如何选择这些点作为”高速公路“？它们可以是预先确定的或随机选择的。这些节点的随机选择是Small World和NHSW中数据构建的重要步骤，我将在后面介绍。

![image-20241023113341865](D:\dev\php\magook\trunk\server\md\img\image-20241023113341865.png)



#### 由跳表的思路延伸到Small World

小世界（small world）网络是一种特殊的网络，在这种网络中，你可以快速地联系到网络中的其他人或点。这有点像“凯文·培根的六度”(Six Degrees of Kevin Bacon)游戏，在这个游戏中，你可以通过一系列其他演员，在不到六个步骤的时间里，将任何演员与凯文·培根联系起来。

想象一下，你有一群朋友排成一个圆圈，如图5所示。每个朋友都与坐在他们旁边的人直接相连。我们称它为“原始圆”。

现在，这就是奇迹发生的地方。你可以随机选择将其中一些连接改变给圆圈中的其他人，就像图5中的红色连接线一样。这就像这些连接的“抢椅子”游戏。有人跳到另一把椅子上的几率用概率p表示。如果p很小，移动的人就不多，网络看起来就很像原来的圆圈。但如果p很大，很多人就会跳来跳去，网络就会变得有点混乱。当您选择正确的p值(不太小也不太大)时，红色连接是最优的。网络变成了一个小世界网络。你可以很快地从一个朋友转到另一个朋友(这就是“小世界”的特点)。

![image-20241023113637773](D:\dev\php\magook\trunk\server\md\img\image-20241023113637773.png)



#### 从小世界到HNSW

现在我们要扩展到高维空间。图中的每个节点都是一个高维向量。在高维空间中，搜索速度会变慢。这是不可避免的“维度的诅咒”。HNSW是一种高级数据结构，用于优化高维空间中的相似性搜索。

让我们看看HNSW如何构建图的层次结构。HNSW从图(6)中的第0层这样的基础图开始。它通常使用随机初始化数据点来构建。

![image-20241023113803599](D:\dev\php\magook\trunk\server\md\img\image-20241023113803599.png)

HNSW在层次结构中的基础层之上构造附加层。每个层将有更少的顶点和边的数量。可以把高层中的顶点看作是跳跃列表中访问“高速公路”的点。你也可以将这些顶点视为游戏“Six Degrees of Kevin Bacon”中的演员Kevin Bacon，其他顶点可以在不到6步的时间内连接到他。

一旦构建了上面的层次结构，数据点就被编入索引，并准备进行查询搜索。假设查询点是桃色数据点。为了找到一个近似最近的邻居，HNSW从入门级(第2层)开始，并通过层次结构向下遍历以找到最近的顶点。在遍历的每一步，算法检查从查询点到当前节点邻居的距离，然后选择距离最小的相邻节点作为下一个基本节点。查询点到最近邻居之间的距离是常用的度量，如欧几里得距离或余弦相似度。当满足某个停止条件(例如距离计算次数)时，搜索终止。



#### HNSW如何构建数据结构?

HNSW首先初始化一个空图作为数据结构的基础。该图表示一个接一个插入数据点的空间。HNSW将数据点组织成多层。每一层表示数据结构中不同级别的粒度。层数是预定义的，通常取决于数据的特征。

每个数据点随机分配到一个层。最高的一层用于最粗略的表示，随着层的向下移动，表示变得更精细。这个任务是用一个特定的概率分布来完成的，这个概率分布叫做指数衰减概率分布。这种分布使得数据点到达更高层的可能性大大降低。如果你还记得跳跃列表中随机选择的数据点作为“高速公路”，这里的一些数据点是随机选择到最高层的。在后面的代码示例中，我们将看到每层中的数据点数量，并且数据点的数量在更高层中呈指数级减少。

为了在每一层内有效地构建连接，HNSW使用贪婪搜索算法。它从顶层开始，试图将每个数据点连接到同一层内最近的邻居。一旦建立了一层中的连接，HNSW将使用连接点作为搜索的起点继续向下扩展到下一层。构建过程一直持续到处理完所有层，并且完全构建了数据结构。

让我们简单总结一下HNSW中数据结构的构造。让我也参考Malkov和Yashunin[3]中的符号，并在附录中解释HNSW算法。您可能会发现它们有助于更明确地理解HNSW的算法。HNSW声明一个空结构并逐个插入数据元素。它保持每个数据点每层最多有*M*个连接的属性，并且每个数据点的连接总数不超过最大值(*Mmax*)。在每一层中，HNSW找到与新数据点最近的K个邻居。然后，它根据距离更新候选数据点集和找到的最近邻居列表(*W*)。如果*W*中的数据点数量超过了动态候选列表(*ef*)的大小，则该函数从*W*中删除最远的数据点。



![image-20241022213902677](D:\dev\php\magook\trunk\server\md\img\image-20241022213902677.png)



![image-20241022213921052](D:\dev\php\magook\trunk\server\md\img\image-20241022213921052.png)



![image-20241022214233583](D:\dev\php\magook\trunk\server\md\img\image-20241022214233583.png)



![image-20241022214349550](D:\dev\php\magook\trunk\server\md\img\image-20241022214349550.png)



![image-20241022214426677](D:\dev\php\magook\trunk\server\md\img\image-20241022214426677.png)



#### 代码示例

接下来，让我们使用库FAISS执行HNSW搜索。我将使用NLP中包含新闻文章的流行数据集。然后，我使用“SentenceTransformer”执行Embeddings。然后，我将向您展示如何使用HNSW通过查询搜索类似的文章。

**Data**

总检察长的新闻文章语料库由[A.]Gulli](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)，是一个从2000多个新闻来源收集100多万篇新闻文章的大型网站。Zhang、Zhao和LeCun在论文中构建了一个较小的集合，其中采样了“世界”、“体育”、“商业”和“科学”等新闻文章，并将其用作文本分类基准。这个数据集“ag_news”已经成为一个经常使用的数据集，可以在Kaggle、PyTorch、Huggingface和Tensorflow中使用。让我们下载[数据从Kaggle](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset)。训练样本和测试样本分别有12万篇和7600篇新闻文章。



```python
import pandas as pd
import numpy as np
import faiss
pd.set_option('display.max_colwidth', -1)
path = "/content/gdrive/My Drive/data"
train = pd.read_csv(path + "/gensim/ag_news_train.csv")
print(train.shape)
print(train.columns)
train['Description'][0:5]
```

输出形状为(120000,3)，列为[‘ Class Index ‘， ‘ Title ‘， ‘ Description ‘]。我们对“描述”栏感兴趣。以下是排名前五的记录。

- *路透社——卖空者，华尔街日益减少的\band极端愤世嫉俗者，又看到了绿色*
- *路透——私人投资公司凯雷投资集团(\which)以在国防工业投资时机恰当、偶尔引发争议而闻名，该公司已悄然将赌注押在了市场的另一个领域*
- *路透社——油价飙升，加上对\about经济和盈利前景的担忧，预计将在下周\summer经济低迷的深度\hang拖累股市*
- *路透社——一位石油官员周六表示，在\intelligence显示反叛民兵可能袭击\infrastructure后，当局已经停止了伊拉克南部主要管道\flows的石油出口*
- *法新社——在距离美国总统大选仅剩三个月的时间里，世界油价不断刷新纪录，人们的钱包越来越紧，这给经济带来了新的威胁*

**数据嵌入**

出于说明的目的，我只使用10,000条记录进行Embeddings。

```python
sentences = train['Description'][0:10000]
```

您需要pip安装“sentence_transformers”库。

```python
!pip install sentence_transformers
from sentence_transformers import SentenceTransformer
```

然后让我们使用预训练模型“bert-base-nli-mean-tokens”来声明模型。在[本页](https://www.sbert.net/docs/pretrained_models.html)上有许多预先训练好的模型。

```python
model = SentenceTransformer('bert-base-nli-mean-tokens')
```

然后我们将“句子”编码为“sentence_embeddings”。

```python
sentence_embeddings = model.encode(sentences)
print(sentence_embeddings.shape)
sentence_embeddings[0:5]
```

输出是10,000个列表。每个列表或向量的维数为768。下面是前5个Embeddings的输出。

*array([[-0.26105028, 0.8585296 , 0.03941074, …, 1.0689917 , 1.1770816 , -0.74388623], [-0.2222097 , -0.03594436, 0.5209106 , …, 0.15727971, -0.3867779 , 0.49948674], [-0.3001758 , -0.41582862, 0.86036515, …, -0.6246218 , 0.52692914, -0.36817163], [ 0.3295024 , 0.22334357, 0.30229023, …, -0.41823167, 0.01728885, -0.05920589], [-0.22277102, 0.7840586 , 0.2004052 , …, -0.9121561 , 0.2918987 , -0.12284964]], dtype=float32)*

这有助于保存Embeddings以备将来使用。

```python
with open(path + '/AG_news.npy', 'wb') as file:
    np.save(file, sentence_embeddings)
```

在上面的代码中，我使用了“npy”文件扩展名，这是NumPy数组文件的常规扩展名。下面是加载数据的代码:

```python
with open (path + '/AG_news.npy', 'rb') as f:
    sentence_embeddings = np.load(f, allow_pickle=True)
```

有了这些Embeddings，我们就可以在HNSW数据结构中组织它们了。



**使用FAISS构建NHSW数据结构索引**

您需要像下面这样pip安装FAISS库:

```python
!pip install faiss-cpu --no-cache
```

我将使用HNSWFlat(dim, m)类来构建HNSW。它需要预先确定的参数dim表示向量的维数，m表示数据元素与其他元素连接的边数。

```python
import faiss
m = 32
dim = 768
index = faiss.IndexHNSWFlat(dim, m)
```

如前所述，HNSW指数的创建分为两个不同的阶段。在初始阶段，该算法采用概率分布来预测引入新数据节点的最上层。在接下来的阶段，收集每个数据点的最近邻居，然后用一个表示为m的值进行修剪(在我们的例子中是m=16)。整个过程是迭代的，从插入层开始，一直到底层。

HNSW中有两个重要参数“efConstruction”和“efSearch”。这两个参数控制着索引结构构建的效率和有效性。它们帮助您在HNSW索引结构中的索引构建和最近邻搜索操作的速度和质量之间取得平衡。

1. efConstruction:该参数用于HNSW索引的构建。它控制了构建索引结构的速度和结构质量之间的权衡。” efConstruction “值决定了在构建阶段要考虑多少候选项目。较高的“efConstruction”值将产生更准确的索引结构，但也会使构建过程变慢。
2. efSearch:该参数用于在HNSW索引中查找查询点的最近邻居。“efSearch”值控制搜索速度和搜索质量之间的权衡。较高的“efSearch”值将导致更准确和详尽的搜索，但也会更慢。我们将“efConstruction”和“efSearch”分别设置为40和16:

```python
index.hnsw.efConstruction = 40 
index.hnsw.efSearch = 16  
```

我们已经声明了上面的数据结构。现在我们准备将数据“sentence_embeddings”一个接一个地插入到数据结构中:

```python
index.add(sentence_embeddings)
```

一旦完成，我们可以检查HNSW数据结构中有多少数据元素:

```python
index.ntotal
```

输出为10000。它是“sentence_embeddings”中的数据点数。接下来，HNSW建造了多少层?让我们来检查最大级别:

```python
# the HNSW index starts with no levels
index.hnsw.max_level
```

最高级别为2.0。这意味着有第0层，第1层和第2层。接下来，您可能想知道每层中数据元素的数量。让我们来看看:

```python
levels = faiss.vector_to_array(index.hnsw.levels)
np.bincount(levels)
```

输出为array([0,9713,280,7])。“0”没有意义，你可以忽略它。它说第0层有9713个数据元素，第1层有280个元素，第2层只有7个元素。注意，9713 + 280 + 7 = 10000。您是否发现，较高层的数据元素数量比前几层呈指数级减少?这是因为数据元素的层分配采用指数衰减概率分布。

**FAISS为HNSW搜索示例**

假设我们的搜索查询是“经济繁荣与股市（economic booming and stock market）”。我们希望找到与我们的搜索查询相关的文章。我们将首先嵌入搜索查询:

```python
qry1 = model.encode(["economic booming and stock market"])
```

使用代码index.search()，搜索过程非常简单。这里k是最近邻居的个数。我们将其设置为5以返回5个邻居。index.search()函数返回两个值” d “和” I “。

- “d”:查询向量与k个最近邻居之间的距离列表。默认的距离度量是欧几里得距离。
- “I”:它是索引中k个最近邻居的位置对应的索引列表。这些索引可用于查找数据集中的实际数据点。

```python
%%time
k=5
d, I = index.search(qry1, k)
print(I)
print(d)
```

索引列表的输出是[[1467 4838 4464 7461 8299]]。我们将使用这些索引打印出搜索结果。

注意，我使用“%%time”来度量执行时间。它输出

*CPU时间:user: 5.57 ms, sys: 5µs, total: 5.58 ms

这意味着搜索只需要几毫秒。这确实是令人难以置信的快!

距离输出列表为:[[158.19066 163.69077 164.47517 164.64172 164.64172]]

```python
for i in I[0]:
  print(train['Description'][i])
```

输出：

*‘Rising oil prices are expected to hit China’s growth rate this year.’*

*‘Developing countries are starting to flex their financial muscles and invest overseas.*

*‘The Tehran Stock Exchange has performed magnificently, but the market’s list of risks is outsized.’*

*‘Federal Express raised its earnings forecast, citing strong demand for its international express, ground and less-than-truckload services.’*

*‘Federal Express raised its earnings forecast, citing strong demand for its international express, ground and less-than-truckload services.’* (Our data have duplications)

这些文章都是关于经济和股票市场的新闻。搜索速度以毫秒计非常快。这不仅仅是结果在哪里的问题，而是如何快速得到结果的问题，不是吗?

您可以通过[此链接](https://github.com/dataman-git/codes_for_articles/blob/master/HNSW.ipynb)下载笔记本进行上述搜索。



**总结**

我希望这篇文章能帮助你理解近似近邻(ANN)，以及它是如何提供高效搜索的。这篇文章解释了不同的人工神经网络算法，包括基于图的HNSW，基于哈希的LSH或产品量化，以及基于树的KD-Trees。这篇文章解释了HNSW如何构建其数据结构并逐个插入数据元素。本文演示了如何使用FAISS库构建用于查询搜索的HNSW。在下一篇文章“[搜索像光速- (2)LSH，](https://dataman-ai.medium.com/search-like-light-speed-2-lsh-b66c90349c66?sk=06225de6acda20982f04699b20428dc4)”中，我将讨论基于哈希的算法。



**附录**

在Malkov和Yashunin[3]的论文中，算法1到5伪代码中提供了HNSW方法。伪代码给出了算法的具体定义。我将这些描述添加到伪代码中，因为一些读者可能会发现它们有助于理解HNSW。算法1、算法2和算法3或算法4中的一个用于完成数据结构。一旦数据结构完成，以后的任何查询搜索都只使用算法5。

- 算法1:“INSERT”函数构建数据结构
- 算法2:“SEARCH-LAYER”函数计算KNN并存储邻居
- 算法3:“SEARCH-NEIGHBORS-SIMPLE”是一种选择邻居的简单方法
- 算法4:“SELECT-NEIGHBORS-HEURISTIC”函数是一种更复杂的选择邻居的方法
- 算法5:“KNN-SEARCH”函数进行查询搜索

让我们从算法1开始。

```
Algorithm 1: INSERT()

INSERT(hnsw, q, M, Mmax, efConstruction, mL)
Input: multilayer graph hnsw, new element q, number of established
connections M, maximum number of connections for each element
per layer Mmax, size of the dynamic candidate list efConstruction, nor-
malization factor for level generation mL
Output: update hnsw inserting element q
1 W ← ∅ // list for the currently found nearest elements
2 ep ← get enter point for hnsw
3 L ← level of ep // top layer for hnsw
4 l ← ⌊-ln(unif(0..1))∙mL⌋ // new element’s level
5 for lc ← L … l+1
6 W ← SEARCH-LAYER(q, ep, ef=1, lc)
7 ep ← get the nearest element from W to q
8 for lc ← min(L, l) … 0
9 W ← SEARCH-LAYER(q, ep, efConstruction, lc)
10 neighbors ← SELECT-NEIGHBORS(q, W, M, lc) // alg. 3 or alg. 4
11 add bidirectionall connectionts from neighbors to q at layer lc
12 for each e ∈ neighbors // shrink connections if needed
13 eConn ← neighbourhood(e) at layer lc
14 if │eConn│ > Mmax // shrink connections of e
// if lc = 0 then Mmax = Mmax0
15 eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc)
// alg. 3 or alg. 4
16 set neighbourhood(e) at layer lc to eNewConn
17 ep ← W
18 if l > L
19 set enter point for hnsw to q
```

它在多层图中插入一个新元素q，保持每个元素每层最多有M个连接，并且每个元素的连接总数不超过Mmax的属性。该算法还保证连接元素之间的距离不大于某一最大距离，并且每层的连接数是均衡的。步骤如下:

1. W←∅:初始化一个空列表W来存储当前找到的最近的元素。
2. ep←get enter point for hnsw:获取多层图hnsw的进入点(即起始点)。
3. L←ep的电平:获取进入点ep的电平。
4. l←ln(unitif(0..1))∙mL⌋:为新元素q生成一个介于0和mL之间的随机级别，其中mL是级别生成的归一化因子。
5. for lc←L…L +1:循环从L到L +1的层。
6. W←SEARCH LAYER(q, ep, ef=1, lc):使用进入点ep和最大距离ef=1在lc层中搜索离q最近的元素。将找到的元素存储在列表W中。
7. ep←取W到q最近的元素:取W到q最近的元素。
8. for lc←min(L, L)…0:循环遍历从min(L, L)到0的层。
9. W←SEARCH LAYER(q, ep, efConstruction, lc):使用进入点ep和最大距离efConstruction搜索层lc中离q最近的元素。将找到的元素存储在列表W中。
10. neighbors←SELECT neighbors (q, W, M, lc):选择W到q最近的M个邻居，只考虑lc层的元素。
11. 在lc层添加邻居到q的双向连接:在lc层添加q与所选邻居之间的双向连接。
12. 对于每个e∈neighbors: //如果需要收缩连接
    对于q的每个邻居e，检查e的连接数是否超过Mmax。如果是这样，使用SELECT neighbors (e, eConn, Mmax, lc)选择一组新的邻居来收缩e的连接，其中eConn是e在lc层的当前连接集。
13. eNewConn←SELECT NEIGHBORS(e, eConn, Mmax, lc):为e选择一组新的邻居，只考虑lc层的元素，保证连接数不超过Mmax。
14. `set neighborhood (e) at layer lc to eNewConn`:将层lc的e的连接集更新为新的set eNewConn。
15. `ep <- W`:设置hnsw的进入点为q。
16. `if 1 > L`:将hnsw的起始点设为q，因为新元素q现在是图的一部分。
17. `return hnsw`:返回更新后的多层图hnsw。

让我们看看算法2。

它在HNSW数据结构上执行K近邻搜索，以查找特定层lc中与查询元素q最近的K个元素。然后，它根据查询元素q与候选元素C和e之间的距离更新候选元素C的集合和找到的最近邻居列表W。最后，如果W中的元素数量超过了动态候选列表ef的大小，则该函数删除从W到q最远的元素。

```
Algorithm 2: SEARCH-LAYER()

SEARCH-LAYER(q, ep, ef, lc)
Input: query element q, enter points ep, number of nearest to q ele-
ments to return ef, layer number lc
Output: ef closest neighbors to q
1 v ← ep // set of visited elements
2 C ← ep // set of candidates
3 W ← ep // dynamic list of found nearest neighbors
4 while │C│ > 0
5 c ← extract nearest element from C to q
6 f ← get furthest element from W to q
7 if distance(c, q) > distance(f, q)
8 break // all elements in W are evaluated
9 for each e ∈ neighbourhood(c) at layer lc // update C and W
10 if e ∉ v
11 v ← v ⋃ e
12 f ← get furthest element from W to q
13 if distance(e, q) < distance(f, q) or │W│ < ef
14 C ← C ⋃ e
15 W ← W ⋃ e
16 if │W│ > ef
17 remove furthest element from W to q
18 return W
```

以下是上述代码的步骤:

1. 初始化变量v为当前的入口点ep。
2. 初始化集合C为当前候选集合。
3. 初始化一个空列表W来存储找到的最近邻。
4. 循环直到候选集合C中的所有元素都求值为止。
5. 从候选元素集合c中提取离查询元素q最近的元素c。
6. 获取从找到的最近邻W到查询元素q的列表中最远的元素f。
7. 如果c到q的距离大于f到q的距离:
8. 然后打破这个循环。
9. 对于lc层c邻域内的每个元素e:
10. 如果e不在访问元素v的集合中，则:
11. 将e添加到访问元素v的集合中。
12. 设f为从W到q的最远的元素。
13. 如果e和q之间的距离小于等于f和q之间的距离，或者W中的元素个数大于等于ef(动态候选列表的大小)，则:
14. 将候选集C更新为C∈e。
15. 将发现的最近邻居W的列表更新为W∈e。
16. 如果W中的元素个数大于等于ef，则:
17. 移除从W到q的最远的元素。
18. 返回找到的最近邻居W的列表。

算法3.

这是一个简单的最近邻选择算法，它接受一个基本元素q、一组候选元素C和一些邻居M作为输入。它返回候选元素C集合中离q最近的M个元素。

```
Algorithm 3: SELECT-NEIGHBORS-SIMPLE()

SELECT-NEIGHBORS-SIMPLE(q, C, M)
Input: base element q, candidate elements C, number of neighbors to
return M
Output: M nearest elements to q
return M nearest elements from C to q
```

步骤如下:

1. 初始化一个空集R来存储选中的邻居。
2. 初始化一个工作队列W来存储候选元素。
3. 如果设置了extendCandidates标志(即true)，则通过将C中每个元素的邻居添加到队列W来扩展候选列表。
4. 而W的大小大于0,R的大小小于M:
5. 从W到q中提取最近的元素e。
6. 如果e比R中的任何元素更接近q，把e加到R中。
7. 否则，将e添加到丢弃队列Wd中。
8. 如果设置了keepPrunedConnections标志(即true)，则从Wd添加一些丢弃的连接到R。
9. 返回R。

让我们看看算法4。

这是一个更复杂的最近邻选择算法，它接受一个基本元素q、一组候选元素C、若干个邻居M、一个层数lc和两个标志extendCandidates和keepPrunedConnections作为输入。它返回由启发式选择的M个元素。

```
Algorithm 4: SELECT-NEIGHBORS-HEURISTIC()

SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keep-
PrunedConnections)
Input: base element q, candidate elements C, number of neighbors to
return M, layer number lc, flag indicating whether or not to extend
candidate list extendCandidates, flag indicating whether or not to add
discarded elements keepPrunedConnections
Output: M elements selected by the heuristic
1 R ← ∅
2 W ← C // working queue for the candidates
3 if extendCandidates // extend candidates by their neighbors
4 for each e ∈ C
5 for each eadj ∈ neighbourhood(e) at layer lc
6 if eadj ∉ W
7 W ← W ⋃ eadj
8 Wd ← ∅ // queue for the discarded candidates
9 while │W│ > 0 and │R│< M
10 e ← extract nearest element from W to q
11 if e is closer to q compared to any element from R
12 R ← R ⋃ e
13 else
14 Wd ← Wd ⋃ e
15 if keepPrunedConnections // add some of the discarded
// connections from Wd
16 while │Wd│> 0 and │R│< M
17 R ← R ⋃ extract nearest element from Wd to q
18 return R
```

步骤如下:

1. 初始化三个队列:R用于选择的邻居，W用于工作的候选，Wd用于丢弃的候选。
2. 设置R的大小为0,W的大小为C的大小。
3. 如果extendCandidates被设置(即，true):
4. 对于C中的每个元素e:
5. 对于第lc层e的每一个邻居eadj:
6. 如果eadj不在W中，则在W中添加它。
7. 而W的大小大于0,R的大小小于M:
8. 从W到q中提取最近的元素e。
9. 如果e比R中的任何元素更接近q，把e加到R中。
10. 否则，将e加到Wd。
11. 如果设置了keepPrunedConnections(即true):
12. 而Wd的大小大于0,R的大小小于M:
13. 从Wd到q中提取最近的元素e。
14. 如果e比R中的任何元素更接近q，就把e加到R中。
15. 返回R。

最后，让我们看看算法5。

这个搜索算法与算法1基本相同。

```
Algorithm 5: K-NN-SEARCH()

K-NN-SEARCH(hnsw, q, K, ef)
Input: multilayer graph hnsw, query element q, number of nearest
neighbors to return K, size of the dynamic candidate list ef
Output: K nearest elements to q
1 W ← ∅ // set for the current nearest elements
2 ep ← get enter point for hnsw
3 L ← level of ep // top layer for hnsw
4 for lc ← L … 1
5 W ← SEARCH-LAYER(q, ep, ef=1, lc)
6 ep ← get nearest element from W to q
7 W ← SEARCH-LAYER(q, ep, ef, lc =0)
8 return K nearest elements from W to q
```

步骤如下:

1. 初始化一个空集合W(当前最近元素的集合)，并将进入点ep设置为网络的顶层。
2. 设置进入点ep的水平为顶层L。
3. 对于每一层lc，从L到1(即从顶层到底层):
4. 使用查询元素q和当前最近的元素W搜索当前层，并将最近的元素添加到W中。
5. 将进入点ep设置为W到q最近的元素。
6. 使用查询元素q和当前最近的元素W搜索下一层，并将最近的元素添加到W中。
7. 返回W中最接近的K个元素作为输出。

**引用**

- [1] [Pugh, W. (1990). Skip lists: A probabilistic alternative to balanced trees. *Communications of the ACM, 33*(6), 668–676. doi:10.1145/78973.78977. S2CID 207691558.](https://15721.courses.cs.cmu.edu/spring2018/papers/08-oltpindexes1/pugh-skiplists-cacm1990.pdf)
- [2] [Xiang Zhang, Junbo Zhao, & Yann LeCun. (2015). Character-level Convolutional Networks for Text Classification](https://www.tensorflow.org/datasets/catalog/ag_news_subset)
- [3] [Yu. A. Malkov, & D. A. Yashunin. (2018). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.](https://arxiv.org/abs/1603.09320)
- [4] Aristides Gionis, Piotr Indyk, Rajeev Motwani, et al. 1999. Similarity search in high dimensions via hashing. In Vldb, Vol. 99. 518–529.