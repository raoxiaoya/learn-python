词向量 text embedding 的原理

### 1、词向量的发展历程

#### 1.1、word embedding 词向量

单词嵌入模型将单词表示为密集数字向量。这些向量旨在捕获单词的语义属性 - 向量靠近在一起的单词在语义上应该是相似的。在一个训练较好的embedding中，向量空间中的方向与单词意义的不同方面相关联。例如，“加拿大”的向量可能在一个方向上接近“法国”，而在另一个方向上接近“多伦多”。

一段时间以来，自然语言处理（NLP）和搜索社区一直对单词的向量表示感兴趣。在过去的几年中，人们对单词嵌入的兴趣再次兴起，当时许多传统的任务正在使用神经网络进行重新审视。开发了一些成功的Word embedding算法，包括[word2vec](https://link.zhihu.com/?target=https%3A//papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)和[GloVe](https://link.zhihu.com/?target=https%3A//nlp.stanford.edu/pubs/glove.pdf)，还有后来的训练速度更快的[Fasttext](https://link.zhihu.com/?target=https%3A//github.com/facebookresearch/fastText/)。这些方法使用大型文本集合，并检查每个单词出现的上下文以确定其向量表示：

- word2vec：Skip-gram模型训练神经网络以预测句子中单词周围的上下文单词。
- GloVe：单词的相似性取决于它们与其他上下文单词出现的频率。该算法训练单词共现计数的简单线性模型。
- Fasttext：Facebook的词向量模型，其训练速度比word2vec的训练速度更快，效果又不丢失。

许多研究小组分发的模型已在大型文本语料库（如维基百科）上进行了预训练，使其便于下载和插入下游任务。尽管经常使用预训练版本，但调整模型以适应特定目标数据集和任务会很有帮助。这通常通过在预训练模型上运行轻量级微调步骤来实现。

Word embedding已被证明非常强大和有效，现在NLP任务（如机器翻译和情感分类）中使用Word embedding已经越来越多。

#### 1.2、sentence embedding 句子向量

最近，研究人员不仅关注单词级别的Word embedding，而且开始关注较长的文本如何进行词向量表示。当前大多数的方法基于复杂的神经网络架构，并且有时在训练期间需要不断标记数据以帮助捕获语义信息和提高训练效果。

一旦经过训练，模型就能够获取一个句子并为上下文中的每个单词生成一个向量，以及整个句子的向量。与嵌入字词类似，许多模型的预训练版本可用，允许用户跳过昂贵的培训过程。虽然训练过程可能非常耗费资源，但调用模型的重量要轻得多。训练好的Sentence embeddings足够快，可以用作实时应用程序的一部分。

一些常见的句子嵌入技术包括[InferSent](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1705.02364)，[Universal Sentence Encoder](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1803.11175)，[ELMo](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1802.05365)和[BERT](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1810.04805)。改进单词和句子嵌入是一个活跃的研究领域，并且可能会引入更多强大的模型。

### 2、与传统检索方式的比较

这里所说的检索，是计算句子的相似度，从而来匹配检索结果。在传统的信息检索中，我们基于大多使用TF-IDF等基于单词个数的搜索方法，我们只是计算单词出现而不考虑句子结构。而基于text embedding等技术的搜索，将会考虑句子意思。比如“上午吃饭吗”和“我eat早餐了”这两个句子没有一个单词一样，但是其语义是完全接近的，使用text embedding将能够很好的搜索出来。

文本嵌入在某些重要方面与传统的矢量表示不同：

- Text embedding的向量通常纬度比较低，100~1000。而传统的words vectors纬度可以到5000+。Text embedding技术将文本编码为低维空间向量，同义词和短语在新的向量空间中表示形式会十分相似。
- 在确定向量表示时，Text embedding可以考虑单词的顺序。例如，短语“明天”可以被映射为与“天明”非常不同的向量。
- Text embedding通常适用于短文本。



参考TODO：

word2vec & doc2vec & text2vec
https://blog.csdn.net/qq_16633405/article/details/80480300

词向量embedding的原理
https://blog.csdn.net/mpk_no1/article/details/72458003
https://zhuanlan.zhihu.com/p/384452959
https://blog.csdn.net/u010280923/article/details/130555437

论文：

https://github.com/wzhe06/Reco-papers