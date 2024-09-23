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







文章：https://zhuanlan.zhihu.com/p/384452959



**Word2Vec必读paper**

**1.** [[Word2Vec\] Efficient Estimation of Word Representations in Vector Space (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Efficient%20Estimation%20of%20Word%20Representations%20in%20Vector%20Space%20%28Google%202013%29.pdf)

Google的Tomas Mikolov提出word2vec的两篇文章之一，这篇文章更具有综述性质，列举了NNLM、RNNLM等诸多词[向量模型](https://zhida.zhihu.com/search?q=向量模型&zhida_source=entity&is_preview=1)，但最重要的还是提出了CBOW和Skip-gram两种word2vec的模型结构。虽然词向量的研究早已有之，但不得不说还是Google的word2vec的提出让词向量重归主流，拉开了整个embedding技术发展的序幕。

**2**. [[Word2Vec\] Distributed Representations of Words and Phrases and their Compositionality (Google 2013)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Distributed%20Representations%20of%20Words%20and%20Phrases%20and%20their%20Compositionality%20%28Google%202013%29.pdf)

Tomas Mikolov的另一篇word2vec奠基性的文章。相比上一篇的综述，本文更详细的阐述了Skip-gram模型的细节，包括模型的具体形式和 Hierarchical Softmax和 Negative Sampling两种可行的训练方法。

**3**. [[Word2Vec\] Word2vec Parameter Learning Explained (UMich 2016)](https://link.zhihu.com/?target=https%3A//github.com/wzhe06/Reco-papers/blob/master/Embedding/%5BWord2Vec%5D%20Word2vec%20Parameter%20Learning%20Explained%20%28UMich%202016%29.pdf)

虽然Mikolov的两篇代表作标志的word2vec的诞生，但其中忽略了大量技术细节，如果希望完全读懂word2vec的原理和实现方法，比如[词向量](https://zhida.zhihu.com/search?q=词向量&zhida_source=entity&is_preview=1)具体如何抽取，具体的训练过程等，强烈建议大家阅读UMich Xin Rong博士的这篇针对word2vec的解释性文章。

Word2Vec算法原理：







参考TODO：

word2vec & doc2vec & text2vec
https://blog.csdn.net/qq_16633405/article/details/80480300

词向量embedding的原理
https://blog.csdn.net/mpk_no1/article/details/72458003

https://blog.csdn.net/u010280923/article/details/130555437

论文：

https://github.com/wzhe06/Reco-papers





最近引入的连续 Skip-gram 模型是一种有效的方法
学习捕获大量精确句法和语义词关系的高质量分布式向量表示。在本文中，我们介绍
提高向量质量和训练的几个扩展
速度。通过对常用词进行二次采样，我们获得了显着的加速和
还要学习更有规律的单词表示。我们还描述了分层 softmax 的一个简单替代方案，称为负采样。
词表示的一个固有限制是它们对词序的漠不关心
以及他们无法代表惯用短语。例如，的含义
「加拿大」和「航空」不能轻易结合获得「加拿大航空」
通过这个例子，我们展示了一种在文本中查找短语的简单方法，并显示
学习数百万个短语的良好向量表示是可能的