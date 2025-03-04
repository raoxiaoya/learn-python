基于BERT的NER（命名实体识别）模型



写在之前

This post is all you need（层层剥开Transformer附全文高清PDF）https://mp.weixin.qq.com/s/uch_AGcSB8OSAeVu2sme8A



This Post Is All You Need（下）——步步走进BERT https://mp.weixin.qq.com/s/tQELF16n5O1-3cvapeaf4Q



Transformer网络 https://mp.weixin.qq.com/s/4vrU81JfgVDklRJkndiD4g



Transformer结构 https://mp.weixin.qq.com/s/sb8JyaXh9YdXu5KNtgsOfA



第10.6节 BERT网络 https://mp.weixin.qq.com/s/S8cCRdzielk2O95K85czYA

第10.7节 从零实现BERT https://mp.weixin.qq.com/s/0MM0gbBSFxifuQpXOTg6GQ

第10.8节 BERT文本分类模型 https://mp.weixin.qq.com/s/HFdEZV9m_Sc87bae57oD2g

第10.9节 BERT问题选择模型 https://mp.weixin.qq.com/s/O_hNiKdYESaGfJuF0IkaAQ

第10.10节 BERT问题回答模型 https://mp.weixin.qq.com/s/oWDGBttbX4ujkFEpwJqkqQ

第10.11节 BERT命名体识别模型 https://mp.weixin.qq.com/s/wAK-_NJb7mjdH3R-zI63Ug

第10.12节 BERT从零训练 https://mp.weixin.qq.com/s/nxIss6Zeex_6omGvLaHW1w









NER：命名实体识别（Named Entity Recognition），即识别出句子中的实体，例如人名、地名、组织，专有名称，学术名词等等。



https://github.com/zjy-ucas/ChineseNER







https://mp.weixin.qq.com/s/Eaqx55P6cBXb22FW6Q-Xrw

https://blog.csdn.net/weixin_45101959/article/details/122982404

https://www.jianshu.com/p/1d6689851622









在大模型领域，全量训练一个大模型，比如 7B，就需要8块A100，因此普通玩家在大模型领域能做的就三个方面：Prompt工程，微调（Fine-Tune）和 RAG。



大模型LLM微调Fine-Tune，在实践中，我们觉得Fine-Tune还是有点奢侈的，特别是数据整理是个大问题，基本上当时的微调都需要按目标数据和通用数据比1:7的比例喂进去，微调成本（不仅仅是money）过于高，有点像是给马儿打造一辆马车。所以，对于不是专职做大模型的团队来说，微调至少不能当做常规手段，尽管我们依然会对LLM和embedding模型做一些微调。



再接下来就是Prompt工程，当时宝玉翻译的吴恩达视频算是一个入门，后面更是直接看外网资料学习，Prompt工程确实有效，而且很自然地成了RAG中的一环。而在我们的实践中，**RAG最终成了最靠谱的解决方案**。

-------

















---



什么是BERT模型，它的特点是什么？



BERT（Bidirectional Encoder Representations from Transformers）双向编码器语言表征模型。

特点：与传统的语言表征模型不同，BERT使用的是深度双向表征，即在每一层中同时基于左边和右边的context来做预测。
优势：预训练的BERT模型只需要在上面增加额外的一层，就能广泛地于多种NLP任务中进行fine-tune.



BERT模型基于Transformer中的Encoder。它是基于Transformer论文的一个应用，主要是它在NLP领域的强大，才使得人们关注到Transformer。关于 Bert 所取得的成就这里就不再细说，用论文里面作者的描述来说就是：BERT 不仅在概念上很简单，并且从经验上来看它也非常强大，以至于直接刷新了 11 项 NLP 记录。



作者之所以说 BERT 在概念上很简单，这是因为 BERT 本质上是根据 Transformer 中 Encoder 堆叠而来所以说它简单；而作者说 BERT 从经验上来看非常强大，是因为 BERT 在训练过程中所基于的 Mask LM 和 NSP 这两个任务所以说其强大。



BERT 有两个预训练的任务，分别是 MLM 和 NSP

**MLM**：mask language model，其实就是完形填空，在句子中随机挖出一些词使用 [MASK] 替代，然后训练模型预测它的值。随机mask 15%的单词，也就是100个单词里面挑出15个单词来mask。这15%的单词又不是全部真正的mask，而是10%替换成其它单词，10%原封不动，80%替换成mask。

**NSP**：Next Sentence Prediction，即每个样本都是由A和B两句话构成，分为两种情况：①、句子B确实是句子A的下一句话，样本标签为IsNext；②、句子B不是句子A的下一句，句子B为语料中的其他随机句子，样本标签为NotNext。在样本集合中，两种情况的样本占比均为50%。此任务的样本是句子对，比如

样本1：[CLS] sentence1 [SEP] sentence2 [SEP]

标签：IsNext

样本2：[CLS] sentence2 [SEP] sentence4 [SEP]

标签：NotNext



发明BERT的动机

在论文的介绍部分作者谈到，预训练语言模型（Language model pre-training）对于下游很多自然语言处理任务都有着显著的改善。但是作者继续说到，现有预训练模型的网络结构限制了模型本身的表达能力，其中最主要的限制就是没有采用双向编码的方法来对输入进行编码。

在 OpenAI GPT 模型中，它使用了从左到右（left-to-right）的网络架构，这就使得模型在编码过程中只能够看到当前时刻之前的信息，而不能够同时捕捉到当前时刻之后的信息。比如有一个句子“the animal didn't cross the street,because it was too tired.”，无论模型采用的是 left-to-right 的编码方式还是采用 right-to-left 的编码方式，模型在对“it”进行编码的时候都不能很好地捕捉到其具体所指代的信息。这就类似于我们人在阅读这句话时一样，在没有看到“tired”这个词之前我们同样也无法判断“it”具体所指代的事物。例如：如果我们把“tired”这个词换成“wide”，则“it”指代的就变成了“street”。

所以，如果模型采用的是双向编码的方式，那么从理论上来看就能够很好的避免这个问题，

![image-20241029092708852](D:\dev\php\magook\trunk\server\md\img\image-20241029092708852.png)

在图 1-4 中，橙色线条表示“it”应该将注意力集中在哪些位置上，颜色越深则表示注意力权重越大。通过这幅图可以发现，模型在对“it”进行编码时，将大部分注意力都集中在了“The animal”上，而这也符合我们预期的结果。

在论文中，作者提出了采用 BERT（Bidirectional Encoder Representations from Transformers）这一网络结构来实现模型的双向编码学习能力。同时，为了使得模型能够有效的学习到双向编码的能力，BERT 在训练过程中使用了基于掩盖的语言模型（Masked Language Model, MLM），即随机对输入序列中的某些位置进行遮蔽，然后通过模型来对其进行预测。

作者继续谈到，由于 MLM 预测任务能够使得模型编码得到的结果同时包含上下文的语境信息，因此有利于训练得到更深的 BERT 网络模型。除此之外，在训练 BERT 的过程中作者还加入了下句预测任务（Next Sentence Prediction, NSP），即同时输入两句话到模型中，然后预测第 2 句话是不是第 1 句话的下一句话。



技术实现

对于技术实现这部分内容，掌柜将会分为三个大的部分来进行介绍。第一部分主要介绍BERT的网络结构原理以及MLM和NSP这两种任务的具体原理；第二部分将主要介绍如何实现 BERT 以及 BERT 预训练模型在下游任务中的使用；第三部分则是介绍如何利用MLM和NSP这两个任务来训练BERT模型（可以是从头开始，也可以是基于开源的 BERT 预训练模型开始）。下面，掌柜将先对第一部分的内容进行介绍。



BERT 网络结构

![image-20241029093722298](D:\dev\php\magook\trunk\server\md\img\image-20241029093722298.png)

BERT 网络结构整体上就是由多层的 Transformer Encoder堆叠所形成。如图 1-5 所示便是一个详细版的 BERT 网络结构图，可以发现其上半部分的结构与之前介绍的 Transformer Encoder 差不多，只不过在 Input 部分多了一个Segment Embedding。

最值得注意的一点就是 **BERT开源的预训练模型最大只支持 512** 个token的长度，这是因为其在训练过程中（位置 Positional Embedding）词表的最大长度只有512。max_position_embeddings



参数

BERT-Base：L=12，H=768，A=12

BERT-Large：L=24，H=1024，A=16

其中，L为BertLayer（即 Transformer Encoder）的层数，H是模型的维度，A是多头注意力中多头的个数。



为了能够更好训练 BERT 网络，论文作者在 BERT 的训练过程中引入两个任务，MLM 和 NSP。对于 MLM 任务来说，其做法是随机掩盖掉输入序列中 15%的 Token（即用“[MASK]”替换掉原有的 Token），然后在 BERT 的输出结果中取对应掩盖位置上的向量进行真实值预测。

接着作者提到，虽然 MLM 的这种做法能够得到一个很好的预训练模型，但是仍旧存在不足之处。由于在 fine-tuning 时，由于输入序列中并不存在“[MASK]”这样的 Token，因此这将导致 pre-training 和 fine-tuning 之间存在不匹配不一致的问题（GAP）。

为了解决这一问题，作者在原始 MLM 的基础了做了部分改动，即先选定15% 的 Token，然后将其中的 80% 替换为“[MASK]”、 10% 随机替换为其它Token、剩下的 10% 不变。最后取这 15% 的 Token 对应的输出做分类来预测其真实值。

由于很多下游任务需要依赖于分析两句话之间的关系来进行建模，例如问题回答等。为了使得模型能够具备有这样的能力，作者在论文中又提出了二分类的下句预测任务。

具体地，对于每个样本来说都是由 A 和 B 两句话构成，其中 50% 的情况 B确实为 A 的下一句话（标签为 IsNext），另外的 50% 的情况是 B 为语料中其它的随机句子（标签为 NotNext），然后模型来预测 B 是否为 A 的下一句话。

如图 1-10 所示便是 ML 和 NSP 这两个任务在 BERT 预训练时的输入输出示意图，其中最上层输出的 *C* 在预训练时用于 NSP 中的分类任务；其它位置上的 T 则用于预测被掩盖的 Token。

![image-20241029102102830](D:\dev\php\magook\trunk\server\md\img\image-20241029102102830.png)

到此，对于 BERT 模型的原理以及 NSP、MLM 这两个任务的内容就介绍完了。总的来说，如果单从网络结构上来看 BERT 并没有太大的创新，这也正如作者所说“BERT 整体上就是由多层的 Transformer Encoder 堆叠而来”，并且所谓的“bidirectional”其实指的也就是 Transformer 中的 self-attention 机制。同时，在掌柜看来真正让 BERT 表现出色的应该是基于 MLM 和 NSP 这两种任务的预训练过程，使得训练得到的模型具有强大的表征能力。