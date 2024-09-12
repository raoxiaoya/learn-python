自注意力机制Self-Attention与Transformer模型



#### 自注意力机制


$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right)V \\
\\
Q=W_{Q}X;~K=W_{K}X;~V=W_{V}X \\
W_{Q}, W_{K}, W_{V}是全局共用的，是通过大规模训练得到。
$$
1、对句子进行分词，每一个分词代表一个属性特征。对分词进行embedding向量化，转换到N维空间中；

2、QKV：Query, Key, Value；QKV本质上都是X的线性变换。

3、Q和K相乘得到两个向量的相似程度，即两个token的相似程度。

4、softmax是归一化处理，得到一个score。

5、V表示的是token经过线性变换后的特征。

6、根号dk是为是里面的值在训练过程中梯度值保持稳定。

7、位置编码(Positional Encoding)，要记住各个token在句子中的位置，一般是正弦或余弦函数。

8、应该把词向量embedding理解为一个N维的空间，每个维度表示一个特征，矩阵的运算其实就是空间变换，向量的内积求的是向量的夹角cosθ。



为什么要使用Q和K两个矩阵，而不是直接训练出一个矩阵呢，有一个比较好的解释是：
$$
单个矩阵的乘法只能代表线性变换，比如：A=X·W\\
而改成两个之后就变成了非线性的变换，比如：A=X·W_{Q}·[X·W_{K}]^{T}\\
=X·[W_{Q}·W_{K}^{T}]·X^{T}\\
=X·W·X^{T}\\
可以类比于 y=ax^2；非线性才有更强的表达能力，能够表达更复杂的情况。
$$


位置编码函数，分别表示奇数位和偶数位，pos表示单词在序列中的位置，d表示了总的维度数。
$$
PE_{(pos, 2i)}=sin(\frac{pos}{10000^{2i/d}}) \\
PE_{(pos, 2i+1)}=cos(\frac{pos}{10000^{2i/d}}) \\
$$
将位置编码直接与embedding矩阵相加得到新的embedding

![image-20240904111447277](D:\dev\php\magook\trunk\server\md\img\image-20240904111447277.png)



[self-Attention｜自注意力机制 ｜位置编码 ｜ 理论 + 代码_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1qo4y1F7Ep/?spm_id_from=333.788&vd_source=0c75dc193ee55511d0515b3a8c375bd0)



[从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1XH4y1T76e/?spm_id_from=333.999.0.0&vd_source=0c75dc193ee55511d0515b3a8c375bd0)



关于 embeddings：https://mp.weixin.qq.com/s/z7ukkqUc8VbI38_lKf8xGA

关于 Self-Attention：https://mp.weixin.qq.com/s/dWlm3UmYoOPr4R7NN9Ca8Q

关于 Self-Attention：https://zhuanlan.zhihu.com/p/410776234



```python

```

![image-20240902222812255](D:\dev\php\magook\trunk\server\md\img\image-20240902222812255.png)

![image-20240902222857198](D:\dev\php\magook\trunk\server\md\img\image-20240902222857198.png)

![image-20240902223518689](D:\dev\php\magook\trunk\server\md\img\image-20240902223518689.png)

![image-20240902223839580](D:\dev\php\magook\trunk\server\md\img\image-20240902223839580.png)

```python
# Muti-head Attention 机制的实现
from math import sqrt
import torch
import torch.nn


class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output
```



#### Transformer模型



https://mp.weixin.qq.com/s/K82ZcmbA8m6bhgjIvge3Wg

![image-20240904104115489](D:\dev\php\magook\trunk\server\md\img\image-20240904104115489.png)

1)词向量层(Embedding)

2)前馈神经网络(Feed-Forward)

3)残差连接(Add)和层标准化(Norm)

4)线性层(Linear)和softmax层





以机器翻译为例，训练过程如下，将中文和英文分别从编码器和解码器输入，编码器和解码器都能实现embedding，然后编码器将输出注入到解码器中，然后计算损失函数，然后就是梯度反向传播，优化收敛。



在训练过程中，我们将目标序列作为解码器的输入，这种方法被称为教师强制。为什么采用这种方法，以及这个术语的含义是什么呢？在训练时，我们本可以采用与推理时相同的方法，即通过一个循环，取输出序列的最后一个单词，附加到解码器输入，并送入解码器进行下一轮迭代。但这样做不仅会使训练过程耗时更长，还会增加模型训练的难度。因为模型需要基于可能错误的第一个预测单词来预测第二个单词，以此类推。相反，通过给解码器提供目标序列作为输入，我们实际上是在给模型一个提示，就像老师指导学生一样。即使模型预测的第一个单词是错误的，它也能利用正确的第一个单词来预测第二个单词，从而避免错误不断累积。此外，Transformer能够并行输出所有单词，无需循环，这大大加快了训练速度。

![image-20240903223033357](D:\dev\php\magook\trunk\server\md\img\image-20240903223033357.png)

![image-20240903223052125](D:\dev\php\magook\trunk\server\md\img\image-20240903223052125.png)



以机器翻译为例，翻译过程如下，从编码器输入中文，从解码器输入一个开始符，在解码器这边将输入的embedding与模型的向量空间进行匹配，将得分最高的输出，将新的输出作为输入在进行匹配，不断累加，直到碰到终止条件，所以Transformer模型的输出是间歇性的。



![image-20240903223324179](D:\dev\php\magook\trunk\server\md\img\image-20240903223324179.png)



GPT只有解码器这个部分，GPT的原理是将你的问题作为起始的输入信息，输入编码器，然后就重复后面的过程，所谓的提示词Prompt其实就是其实输入引导词，引导词越详细匹配越精准。

那么为什么会有终止条件呢，因为随着一轮一轮的匹配，输入信息越滚越多，那么模型的向量空间中能匹配到的信息会越来越少，直到得分低于一个阈值就宣告生成结束。