自注意力机制Self-Attention


$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{Q K^{T}}{\sqrt{d_{k}}}\right)V \\
\\
Q=W_{Q}X;~K=W_{K}X;~V=W_{V}X
$$
1、对句子进行分词，每一个分词代表一个属性特征；

2、对分词进行embedding向量化，转换到N维空间中；



QKV：Query, Key, Value；QKV本质上都是X的线性变换。QKV是通过大规模训练得到。

Q和K相乘得到两个向量的相似程度，即两个token的相似程度。

softmax是归一化处理，得到一个score。

V表示的是token经过线性变换后的特征。

根号dk是为是里面的值在训练过程中梯度值保持稳定。

位置编码，要记住各个token的位置。



[self-Attention｜自注意力机制 ｜位置编码 ｜ 理论 + 代码_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1qo4y1F7Ep/?spm_id_from=333.788&vd_source=0c75dc193ee55511d0515b3a8c375bd0)



[从编解码和词嵌入开始，一步一步理解Transformer，注意力机制(Attention)的本质是卷积神经网络(CNN)_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1XH4y1T76e/?spm_id_from=333.999.0.0&vd_source=0c75dc193ee55511d0515b3a8c375bd0)



关于 Self-Attention：https://mp.weixin.qq.com/s/dWlm3UmYoOPr4R7NN9Ca8Q



关于 embeddings：https://mp.weixin.qq.com/s/z7ukkqUc8VbI38_lKf8xGA



关于 Self-Attention：https://zhuanlan.zhihu.com/p/410776234



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

