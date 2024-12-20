深度学习 Deep Learning 笔记（7）-深度学习中的正则化



深度学习模型在处理复杂任务时表现出了强大的能力，但同时也容易出现[过拟合](https://zhida.zhihu.com/search?q=过拟合&zhida_source=entity&is_preview=1)的问题。为了解决过拟合问题，正则化技术成为了深度学习中的重要手段。

正则化的目标：正则化的目标是降低模型的复杂数度，减少过拟合现象，提升模型的泛化能力。通过在损失函数中引入正则化项，限制模型参数的大小，使模型更加简单、稳定。

#### 2. 常见的正则化方法

假设模型的目标函数如下
$$
\breve{J}\left(\theta;X,y\right)=J(\theta;X,y)+\alpha\Omega(\theta)
$$
如果目标函数的全部参数为w，那么这里的θ表示不包含偏移项的参数，α是超参数， *α* *∈* [0*,* *∞*) ，用于控制正则化的力度，Ω(θ)表示正则项。将 *α* 设为 0 表示没有正则化。*α* 越大，对应正则化惩罚越大。



##### 2.1、L1正则化与L2正则化

$$
L1：\Omega(\theta)=\sum\left|\theta\right| \\
L2：\Omega(\theta)=\frac{1}{2}\sum \theta^2
$$

L1 可以起到过滤某些特征的效果，即特征选择，稀疏化，类似于ReLU函数。

L2 可以起到权重衰减的效果，因为求梯度之后，正好与原梯度相加减。



##### 2.2、Dropout

Dropout是一种常用的正则化技术，它在训练过程中以一定概率随机关闭部分神经元，使模型不依赖于单个神经元，增强模型的[鲁棒性](https://zhida.zhihu.com/search?q=鲁棒性&zhida_source=entity&is_preview=1)。

为什么Dropout最有效：https://blog.csdn.net/stdcoutzyx/article/details/49022443



##### 2.3、批量归一化（Batch Normalization）

批量归一化是一种提升模型训练稳定性的技术，它通过对每一批数据进行归一化处理，使数据分布更加稳定，从而加速模型训练、减小初始化对模型的影响，并起到一定的正则化作用。