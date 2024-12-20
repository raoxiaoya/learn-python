深度学习 Deep Learning 笔记（3）-概率与信息论



#### 3.1、为什么要使用概率

频率派概率：代表一件事情发生的概率，比如掷骰子游戏，只要次数足够多就能够得到想要的点数。

贝叶斯概率：一件事情发生的可能性大小，比如天气预报，这个与次数无关。



#### 3.2、随机变量

可以随机地取不同值的变量。

随机变量可以是离散的或者连续的。

#### 3.3、概率分布

##### 3.3.1、离散型变量和概率质量函数

概率质量函数（probability mass function, PMF）：离散型变量的概率分布。

概率质量函数将随机变量能够取得的每个状态映射到随机变量取得该状态的概率。x = *x* 的概率用 *P*(*x*) 来表示，概率为 1 表示 x = *x* 是确定的，概率为 0 表示x = *x* 是不可能发生的。有时为了使得PMF的使用不相互混淆，我们会明确写出随机变量的名称：*P*(x = *x*)。有时我们会先定义一个随机变量，然后用 *∼* 符号来说明它遵循的分布：x *∼* *P*(x)。

概率质量函数可以同时作用于多个随机变量。这种多个变量的概率分布被称为 联合概率分布（joint probability distribution）。*P*(x = *x,* y = *y*) 表示 x = *x* 和y = *y* 同时发生的概率。我们也可以简写为 *P*(*x, y*)。

如果一个函数 *P* 是随机变量 x 的 PMF，必须满足下面这几个条件：

- *P* 的定义域必须是 x 所有可能状态的集合。

- 0 *≤* *P*(x) *≤* 1. 不可能发生的事件概率为 0，并且不存在比这概率更低。的状态。类似的，能够确保一定发生的事件概率为 1，而且不存在比这概率更高的状态。

- 归一化（normalized）。
  $$
  \sum_{x∈\mathbf{x}}P(x) = 1
  $$
  例如，考虑一个离散型随机变量 x 有 *k* 个不同的状态。我们可以假设 x 是 均匀分布（uniform distribution）的），通过将它的PMF设为
  $$
  P(\mathbf{x} = x_i) = {\frac{1}{k}}
  $$
  对于所有的 *i* 都成立。我们可以看出这满足上述成为概率质量函数的条件。因为 *k*是一个正整数，所以 1/k 是正的。我们也可以看出
  $$
  \sum_{i}{P(\mathbf{x}=x_{i})}=\sum_{i}{\frac{1}{k}}={\frac{k}{k}}=1
  $$
  因此分布也满足归一化条件。

  

##### 3.3.2、连续型变量和概率密度函数

当我们研究的对象是连续型随机变量时，我们用概率密度函数（probability density function, PDF）而不是概率质量函数来描述它的概率分布。

如果一个函数 *p*是概率密度函数，必须满足下面这几个条件：

![image-20240701144309375](D:\dev\php\magook\trunk\server\md\img\image-20240701144309375.png)

概率密度函数 *p*(*x*) 并没有直接对特定的状态给出概率，相对的，它给出了落在面积为 *δx* 的无限小的区域内的概率为 *p*(*x*)*δx*。注意p(x)是密度。

我们可以对概率密度函数求积分来获得点集的真实概率质量。特别地，*x* 落在集合 S 中的概率可以通过 *p*(*x*) 对这个集合求积分来得到。在单变量的例子中，*x* 落在区间 [*a, b*] 的概率是：
$$
\int_{[a,b]}p(x)dx
$$

#### 3.4、边缘概率

![image-20240701152457973](D:\dev\php\magook\trunk\server\md\img\image-20240701152457973.png)

![image-20240701152730780](D:\dev\php\magook\trunk\server\md\img\image-20240701152730780.png)

#### 3.5、条件概率

在很多情况下，我们感兴趣的是某个事件，在给定其他事件发生时出现的概率。这种概率叫做条件概率。我们将给定 x = *x*为条件，那么y = *y* 发生的条件概率记为*P*(y = *y* *|* x = *x*)。这个条件概率可以通过下面的公式计算：
$$
P(y=y\mid\mathbf{x}=x)={\frac{P({\mathbf{y}}=y,{\mathbf{x}}=x)}{P({\mathbf{x}}=x)}}
$$
条件概率只在 *P*(x = *x*) *>* 0 时有定义。我们不能计算给定在永远不会发生的事件上的条件概率。

竖线后面的为条件。

#### 3.6、条件概率的链式法则

任何多维随机变量的联合概率分布，都可以分解成只有一个变量的条件概率相乘的形式：
$$
P({\bf x}^{(1)},\cdot\cdot\cdot,{\bf x}^{(n)})=P({\bf x}^{(1)})\Pi_{i=2}^{n}P({\bf x}^{(i)}\mid{\bf x}^{(1)},\cdot\cdot\cdot,{\bf x}^{(i-1)})
$$
公式的意思是计算x1到xn同时发生的概率。

这个规则被称为概率的 链式法则（chain rule）或者 乘法法则（product rule）。它可以直接从式 (3.5) 条件概率的定义中得到。例如，使用两次定义可以得到

![image-20240701154252200](D:\dev\php\magook\trunk\server\md\img\image-20240701154252200.png)

#### 3.7、独立性和条件独立性

![image-20240701154628813](D:\dev\php\magook\trunk\server\md\img\image-20240701154628813.png)

#### 3.8、期望、方差、协方差

**1、期望**

函数 *f*(*x*) 关于某个分布 *P*(x) 的 期望（expectation）或者 期望值（expectedvalue）是指，当 *x* 由 *P* 产生，*f* 作用于 *x* 时，*f*(*x*) 的`平均值`。

对于离散型随机变量，这可以通过求和得到：
$$
\mathbb{E}_{\mathbf{x}\sim P}[f(x)]=\sum_{x}P(x)f(x)
$$
对于连续型随机变量可以通过求积分得到：
$$
\mathbb{E}_{\mathbf{x}\sim P}[f(x)]=\int P(x)f(x)dx
$$
期望是线性的，因此
$$
\mathbb{E}_{\mathbf{x}}[\alpha f(x)+\beta g(x)]=\alpha\mathbb{E}_{\mathbf{x}}[f(x)]+\beta\mathbb{E}_{\mathbf{x}}[g(x)]
$$


**2、方差**

方差（variance）衡量的是当我们对 *x* 依据它的概率分布进行采样时，随机变量 x 的函数值会呈现多大的差异：
$$
\mathrm{Var}(f(x))=\mathbb{E}\left[(f(x)-\mathbb{E}[f(x)])^{2}\right]
$$
当方差很小时，*f*(*x*) 的值形成的簇比较接近它们的期望值。方差的平方根被称为`标准差`（standard deviation）。



注意，这里的期望和方差与国内的教材解释的不一样！！国内的解释如下。

有一个概率函数P(x)，那么期望表示随机变量的中心位置。
$$
E(x)=\mu=\sum x P(x)
$$
方差用于表示数据的分散程度。数据波动越大，方差就越大。
$$
Var(x)=\sigma^{2}=\sum(x-\mu)^{2}P(x)=E((x-\mu)^{2})=E((x-E(x))^{2}) \\
\sigma 为标准差，通常使用 (\mu, \sigma^{2})来表示一个分布。
$$
如果 x 是由 f(y) 函数的结果，那么就跟上面的公式一样了。



**3、协方差**

协方差（covariance）在某种意义上给出了两个变量线性相关性的强度以及这些变量的尺度：
$$
\mathrm{{Cov}}(f(x),g(y))=\mathbb{E}\left[\left(f(x)-\mathbb{E}\left[f(x)\right]\right)(g(y)-\mathbb{E}\left[g(y)\right])\right] \\
简单点， Cov(x, y) = E(xy) - E(x)E(y)
$$
协方差的`绝对值`如果很大则意味着变量值变化很大并且它们同时距离各自的均值很远。如果协方差是正的，那么两个变量都倾向于同时取得相对较大的值。如果协方差是负的，那么其中一个变量倾向于取得相对较大的值的同时，另一个变量倾向于取得相对较小的值，反之亦然。其他的衡量指标如 `相关系数`（correlation）将每个变量的贡献归一化，为了只衡量变量的相关性而不受各个变量尺度大小的影响。

协方差和相关性是有联系的，但实际上是不同的概念。它们是有联系的，因为两个变量如果相互独立那么它们的协方差为零，如果两个变量的协方差不为零那么它们一定是相关的。然而，独立性又是和协方差完全不同的性质。两个变量如果协方差为零，它们之间一定没有线性关系。独立性比零协方差的要求更强，因为独立性还排除了非线性的关系。

两个变量相互依赖但具有零协方差是可能的。例如，假设我们首先从区间 [*−*1*,* 1] 上的均匀分布中采样出一个实数 *x*。然后我们对一个随机变量 *s* 进行采样。*s* 以 1/2 的概率值为 1，否则为-1。我们可以通过令 *y* = *sx* 来生成一个随机变量 *y*。显然，*x* 和 *y* 不是相互独立的，因为 *x* 完全决定了 *y* 的尺度。然而，Cov(*x, y*) = 0。

![image-20240701164727630](D:\dev\php\magook\trunk\server\md\img\image-20240701164727630.png)

**4、总结**

x 服从于 概率分布 P(x)，现在有一个函数 f(x)，那么

期望：f(x) 的平均值。

方差：f(x) 曲线的波动的剧烈程度。

标准差：方差的平方根。

协方差：两个变量 x 和 y 之间的相互影响或者相关性。



#### 3.9、常用的概率分布

##### 3.9.1、 Bernoulli 分布（伯努利分布，二项分布）

是单个二值随机变量的分布。它由单个参数 *ϕ* *∈* [0*,* 1] 控制，*ϕ* 给出了随机变量等于 1 的概率。它具有如下的一些性质：
$$
P({\bf x}=1) = \phi \\
P({\bf x}=0) = 1 - \phi \\
通用的，P({\bf x}=x) = \phi^{x}(1-\phi)^{1-x} \\
\mathbb{E}_{\bf x}[{\bf x}]=\phi \\
Var_{\bf x}({\bf x})=\phi(1-\phi)
$$
phi 读作 fai

##### 3.9.2、Multinoulli 分布

Multinoulli 分布，也称 范畴分布，分类分布（categotical distribution），是 Bernoulli分布从两个取值状态到多个取值状态的扩展。具体来说，Mutinoulli分布是指在具有k个不同状态的单个离散型随机变量上的分布，其中k是一个有限值，且满足k个状态的概率之和为1。

Bernoulli 分布和 Multinoulli 分布足够用来描述在它们领域内的任意分布。它们能够描述这些分布，不是因为它们特别强大，而是因为它们的领域很简单；它们可以对那些，能够将所有的状态进行枚举的离散型随机变量进行建模。



##### 3.9.3、高斯分布（正态分布）

实数上最常用的分布就是 正态分布（normal distribution），也称为 高斯分布（Gaussian distribution）：
$$
\mathcal{N}(x;\mu,\sigma^{2})=\sqrt{\frac{1}{2\pi\sigma^{2}}}\exp\left(-\frac{1}{2\sigma^{2}}(x-\mu)^{2}\right)
$$
![image-20240702111155378](D:\dev\php\magook\trunk\server\md\img\image-20240702111155378.png)



公式中的`exp`就是指数`e`，是个无理数，值为 2.71828...

y=e的x次方的曲线是个经典的曲线

![image-20240702141650860](D:\dev\php\magook\trunk\server\md\img\image-20240702141650860.png)



高斯分布的期望：
$$
\mathbb{E}[{\bf x}]=\mu
$$
高斯分布的方差：
$$
Var({\bf x}) = \sigma^2
$$
我们使 *β* 等于 *σ* 的平方的倒数，则
$$
N(x;\mu,\beta^{-1})=\sqrt{\frac{\beta}{2\pi}}\exp\left(-\frac12\beta(x-\mu)^{2}\right)
$$
正态分布可以推广到 *n* 维空间，这种情况下被称为 多维正态分布（multivariate normal distribution）。它的参数是一个正定对称矩阵 **Σ**：
$$
\mathcal{N}(x;\mu,\Sigma)=\sqrt{\frac{1}{(2\pi)^{n}\operatorname*{det}(\Sigma)}}\exp\left(-\frac{1}{2}(x-\mu)^{\top}{\Sigma}^{-1}(x-\mu)\right)
$$
参数 **µ** 仍然表示分布的均值，只不过现在是向量值。参数 **Σ** 给出了分布的协方差矩阵。
$$
{\cal N}(x;\mu,\beta^{-1})=\sqrt{\frac{\operatorname*{det}(\beta)}{(2\pi)^{n}}}\exp\left(-\frac{1}{2}(x-\mu)^{\top}\beta(x-\mu)\right)
$$
我们常常把协方差矩阵固定成一个对角阵。一个更简单的版本是 各向同性（isotropic）高斯分布，它的协方差矩阵是一个标量乘以单位阵。



##### 3.9.4、指数分布与Laplace分布

在深度学习中，我们经常会需要一个在 *x* = 0 点处取得边界点 (sharp point) 的分布。为了实现这一目的，我们可以使用 指数分布（exponential distribution）：
$$
p(x;\lambda)=\lambda\mathbf{1}_{x\geq0}\exp(-\lambda x)
$$
指数分布使用指示函数(indicator function) ***I***(x>=0)来使得当 *x* 取负值时的概率为零。

一个联系紧密的概率分布是 **Laplace** 分布（Laplace distribution），它允许我们在任意一点 *µ* 处设置概率质量的峰值：
$$
\operatorname{Laplace}(x;\mu,\gamma)={\frac{1}{2\gamma}}\exp\left(-{\frac{|x-\mu|}{\gamma}}\right)
$$

![image-20240812172601419](D:\dev\php\magook\trunk\server\md\img\image-20240812172601419.png)



##### 3.9.5、Dirac分布和经验分布

在一些情况下，我们希望概率分布中的所有质量都集中在一个点上。这可以通过 **Dirac delta** 函数（Dirac delta function）*δ*(*x*) 定义概率密度函数来实现：
$$
p(x) = \delta(x − \mu)
$$
Dirac delta 函数被定义成在除了 0 以外的所有点的值都为 0，但是积分为 1。Dirac delta 函数不像普通函数一样对 *x* 的每一个值都有一个实数值的输出，它是一种不同类型的数学对象，被称为 广义函数（generalized function），广义函数是依据积分性质定义的数学对象。我们可以把 Dirac delta 函数想成一系列函数的极限点，这一系列函数把除 0 以外的所有点的概率密度越变越小。

通过把 *p*(*x*) 定义成 *δ* 函数左移 *−**µ* 个单位，我们得到了一个在 *x* = *µ* 处具有无限窄也无限高的峰值的概率质量。

![image-20240702114442707](D:\dev\php\magook\trunk\server\md\img\image-20240702114442707.png)



##### 3.9.6、分布的混合

通过组合一些简单的概率分布的混合可以实现任意的概率分布。



![image-20240812173225469](D:\dev\php\magook\trunk\server\md\img\image-20240812173225469.png)

![image-20240812173303976](D:\dev\php\magook\trunk\server\md\img\image-20240812173303976.png)

#### 3.10、常用函数的有用性质

https://zhuanlan.zhihu.com/p/364620596

某些函数在处理概率分布时经常会出现，尤其是深度学习的模型中用到的概率分布。

**1、logistic sigmoid 函数**
$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$
它通常用来产生 Bernoulli 分布中的参数 *ϕ*，因为它的范围是(0*,* 1)，处在 *ϕ* 的有效取值范围内。

sigmoid 函数在变量取绝对值非常大的正值或负值时会出现 饱和（saturate）现象，意味着函数会变得很平，并且对输入的微小改变会变得不敏感。

![image-20240702142058273](D:\dev\php\magook\trunk\server\md\img\image-20240702142058273.png)

在深度学习里面会使用 logistic sigmoid 作为中间层的激活函数来将所有输入挤压到 (0, 1) 之间，具有归一化的效果。

在神经网络刚发展起来的时候，这确实是一个被广泛利用的激活函数，但随着发展，它的弊端也慢慢暴露，其中最为致命的就是它的“梯度消失现象”，也就是图中看到的左右两边的导数几乎为零，我们知道神经网络是通过求偏微分实现神经元的反向传播，这将导致神经元无法继续进行反向传播，因此这个函数慢慢被看似简单的ReLu激活函数代替了。

SIgmoid 的导数：
$$
f'(x) = f(x)(1-f(x))
$$
在 f(x)=1/2 处导数取得最大值 1/4。

![image-20240815160758046](D:\dev\php\magook\trunk\server\md\img\image-20240815160758046.png)



**在什么情况下适合使用 Sigmoid 激活函数呢？**

- Sigmoid 函数的输出范围是 0 到 1。由于输出值限定在 0 到1，因此它对每个神经元的输出进行了归一化；
- 用于将预测概率作为输出的模型。由于概率的取值范围是 0 到 1，因此 Sigmoid 函数非常合适；
- 梯度平滑，避免「跳跃」的输出值；
- 函数是可微的。这意味着可以找到任意两个点的 sigmoid 曲线的斜率；
- 明确的预测，即非常接近 1 或 0。

**Sigmoid 激活函数存在的不足：**

- **梯度消失**：注意：Sigmoid 函数趋近 0 和 1 的时候变化率会变得平坦，也就是说，Sigmoid 的梯度趋近于 0。神经网络使用 Sigmoid 激活函数进行反向传播时，输出接近 0 或 1 的神经元其梯度趋近于 0。这些神经元叫作饱和神经元。因此，这些神经元的权重不会更新。此外，与此类神经元相连的神经元的权重也更新得很慢。该问题叫作梯度消失。因此，想象一下，如果一个大型神经网络包含 Sigmoid 神经元，而其中很多个都处于饱和状态，那么该网络无法执行反向传播。
- **不以零为中心**：Sigmoid 输出不以零为中心的,，输出恒大于0，非零中心化的输出会使得其后一层的神经元的输入发生偏置偏移（Bias Shift），并进一步使得梯度下降的收敛速度变慢。
- **计算成本高昂**：exp() 函数与其他非线性激活函数相比，计算成本高昂，计算机运行起来速度较慢。



**2、Tanh函数**

双曲正切函数，取值区间(-1, 1)
$$
f(x)=tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}=\frac{2}{1+e^{-2x}}-1
$$
导数
$$
f^{\prime}(x)=1-(tamh(x))^{2}
$$
![image-20240815151314671](D:\dev\php\magook\trunk\server\md\img\image-20240815151314671.png)

tanh函数跟sigmoid函数一样，都存在梯度消失问题。

tanh可以看成是放大并平移的sigmoid函数，即
$$
tanh(x)=2~sigmoid(2x)-1
$$




**3、ReLu函数**

最简单却是最好用的函数。在多分类问题中，要解决的核心问题就是如何画一条曲线，将样本空间中的样本按类别隔开。

把所有的负值都变为0，而正值不变，这种操作被成为**单侧抑制**。relu函数可以给与一个`系数`使折线的角度朝向不同的方向。这个折线是如此的形象，特别像过安检的时候设置的那些隔离带。

![image-20240815164803326](D:\dev\php\magook\trunk\server\md\img\image-20240815164803326.png)
$$
ReLU(x)={\left\{\begin{array}{l l}{x}&{x \gt 0} \\ 
{0}&{x \leq 0}\end{array}\right.}
$$




RELU函数的优点：

- 相关大脑方面的研究表明生物神经元的信息编码通常是比较分散及稀疏的。通常情况下，大脑中在同一时间大概只有1%-4%的神经元处于活跃状态。使用线性修正以及正则化（regularization）可以对机器神经网络中神经元的活跃度（即输出为正值）进行调试；相比之下，逻辑函数在输入为0时达到1/2 , 即已经是半饱和的稳定状态，不够符合实际生物学对模拟神经网络的期望。不过需要指出的是，一般情况下，在一个使用修正线性单元（即线性整流）的神经网络中大概有50%的神经元处于激活态；
- 更加有效率的梯度下降以及反向传播：避免了梯度爆炸和梯度消失问题；
- 简化计算过程：没有了其他复杂激活函数中诸如指数函数的影响；同时活跃度的分散性使得神经网络整体计算成本下降。

RELU函数的不足：

- Dead ReLU 问题。当输入为负时，ReLU 完全失效，在正向传播过程中，这不是问题。有些区域很敏感，有些则不敏感。但是在反向传播过程中，如果输入负数，则梯度将完全为零；

  > **【Dead ReLU问题】**ReLU神经元在训练时比较容易“死亡”。在训练时，如果参数在一次不恰当的更新后，第一个隐藏层中的某个ReLU 神经元在所有的训练数据上都不能被激活，那么这个神经元自身参数的梯度永远都会是0，在以后的训练过程中永远不能被激活。这种现象称为死亡ReLU问题，并且也有可能会发生在其他隐藏层。

- 不以零为中心：和 Sigmoid 激活函数类似，ReLU 函数的输出不以零为中心，ReLU 函数的输出为 0 或正数,给后一层的神经网络引入偏置偏移，会影响梯度下降的效率。

RELU总结：

- 虽然ReLU函数大于零的部分和小于零的部分分别都是线性函数，但是整体并不是线性函数，所以仍然可以做为激活函数，ReLU函数其实是分段线性函数，把所有的负值都变为0，而正值不变，这种操作被成为单侧抑制。
- 在训练数据的时候，由于对于不同的任务，可能某些神经元的输出影响就比较大，有些则小，甚至有些则无，类似于人的大脑，左脑和右脑分别管理逻辑能力与想象力，当使用右脑的时候，就抑制左脑，当使用左脑的时候，抑制右脑，RELU函数正好可以实现小于0的数直接受到抑制，这就使得神经网络的激活更接近于生物学上的处理过程，给神经网络增加了生命。



好用程度：Relu > Tanh > Sigmoid



**Leaky ReLU**

为了解决 ReLU 激活函数中的梯度消失问题，当 x < 0 时，我们使用 Leaky ReLU——该函数试图修复 dead ReLU 问题。
$$
LeakyReLU(x)={\left\{\begin{array}{l l}{x}&{x \gt 0} \\ 
{\gamma x}&{x \leq 0}\end{array}\right.}
$$
其中 γ 是一个很小的数，如 0.1，0.01。

![image-20240815164952326](D:\dev\php\magook\trunk\server\md\img\image-20240815164952326.png)



**ELU激活函数**

ELU（Exponential Linear Unit） 的提出同样也是针对解决 ReLU负数部分存在的问题，由Djork等人提出,被证实有较高的噪声鲁棒性。ELU激活函数对 x 小于零的情况采用类似指数计算的方式进行输出。与 ReLU 相比，ELU 有负值，这会使激活的平均值接近零。均值激活接近于零可以使学习更快，因为它们使梯度更接近自然梯度。 函数表达式为
$$
ELU(x)={\left\{\begin{array}{l l}{x}&{x \gt 0} \\ 
{\alpha(e^{x}-1)}&{x \leq 0}\end{array}\right.}
$$
![image-20240815165527270](D:\dev\php\magook\trunk\server\md\img\image-20240815165527270.png)

显然，ELU 具有 ReLU 的所有优点，并且：

- 没有 Dead ReLU 问题，输出的平均值接近 0，以 0 为中心；
- ELU 通过减少偏置偏移的影响，使正常梯度更接近于单位自然梯度，从而使均值向零加速学习；
- ELU 在较小的输入下会饱和至负值，从而减少前向传播的变异和信息。

一个小问题是它的**计算强度更高，计算量较大**。与 Leaky ReLU 类似，尽管理论上比 ReLU 要好，但目前在实践中没有充分的证据表明 ELU 总是比 ReLU 好。



**4、softplus 函数**

log 为对数，是求幂的逆运算
$$
a^x=N\space =>\space log_{a}N = x \\
如果\space a=10，则可以写成 \space lgN \\
如果\space a=e，则可以写成 \space lnN
$$
其图像为

![image-20240702144526486](D:\dev\php\magook\trunk\server\md\img\image-20240702144526486.png)



softplus 函数
$$
\zeta(x)=\log(1+e^{x})
$$
可以用来产生正态分布的 *β* 和 *σ* 参数，因为它的范围是 (0*,* *∞*)。当处理包含 sigmoid 函数的表达式时它也经常出现。softplus 函数名来源于它是另外一个函数的平滑（或 ‘‘软化’’）形式，这个函数是
$$
x^+ = max(0, x)
$$
![image-20240702143614695](D:\dev\php\magook\trunk\server\md\img\image-20240702143614695.png)

`特别注意`：在本书中为了书写方便，`log` 不写下标的话，其底数为 `e`

softplus 的导数恰好是sigmoid函数：
$$
f'(x)=\frac{e^{x}}{1+e^{x}}=\frac{1}{1+e^{-x}}=sigmoid(x)
$$




针对上面的 *σ*(*x*)  和 *ζ*(*x*) 函数，延伸出了一下特性：

![image-20240702150634004](D:\dev\php\magook\trunk\server\md\img\image-20240702150634004.png)



**5、softmax 激活函数**

softmax用于`多分类`过程中，它作用在输出层将每个神经元的输出，映射到 [0，1] 区间，而这些值的累加为1（归一化），从而来进行多分类。它是分类神经网络最后全连接层所接的激活函数。
$$
f(x) = \frac{e^x}{\sum_{n=1}^{k}e^n}
$$
如果有 k 个输入值，分母的意思是先对这些输入值进行e运算，然后求和。分子就是某个输入值的e运算。其实就是算权重。`其中 e 指数可以达到的效果是使大的更大小的更小`，我们就可以选取概率最大的结点作为我们的预测目标！

softmax 主要用于离散化概率分布。

| 激活函数 | Softmax      | Sigmoid    |
| -------- | ------------ | ---------- |
| 本质     | 离散概率分布 | 非线性映射 |
| 任务     | 多分类       | 二分类     |
| 定义域   | 某个一维向量 | 单个数值   |
| 值域     | [0, 1]       | （0, 1）   |
| 结果之和 | 一定为1      | 为某个正数 |



公式汇总

![image-20240815171047443](D:\dev\php\magook\trunk\server\md\img\image-20240815171047443.png)



#### 3.11、贝叶斯规则-很重要

贝叶斯规则也叫贝叶斯方程，贝叶斯公式，贝叶斯定理。

我们经常会需要在已知 *P*(y *|* x) 时计算 *P*(x *|* y)。幸运的是，如果还知道 *P*(x)，我们可以用 贝叶斯规则（Bayes’ rule）来实现这一目的：
$$
P(\mathbf{x}\mid\mathbf{y})={\frac{P(\mathbf{x})P(\mathbf{y}\mid\mathbf{x})}{P(\mathbf{y})}} \\
即 \\
P(\mathbf{x}\mid\mathbf{y})P(\mathbf{y})=P(\mathbf{y}\mid\mathbf{x})P(\mathbf{x})
$$
注意到 *P*(y) 出现在上面的公式中，它通常使用 下面公式来计算，所以我们并不需要事先知道 *P*(y) 的信息。
$$
P(\mathbf{y}) = \sum_{x}P(\mathbf{y}|\mathbf{x})P(\mathbf{x})
$$
贝叶斯规则可以从条件概率的定义直接推导得出，但我们最好记住这个公式的名字，因为很多文献通过名字来引用这个公式。这个公式是以牧师 Thomas Bayes的名字来命名的，他是第一个发现这个公式特例的人。这里介绍的一般形式由Pierre-Simon Laplace 独立发现。



![image-20240702220653012](D:\dev\php\magook\trunk\server\md\img\image-20240702220653012.png)



P(c) 为先验概率，表示每种类别分布的概率。

P(x|c) 为类条件概率，表示在某种类别前提下，某事发生的概率。

P(c|x) 为后验概率。表示某事发生了，并且它属于某一类别的概率。



有了这个后验概率，我们就可以对样本进行分类。后验概率越大，说明某事物属于这个类别的可能性越大，我们越有理由把它归到这个类别下。



举例说明：

**已知：**在夏季，某公园男性穿凉鞋的概率为1/2，女性穿凉鞋的概率为2/3，并且该公园中男女比例通常为2:1。

**问题：**若你在公园中随机遇到一个穿凉鞋的人，请问他的性别为男性或女性的概率分别为多少？



设 w1 = 男性，w2 = 女性，x = 穿脱鞋

可知，先验概率 P(w1) = 2/3，P(w2) = 1/3；类条件概率 P(x|w1) = 1/2，P(x|w2) = 2/3

所以穿脱鞋的概率：P(x) = P(x|w1) P(w1) + P(x|w2) P(w2)  = 5/9

所以
$$
P(w_{1}|x) = \frac{P(x|w_{1})P(w_{1})}{P(x)} = \frac{3}{5} \\
\\
P(w_{2}|x) = \frac{P(x|w_{2})P(w_{2})}{P(x)} = \frac{2}{5}
$$


若只考虑分类问题，只需要比较后验概率的大小，取值并不重要。



**朴素贝叶斯分类器**

![image-20240703092043055](D:\dev\php\magook\trunk\server\md\img\image-20240703092043055.png)





#### 3.12、连续型变量的技术细节



#### 3.13、信息论

信息论的基本想法是一个不太可能的事件居然发生了，要比一个非常可能的事件发生，能提供更多的信息。消息说：‘‘今天早上太阳升起’’ 信息量是如此之少以至于没有必要发送，但一条消息说：‘‘今天早上有日食’’ 信息量就很丰富。

![image-20240702161206098](D:\dev\php\magook\trunk\server\md\img\image-20240702161206098.png)

![image-20240702161943160](D:\dev\php\magook\trunk\server\md\img\image-20240702161943160.png)

![image-20240702162152575](D:\dev\php\magook\trunk\server\md\img\image-20240702162152575.png)

![image-20240702162948625](D:\dev\php\magook\trunk\server\md\img\image-20240702162948625.png)

![image-20240702163006860](D:\dev\php\magook\trunk\server\md\img\image-20240702163006860.png)

![image-20240702163020663](D:\dev\php\magook\trunk\server\md\img\image-20240702163020663.png)



#### 3.14、结构化概率模型

看不懂



---

#### 3.15、补充部分

##### 3.15.1、极大似然估计

极大似然估计属于信息论中的熵的概念，它的概念来源如下：

有两外形完全相同的箱子，甲箱中有99只白球1只黑球；乙箱中有99只黑球1只白球。一次实验取出1球，结果取出的是黑球。

问：黑球从哪个箱子中取出？

分析：人们的第一印象就是”此黑球最像是从乙箱中取出的“。这个推断符合人们的经验事实，“最像”就是“最大似然”之意。



总结起来，最大似然估计的目的就是：利用已知的样本结果，反推最有可能（最大概率）导致这样结果的参数值。即，面对一个数据样本集，我们可能知道用哪个模型，但是不知道模型的参数，于是我们假设它的参数有n种组合，我们将实验的结果带入参数中来计算概率，我们取计算结果最大的那一组参数作为模型的参数，但是这个可能不是百分之百对，我们称这个为极大似然估计。



举个例子：

我们知道，抛硬币得到正反的概率都是50%，但是假设我们不知道这个结论，我们可以使用极大似然估计法来估计出各自的概率。

我们假设正反的概率有：(0.1, 0.9), (0.7, 0.3), (0.8, 0.2), (0.6, 0.4)

我们做了一个实验，抛十次硬币，结果为 7正3反。

我们分别使用上面的四组概率分布来计算“抛十次硬币，结果为 7正3反”这个事件发生的概率。即
$$
p1 = 0.1^{7} * 0.9^{3} = 7.29E-8 \\
p2 = 0.7^{7} * 0.3^{3} = 0.0022235661 \\
p3 = 0.8^{7} * 0.2^{3} = 0.0016777216 \\
p4 = 0.6^{7} * 0.4^{3} = 0.0017915904
$$
从结果来看，p2 最大，我们选择第二个概率分布。



为什么会有极大似然估计？

但是在实际问题中并不都是这样幸运的，我们能获得的数据可能只有有限数目的样本数据，而先验概率和类条件概率都是未知的。根据仅有的样本数据进行分类时，一种可行的办法是我们需要先对先验概率和类条件概率进行估计，然后再套用贝叶斯分类器。

先验概率的估计较简单，1、每个样本所属的自然状态都是已知的（有监督学习）；2、依靠经验；3、用训练样本中各类出现的频率估计。

类条件概率的估计（非常难），原因包括：概率密度函数包含了一个随机变量的全部信息；样本数据可能不多；特征向量x的维度可能很大等等。总之要直接估计类条件概率的密度函数很难。解决的办法就是，把估计完全未知的概率密度转化为估计参数。这里就将概率密度估计问题转化为参数估计问题，极大似然估计就是一种参数估计方法。当然了，概率密度函数的选取很重要，模型正确，在样本区域无穷时，我们会得到较准确的估计值，如果模型都错了，那估计半天的参数，肯定也没啥意义了。



极大似然估计值

极大似然估计函数



最大似然估计的特点：

- 比其他估计方法更加简单；
- 收敛性：无偏或者渐近无偏，当样本数目增加时，收敛性质会更好；
- 如果假设的类条件概率模型正确，则通常能获得较好的结果。但如果假设模型出现偏差，将导致非常差的估计结果。



##### 3.15.2、其他

离散的数据对应的是分类的问题；连续的数据对应的是回归问题。

两个事件互斥是指这两个事件同时发生的概率为0

两个时间独立是指两个事件同时发生的概率为P(A)乘以P(B)