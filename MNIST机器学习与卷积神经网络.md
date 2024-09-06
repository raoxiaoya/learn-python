MNIST机器学习手写数字识别



https://www.bilibili.com/video/BV1y7411v7zd/?spm_id_from=333.337.search-card.all.click&vd_source=0c75dc193ee55511d0515b3a8c375bd0



两个示例

https://mp.weixin.qq.com/s/FwgabUUSkcCd-gWHnBnL9w

https://mp.weixin.qq.com/s/zuOVHVVjfEYCY39jUTLpvg



在深度学习中，“Hello World”通常指的是一个简单的示例，展示了如何使用深度学习框架来解决一个基本的问题，最经典的“Hello World”示例是使用神经网络来识别手写数字，这个问题通常被称为MNIST手写数字识别问题，在这个问题中，目标是训练一个卷积神经网络CNN，使其能够准确地识别手写数字图像中的数字，当然接下来的代码是通过调用相关库进行实现。



这个问题之所以成为“Hello World”，是因为它是深度学习中最简单、最基础的问题之一，很多人在学习深度学习时会从这个问题开始，通过解决这个问题，可以学会如何搭建神经网络、准备数据、进行训练和评估模型的性能等基本技能。



**关于 keras 中的 Sequential 函数**

它是一个用来构建神经网络模型的对象，包括全连接神经网络、卷积神经网络(CNN)、循环神经网络(RNN)、等等，通过堆叠许多层，构建出深度神经网络。

Sequential模型的核心操作是添加layers（图层），以下展示如何将一些最流行的图层添加到模型中：

```python
model = Sequential()

# 卷积层
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(28,28,1)))
# 最大池化层
model.add(MaxPooling2D(pool_size=(2, 2)))
# 全连接层
model.add(Dense(256, activation='relu'))
# Dropout层
model.add(Dropout(0.5))
# Flatten展平层，即一维向量
model.add(Flatten(input_shape=(28, 28)))
```



**python代码实现1-使用普通的神经网络**

**1. 数据加载**

```python
import tensorflow as tf
# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist 
# 加载训练数据和测试数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 打印数据集的维度信息
print("训练数据集维度:", x_train.shape)
print("训练标签集维度:", y_train.shape)
print("测试数据集维度:", x_test.shape)
print("测试标签集维度:", y_test.shape)
```

```bash
训练数据集维度: (60000, 28, 28)
训练标签集维度: (60000,)
测试数据集维度: (10000, 28, 28)
测试标签集维度: (10000,)
```

MNIST 数据集中的图像已经经过预处理，转换为了数值化的形式，具体来说，每张图像都是一个灰度图，表示为 28x28 的矩阵，其中每个元素代表了对应位置的像素强度值。



如果MNIST数据集不存在就会自动下载，其地址为 https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz



**2.数据预处理**

```python
# 把数值转换到0到1
x_train, x_test = x_train / 255.0, x_test / 255.0
```

像素的取值范围是0到255，表示像素的灰度值，其中0表示黑色，255表示白色，对数据进行预处理 加速模型的训练，并且有助于模型更好地收敛

**3.构建模型**

```python
# 构建神经网络模型
model = tf.keras.models.Sequential([ 
    # 将二维的图像展平为一维向量 
    tf.keras.layers.Flatten(input_shape=(28, 28)),   
    # 添加两个具有16个神经元的全连接层，激活函数为ReLU
    tf.keras.layers.Dense(16, activation='relu'),      
    tf.keras.layers.Dense(16, activation='relu'), 
    # 添加一个Dropout层，防止过拟合 正则化有20%的神经元被丢弃以减少过拟合
    tf.keras.layers.Dropout(0.2),
    # 添加一个具有10个神经元的输出层，激活函数为softmax
    tf.keras.layers.Dense(10, activation='softmax')  
])
# 编译模型
model.compile(
    optimizer='adam', # 优化器   
    loss='sparse_categorical_crossentropy', # 损失函数               
    metrics=['accuracy'] # 评价指标：准确率
) 
# 打印模型的结构信息
model.summary()
```

```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dense (Dense)                (None, 16)                12560
_________________________________________________________________
dense_1 (Dense)              (None, 16)                272
_________________________________________________________________
dropout (Dropout)            (None, 16)                0
_________________________________________________________________
dense_2 (Dense)              (None, 10)                170
=================================================================
Total params: 13,002
Trainable params: 13,002
Non-trainable params: 0
_________________________________________________________________
```

**模型解释：**

- Flatten 层: 这是输入层，用于将输入的二维图像数据展平成一维向量，输入图像的尺寸为 28x28 像素，所以展平后的向量长度为 784

- Dense 层: 这是第一个隐藏层，包含 16 个神经元，该层是全连接层，每个神经元与上一层中的所有神经元相连，参数数量为 784 (输入向量长度) * 16 (神经元数量) + 16 (偏置项) = 12560

- Dense 层: 这是第二个隐藏层，同样包含 16 个神经元，参数数量为 16 (上一层神经元数量) * 16 (神经元数量) + 16 (偏置项) = 272

- Dropout 层: 这是一个 Dropout 层，用于在训练过程中随机将一部分神经元的输出置为零，以防止过拟合，在本模型中，Dropout 层的输出形状与上一层相同，即 (None, 16)

- Dense 层: 这是输出层，包含 10 个神经元，对应着 10 个类别（0 到 9 的数字），参数数量为 16 (上一层神经元数量) * 10 (神经元数量) + 10 (偏置项) = 170

总参数数量为13002，其中所有参数都是可训练的，模型的训练目标是最小化损失函数，使得模型能够准确地预测输入图像对应的数字类别。



关于损失函数`sparse_categorical_crossentropy`与`categorical_crossentropy`，虽然都是交叉熵，但是计算格式不一样，前者的targets是数字编码，比如`1, 2, 3`，后者是one-hot编码，比如`[0, 1, 0]`。他们都应用于多分类问题。在这个例子中，我们的样本中每一个图像的标签是0-9的数值，因此选择`sparse_categorical_crossentropy`。

数值转换成one-hot：对于一个数9，将其转换成10位的one-hot向量为`[0,0,0,0,0,0,0,0,0,1]`



**4.训练模型**

```python
import matplotlib.pyplot as plt
# 记录训练过程中的损失值
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))
# 绘制损失曲线
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

![image-20240906115901155](D:\dev\php\magook\trunk\server\md\img\image-20240906115901155.png)

**5.模型评价以及模型保存**

```python
# 在测试集上评估模型性能
test_loss, test_acc = model.evaluate(x_test, y_test)
print("测试集损失:", test_loss)
print("测试集准确率:", test_acc)
# 保存模型
model.save('model.keras')
```



**6.模型调用识别数字**

![图片](https://mmbiz.qpic.cn/mmbiz_png/cCtGVD6h9medUfD2dmZdcBBJs4q8IKlCYDmnt47pHkhI8odN1hXxTQypDRldBbeMD1JRgP2jBQ3o9iaeHJ3QEyA/640?wx_fmt=png&from=appmsg&tp=wxpic&wxfrom=5&wx_lazy=1&wx_co=1)

加载一个已经训练好的神经网络模型，该模型用于识别手写数字，然后读取一张包含手写数字的图像（上图），并对图像进行预处理，使其与模型的输入格式相匹配，最后通过该模型对预处理后的图像进行预测，输出预测的手写数字

```python
import tensorflow as tf
import cv2
import numpy as np
# 加载训练好的模型
model = tf.keras.models.load_model('model.keras')
# 读取现实中的手写数字图像
image = cv2.imread('handwritten_digit.jpg', cv2.IMREAD_GRAYSCALE)
# 调整图像尺寸为模型输入的大小（28x28）
image_resized = cv2.resize(image, (28, 28))
# 对图像进行归一化处理
image_normalized = image_resized / 255.0
# 将图像转换为模型所需的形状 (1, 28, 28, 1)
image_input = np.expand_dims(image_normalized, axis=0)
image_input = np.expand_dims(image_input, axis=-1)
prediction = model.predict(image_input)
predicted_digit = np.argmax(prediction)
print("预测结果:", predicted_digit)
```



**python代码实现2-使用卷积神经网络**

CNN的通用架构：`卷积-池化-卷积-池化-全连接-全连接-输出`

```python
import matplotlib.pyplot as plt
import tensorflow as tf

# 1、构建CNN模型
# 构建一个最基础的连续的模型，所谓连续，就是一层接着一层
model = tf.keras.models.Sequential()
# 第一层为一个卷积，卷积核大小为(3,3), 输出通道32，使用 relu 作为激活函数
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 第二层为一个最大池化层，池化核为（2,2)
# 最大池化的作用，是取出池化核（2,2）范围内最大的像素点代表该区域
# 可减少数据量，降低运算量。
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 又经过一个（3,3）的卷积，输出通道变为64，也就是提取了64个特征。
# 同样为 relu 激活函数
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
# 上面通道数增大，运算量增大，此处再加一个最大池化，降低运算
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# dropout 随机设置一部分神经元的权值为零，在训练时用于防止过拟合
# 这里设置25%的神经元权值为零
model.add(tf.keras.layers.Dropout(0.25))
# 将结果展平成1维的向量
model.add(tf.keras.layers.Flatten())
# 增加一个全连接层，用来进一步特征融合
model.add(tf.keras.layers.Dense(128, activation='relu'))
# 再设置一个dropout层，将50%的神经元权值为零，防止过拟合
# 由于一般的神经元处于关闭状态，这样也可以加速训练
model.add(tf.keras.layers.Dropout(0.5))
# 最后添加一个全连接softmax激活，输出10个分类，分别对应0-9 这10个数字
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 编译上述构建好的神经网络模型
# 指定优化器为 rmsprop
# 制定损失函数为交叉熵损失
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# 打印模型的结构信息
model.summary()

# 2、处理数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 给标签增加维度,使其满足模型的需要 
# 原始标签，比如训练集标签的维度信息是[60000, 28, 28, 1]
X_train = x_train.reshape(60000, 28, 28, 1)
X_test = x_test.reshape(10000, 28, 28, 1)
# 特征转换为one-hot编码
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# 3、开始训练              
model.fit(
    X_train, y_train,  # 指定训练特征集和训练标签集
    validation_split=0.3,  # 部分训练集数据拆分成验证集
    epochs=5,  # 训练轮次为5轮
    batch_size=128)  # 以128为批量进行训练

# 在测试集上进行模型评估
score = model.evaluate(X_test, y_test)
print('测试集预测准确率:', score[1])  #  打印测试集上的预测准确率


#  预测验证集第一个数据
pred = model.predict(X_test[0].reshape(1, 28, 28, 1))
# 把one-hot码转换为数字
print(pred[0], "转换一下格式得到：", pred.argmax())
# 导入绘图工具包
# 输出这个图片
plt.imshow(X_test[0].reshape(28, 28), cmap='Greys')

```

模型信息



```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
dropout (Dropout)            (None, 5, 5, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
dense (Dense)                (None, 128)               204928
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 225,034
Trainable params: 225,034
Non-trainable params: 0
_________________________________________________________________
```



to_categorical 的作用是将样本标签转为 one-hot 编码，而 one-hot  编码的作用是可以对于类别更好的计算概率或得分。这个例子中，数字 0-9 转换为的独热编码为：

```bash
array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]]
```



**卷积核的通道数以及参数量的计算**

卷积神经网络的原理：https://www.bilibili.com/video/BV1f54y1f7rs/?spm_id_from=333.788&vd_source=dd488f2825c3a352e192887d5d63e429

卷积核可以是多层立体的，就像魔方那样，层数就是深度也叫通道数，比如RGB图像有三个通道，我们也应该用三层的卷积核去扫它，每一层负责扫一个通道。

所以卷积核一般不用设置输入的通道数，因为它与输入图像的通道数一样。

但是需要设置输出结果的通道数，比如你可以将RGB三通道图像经过卷积核变成32通道的图像。通道越多相当于升维了，便于提取特征。

![image-20240906151300918](D:\dev\php\magook\trunk\server\md\img\image-20240906151300918.png)

卷积层的参数量跟卷积核有关。我们先来看看多层卷积核是怎么计算的，此处以三通道为例。

![image-20240906152359129](D:\dev\php\magook\trunk\server\md\img\image-20240906152359129.png)

![image-20240906154046640](D:\dev\php\magook\trunk\server\md\img\image-20240906154046640.png)

从图中可以看出，一个卷积核的输出结果为一个通道，其参数量：`N = 输入的通道数 * 核宽 * 核高 + 1`，这个`1`表示偏置量。如果输出的通道数为P，那么就需要P个卷积核，参数量就是`P*N`。总结下来就是

```bash
L = P * (D * U * V + 1)
P：输出通道数
D：输入通道数
U：核宽
V：核高
```



于是示例2中的`conv2D`层的参数计算如下：

```bash
32 * (1 * 3 * 3 + 1) = 320
64 * (32 * 3 * 3 + 1) = 18496
```



**可视化MNIST的运行过程**

https://tensorspace.org/index.html

![image-20240906161235863](D:\dev\php\magook\trunk\server\md\img\image-20240906161235863.png)
