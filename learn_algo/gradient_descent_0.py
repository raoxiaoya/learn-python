import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
批量梯度下降法

没有添加截距，有问题

https://mp.weixin.qq.com/s/gizUmdNcvSyUFhSRWVjXtA
'''

# 生成模拟数据
np.random.seed(42)
num_samples = 100
# 矩阵 10X3，假设3个特征，rand函数生成符合标准正态分布的样本，取值在[0, 1)
X = np.random.rand(num_samples, 3) * 100
# print(X.shape) # (10, 3)
noise = np.random.randn(num_samples) * 10  # 矩阵 10X1，生成随机数，噪音数据
coefficients = [3.5, 2.1, -1.8]  # 权重，3X1

# 线性关系 y=ax1+bx2+cx3+5+k
# dot 矩阵点乘
y = X.dot(coefficients) + 5 + noise  # 10X1

# 数据预处理
# 标准化，即转换为均值为0，方差为1的正态分布。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


def compute_cost(X, y, theta):
    # 梯度下降法实现线性回归
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost


def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = X.dot(theta)  # [0. 0. 0. 0. 0. 0. 0. 0.]
        errors = predictions - y  # 10X1
        gradients = (1/m) * X.T.dot(errors)  # 3X1，计算梯度，T为矩阵的转置
        theta -= learning_rate * gradients
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history


# 初始化参数
theta = np.zeros(X_train.shape[1])  # 3X1，权重初始值，[0,0,0]
learning_rate = 0.05  # 步长
iterations = 100

# 训练模型
theta_opt, cost_history = gradient_descent(
    X_train, y_train, theta, learning_rate, iterations)

# 打印模型参数
print("Optimized Theta Parameters:", theta_opt)
print("Min Lost:", cost_history[len(cost_history)-1])

# 模型评估
# y_train_predictions = X_train.dot(theta_opt)
y_test_predictions = X_test.dot(theta_opt)


def showLost():
    # 可视化损失函数下降过程
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), cost_history, color='blue')
    plt.title('Cost Function Decrease over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


def showResult():
    # 可视化预测结果与真实值的对比
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_predictions, color='red',
                alpha=0.6, label='Predictions vs. Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             color='blue', label='Ideal Fit Line')
    plt.title('Predictions vs. Actual Values')
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.legend()
    plt.grid(True)
    plt.show()


def showDistribute():
    # 数据分布和预测误差分析
    residuals = y_test - y_test_predictions
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    showDistribute()
