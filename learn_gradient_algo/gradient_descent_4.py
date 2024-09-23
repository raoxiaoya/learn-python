import numpy as np
import matplotlib.pyplot as plt


def predict(θ_0, θ_1, x):
    # 预测目标变量y的值
    return θ_0 + θ_1*x


def loop(m, θ_0, θ_1, x, y):
    # 遍历整个样本数据，计算偏差，使用批量梯度下降法
    sum0 = 0
    sum1 = 0
    error = 0
    for i in range(m):
        a = predict(θ_0, θ_1, x[i]) - y[i]  # 误差
        sum0 += a
        sum1 += a * x[i]
        error += a*a  # 误差的评平方

    gradient_0 = sum0 / m  # θ_0 的梯度
    gradient_1 = sum1 / m  # θ_1 的梯度
    error = error / m

    return gradient_0, gradient_1, error


def batch_gradient_descent(x, y, θ_0, θ_1, alpha, m):
    # 批量梯度下降法进行更新θ的值
    # 设定一个阀值，当梯度的绝对值小于0.001时即不再更新了
    while True:
        res = loop(m, θ_0, θ_1, x, y)
        θ_0 = θ_0 - alpha*res[0]
        θ_1 = θ_1 - alpha*res[1]
        if abs(res[1]) <= 0.01:
            break

    return θ_0, θ_1, res[2]


# 样本数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
y = [3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 10, 11, 13, 13, 16, 17, 16, 17, 18, 20]
m = 20  # 样本数量
alpha = 0.01  # 学习率
θ_0 = 1  # 初始化θ_0的值
θ_1 = 1  # 初始化θ_1的值

res = batch_gradient_descent(x, y, θ_0, θ_1, alpha, m)
θ_0 = res[0]
θ_1 = res[1]
error = res[2]
print("The θ_0 is %f, The θ_1 is %f, The The Mean Squared Error is %f " %
      (θ_0, θ_1, error))

plt.figure(figsize=(6, 4))  # 新建一个画布
plt.scatter(x, y, label='y')  # 绘制样本散点图
plt.xlim(0, 21)  # x轴范围
plt.ylim(0, 22)  # y轴范围
plt.xlabel('x', fontsize=20)  # x轴标签
plt.ylabel('y', fontsize=20)  # y轴标签

x = np.array(x)
y_predict = np.array(θ_0 + θ_1*x)
plt.plot(x, y_predict, color='red')  # 绘制拟合的函数图
plt.show()
