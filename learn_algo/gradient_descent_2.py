import matplotlib.pyplot as plt
import numpy as np


'''
二维空间梯度下降法

# https://mp.weixin.qq.com/s/IAkVWrWGMZCGAwxIhRysLw
'''


def fx(x):
    # fx的函数值
    return x**2


def gradient_descent():
    # 定义梯度下降算法
    times = 1000  # 迭代次数
    alpha = 0.05  # 学习率
    x = 10  # 设定x的初始值
    x_axis = np.linspace(-10, 10)  # 设定x轴的坐标系
    fig = plt.figure(1, figsize=(5, 5))  # 设定画布大小
    ax = fig.add_subplot(1, 1, 1)  # 设定画布内只有一个图
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.plot(x_axis, fx(x_axis))  # 作图

    for i in range(times):
        x1 = x
        y1 = fx(x)
        print("第%d次迭代：x=%f，y=%f" % (i + 1, x, y1))
        j = 2 * x
        if j <= 0.1:
            break
        x = x - alpha * j
        y = fx(x)
        ax.plot([x1, x], [y1, y], 'ko', lw=1, ls='-', color='coral')

    plt.show()


if __name__ == "__main__":
    gradient_descent()
