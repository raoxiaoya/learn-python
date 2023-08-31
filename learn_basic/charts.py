
'''
本例在 jupyter中运行，文件为 charts.ipynb
'''

import pygal
import random
import matplotlib.pyplot as plt

############################ matplotlib.pyplot ###########################

'''
基于 matplotlib 开发图表图形

官网 http://matplotlib.org/

plot 图表
'''


def f1():
    squares = [1, 4, 9, 16, 25]
    plt.plot(squares)
    # 可以通过 gui 来展示，也可以在 jupyter notebook 中展示
    plt.show()


def f2():
    squares = [1, 4, 9, 16, 25]

    # 线宽为 5
    plt.plot(squares, linewidth=5)

    # 设置图表标题，并给坐标轴加上标签
    plt.title("Square Numbers", fontsize=24)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Square of Value", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', labelsize=14)

    plt.show()


def f3():
    xlist = [0, 1, 2, 3, 4, 5]
    ylist = [0, 1, 4, 9, 16, 25]

    # 线宽为 5
    plt.plot(xlist, ylist, linewidth=5)

    # 设置图表标题，并给坐标轴加上标签
    plt.title("Square Numbers", fontsize=24)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Square of Value", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', labelsize=14)

    plt.show()


'''
scatter 散点图
'''


def f4():
    dot = [1, 2, 3, 4, 5]
    index = 1
    for d in dot:
        plt.scatter(index, d)
        index = index+1

    # 设置图表标题并给坐标轴加上标签
    plt.title("Square Numbers", fontsize=24)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Square of Value", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.show()


def f5():
    xlist = [0, 1, 2, 3, 4, 5]
    ylist = [0, 1, 4, 9, 16, 25]

    # s 为点的尺寸
    plt.scatter(xlist, ylist, s=100)

    # 设置图表标题并给坐标轴加上标签
    plt.title("Square Numbers", fontsize=24)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Square of Value", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.show()


def f6():
    xlist = list(range(1, 1001))
    ylist = [x**2 for x in xlist]

    # s 为点的尺寸
    plt.scatter(xlist, ylist, s=100)

    # 设置图表标题并给坐标轴加上标签
    plt.title("Square Numbers", fontsize=24)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Square of Value", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='major', labelsize=14)

    # 设置每个坐标轴的取值范围
    plt.axis([0, 1100, 0, 1100000])

    plt.show()


def f6():
    xlist = [0, 1, 2, 3, 4, 5]
    ylist = [0, 1, 4, 9, 16, 25]

    # s 为点的尺寸
    # 点的颜色默认为蓝色点和黑色轮廓
    # edgecolor 参数设置点的轮廓颜色，字符串类型，none 是不需要轮廓
    # c = 'red' 设置点的填充颜色
    # c=(0, 0, 0.8) 也可以使用 RGB 颜色，取值0到1
    plt.scatter(xlist, ylist, s=100, edgecolor='none', c='red')

    # 设置图表标题并给坐标轴加上标签
    plt.title("Square Numbers", fontsize=24)
    plt.xlabel("Value", fontsize=14)
    plt.ylabel("Square of Value", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.show()


def f7():
    '''
    颜色映射 （colormap）是一系列颜色，它们从起始颜色渐变到结束颜色。
    在可视化中，颜色映射用于突出数据的规律，例如，你可能用较浅的颜色来显
    示较小的值，并使用较深的颜色来显示较大的值。

    plt.cm.Blues 为自带的颜色映射map
    '''
    x_values = list(range(1001))
    y_values = [x**2 for x in x_values]
    plt.scatter(x_values, y_values, c=y_values,
                cmap=plt.cm.Blues, edgecolor='none', s=40)
    plt.show()


def f8():
    x_values = list(range(1001))
    y_values = [x**2 for x in x_values]
    plt.scatter(x_values, y_values, c=y_values,
                cmap=plt.cm.Blues, edgecolor='none', s=40)

    # bbox_inches 实参指定将图表多余的空白区域裁剪掉。如果要保留图表周围多余的空白区域，可省略这个实参。
    plt.savefig('squares_plot.png', bbox_inches='tight')


def fill_walk():
    """计算随机漫步包含的所有点"""
    num_points = 5000
    x_values = [0]
    y_values = [0]

    # 不断漫步，直到列表达到指定的长度
    while len(x_values) < num_points:
        # 决定前进方向以及沿这个方向前进的距离
        x_direction = random.choice([1, -1])
        x_distance = random.choice([0, 1, 2, 3, 4])
        x_step = x_direction * x_distance

        y_direction = random.choice([1, -1])
        y_distance = random.choice([0, 1, 2, 3, 4])
        y_step = y_direction * y_distance

        # 拒绝原地踏步
        if x_step == 0 and y_step == 0:
            continue

        # 计算下一个点的x和y值
        next_x = x_values[-1] + x_step
        next_y = y_values[-1] + y_step
        x_values.append(next_x)
        y_values.append(next_y)

    plt.scatter(x_values, y_values, s=100)

    plt.axes().get_xaxis().set_visible(False)  # 隐藏坐标轴
    plt.axes().get_yaxis().set_visible(False)  # 隐藏坐标轴

    # 用于指定图表的宽度、高度、分辨率和背景色
    #
    # figsize 指定宽度、高度单位为英寸
    # dpi=128 指定分辨率
    plt.figure(figsize=(10, 6))

    plt.show()


############################ pypgal ###########################
'''
pip install pygal
官网 http://www.pygal.org/
生成的是 svg 图像，浏览器中打开
'''


def gal():
    # 掷几次骰子，并将结果存储在一个列表中
    results = []
    for roll_num in range(100):
        result = random.randint(1, 6)
        results.append(result)

    # 统计
    frequencies = []
    for index in range(1, 7):
        frequencies.append(results.count(index))

    # 对结果进行可视化
    hist = pygal.Bar()
    hist.title = "Results of rolling one D6 1000 times."
    hist.x_labels = ['1', '2', '3', '4', '5', '6']
    hist.x_title = "Result"
    hist.y_title = "Frequency of Result"
    hist.add('D6', frequencies)
    # hist.render_in_browser()
    hist.render_to_file('die_visual.svg')
