'''
操作文件
'''


# 手动关闭文件，存在的问题就是中途出bug导致文件没有关闭
import csv
fo = open('files/data.txt')
contents = fo.read()
print(contents.rstrip())  # read读出来的内容最后会多出一个空字符串，需要处理
fo.close()

print("--------------------------------------")

# with 关键字会在代码块结束后关闭文件，即便出现bug也能做到关闭文件
with open('files/data.txt') as file_object:
    contents = file_object.read()
    print(contents.rstrip())

print("--------------------------------------")

# 逐行读取
with open('files/data.txt') as fo:
    for line in fo:
        print(line.rstrip())  # 每一行都有一个换行符，而print会打印出换行符

print("--------------------------------------")

with open('files/data.txt') as fo:
    lines = fo.readlines()

# 居然能访问到 with 代码块中的变量
for line in lines:
    print(line.rstrip())

print("--------------------------------------")

# 写文件
#
# 打开文件的模式：r 只读, w 只写, a 附加模式, r+ 读写
#
# 如果你要写入的文件不存在，函数open()将自动创建它。然而，以写入（'w'或'r+'）模式打开文件时千万要小心，
# 因为如果指定的文件已经存在，Python将在返回文件对象前清空该文件。如果不希望清空，需要使用附加模式('a')打开
#
with open('files/data_2.txt', 'a') as fo:
    fo.write("write some thing...\n")
    fo.write(str(100)+"\n")
    fo.writelines(["write some thing...\n"])

print("--------------------------------------")

# 文件不存在的异常
filename = 'files/data_3.txt'
try:
    with open(filename) as file_object:
        contents = file_object.read()
        print(contents.rstrip())
except FileNotFoundError:
    msg = "Sorry, the file " + filename + " does not exist."
    print(msg)

print("--------------------------------------")

# 读取 csv 文件
# import csv


with open('files/data.csv') as of:
    reader = csv.reader(of)

    # next 从迭代器中取出一个元素，具体到此处，每一个元素是一行。
    # 每次调用 next，游标会下移一个
    # 而 csv 每一行是由逗号分隔的数据，因此 line 是一个列表类型
    # ['rao', '12', '170', '60']
    line = next(reader)
    print(line)

print("--------------------------------------")
