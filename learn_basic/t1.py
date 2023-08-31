########################################### 字符串 ########################################

print("hello world!")
print('Hello', 'world!', sep=' ')

# 声明变量
msg = "hello world!"
print(msg)

name = "ada lovelace"
print(name.title())  # Ada Lovelace
print(name.upper())  # ADA LOVELACE
print(name.lower())  # ada lovelace

print(msg+" "+name)  # 字符串拼接

print("Languages:\nPython\nC\nJavaScript")  # \n 换行

word = " this is word "
print(word.strip())  # 删除空白  lstrip, rstrip, strip

name = 'rao'
age = 10
print("hello %s, age %d" % (name, age))  # 格式化输出

print("hello {}, age {}".format(name, age))  # 格式化输出

print('我们'*3)  # 重复拼接3次：我们我们我们

############################################## 数值型 #####################################

print(100+100)
print(1/2)  # 0.5
print(3//2)  # 向下取整，结果为 1
print(2**3)  # 两个乘号表示乘方

age = 23
message = "Happy " + str(age) + "rd Birthday!"  # str 函数将非字符串对象转换成字符串对象
print(message)

# python2 中的除法：
# 3/2=1
# 3.0/2=1.5

############################################## 列表 list #####################################
# 列表是引用传递

# 列表的元素类型可以是混合的
mixlist = [12, 'rao', 0.13, [1, 2, 3]]

bicycles = ['trek', 'redline', 'cannondale', 'specialized']

print(bicycles)

print(bicycles[0].title())
print(bicycles[-1])  # 倒数第一个

bicycles.append("rao")  # 在末尾添加
print(bicycles)

bicycles.insert(0, "hhhh")  # 在特定索引之前添加
print(bicycles)

val = bicycles.pop(0)  # 剔除指定位置的元素，默认为最后一个
print(bicycles, val)

del bicycles[0]  # 剔除定位位置的元素
print(bicycles)

bicycles.remove("rao")  # 删除指定的值
print(bicycles)

bicycles.sort()  # 递增排序
print(bicycles)

bicycles.sort(reverse=True)  # 递减排序
print(bicycles)

bicycles2 = sorted(bicycles, reverse=False)  # 临时排序，不改变原List的顺序
print(bicycles2, bicycles)

bicycles.reverse()  # 列表顺序翻转，再次 reverse 即可恢复原状

print(bicycles)
print(len(bicycles))  # 列表长度

print('redline' in bicycles)  # True  判断是否在其中

# 遍历列表
for c in bicycles:
    print(c)

# 生成 range object
a = range(1, 10, 1)
print(a)  # range(1, 10)

for v in a:
    print(v)

# 从 range object 中得到 List
alist = list(a)

min(alist)
max(alist)
sum(alist)
set(alist)  # 返回一个集合，集合的特性是去掉重复元素

# 列表解析：根据已有列表，高效创建新列表的方式
#   语法：
#       [expression for iter_val in iterable]
#       [expression for iter_val in iterable if cond_expr]
#
# 快速从 range object 生成 list
blist = [value ** 2 for value in range(1, 5)]
print(blist)  # [1, 4, 9, 16]


# 列表可以直接相加
list1 = [1, 2, 3]
list2 = [1, 2, 4]
list = list1+list2  # [1, 2, 3, 1, 2, 4]

# join 操作，需要字符串
print('-'.join([str(v) for v in list1]))

# count 统计
list1.count(1)

# 使用 enumerate() 类型，将 list 转换成带索引的，从 0 开始
for index, value in enumerate(list1):
    print(index, '-', value)

# 变量的类型
typename = type(list1)

############################################## 切片 slice #####################################

players = ['charles', 'martina', 'michael', 'florence', 'eli']
# [start:end) 不包含右边，
# 此种方式得到的两个独立的对象，相互不会影响
players[0:3]  # ['charles', 'martina', 'michael']
players[:3]
players[1:]
players[-3:]  # 最后三个
playersb = players[:]  # 全部

players.append("xxx")
playersb.append("yyy")

print(players, playersb)
# ['charles', 'martina', 'michael', 'florence', 'eli', 'xxx'] ['charles', 'martina', 'michael', 'florence', 'eli', 'yyy']

# 如果不采用切片的形式来赋值，那么是会相互影响的，因为列表是引用的关系
playersc = players
players.append("aaa")
playersc.append("bbb")
print(players, playersc)
# ['charles', 'martina', 'michael', 'florence', 'eli', 'xxx', 'aaa', 'bbb'] ['charles', 'martina', 'michael', 'florence', 'eli', 'xxx', 'aaa', 'bbb']

############################################## 元组 tuple #####################################
# 元组类似于列表，但是数据不可修改
dimensions = (200, 50)
dimensions[0]
# dimensions[0] = 300  # Error
for dimension in dimensions:
    print(dimension)

# 虽然元组的元素不能修改，但是可以修改元组变量
dimensions = (300, 50)

# 注意，只有一个元素的元组最后需要有个逗号，否则会被当成小括号处理

############################################## if 语句 #####################################
# 判断 bool
names = ['aaa', 'bbb', 'ccc']
for name in names:
    if name == 'aaa':
        print('ok')
    else:
        print('!ok')

# 判断 in, not in
if 'ddd' in names:
    print('in')
else:
    print('not in')

# if...elif...else
num = 10
if num == 1:
    print('a')
elif num > 1:
    print('b')
else:
    print('c')

# 判断列表是否为空
if names:
    print('a')
else:
    print('b')

# python的这种缩进代码块模式，缩进块必须要有内容，否则报错
# pass 关键字，就可以满足这一点
if names:
    pass
else:
    print('b')

############################################## 字典 map #####################################
# 字典是没有顺序的
# 读取不存在的key会报错

alien_0 = {'color': 'green', 'points': 5}
print(alien_0['color'])
print(alien_0['points'])

len(alien_0)  # 有几个键值对，即长度

# 判断key是否存在
print("color" in alien_0.keys())  # True
print("color" in alien_0)  # True

alien_1 = {}
alien_1['color'] = 'green'
alien_1['points'] = 5

del alien_1['color']  # 删除指定元素
alien_1.clear('color')  # 删除指定元素
alien_1.clear()  # 清空
alien_1.pop('color')  # 删除指定元素

user_0 = {
    'username': 'efermi',
    'first': 'enrico',
    'last': 'fermi',
    'age': 12,
}

for key, value in user_0.items():
    print("\nKey: " + key)
    print("Value: " + str(value))

for key in user_0.keys():
    print("Key: " + key)

for val in user_0.values():
    print("Value: " + str(val))

# 按顺序遍历字典中的所有键
favorite_languages = {
    'jen': 'python',
    'sarah': 'c',
    'edward': 'ruby',
    'phil': 'python',
}
for name in sorted(favorite_languages.keys()):
    print(name.title() + ", thank you for taking the poll.")

aaa = [1, 2, 3, 1, 1, 5, 6]
print(set(aaa))

# 遍历字典中的所有键，去重
for val in set(favorite_languages.values()):
    print(val)


# 字典与字典，列表与列表，字典与列表 可以相互嵌套

############################################## 用户输入 #####################################
# 输入的都是字符串

message = input(
    "Tell me something, and I will repeat it back to you: ")  # 会阻塞，直到回车事件
print(message)

age = input("How old are you? ")
age = int(age)

# 还有 float()

############################################## while #####################################

current_number = 1
while current_number <= 5:
    print(current_number)
    current_number += 1


############################################## 逻辑运算符 #####################################
# and, or, not

# &, |, ~, ^, <<, >>

############################################## 常量 #####################################
MAX_LOG_OUTPUT = 6000

############################################## sprintf 风格 #####################################
# python 中没有相应的函数，使用的是 % 来处理
# 格式规则是通用的

sprintf_str = ('I am %s, age %d, weight %.2f kg') % ('rao', 12, 50)
# I am rao, age 12, weight 50.00 kg

print(sprintf_str)

############################################## None #####################################
# None是一个特殊的常量。
# None和False不同。
# None不是0。
# None不是空字符串。
# None和任何其他的数据类型比较永远返回False。
# None有自己的数据类型NoneType。
# 你可以将None复制给任何变量，但是你不能创建其他NoneType对象。
# 类似于C中的 NULL，和Go中的 Nil

############################################## help #####################################
# 进入 Python 交互程序
# help('keywords') 列出所以关键字
# help('del') 查看具体关键字的说明
