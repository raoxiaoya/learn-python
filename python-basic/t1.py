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

# 列表解析：快速从 range object 生成 list
blist = [value ** 2 for value in range(1, 5)]
print(blist)  # [1, 4, 9, 16]

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

############################################## 逻辑运算符 #####################################
# and, or,
