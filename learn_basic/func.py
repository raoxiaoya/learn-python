"""
函数说明块
"""

#############################################################
'''
新的函数会把旧的函数覆盖，函数名字相同就会覆盖，参数部分可以相同也可以不同
'''




from typing import List, Dict, Union, Iterable, Any
import plot.poltmet as pm
import module1
def greet_user():
    """显示简单的问候语"""
    print("hello")


greet_user()  # 正常执行


def greet_user(username, gener='male'):
    """显示简单的问候语"""
    print("Hello, " + username.title() + "!")


greet_user('jesse')  # 正常执行

greet_user(username='jesse')  # 正常执行

# greet_user()  # 报错，没有传递参数


def greet_user(username, gener='male'):
    """显示简单的问候语"""
    return "hell "+username + ": "+gener


print(greet_user('rao'))

print("--------------------------------------")

#############################################################
# 列表是引用传递

fromList = [1, 2, 3, 4]
toList = []


def move(list1, list2):
    while list1:
        i = list1.pop()
        list2.append(i)


move(fromList, toList)

# move(fromList[:], toList) # 传入的是切片，则不会影响原来的

print(fromList, toList)  # [] [4, 3, 2, 1]

print("--------------------------------------")

#############################################################
# 传递任意数量的实参
# 得到的是元组
# 使用 list() 函数转换成列表


def make_pizza(*toppings):
    """打印顾客点的所有配料"""
    arglist = list(toppings)
    print(toppings)


# ('pepperoni',)
make_pizza('pepperoni')

# ('mushrooms', 'green peppers', 'extra cheese')
make_pizza('mushrooms', 'green peppers', 'extra cheese')


print("--------------------------------------")

#############################################################
# 任意数量的键值对参数，此参数必须放在最后
# 得到的是字典
# 注意传参形式


def rand_num_kv(first_name, last_name, **toppings):
    info = {}
    info['first_name'] = first_name
    info['last_name'] = last_name
    for k, v in toppings.items():
        info[k] = v

    return info


re = rand_num_kv("rao", "ya", age=10, score=100)
print(re)

print("--------------------------------------")

#############################################################


def many(*args, scalex=True, scaley=True, data=None, **kwargs):
    '''这也是可以的, args 为元组, kwargs 为字典, 但是字典的key要避开scalex和scaley'''
    pass


print("--------------------------------------")

#############################################################
# 函数可以返回多个值，但是实际上得到是一个 tuple，即函数返回值接收者只能是一个变量


def morereturn():
    return 1, 'rao', [2, 3]


print("--------------------------------------")

#############################################################
# 引入模块，模块名就是文件名
# import module1
# 设置别名：import module1 as p
#
# 调用时：module1.func_name

# 导入特定的函数
# from module_name import function_name
# from module_name import function_0, function_1, function_2
# 设置别名：from module1 import make_pizza as mp
#
# 调用时：function_0()
#


re = module1.rand_num_kv2("rao2", "ya2", age=10, score=100)
print(re)

print("--------------------------------------")

# 比如要导入某个目录下的文件，plot/poltmet.py
# import plot.poltmet as pm
pm.show()

print("--------------------------------------")

#############################################################

'''
函数形参指定数据类型，一方面限制参数类型，另一方面，编辑器才有对应类型的语法提示

复合类型需要 from typing import List, Dict, Union

Union 表示多种数据类型

有默认值的放在后面

Dict[str, Any] 表示key是str, val 是任意类型

任意类型使用 Any
object 和 Any 是有区别的，使用 object 示意一个值可以类型安全地兼容任何类型，使用 Any 示意一个值地类型是动态定义的

Optional[int] 等价于 Union[int, None]，Optional[] 里面只能写一个数据类型

也可以指定任意class作为类型

使用 -> str 来定义返回值类型
'''


def f22(name: str, age: int, lst: List[str], ps: Dict[str, Any], dos: Iterable, ban: Union[str, int] = None, sit: bool = False) -> str:
    return


print("--------------------------------------")
