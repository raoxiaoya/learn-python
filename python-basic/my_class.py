from dog import Dog, Hashiqi

# 导入模块中的多个类

my_dog = Dog("rao", 10)
print(my_dog)  # <__main__.Dog object at 0x0000029D42C407D0>

# 没有返回值的方法，print 会打印出 None
print(my_dog.sit(), my_dog.roll_over())

print(my_dog.name)
print(my_dog.age)

print("--------------------------------------")

##########################################################

my_hsq = Hashiqi("xiao", 2)
my_hsq.sit("here")
my_hsq.roll_over()
my_hsq.eat()

print("--------------------------------------")


class Keji(Dog):
    info = 'info...'

    def __init__(self, name, age):
        super().__init__(name, age)


keji = Keji("xxx", 2)
keji.sit()
