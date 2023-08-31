import dog

# 导入整个模块

my_hsq = dog.Hashiqi("xiao", 2)
my_hsq.sit("here")
my_hsq.roll_over()
my_hsq.eat()


class Mal(dog.Dog):
    def __init__(self, name, age):
        super().__init__(name, age)


mal = Mal("xxx", 2)
mal.sit()
