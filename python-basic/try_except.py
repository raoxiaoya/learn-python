# 异常处理

try:
    print(5/0)
except ZeroDivisionError:
    print("You can't divide by zero!")
else:
    print('...')

print('ok')
