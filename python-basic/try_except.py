# 异常处理
# 关键字： try, except, else, finally, raise, pass, as
'''

try:
    尝试执行的代码
    pass
except 错误类型1：
    针对错误类型1，对应的代码处理
    pass
except (错误类型2，错误类型3)：
    针对错误类型2 和 3，对应的代码处理
    pass
except Exception as result:
    打印错误信息
    print("未知错误 %s" % result)
else:
    没有异常才会执行的代码
    pass
finally：
    无论是否有异常，都会执行的代码
    print("无论是否有异常，都会执行的代码")

'''


def except1():
    try:
        print(5/0)
    except ZeroDivisionError as zr:
        print("You can't divide by zero!")

        # 获取错误信息，元组 tuple
        print(zr.args)
        print(zr.with_traceback)

        # 获取文件名
        print(f'error file:{zr.__traceback__.tb_frame.f_globals["__file__"]}')
        # 获取行号
        print(f"error line:{zr.__traceback__.tb_lineno}")
    else:
        print('...')
    finally:
        # 最终都会执行
        print('ok')


def except2():
    '''捕获通用异常'''
    try:
        print(5/0)
    except Exception as e:
        print(('error: %s, in file %s, on line %d') %
              (','.join(list(e.args)), e.__traceback__.tb_frame.f_globals["__file__"], e.__traceback__.tb_lineno))


def throwException():
    '''抛出异常'''
    raise Exception('some exception')


except2()
