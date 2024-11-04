import tensorflow as tf

# print(tf.__version__) # 2.4.0

# 查看物理设备
cpus = tf.config.experimental.list_physical_devices('CPU') # 显示CPU信息
gpus = tf.config.experimental.list_physical_devices('GPU') # 显示GPU信息
print(cpus)
print(gpus)

# 指定设备运行代码
with tf.device('/cpu:0'):
    a = tf.random.normal([10000, 1000])
    b = tf.random.normal([1000, 2000])
    c = tf.matmul(a, b)
    print(a.device, b.device, c.device)

# 查看GPU是否可用
print(tf.test.is_gpu_available())

with tf.device('/gpu:0'):
    a = tf.random.normal([10000, 1000])
    b = tf.random.normal([1000, 2000])
    c = tf.matmul(a, b)
    print(a.device, b.device, c.device)

# 对比CPU、GPU运行时间

def cpu_run():
    with tf.device('/cpu:0'):
        a = tf.random.normal([10000, 1000])
        b = tf.random.normal([1000, 2000])
        c = tf.matmul(a, b)
    return c

def gpu_run():
    with tf.device('/gpu:0'):
        a = tf.random.normal([10000, 1000])
        b = tf.random.normal([1000, 2000])
        c = tf.matmul(a, b)
    return c

import timeit

cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('cpu time:', cpu_time)
print('gpu time:', gpu_time)

cpu_time = timeit.timeit(cpu_run, number=100)
gpu_time = timeit.timeit(gpu_run, number=100)
print('cpu time:', cpu_time)
print('gpu time:', gpu_time)

'''
在 tensorflow 中，张量可以运行在CPU，GPU，TPU上。一般无需指定设备，tensorflow会自动选择设备。


'''