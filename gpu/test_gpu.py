'''
conda activate tensorflow2.19.0

'''

import cupy as cp


def matrix_multiplication():
    # 定义矩阵的大小
    N = 2  # 矩阵的行或列数
    A = cp.array([[1, 2], [3, 4]])  # 矩阵 A
    B = cp.array([[5, 6], [7, 8]])  # 矩阵 B

    # 执行矩阵乘法
    C = cp.matmul(A, B)

    # 将结果从 GPU 内存复制回主机内存
    result = cp.asnumpy(C)

    # 输出结果
    print("Result C:")
    print(result)

if __name__ == "__main__":
    matrix_multiplication()
