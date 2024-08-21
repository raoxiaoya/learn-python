import numpy as np

'''
信息熵
'''
def H(px):
    '''
    计算信息熵
    '''
    sum = 0
    for p in px:
        sum += p*np.log2(1/p)

    return sum


def generate():
    '''
    随机生成分布
    '''
    lt = np.random.randn(10)*100
    lt = abs(lt)
    sum = 0
    ltint = []
    p = []
    for i in lt:
        ins = int(i)
        ltint.append(ins)
        sum += ins

    for i in ltint:
        k = round(i/sum, 2)
        if k == 0:
            generate()

        p.append(k)

    psum = 0
    for i in p:
        psum += i

    return p


if __name__ == "__main__":
    r1 = H([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 均匀分布

    p = generate()
    print(p)

    r2 = H(p)

    print(r1)
    print(r2)
    print(np.log2(10))  # 熵的最大值为 logN
