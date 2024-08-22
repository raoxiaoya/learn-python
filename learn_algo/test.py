import numpy as np

def tt():
    t=[1,2,3,4]
    print(t[:3])

def t2():
    a = np.array([1,2,3])
    print(a.shape)
    b = np.array([5])
    print(b.shape)
    c = np.insert(a, 3, values=b, axis=0)
    print(c.shape)


if __name__ == '__main__':
    t2()