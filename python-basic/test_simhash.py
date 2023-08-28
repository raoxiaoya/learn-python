
from math import sqrt
from simhash import Simhash

'''
simhash 算法计算两个文章的相似度
'''


def simhash_similarity(text1, text2):
    a_simhash = Simhash(text1)
    b_simhash = Simhash(text2)
    print("a_simhash:", a_simhash.value)
    print("b_simhash:", b_simhash.value)
    max_hashbit = max(len(bin(a_simhash.value)), len(bin(b_simhash.value)))
    print("max_hashbit:", max_hashbit)

    # 汉明距离
    distince = a_simhash.distance(b_simhash)
    print("distince:", distince)
    similar = distince/max_hashbit
    return similar


def f1():
    text1 = open("files/article1.txt", "r", encoding="utf-8")
    text2 = open("files/article2.txt", "r", encoding="utf-8")
    similar = simhash_similarity(text1, text2)
    # 相相似度
    print(similar)
    text1.close()
    text2.close()


f1()
##############################################################################


def get_same_Item(prefs, person1, person2):
    '''找到二者相同评分项'''
    si = {}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item] = 1
            return si


def sim_euclid(prefs, p1, p2):
    '''欧几里得相似度算法'''
    si = get_same_Item(prefs, p1, p2)
    if len(si) == 0:
        return 0

    sum_of_squares = sum(
        [pow(prefs[p1][item] - prefs[p2][item], 2) for item in si])

    return 1 / (1 + sqrt(sum_of_squares))


def f2():
    critics = {
        'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5, 'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5, 'The Night Listener': 3.0},

        'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5, 'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0, 'You, Me and Dupree': 3.5}
    }
    # 0.29429805508554946
    print(sim_euclid(critics, 'Lisa Rose', 'Gene Seymour'))

##############################################################################


##############################################################################
