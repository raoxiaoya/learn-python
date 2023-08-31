import requests
import json

'''
网络请求
'''


def r1():
    url = 'https://api.github.com/search/repositories?q=language:python&sort=stars'
    r = requests.get(url)
    print("Status code:", r.status_code)

    response_dict = r.json()

    print(response_dict.keys())


def r2():
    '''
    POST
    x-www-form-urlencoded
    json
    '''
    url1 = 'http://localhost:8080/server.php'
    data = {}
    data['name'] = 'rao'
    data['age'] = 12

    # 传入 map
    res = requests.post(url=url1, data=data)
    print(res.text)

    url2 = 'http://localhost:8080/server.php'
    data = {}
    data['name'] = 'rao'
    data['age'] = 12

    # 传入 json str
    res2 = requests.post(url=url2, json=data)
    print(res2.text)


r2()
