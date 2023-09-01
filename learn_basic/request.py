import requests
from typing import List

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

    # res2.text 为字符串
    # res2.content 为字节


def download_batch(urls: List[str]):
    '''
    批量下载图片
    '''
    if len(urls) == 0:
        return
    for url in urls:
        l = url.split('/')
        filename = l[-1]
        r = requests.get(url)
        with open('../imgs/'+filename, 'wb') as f:
            f.write(r.content)


download_batch([
    'https://img-blog.csdnimg.cn/34794c1b0b274694966df50a7ebdd63b.png',
    'https://pic3.zhimg.com/80/v2-9d0b71d384e7410f02f965ea58d0f966_720w.webp',
    'https://img-blog.csdnimg.cn/e69750034ba745ec96068e3d7d7f8eb4.png',
    'https://img-blog.csdnimg.cn/7342931d9d8046e789b50361f380b080.png',
    'https://img-blog.csdnimg.cn/f4ab8d3085434b858406f7101711f78c.png',
    'https://img-blog.csdnimg.cn/5637d0417bcd43569105fda4dc20ccf1.png',
    'https://img-blog.csdnimg.cn/d83a46e1193f4553836b39e8989417cf.png',
    'https://img-blog.csdnimg.cn/0b4031d5e6294cf98700a447a9f06c5c.png',
    'https://img-blog.csdnimg.cn/c9aa0f39254741bca476b504edacd950.png',
    'https://img-blog.csdnimg.cn/d3779edf3830434bafb2cfd6decd708e.png',
    'https://img-blog.csdnimg.cn/8cc8ce489bcb4652a92451899d667903.png',
    'https://img-blog.csdnimg.cn/a7687e0988ce47f7913ed67d4f634cc8.png',
    'https://img-blog.csdnimg.cn/25bdcb4b9cbf4196bd9535d68e8afd73.png',
    'https://img-blog.csdnimg.cn/7054d001801947df9ff0b337f7dc8c90.png',
    'https://img-blog.csdnimg.cn/9463dcce7ed84bfea384dfebf99f1581.png'
])
