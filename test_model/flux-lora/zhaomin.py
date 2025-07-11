'''
modelscope平台提供的API推理

设置访问令牌：https://modelscope.cn/my/myaccesstoken

API Inference文档：https://www.modelscope.cn/docs/model-service/API-Inference/intro

'''

import requests
import json
from PIL import Image
from io import BytesIO

url = 'https://api-inference.modelscope.cn/v1/images/generations'

payload = {
    'model': 'raoxiaoya/zhaominplus',  # ModelScope Model-Id,required
    'prompt': 'A person dressed in traditional Chinese clothing, exquisitely styled, riding majestically on horseback, holding a folding fan in one hand, with a powerful and commanding gaze.'  # required
}
headers = {
    'Authorization': 'Bearer 7b51bec1-ecdd-4f5a-9aac-5dfe522c9562', # 这个 Bearer 不能删
    'Content-Type': 'application/json'
}

response = requests.post(url, data=json.dumps(
    payload, ensure_ascii=False).encode('utf-8'), headers=headers)

response_data = response.json()
image = Image.open(BytesIO(requests.get(
    response_data['images'][0]['url']).content))
image.save('result_image.jpg')
