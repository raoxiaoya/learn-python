from swift.utils import seed_everything
from swift.llm import (
    get_model_tokenizer, get_template, inference_stream, ModelType, get_default_template_type,
)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

'''
pip install ms-swift

swift 实现流式输出模型的输出，如果模型不存在会自动下载

参考
https://blog.csdn.net/chrnhao/article/details/136284249
'''


model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(
    model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江的省会在哪里？'
gen = inference_stream(model, template, query)
print(f'query: {query}')
for response, history in gen:
    print(f'response: {response}')

query = '这有什么好吃的？'
gen = inference_stream(model, template, query, history)
print(f'query: {query}')
for response, history in gen:
    print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: 浙江的省会在哪里？
...
response: 浙江省的省会是杭州。
query: 这有什么好吃的？
...
response: 杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。
history: [('浙江的省会在哪里？', '浙江省的省会是杭州。'), ('这有什么好吃的？', '杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。')]
"""
