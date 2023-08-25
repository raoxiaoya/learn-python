'''
自带的标准库模块
'''

from collections import OrderedDict
import json

favorite_languages = OrderedDict()
favorite_languages['jen'] = 'python'
favorite_languages['sarah'] = 'c'
favorite_languages['edward'] = 'ruby'
favorite_languages['phil'] = 'python'

for name, language in favorite_languages.items():
    print(name.title() + "'s favorite language is " + language.title() + ".")

############################################## json操作 #####################################

# 基础操作
mm = {'name': 'rao', 'age': 12}

mmstr = json.dumps(mm)
print(mmstr)  # {"name": "rao", "age": 12}
print(type(mmstr))  # <class 'str'>

mm2 = json.loads(mmstr)
print(mm2)  # {'name': 'rao', 'age': 12}
print(type(mm2))  # <class 'dict'>

# 操作文件
da = [1, 2, 3, 4, 5]
with open("files/jsondata.json", 'w') as of:
    json.dump(da, of)

with open("files/jsondata.json") as of:
    data = json.load(of)
print(data)
