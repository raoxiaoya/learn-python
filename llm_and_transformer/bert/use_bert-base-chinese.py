from transformers import BertTokenizer

'''
参考 
https://zhuanlan.zhihu.com/p/610171544
https://zhuanlan.zhihu.com/p/668482092
'''

# 加载预训练字典和分词方法
tokenizer = BertTokenizer.from_pretrained(
    # 可选，huggingface 中的预训练模型名称或路径，默认为 bert-base-chinese
    pretrained_model_name_or_path='C:/Users/Administrator.DESKTOP-TPJL4TC/.cache/modelscope/hub/tiansz/bert-base-chinese',
    cache_dir=None,  # 将数据保存到的本地位置，使用cache_dir 可以指定文件下载位置
    force_download=False,
)

sents = [
    '选择珠江花园的原因就是方便。',
    '笔记本的键盘确实爽。',
    '房间太小。其他的都一般。',
    '今天才知道这书还有第6卷,真有点郁闷.',
    '机器背面似乎被撕了张什么标签，残胶还在。',
]

print(type(tokenizer))


# 编码两个句子
out = tokenizer.encode(
    text=sents[0],
    text_pair=sents[1],  # 一次编码两个句子，若没有text_pair这个参数，就一次编码一个句子

    # 当句子长度大于max_length时,截断
    truncation=True,

    # 一律补pad到max_length长度
    padding='max_length',   # 少于max_length时就padding
    add_special_tokens=True,
    max_length=30,
    return_tensors=None,  # None表示不指定数据类型，默认返回list
)

print(out)

tokenizer.decode(out)
'''
bert-base-chinese是以一个字作为一个词，开头是特殊符号 [CLS]，两个句子中间用 [SEP] 分隔，句子末尾也是 [SEP]，最后用 [PAD] 将句子填充到 max_length 长度
'''

#增强的编码函数
out = tokenizer.encode_plus(
    text=sents[0],
    text_pair=sents[1],

    #当句子长度大于max_length时,截断
    truncation=True,

    #一律补零到max_length长度
    padding='max_length',
    max_length=30,
    add_special_tokens=True,

    #可取值tensorflow,pytorch,numpy,默认值None为返回list
    return_tensors=None,

    #返回token_type_ids
    return_token_type_ids=True,

    #返回attention_mask
    return_attention_mask=True,

    #返回special_tokens_mask 特殊符号标识
    return_special_tokens_mask=True,

    #返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
    #return_offsets_mapping=True,

    #返回length 标识长度
    return_length=True,
)

print(out)   # 字典

for k, v in out.items():
    print(k, ':', v)

tokenizer.decode(out['input_ids'])

'''
input_ids 就是编码后的词，即将句子里的一个一个词变为一个一个数字
token_type_ids 第一个句子和特殊符号的位置是0，第二个句子的位置是1（含第二个句子末尾的 [SEP]）
special_tokens_mask 特殊符号的位置是1，其他位置是0
attention_mask pad的位置是0，其他位置是1
length 返回句子长度
'''

# 上述方式是一次编码一个或者一对句子，但是实际操作中需要批量编码句子

