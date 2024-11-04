import torch
from transformers import BertTokenizer, BertModel
import numpy as np

model_name = "C:/Users/Administrator.DESKTOP-TPJL4TC/.cache/modelscope/hub/tiansz/bert-base-chinese"

sentences = ['春眠不觉晓', '大梦谁先觉', '浓睡不消残酒', '东临碣石以观沧海']

tokenizer = BertTokenizer.from_pretrained(model_name)
# print(type(tokenizer)) # <class 'transformers.models.bert.tokenization_bert.BertTokenizer'>

model = BertModel.from_pretrained(model_name)
# print(type(model)) # <class 'transformers.models.bert.modeling_bert.BertModel'>


def test_similarity():
    with torch.no_grad():
        vs = [sentence_embedding(sentence).numpy() for sentence in sentences]
        nvs = [v / np.linalg.norm(v) for v in vs]  # normalize each vector
        m = np.array(nvs).squeeze(1)  # shape (4, 768)
        print(np.around(m @ m.T, decimals=2))  # pairwise cosine similarity


def sentence_embedding(sentence):
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    output = model(input_ids)
    return torch.mean(output.last_hidden_state, dim=1)


def test_encode():
    input_ids = tokenizer.encode('春眠不觉晓', return_tensors='pt') # shape (1, 7)
    output = model(input_ids)
    # print(output.last_hidden_state.shape)  # shape (1, 7, 768)
    v = torch.mean(output.last_hidden_state, dim=1)  # shape (1, 768)
    # print(v.shape)  # shape (1, 768)
    # print(output.pooler_output.shape)  # shape (1, 768)


def test_tokenizer():
    context = '海洋之所以呈现蓝色，首先是因为光的吸收和散射特性。太阳光是由红、橙、黄、绿、青、蓝、紫七种颜色组成的，其中波长较长的红光、橙光和黄光在射入海水后，由于透射力相对较弱，很容易被海水吸收。而波长较短的蓝光和紫光遇到纯净海水时，由于穿透力弱，最易被散射和反射。这就是为什么我们所见到的海洋多呈蔚蓝色或深蓝色的原因。'

    question = '海洋为什么是蓝色的'

    # <class 'transformers.tokenization_utils_base.BatchEncoding'>
    # inputs = tokenizer(question, context, return_tensors="pt")

    inputs = tokenizer.encode(question, context, return_tensors="pt")
    print(type(inputs))
    print(inputs.shape)
    '''
    <class 'torch.Tensor'>
    torch.Size([1, 166])
    '''


if __name__ == '__main__':
    test_similarity()
