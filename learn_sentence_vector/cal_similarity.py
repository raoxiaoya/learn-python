'''
计算中文文本相似度

https://www.zhihu.com/tardis/zm/ans/2147835262?source_id=1004

https://github.com/terrifyzhao/text_matching
'''


# 先安装sentence_transformers
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers import InputExample, evaluation, losses
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import faiss                   # make faiss available

# Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('distiluse-base-multilingual-cased')
# distiluse-base-multilingual-cased 蒸馏得到的，官方预训练好的模型

# 加载数据集


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            try:
                text1, text2, label = l.strip().split(',')
                D.append((text1, text2, int(label)))
            except ValueError:
                _
    return D


train_data = load_data('text_matching/input/train.csv')
valid_data = load_data('text_matching/input/dev.csv')
test_data = load_data('text_matching/input/test.csv')


# Define your train examples.
train_datas = []
for i in train_data:
    train_datas.append(InputExample(texts=[i[0], i[1]], label=float(i[2])))

# Define your evaluation examples
sentences1, sentences2, scores = [], [], []
for i in valid_data:
    sentences1.append(i[0])
    sentences2.append(i[1])
    scores.append(float(i[2]))

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1, sentences2, scores)


# Define your train dataset, the dataloader and the train loss
train_dataset = SentencesDataset(train_datas, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Define your train dataset, the dataloader and the train loss
train_dataset = SentencesDataset(train_datas, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100,
          evaluator=evaluator, evaluation_steps=200, output_path='./two_albert_similarity_model')


# 向量相似度的测评：
def f1():
    # Define your evaluation examples
    sentences1, sentences2, scores = [], [], []
    for i in test_data:
        sentences1.append(i[0])
        sentences2.append(i[1])
        scores.append(float(i[2]))

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores)
    model.evaluate(evaluator)

# 0.68723840499831


# 模型准确度的测评
def f2():
    '''
    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of 
    identifying similar and dissimilar sentences. The metrics are the cosine similarity as well 
    as euclidean and Manhattan distance The returned score is the accuracy with a specified metric.
    '''
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1, sentences2, scores)
    model.evaluate(evaluator)

# 0.8906211331111515


# 测试模型获取向量

def f3():
    model = SentenceTransformer('./two_albert_similarity_model')

    # Sentences are encoded by calling model.encode()
    emb1 = model.encode('什么情况导致评估不过')
    emb2 = model.encode("个人信用怎么评估出来的")
    print(emb1)
    print(emb2)

    cos_sim = util.pytorch_cos_sim(emb1, emb2)
    print("Cosine-Similarity:", cos_sim)


# 模型向量召回

def f4():
    ALL = []
    for i in tqdm(test_data):
        ALL.append(i[0])
        ALL.append(i[1])
    ALL = list(set(ALL))

    DT = model.encode(ALL)
    DT = np.array(DT, dtype=np.float32)

    # https://waltyou.github.io/Faiss-Introduce/
    index = faiss.IndexFlatL2(DT[0].shape[0])   # build the index
    print(index.is_trained)
    index.add(DT)                  # add vectors to the index
    print(index.ntotal)


# 查询最相似的文本
def f5():
    k = 10                          # we want to see 10 nearest neighbors
    aim = 220
    D, I = index.search(DT[aim:aim+1], k)  # sanity check
    print(I)
    print(D)
    print([ALL[i]for i in I[0]])


# 先获取特征再查询最相似的文本

def f6():
    query = [model.encode('1万元的每天利息是多少')]
    query = np.array(query, dtype=np.float32)
    D, I = index.search(query, 10)  # sanity check
    print(I)
    print(D)
    print([ALL[i]for i in I[0]])
