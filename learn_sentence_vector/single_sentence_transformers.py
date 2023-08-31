'''
将句子向量化的一个脚本
'''


from copy import deepcopy
from random import randint
from sentence_transformers import SentenceTransformer, util

MODEL_CACHE_FOLDER = "D:/dev/php/magook/trunk/server/torch/sentence_transformers"
# D:\dev\php\magook\trunk\server\torch\sentence_transformers\sentence-transformers_multi-qa-MiniLM-L6-cos-v1


def f1():
    '''
    单个句子
    '''
    model = SentenceTransformer(
        'all-MiniLM-L6-v2', cache_folder=MODEL_CACHE_FOLDER)

    # The sentence we like to encode.
    sentences = ["今天天气是真的好啊"]

    # Compute sentence embeddings.
    embeddings = model.encode(sentences)

    # Create a list object, comma separated.
    vector_embeddings = list(embeddings)

    print(embeddings)
    print(vector_embeddings)


def f2():
    '''
    多个句子
    '''
    model = SentenceTransformer(
        'all-MiniLM-L6-v2', cache_folder=MODEL_CACHE_FOLDER)

    sentences = ["明月几时有", "把酒问青天"]
    embeddings = model.encode(sentences)
    for sentence, embedding in zip(sentences, embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")


def f22():
    '''
    句子相似度计算
    '''
    model = SentenceTransformer(
        'all-MiniLM-L6-v2', cache_folder=MODEL_CACHE_FOLDER)

    emb1 = model.encode("我很生气")
    emb2 = model.encode("气死我了")

    cos_sim = util.cos_sim(emb1, emb2)
    print("Cosine-Similarity:", cos_sim)


def f3():
    '''
    相似度计算与检索
    '''
    import scipy

    model = SentenceTransformer(
        'all-MiniLM-L6-v2', cache_folder=MODEL_CACHE_FOLDER)
    sentences = ['Lack of saneness',
                 'Absence of sanity',
                 'A man is eating food.',
                 'A man is eating a piece of bread.',
                 'The girl is carrying a baby.',
                 'A man is riding a horse.',
                 'A woman is playing violin.',
                 'Two men pushed carts through the woods.',
                 'A man is riding a white horse on an enclosed ground.',
                 'A monkey is playing drums.',
                 'A cheetah is running behind its prey.']
    sentence_embeddings = model.encode(sentences)

    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")

    # A query sentence uses for searching semantic similarity score.
    query = 'Nobody has sane thoughts'
    queries = [query]
    query_embeddings = model.encode(queries)

    # 用scipy库计算两个向量的余弦距离
    print("Semantic Search Results")
    number_top_matches = 3
    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist(
            [query_embedding], sentence_embeddings, "cosine")[0]
        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])
        print("Query:", query)
        print("\nTop {} most similar sentences in corpus:".format(number_top_matches))

        for idx, distance in results[0:number_top_matches]:
            print(sentences[idx].strip(),
                  "(Cosine Score: %.4f)" % (1-distance))
    # distance表示两个句子的余弦距离，距离越小语义越接近，或者说，1-distance可以理解为两个句子的余弦分数，分数越大表示两个句子的语义越相近


def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while (m):
        m -= 1
        i = randint(0, m)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst


def f4():
    '''
    fine-tune 微调模型
    参考 https://blog.csdn.net/javastart/article/details/119917405
    '''
    from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
    from torch.utils.data import DataLoader

    import xlrd
    f = xlrd.open_workbook('Ko2Cn.xlsx').sheet_by_name('Xbench QA')
    Ko_list = f.col_values(0)  # 　所有的中文句子
    Cn_list = f.col_values(1)  # 　所有的韩语句子

    shuffle_Cn_list = shuffle(Cn_list)  # 所有的中文句子打乱排序
    shuffle_Ko_list = shuffle(Ko_list)  # 　所有的韩语句子打乱排序

    train_size = int(len(Ko_list) * 0.8)
    eval_size = len(Ko_list) - train_size

    # train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8), InputExample(
    #     texts=['Another pair', 'Unrelated sentence'], label=0.3)]

    # Define your train examples.
    train_data = []
    for idx in range(train_size):
        train_data.append(InputExample(
            texts=[Ko_list[idx], Cn_list[idx]], label=1.0))
        train_data.append(InputExample(
            texts=[shuffle_Ko_list[idx], shuffle_Cn_list[idx]], label=0.0))

    # Define your evaluation examples
    sentences1 = Ko_list[train_size:]
    sentences2 = Cn_list[train_size:]
    sentences1.extend(list(shuffle_Ko_list[train_size:]))
    sentences2.extend(list(shuffle_Cn_list[train_size:]))
    scores = [1.0] * eval_size + [0.0] * eval_size

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores)
    # Define your train dataset, the dataloader and the train loss
    train_dataset = SentencesDataset(train_data, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model)

    # Define the model. Either from scratch of by loading a pre-trained model
    model = SentenceTransformer(
        'distiluse-base-multilingual-cased', cache_folder=MODEL_CACHE_FOLDER)

    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100,
              evaluator=evaluator, evaluation_steps=100, output_path='./Ko2CnModel')
    # 每隔100次训练集的迭代，进行一次验证，并且它会自动将在验证集上表现最好的模型保存到output_path


def f5():
    '''
    语义检索
    '''
    model = SentenceTransformer(
        'multi-qa-MiniLM-L6-cos-v1', cache_folder=MODEL_CACHE_FOLDER)

    query_embedding = model.encode('How big is London')
    passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                      'London is known for its finacial district'])

    print("Similarity:", util.dot_score(query_embedding, passage_embedding))


f5()
