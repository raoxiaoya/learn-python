'''
此程序为词向量 word embedding 的使用，参考 https://zhuanlan.zhihu.com/p/80737146

此程序包含分词，向量化，Elasticsearch，检索等全过程

jieba 将句子分词
pip install jieba

Gensim 训练词向量
pip install gensim
关于 Gensim 模块： https://zhuanlan.zhihu.com/p/40016964

'''

import jieba
import gensim
import json
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk


####################################### 训练模型 #######################################


def filterWords():
    # 停用词
    stopwords = [line.strip() for line in open(
        './data/ChineseStopWords.txt', encoding='UTF-8').readlines()]
    return stopwords


def segment(sentence: str):
    """
    结巴分词，并去除停用词
    """
    resp = []
    sentence_depart = jieba.cut(sentence.strip())
    for word in sentence_depart:
        if word not in filterWords():
            if word != "":
                resp.append(word)
    return resp


def read_corpus(f_name):
    """
    读数据
    """
    with open(f_name, encoding="utf-8") as f:
        for i, line in enumerate(f):
            yield gensim.models.doc2vec.TaggedDocument(segment(line), [i])


def train():
    """
    训练 Doc2Vec 模型
    """
    train_file = "./data/train_data.txt"

    train_corpus = list(read_corpus(train_file))

    model = gensim.models.doc2vec.Doc2Vec(
        vector_size=300, min_count=2, epochs=10)

    print(len(train_corpus))

    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count,
                epochs=model.epochs)

    model.save("doc2vec.model")


####################################### 数据处理 #######################################
'''
要想提供搜索能力，就必须有基础数据，比如，医院的所有病例中记载了病人在具有什么疾病特征后就得了什么病，
我们要整理这些疾病的描述，逐条制作成向量，写入 ES 中。

ES中使用指定的mapping创建索引。这里需要将向量这个Field“feature_vector”的类型设置为“dense_vector”，由于我们在model训练期间设置的纬度是300，这里需要指定dims为300.
'''

ELASTIC_ADDRESS = "http://localhost:9200"


def get_es_client():
    return Elasticsearch(hosts=[ELASTIC_ADDRESS])


def create_index():
    print("begin create index")
    setting = {
        "settings": {
            "number_of_replicas": 0,
            "number_of_shards": 2
        },
        "mappings": {
            "properties": {
                "name": {
                    "type": "keyword"
                },
                "department": {
                    "type": "keyword"
                },
                "feature": {
                    "type": "text"
                },
                "feature_vector": {
                    "type": "dense_vector",
                    "dims": 300
                }
            }
        }
    }
    get_es_client().indices.create(index='hospital_recommend', body=setting)
    print("end create index")


def embed_text(sentences):
    """
    将所有的句子转化为向量
    """
    model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
    resp = []
    for s in sentences:
        resp.append(model.infer_vector(segment(s)).tolist())
    return resp


def bulk_index_data():
    """
    将元数据及其向量索引到es中，且其中包含描述的特征向量字段
    """
    print("begin embed index data to vector")
    with open("./data/data.json") as file:
        load_dict = json.load(file)
    features = [doc["feature"] for doc in load_dict]
    print("number of lines to embed:", len(features))
    features_vectors = embed_text(features)
    print("begin index data to es")
    requests = []
    for i, doc in enumerate(load_dict):
        request = {'_op_type': 'index',  # 操作 index update create delete
                   '_index': 'hospital_recommend',  # index
                   '_id': doc["id"],
                   '_source':
                       {
                           'name': doc["name"],
                           'department': doc["department"],
                           'feature': doc["feature"],
                           'feature_vector': features_vectors[i],
                   }
                   }
        requests.append(request)
    bulk(get_es_client(), requests)
    print("end index data to es")

####################################### 检索数据 #######################################


'''
用户输入，我们假设从命令行输入即可。转化为向量也是使用最初训练的model进行了embed text，函数为上一个步骤使用过的embed_text。当用户的症状描述转化为一个向量时候，这时候即可从Es中进行搜索即可，在搜索的时候，需要使用Es的script_score的query，在query的scrip脚本中，将用户的向量放到查询语句的参数中，即可进行搜索，这里的搜索不是简单的文本匹配了，而是进行了语义层面的搜索。搜索结果中，我们将用户最大可能患有的疾病进行输出即可。
'''


def test():
    model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
    es = get_es_client()
    while True:
        try:
            query = input("Enter query: ")
            input_vector = model.infer_vector(segment(query)).tolist()
            resp = es.search(index='hospital_recommend', body={
                "_source": ["name", "feature"],
                "query": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.queryVector, doc['feature_vector'])+1",
                            "params": {
                                "queryVector": input_vector
                            }
                        }
                    }
                }
            })
            print("可能获得的疾病是：", end=" ")
            for hit in resp["hits"]["hits"]:
                print(hit["_source"]["name"], end="\t")
            print("\n")
        except KeyboardInterrupt:
            return
