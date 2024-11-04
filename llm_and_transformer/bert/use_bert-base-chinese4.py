import gradio as gr
import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# 加载bert-base-chinese模型和分词器
model_name = "C:/Users/Administrator.DESKTOP-TPJL4TC/.cache/modelscope/hub/tiansz/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


def question_answering(context, question):
    # 使用分词器对输入进行处理
    inputs = tokenizer(question, context, return_tensors="pt")
    # 调用模型进行问答
    outputs = model(**inputs)
    # 获取答案的起始和结束位置
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    # 获取最佳答案
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores) + 1
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])
    return answer


# 创建Gradio界面
interface = gr.Interface(
    fn=question_answering,
    inputs=["text", "text"],  # 输入分别为context和question
    outputs="text",  # 输出为答案
)

interface.launch()
