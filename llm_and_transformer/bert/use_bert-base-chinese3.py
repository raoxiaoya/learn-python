from modelscope import AutoModelForCausalLM
from modelscope import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tiansz/bert-base-chinese")
model = AutoModelForCausalLM.from_pretrained("tiansz/bert-base-chinese")
