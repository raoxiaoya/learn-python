from modelscope import AutoModelForCausalLM, AutoTokenizer
from swift.llm import inference_stream, get_template

model_path = "D:\qwen\Qwen-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", trust_remote_code=True).eval()
template_type = 'qwen'
template = get_template(template_type, tokenizer)
history = None

before_len = 0

while True:
    query = input('User:')
    gen = inference_stream(model, template, query, history)
    print(f'System:', end="")
    for response, h in gen:

        print(response[before_len:], end="")
        before_len = len(response)
        history = h
    print()
