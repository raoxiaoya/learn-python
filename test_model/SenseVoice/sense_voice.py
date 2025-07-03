'''
conda activate tensorflow2.19.0

语音识别（Speech to text）

https://modelscope.cn/models/iic/SenseVoiceSmall
多语言音频理解模型，具有包括语音识别、语种识别、语音情感识别，声学事件检测能力。

注意：此模型在本机上无法正常运行，需要到 modelscope Notebook 上运行！！！

体验地址：https://www.modelscope.cn/studios/iic/SenseVoice

'''


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
    disable_update=True,
    # device="cuda:0",
)

rec_result = inference_pipeline(
    'https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav')
print(rec_result)

'''

输出
[{'key': 'asr_example_zh', 'text': '<|zh|><|NEUTRAL|><|Speech|><|woitn|>欢迎大家来体验达摩院推出的语音识别模型'}]

'''