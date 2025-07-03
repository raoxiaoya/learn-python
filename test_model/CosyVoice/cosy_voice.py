'''
conda create -n cosyvoice python=3.10.16
conda activate cosyvoice

语音生成TTS（Text To Speech）

https://modelscope.cn/models/iic/CosyVoice2-0.5B

zero_shot 零样本

注意：测试代码需要在 D:\dev\php\magook\trunk\server\CosyVoice 项目中运行，文件为 test.py，此处只是个备份。教程在 总结.md 中查看。

'''

from modelscope import snapshot_download
import torchaudio
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
import sys

sys.path.append('third_party/Matcha-TTS')

download_dir = 'D:/dev/php/magook/trunk/server/ai_models/pretrained_models'
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)


def download_models():
    snapshot_download('iic/CosyVoice2-0.5B',
                      local_dir=download_dir+'/CosyVoice2-0.5B')
    snapshot_download('iic/CosyVoice-300M',
                      local_dir=download_dir+'/CosyVoice-300M')
    snapshot_download('iic/CosyVoice-300M-SFT',
                      local_dir=download_dir+'/CosyVoice-300M-SFT')
    snapshot_download('iic/CosyVoice-300M-Instruct',
                      local_dir=download_dir+'/CosyVoice-300M-Instruct')
    snapshot_download('iic/CosyVoice-ttsfrd',
                      local_dir=download_dir+'/CosyVoice-ttsfrd')


def run_models():
    cosyvoice = CosyVoice2(download_dir+'/CosyVoice2-0.5B',
                           load_jit=False, load_trt=False, load_vllm=False, fp16=False)

    # NOTE if you want to reproduce the results on https://funaudiollm.github.io/cosyvoice2, please add text_frontend=False during inference
    # zero_shot usage
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    # save zero_shot spk for future usage
    assert cosyvoice.add_zero_shot_spk(
        '希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
    for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)
    cosyvoice.save_spkinfo()

    # fine grained control, for supported control, check cosyvoice/tokenizer/tokenizer.py#L248
    for i, j in enumerate(cosyvoice.inference_cross_lingual('在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k, stream=False)):
        torchaudio.save('fine_grained_control_{}.wav'.format(i),
                        j['tts_speech'], cosyvoice.sample_rate)

    # instruct usage
    for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
        torchaudio.save('instruct_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)

    # bistream usage, you can use generator as input, this is useful when using text llm model as input
    # NOTE you should still have some basic sentence split logic because llm can not handle arbitrary sentence length


def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '今天真是太开心啦，吃到了最想吃的正新鸡排，抢到了梦寐以求的Labubu，和闺蜜们在东湖绿道骑车 吹着凉爽的风，欣赏着醉人的景，一切都是那么美好。'


def run_models2():
    cosyvoice = CosyVoice2(download_dir+'/CosyVoice2-0.5B',
                           load_jit=False, load_trt=False, load_vllm=False, fp16=False)
    for i, j in enumerate(cosyvoice.inference_zero_shot(text_generator(), '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
        torchaudio.save('zero_shot_{}.wav'.format(
            i), j['tts_speech'], cosyvoice.sample_rate)


if __name__ == '__main__':
    download_models()
    # run_models()
    # run_models2()
