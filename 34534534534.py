import torch
print(torch.cuda.is_available())
print(11111111111111111111)
import time







import paddle
import time

# 测试

from paddlespeech.cli.tts import TTSExecutor
 
tts_executor = TTSExecutor()
print(paddle.get_device())
fdsaf=paddle.get_device()
adsf=time.time()
wav_file = tts_executor(
    text='今天的天气不错啊',
    output='output.wav',
    am='fastspeech2_csmsc',
    am_config=None,
    am_ckpt=None,
    am_stat=None,
    spk_id=174,
    phones_dict=None,
    tones_dict=None,
    speaker_dict=None,
    voc='pwgan_csmsc',
    voc_config=None,
    voc_ckpt=None,
    voc_stat=None,
    lang='zh',
    device=paddle.get_device())
print(wav_file)
print(time.time()-adsf)