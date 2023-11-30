import torch
print(torch.cuda.is_available())
print(11111111111111111111)
import time
adsf=time.time()






import paddle
import time

# 测试

from paddlespeech.cli.tts import TTSExecutor
 
tts_executor = TTSExecutor()
print(paddle.get_device())
fdsaf=paddle.get_device()
wav_file = tts_executor(
    text='你好，我叫 QAGLM，是一个由清华大学 KEG 实验室和智谱AI训练的大型语言模型。',
    output='output.wav',
    am='fastspeech2_mix',
    spk_id=0,
    voc='hifigan_csmsc',
    lang='mix',
    device=paddle.get_device())
print(wav_file)
print(time.time()-adsf)