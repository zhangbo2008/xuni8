






## https://www.paddlepaddle.org.cn/modelsDetail?modelId=26
from paddlespeech.cli.tts import TTSExecutor
## 如果找不到nltk的zip，就手动执行下 nltk_download.py
 
tts_executor = TTSExecutor()
wav_file = tts_executor(
    text="热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！",
    output='output.wav',
    am='fastspeech2_mix',
    voc='pwgan_aishell3',
    lang='mix',
    spk_id=170)