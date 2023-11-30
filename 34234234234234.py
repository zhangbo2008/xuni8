from  paddlespeech.cli.tts import TTSExecutor
tts_executor = TTSExecutor()
wav_file = tts_executor(
 text="热烈欢迎您在 Discussions 中提交问题，并在 Issues 中指出发现的 bug。此外，我们非常希望您参与到 Paddle Speech 的开发中！",
output='output.wav',
am='fastspeech2_mix',
voc='hifigan_csmsc',
lang='mix',
spk_id=174)