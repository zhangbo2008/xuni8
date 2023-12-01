import audio
aaa=audio.load_wav('my_data_preprocessed/20171116/section_1_024/section_1_024.07_026.30.wav',sr=None)
print(1)
import wave
wave_read = wave.open('my_data_preprocessed/20171116/section_1_024/section_1_024.07_026.30.wav',mode="rb")
bbb=audio.load_wav('my_data_preprocessed/20171116/section_1_024/section_1_024.07_026.30.wav',sr=16000)
print(wave_read._framerate)

