# -*- coding: utf-8 -*-  
import os
import argparse

#引入飞桨生态的语音和GAN依赖
from PaddleTools.TTS import TTSExecutor
from PaddleTools.GAN import wav2lip

parser = argparse.ArgumentParser()
parser.add_argument('--human', type=str,default='', help='human video', required=False)
parser.add_argument('--output', type=str, default='output.mp4', help='output video')
parser.add_argument('--text', type=str,default='', help='human video', required=False)

if __name__ == '__main__':
    args = parser.parse_args()


    args.human='file/input/zimeng.mp4'
    args.human='file/input/test.png'
    args.text='各位开发者大家好，我是您的专属虚拟主播，很高兴能为您服务。'

    TTS = TTSExecutor('default.yaml') #PaddleSpeech语音合成模块
    wavfile = TTS.run(text=args.text,output='output.wav') #合成音频
    print('音频结束')
    video = wav2lip(args.human,wavfile,args.output) #将音频合成到唇形视频
    os.remove(wavfile) #删除临时的音频文件6
    print('视频生成完毕，输出路径为：{}'.format(args.output))