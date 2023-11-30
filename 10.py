
def main(ttttt):
	#=固定原始视频的情况下,如何中间保存变量,从而进行加速.
	# apt-get install ffmpeg
	# apt install nvidia-cuda-toolkit
	# pip install paddlepaddle-gpu==2.4.2
	#  pip install paddlepaddle==2.4.2
	# pip install typeguard==2.13.3

	# -*- coding: utf-8 -*-    python3 -m pip install paddlespeech==1.0.0
	# ========简化代码.     python3 -m pip install paddleaudio==1.0.1  
	# cp nltk_data  /root/nltk_data -r
	import nltk

	print(nltk.data.path)
	# ['/root/nltk_data', '/root/miniconda3/nltk_data', '/root/miniconda3/share/nltk_data', '/root/miniconda3/lib/nltk_data', '/usr/share/nltk_data', '/usr/local/share/nltk_data', '/usr/lib/nltk_data', '/usr/local/lib/nltk_data']  #==========去这些里面看文件是否下好.
	tttttt=ttttt
	# tttttt='掼蛋，一种从“跑得快”和“八十分”演化而来的升级类纸牌游戏，在这两年快速完成了“升级”，从民间牌桌一跃进入了竞技体育赛场。上个周末，上海市棋牌运动管理中心二楼大厅内，全国掼牌（掼蛋）公开赛上海站比赛落下大幕。这是继掼蛋被列为今年10月下旬在安徽合肥举办的智力运动会表演项目之后，全国大赛的第一站比赛。为什么掼蛋能够在过去两三年以江苏和安徽为中心，迅速风靡全国，渗透到不同的社交圈层和领域？用上海市休闲棋牌协会副会长金方伟的话来总结：“它（掼蛋）不像桥牌那么复杂，但又融合了几种牌类的玩法，而且具有社交属性。通过团队合作打升级的方式，体现了一种趣味性，也有竞技性，所以也符合体育精神。比赛现场。社交属性突显掼蛋魅力在上海举行的这场全国掼牌（掼蛋）公开赛第一站比赛，吸引了一共72支队伍144名选手参赛，其中包括了各区体育局、行业集团、上海市休闲棋牌协会等单位，以及外围赛晋级选手。'

	import os
	import argparse
	#================第一部分,从文字生成音频.
	#引入飞桨生态的语音和GAN依赖
	# from PaddleTools.TTS import TTSExecutor
	# from PaddleTools.GAN import wav2lip

	parser = argparse.ArgumentParser()
	parser.add_argument('--human', type=str,default='', help='human video', required=False)
	parser.add_argument('--output', type=str, default='output.mp4', help='output video')
	parser.add_argument('--text', type=str,default='', help='human video', required=False)
	import torch
	print(torch.cuda.is_available())
	print(11111111111111111111)
	import time

	import paddle
	import time


	#====================# 预加载一些数据 放这里.

	import pickle

	with open("./marks", 'rb') as fr:
		face_det_results = pickle.load(fr)
	face_det_results=face_det_results

	args = parser.parse_args()


	args.human='file/input/zimeng.mp4'
	args.human='file/input/test.png'
	args.text=tttttt
	adsf=time.time()      #=============开始时间记录.
	print('生成音频使用了',time.time()-adsf)
	args2=args	
	wavfile='examples/driven_audio/bus_chinese.wav'






	#=======推理代码. 参数都已经设置好了,直接跑即可. 结果再results/result_voice.mp4里面.







	from os import listdir, path
	import numpy as np
	import scipy, cv2, os, sys, argparse, audio
	import json, subprocess, random, string
	from tqdm import tqdm
	from glob import glob
	import torch, face_detection
	from models import Wav2Lip
	import platform

	parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

	parser.add_argument('--checkpoint_path', type=str, 
						help='Name of saved checkpoint to load weights from', required=False)

	parser.add_argument('--face', type=str, 
						help='Filepath of video/image that contains faces to use', required=False)
	parser.add_argument('--audio', type=str, 
						help='Filepath of video/audio file to use as raw audio source', required=False)
	parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
									default='result99999999.mp4')

	parser.add_argument('--static', type=bool, 
						help='If True, then use only first video frame for inference', default=False)
	parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
						default=25., required=False)

	parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
						help='Padding (top, bottom, left, right). Please adjust to include chin at least')

	parser.add_argument('--face_det_batch_size', type=int, 
						help='Batch size for face detection', default=16)
	parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

	parser.add_argument('--resize_factor', default=1, type=int, 
				help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

	parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
						help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
						'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

	parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
						help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
						'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

	parser.add_argument('--rotate', default=False, action='store_true',
						help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
						'Use if you get a flipped result, despite feeding a normal looking video')

	parser.add_argument('--nosmooth', default=False, action='store_true',
						help='Prevent smoothing face detections over a short temporal window')

	args = parser.parse_args()
	args.img_size = 96
	args.face_det_batch_size = 4
	args.checkpoint_path = 'checkpoints/wav2lip_gan.pth'
	args.wav2lip_batch_size =40



	#========改这里啊就行.
	args.face = args2.human

	args.face = 'test3.mp4'
	args.audio = wavfile
	print(1)













	if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
		args.static = True

	def get_smoothened_boxes(boxes, T):
		for i in range(len(boxes)):
			if i + T > len(boxes):
				window = boxes[len(boxes) - T:]
			else:
				window = boxes[i : i + T]
			boxes[i] = np.mean(window, axis=0)
		return boxes

	def face_detect(images):
		detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
												flip_input=False, device=device)

		batch_size = args.face_det_batch_size
		
		while 1:
			predictions = []
			try:
				for i in tqdm(range(0, len(images), batch_size)):
					predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
			except RuntimeError:
				if batch_size == 1: 
					raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
				batch_size //= 2
				print('Recovering from OOM error; New batch size: {}'.format(batch_size))
				continue
			break

		results = []
		pady1, pady2, padx1, padx2 = args.pads
		for rect, image in zip(predictions, images):
			if rect is None:
				cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
				raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

			gaodu=rect[3]-rect[1] #========优化人脸位置.
			chagndu=rect[2]-rect[0]
			pady1=round(gaodu*0.1)
			pady2=round(gaodu*0.12)
			padx2=round(chagndu*0.1)
			padx1=round(chagndu*0.1)

			y1 = max(0, rect[1] - pady1)
			y2 = min(image.shape[0], rect[3] + pady2)
			x1 = max(0, rect[0] - padx1)
			x2 = min(image.shape[1], rect[2] + padx2)
			
			results.append([x1, y1, x2, y2])

		boxes = np.array(results)
		if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)

		# y1=max(0,y1-0.1*gaodu)
		# y2=y2+0.1*gaodu
		# boxes
		results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]


		del detector
		return results 
	# a=3

	# print(face_det_results)
	def datagen(frames, mels):
		nonlocal face_det_results
		# print(face_det_results)
		# print(a)
		img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []


		#=====写入:
		if 1:
			frames=frames[:50]
			if args.box[0] == -1:
				if not args.static: # 需要每一个帧进行处理.
					face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
				else:
					face_det_results = face_detect([frames[0]])
			else:
				print('Using the specified bounding box instead of face detection...')
				y1, y2, x1, x2 = args.box
				face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
		if 0:
				import pickle
				picklefile = open('marks', 'wb')
				# Pickle the dictionary and write it to file
				pickle.dump(face_det_results, picklefile)
				# Close the file
				picklefile.close()

		if 1:
			print(1)





		for i, m in enumerate(mels):
			idx = 0 if args.static else i%len(face_det_results) # 静态就是每个图片都是第一针.这里面已经保证了视频帧进行循环.
			frame_to_save = frames[idx].copy()
			face, coords = face_det_results[idx].copy()

			face = cv2.resize(face, (args.img_size, args.img_size))
				
			img_batch.append(face)
			mel_batch.append(m)
			frame_batch.append(frame_to_save)
			coords_batch.append(coords)

			if len(img_batch) >= args.wav2lip_batch_size:#========一直往[]里面添加.数量够了就yield出去.
				img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

				img_masked = img_batch.copy()
				img_masked[:, args.img_size//2:] = 0

				img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
				mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

				yield img_batch, mel_batch, frame_batch, coords_batch
				img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

		if len(img_batch) > 0: #========剩余的最后一组,
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch

	mel_step_size = 16
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print('Using {} for inference.'.format(device))

	def _load(checkpoint_path):
		if device == 'cuda':
			checkpoint = torch.load(checkpoint_path)
		else:
			checkpoint = torch.load(checkpoint_path,
									map_location=lambda storage, loc: storage)
		return checkpoint

	def load_model(path):
		model = Wav2Lip()
		print("Load checkpoint from: {}".format(path))
		checkpoint = _load(path)
		s = checkpoint["state_dict"]
		new_s = {}
		for k, v in s.items():
			new_s[k.replace('module.', '')] = v
		model.load_state_dict(new_s)

		model = model.to(device)
		return model.eval()

	if 1:
		if not os.path.isfile(args.face):
			raise ValueError('--face argument must be a valid path to video/image file')

		elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
			full_frames = [cv2.imread(args.face)]
			fps = args.fps

		else:
			video_stream = cv2.VideoCapture(args.face) #cv2.videocapture作为opencv中常用的视频读取函数，其主要作用是从本地或者网络中读取视频帧，并预存储到内存中，便于图片处理或者特征提取等操作。
			fps = video_stream.get(cv2.CAP_PROP_FPS)
			print('是用视频的帧率是',fps)
			print('Reading video frames...')

			full_frames = []
			while 1:
				still_reading, frame = video_stream.read()
				if not still_reading:
					video_stream.release()
					break
				if args.resize_factor > 1:
					frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

				if args.rotate:
					frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

				y1, y2, x1, x2 = args.crop
				if x2 == -1: x2 = frame.shape[1]
				if y2 == -1: y2 = frame.shape[0]

				frame = frame[y1:y2, x1:x2]

				full_frames.append(frame)

		print ("Number of frames available for inference: "+str(len(full_frames)))

		if not args.audio.endswith('.wav'):
			print('Extracting raw audio...')
			command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

			subprocess.call(command, shell=True)
			args.audio = 'temp/temp.wav'

		wav = audio.load_wav(args.audio, 16000)
		mel = audio.melspectrogram(wav)
		print(mel.shape)

		if np.isnan(mel.reshape(-1)).sum() > 0:
			raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

		mel_chunks = []
		mel_idx_multiplier = 80./fps  # 梅尔一秒是80, fps是我们采样率.
		i = 0
		while 1:
			start_idx = int(i * mel_idx_multiplier)
			if start_idx + mel_step_size > len(mel[0]): # 每16个mel取一个区间.
				mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
				break
			mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
			i += 1

		print("Length of mel chunks: {}".format(len(mel_chunks)))

		# full_frames = full_frames[:len(mel_chunks)]
		import time
		start=time.time()
		batch_size = args.wav2lip_batch_size
		gen = datagen(full_frames.copy(), mel_chunks)

		for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
												total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
			if i == 0:
				model = load_model(args.checkpoint_path)
				print ("Model loaded")

				frame_h, frame_w = full_frames[0].shape[:-1]
				out = cv2.VideoWriter('temp/result.avi', 
										cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

			with torch.no_grad():
				pred = model(mel_batch, img_batch)

			pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
			cnt=1
			for p, f, c in zip(pred, frames, coords):
				y1, y2, x1, x2 = c
				p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
				cv2.imwrite(f'temp/{cnt}.png',p)
				f[y1:y2, x1:x2] = p #======新的脸部贴上.
				out.write(f)
				cnt+=1

		out.release()
		#-========给视频加上音频.
		command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
		subprocess.call(command, shell=platform.system() != 'Windows')
		print('文件保存在',args.outfile)
		print('总时间',time.time()-adsf)





print(main('dsfasdfd'))