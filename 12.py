# 最新数字人训练测试一体代码. 目标解决黑框问题, debug看看.到底咋回事2023-12-08,15点01        第一部分.训练syncnet


print('先训练syncnet模型')


train_steps=400
# 先需要训练一个syncnet给4.py用. 目标训练一个音频视频是否同步的分类器. 输出是不是同步的概率.
# import trl

from os.path import dirname, join, basename, isfile
from tqdm import tqdm
import cv2
from models import SyncNet_color as SyncNet
import audio
print(1)
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
import numpy as np
print(2)
from glob import glob

import os, random, cv2, argparse
from hparams import hparams, get_image_list

parser = argparse.ArgumentParser(description='Code to train the expert lip-sync discriminator')

parser.add_argument("--data_root", help="Root folder of the preprocessed LRS2 dataset", required=False)

parser.add_argument('--checkpoint_dir', help='Save checkpoints to this directory', required=False, type=str)
parser.add_argument('--checkpoint_path', help='Resumed from this checkpoint', default=None, type=str)

args = parser.parse_args('') # kaggle需要传入一个空字符串.

ppp='/kaggle/working/Wav2Lip/'
args.data_root='lrs2_preprocessed/LRS2_partly'
args.checkpoint_dir='./tmp2'
args.checkpoint_path='checkpoints/lipsync_expert.pth'

global_step = 0
global_epoch = 0
use_cuda = torch.cuda.is_available()
print('use_cuda: {}'.format(use_cuda))

syncnet_T = 5
syncnet_mel_step_size = 16
hparams.syncnet_checkpoint_interval=300
hparams.num_workers=0
hparams.syncnet_batch_size=10
hparams.syncnet_lr=3e-5
class Dataset(object):
    def __init__(self, split):
        # self.all_videos = get_image_list(args.data_root, split)
        self.all_videos =glob('my_data_preprocessed/20171116/*')
        print(self.all_videos)
    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def crop_audio_window(self, spec, start_frame):
        # num_frames = (T x hop_size * fps) / sample_rate
        start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]


    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            vidname = self.all_videos[idx] # 随便抽取一个视频.

            img_names = list(glob(join(vidname, '*.jpg')))
            if len(img_names) <= 3 * syncnet_T:
                continue
            img_name = random.choice(img_names)
            wrong_img_name = random.choice(img_names)
            while wrong_img_name == img_name:
                wrong_img_name = random.choice(img_names)
            #选一个真或者假照片.
            if random.choice([True, False]):
                y = torch.ones(1).float()
                chosen = img_name
            else:
                y = torch.zeros(1).float()
                chosen = wrong_img_name

            window_fnames = self.get_window(chosen)
            if window_fnames is None:
                continue

            window = []
            all_read = True
            for fname in window_fnames:
                img = cv2.imread(fname)
                if img is None:
                    all_read = False
                    break
                try:
                    img = cv2.resize(img, (hparams.img_size, hparams.img_size))
                except Exception as e:
                    all_read = False
                    break

                window.append(img)

            if not all_read: continue

            if 1:
#                 print(88888888888888888,vidname)
                aaa=list(glob(join(vidname, '*.wav')))[0]
                wavpath = join(vidname, "audio.wav")
#                 print(aaa,333333333)
                wav = audio.load_wav(aaa, hparams.sample_rate) # 输入音频原始的sr不用管, 这里面设置好我们需要的sr即可.16000.
#                 print(222222222)
                orig_mel = audio.melspectrogram(wav).T
#             except Exception as e:
#                 continue
#=======根据片段拿到真实的读音.
            mel = self.crop_audio_window(orig_mel.copy(), img_name)

            if (mel.shape[0] != syncnet_mel_step_size):
                continue

            # H x W x 3 * T
            x = np.concatenate(window, axis=2) / 255.  # window是5个人脸图片# channel变成15
            x = x.transpose(2, 0, 1)
            x = x[:, x.shape[1]//2:]  #################????????????????????????????????????????????????????为啥要切一半呢?????????????????我理解是人脸嘴的部分一定在图片的下半部分, 所以去掉上面, 会加速网络收敛.  # x只保留嘴部. 不要其他部分了!!!!!!
#  cv2.imwrite('tmp.png',x[:3,:,].transpose(1,2,0)*255)          window[0]      cv2.imwrite('tmp.png',x[:,:,:3]*255) 
            x = torch.FloatTensor(x) # x现在是 5个嘴部特写的concat
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            return x, mel, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):

    global global_step, global_epoch
    resumed_step = global_step
    
    while global_epoch < nepochs:
        running_loss = 0.
        # prog_bar = tqdm(enumerate(train_data_loader))
#         print(global_epoch,'global_epoch')
        for step, (x, mel, y) in enumerate(train_data_loader):
#             print(step,'step')
            if global_step==train_steps:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
                print('训完')
                return
            model.train()
            optimizer.zero_grad()
            # Transform data to CUDA device
            x = x.to(device)
            mel = mel.to(device)
            a, v = model(mel, x) # a:audio v:video 都是512向量.
            y = y.to(device)
            loss = cosine_loss(a, v, y)
            loss.backward()
            optimizer.step()
            global_step += 1
            cur_session_steps = global_step - resumed_step
            running_loss += loss.item()
            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch)
            if global_step%10==0:
              print(f'Loss: {running_loss / (step + 1)},global_step:{global_step}')
        global_epoch += 1


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch):

    checkpoint_path = join(
        checkpoint_dir, "lipsync_expert.pth")
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
#     global_step = checkpoint["global_step"]
#     global_epoch = checkpoint["global_epoch"]

    return model

if __name__ == "__main__":
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    # Dataset and Dataloader setup
    train_dataset = Dataset('train')
    test_dataset = Dataset('val')

    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.syncnet_batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.syncnet_batch_size,
        num_workers=8)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Model
    model = SyncNet().to(device)
    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.syncnet_lr)

    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer=False)

    train(device, model, train_data_loader, test_data_loader, optimizer,
          checkpoint_dir=checkpoint_dir,
          checkpoint_interval=hparams.syncnet_checkpoint_interval,
          nepochs=hparams.nepochs)











