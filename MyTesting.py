import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
# from scipy import misc
from lib.FCCNet import FCCNet
from utils_2.data_val import test_dataset
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--test_path',type=str,default='./TestDataset/')
parser.add_argument('--ckpt_url', type=str, default="./val_epoch_best.pth",help='model_name')  # TODO
parser.add_argument('--save_path', type=str, default='./result/')

opt = parser.parse_args()
# save_mode_path = '/model/{}'.format(opt.modelname)
model = FCCNet(channel=32, pretrained=False)
model.load_state_dict(torch.load(opt.ckpt_url))
model.cuda()
model.eval()

os.makedirs(opt.save_path, exist_ok=True)
image_root = '{}/images/'.format(opt.test_path)
gt_root = '{}/masks/'.format(opt.test_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize)

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    image = image.cuda()
    output_shape = gt.shape
    res5, res4, res3, res2,oe = model(image)
    res = res2
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res = Image.fromarray((res * 255).astype(np.uint8), mode='L')
    name_split = name.split('-')
    os.makedirs(os.path.join(opt.save_path, name_split[2]), exist_ok=True)
    print("->")
