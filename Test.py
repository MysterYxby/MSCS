# -*- coding: utf-8 -*-
# @Time    : 2022/7/29 9:45
# @Author  : Xu
# @Updating Time : 2023/2/8 17:33
import torch
from torch.autograd import Variable
from Net import *
import utils
import os
from PIL import Image
import os
from torchvision import transforms

#模型加载
Retinex_path = "model/RTNet.pth"
encoder_path = 'model/encoder.pth'
decoder_path = 'model/decoder.pth'
AFB_path     = 'model/AFB.pth'
Retinex = torch.load(Retinex_path,map_location='cpu')
encoder = torch.load(encoder_path,map_location='cpu')
decoder = torch.load(decoder_path,map_location='cpu')
AFB = torch.load(AFB_path,map_location='cpu')

def transform_img(img):
    trans = transforms.Compose([
        # transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Resize([256,360])
    ])
    return trans(img).unsqueeze(0)

vis_path = 'TNO/vis'
infrared_path = 'TNO/ir'
filenames_vis = os.listdir(vis_path)
filenames_ir = os.listdir(infrared_path)
for i in range (len(filenames_vis)):
    filepath = os.path.join(vis_path,filenames_vis[i])
    img_vis = Image.open(filepath)
    img_vis = transform_img(img_vis)
    filepath = os.path.join(infrared_path,filenames_ir[i])
    img_ir = Image.open(filepath).convert('L')
    img_ir = transform_img(img_ir)
    img_ir = Variable(img_ir, requires_grad=False)
    img_vis = Variable(img_vis, requires_grad=False)
    RI,RR = Retinex(img_ir)
    VI,VR = Retinex(img_vis)
    fr1,fi1 = encoder(RR,RI)
    fr2,fi2 = encoder(VR,VI)      
    f = AFB(fr1,fi1,fr2,fi2)
    S = decoder(f)
    save_path = 'outputs/' + str(i+1) + '.png'
    utils.save_PIL(S,save_path)