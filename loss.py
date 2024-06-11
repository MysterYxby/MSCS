import torch 
import torch.nn as nn
import numpy as np
import cv2 
import torch.nn.functional as F
from math import exp
from PIL import Image
from torchvision import transforms 
from torch.autograd import Variable

#高斯核初始化
g_kernel_size = 5
g_padding = 2
sigma = 3
reffac = 1
kx = cv2.getGaussianKernel(g_kernel_size,sigma)
ky = cv2.getGaussianKernel(g_kernel_size,sigma)
gaussian_kernel = np.multiply(kx,np.transpose(ky))
gaussian_kernel = torch.FloatTensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)


#归一化
def normalize1 (img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)

#梯度平方计算
def gradient(x):
    laplace = [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    kernel = torch.FloatTensor(laplace).unsqueeze(0).unsqueeze(0)
    return F.conv2d(x, kernel, stride=1, padding=1)

#重构损失
def reconstruction_loss(image, illumination, reflectance):
    reconstructed_image = illumination*reflectance
    return torch.norm(image-reconstructed_image, 1)
    
    
#照明度损失
def illumination_smooth_loss(image, illumination):
    gradient_gray_h, gradient_gray_w = gradient(image)
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=gaussian_kernel, padding=g_padding)+0.0001)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=gaussian_kernel, padding=g_padding)+0.0001)
    mean = ((torch.mean(image)) * (torch.ones(image.size())))
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    image.detach()
    mean.detach()
    return loss_h.sum() + loss_w.sum() + torch.norm(illumination - mean, 1)

def reflectance_smooth_loss(image, illumination, reflectance):
    gradient_gray_h, gradient_gray_w = gradient(image)
    gradient_reflect_h, gradient_reflect_w = gradient(reflectance)
    weight = 1/(illumination*gradient_gray_h*gradient_gray_w+0.0001)
    weight = normalize1(weight)
    weight.detach()
    loss_h = weight * gradient_reflect_h
    loss_w = weight * gradient_reflect_w
    refrence_reflect = image/illumination
    refrence_reflect.detach()
    return loss_h.sum() + loss_w.sum() + reffac*torch.norm(refrence_reflect - reflectance, 1)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def loss_RT(S,I,R):
    b,c,w,h = S.shape
    loss_res = reconstruction_loss(S, I, R)
    loss_I = 0.01*illumination_smooth_loss(S,I)
    loss_R = 0.5*reflectance_smooth_loss(S,I,R)
    loss = (loss_res + loss_I + loss_R)*0.01
    print(loss_res)
    print(loss_I)
    print(loss_R)
    return loss

def loss_AFB(O,ir,vis):
    a1 = 1
    a2 = 1
    a3 = 1
    b,c,h,w = O.size()
    loss_det = 1 - ssim(O,vis)
    loss_int = torch.norm(O-ir,2)
    loss_tex = (torch.norm(gradient(O)-torch.max(gradient(ir),gradient(vis)),1))/(w*h)
    return loss_det/b,loss_int/b,loss_tex/b,(a1*loss_det+a2*loss_int+a3*loss_tex)/b

def loss_Fusion(O,ir,vis):
    loss_ssim    = 1-(ssim(O,ir)+ssim(O,vis))/2
    loss_content = torch.norm(gradient(O)-gradient(ir),1)+torch.norm(gradient(O)-gradient(vis),1)
    print(loss_ssim)
    print(loss_content)
    return loss_content + loss_ssim

if __name__ == '__main__':
    img1 = Image.open("input/test/TNO/ir/1.png")
    img2 = Image.open("outputs/pre_fusion/TNO/1pre2.bmp")
    img3 = Image.open("input/test/TNO/vis/1.png")
    trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Resize((256,256))
    ])
    img1 = trans(img1)
    img1 = img1.unsqueeze(0)
    img2 = trans(img2)
    img2 = img2.unsqueeze(0)
    img3 = trans(img3)
    img3 = img3.unsqueeze(0)
    loss_det,loss_int,loss_tex,loss = loss_AFB(img2,img1,img3)
    print(loss)
    print(loss_det)
    print(loss_int)
    print(loss_tex)
    # a = torch.ones([8,1,460,460])
    # print(a.shape)
    # b = trans(a)
    # print(b.shape)

