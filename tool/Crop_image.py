import torch
from torchvision import transforms 
from PIL import Image
import utils

def resize_img():
    trans  = trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256,256])
        ])

    ir_path = 'E:/image_fusion_data/test_50/infrared/'
    vis_path = 'E:/image_fusion_data/test_50/visible/'
    num = 50
    for index in range(num):
        ir_image_paths = ir_path + str(index+1) +'.jpg'
        vis_image_paths = vis_path + str(index+1) +'.jpg'
        ir = Image.open(ir_image_paths)
        vis = Image.open(vis_image_paths)
        vis_crop = trans(vis)
        ir_crop = trans(ir)
        vis_paths = 'LLVIP/' + 'vis/'  +str(index+1) + '.png'
        ir_paths  = 'LLVIP/' + 'ir/' +str(index+1) + '.png'
        utils.save_PIL(vis_crop,vis_paths)
        utils.save_PIL(ir_crop,ir_paths)

def crop_images():
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(256)
    ])

    data_num = 24
    crop_num = 100

    ir_path = 'input/train/Ir/'
    vis_path = 'input/train/Iv/'
    num = 0
    for index in range(data_num):
        ir_image_paths = ir_path + str(index+1) +'.bmp'
        vis_image_paths = vis_path + str(index+1) +'.bmp'
        ir = Image.open(ir_image_paths)
        vis = Image.open(vis_image_paths)
        for i in range(1,crop_num+1):
            num +=1
            seed = torch.random.seed()
            torch.random.manual_seed(seed)
            vis_crop = trans(vis)
            torch.random.manual_seed(seed)
            ir_crop = trans(ir)
            vis_paths = 'crop_img/' + 'vis/'  +str(num) + '.png'
            ir_paths  = 'crop_img/' + 'ir/' +str(num) + '.png'
            utils.save_PIL(vis_crop,vis_paths)
            utils.save_PIL(ir_crop,ir_paths)

path = 'E:/ir_vis_data/crop_img/vis/'
for i in range (2400):
    path = path + str(i+1) + '.png'
    out_path = 'E:\image_fusion_data' + str(i+2401) + '.png'
    img = Image.open(path)
    img.save(out_path)
# crop_images()
#resize_img()