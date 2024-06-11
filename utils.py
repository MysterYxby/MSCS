from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        elif name.endswith('.tif'):
            images.append(join(directory, file))
        elif name.endswith('.bmp'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    #random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches

def get_image(path, mode='L'):
    if mode == 'L':
        image = Image.open(path).convert('L')
    else:
        image = Image.open(path).convert('RGB')
    image = np.array(image)
    return image

def get_train_images_auto(paths, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_test_images(paths, height=None, width=None, mode='L'):
    ImageToTensor = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize([128,128])])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])

    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images/255


def normalize1 (img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)

def toPIL(img):
    img = normalize1(img)
    TOPIL = transforms.ToPILImage()
    img = img.squeeze()
    return TOPIL(img)

def save_PIL(img,path):
    img  = toPIL(img)
    # img.convert('RGB')
    img.save(path)

def l1_norm(img1,img2):
    w1 = torch.mean(img1,1)
    w2 = torch.mean(img2,1)
    # w1 = torch.norm(img1,1)
    # w2 = torch.norm(img2,1)
    w = torch.exp(w1)/(torch.exp(w2)+torch.exp(w2))
    # w = w1/(w1+w2)
    f = w*img1 + (1-w)*img2
    return f
