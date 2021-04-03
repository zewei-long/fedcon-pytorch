

from torchvision import transforms
from PIL import Image, ImageOps


imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from torchvision import datasets, transforms
import torch

import logging
import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

logger = logging.getLogger(__name__)

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v)


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)


def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v)


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold)


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img


class RandomTranslateWithReflect:
    
    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image

class Mnist_Transform: # Table 6 
    def __init__(self):


        self.trans_mnist1 = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=28,
#                                   padding=int(28*0.125),
#                                   padding_mode='reflect'),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])

        self.trans_mnist2 = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=28,
#                                   padding=int(28*0.125),
#                                   padding_mode='reflect'),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    def __call__(self, x):
        x1 = self.trans_mnist1(x) 
        x2 = self.trans_mnist2(x) 
        return x1, x2




    
class Cifar_Transform: # Table 6 
    def __init__(self):


        self.trans_cifar1 = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=32,
#                                   padding=int(32*0.125),
#                                   padding_mode='reflect'),
#             RandAugmentMC(n=15, m=10),            
#             transforms.ToTensor(), 
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        

        self.trans_cifar2 = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=32,
#                                   padding=int(32*0.125),
#                                   padding_mode='reflect'),
#             RandAugmentMC(n=15, m=10),            
#             transforms.ToTensor(), 
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        

    def __call__(self, x):
        x1 = self.trans_cifar1(x) 
        x2 = self.trans_cifar2(x) 
        return x1, x2

class Svhn_Transform: # Table 6 
    def __init__(self):


        self.trans_svhn1 = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        

        self.trans_svhn2 = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        

    def __call__(self, x):
        x1 = self.trans_svhn1(x) 
        x2 = self.trans_svhn2(x) 
        return x1, x2


    
class Mnist_Transform_t():
    def __init__(self):

        self.trans_mnist = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
    def __call__(self, x):
        return self.trans_mnist(x)

class Cifar_Transform_t():
    def __init__(self):
        
        self.trans_cifar = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
    def __call__(self, x):
        return self.trans_cifar(x)

class Svhn_Transform_t():
    def __init__(self):

        self.trans_svhn = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
    def __call__(self, x):
        return self.trans_svhn(x)





def get_aug(name, train, train_classifier=True):

    if train==True:
        if name == 'mnist':
            augmentation = Mnist_Transform()
        elif name == 'cifar10':
            augmentation = Cifar_Transform()
        elif name == 'svhn':
            augmentation = Svhn_Transform()
        else:
            raise NotImplementedError
    elif train==False:
        if name == 'mnist':
            augmentation = Mnist_Transform_t()
        elif name == 'cifar10':
            augmentation = Cifar_Transform_t()
        elif name == 'svhn':
            augmentation = Svhn_Transform_t()
        else:
            raise NotImplementedError
    
    return augmentation

class Mnist_Transform_fedmatch: # Table 6 
    def __init__(self):

        self.trans_mnist1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28*0.125),
                                  padding_mode='reflect'),
            transforms.RandomGrayscale(p=0.1),  
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),      
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        
        self.trans_mnist2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])



    def __call__(self, x):
        x1 = self.trans_mnist1(x) 
        x2 = self.trans_mnist2(x) 
        return x1, x2

class Cifar_Transform_fedmatch: # Table 6 
    def __init__(self):


        self.trans_cifar1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=15, m=10),            
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])      
        

        self.trans_cifar2 = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        

    def __call__(self, x):
        x1 = self.trans_cifar1(x) 
        x2 = self.trans_cifar2(x) 
        return x1, x2

class Svhn_Transform_fedmatch: # Table 6 
    def __init__(self):


        self.trans_svhn1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=15, m=10),              # 1 改成10 10
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])  
        

        self.trans_svhn2 = transforms.Compose([
            RandomTranslateWithReflect(4),          
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        

    def __call__(self, x):
        x1 = self.trans_svhn1(x) 
        x2 = self.trans_svhn2(x) 
        return x1, x2

# class Svhn_Transform_fedmatch: # Table 6 
#     def __init__(self):


#         self.trans_svhn1 = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=32,
#                                   padding=int(32*0.125),
#                                   padding_mode='reflect'),
#             RandAugmentMC(n=5, m=5),              # 1 改成10 10
#             transforms.ToTensor(), 
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])  
        

#         self.trans_svhn2 = transforms.Compose([
#             transforms.ToTensor(), 
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])
        

#     def __call__(self, x):
#         x1 = self.trans_svhn1(x) 
#         x2 = self.trans_svhn2(x) 
#         return x1, x2
    
def get_aug_fedmatch(name, train, train_classifier=True):

    if train==True:
        if name == 'mnist':
            augmentation = Mnist_Transform_fedmatch()
        elif name == 'cifar10':
            augmentation = Cifar_Transform_fedmatch()
        elif name == 'svhn':
            augmentation = Svhn_Transform_fedmatch()
        else:
            raise NotImplementedError
    elif train==False:
        if name == 'mnist':
            augmentation = Mnist_Transform_t()
        elif name == 'cifar10':
            augmentation = Cifar_Transform_t()
        elif name == 'svhn':
            augmentation = Svhn_Transform_t()
        else:
            raise NotImplementedError
    
    return augmentation


class Mnist_Transform_uda: # Table 6 
    def __init__(self):

        self.trans_mnist1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.trans_mnist2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28*0.125),
                                  padding_mode='reflect'),
            transforms.RandomGrayscale(p=0.1),  
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),      
            transforms.ToTensor(), 
            transforms.Normalize((0.1307,), (0.3081,))
            ])
    


    def __call__(self, x):
        x1 = self.trans_mnist1(x) 
        x2 = self.trans_mnist2(x) 
        return x1, x2

class Cifar_Transform_uda: # Table 6 
    def __init__(self):

        self.trans_cifar1 = transforms.Compose([
            RandomTranslateWithReflect(4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.trans_cifar2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=15, m=10),            
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])      
        
  

    def __call__(self, x):
        x1 = self.trans_cifar1(x) 
        x2 = self.trans_cifar2(x) 
        return x1, x2

class Svhn_Transform_uda: # Table 6 
    def __init__(self):


        self.trans_svhn1 = transforms.Compose([
            RandomTranslateWithReflect(4),          
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        self.trans_svhn2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=15, m=10),              # 1 改成10 10
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])  
        

        

    def __call__(self, x):
        x1 = self.trans_svhn1(x) 
        x2 = self.trans_svhn2(x) 
        return x1, x2
    
    
def get_aug_uda(name, train, train_classifier=True):

    if train==True:
        if name == 'mnist':
            augmentation = Mnist_Transform_uda()
        elif name == 'cifar10':
            augmentation = Cifar_Transform_uda()
        elif name == 'svhn':
            augmentation = Svhn_Transform_uda()
        else:
            raise NotImplementedError
    elif train==False:
        if name == 'mnist':
            augmentation = Mnist_Transform_t()
        elif name == 'cifar10':
            augmentation = Cifar_Transform_t()
        elif name == 'svhn':
            augmentation = Svhn_Transform_t()
        else:
            raise NotImplementedError
    
    return augmentation



