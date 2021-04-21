from torchvision import transforms
from statistics import *
from jpeg2dct.numpy import loads
from PIL import Image
from turbojpeg import TurboJPEG

import numpy as np
import transform_functions as F
import cv2
import random, math, collections
import torch

#upsamples the CbCr channels by 2 to make them same size as Y
class UpsampleCbCr():
    def __init__(self, upscale_factor=2, interpolation='BILINEAR'):
        self.upscale_factor = upscale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        y, cb, cr = img[0], img[1], img[2]

        dh, dw, _ = y.shape
        cb = F.upscale(cb, desize_size=(dh, dw), interpolation=self.interpolation)
        cr = F.upscale(cr, desize_size=(dh, dw), interpolation=self.interpolation)

        return y, cb, cr

#we will go with the 'learned' subset; this reduces the 3rd dimensions to (14, 5, 5) for Y, Cb, Cr
class SubsetDCT2():
    def __init__(self, channels=24, pattern='learned'):
        if pattern == 'square':
            self.subset_channel_index = subset_channel_index_square
        elif pattern == 'learned':
            self.subset_channel_index = subset_channel_index_learned
        elif pattern == 'triangle':
            self.subset_channel_index = subset_channel_index_triangle

        if channels < 192:
            self.subset_y =  self.subset_channel_index[channels][0]
            self.subset_cb = self.subset_channel_index[channels][1]
            self.subset_cr = self.subset_channel_index[channels][2]

    def __call__(self, tensor):
        dct_y, dct_cb, dct_cr = tensor[0], tensor[1], tensor[2]
        dct_y, dct_cb, dct_cr = dct_y[:,:, self.subset_y], dct_cb[:, :, self.subset_cb], dct_cr[:, :, self.subset_cr]

        return dct_y, dct_cb, dct_cr

#randomly resizes HxW to a square; all Y, Cb, and Cr are resized to the same size, and keep their 3rd dimension(frequency)
class RandomResizedCropDCT():
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), L=1, M=1, N=8,
                 interpolation='BILINEAR', upscale_method='raw'):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.upscale_method = upscale_method

    def get_params(self, img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        y, cb, cr = img[0], img[1], img[2]
        i, j, h, w = self.get_params(y, self.scale, self.ratio)

        if self.upscale_method == 'raw':
            y  = F.resized_crop(y, i, j, h, w, self.size, self.interpolation)
            cb = F.resized_crop(cb, i, j, h, w, self.size, self.interpolation)
            cr = F.resized_crop(cr, i, j, h, w, self.size, self.interpolation)
        else:
            y  = F.resized_crop_dct(y, i, j, h, w, self.size, A1=self.A1, A2=self.A2)
            cb = F.resized_crop_dct(cb, i, j, h, w, self.size, A1=self.A1, A2=self.A2)
            cr = F.resized_crop_dct(cr, i, j, h, w, self.size, A1=self.A1, A2=self.A2)

        return y, cb, cr

#stacks together to get tensor 1 in figure 4: 56x56x24
class Aggregate():
    def __call__(self, img):
        dct_y, dct_cb, dct_cr = img[0], img[1], img[2]
        try:
            dct_y = np.concatenate((dct_y, dct_cb, dct_cr), axis=2)
        except:
            print('Y: {}, Cb: {}, Cr: {}'.format(dct_y.shape, dct_cb.shape, dct_cr.shape))
        return dct_y

#random horizontal flip
class RandomHorizontalFlip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            if not isinstance(img, np.ndarray) and (img.ndim in {2, 3}):
                raise TypeError('img should be CV Image. Got {}'.format(type(img)))
            return cv2.flip(img, 1)
        return img

#convert to tensor and reshape to [channel, width, height]
class ToTensorDCT():
    def __call__(self, img):
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        return img

#normalize between 0-1; returns a tuple with tensor, None, None so only first return is needed
class NormalizeDCT():
    def __init__(self, y_mean, y_std, cb_mean=None, cb_std=None, cr_mean=None, cr_std=None, channels=None, pattern='square'):
        self.y_mean,  self.y_std = y_mean, y_std
        self.cb_mean, self.cb_std = cb_mean, cb_std
        self.cr_mean, self.cr_std = cr_mean, cr_std

        if channels < 192:
            if pattern == 'square':
                self.subset_channel_index = subset_channel_index_square
            elif pattern == 'learned':
                self.subset_channel_index = subset_channel_index_learned
            elif pattern == 'triangle':
                self.subset_channel_index = subset_channel_index_triangle

            self.subset_y  = self.subset_channel_index[channels][0]
            self.subset_cb = self.subset_channel_index[channels][1]
            self.subset_cb = [64+c for c in self.subset_cb]
            self.subset_cr = self.subset_channel_index[channels][2]
            self.subset_cr = [128+c for c in self.subset_cr]
            self.subset = self.subset_y + self.subset_cb + self.subset_cr
            self.mean_y, self.std_y = [y_mean[i] for i in self.subset], [y_std[i] for i in self.subset]
        else:
            self.mean_y, self.std_y = y_mean, y_std


    def __call__(self, tensor):
        if isinstance(tensor, list):
            y, cb, cr = tensor[0], tensor[1], tensor[2]
            y  = F.normalize(y,  self.y_mean,  self.y_std)
            cb = F.normalize(cb, self.cb_mean, self.cb_std)
            cr = F.normalize(cr, self.cr_mean, self.cr_std)
            return y, cb, cr
        else:
            y = F.normalize(tensor, self.mean_y, self.std_y)
            return y, None, None

def adjust_size(y_size, cbcr_size):
    #if y = cbcr size, change nothing
    #if y is odd, make it even, then make cbcr half the size as in jpeg compression
    if y_size == cbcr_size:
        return y_size, cbcr_size
    elif np.mod(y_size, 2) == 1:
        y_size -= 1
        cbcr_size = y_size // 2
    return y_size, cbcr_size


if __name__ == '__main__':

    #testing on local computer to see that dataloader works
    input_size = 56 #resnet

    path = '/home/glenn/CP8322/code/data/val/n01440764/ILSVRC2012_val_00000293.JPEG'
    image = Image.open(path)

    with open(path, 'rb') as src:
        buffer = src.read()
    dct_y, dct_cb, dct_cr = loads(buffer)

    y_height, y_width = dct_y.shape[:-1]
    cbcr_height, cbcr_width = dct_cb.shape[:-1]

    y_height, cbcr_height = adjust_size(y_height, cbcr_height)
    y_width, cbcr_width = adjust_size(y_width, cbcr_width)
    dct_y = dct_y[:y_height, :y_width]
    dct_cb = dct_cb[:cbcr_height, :cbcr_width]
    dct_cr = dct_cr[:cbcr_height, :cbcr_width]
    sample = [dct_y, dct_cb, dct_cr]

    y_h, y_w, _ = dct_y.shape
    cbcr_h, cbcr_w, _ = dct_cb.shape

    print ("Y component DCT shape {} and type {}".format(dct_y.shape, dct_y.dtype))
    print ("Cb component DCT shape {} and type {}".format(dct_cb.shape, dct_cb.dtype))
    print ("Cr component DCT shape {} and type {}".format(dct_cr.shape, dct_cr.dtype))
    #print ("Shape of original image {} and type {}".format(np.array(image).shape, np.array(image).dtype))

    #Simplify: randomhorizontalflip
    transform = transforms.Compose([UpsampleCbCr(), 
        SubsetDCT2(channels=24, pattern='learned'),
        RandomResizedCropDCT(size=input_size),
        Aggregate(),
        RandomHorizontalFlip(),
        ToTensorDCT2(),
        NormalizeDCT(
            train_upscaled_static_dct_direct_mean_interp,
            train_upscaled_static_dct_direct_std_interp,
            channels=24,
            pattern='learned')])

    tensor = transform(sample)
    print('Random Resized Crop DCT: ', tensor[0].shape)