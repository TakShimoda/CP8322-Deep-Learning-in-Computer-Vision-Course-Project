import cv2
import random, math, collections
import numpy as np
import torch

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}

def crop(img, x, y, h, w):
    assert isinstance(img, np.ndarray) and (img.ndim in {2, 3}), 'img should be CV Image. Got {}'
    assert h > 0 and w > 0, 'h={} and w={} should greater than 0'.format(h, w)

    x1, y1, x2, y2 = round(x), round(y), round(x+h), round(y+w)

    try:
        check_point1 = img[x1, y1, ...]
        check_point2 = img[x2-1, y2-1, ...]
    except IndexError:
        # warnings.warn('crop region is {} but image size is {}'.format((x1, y1, x2, y2), img.shape))
        img = cv2.copyMakeBorder(img, - min(0, x1), max(x2 - img.shape[0], 0),
                                 -min(0, y1), max(y2 - img.shape[1], 0), cv2.BORDER_CONSTANT, value=[0, 0, 0])
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)

    finally:
        return img[x1:x2, y1:y2, ...].copy()

def resized_crop(img, i, j, h, w, size, interpolation='BILINEAR'):

    assert isinstance(img, np.ndarray) and (img.ndim in {2, 3}), 'img should be CV Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img

def resize(img, size, interpolation='BILINEAR'):
    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
    else:
        oh, ow = size
        return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])

def upscale(img, upscale_factor=None, desize_size=None, interpolation='BILINEAR'):
    h, w, c = img.shape
    if upscale_factor is not None:
        dh, dw = upscale_factor*h, upscale_factor*w
    elif desize_size is not None:
        # dh, dw = desize_size.shape
        dh, dw = desize_size
    else:
        raise ValueError
    return cv2.resize(img, dsize=(dw, dh), interpolation=INTER_MODE[interpolation])

def normalize(tensor, mean, std):

    if torch.is_tensor(tensor) and tensor.ndimension() == 3:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor
    elif _is_numpy_image(tensor):
        return (tensor.astype(np.float32) - 255.0 * np.array(mean))/np.array(std)
    else:
        raise RuntimeError('Undefined type')
