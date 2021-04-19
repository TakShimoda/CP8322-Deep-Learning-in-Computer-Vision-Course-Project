from torchvision import transforms
from statistics import *
from jpeg2dct.numpy import loads
import numpy as np
from PIL import Image
import cv2

#upsamples the CbCr channels by 2 to make them same size as Y
class UpsampleCbCr(object):
    def __init__(self, upscale_factor=2, interpolation='BILINEAR'):
        self.upscale_factor = upscale_factor
        self.interpolation = interpolation

    def __call__(self, img):
        y, cb, cr = img[0], img[1], img[2]

        dh, dw, _ = y.shape
        # y  = F.upscale(y,  desize_size=(dh, dw), interpolation=self.interpolation)
        cb = upscale(cb, desize_size=(dh, dw), interpolation=self.interpolation)
        cr = upscale(cr, desize_size=(dh, dw), interpolation=self.interpolation)

        return y, cb, cr

#we will go with the 'learned' subset; this reduces the 3rd dimensions to (14, 5, 5) for Y, Cb, Cr
class SubsetDCT2(object):
    def __init__(self, channels=20, pattern='square'):
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
#requires many functions: crop, resized_crop, and resize
class RandomResizedCropDCT(object):
    """Crop the given CV Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), L=1, M=1, N=8,
                 interpolation='BILINEAR', upscale_method='raw'):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.upscale_method = upscale_method

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (CV Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
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
        """
        Args:
            img (np.ndarray): Image to be cropped and resized.

        Returns:
            np.ndarray: Randomly cropped and resized image.
        """
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
class Aggregate2(object):
    def __call__(self, img):
        dct_y, dct_cb, dct_cr = img[0], img[1], img[2]
        try:
            dct_y = np.concatenate((dct_y, dct_cb, dct_cr), axis=2)
        except:
            print('Y: {}, Cb: {}, Cr: {}'.format(dct_y.shape, dct_cb.shape, dct_cr.shape))
        return dct_y

#random horizontal flip
class RandomHorizontalFlip(object):
    """Horizontally flip the given CV Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (CV Image): Image to be flipped.

        Returns:
            CV Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

#convert to tensor; can be made simpler without the function used
class ToTensorDCT2(object):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        """
        Args:
            pic (numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor_dct(img)

#normalize between 0-1; returns a tuple with tensor, None, None so only first return is needed
class NormalizeDCT(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """
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
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if isinstance(tensor, list):
            y, cb, cr = tensor[0], tensor[1], tensor[2]
            y  = F.normalize(y,  self.y_mean,  self.y_std)
            cb = F.normalize(cb, self.cb_mean, self.cb_std)
            cr = F.normalize(cr, self.cr_mean, self.cr_std)
            return y, cb, cr
        else:
            y = F.normalize(tensor, self.mean_y, self.std_y)
            return y, None, None

def upscale(img, upscale_factor=None, desize_size=None, interpolation='BILINEAR'):
    INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}
    h, w, c = img.shape
    if upscale_factor is not None:
        dh, dw = upscale_factor*h, upscale_factor*w
    elif desize_size is not None:
        # dh, dw = desize_size.shape
        dh, dw = desize_size
    else:
        raise ValueError
    return cv2.resize(img, dsize=(dw, dh), interpolation=INTER_MODE[interpolation])

def adjust_size(y_size, cbcr_size):
    #if y = cbcr size, change nothing
    #if y is odd, make it even, then make cbcr half the size as in jpeg compression
    if y_size == cbcr_size:
        return y_size, cbcr_size
    elif np.mod(y_size, 2) == 1:
        y_size -= 1
        cbcr_size = y_size // 2
    return y_size, cbcr_size

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
print ("Shape of original image {} and type {}".format(np.array(image).shape, np.array(image).dtype))

transform = transforms.Compose([UpsampleCbCr(), 
    SubsetDCT2(channels=192, pattern='square')])

tensor = transform(sample)

# transform = transforms.Compose([UpsampleCbCr(), 
#     SubsetDCT2(channels=192, pattern='square'),
#     RandomResizedCropDCT(size=input_size),
#     transforms.Aggregate2(),
#     RandomHorizontalFlip(),
#     ToTensorDCT2(),
#     NormalizeDCT(train_upscaled_static_dct_direct_mean_interp, train_upscaled_static_dct_direct_std_interp, channels=192, pattern='square')
# ])