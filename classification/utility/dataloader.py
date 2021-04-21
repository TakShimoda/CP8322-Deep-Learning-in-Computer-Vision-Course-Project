import os
import sys
import torch
import torch.utils.data as data
import torchvision
import transformations
import numpy as np

from statistics import *
from torchvision import transforms
from PIL import Image
from jpeg2dct.numpy import loads

extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

#The image loader
class DataSetFolder(data.Dataset):
    def __init__(self, dir, device, transform, DCT=False, is_valid_file=None):
        if isinstance(dir, torch._six.string_classes):
            dir = os.path.expanduser(dir)
        self.DCT = DCT	
        self.transform = transform
        self.dir = dir
        self.device = device
        classes, class_to_idx = self._find_classes(dir)
        self.classes = classes
        self.class_to_idx = class_to_idx
        samples = make_dataset(dir, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0: 
            raise (RuntimeError("Found 0 files in subfolders of: " + self.dir + "\n" + "Supported extensions are: " + ",".join(extensions)))
        self.samples = samples

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.DCT:
            #jpeg DCT transform and chroma subsampling
            with open(path, 'rb') as src:
                buffer = src.read()
                dct_y, dct_cb, dct_cr = loads(buffer)

            y_size_h, y_size_w = dct_y.shape[:-1]
            cbcr_size_h, cbcr_size_w = dct_cb.shape[:-1]
            y_size_h, cbcr_size_h = adjust_size(y_size_h, cbcr_size_h)
            y_size_w, cbcr_size_w = adjust_size(y_size_w, cbcr_size_w)

            dct_y = dct_y[:y_size_h, :y_size_w]
            dct_cb = dct_cb[:cbcr_size_h, :cbcr_size_w]
            dct_cr = dct_cr[:cbcr_size_h, :cbcr_size_w]
            sample = [dct_y, dct_cb, dct_cr]

            y_h, y_w, _ = dct_y.shape
            cbcr_h, cbcr_w, _ = dct_cb.shape
            
            #--now transform--#
            dct_y, dct_cb, dct_cr = self.transform(sample)
            input_batch = dct_y
        else:
            input_image = pil_loader(path)
            input_batch = self.transform(input_image)
        return input_batch, target

    def __len__(self):
        return len(self.samples)
    
def load_data(model, device, valdir, batch_size, DCT=False, channels=24, pattern='learned'):
    if model == 'resnet':
        input_size = 56
    if DCT is False:
    	transform = transforms.Compose([
		    transforms.Resize(256),
		    transforms.CenterCrop(224),
		    transforms.ToTensor(),
		    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])
    else:
    	transform = transforms.Compose([
		transformations.UpsampleCbCr(),
		transformations.SubsetDCT2(channels=channels, pattern=pattern),
		transformations.RandomResizedCropDCT(size=input_size),
		transformations.Aggregate(),
		transformations.RandomHorizontalFlip(),
		transformations.ToTensorDCT(),
		transformations.NormalizeDCT(
		    train_upscaled_static_dct_direct_mean_interp,
		    train_upscaled_static_dct_direct_std_interp,
		    channels=channels,
		    pattern=pattern
		)
	    ])

    #instantiate dataset and data loader. the __getitem__ returns preprocessed image and labels
    dataset = DataSetFolder(valdir, device, transform, DCT=DCT)
    val_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    return val_loader

def adjust_size(y_size, cbcr_size):
    #if y = cbcr size, change nothing
    #if y is odd, make it even, then make cbcr half the size as in jpeg compression
    if y_size == cbcr_size:
        return y_size, cbcr_size
    elif np.mod(y_size, 2) == 1:
        y_size -= 1
        cbcr_size = y_size // 2
    return y_size, cbcr_size

def pil_loader(path):
    """open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    Args:
        path: (string)
            the path to the image, e.g. '/val/n01484850/ILSVRC2012_val_00002338.JPEG'
    Returns:
        img.convert('RGB')
            the image as loaded from PILLOW, converted to RGB(if it's already not)
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images
