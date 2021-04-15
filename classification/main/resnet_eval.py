import os
import sys
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
from PIL import Image

#Global variables
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print('===========USING CUDA===========')
    device = 'cuda'
else:
    print('===========USING CPU===========')
    device = 'cpu'
dir = '/val'
extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
model = torchvision.models.resnet50(pretrained = True)
model.eval()
model.to(device)
top_1_accuracy = 0
top_5_accuracy = 0 

#copies DataSetFolderDCT(); we do not inherit from another superclass like the authors
class DataSetFolder(data.Dataset):
    def __init__(self, dir, is_valid_file=None):
        if isinstance(dir, torch._six.string_classes):
            dir = os.path.expanduser(dir)
        self.dir = dir
        classes, class_to_idx = self._find_classes(dir)
        self.classes = classes
        self.class_to_idx = class_to_idx
        samples = make_dataset(dir, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.dir + "\n"
                                "Supported extensions are: " + ",".join(extensions)))
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
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = pil_loader(path)
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)
        return input_batch, target

    def __len__(self):
        return len(self.samples)

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

def accuracy(pred, target):
    """ Returns 1 if top 1 or top 5 is correct
    Args:
        pred(torch.tensor): the top 5 predictions as a tensor
        target(int): the correct label
    Returns:
        top 1: 1 for correct, 0 otherwise
        top 5: 1 for correct, 0 otherwise
    """
    answers = [i.item()==target for i in pred]
    return answers[0], sum(answers[0:5])

if __name__ == '__main__':

    val_loader = DataSetFolder(dir)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    for index, (img, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, pred = probabilities.data.topk(5)
        answer = torch.argmax(probabilities).item()
        top_1, top_5 = accuracy(pred, target)
        top_1_accuracy += top_1
        top_5_accuracy += top_5
        if index % 1000 == 0:
            print('Iteration number: ', index)

    print('Top 1 accuracy: ', top_1_accuracy/(index))
    print('Top 5 accuracy: ', top_5_accuracy/(index))



