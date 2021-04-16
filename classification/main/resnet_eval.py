import torch
import torchvision
from torchvision import transforms
from PIL import Image
from dataloader import *

#Global variables
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print('===========USING CUDA===========')
    device = 'cuda'
else:
    print('===========USING CPU===========')
    device = 'cpu'
dir = '../data/val'
model_name = 'resnet'
model = torchvision.models.resnet50(pretrained = True)
model.eval()
model.to(device)
top_1_accuracy = 0
top_5_accuracy = 0
batch_size = 32

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk):
    """ Returns the top-1 and top-5 accuracy on the batch
    Args:
        output(torch.tensor): the raw output of the model on the batch
        target(torch.tensor): the target labels on the batch
        topk(tuple): a tuple representing the topk to use: is (1, 5)
    Returns:
        res(list): a list of torch tensors for top1 and top5 accuracy
    """
    probabilities = torch.nn.functional.softmax(output)
    _, pred = probabilities.data.topk(5, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    top1, top5 = AverageMeter(), AverageMeter()
    val_loader = load_data(model_name, device, dir)
    for index, (img, target) in enumerate(val_loader):
        img, target = img.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(img)
        top_1, top_5 = accuracy(output, target, (1, 5))
        top1.update(top_1.item(), img.size(0))
        top5.update(top_5.item(), img.size(0))
        if index % 100 == 0:
            print('Batch number: %d || Top 1 Accuracy: %-2.4f || Top 5 Accuracy: %-2.4f' %(index, top1.avg, top5.avg))

    print('Top 1 Accuracy: %-2.4f' % (top1.avg))
    print('Top 5 Accuracy: %-2.4f' % (top5.avg))



