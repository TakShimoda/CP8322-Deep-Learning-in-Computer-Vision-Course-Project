import torch
import torchvision
from torchvision import transforms
from PIL import Image
from dataloader import *
from eval import AverageMeter, accuracy

#Global variables
if torch.cuda.is_available():
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

if __name__ == '__main__':
    top1, top5 = AverageMeter(), AverageMeter()
    val_loader = load_data(model_name, device, dir, batch_size)
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



