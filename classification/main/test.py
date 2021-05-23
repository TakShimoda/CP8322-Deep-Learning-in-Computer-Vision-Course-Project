import torch
import torchvision
import argparse
import sys
from torchvision import transforms
from PIL import Image
from resnet import ResNetDCT_Upscaled_Static

sys.path.insert(0, '../utility')
from dataloader import *
from eval import AverageMeter, accuracy

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description='Testing ResNet-50 with DCT')

parser.add_argument('--batch', default=200, type=int, metavar='N', help='test batchsize (default: 200)')
parser.add_argument('--DCT', default='False', type=str2bool, help='Use DCT or not on the mdoel (default: False)')
parser.add_argument('--pretrained', default='True', type=str2bool, help='load pretrained model or not')

args = parser.parse_args()

#Global variables
if torch.cuda.is_available():
    print('===========USING CUDA===========')
    device = 'cuda'
else:
    print('===========USING CPU===========')
    device = 'cpu'

dir = '../data/val'
DCT = args.DCT
pretrained = args.pretrained
if DCT is True:
    model = ResNetDCT_Upscaled_Static(channels=24, pretrained=pretrained)
else:
    model = torchvision.models.resnet50(pretrained = pretrained)
model_name = 'resnet'
model.eval()
model.to(device)
top_1_accuracy = 0
top_5_accuracy = 0
batch_size = args.batch

if __name__ == '__main__':
    top1, top5 = AverageMeter(), AverageMeter()
    val_loader = load_data(model_name, device, dir, batch_size, DCT=DCT)
    print('====== Training on resnet50 with batch size ', batch_size, '|| Using DCT: ', DCT, '======\n\n')

    for index, (img, target) in enumerate(val_loader):
        img, target = img.cuda(non_blocking=True), target.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(img)
        top_1, top_5 = accuracy(output, target, (1, 5))
        top1.update(top_1.item(), img.size(0))
        top5.update(top_5.item(), img.size(0))
        if index % 50 == 0 and index > 0:
            print('Image number: %d || Top 1 Accuracy: %-2.4f || Top 5 Accuracy: %-2.4f' %(index*batch_size, top1.avg, top5.avg))

    print('Top 1 Accuracy: %-2.4f' % (top1.avg))
    print('Top 5 Accuracy: %-2.4f' % (top5.avg))



