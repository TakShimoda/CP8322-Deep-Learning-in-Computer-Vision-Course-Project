import torch
import torch.nn as nn
import torch.optim as optim
import os

from resnet import ResNetDCT_Upscaled_Static
from logger import Logger
from train_dataloader import *
from eval import AverageMeter, accuracy

#Define global parameters. Training parameters based on the paper outlined in section 4.1
lr = 0.1
momentum = 0.9
weight_decay = 4e-5 #for some reason 1e-4 in code??
start_epoch = 0
epochs = 210
batch_size = 32
if torch.cuda.is_available():
    print('===========USING CUDA===========')
    device = 'cuda'
else:
    print('===========USING CPU===========')
    device = 'cpu'
dir = '../data/val'
best_prec1 = 0  # best test accuracy

#parser arguments
channels = '24'
pattern = 'square' #
checkpoint = 'checkpoints/imagenet/resnet50dct_upscaled_static_' + pattern + '_' + channels
title = 'ImageNet-ResNetDCT_Upscaled_Static'
resume = ''
dir = '../data/train'

def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to train mode
    model.train()

    for batch_idx, (image, target) in enumerate(train_loader):
        # measure data loading time
        #data_time.update(time.time() - end)

        image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)

        # compute output
        output = model(image)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        top5.update(prec5.item(), image.size(0))

    return (losses.avg, top1.avg, top5.avg)

if __name__ == '__main__':
    model = ResNetDCT_Upscaled_Static(channels=int(channels), pretrained=True)
    model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)

    if resume:
        # Load checkpoint.
        print('==== Resuming from checkpoint..')
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        checkpoint = os.path.dirname(resume)
        logger = Logger(os.path.join(checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc Top1.', 'Valid Acc Top5.'])

    # Data loading code
    train_loader = train_loader(data, 'resnet', device, dir, batch_size)
    #val_loader = validation_loader(args, model='resnet')

    for epoch in range(start_epoch, epochs):
        print('Epoch: [%d | %d] LR: %f \n' % (epoch + 1, epochs, lr))

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        #val_loss, prec1, prec5 = test(val_loader, model, criterion)

        # append logger file
        #logger.append([state['lr'], train_loss, val_loss, train_acc, prec1, prec5])
        #logger.append([lr, train_loss, train_acc, prec1, prec5])

        # save model
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        # save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_prec1': best_prec1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, epoch, is_best, checkpoint=args.checkpoint)

        # logger.close()
        # logger.plot()

        #-------------------pick up from here and train; reference def train() in imagenet_resnet_upscaled_static.py
        # also need to work on transform!

