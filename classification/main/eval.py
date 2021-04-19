import torch

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
    batch_size = target.size(0)
    probabilities = torch.nn.functional.softmax(output)
    _, pred = probabilities.data.topk(5, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res