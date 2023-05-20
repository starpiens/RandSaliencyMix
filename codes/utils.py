import torch


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.val = None
        self.cnt = 0
        self.sum = 0.

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        assert type(val) in (int, float, tuple, list)
        if self.val is not None:
            assert type(self.val) == type(val)

        self.val = val
        self.cnt += n

        if type(val) in (tuple, list):
            if type(self.sum) is float:
                self.sum = [0.] * len(val)
            self.sum = [i + j * n for i, j in zip(self.sum, val)]
            self.avg = [i / self.cnt for i in self.sum]
            
        else:
            self.sum += val * n
            self.avg = self.sum / self.cnt


class TopkError:
    
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.maxk = max(topk)

    @torch.no_grad()
    def __call__(self, output, target):
        batch_size = target.shape[0]
        _, pred = output.topk(self.maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            wrong_k = batch_size - correct_k
            res.append(wrong_k.item() * 100 / batch_size)

        return res
