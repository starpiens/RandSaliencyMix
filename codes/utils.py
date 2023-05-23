class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.val = []
        self.sum = [0.]
        self.avg = [0.]
        self.cnt = 0

    def reset(self):
        self.__init__()

    def update(self, 
               val: int | float | list[int | float] | tuple[int | float], 
               n: int = 1):
        if isinstance(val, int) or isinstance(val, float):
            val = [val]
        elif isinstance(val, tuple):
            val = list(val)

        self.val = val
        self.cnt += n
        self.sum = [i + j * n for i, j in zip(self.sum, self.val)]
        self.avg = [i / self.cnt for i in self.sum]


class TopkError:
    def __init__(self, topk=(1,)):
        self.topk = topk
        self.maxk = max(topk)

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
