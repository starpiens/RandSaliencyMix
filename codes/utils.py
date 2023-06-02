from torch import Tensor


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.val = 0
        self.sum = 0
        self.avg = 0.0
        self.cnt = 0

    def reset(self):
        self.__init__()

    def update(self, val, n=1):
        self.val = val
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt


class TopkError:
    def __init__(self, k: int = 1):
        self.k = k

    def __call__(self, output: Tensor, label: Tensor) -> float:
        batch_size = output.shape[0]
        label = label.argmax(1, keepdim=True)
        _, pred = output.topk(self.k, dim=1)
        num_correct = (label == pred).sum()
        num_wrong = batch_size - num_correct
        error = 100 * num_wrong / batch_size
        return error
