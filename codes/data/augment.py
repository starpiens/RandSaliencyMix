import torch
import numpy as np
import cv2


def saliency_bbox(img, lam):
    size = img.size()
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Initialize OpenCV's static fine grained saliency detector
    # and compute the saliency map.
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliencyMap) = saliency.computeSaliency(temp_img)
    saliencyMap = (saliencyMap * 255).astype("uint8")

    maximum_indices = np.unravel_index(
        np.argmax(saliencyMap, axis=None), saliencyMap.shape
    )
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, W)
    bby1 = np.clip(y - cut_h // 2, 0, H)
    bbx2 = np.clip(x + cut_w // 2, 0, W)
    bby2 = np.clip(y + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class SaliencyMix:
    """SaliencyMix implementation from the authors.
    https://github.com/afm-shahab-uddin/SaliencyMix/
    """

    def __init__(self, beta: float):
        self.beta = beta

    def __call__(self, inp, tar):
        # Generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(inp.shape[0])
        tar_a = tar
        tar_b = tar[rand_index]
        bbx1, bby1, bbx2, bby2 = saliency_bbox(inp[rand_index[0]], lam)
        inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inp.shape[-1] * inp.shape[-2]))
        # Compute output
        inp_var = torch.autograd.Variable(inp, requires_grad=True)
        tar = tar_a * lam + tar_b * (1 - lam)
        return inp_var, tar


class SaliencyMixFixed:
    """Fixed SaliencyMix implementation,
    which fixes wrong implementation of the authors."""

    def __init__(self, beta: float):
        self.beta = beta

    def __call__(self, inp, tar):
        num_items = inp.shape[0]
        for paste_idx in range(num_items):
            copy_idx = np.random.randint(num_items)
            lam = np.random.beta(self.beta, self.beta)
            x1, y1, x2, y2 = saliency_bbox(inp[copy_idx], lam)
            inp[paste_idx, :, x1:x2, y1:y2] = inp[copy_idx, :, x1:x2, y1:y2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (inp.shape[-1] * inp.shape[-2]))
            tar[paste_idx] = tar[paste_idx] * lam + tar[copy_idx] * (1 - lam)
        return inp, tar
