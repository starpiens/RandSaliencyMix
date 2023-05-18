import torch
from torch.utils.data import Dataset
import numpy as np
import cv2


def saliency_bbox(img, lam):
    size = img.size()
    width = size[1]
    height = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency_detector.computeSaliency(temp_img)
    saliency_map = (saliency_map * 255).astype("uint8")

    maximum_indices = np.unravel_index(np.argmax(saliency_map, axis=None), saliency_map.shape)
    x = maximum_indices[0]
    y = maximum_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, width)
    bby1 = np.clip(y - cut_h // 2, 0, height)
    bbx2 = np.clip(x + cut_w // 2, 0, width)
    bby2 = np.clip(y + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2


def saliency_mix(inp, tar, beta):
    # Generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(inp.shape[0])
    tar_a = tar
    tar_b = tar[rand_index]
    bbx1, bby1, bbx2, bby2 = saliency_bbox(inp[rand_index[0]], lam)
    inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inp.shape[-1] * inp.shape[-2]))

    return inp, tar_a, tar_b, lam