import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision.utils import save_image

def saliency_bbox(inp, lam):
    size = inp.size()
    width = size[1]
    height = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    # initialize OpenCV's static fine grained saliency detector and compute the saliency map
    temp_img = inp.cpu().numpy().transpose(1, 2, 0)
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
    
    # bbx1 ~ bbx2, bby1 ~ bby2 : Salient Region
    Is = np.sum(saliency_map)
    Ps = np.sum(saliency_map[bbx1 : bbx2, bby1 : bby2])

    return bbx1, bby1, bbx2, bby2, Is, Ps
    

def saliency_sum(inp_img, bbx1, bby1, bbx2, bby2, lam):
    # Calculate Salient Region's Sum 
    temp_img = inp_img.cpu().numpy().transpose(1, 2, 0)
    saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency_detector.computeSaliency(temp_img)
    saliency_map = (saliency_map * 255).astype("uint8")
    It = np.sum(saliency_map)
    
    # Case : Sal to Corr
    Pt = np.sum(saliency_map[bbx1 : bbx2, bby1 : bby2])

    return It, Pt


def saliency_mix(inp, tar, beta):
    # Generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(inp.shape[0])
    tar_a = tar 
    tar_b = tar[rand_index] 
    bbx1, bby1, bbx2, bby2, Is, Ps = saliency_bbox(inp[rand_index[0]], lam)
    
    # Corr : Salient Region Sum of inp images
    It, Pt = [], []
    for inp_img in inp:
        temp_It, temp_Pt = saliency_sum(inp_img, bbx1, bby1, bbx2, bby2, lam)
        It.append(temp_It)
        Pt.append(temp_Pt)

    inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]

    # # augmented sample label Ya = Cs * Ys + Ct * Yt
    Cs = Ps / Is
    Ct = [1 - (Pt[i] / It[i]) for i in range(len(Pt))]

    # normalization : Cs + Ct = 1
    norm_Cs = Cs / (Cs + Ct)
    norm_Ct = [Ct[i] / (Cs + Ct[i]) for i in range(len(Ct))]

    return inp, tar_a, tar_b, Cs, Ct
