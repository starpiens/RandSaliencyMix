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

def threshold_rand_saliency_bbox(img, lam):
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

    threshold = 150
    rand_saliency = saliency_map > threshold
    true_indices = np.where(rand_saliency)
    x_index = np.random.choice(len(true_indices[0]))
    y_index = np.random.choice(len(true_indices[1]))
    rand_indices = (true_indices[0][x_index], true_indices[1][y_index])
    x = rand_indices[0]
    y = rand_indices[1]

    bbx1 = np.clip(x - cut_w // 2, 0, width)
    bby1 = np.clip(y - cut_h // 2, 0, height)
    bbx2 = np.clip(x + cut_w // 2, 0, width)
    bby2 = np.clip(y + cut_h // 2, 0, height)

    return bbx1, bby1, bbx2, bby2

def local_mean_saliency_bbox(img, lam):
    size = img.size()
    width = size[1]
    height = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)
    
    temp_img = img.cpu().numpy().transpose(1, 2, 0)
    saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    _, saliency_map = saliency_detector.computeSaliency(temp_img)
    saliency_map = (saliency_map * 255).astype("uint8")

    saliency_map_ = saliency_map.copy()
    saliency_map_[:cut_w//2,:]= False
    saliency_map_[:,:cut_h//2] = False
    saliency_map_[width-cut_w//2:,:] = False
    saliency_map_[:,height-cut_h//2:] = False
    
    x_y_indices = np.where(saliency_map_)
    
    mean_values = []
    bboxes = []
    for i in range(10):
        x_index = np.random.choice(len(x_y_indices[0]))
        y_index = np.random.choice(len(x_y_indices[1]))
        random_indices = (x_y_indices[0][x_index], x_y_indices[1][y_index])
        x = random_indices[0]
        y = random_indices[1]

        bbx1 = np.clip(x - cut_w // 2, 0, width)
        bby1 = np.clip(y - cut_h // 2, 0, height)
        bbx2 = np.clip(x + cut_w // 2, 0, width)
        bby2 = np.clip(y + cut_h // 2, 0, height)

        bboxes.append([bbx1,bbx2,bby1,bby2])
        
        mean_values.append(saliency_map[bbx1:bbx2, bby1:bby2].mean())
        
    best = np.argmax(mean_values)
    f_bbx1,f_bbx2,f_bby1,f_bby2 = bboxes[best]
        
    return f_bbx1, f_bby1, f_bbx2, f_bby2

def saliency_mix(inp, tar, beta):
    # Generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(inp.shape[0])
    tar_a = tar
    tar_b = tar[rand_index]
    #bbx1, bby1, bbx2, bby2 = saliency_bbox(inp[rand_index[0]], lam)
    bbx1, bby1, bbx2, bby2 = threshold_rand_saliency_bbox(inp[rand_index[0]], lam)
    bbx1, bby1, bbx2, bby2 = local_mean_saliency_bbox(inp[rand_index[0]], lam)
    
    inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inp.shape[-1] * inp.shape[-2]))

    return inp, tar_a, tar_b, lam
