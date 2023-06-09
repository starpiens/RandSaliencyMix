from multiprocessing import Queue, Process

import torch
import numpy as np
import cv2
from torch import Tensor
from numpy import ndarray


def _pick_most_salient_pixel(
    saliency_map: ndarray, lam: float
) -> tuple[int, int, int, int]:
    h, w = saliency_map.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    row, col = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    bbr1 = int(np.clip(row - cut_h // 2, 0, h))
    bbc1 = int(np.clip(col - cut_w // 2, 0, w))
    bbr2 = int(np.clip(row + cut_h // 2, 0, h))
    bbc2 = int(np.clip(col + cut_w // 2, 0, w))

    return bbr1, bbc1, bbr2, bbc2


class SaliencyMix:
    """SaliencyMix implementation from the authors.
    https://github.com/afm-shahab-uddin/SaliencyMix/
    """

    def __init__(self, beta: float) -> None:
        self.beta = beta

    @torch.no_grad()
    def __call__(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        # Generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(images.shape[0])
        tar_a = labels
        tar_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = self._saliency_bbox(images[rand_index[0]], lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (images.shape[-1] * images.shape[-2])
        )
        # Compute output
        inp_var = torch.autograd.Variable(images, requires_grad=True)
        labels = tar_a * lam + tar_b * (1 - lam)
        return inp_var, labels

    def _saliency_bbox(self, image: Tensor, lam: float) -> tuple[int, int, int, int]:
        size = image.size()
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Initialize OpenCV's static fine grained saliency detector
        # and compute the saliency map.
        temp_img = image.cpu().numpy().transpose(1, 2, 0)
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliencyMap) = saliency.computeSaliency(temp_img)
        saliencyMap = (saliencyMap * 255).astype("uint8")

        maximum_indices = np.unravel_index(
            np.argmax(saliencyMap, axis=None), saliencyMap.shape
        )
        x = maximum_indices[0]
        y = maximum_indices[1]

        bbx1 = int(np.clip(x - cut_w // 2, 0, W))
        bby1 = int(np.clip(y - cut_h // 2, 0, H))
        bbx2 = int(np.clip(x + cut_w // 2, 0, W))
        bby2 = int(np.clip(y + cut_h // 2, 0, H))

        return bbx1, bby1, bbx2, bby2


class SaliencyMixFixed:
    """Fixed SaliencyMix implementation,
    which fixes wrong implementation of the authors."""

    def __init__(self, beta: float) -> None:
        self.beta = beta

    @torch.no_grad()
    def __call__(
        self, images: Tensor, labels: Tensor, sal_maps: ndarray
    ) -> tuple[Tensor, Tensor]:
        num_items = images.shape[0]

        for paste_idx in range(num_items):
            copy_idx = np.random.randint(num_items)
            lam = np.random.beta(self.beta, self.beta)
            r1, c1, r2, c2 = _pick_most_salient_pixel(sal_maps[copy_idx], lam)
            images[paste_idx, :, r1:r2, c1:c2] = images[copy_idx, :, r1:r2, c1:c2]

            copy_area = (r2 - r1) * (c2 - c1)
            total_area = images.shape[-1] * images.shape[-2]
            lam = 1 - copy_area / total_area
            labels[paste_idx] = labels[paste_idx] * lam + labels[copy_idx] * (1 - lam)

        return images, labels


class ErrorMix:
    """Mixes two images using SaliencyMix based on error matrix."""

    def __init__(self, beta: float, num_classes: int, exp_weight: float = 1.0) -> None:
        self.beta = beta
        self.num_classes = num_classes
        self.exp_weight = exp_weight
        self.error_matrix = torch.full((num_classes, num_classes), 1 / num_classes)

    def __call__(
        self, images: Tensor, labels: Tensor, sal_maps: ndarray
    ) -> tuple[Tensor, Tensor]:
        num_items = images.shape[0]
        labels = labels.argmax(1)

        for paste_idx in range(num_items):
            # Pick an index to be copied.
            prob = 1 - self.error_matrix[labels[paste_idx], labels].to(torch.float64)
            prob /= prob.sum()
            copy_idx = np.random.choice(num_items, p=prob)

            lam = np.random.beta(self.beta, self.beta)
            x1, y1, x2, y2 = _pick_most_salient_pixel(sal_maps[copy_idx], lam)
            images[paste_idx, :, x1:x2, y1:y2] = images[copy_idx, :, x1:x2, y1:y2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (images.shape[-1] * images.shape[-2]))
            labels[paste_idx] = labels[paste_idx] * lam + labels[copy_idx] * (1 - lam)

        return images, labels

    @torch.no_grad()
    def update_error_matrix(self, outputs: Tensor, labels: Tensor) -> None:
        outputs = outputs.cpu()
        labels = labels.cpu()
        num_items = outputs.shape[0]
        diff_matrix = (outputs - labels).abs()
        for i in range(num_items):
            label_index = labels[i].argmax()
            self.error_matrix[label_index, :] = (
                self.exp_weight * diff_matrix[i, :]
                + (1 - self.exp_weight) * self.error_matrix[label_index, :]
            )

                
class randsalMix:

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def __call__(self, images: Tensor, labels: Tensor, sal_maps):
        num_items = images.shape[0]
        temp_labels = labels
        
        for paste_idx in range(num_items):
            copy_idx = np.random.randint(num_items)
            lam = np.random.beta(self.beta, self.beta)
            
            val = 0
            cx1, cy1, cx2, cy2, val = self.randsal_bbox(sal_maps[copy_idx], lam, val)
            px1, py1, px2, py2, val =  self.randsal_bbox(sal_maps[paste_idx], lam, val)

            if val == -1:
                continue

            images[paste_idx, :, px1:px2, py1:py2] = images[copy_idx, :, cx1:cx2, cy1:cy2]            
            copy_area = (cx2 - cx1) * (cy2 - cy1)
            total_area = images.shape[-1] * images.shape[-2]
            lam = 1 - copy_area / total_area
            labels[paste_idx] = temp_labels[paste_idx] * lam + temp_labels[copy_idx] * (1 - lam)

        return images, labels


    def randsal_bbox(self, saliency_map, lam, val):
        size = saliency_map.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(size[0] * cut_rat)
        cut_h = int(size[1] * cut_rat)

        saliency_map[:cut_w//2, :] = 0.0
        saliency_map[-cut_w//2:, :] = 0.0
        saliency_map[:, :cut_h//2] = 0.0
        saliency_map[:, -cut_h//2:] = 0.0
        saliency_map_1d = np.array(saliency_map).flatten()

        s_nonzero = saliency_map_1d[saliency_map_1d != 0]
        s_sum = np.sum(s_nonzero)
        prob = np.zeros_like(saliency_map_1d, dtype=float)
        #copy
        if val == 0:
            if s_sum != 0:
                prob[saliency_map_1d != 0] = s_nonzero / s_sum
            val = 1
        #paste
        elif val == 1:
            if s_sum != 0:
                prob[saliency_map_1d != 0] = (255 - s_nonzero) / (255 - s_nonzero).sum()
            
        if prob.sum() == 1:
            pick = np.random.choice(len(saliency_map_1d), p=prob)
            pick_idx = np.unravel_index(pick, saliency_map.shape)
            cx, cy = int(pick_idx[0]), int(pick_idx[1])
            bbx1 = cx - cut_w // 2
            bby1 = cy - cut_h // 2
            bbx2 = cx + cut_w // 2
            bby2 = cy + cut_h // 2

            return bbx1, bby1, bbx2, bby2, val
      
        else:
            return -1, -1, -1, -1, -1
        
        
class randsalwithlabelMix:
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def __call__(self, images: Tensor, labels: Tensor, sal_maps):
        num_items = images.shape[0]
        temp_labels = copy.deepcopy(labels)
        rand_index = torch.randperm(num_items)
        for paste_idx in range(num_items):
            #copy_idx = np.random.randint(num_items)
            copy_idx = rand_index[paste_idx]
            lam = np.random.beta(self.beta, self.beta)
            
            val = 0
            cx1, cy1, cx2, cy2, val = self.randsal_bbox(sal_maps[copy_idx], lam, val)
            px1, py1, px2, py2, val =  self.randsal_bbox(sal_maps[paste_idx], lam, val)

            if val == -1:
                continue

            images[paste_idx, :, px1:px2, py1:py2] = images[copy_idx, :, cx1:cx2, cy1:cy2]

            Ss = sal_maps[copy_idx]
            Is = torch.sum(Ss)
            Ps = torch.sum(Ss[cx1 : cx2, cy1 : cy2])
            Cs = Ps / Is

            St = sal_maps[paste_idx]
            It = torch.sum(St)
            Pt = torch.sum(St[px1 : px2, py1 : py2])
            Ct = 1 - (Pt / It)

            labels[paste_idx] = temp_labels[copy_idx] * Cs + temp_labels[paste_idx] * Ct

        return images, labels


    def randsal_bbox(self, saliency_map, lam, val):
        size = saliency_map.size()
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(size[0] * cut_rat)
        cut_h = int(size[1] * cut_rat)

        saliency_map[:cut_w//2, :] = 0.0
        saliency_map[-cut_w//2:, :] = 0.0
        saliency_map[:, :cut_h//2] = 0.0
        saliency_map[:, -cut_h//2:] = 0.0

        #saliency_map = saliency_map > 100
        saliency_map_1d = np.array(saliency_map).flatten()
        s_nonzero = saliency_map_1d[saliency_map_1d != 0]
        s_sum = np.sum(s_nonzero)
        prob = np.zeros_like(saliency_map_1d, dtype=float)
        #copy
        if val == 0:
            if s_sum != 0:
                prob[saliency_map_1d != 0] = s_nonzero / s_sum
            val = 1
        #paste
        elif val == 1:
            if s_sum != 0:
                prob[saliency_map_1d != 0] = (255 - s_nonzero) / (255 - s_nonzero).sum()
               
        if prob.sum() == 1:
            pick = np.random.choice(len(saliency_map_1d), p=prob)
            pick_idx = np.unravel_index(pick, saliency_map.shape)
   
            cx, cy = int(pick_idx[0]), int(pick_idx[1])
            bbx1 = cx - cut_w // 2
            bby1 = cy - cut_h // 2
            bbx2 = cx + cut_w // 2
            bby2 = cy + cut_h // 2

            return bbx1, bby1, bbx2, bby2, val
      
        else:
            return -1, -1, -1, -1, -1
