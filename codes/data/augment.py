import queue
from multiprocessing import Queue, Process

import torch
import numpy as np
import cv2
from torch import Tensor
from numpy import ndarray


def _compute_saliency_maps(imgs: Tensor, num_workers=16):
    def _worker(idx, imgs, result_queue):
        computer = cv2.saliency.StaticSaliencyFineGrained_create()
        result = np.zeros(imgs.shape[:-1], dtype="uint8")
        num_imgs = imgs.shape[0]
        for i in range(num_imgs):
            (success, saliency_map) = computer.computeSaliency(imgs[i])
            saliency_map = (saliency_map * 255).astype("uint8")
            result[i] = saliency_map
        result_queue.put((idx, result))

    imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1)
    num_imgs = imgs.shape[0]
    batch_size = num_imgs // num_workers
    assert num_imgs % num_workers == 0

    result_queue = Queue()
    processes = []
    for i in range(num_workers):
        imgs_batch = imgs[i * batch_size : (i + 1) * batch_size, ...]
        p = Process(target=_worker, args=(i, imgs_batch, result_queue))
        processes.append(p)
        p.start()

    results = []
    for _ in range(num_workers):
        results.append(result_queue.get(timeout=2))
    results.sort(key=lambda r: r[0])
    results = np.concatenate([r[1] for r in results], axis=0)

    for p in processes:
        p.join()

    return results


def _pick_most_salient_pixel(saliency_map: ndarray, lam: float):
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
    def __call__(self, inp: Tensor, tar: Tensor):
        # Generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(inp.shape[0])
        tar_a = tar
        tar_b = tar[rand_index]
        bbx1, bby1, bbx2, bby2 = self._saliency_bbox(inp[rand_index[0]], lam)
        inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inp.shape[-1] * inp.shape[-2]))
        # Compute output
        inp_var = torch.autograd.Variable(inp, requires_grad=True)
        tar = tar_a * lam + tar_b * (1 - lam)
        return inp_var, tar

    def _saliency_bbox(self, img: Tensor, lam: float):
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
    def __call__(self, inp: Tensor, tar: Tensor):
        num_items = inp.shape[0]
        saliency_maps = _compute_saliency_maps(inp)

        for paste_idx in range(num_items):
            copy_idx = np.random.randint(num_items)
            lam = np.random.beta(self.beta, self.beta)
            x1, y1, x2, y2 = _pick_most_salient_pixel(saliency_maps[copy_idx], lam)
            inp[paste_idx, :, x1:x2, y1:y2] = inp[copy_idx, :, x1:x2, y1:y2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (inp.shape[-1] * inp.shape[-2]))
            tar[paste_idx] = tar[paste_idx] * lam + tar[copy_idx] * (1 - lam)

        return inp, tar


class ErrorMix:
    """Mixes two images using SaliencyMix based on error matrix."""

    def __init__(self, beta: float, num_classes: int, exp_weight: float = 1.0) -> None:
        self.beta = beta
        self.num_classes = num_classes
        self.exp_weight = exp_weight
        self.error_matrix = torch.full((num_classes, num_classes), 1 / num_classes)

    def __call__(self, inp: Tensor, tar: Tensor):
        num_items = inp.shape[0]
        labels = tar.argmax(1)
        saliency_maps = _compute_saliency_maps(inp)

        for paste_idx in range(num_items):
            # Pick an index to be copied.
            prob = 1 - self.error_matrix[labels[paste_idx], labels].to(torch.float64)
            prob /= prob.sum()
            copy_idx = np.random.choice(num_items, p=prob)

            lam = np.random.beta(self.beta, self.beta)
            x1, y1, x2, y2 = _pick_most_salient_pixel(saliency_maps[copy_idx], lam)
            inp[paste_idx, :, x1:x2, y1:y2] = inp[copy_idx, :, x1:x2, y1:y2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (inp.shape[-1] * inp.shape[-2]))
            tar[paste_idx] = tar[paste_idx] * lam + tar[copy_idx] * (1 - lam)

        return inp, tar

    @torch.no_grad()
    def update_error_matrix(self, out: Tensor, tar: Tensor) -> None:
        num_items = out.shape[0]
        diff_matrix = (out - tar).abs()
        for i in range(num_items):
            label = tar[i].argmax()
            self.error_matrix[label, :] = (
                self.exp_weight * diff_matrix[i, :]
                + (1 - self.exp_weight) * self.error_matrix[label, :]
            )


class randsalMix:

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def __call__(self, inp: Tensor, tar: Tensor):
        num_items = inp.shape[0]
        center_list = [0] * num_items
        saliency_maps = _compute_saliency_maps(inp)

        for i in range(num_items):
            self.pick_bbox_center(i, saliency_maps[i], center_list)

        copy_idx_list = torch.randperm(num_items)
        for paste_idx in range(num_items):
            copy_idx = copy_idx_list[paste_idx]
            lam = np.random.beta(self.beta, self.beta)
            cx1, cy1, cx2, cy2, px1, py1, px2, py2 = self.get_bbox(inp[paste_idx], paste_idx, copy_idx, lam, center_list)
            inp[paste_idx, :, px1:px2, py1:py2] = inp[copy_idx, :, cx1:cx2, cy1:cy2]
            
            lam = 1 - ((cx2 - cx1) * (cy2 - cy1) / (inp.shape[-1] * inp.shape[-2]))
            tar[paste_idx] = tar[paste_idx] * lam + tar[copy_idx] * (1 - lam)

        return inp, tar


    def pick_bbox_center(self, idx, saliency_maps, center_list):
        # make saliency map -> probability map
        saliency_map_1d = np.array(saliency_maps).flatten()
        copy_prob = saliency_map_1d / saliency_map_1d.sum()
        paste_prob = (255 - saliency_map_1d) / (255 - saliency_map_1d).sum()

        cc = np.random.choice(copy_prob, 1, p=copy_prob, replace=False)
        pp = np.random.choice(paste_prob, 1, p=paste_prob, replace=False)

        #find the index
        copy_idx = np.unravel_index(np.where(copy_prob == float(cc)), saliency_maps.shape)
        if len(copy_idx[0][0]) > 1:
            c = np.random.randint(0, len(copy_idx[0][0])-1)
        else:
            c = 0

        paste_idx = np.unravel_index(np.where(paste_prob == float(pp)), saliency_maps.shape)
        if len(paste_idx[0][0]) > 1:
            p = np.random.randint(0, len(paste_idx[0][0])-1)
        else:
            p = 0
  
        center_list[idx] = [copy_idx[0][0][c], copy_idx[1][0][c]],[paste_idx[0][0][p], paste_idx[1][0][p]]


    def get_bbox(self, img, idx, rand_idx, lam, center_list):
        size = img.size()
        width = size[1]
        height = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(width * cut_rat)
        cut_h = int(height * cut_rat)

        cx, cy = center_list[rand_idx][0]
        px, py = center_list[idx][1]

        cbbx1 = np.clip(cx - cut_w // 2, 0, width)
        cbby1 = np.clip(cy - cut_h // 2, 0, height)
        cbbx2 = np.clip(cx + cut_w // 2, 0, width)
        cbby2 = np.clip(cy + cut_h // 2, 0, height)

        # Adjust cut_w/cut_h
        x_l = cx - cbbx1
        x_h = cbbx2 - cx
        y_l = cy - cbby1
        y_h = cbby2 - cy

        # Adjust cbbox to exactly match paste region
        if (px - x_l < 0):
            cbbx1 = cx - px

        if (py - y_l < 0):
            cbby1 = cy - py

        if (px + x_h > width):
            cbbx2 = cx + (width - px)
    
        if (py + y_h > height):
            cbby2 = cy + (height - py)

        # Adjust cut_w/cut_h
        x_l = cx - cbbx1
        x_h = cbbx2 - cx
        y_l = cy - cbby1
        y_h = cbby2 - cy
    
        pbbx1 = px - x_l
        pbby1 = py - y_l
        pbbx2 = px + x_h
        pbby2 = py + y_h

        return cbbx1, cbby1, cbbx2, cbby2, pbbx1, pbby1, pbbx2, pbby2
