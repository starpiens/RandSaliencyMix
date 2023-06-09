from multiprocessing import Queue, Process

import torch
from torch import Tensor
from torch.nn.functional import normalize
import numpy as np
import cv2
from numpy import ndarray


class SaliencyLabelMix:
    """SaliencyMix implementation from the authors.
    https://github.com/afm-shahab-uddin/SaliencyMix/
    """

    def __init__(self, beta: float) -> None:
        self.beta = beta

    @torch.no_grad()
    def __call__(self, inp: Tensor, tar: Tensor, sal_maps: ndarray) -> tuple[Tensor, Tensor]:
        # Generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = np.random.randint(inp.shape[0])
        tar_src = tar[rand_index]
        Ss = sal_maps[rand_index]
        bbx1, bby1, bbx2, bby2 = self._saliency_bbox(inp[rand_index], lam, Ss)
        Is = torch.sum(Ss)
        Ps = torch.sum(Ss[bbx1 : bbx2, bby1 : bby2])
        
        inp[:, :, bbx1:bbx2, bby1:bby2] = inp[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # Perform SaliencyLabelMix with prob 0.5
        Cs = Ps / Is
        prob = np.random.rand(1)
        if prob > 0.5:
            for index, inp_img in enumerate(inp): 
                St = sal_maps[index]
                It = torch.sum(St)
                Pt = torch.sum(St[bbx1 : bbx2, bby1 : bby2])
                Ct = 1 - (Pt / It)
                # Define new label definition
                tar[index] = tar_src * Cs + tar[index] * Ct

        # Else: don't perform SaliencyLabelMix
        inp_var = torch.autograd.Variable(inp, requires_grad=True)
        
        return inp_var, tar

    def _saliency_bbox(self, img: Tensor, lam: float, saliencyMap: ndarray) -> tuple[int, int, int, int]:
        size = img.size()
        W = size[1]
        H = size[2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

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


class ErrorMix:
    """Mixes two images using SaliencyMix based on error matrix."""

    def __init__(self, beta: float, num_classes: int, exp_weight: float = 1.0) -> None:
        self.beta = beta
        self.num_classes = num_classes
        self.exp_weight = exp_weight
        self.error_matrix = torch.full((num_classes, num_classes), 1 / num_classes)

    def __call__(self, inp: Tensor, tar: Tensor) -> tuple[Tensor, Tensor]:
        num_items = inp.shape[0]
        labels = tar.argmax(1)
        for paste_idx in range(num_items):
            prob = 1 - self.error_matrix[labels[paste_idx], labels].to(torch.float64)
            prob /= prob.sum()
            copy_idx = np.random.choice(num_items, p=prob)
            lam = np.random.beta(self.beta, self.beta)
            x1, y1, x2, y2 = _saliency_bbox(inp[copy_idx], lam)
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

