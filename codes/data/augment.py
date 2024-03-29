import torch
import numpy as np
import cv2
import torch.nn.functional as F
from torch import Tensor
from skimage.util import random_noise


def _pick_most_salient_pixel(
    saliency_map: Tensor, lam: float
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
        self, images: Tensor, labels: Tensor, sal_maps: Tensor
    ) -> tuple[Tensor, Tensor]:
        num_items = images.shape[0]
        copy_indices = np.random.permutation(num_items)
        new_images = torch.zeros_like(images)
        new_labels = torch.zeros_like(labels)

        for paste_idx, copy_idx in enumerate(copy_indices):
            lam = np.random.beta(self.beta, self.beta)
            r1, c1, r2, c2 = _pick_most_salient_pixel(sal_maps[copy_idx], lam)

            # Adjust lambda to exactly match pixel ratio.
            copy_area = (r2 - r1) * (c2 - c1)
            total_area = images.shape[-1] * images.shape[-2]
            lam = 1 - copy_area / total_area

            new_images[paste_idx, ...] = images[paste_idx, ...]
            new_images[paste_idx, :, r1:r2, c1:c2] = images[copy_idx, :, r1:r2, c1:c2]
            new_labels[paste_idx, :] = labels[paste_idx, :] * lam
            new_labels[paste_idx, :] += labels[copy_idx, :] * (1 - lam)

        return new_images, new_labels


class ErrorMix:
    """Mixes two images using SaliencyMix based on error matrix."""

    def __init__(self, beta: float, num_classes: int, exp_weight: float = 1.0) -> None:
        self.beta = beta
        self.num_classes = num_classes
        self.exp_weight = exp_weight
        self.error_matrix = np.full(
            (num_classes, num_classes), 1 / num_classes, dtype=np.float64
        )
        self.eps = 10**-12

    def __call__(
        self, images: Tensor, labels: Tensor, sal_maps: Tensor
    ) -> tuple[Tensor, Tensor]:
        num_items = images.shape[0]
        new_images = torch.zeros_like(images)
        new_labels = torch.zeros_like(labels)
        labels_idx = labels.argmax(1)

        for paste_idx in range(num_items):
            # Pick an index to be copied.
            prob = 1 - self.error_matrix[labels_idx[paste_idx], labels_idx]
            prob += self.eps
            prob[paste_idx] = 0
            prob /= prob.sum()
            copy_idx = np.random.choice(num_items, p=prob)

            lam = np.random.beta(self.beta, self.beta)
            r1, c1, r2, c2 = _pick_most_salient_pixel(sal_maps[copy_idx], lam)

            # Adjust lambda to exactly match pixel ratio.
            copy_area = (r2 - r1) * (c2 - c1)
            total_area = images.shape[-1] * images.shape[-2]
            lam = 1 - copy_area / total_area

            new_images[paste_idx, ...] = images[paste_idx, ...]
            new_images[paste_idx, :, r1:r2, c1:c2] = images[copy_idx, :, r1:r2, c1:c2]
            new_labels[paste_idx, :] = labels[paste_idx, :] * lam
            new_labels[paste_idx, :] += labels[copy_idx, :] * (1 - lam)

        return new_images, new_labels

    @torch.no_grad()
    def update_error_matrix(self, outputs: Tensor, labels: Tensor) -> None:
        labels_arr = labels.cpu().numpy()  # Assuming already normalized
        outputs_arr = outputs.cpu().numpy()  # Needs to be normalized
        outputs_arr -= outputs_arr.min(1, keepdims=True)
        outputs_arr /= outputs_arr.sum(1, keepdims=True)
        diff_matrix = np.absolute(outputs_arr - labels_arr)

        num_items = outputs_arr.shape[0]
        for i in range(num_items):
            label_index = labels_arr[i].argmax()
            self.error_matrix[label_index, :] = (
                self.exp_weight * diff_matrix[i, :]
                + (1 - self.exp_weight) * self.error_matrix[label_index, :]
            )


class LocalMeanSaliencyMix:
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

        saliency_map_ = saliencyMap.copy()
        saliency_map_[: cut_w // 2, :] = False
        saliency_map_[:, : cut_h // 2] = False
        saliency_map_[W - cut_w // 2 :, :] = False
        saliency_map_[:, H - cut_h // 2 :] = False

        x_y_indices = np.where(saliency_map_)

        mean_values = []
        bboxes = []
        for i in range(10):
            x_index = np.random.choice(len(x_y_indices[0]))
            y_index = np.random.choice(len(x_y_indices[1]))
            random_indices = (x_y_indices[0][x_index], x_y_indices[1][y_index])
            x = random_indices[0]
            y = random_indices[1]

            bbx1 = int(np.clip(x - cut_w // 2, 0, W))
            bby1 = int(np.clip(y - cut_h // 2, 0, H))
            bbx2 = int(np.clip(x + cut_w // 2, 0, W))
            bby2 = int(np.clip(y + cut_h // 2, 0, H))

            bboxes.append([bbx1, bbx2, bby1, bby2])

            mean_values.append(saliencyMap[bbx1:bbx2, bby1:bby2].mean())

        best = np.argmax(mean_values)
        f_bbx1, f_bbx2, f_bby1, f_bby2 = bboxes[best]

        return f_bbx1, f_bby1, f_bbx2, f_bby2


class LocalMeanSaliencyMixFixed:
    """SaliencyMix implementation from the authors.
    https://github.com/afm-shahab-uddin/SaliencyMix/
    """

    def __init__(self, beta: float) -> None:
        self.beta = beta

    @torch.no_grad()
    def __call__(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        # Generate mixed sample
        num_items = images.shape[0]

        for paste_idx in range(num_items):
            copy_idx = np.random.randint(num_items)
            lam = np.random.beta(self.beta, self.beta)
            r1, c1, r2, c2 = self._saliency_bbox(images[copy_idx], lam)
            images[paste_idx, :, r1:r2, c1:c2] = images[copy_idx, :, r1:r2, c1:c2]

            # Adjust lambda to exactly match pixel ratio
            copy_area = (r2 - r1) * (c2 - c1)
            total_area = images.shape[-1] * images.shape[-2]
            lam = 1 - copy_area / total_area

            labels[paste_idx] = labels[paste_idx] * lam + labels[copy_idx] * (1 - lam)

        return images, labels

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

        saliency_map_ = saliencyMap.copy()
        saliency_map_[: cut_w // 2, :] = False
        saliency_map_[:, : cut_h // 2] = False
        saliency_map_[W - cut_w // 2 :, :] = False
        saliency_map_[:, H - cut_h // 2 :] = False

        x_y_indices = np.where(saliency_map_)

        mean_values = []
        bboxes = []
        for i in range(10):
            x_index = np.random.choice(len(x_y_indices[0]))
            y_index = np.random.choice(len(x_y_indices[1]))
            random_indices = (x_y_indices[0][x_index], x_y_indices[1][y_index])
            x = random_indices[0]
            y = random_indices[1]

            bbx1 = int(np.clip(x - cut_w // 2, 0, W))
            bby1 = int(np.clip(y - cut_h // 2, 0, H))
            bbx2 = int(np.clip(x + cut_w // 2, 0, W))
            bby2 = int(np.clip(y + cut_h // 2, 0, H))

            bboxes.append([bbx1, bbx2, bby1, bby2])

            mean_values.append(saliencyMap[bbx1:bbx2, bby1:bby2].mean())

        best = np.argmax(mean_values)
        f_bbx1, f_bbx2, f_bby1, f_bby2 = bboxes[best]

        return f_bbx1, f_bby1, f_bbx2, f_bby2


class RandSaliencyMix:
    def __init__(
        self,
        beta: float,
        use_sal_labelmix: bool,
        use_patch_prob: bool,
        use_error_mix: bool,
        num_classes: int,
        noise_std_dev: float,
    ) -> None:
        self.beta = beta
        self.use_sal_labelmix = use_sal_labelmix
        self.use_patch_prob = use_patch_prob
        self.use_error_mix = use_error_mix
        self.noise_std_dev = noise_std_dev
        self.exp_weight = 0.5
        if use_error_mix:
            self.error_matrix = np.full(
                (num_classes, num_classes), 1 / num_classes, dtype=np.float64
            )

    @torch.no_grad()
    def __call__(self, images: Tensor, labels: Tensor, sal_maps: Tensor):
        num_items = images.shape[0]
        new_images = torch.zeros_like(images)
        new_labels = torch.zeros_like(labels)
        labels_idx = labels.argmax(1)

        for paste_idx in range(num_items):
            # Pick an index to be copied.
            if self.use_error_mix:
                prob = self.error_matrix[labels_idx[paste_idx], labels_idx]
                prob += 0.0001
                prob[paste_idx] = 0
                prob /= prob.sum()
                copy_idx = np.random.choice(num_items, p=prob)
            else:
                copy_idx = np.random.choice(num_items)

            # Pick regions to copy/paste.
            lam = np.random.beta(self.beta, self.beta)
            cr1, cc1, cr2, cc2 = self.get_bbox(sal_maps[copy_idx], lam, True)
            pr1, pc1, pr2, pc2 = self.get_bbox(sal_maps[paste_idx], lam, False)
            patch = images[copy_idx, :, cr1:cr2, cc1:cc2]

            # Add noise.
            if patch.numel() != 0 and self.noise_std_dev != 0:
                patch = random_noise(
                    patch, mode="gaussian", var=self.noise_std_dev**2, clip=False
                )
                patch = torch.tensor(patch)

            # Make an augmented image.
            new_images[paste_idx, ...] = images[paste_idx, ...]
            new_images[paste_idx, :, pr1:pr2, pc1:pc2] = patch

            if self.use_sal_labelmix:
                copy_img_saliency = sal_maps[copy_idx].sum()
                paste_img_saliency = sal_maps[paste_idx].sum()

                if copy_img_saliency == 0:
                    copy_ratio = 0
                else:
                    copy_patch_saliency = sal_maps[copy_idx, cr1:cr2, cc1:cc2].sum()
                    copy_ratio = copy_patch_saliency / copy_img_saliency

                if paste_img_saliency == 0:
                    paste_ratio = 0
                else:
                    paste_patch_saliency = sal_maps[paste_idx, pr1:pr2, pc1:pc2].sum()
                    paste_ratio = 1 - paste_patch_saliency / paste_img_saliency

                norm = copy_ratio + paste_ratio
                lam = paste_ratio / norm if norm != 0 else 0.5

            # Make a label.
            new_labels[paste_idx, :] = labels[paste_idx, :] * lam
            new_labels[paste_idx, :] += labels[copy_idx, :] * (1 - lam)

        return new_images, new_labels

    def get_bbox(
        self, sal_map: Tensor, lam: float, copy_patch: bool
    ) -> tuple[int, int, int, int]:
        H, W = sal_map.shape
        patch_ratio = np.sqrt(1.0 - lam)
        h_2 = int(H * patch_ratio) // 2
        w_2 = int(W * patch_ratio) // 2

        if self.use_patch_prob:
            cum_map = torch.cumsum(sal_map, 0)
            cum_map = torch.cumsum(cum_map, 1)
            cum_map = F.pad(cum_map, (1, 0, 1, 0))
            h = h_2 * 2
            w = w_2 * 2

            prob_map = (
                cum_map[h : H + 1, w : W + 1]
                - cum_map[0 : H - h + 1, w : W + 1]
                - cum_map[h : H + 1, 0 : W - w + 1]
                + cum_map[0 : H - h + 1, 0 : W - w + 1]
            ).numpy().astype(np.float64) + 0.0001
            prob_map /= prob_map.sum()

            idx = np.random.choice(prob_map.size, p=prob_map.flatten())
            row, col = np.unravel_index(idx, prob_map.shape)
            row += h_2
            col += w_2

        else:
            prob_map = np.zeros((H, W), dtype=np.float64)
            prob_map[h_2 : H - h_2 + 1, w_2 : W - w_2 + 1] = (
                255 - sal_map[h_2 : H - h_2 + 1, w_2 : W - w_2 + 1]
                if not copy_patch
                else sal_map[h_2 : H - h_2 + 1, w_2 : W - w_2 + 1]
            ) + 0.0001
            prob_map /= prob_map.sum()

            idx = np.random.choice(prob_map.size, p=prob_map.flatten())
            row, col = np.unravel_index(idx, prob_map.shape)

        bbr1 = int(row - h_2)
        bbc1 = int(col - w_2)
        bbr2 = int(row + h_2)
        bbc2 = int(col + w_2)

        return bbr1, bbc1, bbr2, bbc2

    @torch.no_grad()
    def update_error_matrix(self, outputs: Tensor, labels: Tensor) -> None:
        if not self.use_error_mix:
            return
        
        labels_arr = labels.cpu().numpy()  # Assuming already normalized
        outputs_arr = outputs.cpu().numpy()  # Needs to be normalized
        outputs_arr -= outputs_arr.min(1, keepdims=True)
        outputs_arr /= outputs_arr.sum(1, keepdims=True)
        diff_matrix = np.absolute(outputs_arr - labels_arr)

        num_items = outputs_arr.shape[0]
        for i in range(num_items):
            label_index = labels_arr[i].argmax()
            self.error_matrix[label_index, :] = (
                self.exp_weight * diff_matrix[i, :]
                + (1 - self.exp_weight) * self.error_matrix[label_index, :]
            )
