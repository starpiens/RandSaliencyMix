from typing import Literal

import torch
import cv2
import numpy as np
from torch import Tensor
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from torchvision.transforms import functional as TF
from numpy import ndarray

from . import preprocess


class ImageNet(Dataset):
    """ImageNet dataset."""

    def __init__(self, path: str, num_classes: int, train: bool) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Define transform.
        if train:
            self.pre_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
            self.post_transform = transforms.Compose(
                [
                    preprocess.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4
                    ),
                    preprocess.Lighting(
                        alphastd=0.1,
                        eigval=[0.2175, 0.0188, 0.0045],
                        eigvec=[
                            [-0.5675, 0.7192, 0.4009],
                            [-0.5808, -0.0045, -0.8140],
                            [-0.5836, -0.6948, 0.4203],
                        ],
                    ),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.pre_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
            self.post_transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.dataset = datasets.ImageFolder(path)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        image = self.pre_transform(image)
        image = self.post_transform(image)
        label = torch.tensor(label)
        label = one_hot(label, num_classes=self.num_classes)
        label = label.float()
        return image, label

    def _getitem_no_transform(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        label = torch.tensor(label)
        label = one_hot(label, num_classes=self.num_classes)
        label = label.float()
        return image, label


class ImageNetWithSaliencyMap(ImageNet):
    """ImageNet dataset with saliency map."""

    def __init__(
        self,
        path: str,
        num_classes: int,
        train: bool,
        cache_salmap: bool = True,
    ) -> None:
        super().__init__(path, num_classes, train)
        self.cache_salmap = cache_salmap
        if self.cache_salmap:
            self.salmaps = [None] * len(self)
        self.saliency_computer = cv2.saliency.StaticSaliencyFineGrained_create()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if self.cache_salmap:
            image, label = self._getitem_no_transform(index)
            image = TF.to_tensor(image)
            if self.salmaps[index] is None:
                salmap = self.compute_salmap(image)
                salmap = salmap.to(torch.float32)
                salmap /= 255
                salmap = torch.unsqueeze(salmap, dim=0)
                self.salmaps[index] = salmap

            salmap = self.salmaps[index]
            img_and_sal = torch.cat((image, salmap), dim=0)
            img_and_sal = TF.to_pil_image(img_and_sal)
            img_and_sal = self.pre_transform(img_and_sal)

            image, salmap = torch.split(img_and_sal, [3, 1], dim=0)
            salmap = torch.squeeze(salmap, dim=0)
            salmap *= 255
            salmap = salmap.to(torch.uint8)
            image = self.post_transform(image)
            return image, label, salmap

        else:
            image, label = super().__getitem__(index)
            salmap = self.compute_salmap(image)
            return image, label, salmap

    def compute_salmap(self, image: Tensor) -> Tensor:
        img_arr = image.numpy().transpose(1, 2, 0)
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        img_arr = (img_arr * 255).astype(np.uint8)
        success, salmap = self.saliency_computer.computeSaliency(img_arr)
        if not success:
            raise RuntimeError("Failed to compute saliency map.")
        salmap = (salmap * 255).astype("uint8")
        salmap = torch.from_numpy(salmap)
        return salmap


class CIFAR(Dataset):
    """CIFAR-10/100 dataset."""

    def __init__(
        self,
        path: str,
        num_classes: Literal[10, 100],
        data_aug: bool,
        train: bool,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.data_aug = data_aug

        # Define transform.
        if train and data_aug:
            self.pre_transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.pre_transform = transforms.ToTensor()
        self.post_transform = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )

        if num_classes == 10:
            self.dataset = datasets.CIFAR10(root=path, train=train, download=True)
        else:
            self.dataset = datasets.CIFAR100(root=path, train=train, download=True)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        image = self.pre_transform(image)
        image = self.post_transform(image)
        label = torch.tensor(label)
        label = one_hot(label, num_classes=self.num_classes)
        label = label.float()
        return image, label

    def _getitem_no_transform(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        label = torch.tensor(label)
        label = one_hot(label, num_classes=self.num_classes)
        label = label.float()
        return image, label


class CIFARWithSaliencyMap(CIFAR):
    """CIFAR-10/100 dataset with saliency map."""

    def __init__(
        self,
        path: str,
        num_classes: Literal[10, 100],
        data_aug: bool,
        train: bool,
        cache_salmap: bool = True,
    ) -> None:
        super().__init__(path, num_classes, data_aug, train)
        self.cache_salmap = cache_salmap
        if self.cache_salmap:
            self.salmaps = [None] * len(self)
        self.saliency_computer = cv2.saliency.StaticSaliencyFineGrained_create()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, ndarray]:
        if self.cache_salmap:
            image, label = self._getitem_no_transform(index)
            image = TF.to_tensor(image)
            if self.salmaps[index] is None:
                salmap = self.compute_salmap(image)
                salmap = salmap.to(torch.float32)
                salmap /= 255
                salmap = torch.unsqueeze(salmap, dim=0)
                self.salmaps[index] = salmap

            salmap = self.salmaps[index]
            img_and_sal = torch.cat((image, salmap), dim=0)
            img_and_sal = TF.to_pil_image(img_and_sal)
            img_and_sal = self.pre_transform(img_and_sal)

            image, salmap = torch.split(img_and_sal, [3, 1], dim=0)
            salmap = torch.squeeze(salmap, dim=0)
            salmap *= 255
            salmap = salmap.to(torch.uint8)
            image = self.post_transform(image)
            return image, label, salmap

        else:
            image, label = super().__getitem__(index)
            salmap = self.compute_salmap(image)
            return image, label, salmap

    def compute_salmap(self, image: Tensor) -> Tensor:
        img_arr = image.numpy().transpose(1, 2, 0)
        img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        img_arr = (img_arr * 255).astype(np.uint8)
        success, salmap = self.saliency_computer.computeSaliency(img_arr)
        if not success:
            raise RuntimeError("Failed to compute saliency map.")
        salmap = (salmap * 255).astype("uint8")
        salmap = torch.from_numpy(salmap)
        return salmap
