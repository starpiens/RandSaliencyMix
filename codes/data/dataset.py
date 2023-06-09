import torch
import cv2
import numpy as np
from torch import Tensor
from torch.nn.functional import one_hot
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from numpy import ndarray

from . import preprocess


class ImageNet(Dataset):
    """ImageNet dataset."""

    def __init__(self, path: str, num_classes: int, train: bool = True):
        super().__init__()
        self.num_classes = num_classes

        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
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
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        self.dataset = datasets.ImageFolder(path, transform=transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image, label = self.dataset[index]
        label = torch.tensor(label)
        label = one_hot(label, num_classes=self.num_classes)
        label = label.float()
        return image, label


class ImageNetWithSaliencyMap(ImageNet):
    """ImageNet dataset with saliency map."""

    def __init__(self, path: str, num_classes: int, train=True):
        super().__init__(path, num_classes, train)
        self.saliency_computer = cv2.saliency.StaticSaliencyFineGrained_create()

    def __len__(self) -> int:
        return super().__len__()

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, ndarray]:
        image, label = super().__getitem__(index)
        image_arr = image.numpy().transpose(1, 2, 0)
        image_arr = (
            (image_arr - image_arr.min()) / (image_arr.max() - image_arr.min()) * 255
        ).astype(np.uint8)
        success, saliency_map = self.saliency_computer.computeSaliency(image_arr)
        if not success:
            raise RuntimeError("Failed to compute saliency map.")
        saliency_map = (saliency_map * 255).astype("uint8")
        return image, label, saliency_map
