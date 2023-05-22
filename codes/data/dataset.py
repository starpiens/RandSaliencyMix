from torchvision import transforms, datasets
from torch.utils.data import Dataset

from . import preprocess


class ImageNet(Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                preprocess.ColorJitter(brightness=0.4, 
                                       contrast=0.4, saturation=0.4),
                preprocess.Lighting(alphastd=0.1,
                                    eigval=[0.2175, 0.0188, 0.0045],
                                    eigvec=[[-0.5675, 0.7192, 0.4009],
                                            [-0.5808, -0.0045, -0.8140],
                                            [-0.5836, -0.6948, 0.4203]]),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        
        self.dataset = datasets.ImageFolder(path, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
