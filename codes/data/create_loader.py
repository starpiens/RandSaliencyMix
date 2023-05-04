from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets

from . import preprocess


def create_train_loader(train_dataset_cfg):
    name = train_dataset_cfg['name']
    path = train_dataset_cfg['path']
    batch_size = train_dataset_cfg['batch_size']

    if name == 'imagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            preprocess.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            preprocess.Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    else:
        raise NotImplementedError('Only imagenet is supported.')

    return loader


def create_val_loader(val_dataset_cfg):
    name = val_dataset_cfg['name']
    path = val_dataset_cfg['path']
    batch_size = val_dataset_cfg['batch_size']

    if name == 'imagenet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    else:
        raise NotImplementedError('Only imagenet is supported.')

    return loader
