import data
import models

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets


def create_train_loader(train_dataset_cfg):
    name = train_dataset_cfg['name']
    path = train_dataset_cfg['path']
    batch_size = train_dataset_cfg['batch_size']

    if name == 'ImageNet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            data.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            data.Lighting(alphastd=0.1,
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
        raise NotImplementedError(f'Dataset "{name}" is not supported.')

    return loader


def create_val_loader(val_dataset_cfg):
    name = val_dataset_cfg['name']
    path = val_dataset_cfg['path']
    batch_size = val_dataset_cfg['batch_size']

    if name == 'ImageNet':
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
        raise NotImplementedError(f'Dataset "{name}" is not supported.')

    return loader


def create_model(model_cfg):
    name = model_cfg['name']
    args = model_cfg['args']

    if name == 'ResNet':
        model = models.ResNet(**args)
    else:
        raise NotImplementedError(f'Model "{name}" is not supported.')

    return model


def create_optimizer(optim_cfg, model):
    name = optim_cfg['name']
    args = optim_cfg['args']

    if name == 'SGD':
        optim = torch.optim.SGD(model.parameters(), **args)
    else:
        raise NotImplementedError(f'Optimizer "{name}" is not supported.')

    return optim


def create_scheduler(sch_cfg, optim):
    name = sch_cfg['name']
    args = sch_cfg['args']

    if name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, **args)
    else:
        raise NotImplementedError(f'Scheduler "{name}" is not supported.')

    return scheduler
