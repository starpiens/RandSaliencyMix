import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision

import data
import models


def create_dataset(dataset_cfg: dict) \
        -> torch.utils.data.Dataset:
    """Creates a dataset."""
    name = dataset_cfg['name']
    args = dataset_cfg.get('args', {})
    try:
        dataset_cls = getattr(data, name)
    except AttributeError:
        dataset_cls = getattr(torchvision.datasets, name)
    dataset = dataset_cls(**args)
    return dataset


def create_train_loader(train_dataloader_cfg: dict) \
        -> torch.utils.data.DataLoader:
    """Creates a data loader for the training."""
    dataset = create_dataset(train_dataloader_cfg['dataset'])
    args = train_dataloader_cfg.get('args', {})
    args.setdefault('batch_size', 1)
    args.setdefault('shuffle', True)
    args.setdefault('num_workers', 8)
    args.setdefault('pin_memory', True)
    args.setdefault('drop_last', True)
    loader = DataLoader(dataset, **args)
    return loader


def create_val_loader(val_dataloader_cfg: dict) \
        -> torch.utils.data.DataLoader:
    """Creates a data loader for the validation."""
    dataset = create_dataset(val_dataloader_cfg['dataset'])
    args = val_dataloader_cfg.get('args', {})
    args.setdefault('batch_size', 1)
    args.setdefault('shuffle', False)
    args.setdefault('num_workers', 8)
    args.setdefault('pin_memory', True)
    args.setdefault('drop_last', False)
    loader = DataLoader(dataset, **args)
    return loader


def create_model(model_cfg: dict) \
        -> torch.nn.Module:
    """Creates a model."""
    name = model_cfg['name']
    args = model_cfg.get('args', {})
    try:
        model_cls = getattr(models, name)
    except AttributeError:
        model_cls = getattr(torchvision.models, name)
    model = model_cls(**args)
    return model


def create_optimizer(optim_cfg: dict,
                     model: torch.nn.Module) \
        -> torch.optim.Optimizer:
    """Creates an optimizer."""
    name = optim_cfg['name']
    args = optim_cfg.get('args', {})
    optim_cls = getattr(torch.optim, name)
    optim = optim_cls(model.parameters(), **args)
    return optim


def create_scheduler(sch_cfg: dict,
                     optim: torch.optim.Optimizer) \
        -> torch.optim.lr_scheduler.LRScheduler:
    """Creates a learning rate scheduler."""
    name = sch_cfg['name']
    args = sch_cfg.get('args', {})
    sch_cls = getattr(torch.optim.lr_scheduler, name)
    sch = sch_cls(optim, **args)
    return sch


def create_loss_fn(loss_cfg: dict) \
        -> torch.nn.Module:
    """Creates a loss function."""
    name = loss_cfg['name']
    args = loss_cfg.get('args', {})
    loss_cls = getattr(torch.nn.modules.loss, name)
    loss_fn = loss_cls(**args)
    return loss_fn
