from typing import Callable, Tuple

import torch
from torch import Tensor
from torch.nn import Module, DataParallel
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
import torchvision

import data
import models


def prepare_training(
    cfg: dict,
) -> Tuple[
    DataLoader,
    DataLoader,
    Module,
    Optimizer,
    (LRScheduler | None),
    Module,
    (Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]] | None),
]:
    train_loader = create_train_loader(cfg["train_dataloader"])
    val_loader = create_val_loader(cfg["val_dataloader"])
    model = create_model(cfg["model"])
    model = DataParallel(model)
    model = model.cuda()
    optim = create_optimizer(cfg["optimizer"], model)
    loss_fn = create_loss_fn(cfg["loss"])
    sched = create_scheduler(cfg["scheduler"], optim) if "scheduler" in cfg else None
    aug_fn = create_augment_fn(cfg["augment"]) if "augment" in cfg else None
    return train_loader, val_loader, model, optim, sched, loss_fn, aug_fn


def create_dataset(dataset_cfg: dict) -> Dataset:
    """Creates a dataset."""
    name = dataset_cfg["name"]
    args = dataset_cfg.get("args", {})
    try:
        dataset_cls = getattr(data, name)
    except AttributeError:
        dataset_cls = getattr(torchvision.datasets, name)
    dataset = dataset_cls(**args)
    return dataset


def create_train_loader(train_dataloader_cfg: dict) -> DataLoader:
    """Creates a data loader for the training."""
    dataset = create_dataset(train_dataloader_cfg["dataset"])
    args = train_dataloader_cfg.get("args", {})
    args.setdefault("batch_size", 1)
    args.setdefault("shuffle", True)
    args.setdefault("num_workers", 8)
    args.setdefault("pin_memory", True)
    args.setdefault("drop_last", True)
    loader = DataLoader(dataset, **args)
    return loader


def create_val_loader(val_dataloader_cfg: dict) -> DataLoader:
    """Creates a data loader for the validation."""
    dataset = create_dataset(val_dataloader_cfg["dataset"])
    args = val_dataloader_cfg.get("args", {})
    args.setdefault("batch_size", 1)
    args.setdefault("shuffle", False)
    args.setdefault("num_workers", 8)
    args.setdefault("pin_memory", True)
    args.setdefault("drop_last", False)
    loader = DataLoader(dataset, **args)
    return loader


def create_model(model_cfg: dict) -> Module:
    """Creates a model."""
    name = model_cfg["name"]
    args = model_cfg.get("args", {})
    try:
        model_cls = getattr(models, name)
    except AttributeError:
        model_cls = getattr(torchvision.models, name)
    model = model_cls(**args)
    return model


def create_optimizer(optim_cfg: dict, model: Module) -> Optimizer:
    """Creates an optimizer."""
    name = optim_cfg["name"]
    args = optim_cfg.get("args", {})
    optim_cls = getattr(torch.optim, name)
    optim = optim_cls(model.parameters(), **args)
    return optim


def create_scheduler(sch_cfg: dict, optim: Optimizer) -> LRScheduler:
    """Creates a learning rate scheduler."""
    name = sch_cfg["name"]
    args = sch_cfg.get("args", {})
    sch_cls = getattr(torch.optim.lr_scheduler, name)
    sch = sch_cls(optim, **args)
    return sch


def create_loss_fn(loss_cfg: dict) -> Module:
    """Creates a loss function."""
    name = loss_cfg["name"]
    args = loss_cfg.get("args", {})
    loss_cls = getattr(torch.nn.modules.loss, name)
    loss_fn = loss_cls(**args)
    return loss_fn


def create_augment_fn(
    augment_cfg: dict,
) -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Creates an augmentation function."""
    name = augment_cfg["name"]
    print(name)
    args = augment_cfg.get("args", {})
    aug_cls = getattr(data.augment, name)
    aug_fn = aug_cls(**args)
    return aug_fn
