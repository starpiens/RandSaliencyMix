import os
import argparse

import tqdm
import yaml
import torch
from torch import nn
from torch.backends import cudnn

from data import saliency_mix
from forge import create_train_loader, create_val_loader, create_model, \
                  create_optimizer, create_scheduler, create_loss_fn
from utils import AverageMeter, TopkError


def train(loader, model, optim, loss_fn, cfg, criterion_fns=[]):
    model.train()
    results = [AverageMeter() for _ in range(len(criterion_fns))]

    for idx, (inp, tar) in enumerate(tqdm.tqdm(loader, desc="Training")):
        # Perform augmentation.
        if 'augment' in cfg:
            aug_name = cfg['augment']['name']
            aug_args = cfg['augment']['args']
            if aug_name == 'SaliencyMix':
                inp, tar_a, tar_b, lam = saliency_mix(inp, tar, **aug_args)
                inp = inp.cuda()
                tar = tar.cuda()
                tar_a = tar_a.cuda()
                tar_b = tar_b.cuda()
                out = model(inp)
                loss = loss_fn(out, tar_a) * lam + loss_fn(out, tar_b) * (1 - lam)

            else:
                raise NotImplementedError(f'Augmentation "{aug_name}" is not supported.')

        else:
            inp = inp.cuda()
            tar = tar.cuda()
            out = model(inp)
            loss = loss_fn(out, tar)

        # Update train results.
        num_items = inp.shape[0]
        for i in range(len(criterion_fns)):
            with torch.no_grad():
                result = criterion_fns[i](out, tar)
            if type(result) is torch.Tensor:
                result = result.item()
            results[i].update(result, num_items)

        optim.zero_grad()
        loss.backward()
        optim.step()

    return [i.avg for i in results]


@torch.no_grad()
def validate(loader, model, criterion_fns=[]):
    model.eval()
    results = [AverageMeter() for _ in range(len(criterion_fns))]
    
    for idx, (inp, tar) in enumerate(tqdm.tqdm(loader, desc="Validating")):
        inp = inp.cuda()
        tar = tar.cuda()
        out = model(inp)

        # Update validation results.
        num_items = inp.shape[0]
        for i in range(len(criterion_fns)):
            result = criterion_fns[i](out, tar)
            if type(result) is torch.Tensor:
                result = result.item()
            results[i].update(result, num_items)
        
    return [i.avg for i in results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to config .yaml file')
    parser.add_argument('--gpu', type=str, default=None,
                        help='Index of gpu(s) to use')
    args = parser.parse_args()

    # Load config.
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        if cfg is not None:
            print(f'Loaded config at: {f.name}', flush=True)
        else:
            raise IOError('Failed to load config.')

    # Setup environment.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'Using GPU(s): {args.gpu}.', flush=True)
    else:
        print(f'Using all {torch.cuda.device_count()} GPU(s).', flush=True)
    cudnn.benchmark = True

    # Prepare training. 
    train_loader = create_train_loader(cfg['train_dataloader'])
    val_loader = create_val_loader(cfg['val_dataloader'])
    model = create_model(cfg['model'])
    model = nn.DataParallel(model)
    model = model.cuda()
    optim = create_optimizer(cfg['optimizer'], model)
    scheduler = create_scheduler(cfg['scheduler'], optim)
    loss_fn = create_loss_fn(cfg['loss'])
    topk_err_fn = TopkError(topk=(1, 5))
    
    print('Prepared training with:' )
    print('\tTrain data: class "{}" from "{}".'.format(
        train_loader.dataset.__class__.__name__,
        train_loader.dataset.__class__.__module__))
    print('\tVal data:   class "{}" from "{}".'.format(
        val_loader.dataset.__class__.__name__,
        val_loader.dataset.__class__.__module__))
    print('\tModel:      class "{}" from "{}".'.format(
        model.module.__class__.__name__,
        model.module.__class__.__module__))
    print('\tOptimizer:  class "{}" from "{}".'.format(
        optim.__class__.__name__,
        optim.__class__.__module__))
    print('\tScheduler:  class "{}" from "{}".'.format(
        scheduler.__class__.__name__,
        scheduler.__class__.__module__))
    print('\tLoss:       class "{}" from "{}".'.format(
        loss_fn.__class__.__name__,
        loss_fn.__class__.__module__))
    print()
    
    # Train.
    print('Starting training...', flush=True)
    for epoch in range(1, cfg['epochs'] + 1):
        print(f'Epoch {epoch}/{cfg["epochs"]}')
        loss, topk_err = train(train_loader, model, optim, 
                                     loss_fn, cfg, [loss_fn, topk_err_fn])
        print(f'\tLoss:     ', loss)
        print(f'\tTop-1 err:', topk_err[0])     # type: ignore
        print(f'\tTop-5 err:', topk_err[1])     # type: ignore

        loss, topk_err = validate(val_loader, model, [loss_fn, topk_err_fn])
        print(f'\tLoss:     ', loss)
        print(f'\tTop-1 err:', topk_err[0])     # type: ignore
        print(f'\tTop-5 err:', topk_err[1])     # type: ignore

        scheduler.step()
        print()


if __name__ == '__main__':
    main()
