import os
import argparse
from datetime import datetime
import math

import tqdm
import yaml
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.tensorboard.writer import SummaryWriter

from data import saliency_mix
from forge import create_train_loader, create_val_loader, create_model, \
                  create_optimizer, create_scheduler, create_loss_fn
from utils import AverageMeter, TopkError


def prepare_training(cfg):
    train_loader = create_train_loader(cfg['train_dataloader'])
    val_loader = create_val_loader(cfg['val_dataloader'])
    model = create_model(cfg['model'])
    model = nn.DataParallel(model)
    model = model.cuda()
    optim = create_optimizer(cfg['optimizer'], model)
    sched = create_scheduler(cfg['scheduler'], optim)
    loss_fn = create_loss_fn(cfg['loss'])
    return train_loader, val_loader, model, optim, sched, loss_fn


def save_checkpoint(model, optim, sched, epoch, cfg, args, run_name, sv_path):
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'sched': sched.state_dict(),
        'epoch': epoch,
        'cfg': cfg,
        'tb_log_dir': args.tb_log_dir,
        'save_dir': args.save_dir,
        'run_name': run_name
    }, sv_path)


def train(loader, model, optim, loss_fn, cfg, criterion_fns=[]):
    model.train()
    results = [AverageMeter() for _ in range(len(criterion_fns))]

    for idx, (inp, tar) in enumerate(tqdm.tqdm(loader, desc="Training  ")):
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

        # Update training results.
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

        if idx == 3:
            break

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
    parser.add_argument(
        '--config', type=str,
        help='Path to config .yaml file.')
    parser.add_argument(
        '--gpu', type=str, default=None,
        help='Index of gpu(s) to use.')
    parser.add_argument(
        '--tb_log_dir', type=str, default='runs/',
        help='Tensorboard save directory location.')
    parser.add_argument(
        '--save_dir', type=str, default='save/',
        help='PyTorch save directory location.')
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Resume training from specified directory. '
             'Ignores --config, --tb_log_dir and --save_dir options, '
             'and uses saved values.')
    args = parser.parse_args()

    # Setup environment.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        print(f'Using GPU(s): {args.gpu}.', flush=True)
    else:
        print(f'Using all {torch.cuda.device_count()} GPU(s).', flush=True)
    cudnn.benchmark = True

    # Prepare training from the beginning.
    if args.resume is None:
        # Load config.
        with open(args.config, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        if cfg is not None:
            print(f'Loaded config at: {f.name}', flush=True)
        else:
            raise IOError('Failed to load config.')
            
        train_loader, val_loader, model, optim, sched, loss_fn \
            = prepare_training(cfg)
        start_epoch = 1
        now = datetime.now()
        run_name = '{}_{}'.format(
            now.strftime('%y%m%d_%H%M%S'),
            os.path.splitext(os.path.basename(args.config))[0]
        )

    # Prepare training from the checkpoint.
    else:
        checkpoint = torch.load(args.resume)
        cfg = checkpoint['cfg']
        train_loader, val_loader, model, optim, sched, loss_fn \
            = prepare_training(cfg)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        sched.load_state_dict(checkpoint['sched'])
        args.tb_log_dir = checkpoint['tb_log_dir']
        args.save_dir = checkpoint['save_dir']
        run_name = checkpoint['run_name']

    print('Prepared training with:' )
    print('\tTrain data: class "{}" from "{}" (num: {}).'.format(
        train_loader.dataset.__class__.__name__,
        train_loader.dataset.__class__.__module__,
        len(train_loader)))
    print('\tVal data:   class "{}" from "{}" (num: {}).'.format(
        val_loader.dataset.__class__.__name__,
        val_loader.dataset.__class__.__module__,
        len(val_loader)))
    print('\tModel:      class "{}" from "{}".'.format(
        model.module.__class__.__name__,
        model.module.__class__.__module__))
    print('\tOptimizer:  class "{}" from "{}".'.format(
        optim.__class__.__name__,
        optim.__class__.__module__))
    print('\tScheduler:  class "{}" from "{}".'.format(
        sched.__class__.__name__,
        sched.__class__.__module__))
    print('\tLoss:       class "{}" from "{}".'.format(
        loss_fn.__class__.__name__,
        loss_fn.__class__.__module__))
    print()

    tb_log_path = os.path.join(args.tb_log_dir, run_name)
    best_save_path = os.path.join(args.save_dir, run_name, 'best.pth')
    last_save_path = os.path.join(args.save_dir, run_name, 'last.pth')
    os.makedirs(os.path.join(args.save_dir, run_name), exist_ok=True)
    writer = SummaryWriter(tb_log_path)
    writer.add_text('configs', str(cfg))
    print(f'Tensorboard log path: {tb_log_path}')
    print(f'Best model save path: {best_save_path}')
    print(f'Last model save path: {last_save_path}')
    print()

    # Start training.
    topk_err_fn = TopkError(topk=(1, 5))
    trn_best_loss = math.inf
    trn_best_top1 = math.inf
    trn_best_top5 = math.inf
    val_best_loss = math.inf
    val_best_top1 = math.inf
    val_best_top5 = math.inf
    print(f'Starting training from epoch {start_epoch}...', flush=True)
    
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        print(f'Epoch {epoch}/{cfg["epochs"]}')

        # Train for an epoch.
        loss, topk_err = train(train_loader, model, optim, 
                               loss_fn, cfg, [loss_fn, topk_err_fn])
        loss, top1_err, top5_err = loss[0], topk_err[0], topk_err[1]
        trn_best_loss = min(trn_best_loss, loss)
        trn_best_top1 = min(trn_best_top1, top1_err)
        trn_best_top5 = min(trn_best_top5, top5_err)
        print('\tLoss:      {} (Best: {})'.format(loss, trn_best_loss))
        print('\tTop-1 err: {} (Best: {})'.format(top1_err, trn_best_top1))
        print('\tTop-5 err: {} (Best: {})'.format(top5_err, trn_best_top5))
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/top-1 err', top1_err, epoch)
        writer.add_scalar('train/top-5 err', top5_err, epoch)
        writer.add_scalar('train/loss (best)', trn_best_loss, epoch)
        writer.add_scalar('train/top-1 err (best)', trn_best_top1, epoch)
        writer.add_scalar('train/top-5 err (best)', trn_best_top5, epoch)

        # Validate.
        loss, topk_err = validate(val_loader, model, [loss_fn, topk_err_fn])
        loss, top1_err, top5_err = loss[0], topk_err[0], topk_err[1]
        val_best_loss = min(val_best_loss, loss)
        val_best_top1 = min(val_best_top1, top1_err)
        val_best_top5 = min(val_best_top5, top5_err)
        print('\tLoss:      {} (Best: {})'.format(loss, val_best_loss))
        print('\tTop-1 err: {} (Best: {})'.format(top1_err, val_best_top1))
        print('\tTop-5 err: {} (Best: {})'.format(top5_err, val_best_top5))
        writer.add_scalar('val/loss', loss, epoch)
        writer.add_scalar('val/top-1 err', top1_err, epoch)
        writer.add_scalar('val/top-5 err', top5_err, epoch)
        writer.add_scalar('val/loss (best)', val_best_loss, epoch)
        writer.add_scalar('val/top-1 err (best)', val_best_top1, epoch)
        writer.add_scalar('val/top-5 err (best)', val_best_top5, epoch)

        sched.step()
        writer.flush()
        print()

        # Save model for best top-1 error on validation set. 
        if val_best_top1 == top1_err:
            save_checkpoint(model, optim, sched, epoch + 1, 
                            cfg, args, run_name, best_save_path)

        save_checkpoint(model, optim, sched, epoch + 1, 
                            cfg, args, run_name, last_save_path)

    writer.close()


if __name__ == '__main__':
    main()
