import os
import argparse

import tqdm
import yaml
import torch
from torch import nn
from torch.backends import cudnn

from data import saliency_mix
from forge import *
from utils import AverageMeter, calc_error

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def prepare_training(cfg):
    train_loader = create_train_loader(cfg['train_dataset'])
    val_loader = create_val_loader(cfg['val_dataset'])
    model = create_model(cfg['model'])
    model = nn.DataParallel(model)
    model = model.cuda()
    optim = create_optimizer(cfg['optimizer'], model)
    scheduler = create_scheduler(cfg['scheduler'], optim)
    loss_fn = nn.CrossEntropyLoss().cuda()
    return train_loader, val_loader, model, optim, scheduler, loss_fn


def train(loader, model, optim, loss_fn, cfg):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for i, (inp, tar) in enumerate(tqdm.tqdm(loader)):
        # Perform augmentation
        if 'augment' in cfg:
            aug_name = cfg['augment']['name']
            aug_args = cfg['augment']['args']
            if aug_name == 'SaliencyMix':
                inp, tar_a, tar_b, Cs, Ct = saliency_mix(inp, tar, **aug_args)
                inp = inp.cuda()
                tar = tar.cuda()
                tar_a = tar_a.cuda()
                tar_b = tar_b.cuda()
                Cs = torch.tensor(Cs).cuda()
                Ct = torch.tensor(Ct).cuda()
                output = model.forward(inp)
            
                # # Sum of target image losses
                loss_t = 0 
                for i in range(len(inp)):
                    loss_t += loss_fn(output[i], tar_a[i]) * Ct[i] 

                # # AVERAGE of loss_t
                loss_t /= len(inp)

                # 2
                # weight_tar_a = torch.mul(tar_a, Ct)
                # weight_tar_a = weight_tar_a.type(torch.LongTensor)
                # output = output.cuda()
                # loss_t = loss_fn(output, weight_tar_a)

                # source patch loss
                loss_s = loss_fn(output, tar_b) * Cs

                # final loss = loss_s + loss_t 
                loss = loss_s + loss_t
                print("loss_s :", loss_s)
                print("loss_t :", loss_t)
                print("loss :", loss)
                
            else:
                raise NotImplementedError(f'Augmentation "{aug_name}" is not supported.')

        else:
            inp = inp.cuda()
            tar = tar.cuda()
            output = model.forward(inp)
            loss = loss_fn(output, tar)

        err1, err5 = calc_error(output, tar, topk=(1, 5))
        num_items = inp.shape[0]
        loss_meter.update(loss.item(), num_items)
        top1_meter.update(err1, num_items)
        top5_meter.update(err5, num_items)

        optim.zero_grad()
        loss.backward()
        optim.step()

    return loss_meter.avg, top1_meter.avg, top5_meter.avg


def validate(val_loader, model):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    cudnn.benchmark = True

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        if cfg is not None:
            print('Loaded config.')
        else:
            raise IOError('Failed to load config.')
    train_loader, val_loader, model, optim, scheduler, loss_fn = prepare_training(cfg)

    for epoch in range(cfg['epochs']):
        print("epoch: ",epoch)
        train(train_loader, model, optim, loss_fn, cfg)
        scheduler.step()


if __name__ == '__main__':
    main()
