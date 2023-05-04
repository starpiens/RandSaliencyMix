import os
import argparse

import yaml
import torch
from torch import nn

import data
import models


def prepare_training(cfg):
    train_dataset_cfg = cfg['train_dataset']
    val_dataset_cfg = cfg['val_dataset']
    model_cfg = cfg['model']

    train_loader = data.create_train_loader(train_dataset_cfg)
    val_loader = data.create_val_loader(val_dataset_cfg)
    model = models.create_model(model_cfg)
    model = model.to('cuda')
    model = nn.DataParallel(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--gpu')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    prepare_training(config)


if __name__ == '__main__':
    main()
