from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from os.path import join, exists, isdir, isfile, dirname, abspath
import pprint
import shutil
import copy
import logging
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config, update_config, update_dir, get_model_name
from core.loss import WeightedL1Loss, WeightedL2Loss
from core.loss import SSIM, EdgeSSIM, FlowGrad, EdgeGrad
from core.function_st1 import train, validate
from utils.utils import get_optimizer, get_scheduler, load_partial_weights 
from utils.utils import save_checkpoint, create_logger
from utils.utils import Plabeler, save_infer2coco

import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Landmark Bert Model')
    # general
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', default=config.PRINT_FREQ, type=int, help='frequency of logging')
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    parser.add_argument('--resume', type=str, default='', help='resume path')
    parser.add_argument('--perf_name', type=str, default='MRE', choices=['MRE', 'AP'], help='perf name for evaluate')
    args = parser.parse_args()

    return args

def reset_config(config, args):
    if args.gpus: config.GPUS = args.gpus
    if args.workers: config.WORKERS = args.workers
    if args.resume: config.RESUME = args.resume

def _make_model(cfg, is_train=True):
    bb_name, fuse_name = cfg.MODEL.NAME
    models_dict = {}
    ### Backbone model ###################
    backbone = eval(f'models.Backbones.{bb_name}.get_backbone')(\
        cfg, is_train, collect_feat=False)
    models_dict['backbone'] = backbone
    ### Bert model #######################
    fuse_model = eval(f'models.{fuse_name}.get_net')(cfg)
    models_dict['fuse'] = fuse_model
    ### copy model files #################
    for k, v in models_dict.items():
        models_dict[k] = torch.nn.DataParallel(v).cuda()
    return models_dict

def save_files(args, cfg, final_output_dir):
    pose_name, model_name = cfg.MODEL.NAME
    this_dir = dirname(__file__)
    model_dir = join(this_dir, '../lib/models')
    core_dir = join(this_dir, '../lib/core')
    # copy files for backup
    for fname in ['Backbones/hrnet.py', f'{model_name}.py',\
         'Transformers/transform_block.py']:
        fpath = join(model_dir, fname)
        shutil.copy2(fpath, final_output_dir)
    # copy other files
    for fpath in [args.cfg, abspath(__file__), \
        join(core_dir, 'function_st1.py'), \
        join(core_dir, 'loss.py')]:
        shutil.copy2(fpath, final_output_dir)
    
def _make_loss(cfg):
    loss_dict = {}    
    pose_name, model_name = cfg.MODEL.NAME
    im_channel = 1
    im_w, im_h = cfg.MODEL.IMAGE_SIZE
    hm_w, hm_h = cfg.MODEL.EXTRA.HEATMAP_SIZE
    # reconstruction loss 
    loss_dict['l1'] = WeightedL1Loss(size_average=True)
    loss_dict['l2'] = WeightedL2Loss(size_average=True) #nn.MSELoss(reduction='mean')
    loss_dict['sim'] = SSIM(im_channel, mode='ssim', reduction='mean')
    loss_dict['esim'] = EdgeSSIM(im_channel, (im_w, im_h), mode='ssim', reduction='mean')
    # smoothness loss 
    loss_dict['grad'] = FlowGrad(penalty='l2')
    loss_dict['egrad'] = EdgeGrad((im_w, im_h), penalty='l2', corner_flag=True, T=cfg.LOSS.TEMPERATURE)
   
    if cfg.USE_GPU:
        for name, loss in loss_dict.items():
            loss_dict[name] = loss.cuda()
    return loss_dict

def epoch2lrdecay(epoch, config):
    '''
    epoch -> lr decay (lrbase * lr_decay)
    '''
    base_lr = config.TRAIN.LR 
    stage1 = 250; stage2 = config.TRAIN.END_EPOCH
    if epoch < stage1: 
        lr = 1e-4 
    else: 
        duration = epoch - stage1
        total = stage2 - stage1
        max_lr = 1e-4 
        min_lr = 5e-5 
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(duration / total * np.pi))
    return lr / base_lr

def _make_data(cfg, split_name, is_train, **kwargs):
    split_name = split_name.upper()
    if 'ann_file_path' not in kwargs:
        kwargs['ann_file_path'] = getattr(cfg.DATASET, f'{split_name}_ANN')

    dset = eval(f'dataset.{cfg.DATASET.DATASET}')(
        cfg, cfg.DATASET.ROOT, getattr(cfg.DATASET, f'{split_name}_SET'),
        is_train,  **kwargs)
    
    bsize = cfg.TRAIN.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE 
    shuffle = cfg.TRAIN.SHUFFLE if is_train else False 
    dloader = torch.utils.data.DataLoader(dset, batch_size=bsize, 
        shuffle=shuffle, num_workers=cfg.WORKERS, pin_memory=True)
    return dset, dloader

def main():    
    args = parse_args()
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
    logger.info('=> USING GPUS {}'.format(config.GPUS))
    
    models_dict = _make_model(config, is_train=True)
    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0, 'valid_global_steps': 0,
    }
    
    # make loss & optimizer
    loss_dict = _make_loss(config)
    optimizer = get_optimizer(config, models_dict)

    # Data loading code
    means_3ch = [0.485, 0.456, 0.406]
    stds_3ch = [0.229, 0.224, 0.225]
    # use means of natural image stats for one-channel norm
    mean_1ch = sum(means_3ch) / 3.0
    std_1ch = sum(stds_3ch) / 3.0
    norm= transforms.Normalize(mean=[ mean_1ch ], std=[ std_1ch ])
    inv_norm = transforms.Compose([
        transforms.Normalize(mean=[0.], std=[1 / std_1ch]),
        transforms.Normalize(mean=[-1.0 * mean_1ch], std=[1.])
    ])
    transform = transforms.Compose([transforms.ToTensor(), norm])

    data_name = config.DATASET.DATASET
    perf_name = args.perf_name
    
    if perf_name == 'AP':
        compare_func = lambda cur, best: cur > best
        best_perf = 0.0
        reduce_mode = 'max'
    elif perf_name == 'MRE':
        compare_func = lambda cur, best: cur < best
        best_perf = 1e8
        reduce_mode = 'min'
    
    # make data 
    train_set, train_loader = _make_data(config, 'TRAIN', True,
        transform=transform, return_mask=True)
    val_set, val_loader = _make_data(config, 'VAL', False, 
        transform=transform, return_mask=False)
    
    val_exemplar_set, val_exemplar_loader = _make_data(config, 'TRAIN', False,
        transform=transform, subdir=config.DATASET.TRAIN_SUBDIR, return_mask=True)

    train_set_for_test, train_loader_for_test = _make_data(config, 'TRAIN', False,\
        transform=transform, return_mask=True)

    # reset config attr (dataset)
    config.DATASET.POINT_NAMES = val_set.point_names
    config.DATASET.LINE_PAIRS = val_set.line_pairs
    
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH  
    scheduler_type = config.TRAIN.SCHEDULER  
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e:epoch2lrdecay(e, config))

    best_model = False
    if config.RESUME:
        resume_state = torch.load(config.RESUME)
        for k, model in models_dict.items():
            if 'state_dict_' + k in resume_state:
                logger.info('=> resume model state {}'.format(k))
                pretrained_w, _, _ = load_partial_weights(model.module, '', \
                    pretrained_state=resume_state['state_dict_' + k], logger=logger)

        begin_epoch = resume_state['epoch']
        if 'optimizer' in resume_state: optimizer.load_state_dict(resume_state['optimizer'])
        if 'scheduler' in resume_state: scheduler.load_state_dict(resume_state['scheduler'])        
        best_perf = resume_state['perf']
        logger.info(f'=> resume checkpoint from epoch {begin_epoch:d} with perf {best_perf:.3f}')
    else: save_files(args, config, final_output_dir)
    
    enable_pts = False
    for epoch in range(begin_epoch, end_epoch):
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info(f'Epoch {epoch:d} Lr {cur_lr:.6f}')
        writer_dict['writer'].add_scalar('lr', cur_lr, epoch)

        # train for one epoch
        train(config, train_loader, models_dict, loss_dict, optimizer, epoch,
            final_output_dir, tb_log_dir, writer_dict, perf_name, 
            norm=norm, inv_norm=inv_norm, enable_pts=enable_pts)

        # evaluate on validation set 
        perf_indicator = validate(config, val_exemplar_loader, val_loader, val_set, models_dict,
            loss_dict, final_output_dir, tb_log_dir, writer_dict, perf_name, 
            norm=norm, inv_norm=inv_norm)

        if compare_func(perf_indicator, best_perf):
            best_perf = perf_indicator
            best_model = True
        else: best_model = False

        # save checkpoint 
        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        checkpoint_dict = {'epoch': epoch + 1, 'model': get_model_name(config),
            'perf': perf_indicator, 'optimizer': optimizer.state_dict(), 
            'scheduler': scheduler.state_dict()}
        for k, v in models_dict.items():
            state_dict = v.module.state_dict() if config.USE_GPU else v.state_dict()
            checkpoint_dict[f'state_dict_{k}'] = state_dict
        save_checkpoint(checkpoint_dict, best_model, final_output_dir)

        if scheduler_type == 'reduce': scheduler.step(perf_indicator)
        else: scheduler.step()

        # save pseudo label for train set (unlabeled + exemplar)
        if  (epoch + 1) >= config.TRAIN.ENABLE_PTS_EPOCH:
            # inference on train set 
            pesudo_perf = validate(config, val_exemplar_loader, 
                train_loader_for_test, train_set_for_test, models_dict, loss_dict, \
                final_output_dir, tb_log_dir, writer_dict, perf_name, norm=norm, inv_norm=inv_norm)
            # save to certain path 
            if (epoch+1) == config.TRAIN.ENABLE_PTS_EPOCH:
                plabeler = Plabeler(config, final_output_dir, \
                    join(final_output_dir, 'results'), 'loose', logger)
            else: plabeler.update(epoch+1)
            pseudo_path= plabeler.label_path
        
            # update train loader with pesudo ann
            train_set, train_loader = _make_data(config, 'TRAIN', True, transform=transform, \
                ann_file_path=pseudo_path, return_mask=True)
            enable_pts = True

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()


