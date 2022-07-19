from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from os.path import dirname, abspath, join, isdir, exists
import pprint
import shutil

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
from core.loss import JointsMSELoss
from core.function import validate
from core.function_st2 import train_c2teach
from utils.utils import get_optimizer, get_scheduler
from utils.utils import save_checkpoint
from utils.utils import create_logger, load_partial_weights

import dataset
import models
import copy
import math
from glob import glob 

def parse_args():
    parser = argparse.ArgumentParser(description='landmark ssl or lnl')
    # general
    parser.add_argument('--cfg', required=True, type=str, help='experiment configure file name')
    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent', type=int, default=config.PRINT_FREQ,
                        help='frequency of logging')
    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')
    parser.add_argument('--resume', help='resume path', type=str, default='')
    parser.add_argument('--method', required=True, type=str, help='ssl or lnl method')
    parser.add_argument('--perf_name', type=str, default='MRE', choices=['MRE', 'AP'], help='perf name for evaluate')
    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus: config.GPUS = args.gpus
    if args.workers: config.WORKERS = args.workers
    if args.resume: config.RESUME = args.resume
    if args.method: config.METHOD = args.method

def _make_model(cfg, final_output_dir, is_train):
    name = cfg.MODEL.NAME 
    models_dict = {}
    models_dict['spen'] = eval(f'models.{name}.get_pose_net')(cfg, is_train=True)
    models_dict['spen2'] = eval(f'models.{name}.get_pose_net')(cfg, is_train=True)    
    for k,v in models_dict.items(): models_dict[k] = torch.nn.DataParallel(v).cuda()
    return models_dict

def save_files(args, cfg, final_output_dir):
    save_file_names = [cfg.MODEL.NAME]
    this_dir = dirname(__file__)
    model_dir = join(this_dir, '../lib/models')
    core_dir = join(this_dir, '../lib/core')
    # copy files under model dir 
    for n in save_file_names:
        shutil.copy2(join(model_dir, '{}.py'.format(n)), final_output_dir)
    # copy other files 
    for fpath in [args.cfg, abspath(__file__), join(core_dir, 'function.py'),
        join(core_dir, 'function_st2.py')]:
        shutil.copy2(fpath, final_output_dir)

def _make_loss(cfg):
    loss_dict = {}
    names  = cfg.MODEL.NAME 
    loss_dict['heat'] = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT)
    loss_dict['consist'] = nn.MSELoss(reduction='mean')
    if cfg.USE_GPU:
        for name, loss in loss_dict.items(): loss_dict[name] = loss.cuda()
    return loss_dict

def epoch2lrdecay(epoch, config):
    '''
    epoch -> lr decay (lrbase * lr_decay)
    '''
    base_lr = config.TRAIN.LR 
    stage1 = 60
    stage2 = config.TRAIN.END_EPOCH
    if epoch < stage1: lr = 1e-3
    else: 
        duration = epoch - stage1
        total = stage2 - stage1
        max_lr = 1e-3
        min_lr = 1e-4
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(duration / total * math.pi))
    return lr / base_lr

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

    models_dict = _make_model(config, final_output_dir, True)
    
    writer_dict = {
        'writer': SummaryWriter(logdir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # define loss function (criterion) and optimizer
    loss_dict = _make_loss(config)
    optimizer = get_optimizer(config, models_dict)
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])

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
        
    lab_train_set = eval('dataset.'+data_name)(
        config, config.DATASET.ROOT, config.DATASET.TRAIN_SET,
        True, transform, ann_file_path=config.DATASET.TRAIN_ANN,
        subdir=config.DATASET.TRAIN_SUBDIR, item_type='semi')

    unlab_train_set = eval('dataset.'+data_name)(
        config, config.DATASET.ROOT, config.DATASET.TRAIN_SET,
        True, transform, ann_file_path=config.DATASET.TRAIN_ANN,
        item_type='semi') #,subdir=unlab_subdir
    
    valid_set = eval('dataset.'+data_name)(
        config, config.DATASET.ROOT, config.DATASET.VAL_SET,
        False, transform, ann_file_path=config.DATASET.VAL_ANN, item_type='sup')
 
    # reset config attr (dataset)
    config.DATASET.POINT_NAMES = valid_set.point_names
    config.DATASET.LINE_PAIRS = valid_set.line_pairs
    
    lab_train_loader = torch.utils.data.DataLoader(
        lab_train_set, batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE, num_workers=config.WORKERS, pin_memory=True
    )

    unlab_train_loader = torch.utils.data.DataLoader(
        unlab_train_set, batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=config.TRAIN.SHUFFLE, num_workers=config.WORKERS, pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=config.TEST.BATCH_SIZE, 
        shuffle=False, num_workers=config.WORKERS, pin_memory=True
    )

    begin_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH
    milestones = [int(r * end_epoch) for r in config.TRAIN.LR_STEP]
    lr_decay = config.TRAIN.LR_FACTOR
    
    logger.info('=> Epoch begin {:d} end {:d} with lr decay {} @ milestones {}'.format(
        begin_epoch, end_epoch, lr_decay, milestones))
    
    scheduler_type = config.TRAIN.SCHEDULER
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e:epoch2lrdecay(e, config))
    
    best_model = False
    if config.RESUME:
        resume_state = torch.load(config.RESUME)
        begin_epoch = resume_state['epoch']
        for k, model in models_dict.items():
            resume_k = 'state_dict_{}'.format(k)
            if resume_k in resume_state:
                logger.info('=> resume model state {}'.format(k))
                # model.module.load_state_dict(resume_state[resume_k])
                pretrained_w, _, _ = load_partial_weights(model.module, '', \
                    pretrained_state=resume_state[resume_k], logger=logger)
        if 'optimizer' in resume_state:
            optimizer.load_state_dict(resume_state['optimizer'])
        if 'scheduler' in resume_state:
            lr_scheduler.load_state_dict(resume_state['scheduler'])        
        if 'perf' in resume_state:
            best_perf = resume_state['perf']
    else: save_files(args, config, final_output_dir)
    
    for epoch in range(begin_epoch, end_epoch):
        cur_lr = optimizer.param_groups[0]['lr']
        logger.info('Epoch {:d} Lr {:.6f}'.format(epoch, cur_lr))
        writer_dict['writer'].add_scalar('lr', cur_lr, epoch)
        
        # train for one epoch
        eval(f'train_{config.METHOD}')(config, lab_train_loader, unlab_train_loader, models_dict, 
            loss_dict, optimizer, epoch, final_output_dir, tb_log_dir, writer_dict, perf_name)

        # evaluate on validation set
        perf_indicator = validate(config, valid_loader, valid_set, models_dict,
            loss_dict, final_output_dir, tb_log_dir, writer_dict, perf_name)
    
        if compare_func(perf_indicator, best_perf):
            best_perf = perf_indicator
            best_model = True
        else: best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        checkpoint_dict = {'epoch': epoch + 1, 'model': get_model_name(config),
            'perf': perf_indicator, 'optimizer': optimizer.state_dict(), 
            'scheduler': lr_scheduler.state_dict()}
        for k, v in models_dict.items():
            state_dict = v.module.state_dict() if config.USE_GPU else v.state_dict()
            checkpoint_dict[f'state_dict_{k}'] = state_dict

        save_checkpoint(checkpoint_dict, best_model, final_output_dir)
        if scheduler_type == 'reduce': lr_scheduler.step(perf_indicator)
        else: lr_scheduler.step()

    writer_dict['writer'].close()

    # final test best model file 
    test_set = eval('dataset.'+data_name)(
        config, config.DATASET.ROOT, config.DATASET.TEST_SET,
        False, transform, ann_file_path=config.DATASET.TEST_ANN, item_type='sup')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.TEST.BATCH_SIZE,
        shuffle=False, num_workers=config.WORKERS, pin_memory=True)

    checkpoint_paths = glob(join(final_output_dir, '*epoch_best.pth.tar'))
    checkpoint_paths.sort()
    best_path = checkpoint_paths[-1]
    best_checkpoint = torch.load(best_path)
    logger.info(f'Train Ends. Test Starts with checkpoint {best_path}')
    for k, model in models_dict.items():
        load_k = f'state_dict_{k}'
        if load_k in best_checkpoint:
            model.module.load_state_dict(best_checkpoint[load_k])
        else: raise ValueError(f'Cannot find key {k} in checkpoint {best_path}!')
    final_perf_indicator = validate(config, test_loader, test_set, models_dict, 
        loss_dict, final_output_dir, tb_log_dir, writer_dict, perf_name)


if __name__ == '__main__':
    main()