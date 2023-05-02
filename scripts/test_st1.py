from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, os, sys 
from os.path import join, exists, isdir, isfile, dirname, abspath
this_dir = dirname(abspath(__file__))
base_dir = dirname(this_dir)

import pprint
import shutil
import copy
import logging
from tqdm import tqdm
from glob import glob

import json

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
from core.function_st1 import validate
from utils.utils import get_optimizer, get_scheduler
from utils.utils import save_checkpoint, create_logger
from utils.utils import load_partial_weights, save_infer2coco

import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--cfg', default='', type=str, help='experiment configure file name')
    parser.add_argument('--model', required=True, type=str, help='model state file')
    args, rest = parser.parse_known_args()
    # update config
    if args.cfg == '':
        cfg_path = glob(join(dirname(args.model), '*.yaml'))
        assert len(cfg_path) == 1, 'find more than configuration file !'
        args.cfg = cfg_path[0]
    update_config(args.cfg)

    # testing
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--flip-test', help='use flip test', action='store_true')
    parser.add_argument('--post-process', help='use post process', action='store_true')
    parser.add_argument('--shift-heatmap', help='shift heatmap', action='store_true')
    parser.add_argument('--vis-all', action='store_true', default=False, help='visualize all results')
    parser.add_argument('--local-iter', default=4, type=int, help='iter cnt for local deform stage')
    parser.add_argument('--infer-train', default=False, action='store_true', help='stage I infer pesudo labels for stage II')
    parser.add_argument('--bbox-mode', default='tight', choices=['tight', 'loose'], type=str, help='bbox mode for inference results')
    args = parser.parse_args()
    # args.cfg = cfg_path[0] # reset cfg into args
    return args

def reset_config(config, args):
    if args.gpus: config.GPUS = args.gpus 
    if args.workers: config.WORKERS = args.workers 
    if args.flip_test: config.TEST.FLIP_TEST = args.flip_test 
    if args.post_process: config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap: config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.vis_all: config.DEBUG.VIS_ALL_IMAGES_PRED_GT = args.vis_all
    if args.model: config.TEST.MODEL_FILE = args.model
    if args.local_iter: config.TEST.LOCAL_ITER = args.local_iter
    if args.infer_train: config.TEST.INFER_TRAIN = args.infer_train
    config.DATA_DIR = join(base_dir, config.DATA_DIR)
    config.OUTPUT_DIR = join(base_dir, config.OUTPUT_DIR)
    config.DATASET.ROOT = join(config.DATA_DIR, config.DATASET.ROOT)

def _make_model(cfg, logger):
    pose_name, model_name = cfg.MODEL.NAME
    models_dict = {}
    ### Backbone model ###################
    pose_model = eval(f'models.Backbones.{pose_name}.get_backbone')(cfg, False)
    models_dict['backbone'] = pose_model
    ### Bert model #######################
    bert_model = eval(f'models.{model_name}.get_net')(cfg)
    models_dict['fuse'] = bert_model 
    ### load from checkpoint #################
    checkpoint_path = config.TEST.MODEL_FILE 
    resume_state = torch.load(checkpoint_path)
    meta = {k:resume_state[k] for k in ['epoch', 'model', 'perf']}
    for k, model in models_dict.items():
        load_k = f'state_dict_{k}'
        if load_k in resume_state:
            logger.info(f'=> resume model state {k}')
            resume_w, _, _ = load_partial_weights(model, '', \
                pretrained_state=resume_state[load_k], logger=logger)
        else: raise ValueError(f'Invalid key {load_k} for current checkpoint !')
        if cfg.USE_GPU:
            models_dict[k] = torch.nn.DataParallel(model).cuda()
    return models_dict, meta

def save_files(final_output_dir):
    this_dir = dirname(__file__)
    shutil.copy2(abspath(__file__), final_output_dir)

def _make_loss(cfg):
    loss_dict = {}    
    pose_name, model_name = cfg.MODEL.NAME
    im_channel = 1
    im_w, im_h = cfg.MODEL.IMAGE_SIZE
    # reconstruction loss 
    loss_dict['l1'] = WeightedL1Loss(size_average=True)
    loss_dict['l2'] = WeightedL2Loss(size_average=True)
    loss_dict['sim'] = SSIM(im_channel, mode='ssim', reduction='mean')
    loss_dict['esim'] = EdgeSSIM(im_channel, (im_w, im_h), mode='ssim', reduction='mean')
    # smoothness loss 
    loss_dict['grad'] = FlowGrad(penalty='l2')
    loss_dict['egrad'] = EdgeGrad((im_w, im_h), penalty='l2', corner_flag=True, T=cfg.LOSS.TEMPERATURE)
   
    if cfg.USE_GPU:
        for name, loss in loss_dict.items():
            loss_dict[name] = loss.cuda()
    return loss_dict

def main():
    args = parse_args()
    reset_config(config, args)
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED
    
    # os.environ['CUDA_VISIBLE_DEVICES'] = config.GPUS
    logger.info('=> USING GPUS {}'.format(config.GPUS))

    models_dict, meta = _make_model(config, logger)
    writer_dict = {'writer': SummaryWriter(tb_log_dir), 'valid_global_steps': 0}
    loss_dict = _make_loss(config)

    norm = transforms.Normalize(mean=[0.449], std=[0.226])
    transform = transforms.Compose([transforms.ToTensor(), norm])
    inv_norm = transforms.Compose([
        transforms.Normalize(mean=[0.], std=[1/0.226]),
        transforms.Normalize(mean=[-0.449], std=[1])
    ])

    data_name = config.DATASET.DATASET
    if data_name == 'rsna': perf_name = 'AP'
    elif data_name in ['cepha', 'hand', 'chest']:
        perf_name = 'MRE'
        
    template_dataset = eval('dataset.' + data_name)(
        config, config.DATASET.ROOT, config.DATASET.TRAIN_SET,
        False, transform=transform, subdir=config.DATASET.TRAIN_SUBDIR,
        ann_file_path=config.DATASET.TRAIN_ANN)

    # for inference labels on train set 
    prefix = 'TRAIN' if config.TEST.INFER_TRAIN else 'TEST'

    test_dataset = eval('dataset.'+ data_name)(
        config, config.DATASET.ROOT, getattr(config.DATASET, f'{prefix}_SET'),
        False, transform=transform, subdir=config.DATASET.TEST_SUBDIR,
        ann_file_path=getattr(config.DATASET, f'{prefix}_ANN')
    )

    # reset config attr (dataset)
    config.DATASET.POINT_NAMES = test_dataset.point_names
    config.DATASET.LINE_PAIRS = test_dataset.line_pairs
        
    template_loader = torch.utils.data.DataLoader(
        template_dataset, batch_size=config.TEST.BATCH_SIZE, shuffle=False, 
        num_workers=config.WORKERS, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.TEST.BATCH_SIZE, shuffle=False,
        num_workers=config.WORKERS, pin_memory=True)
  
    perf_indicator = validate(config, template_loader, test_loader, test_dataset, models_dict,
        loss_dict, final_output_dir, tb_log_dir, writer_dict, perf_name,
        norm=norm, inv_norm=inv_norm, save_details=True)

    if config.TEST.INFER_TRAIN: # generate pseudo labels annotation files 
        time_str = dirname(config.TEST.MODEL_FILE).split('_')[-2]
        save_dir = join(config.DATASET.ROOT, 'anno/pseudo')
        save_path = save_infer2coco(config, final_output_dir, save_dir, \
            bbox_mode=args.bbox_mode, logger=logger, time_str=time_str, \
            epoch=meta['epoch'], save_name='pseudo_label_best')
        print('save pseudo label to {}'.format(save_path))
    
if __name__ == '__main__':
    main()

