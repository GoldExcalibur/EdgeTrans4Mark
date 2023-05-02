from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from os.path import join, dirname, exists, isfile
import logging
import time
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2 
import json
import copy

from core.config import get_model_name
from einops import rearrange

def set_logger(log_file_path):
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file_path, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger

def create_logger(cfg, cfg_name, phase='train'):
    if cfg.RESUME != '' and not cfg.FINETUNE:
        final_output_dir = dirname(cfg.RESUME)
        tensorboard_log_dir = join(final_output_dir, 'log')
        log_files = glob(join(final_output_dir, '*.log'))
        assert len(log_files) == 1
        logger = set_logger(log_files[0])
        logger.info('=> resume from {}'.format(cfg.RESUME))
        return logger, final_output_dir, tensorboard_log_dir

    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists() and phase == 'train':
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET.replace(':', '_')
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    experiment_name = join(cfg_name, time_str, phase)
    experiment_name = experiment_name.replace('/', '_')
    if phase == 'train':
        final_output_dir = root_output_dir / dataset / model / Path(experiment_name)
    elif phase in ['valid', 'test']:
        if cfg.TEST.USE_EMA:
            experiment_name += '_ema'
        if cfg.TEST.FLIP_TEST:
            experiment_name += '_flip'
            if cfg.TEST.SHIFT_HEATMAP:
                experiment_name += '_shift'
        if cfg.TEST.POST_PROCESS:
            experiment_name += '_post'
        checkpoint_path = cfg.TEST.MODEL_FILE 
        if isinstance(checkpoint_path, list): checkpoint_path = checkpoint_path[0]
        checkpoint_dir = dirname(checkpoint_path)
        final_output_dir = Path(checkpoint_dir) / experiment_name
    else:
        assert 0, 'unknown phase {}'.format(phase)

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    logger = set_logger(str(final_log_file))

    # tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
    #     (cfg_name + '_' + time_str)
    tensorboard_log_dir = final_output_dir / 'log'
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_optimizer(cfg, model):
    if isinstance(model, dict):
        params = []
        for name, model in model.items():
            params.extend(list(model.parameters()))
    else:
        params = model.parameters()

    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            params,
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            params,
            lr=cfg.TRAIN.LR
        )

    return optimizer

def get_scheduler(cfg, optimizer, params=None):
    lr_decay = params['lr_decay'] if 'lr_decay' in params else None
    scheduler_type = cfg.TRAIN.SCHEDULER

    scheduler = None
    if scheduler_type == 'multistep':
        milestones = params['milestones'] if 'milestones' in params else None
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=lr_decay)
    elif scheduler_type == 'step':
        step_size = params['step_size'] if 'step_size' in params else None
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_decay)
    elif scheduler_type == 'reduce':
        patience = params['patience'] if 'patience' in params else None
        threshold = params['threshold'] if 'threshold' in params else None
        mode = params['mode'] if 'mode' in params else None
        # mode = max or min
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, 
        factor=lr_decay, patience=patience, threshold=threshold, min_lr=1e-8, verbose=True)
    elif scheduler_type == 'cosine':
        max_iter = params['max_iter'] 
        min_lr = params['min_lr'] if 'min_lr' in params else 0.0
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=min_lr)
    else:
        assert 0, 'not implemented lr scheduler {}'.format(cfg.TRAIN.SCHEDULER)
    
    return scheduler

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint', topk=3):
    '''save topk best model'''
    if is_best:
        if 'epoch' in states: filename = '{}epoch_best'.format(states['epoch'])
        else: filename = 'model_best'

    # remove old best states 
    saved_filenames = glob(join(output_dir, '*epoch_best.pth.tar'))
    saved_filenames = [f.split('/')[-1] for f in saved_filenames]
    saved_epochs = [int(n[:n.index('epoch')]) for n in saved_filenames]
    if len(saved_epochs) >= topk:
        saved_epochs.sort()
        oldest_epoch = saved_epochs[0]
        oldest_filename = join(output_dir, f'{oldest_epoch:d}epoch_best.pth.tar')
        os.remove(oldest_filename)

    torch.save(states, join(output_dir, f'{filename}.pth.tar'))
    
def num_trainable_params(model, print_list=False, logger=None):
    """Count number of trainable parameters
    """
    n_trainable = 0
    n_total = 0
    #for child in model.children():
    for param in model.parameters():
        n_total += param.nelement()
        if param.requires_grad == True:
            n_trainable += param.nelement()
    if logger:
        logger.info('=> Trainable {:,} parameters out of {:,}'.format(n_trainable, n_total))
    if print_list:
        print('Trainable parameters:')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print('\t {} \t {} \t {:,}'.format(name, param.size(), param.numel()))
    return n_trainable

def load_partial_weights(model, model_path, pretrained_state=None, cuda_avail=True, logger=None):
    if pretrained_state is None:
        if cuda_avail:
            pretrained_state = torch.load(model_path)
        else:
            pretrained_state = torch.load(model_path, map_location=torch.device('cpu'))
    
    model_state = model.state_dict()
    #print(model_state.keys())
    transfer_state = {k:v for k,v in pretrained_state.items() if k in model_state and v.size() == model_state[k].size()}
    #print('Loading weights for layers:', transfer_state.keys())
    not_in_model_state = [k for k,v in pretrained_state.items() if k not in model_state or v.size() != model_state[k].size()]
    print('Not loaded weights:', not_in_model_state)
    model_state.update(transfer_state)
    print(model.load_state_dict(model_state))
    no_init = [k for k,v in model_state.items() if ('num_batches_tracked' not in k) and (k not in pretrained_state or v.size() != pretrained_state[k].size())]
    print('Randomly initialised weights', no_init)
    n1, n2, n3 = len(transfer_state), len(not_in_model_state), len(no_init)
    n = float(n1 + n2 + n3)
    if logger: 
        logger.info('=> Load Param Ratios: loaded {:.3f} not_loaded {:.3f} random_init {:.3f}'.format(n1/n, n2/n, n3/n))
    return transfer_state.keys(), not_in_model_state, no_init

def read_labeled_split(labels_file):
    """Read text files with labelled and unlabelled indices
    """
    print('=> reading labelled split from {}'.format(labels_file))
    labels_idx = np.genfromtxt(labels_file, dtype='str')
    labelled_idx = np.where(labels_idx[:,1] == '1')[0]
    unlabelled_idx = np.where(labels_idx[:,1] == '-1')[0]
    return labelled_idx, unlabelled_idx

# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)
    
    
class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}

# normalize coords to [-1, 1]
def normalize_coords(coords, w, h, inv=False, **kwargs):
    '''
    coords: (x, y) 0<x<w, 0<y<h
    '''
    norm_type = kwargs['type'] if 'type' in kwargs else '[-1,1]'
    assert norm_type in ['[-1,1]', '[0,1]']
    dev = coords.device
    results = coords.clone()
    if inv:
        if norm_type == '[-1,1]':
            results = (results + 1)/2.0

        results[..., 0] *= (w-1)
        results[..., 1] *= (h-1)
    else:
        results[..., 0] = results[..., 0]/(w-1)
        results[..., 1] = results[..., 1]/(h-1)
        if norm_type == '[-1,1]':
            results = 2 * results - 1
    return results


def ramp(current, length, ramp_type):
    if ramp_type == "sigmoid":
        if length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, length)
            phase = 1.0 - current / length
            return float(np.exp(-5.0 * phase * phase))
    elif ramp_type == "linear":
        # assert current >= 0 and length >= 0
        if length == 0:
            return 1.0
        current = np.clip(current, 0.0, length)
        return current / length
    elif ramp_type == "cosine":
        assert 0 <= current <= length
        return float(.5 * (np.cos(np.pi * current / length) + 1))
    else:
        assert 0, 'unknown ramp type %s' % (ramp_type)


# sample random transform (affine or perspective)
def get_random_trans(bsize, tf=0.1, sf=0.25, rf=0.1, shf=0.01, trans_type='affine'):
    assert trans_type in ['affine', 'perspective'], 'invalid transform type {}'.format(trans_type)

    s = 1.0 + torch.randn(bsize, 2).clamp(-sf, sf)
    r = (rf * np.pi) * torch.randn(bsize).clamp(-2, 2)
    sh = (shf * np.pi) * torch.randn(bsize).clamp(-2, 2)
    t = tf * torch.randn(bsize, 2).clamp(-2, 2)

    rcos, rsin = torch.cos(r), torch.sin(r)
    shtan = torch.tan(sh)
    ndim = 2 if trans_type == 'affine' else 3 
    trans = torch.zeros(bsize, ndim, 3)

    trans[:, 0, 0] = s[:, 0] * rcos
    trans[:, 0, 1] = s[:, 0] * (rcos * shtan + rsin)
    trans[:, 0, 2] = t[:, 0]
    trans[:, 1, 0] = -s[:, 1] * rsin
    trans[:, 1, 1] = s[:, 1] * (-rsin * shtan + rcos)
    trans[:, 1, 2] = t[:, 1]
    
    if ndim == 3: # slight perturb over affine transform 
        trans[:, 2, :2] = (0.15 * torch.randn(bsize, 2)).clamp(-0.5, 0.5)
        trans[:, 2, 2] = 1.0 
    return trans 


def get_grid(theta, out_size, corner_flag):
    '''
    theta: b x ndim x 3, ndim=2/3; return ide_grid if None
    '''
    bsize, c, h, w = out_size 
    ide_m = torch.cat([torch.eye(2), torch.zeros(2, 1)], dim=1)
    ide_grid = F.affine_grid(ide_m[None, ...], (1, c, h, w), align_corners=corner_flag)
    if theta is None: return ide_grid.repeat(bsize, 1, 1, 1)

    dev = theta.device
    ndim = theta.size(1)
    ide_grid = ide_grid.flatten(1, 2).repeat(bsize, 1, 1) # b x (h x w) x 2
    ide_grid = torch.cat([ide_grid, torch.ones(bsize, int(h*w), 1, dtype=ide_grid.dtype)], dim=-1)
    out_grid = torch.bmm(ide_grid.to(dev), theta.transpose(-2, -1)) # b x (h x w) x 2/3
    if ndim == 3:
        out_grid = out_grid[..., :2] / out_grid[..., 2:3]
    out_grid = rearrange(out_grid, 'b (h w) c -> b h w c', h=h)
    return out_grid 


# sample random perspective transform 
# def get_theta_flow(config, input, corner_flag):
#     bsize, c, h, w = input.size()
#     dev = input.device 
#     sf = config.DATASET.SCALE_FACTOR 
#     rf = config.DATASET.ROT_FACTOR / 180.0
#     shf = rf / 10.0 
#     tf = 0.1 
#     batch_trans = get_random_trans(bsize, 
#         tf=tf, sf=sf, rf=rf, shf=shf,
#         trans_type='perspective') # b x 3 x 3
    
#     # gen aff input
#     aff = batch_trans[:, :2].to(dev)
#     aff_grid = F.affine_grid(aff, (bsize, c, h, w), align_corners=corner_flag)
#     out = F.grid_sample(input, aff_grid, mode='bilinear', padding_mode='zeros', align_corners=corner_flag)

#     # flow=perspective - affine, gen flow input
#     per_grid = get_grid(batch_trans, (bsize, c, h, w), corner_flag)  
#     flow = per_grid.to(dev) - aff_grid # b x h x w x 2
#     ide_m = torch.cat([torch.eye(2), torch.zeros(2, 1)], dim=1)
#     ide_grid = F.affine_grid(ide_m[None, ...], (1, c, h, w), align_corners=corner_flag)
#     out = F.grid_sample(out, ide_grid.to(dev) + flow, mode='bilinear', padding_mode='zeros', align_corners=corner_flag)
    
#     return out, aff, flow 

def get_theta_flow(config, input, corner_flag, trans_type='affine'):
    bsize, c, h, w = input.size()
    dev = input.device 
    sf = 0.5 * config.DATASET.SCALE_FACTOR 
    rf = 0.5 * config.DATASET.ROT_FACTOR / 180.0
    shf = rf / 10.0 
    tf = 0.1 
    batch_trans = get_random_trans(bsize, 
        tf=tf, sf=sf, rf=rf, shf=shf,
        trans_type=trans_type) # b x 2/3 x 3
    batch_trans = batch_trans.to(dev)
    
    trans_grid = get_grid(batch_trans, (bsize, c, h, w), corner_flag)  
    trans_grid = trans_grid.to(dev)
    out = F.grid_sample(input, trans_grid, mode='bilinear', padding_mode='zeros', align_corners=corner_flag)
    
    if trans_type == 'affine':
        return out, batch_trans 
    elif trans_type == 'perspective':
        return out, trans_grid 
    else:
        raise ValueError('Invalid trans type {} !'.format(trans_type))

def get_background_mask(trans, src_size, dst_size):
    '''
    src_size / dst_size: (w, h) 
    trans (src->dst): 2 x 3 or 3 x 3
    output: h x w, binary mask
    '''
    sw, sh = src_size 
    dw, dh = dst_size
    xx, yy = np.meshgrid(np.arange(dw), np.arange(dh))
    trans_inv = cv2.invertAffineTransform(trans)
    grid = np.stack([xx, yy], axis=-1)
    grid = np.concatenate([grid, np.ones((dh, dw, 1))], axis=-1)
    grid = grid @ trans_inv.T 
    xmask = (0 < grid[..., 0]) & (grid[..., 0] < sw)
    ymask = (0 < grid[..., 1]) & (grid[..., 1] < sh)
    mask = np.where(xmask & ymask, 1, 0).astype(np.float32)
    return mask

def update_teacher_weights(student_model, teacher_model, alpha, global_steps, param_type='p'):
    alpha = min(1.0 - 1.0/(global_steps + 1), alpha)
    if param_type == 'p':
        t_params = teacher_model.parameters()
        s_params = student_model.parameters()
    elif param_type == 's':
        t_params = list(teacher_model.state_dict().values())
        s_params = list(student_model.state_dict().values())
    else:
        raise ValueError('invalid param type {}'.format(param_type))

    for t_param, s_param in zip(t_params, s_params):
        t_param.data.mul_(alpha).add_(1.0-alpha, s_param.data)

def init_weights(m, fc_std=0.01):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, fc_std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def get_gaussian_maps(mu, shape_wh, inv_std, mode='rot'):
    """
    Generates [B,NMAPS,SHAPE_H,SHAPE_W] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    dev = mu.device
    w, h = int(shape_wh[0]), int(shape_wh[1])
    mu_x, mu_y = mu.chunk(2, dim=2) # b x n_joints x 1
    x = torch.linspace(-1.0, 1.0, w).to(dev)
    y = torch.linspace(-1.0, 1.0, h).to(dev)

    if mode in ['rot', 'flat']:
        mu_y, mu_x = torch.unsqueeze(mu_y, dim=-1), torch.unsqueeze(mu_x, dim=-1)

        y = y.view(1, 1, h, 1)
        x = x.view(1, 1, 1, w)

        g_y = (y - mu_y)**2
        g_x = (x - mu_x)**2
        dist = (g_y + g_x) * inv_std**2

        if mode == 'rot':
            g_yx = torch.exp(-dist)
        else:
            g_yx = torch.exp(-torch.pow(dist + 1e-5, 0.25))

    elif mode == 'ankush':
        y = y.view(1, 1, h)
        x = x.view(1, 1, w)

        g_y = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_y - y) * inv_std)))
        g_x = torch.exp(-torch.sqrt(1e-4 + torch.abs((mu_x - x) * inv_std)))

        g_y = torch.unsqueeze(g_y, dim=3)
        g_x = torch.unsqueeze(g_x, dim=2)
        g_yx = torch.matmul(g_y, g_x)  # [B, NMAPS, H, W]

    else:
        raise ValueError('Unknown mode: ' + str(mode))

    return g_yx

def resize_feats(feats, scale, in_type, out_type, **kwargs):
    '''
    To resize feats according to scale 
    feats: b x c x h x w (img) or b x h x w x c (seq)
    in_type, out_type: [img, seq]
    img: b x c x h x w 
    seq: b x h x w x c
    '''
    corner_flag = kwargs['corner_flag'] if 'corner_flag' in kwargs else True
    mode = kwargs['mode'] if 'mode' in kwargs else 'bilinear'
    if feats is None: return None
    feats_img = feats if in_type ==  'img' else feats.permute(0, 3, 1, 2) 
    feats_img = F.interpolate(feats_img, align_corners=corner_flag, scale_factor=scale, mode=mode)
    feats_img = feats_img if out_type == 'img' else feats_img.permute(0, 2, 3, 1)
    return feats_img


def transform_feats(feats, grid, **kwargs):
    '''
    To warp feats according the warping grid 
    feats: b x c x h x w
    grid: b x h1 x w1 x 2 
    '''
    corner_flag = kwargs['corner_flag'] if 'corner_flag' in kwargs else True 
    pad_mode = kwargs['pad_mode'] if 'pad_mode' in kwargs else 'zeros'
    mode = kwargs['mode'] if 'mode' in kwargs else 'bilinear'
    h, w = feats.size(2), feats.size(3)
    if grid.size(1) != h or grid.size(2) != w:
        grid = F.interpolate(grid.permute(0, 3, 1, 2), size=(h, w), mode=mode, align_corners=corner_flag)
        grid = grid.permute(0, 2, 3, 1) # b x h x w x 2
    return F.grid_sample(feats, grid, mode=mode, padding_mode=pad_mode, align_corners=corner_flag)


def split_seq(seq, stride, original_h):
    # split seq b (h w) c -> (b p1 p2) (h/p1 * w/p2) c
    out = rearrange(seq, 'b (h w) c -> b h w c', h=original_h)
    out = rearrange(out, 'b (p1 h) (p2 w) c -> (b p1 p2) (h w) c', p1=stride, p2=stride)
    # bsize, l, _ = seq.size()
    # original_w = l // original_h
    # out = seq.view(bsize, original_h, original_w, -1)
    # out = out.view(bsize, stride, original_h//stride, stride, original_w//stride, -1)
    # out = out.transpose(2,3).flatten(0,2).flatten(1,2)
    return out

def merge_seq(seq, stride, original_h):
    # merge seq: (b p1 p2) (h*w) c -> b (p1*h * p2*w) c
    out = rearrange(seq, 'b (h w) c -> b h w c', h=original_h)
    out = rearrange(out, '(b p1 p2) h w c -> b (p1 h p2 w) c', p1=stride, p2=stride)
    # bsize, l, _ = seq.size()
    # original_w = l // original_h
    # out = seq.view(bsize, original_h, original_w, -1)
    # out = out.view(bsize//(stride*stride), stride, stride, original_h, original_w, -1)
    # out = out.transpose(2,3).flatten(1, 4)
    return out

def get_stats(input, name, logger=None):
    info = '{} {:.3f}~{:.3f} ({:.3f}/{:.3f})'.format( \
        name, input.min(), input.max(), input.mean(), input.std())
    if logger: logger.info(info)
    else: print(info)
            
def save_infer2coco(cfg, final_output_dir, save_dir, bbox_mode='loose', epoch=None, \
    perf=None, time_str=None, logger=None, save_name='', **kwargs):
    # get path of inference result
    dset_key = cfg.DATASET.TRAIN_SET.replace('/', '_')
    infer_paths = glob('{}/results/*_{}_results.json'.format(final_output_dir, dset_key))
    assert len(infer_paths) == 1, f'find {len(infer_path):d}>1 inferences for {dset_key}!'
    infer_path = infer_paths[0]
    # get save path
    if save_name == '':
        model_name, _ = get_model_name(cfg)
        save_name = [model_name, f'bbox_{bbox_mode}']
        if epoch is not None: save_name.append(f'ep{epoch:d}_best')
        if perf is not None: save_name.append(f'perf{perf:.3f}')
        if time_str is not None: save_name.append(time_str)
        save_name = '_'.join(save_name)
        
    save_path = join(save_dir, '{}.json'.format(save_name))
    if logger:
        logger.info(f'=> save pseudo labels {infer_path} >>>>> {save_path}')
    
    # write pesudo ann 
    gt_ann = json.load(open(join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_ANN), 'r'))
    pseudo_ann = copy.deepcopy(gt_ann)

    infer_res = json.load(open(infer_path, 'r'))
    id2pts = dict(zip(
        [res['image_id'] for res in infer_res],
        [res['keypoints'] for res in infer_res]
    ))
    if bbox_mode == 'loose':
        id2wh = dict(zip(
            [im['id'] for im in pseudo_ann['images']],
            [(im['width'], im['height']) for im in pseudo_ann['images']]
        ))

    fpth = join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SUBDIR)
    exemplar_ids = np.loadtxt(fpth).astype(np.uint16)
    for ann in pseudo_ann['annotations']:
        cur_imgid = ann['image_id']
        if cur_imgid in id2pts:
            pred_pts = np.array(id2pts[cur_imgid]).reshape(-1, 3)
            # replace keypoints & bbox in gt to predictions 
            if cur_imgid not in exemplar_ids:
                ann['keypoints'] = []
                for (x,y,vis) in pred_pts:
                    ann['keypoints'].extend([x, y, 2])
            if bbox_mode == 'tight':
                pesudo_pts = np.array(ann['keypoints']).reshape(-1, 3)
                tlx, tly = pesudo_pts[:, :2].min(axis=0)
                brx, bry = pesudo_pts[:, :2].max(axis=0)
                ann['bbox'] = [float(tlx), float(tly), float(brx - tlx), float(bry - tly)]
            elif bbox_mode == 'loose':
                ann['bbox'] = (0, 0) + id2wh[cur_imgid]
            else: raise Exception(f'Invalid bbox mode {bbox_mode} !')
        else: 
            info = f'Cannot find imgid {cur_imgid} in {infer_path} !'
            # raise Exception(info)
            print(info)
            continue 

    try: json.dump(pseudo_ann, open(save_path, 'w'))
    except: logger.info(f'fail to save pesudo label to {save_path} !')
    return save_path


### class for pseudo labeler 
class Plabeler(object):
    def __init__(self, cfg, final_output_dir, save_dir, bbox_mode, logger, **kwargs):
        self.final_output_dir = final_output_dir 
        self.save_dir = save_dir 
        self.enable_pts_epoch = cfg.TRAIN.ENABLE_PTS_EPOCH 
        
        dset_key = cfg.DATASET.TRAIN_SET.replace('/', '_')
        infer_paths = glob('{}/*_{}_results.json'.format(\
            self.save_dir, dset_key))
        assert len(infer_paths) == 1, f'find {len(infer_paths):d} inference results for {dset_key}!'
        self.infer_path = infer_paths[0]
        
        fpth = join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SUBDIR)
        self.exemplar_ids = np.loadtxt(fpth).astype(np.uint16)
        self.label_path = save_infer2coco(cfg, final_output_dir, save_dir, \
            bbox_mode=bbox_mode, logger=logger)
        self.label_dict = json.load(open(self.label_path, 'r'))
        self.alpha = cfg.TRAIN.EMA_ALPHA #0.9 # alpha for ema 
        
        self.logger = logger

    def update(self, epoch): 
        infer_res = json.load(open(self.infer_path, 'r'))
        id2pred = dict(zip(
            [res['image_id'] for res in infer_res],
            [res['keypoints'] for res in infer_res]
        ))
        for ann in self.label_dict['annotations']:
            cur_imgid = ann['image_id']
            if cur_imgid in self.exemplar_ids: continue 
            if cur_imgid not in id2pred:
                self.logger.info(f'Cannot find imgid {cur_imgid} in {infer_path} !')
            pred = np.array(id2pred[cur_imgid]).reshape(-1, 3)
            for idx, (px, py, pvis) in enumerate(pred):
                ann['keypoints'][3*idx] = self.alpha * \
                    ann['keypoints'][3*idx] + (1.0 -  self.alpha) * px 
                ann['keypoints'][3*idx+1] = self.alpha * \
                    ann['keypoints'][3*idx+1] + (1.0 - self.alpha) * py 

        try: json.dump(self.label_dict, open(self.label_path, 'w'))
        except: self.logger.info(f'fail to save pseudo label to {save_path}')