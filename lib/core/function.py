from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia

from core.config import get_model_name
from core.evaluate import accuracy, cepha_metric, get_max_preds
from core.inference import get_final_preds, get_final_preds_from_coords
from utils.transforms import flip_back, fliplr_joints, get_affine_transform, affine_transform
from utils.vis import save_debug_images, save_batch_image_with_joints, save_batch_image_with_joints_and_gts
from utils.vis import save_batch_crops, save_batch_flows
from utils.utils import AverageMeter, AverageMeterSet, _print_name_value
from utils.utils import normalize_coords, ramp, get_theta_flow, get_gaussian_maps, get_grid
from utils.utils import transform_feats, resize_feats

import cv2
from tqdm import tqdm
import pickle
from einops import rearrange 

logger = logging.getLogger(__name__)

############## PERF_INDICATOR: AP ###########################################
def train(config, train_loader, models_dict, loss_dict, optimizer, epoch, \
    output_dir, tb_log_dir, writer_dict, perf_name):
    batch_time = AverageMeter(); data_time = AverageMeter()
    losses = AverageMeter(); acc = AverageMeter()
    metrics = AverageMeterSet()
    
    display_period = int(np.ceil(len(train_loader) / 3))
    for k in models_dict: models_dict[k].train()

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    end = time.time()

    im_w, im_h = config.MODEL.IMAGE_SIZE 
    hm_w, hm_h = config.MODEL.EXTRA.HEATMAP_SIZE 
    stride = im_w // hm_w 
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        # compute output
        bsize = input.size(0)
        input = input.cuda()
        verbose_flag = True if i == 0 else False
        if isinstance(config.MODEL.NAME, list):
            if 'fuse' in config.MODEL.NAME[-1]:
                feats = models_dict['spen'](input)
                output = models_dict['fuse'](feats, verbose_flag)
        else:
            output = models_dict['spen'](input)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        loss = loss_dict['heat'](output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_steps += 1

        # measure accuracy and record loss
        losses.update(loss.item(), bsize)

        output_hm = output.detach().cpu().numpy()
        target_hm = target.detach().cpu().numpy()
        if perf_name == 'AP':
            _, avg_acc, cnt, pred = accuracy(output_hm, target_hm)
            metrics.update('acc', avg_acc, cnt)
        elif perf_name == 'MRE':
            pred, _ = get_max_preds(output_hm)
            tgt, _ = get_max_preds(target_hm)
            bbox_size = meta['scale'].cpu().numpy() * config.DATASET.PIXEL_STD # bsize * 2
            mre, sd, sdr_dict = cepha_metric(pred, tgt, (hm_w, hm_h), bbox_size)
            metrics.update('mre', np.mean(mre), bsize)
            metrics.update('sd', np.mean(sd), bsize)
            for k, v in sdr_dict.items():
                metrics.update(k, v, bsize) 

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % display_period == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=input.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            if perf_name == 'AP':
                msg += 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(acc=metrics['acc'])
                writer.add_scalar('train_acc', metrics['acc'].val, global_steps)
            elif perf_name == 'MRE':
                msg +=  'MRE {mre.val:.3f} ({mre.avg:.3f})\t' \
                        'SD {sd.val:.3f} ({sd.avg:.3f})\t' \
                        'SDR 2.0mm {sdr1.val:.3f} ({sdr1.avg:.3f})\t' \
                        'SDR 2.5mm {sdr2.val:.3f} ({sdr2.avg:.3f})\t' \
                        'SDR 3.0mm {sdr3.val:.3f} ({sdr3.avg:.3f})\t' \
                        'SDR 4.0mm {sdr4.val:.3f} ({sdr4.avg:.3f})'.format(
                        mre=metrics['mre'], sd=metrics['sd'],
                        sdr1=metrics['sdr_2.0'], sdr2=metrics['sdr_2.5'], 
                        sdr3=metrics['sdr_3.0'], sdr4=metrics['sdr_4.0'])
                writer.add_scalars('train_sdr', sdr_dict, global_steps)

            logger.info(msg)
            writer.add_scalar('train_loss', losses.val, global_steps)

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, input, meta, target, pred * stride, output,
                              prefix, line_pairs=config.DATASET.LINE_PAIRS)

    writer_dict['train_global_steps'] = global_steps

def validate(config, val_loader, val_dataset, models_dict, loss_dict, output_dir, \
    tb_log_dir, writer_dict, perf_name):
    batch_time = AverageMeter()
    losses = AverageMeter()
    metrics = AverageMeterSet()
    display_period = 1 if config.DEBUG.VIS_ALL_IMAGES_PRED_GT \
        else int(np.ceil(len(val_loader) / 3))

    # switch to evaluate mode
    for k in models_dict: models_dict[k].eval()

    num_samples = len(val_dataset)
    if isinstance(val_dataset, torch.utils.data.dataset.ConcatDataset):
        flip_pairs = val_dataset.datasets[0].flip_pairs
        line_pairs = val_dataset.datasets[0].line_pairs
    else:
        flip_pairs = val_dataset.flip_pairs
        line_pairs = val_dataset.line_pairs

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']    
    im_w, im_h = config.MODEL.IMAGE_SIZE 
    hm_w, hm_h = config.MODEL.EXTRA.HEATMAP_SIZE 
    stride = im_w // hm_w 

    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3),
                         dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    # image_path = []
    image_ids = []
    filenames = []
    imgnums = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            # input = input.cuda()
            verbose_flag = True if i == 0 else False
            if isinstance(config.MODEL.NAME, list):
                if 'fuse' in config.MODEL.NAME[-1]:
                    feats = models_dict['spen'](input)
                    output = models_dict['fuse'](feats, verbose_flag)
            else:
                output = models_dict['spen'](input)

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                if isinstance(config.MODEL.NAME, list):
                    if 'fuse' in config.MODEL.NAME[-1]:
                        feats_flipped = models_dict['spen'](input_flipped)
                        output_flipped = models_dict['fuse'](feats_flipped, verbose_flag)
                else:
                    output_flipped = models_dict['spen'](input_flipped)
                output_flipped = flip_back(output_flipped.cpu().numpy(), flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if config.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = \
                        output_flipped.clone()[:, :, :, 0:-1]
                    # output_flipped[:, :, :, 0] = 0

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = loss_dict['heat'](output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            output_hm = output.cpu().numpy()
            target_hm = target.cpu().numpy()
            if perf_name == 'AP':
                _, avg_acc, cnt, pred = accuracy(output_hm, target_hm)
                metrics.update('acc', avg_acc, num_images)
            elif perf_name == 'MRE':
                bbox_size = meta['scale'].cpu().numpy() * config.DATASET.PIXEL_STD
                pred, _ = get_max_preds(output_hm)
                tgt, _ = get_max_preds(target_hm)
                mre, sd, sdr_dict = cepha_metric(pred, tgt, (hm_w, hm_h), bbox_size)
                metrics.update('mre', np.mean(mre), num_images)
                metrics.update('sd', np.mean(sd), num_images)
                for k, v in sdr_dict.items():
                    metrics.update(k, v, num_images) 
           
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()

            final_preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = final_preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * config.DATASET.PIXEL_STD, 1)
            all_boxes[idx:idx + num_images, 5] = score
            # image_path.extend(meta['image'])
            image_ids.extend(meta['image_id'].numpy())
            idx += num_images

            if i % display_period == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses)
                if perf_name == 'AP':
                    msg += 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(acc=metrics['acc'])
                elif perf_name == 'MRE':
                    msg += 'MRE {mre.val:.3f} ({mre.avg:.3f})\t' \
                        'SD {sd.val:.3f} ({sd.avg:.3f})\t' \
                        'SDR 2.0mm {sdr1.val:.3f} ({sdr1.avg:.3f})\t' \
                        'SDR 2.5mm {sdr2.val:.3f} ({sdr2.avg:.3f})\t' \
                        'SDR 3.0mm {sdr3.val:.3f} ({sdr3.avg:.3f})\t' \
                        'SDR 4.0mm {sdr4.val:.3f} ({sdr4.avg:.3f})'.format(
                        mre=metrics['mre'], sd=metrics['sd'],
                        sdr1=metrics['sdr_2.0'], sdr2=metrics['sdr_2.5'], 
                        sdr3=metrics['sdr_3.0'], sdr4=metrics['sdr_4.0'])
                logger.info(msg)
                
                prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
                save_debug_images(config, input, meta, target, pred * stride, output, prefix, line_pairs=config.DATASET.LINE_PAIRS)
                
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, image_ids,
            filenames, imgnums, perf_name=perf_name)
        
        _, full_arch_name = get_model_name(config)
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(logger, name_value, full_arch_name)
        else:
            _print_name_value(logger, name_values, full_arch_name)

        writer.add_scalar('valid_loss', losses.avg, global_steps)
        if perf_name == 'AP':
            writer.add_scalar('valid_acc', metrics['acc'].avg, global_steps)
        elif perf_name == 'MRE':
            writer.add_scalars('valid_sdr', sdr_dict, global_steps)
        if isinstance(name_values, list):
            for name_value in name_values:
                writer.add_scalars('valid', dict(name_value), global_steps)
        else:
            writer.add_scalars('valid', dict(name_values), global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator
