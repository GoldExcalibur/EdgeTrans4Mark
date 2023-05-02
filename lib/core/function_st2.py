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

from core.config import get_model_name 
from core.evaluate import accuracy, cepha_metric, get_max_preds 
from core.inference import get_final_preds, get_final_preds_from_coords
from utils.transforms import flip_back, fliplr_joints, get_affine_transform 
from utils.vis import save_debug_images, save_batch_image_with_joints, save_batch_image_with_joints_and_gts, save_batch_crops 
from utils.utils import AverageMeter, AverageMeterSet, _print_name_value 
from utils.utils import normalize_coords, ramp, update_teacher_weights

import cv2 
import kornia 
from tqdm import tqdm 

logger = logging.getLogger(__name__)

def train(config, lab_loader, unlab_loader, models_dict, loss_dict, optimizer, epoch, 
        output_dir, tb_log_dir, writer_dict, perf_name):
    batch_time = AverageMeter(); data_time = AverageMeter()
    losses = AverageMeterSet(); metrics = AverageMeterSet()

    display_period = int(np.ceil(len(unlab_loader) / 3))
    for k in models_dict: models_dict[k].train()

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    end = time.time()

    im_w, im_h = config.MODEL.IMAGE_SIZE 
    hm_w, hm_h = config.MODEL.EXTRA.HEATMAP_SIZE 
    stride = im_w // hm_w 
    lab_iter = iter(lab_loader)
    filter_flag = config.LOSS.COTEACH 
    forget_rate = min(0.8, ramp(epoch, 30, 'linear')) if filter_flag else 0.0 
    logger.info('Current forget rate {:.3f}'.format(forget_rate))

    w = ramp(epoch, 30, 'linear') * 0.3 
    w_noise, w_consist = 1.0, config.LOSS.CONSIST_W 
    for i, upack in enumerate(unlab_loader):
        try: lpack = lab_iter.next()
        except:
            lab_iter = iter(lab_loader)
            lpack = lab_iter.next()
        
        # load data on cuda 
        linput, ltgt, ltgt_w, lmeta, linput_aug, ltgt_aug, ltgt_w_aug, lM_hm = lpack 
        uinput, utgt, utgt_w, umeta, uinput_aug, utgt_aug, utgt_w_aug, uM_hm = upack 
        data_time.update(time.time() - end)

        lsize, usize = linput.size(0), uinput.size(0)
        bsize = lsize + usize 
        input = torch.cat([linput, uinput], dim=0).cuda()
        input_aug = torch.cat([linput_aug, uinput_aug], dim=0).cuda()
        ltgt_aug, ltgt_w_aug = ltgt_aug.cuda(), ltgt_w_aug.cuda()
        utgt_aug, utgt_w_aug = utgt_aug.cuda(), utgt_w_aug.cuda()
        ltgt, ltgt_w = ltgt.cuda(), ltgt_w.cuda()
        utgt, utgt_w = utgt.cuda(), utgt_w.cuda() 
        M_hm = torch.cat([lM_hm, uM_hm], dim=0).cuda()

        # backbone output heatmap 
        out1 = models_dict['net1'](input)
        out1_aug = models_dict['net1'](input_aug)
        out2 = models_dict['net2'](input)
        out2_aug = models_dict['net2'](input_aug)
        # clean lab loss 
        clean1 = loss_dict['heat'](out1_aug[:lsize], ltgt_aug, ltgt_w_aug) + \
            loss_dict['heat'](out1[:lsize], ltgt, ltgt_w)
        clean2 = loss_dict['heat'](out2_aug[:lsize], ltgt_aug, ltgt_w_aug) + \
            loss_dict['heat'](out1[:lsize], ltgt, ltgt_w)
        clean_hm_loss = clean1 + clean2
        losses.update('clean', clean_hm_loss.item(), lsize)
        # model 1 loss 
        heat1_aug = ((out1_aug[lsize:] - utgt_aug)**2).mean(dim=[-1, -2]).flatten()
        heat1 = ((out1[lsize:] - utgt)**2).mean(dim=[-1, -2]).flatten()
        noise1 = 0.5 * (heat1 + heat1_aug)
        out1_M = kornia.warp_affine(out1, M_hm, out1.size()[-2:])
        with torch.no_grad():
            cons1_self = loss_dict['consist'](out1_aug, out1_M)
        filter1_loss= noise1 * (1.0 - w) + w * cons1_self
        # model 2 loss 
        heat2_aug = ((out2_aug[lsize:] - utgt_aug)**2).mean(dim=[-1, -2]).flatten()
        heat2 = ((out2[lsize:] - utgt)**2).mean(dim=[-1, -2]).flatten()
        noise2 = 0.5 * (heat2 + heat2_aug)
        out2_M = kornia.warp_affine(out2, M_hm, out2.size()[-2:])
        with torch.no_grad():
            cons2_self = loss_dict['consist'](out2_aug, out2_M)
        filter2_loss = noise2 * (1.0 - w) + w * cons2_self 
        # small-loss select for each other
        num_remember = max(1, int((1 - forget_rate) * heat1.size(0))) # >=1 
        sort_idx1 = filter1_loss.argsort()[:num_remember]
        sort_idx2 = filter2_loss.argsort()[:num_remember]
        noise_hm_loss = (noise1[sort_idx2]).mean() + (noise2[sort_idx1]).mean()
        losses.update('noise', noise_hm_loss.item(), usize)
        # consist loss 
        cons1_cross = loss_dict['consist'](out1_aug, out2_M)
        cons2_cross = loss_dict['consist'](out2_aug, out1_M)
        consist_loss = cons1_cross + cons2_cross 
        losses.update('consist', consist_loss.item(), bsize)
        # total loss 
        loss = clean_hm_loss + noise_hm_loss * w_noise + consist_loss * w_consist 
        losses.update('total', loss.item(), bsize)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_steps += 1

        output_hm = out1_aug.detach().cpu().numpy()
        tgt_aug = torch.cat([ltgt_aug, utgt_aug], dim=0)
        target_hm = tgt_aug.detach().cpu().numpy()
        
        pred, _ = get_max_preds(output_hm)
        target, _ = get_max_preds(target_hm)
        scale = torch.cat([lmeta['scale'], umeta['scale']], dim=0)
        bbox_size = scale.cpu().numpy() * config.DATASET.PIXEL_STD 
        mre, sd, sdr_dict = cepha_metric(pred, target, (hm_w, hm_h), bbox_size)
        metrics.update('mre', np.mean(mre), bsize)
        metrics.update('sd', np.mean(sd), bsize)
        for k, v in sdr_dict.items(): metrics.update(k, v, bsize)

        # measure elapsed time 
        batch_time.update(time.time() - end)
        end = time.time()

        if i % display_period == 0: 
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f}) w_consist {w:.4f}\t' \
                  'clean {cl.val:.4f}({cl.avg:.4f}) noise {no.val:.4f}({no.avg:.4f})\t' \
                  'consist {con.val:.4f}({con.avg:.4f})'.format(
                      epoch, i, len(unlab_loader), batch_time=batch_time,
                      speed=bsize/batch_time.val,
                      data_time=data_time, loss=losses['total'], w=w_consist,
                      cl=losses['clean'], no=losses['noise'], con=losses['consist'])
            if perf_name == 'AP': msg += 'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(acc=metrics['acc'])
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
            writer.add_scalars('train_loss', losses.averages(), global_steps)

            prefix = '{}_{}'.format(join(output_dir, 'train_stu1'), i)
            meta_vis = {'joints': torch.cat([lmeta['joints_aug'], umeta['joints_aug']], dim=0),\
                    'joints_vis': torch.cat([lmeta['joints_vis'], umeta['joints_vis']], dim=0)}
            save_debug_images(config, input_aug, meta_vis, tgt_aug, pred * stride, \
                out1_aug, prefix, line_pairs=config.DATASET.LINE_PAIRS)
            
            prefix = '{}_{}'.format(join(output_dir, 'train_stu2'), i)
            pred2, _ = get_max_preds(out2_aug.detach().cpu().numpy())
            save_debug_images(config, input_aug, meta_vis, tgt_aug, pred2 * stride, \
                out2_aug, prefix, line_pairs=config.DATASET.LINE_PAIRS)
