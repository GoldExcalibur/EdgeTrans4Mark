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
from utils.vis import save_batch_heatmaps, save_debug_images, save_batch_image_with_joints, save_batch_image_with_joints_and_gts
from utils.vis import save_batch_crops, save_batch_flows
from utils.utils import AverageMeter, AverageMeterSet, _print_name_value
from utils.utils import normalize_coords, ramp, get_theta_flow, get_gaussian_maps, get_grid
from utils.utils import transform_feats, resize_feats

import cv2
from tqdm import tqdm
import pickle
from einops import rearrange 
from collections import defaultdict

logger = logging.getLogger(__name__)

def theta2pts(pts, theta):
    '''
    pts: bsize x n_pts x 2
    theta: bsize x 2 x 3
    '''
    bsize, n_pts, _ = pts.size()
    out_pts = torch.bmm(
        torch.cat([pts, torch.ones(bsize, n_pts, 1, device=pts.device)], dim=-1),
        theta.transpose(-2, -1)
    ) # b x n_pts x 2
    if out_pts.size(-1) == 3: # persepctive transform
        return out_pts[..., :2] / out_pts[..., 2:]
    else: # affine transform
        return out_pts 

def flow2pts(pts, flow, corner_flag=False, inv=True):
    '''
    pts: bsize x n_pts x 2 
    flow: bsize x h x w x 2
    '''
    pts_flow = F.grid_sample(flow.permute(0, 3, 1, 2), pts.unsqueeze(2), 
        mode='bilinear', padding_mode='zeros', align_corners=corner_flag)
    pts_flow = pts_flow.squeeze(-1).transpose(-2, -1)
    if inv: return pts - pts_flow
    else: return pts + pts_flow

def im2edge(img):
    '''
    img: b x c x h x w -> edge: b x c x h x w
    '''
    return kornia.filters.sobel(img, normalized=True, eps=1e-6)

def train(config, train_loader, models_dict, loss_dict, optimizer, epoch,
        output_dir, tb_log_dir, writer_dict, perf_name, **kwargs):
    batch_time = AverageMeter(); data_time = AverageMeter()
    losses = AverageMeterSet(); metrics = AverageMeterSet()
    
    display_period = int(np.ceil(len(train_loader) / 3))
    # switch to train mode
    for k in models_dict: models_dict[k].train()

    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']
    end = time.time()

    # compute stride: ratio of image_size / heatmap_size
    im_w, im_h = config.MODEL.IMAGE_SIZE
    hm_w, hm_h = config.MODEL.EXTRA.HEATMAP_SIZE
    stride = im_w // hm_w; n_pts = config.MODEL.NUM_JOINTS 
    # compute identity grid
    corner_flag = True # align_corner: True or False, used in (F.affine_grid, F.interpolate, F.grid_sample)
    ide_m = torch.cat([torch.eye(2), torch.zeros(2, 1)], dim=1)
    ide_grid = F.affine_grid(ide_m[None, ...], (1, 3, im_h, im_w), align_corners=corner_flag)
    # function: project coordinates to [-1, 1] or inverse
    norm_func = lambda c: normalize_coords(c, hm_w, hm_h, inv=False, type='[-1,1]')
    inv_norm_func = lambda c: normalize_coords(c, hm_w, hm_h, inv=True, type='[-1,1]')
    img_pad_mode = 'zeros' #if config.DATASET.DATASET != 'chest' else 'border'
    trans_img_func = lambda f,g: transform_feats(f, g, mode='bilinear',
        corner_flag=corner_flag, pad_mode=img_pad_mode)
    trans_mask_func = lambda f,g: transform_feats(f, g, mode='nearest',
        corner_flag=corner_flag, pad_mode='zeros')
    resize_feats_func = lambda f,sc,in_type,out_type: resize_feats(f, sc, \
        in_type, out_type, corner_flag=corner_flag, mode='bilinear')
    
    # local hyper params 
    do_global = config.TRAIN.DO_GLOBAL; local_iter_cnt = config.TRAIN.LOCAL_ITER 
    local_loss_tags = ['local', 'flow', 'struct', 'smooth']
    w_struct = 0.25; w_smooth = 1.0
    w_lab = (1.0 - ramp(epoch, config.LOSS.RAMP_LEN, 'sigmoid')) * 5.0 
    w_local = ramp(epoch, 250, 'sigmoid') * 1.0
    train_flow_inv = True #False 
    enable_pts = kwargs['enable_pts'] if 'enable_pts' in kwargs else False

    pix2ten = lambda batch_t: torch.stack([kwargs['norm'](t) for t in batch_t], dim=0)
    ten2pix = lambda batch_t: torch.stack([kwargs['inv_norm'](t) for t in batch_t], dim=0)
    for i, (input, _, _, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        bsize = input.size(0)
        # load exemplar 
        input = input[:, 0:1].cuda()
        ######### input_pts ############################################
        ### 1) enable_pts = False + input_pts: gt pts for visualization 
        ### 2) enable_pts = True + input_pts: pseudo pts for esim loss
        ################################################################
        input_pts = meta['joints_norm'].cuda()
        input_mask = meta['mask'].cuda()
        # synthetic samples: gen gt matrix (affine) & flow (perspective)
        input_trans, tgt_grid = get_theta_flow(config, ten2pix(input), corner_flag, trans_type='perspective')
        input_trans = pix2ten(input_trans)  
        # unlab: random perturb, different instance
        rand_idx = torch.randperm(bsize, device=input.device)
        input_randp = input[rand_idx, 0:1]
        # batch: lab & unlab, backbone extract feats
        src = torch.cat([input, input], dim=0) # b x c x h x w 
        dst = torch.cat([input_trans, input_randp], dim=0) # b x c x h x w
        src_dst = torch.cat([src, dst, 0.5*(src + dst)], dim=1)      
        feats = models_dict['backbone'](src_dst)

        # initial img & pts list
        verbose_flag = False #True if i == 0 else False 
        src_im = ten2pix(src); dst_im = ten2pix(dst)
        src_mask = torch.cat([input_mask, input_mask], dim=0)
        dst_mask = torch.cat([trans_mask_func(input_mask, tgt_grid), input_mask[rand_idx]], dim=0)
        src_pts = input_pts; dst_pts = input_pts[rand_idx, ...]
        img_list = [src_im]; pts_list = [input_pts]
        mask_list = [src_mask]; tag_list = ['init']
        
        # global stage
        if do_global:
            theta, theta_inv = models_dict['fuse'](feats, 'global', verbose=verbose_flag)
            im_grid = get_grid(theta, dst.size(), corner_flag)
            src_global = trans_img_func(img_list[0], im_grid)
            with torch.no_grad():
                src_global_mask = trans_mask_func(src_mask, im_grid) 
                global_pts = theta2pts(src_pts, theta_inv[bsize:])
            loss_global = loss_dict[config.LOSS.REC_GLOBAL](src_global, dst_im, weight=dst_mask * src_global_mask)
            # add global img & pts to list 
            img_list.append(src_global); pts_list.append(global_pts)
            mask_list.append(src_global_mask); tag_list.append('global')
        else: 
            batch_ide_m = ide_m.unsqueeze(0).repeat(dst.size(0), 1, 1).to(dst.device)
            im_grid = F.affine_grid(batch_ide_m, dst.size(), align_corners=corner_flag)
            loss_global = torch.tensor([0.0], dtype=dst.dtype).to(dst.device)
        losses.update('global', loss_global.item(), dst_im.size(0))
        
        loss_local_total = 0; flow_list = []
        # local stage
        for lid in range(1, local_iter_cnt+1):
            pre_src_im = img_list[-1]; pre_src_mask = mask_list[-1]; pre_src_pts = pts_list[-1]
            pre_src = pix2ten(pre_src_im)
            pre_src_dst = torch.cat([pre_src, dst, 0.5*(pre_src + dst)], dim=1)
            ########### Extract Feats & predict flow & foreground mask #######################
            feats = models_dict['backbone'](pre_src_dst)
            flow = models_dict['fuse'](feats, 'local', verbose=verbose_flag)
            flow = resize_feats_func(flow, float(im_w / flow.size(2)), 'seq', 'seq')
            flow_list.append(flow)
            ########### Flow Loss: supervised (synthesized pairs) ###########################
            flow_loss = loss_dict['l2'](flow[:bsize], tgt_grid - im_grid[:bsize])
            im_grid = im_grid + flow # accumulate warp flow to im_grid
            ########## Compute pts & mask after deformation #################################
            pos_flow_grid = ide_grid.to(flow.device) + flow
            neg_flow_grid = ide_grid.to(flow.device) - flow
            with torch.no_grad():
                post_src_mask = trans_mask_func(pre_src_mask, pos_flow_grid)
                post_dst_mask = trans_mask_func(dst_mask, neg_flow_grid)
                post_pts = flow2pts(pre_src_pts, flow[bsize:], corner_flag, inv=True)
            ########## Struct Loss: unsupervised shuffled pairs #############################
            post_src_im = trans_img_func(pre_src_im, pos_flow_grid)
            struct_loss = loss_dict[config.LOSS.REC_LOCAL](post_src_im, dst_im, weight=dst_mask * post_src_mask, \
                pts=post_pts if enable_pts else None, iter=lid)
            if train_flow_inv: # inv transform dst to src
                post_dst_im = trans_img_func(dst_im, neg_flow_grid)
                inv_struct_loss = loss_dict[config.LOSS.REC_LOCAL](pre_src_im, post_dst_im, weight=pre_src_mask * post_dst_mask, \
                    pts=pre_src_pts if enable_pts else None, iter=lid)
                struct_loss = 0.5 * (struct_loss + inv_struct_loss) 
            ########## Smooth Loss: gradient of deformation field ############################
            smooth_loss = loss_dict[config.LOSS.SMOOTH](inv_norm_func(flow), img=pre_src_im)
            
            loss_local = flow_loss * w_lab + struct_loss * w_struct + smooth_loss * w_smooth
            for tag, cur_loss in zip(local_loss_tags, [loss_local, flow_loss, struct_loss, smooth_loss]):
                losses.update(f'{tag}{lid:d}', cur_loss.item(), bsize)
            loss_local_total += loss_local
            # add local img & pts to list
            img_list.append(post_src_im); pts_list.append(post_pts)
            mask_list.append(post_src_mask); tag_list.append(f'local{lid:d}')

        # total
        loss = loss_global + (loss_local_total / local_iter_cnt) * w_local 
        losses.update('total', loss.item(), 2*bsize)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_steps += 1
        
        # evaluate pseudo pts mre
        for pid, tag in enumerate(tag_list):
            norm_mre = (pts_list[pid].detach() - dst_pts).abs().sum(dim=-1)
            metrics.update(f'{tag}_mre', norm_mre.mean().item())
        img_list.append(dst_im); pts_list.append(dst_pts)
        mask_list.append(dst_mask); tag_list.append('dst')
       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % display_period == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.3f} ({loss.avg:.4f}) w_local={wl:.3f}\t' \
                  'global {g.val:.3f} ({g.avg:.4f})\t' \
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=bsize/batch_time.val, data_time=data_time, 
                      loss=losses['total'], wl=w_local, g=losses['global'])
            # print local loss
            for lid in range(1, local_iter_cnt+1):
                for name in local_loss_tags:
                    msg += '{}{:d} {:.3f} '.format(name, lid, losses[f'{name}{lid}'].avg)
            # print pts mre
            for tag in tag_list:
                if tag == 'dst': continue
                mre_avg = metrics[f'{tag}_mre'].avg
                msg += '{}{:.4f}'.format('mre=' if tag=='init' else '->', mre_avg)
            logger.info(msg)

            writer.add_scalars('train_loss', losses.averages(), global_steps)
            prefix = '{}_{}'.format(join(output_dir, 'train'), i)
            # visualize lab without pts 
            vis_list = [img[:bsize].detach() for img in img_list]
            save_batch_crops(vis_list, f'{prefix}_reg_lab.jpg', nrow=len(vis_list))
            # visualize unlab with pts
            vis_list = [img[bsize:].detach() for img in img_list]
            pts_list = [inv_norm_func(pt.detach()) * stride for pt in pts_list]
            save_batch_image_with_joints(
                torch.stack(vis_list, dim=1).flatten(0,1),
                torch.stack(pts_list, dim=1).flatten(0,1),
                torch.ones(bsize * len(vis_list), n_pts, 1),
                f'{prefix}_reg_unlab.jpg', nrow=len(vis_list)
            )
            save_batch_crops([m.detach() for m in mask_list], f'{prefix}_mask.jpg', nrow=len(mask_list))

    writer_dict['train_global_steps'] = global_steps

@torch.no_grad()
def validate(config, exemplar_loader, val_loader, val_dataset, models_dict, loss_dict, output_dir, 
        tb_log_dir, writer_dict, perf_name, **kwargs):
    batch_time = AverageMeter(); losses = AverageMeterSet(); metrics = AverageMeterSet()
    display_period = 1 if config.DEBUG.VIS_ALL_IMAGES_PRED_GT \
        else int(np.ceil(len(val_loader) / 3))
    # switch to evaluate mode
    for k in models_dict: models_dict[k].eval()
    
    num_samples = len(val_dataset)
    # load hyper params 
    do_global = config.TRAIN.DO_GLOBAL; local_iter_cnt = config.TEST.LOCAL_ITER
    REC_GLOBAL = config.LOSS.REC_GLOBAL; REC_LOCAL = config.LOSS.REC_LOCAL
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps'] 
    # compute stride: ratio of image_size / heatmap_size
    im_w, im_h = config.MODEL.IMAGE_SIZE
    hm_w, hm_h = config.MODEL.EXTRA.HEATMAP_SIZE
    stride = im_w // hm_w; n_pts = config.MODEL.NUM_JOINTS
    # compute identity grid
    corner_flag = True
    ide_m = torch.cat([torch.eye(2), torch.zeros(2, 1)], dim=1)
    ide_grid = F.affine_grid(ide_m[None, ...], (1, 3, im_h, im_w), align_corners=corner_flag)
    # function: project coordinates to [-1, 1] or inverse
    norm_func = lambda c: normalize_coords(c, hm_w, hm_h, inv=False, type='[-1,1]')
    inv_norm_func = lambda c: normalize_coords(c, hm_w, hm_h, inv=True, type='[-1,1]')
    img_pad_mode = 'zeros' if config.DATASET.DATASET in ['chest'] else 'zeros' 
    trans_img_func = lambda f,g: transform_feats(f, g, mode='bilinear',
        corner_flag=corner_flag, pad_mode=img_pad_mode)
    resize_feats_func = lambda f,sc,in_type,out_type: resize_feats(f, sc, \
        in_type, out_type, mode='bilinear', corner_flag=corner_flag)

    pix2ten = lambda batch_t: torch.stack([kwargs['norm'](t) for t in batch_t], dim=0)
    ten2pix = lambda batch_t: torch.stack([kwargs['inv_norm'](t) for t in batch_t], dim=0)
    # final evaluate record table
    all_preds = np.zeros((num_samples, n_pts, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_ids, filenames, imgnums = [], [], []
    end = time.time()
    exemplar, _, _, exemplar_meta = iter(exemplar_loader).next()
    exemplar = exemplar[:, 0:1].cuda()
    exemplar_pts = exemplar_meta['joints_norm'].cuda()
    idx = 0
    for i, (input, _, _, meta) in enumerate(val_loader):
        # compute output
        num_images = input.size(0)
        src = exemplar.repeat(num_images, 1, 1, 1)
        dst = input[:, 0:1].cuda()
        src_dst = torch.cat([src, dst, 0.5* (src + dst)], dim=1)
        feats = models_dict['backbone'](src_dst)

        # img & pts list init
        verbose_flag = False #True if i == 0 else False
        src_im = ten2pix(src); dst_im = ten2pix(dst)
        src_pts = exemplar_pts.repeat(num_images, 1, 1)
        dst_pts = meta['joints_norm'].cuda()
        img_list = [src_im]; pts_list = [src_pts]
        tag_list = ['init']; edge_list = [ im2edge(src_im) ]

        # global stage
        if do_global:
            theta, theta_inv = models_dict['fuse'](feats, 'global', verbose=verbose_flag)
            # aff_grid = F.affine_grid(theta, dst.size(), align_corners=corner_flag)
            aff_grid = get_grid(theta, dst.size(), corner_flag)
            src_global = trans_img_func(img_list[0], aff_grid)
            loss_global = loss_dict[REC_GLOBAL](src_global, dst_im)
            # compute global pts 
            global_pts = theta2pts(src_pts, theta_inv)
            img_list.append(src_global); pts_list.append(global_pts)
            tag_list.append('global'); edge_list.append( im2edge(src_global) )
        else:
            batch_ide_m = ide_m.unsqueeze(0).repeat(dst.size(0), 1, 1).to(dst.device)
            aff_grid = F.affine_grid(batch_ide_m, dst.size(), align_corners=corner_flag)
            loss_global = torch.tensor([0.0], dtype=dst.dtype).to(dst.device) 
        losses.update('global', loss_global.item(), num_images)

        loss_local = 0; flow_list = []
        # local stage
        for lid in range(1, local_iter_cnt+1):
            pre_src_im = img_list[-1]; pre_pts = pts_list[-1]
            pre_src = pix2ten(pre_src_im)
            pre_src_dst = torch.cat([pre_src, dst, 0.5*(pre_src + dst)], dim=1)
            ########### Extract Feats & predict flow & foreground mask #############################
            feats = models_dict['backbone'](pre_src_dst)
            flow = models_dict['fuse'](feats, 'local', verbose=verbose_flag)
            flow = resize_feats_func(flow, float(im_w / flow.size(2)), 'seq', 'seq')
            flow_list.append(flow)
            post_pts = flow2pts(pre_pts, flow, corner_flag, inv=True)
            ########### Warp img according to estimated flow & compute loss ########################
            post_src_im = trans_img_func(pre_src_im, ide_grid.to(flow.device) + flow)
            struct_loss = loss_dict[REC_LOCAL](post_src_im, dst_im, pts=post_pts, iter=lid)
            loss_local += struct_loss  
            img_list.append(post_src_im); pts_list.append(post_pts)
            tag_list.append(f'local{lid:d}')
            edge_list.append( im2edge(post_src_im) )
        losses.update('local', loss_local.item(), num_images * local_iter_cnt)

        # total
        loss = loss_global + loss_local
        losses.update('total', loss.item(), num_images)

        # point prediction according to global warp & local deform
        # get pts & norm->[-1,1], warp
        for pid, tag in enumerate(tag_list):
            norm_mre = (pts_list[pid] - dst_pts).abs().sum(dim=-1)
            metrics.update(f'{tag}_mre', norm_mre.mean().item())
        img_list.append(dst_im); pts_list.append(dst_pts)
        tag_list.append('dst'); edge_list.append( im2edge(dst_im) )
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # final evaluate 
        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()
        
        all_norm_mres = [metrics[f'{tag}_mre'].avg for tag in tag_list if tag != 'dst']
        opt_pid = all_norm_mres.index(min(all_norm_mres))
        opt_pred = inv_norm_func(pts_list[opt_pid]).cpu().numpy()
        final_preds = get_final_preds_from_coords(config, opt_pred, c, s)

        maxvals = np.ones((num_images, n_pts, 1))
        cur_idxs = range(idx, idx+num_images)
        all_preds[cur_idxs, :, 0:2] = final_preds[:, :, 0:2]
        all_preds[cur_idxs, :, 2:3] = maxvals
        # double check this all_boxes parts
        all_boxes[cur_idxs, 0:2] = c[:, 0:2]
        all_boxes[cur_idxs, 2:4] = s[:, 0:2]
        all_boxes[cur_idxs, 4] = np.prod(s * config.DATASET.PIXEL_STD, 1)
        all_boxes[cur_idxs, 5] = score
        # image_path.extend(meta['image'])
        batch_image_ids = meta['image_id'].tolist()
        image_ids.extend(batch_image_ids)
        idx += num_images

        if i % display_period == 0:
            msg = 'Test: [{0}/{1}]\t' \
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                    'global {g.val:.4f} ({g.avg:.4f})\t' \
                    'local {l.val:.4f} ({l.avg:.4f})\t'.format(\
                        i, len(val_loader), batch_time=batch_time,
                        loss=losses['total'], g=losses['global'], l=losses['local'])
            for tag in tag_list:
                if tag == 'dst': continue
                mre_avg = metrics[f'{tag}_mre'].avg
                msg += '{}{:.4f}'.format('mre=' if tag == 'init' else '->', mre_avg)
            logger.info(msg)

            writer.add_scalars('val_loss', losses.averages(), global_steps)
            prefix = '{}_{}'.format(os.path.join(output_dir, 'val'), i)
            vis_list = img_list
            pts_list = [inv_norm_func(pt) * stride for pt in pts_list]
            # visualize deformed imgs
            save_batch_image_with_joints(
                torch.stack(vis_list, dim=1).flatten(0, 1),
                torch.stack(pts_list, dim=1).flatten(0, 1),
                torch.ones(num_images * len(vis_list), n_pts, 1),
                f'{prefix}_reg.jpg', nrow=len(vis_list)
            )
            # visualize imgs with transformed keypoints
            save_batch_image_with_joints_and_gts(input, pts_list[opt_pid], meta['joints'], \
                meta['joints_vis'], config.DATASET.LINE_PAIRS, f'{prefix}_gt.jpg')

            # visualize flow & (learned confidence
            save_batch_flows(flow_list, f'{prefix}_flow.jpg', nrow=len(flow_list))
            # visualize edge 
            save_batch_image_with_joints(
                torch.stack(edge_list, dim=1).flatten(0, 1),
                torch.stack(pts_list, dim=1).flatten(0, 1),
                torch.ones(num_images * len(edge_list), n_pts, 1),
                f'{prefix}_edge.jpg', nrow=len(edge_list)
            )
    

    name_values, perf_indicator = val_dataset.evaluate(
        config, all_preds, output_dir, all_boxes, image_ids,
        filenames, imgnums, perf_name=perf_name)
    
    _, full_arch_name = get_model_name(config)
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(logger, name_value, full_arch_name)
    else: 
        _print_name_value(logger, name_values, full_arch_name)

    if isinstance(name_values, list):
        for name_value in name_values:
            writer.add_scalars('valid', dict(name_value), global_steps)
    else:
        writer.add_scalars('valid', dict(name_values), global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1
    return perf_indicator

