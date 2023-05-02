from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch
import torchvision
import cv2

from core.inference import get_max_preds
from flow_vis import flow_to_color 

def save_batch_crops(batch_crop_list, file_name, nrow=8, padding=2):
    '''
    batch_crops: [batch_size, channel, height, width] list of len num_joints
    '''
    num_joints = len(batch_crop_list)
    batch_size, _, height, width = batch_crop_list[0].size()
    batch_crops = torch.stack(batch_crop_list, dim=1).view(batch_size * num_joints, -1, height, width)
    grid = torchvision.utils.make_grid(batch_crops, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    cv2.imwrite(file_name, ndarr)    

# visualize predicted flow (local iteration)
def save_batch_flows(batch_flow_list, file_name, nrow=8, padding=2):
    '''
    batch_flow_list: len(list) = local_iters
    '''
    batch_flow_list = [
        np.stack([flow_to_color(flow) for flow in batch_flow.cpu().numpy()], axis=0)
    for batch_flow in batch_flow_list]
    batch_flow_list = [torch.from_numpy(batch_flow).permute(0, 3, 1, 2).float()
        for batch_flow in batch_flow_list]
    save_batch_crops(batch_flow_list, file_name, nrow, padding)
    

def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    normalize = False if batch_image.min() > 0 and batch_image.max() < 1 else True
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, normalize)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint0 = x * width + padding + joint[0]
                joint1 = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint0), int(joint1)), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)
    
def get_color(order_idx, cycles=[3, 4]):
    color = [0]
    for cycle in cycles:
        c = (order_idx % cycle)*255.0/cycle
        color.append(int(c))
    return color 
    
def save_batch_image_with_joints_and_gts(batch_image, batch_pred_joints, batch_gt_joints, batch_joints_vis, 
                                 line_pairs, file_name, nrow=8, padding=2, batch_clf_info=None):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_pred_joints: [batch_size, num_joints, 3], 
    batch_gt_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    batch_clf_info: [batch_size, num_joints, 2],
    '''
    # normalize = True
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()
    
    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints_pred = batch_pred_joints[k]
            joints_gt = batch_gt_joints[k]
            joints_vis = batch_joints_vis[k]
             
            pred_joint_list = []
            for idx, (joint_pred, joint_gt, joint_vis) in enumerate(zip(joints_pred, joints_gt, joints_vis)):
                pred_joint0 = int(x * width + padding + joint_pred[0])
                pred_joint1 = int(y * height + padding + joint_pred[1])
                pred_joint_list.append((pred_joint0, pred_joint1))
                gt_joint0= int(x * width + padding + joint_gt[0])
                gt_joint1 = int(y * height + padding + joint_gt[1])
                if joint_vis[0]:
                    # first plot ground truth joint, make sure it is under pred joint
                    cv2.circle(ndarr, (gt_joint0, gt_joint1), 1, [0, 0, 255], 2)
                    cv2.circle(ndarr, (pred_joint0, pred_joint1), 1, [0, 255, 0], 2)
                    # info_str = ' {:d}'.format(int(idx))
                    # if batch_clf_info is not None:
                    #     pred_class_label, pred_prob = batch_clf_info[k, idx]
                    #     info_str = ' {:d} {:.2f}'.format(int(pred_class_label), float(pred_prob))
                    # cv2.putText(ndarr, info_str, (pred_joint0, pred_joint1), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 0], 1)
        
            for (idx1, idx2) in line_pairs:
                x1, y1 = pred_joint_list[idx1]
                x2, y2 = pred_joint_list[idx2]
                cv2.line(ndarr, (x1, y1), (x2, y2), [255, 0, 0], 1, 8)
                
            k = k + 1
    
    cv2.imwrite(file_name, ndarr, [cv2.IMWRITE_JPEG_QUALITY, 95])

def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)
    if batch_image.size(1) == 1:
        batch_image = batch_image.repeat(1, 3, 1, 1)
        
    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))
        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix, line_pairs=[], batch_clf_info=None):
    if not config.DEBUG.DEBUG:
        return

    # if config.DEBUG.SAVE_BATCH_IMAGES_GT:
    #     save_batch_image_with_joints(
    #         input, meta['joints'], meta['joints_vis'],
    #         '{}_gt.jpg'.format(prefix)
    #     )
    # if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
    #     save_batch_image_with_joints(
    #         input, joints_pred, meta['joints_vis'],
    #         '{}_pred.jpg'.format(prefix)
    #     )
        
    if config.DEBUG.SAVE_BATCH_IMAGES_GT and config.DEBUG.SAVE_BATCH_IMAGES_PRED:
         save_batch_image_with_joints_and_gts(
                    input, joints_pred, meta['joints'], meta['joints_vis'], line_pairs,
                    '{}_pred_gt.jpg'.format(prefix), batch_clf_info=batch_clf_info)
        
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )

