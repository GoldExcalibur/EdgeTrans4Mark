from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import copy

from core.inference import get_max_preds


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1

def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros((len(idx) + 1))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i + 1] = dist_acc(dists[idx[i]], thr=thr)
        if acc[i + 1] >= 0:
            avg_acc = avg_acc + acc[i + 1]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0
    if cnt != 0:
        acc[0] = avg_acc
    return acc, avg_acc, cnt, pred

# for cephalometric dataset evaluation
def cepha_metric(pred, target, hm_size, bbox_size):
    # pred, target: b x n x 2 (coords)
    # bbox_size: b x 2
    # if input_type == 'gaussian':
    #     n, c, h, w = output_hm.shape
    #     pred, _ = get_max_preds(output_hm)
    #     target, _ = get_max_preds(target_hm)
    # elif input_type == 'coords':
    #     pred = copy.deepcopy(output_hm)
    #     target = copy.deepcopy(target_hm)
    #     assert 'hm_size' in kwargs
    #     w, h = kwargs['hm_size']
    # else:
    #     assert 0, 'unknown input type {} !'.format(input_type)
    
    w, h = hm_size
    # remap to original size
    wr = bbox_size[:, 0] / w # n 
    hr = bbox_size[:, 1] / h # n 
    x_dis = (pred[:, :, 0] - target[:, :, 0]) * wr[:, np.newaxis]
    y_dis = (pred[:, :, 1] - target[:, :, 1]) * hr[:, np.newaxis]
    # pred: n x c x 2
    # compute distance
    dists = np.sqrt(x_dis * x_dis + y_dis * y_dis) # n x c
    # compute mean & standard deviation
    mre = np.mean(dists, axis=0) # c
    # sd = np.sum((dists - mre)**2, axis=0)/(n-1) if n > 1 else 0.0 # c
    sd = np.std(dists, axis=0) # c
    # compute successful detection rate
    thrs = [2.0, 2.5, 3.0, 4.0]
    sdr_dict = {}
    for thr in thrs:
        mask = np.where(dists < (thr * 10.0), 1, 0)
        sdr = np.sum(mask)/float(mask.flatten().shape[0])
        sdr_dict['sdr_' + str(thr)] = sdr
    return mre, sd, sdr_dict




