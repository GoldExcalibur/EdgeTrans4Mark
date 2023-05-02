from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch 

import random
from PIL import ImageFilter

def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0, pixel_std=200.0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * pixel_std
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T # 3x1
    new_pt = np.dot(t, new_pt) # 2x3 or 3x3
    if new_pt.shape[0] == 3: # perspective transform 
        return (new_pt[:2]/new_pt[-1]).astype(pt.dtype)
    elif new_pt.shape[0] == 2:
        return new_pt
    else:
        assert 0, 'invalid transform matrix shape {}'.format(t.shape)

# def perspective_transform(pt, t):
#     # pt: (2,) np.ndarray
#     #  t: (3, 3)
#     assert isinstance(pt, np.ndarray)
#     pt = np.array([[pt]])
#     new_pt = cv2.perspectiveTransform(pt, t)
#     return new_pt.squeeze()


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0, **kwargs):
    if 'var_ratio' in kwargs:
        var_ratio = kwargs['var_ratio']
        trans = get_perspective_transform(center, scale, rot, output_size, var_ratio)
        dst_img = cv2.warpPerspective(img, trans, (int(output_size[0]), int(output_size[1])),
                    flags=cv2.INTER_LINEAR)
    else:
        trans = get_affine_transform(center, scale, rot, output_size)
        dst_img = cv2.warpAffine(img, trans,(int(output_size[0]), int(output_size[1])),
                    flags=cv2.INTER_LINEAR)
    return dst_img

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

## perspective transform ####
def get_perspective_transform(center, scale, rot, output_size, variation,
                            shift=np.array([0, 0], dtype=np.float32), 
                            inv=0, pixel_std=200.0):
    # first compute affine transform
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    if not isinstance(center, np.ndarray):
        center = np.array(center)
    
    # compute scale -> width length
    scale_tmp = scale * pixel_std
    src_w, src_h = scale_tmp
    dst_w, dst_h = output_size
    
    src = np.zeros((4, 2), dtype=np.float32)
    perturb = np.random.uniform(-0.5*variation, variation, (4, 2)) * scale_tmp
    minx, miny = center - scale_tmp * 0.5
    maxx, maxy = center + scale_tmp * 0.5
    src[0] = np.array([minx, miny])
    src[1] = np.array([minx, maxy])
    src[2] = np.array([maxx, miny])
    src[3] = np.array([maxx, maxy])
        
    M_rot = cv2.getRotationMatrix2D(tuple(center), rot, 1.0)
    src = np.concatenate([src + perturb, np.ones((4, 1))], axis=-1)
    src = src @ M_rot.T

    dst = np.zeros((4, 2), dtype=np.float32)
    dst[0] = np.array([0, 0])
    dst[1] = np.array([0, dst_h])
    dst[2] = np.array([dst_w, 0])
    dst[3] = np.array([dst_w, dst_h])
    
    if inv:
        trans = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

    return trans

def get_M(theta, rot_range, logger=None, inv=False, eps=0.0):
    # theta: bsize, 6
    dev = theta.device

    theta = torch.tanh(theta) # [-1, 1]
    M = torch.zeros_like(theta).to(dev)
    t = theta[:, :2] 
    s = 1.0 + theta[:, 2:4] # [0, 2]
    rot = theta[:, 4] * rot_range
    rcos, rsin = torch.cos(rot), torch.sin(rot)
    sh = torch.tan(theta[:, 5] * rot_range)

    if logger:
        info = 'translation ({:.2f}/{:.2f}/{:.2f} scale ({:.2f}/{:.2f}/{:.2f}) rotation ({:.2f}/{:.2f}/{:.2f})'.format( \
            t.min(), t.max(), t.mean(),
            s.min(), s.max(), s.mean(),
            rot.min(), rot.max(), rot.mean()
        )
        logger.info(info)


    if not inv:
        M = torch.stack([
            s[:, 0] * rcos, s[:, 0] * (rcos * sh + rsin), t[:, 0],
            s[:, 1] * (-rsin), s[:, 1] * (-rsin  * sh + rcos), t[:, 1],
        ], dim=1)
    else:
        ts = t / (s + eps)
        M = torch.stack([
            (rcos - rsin * sh)/s[:, 0],
            (-rsin - rcos * sh)/s[:, 1],
            (-rcos + rsin * sh)*ts[:, 0] + (rsin + rcos * sh) * ts[:, 1],
            rsin/s[:, 0],
            rcos/s[:, 1],
            (-rsin)*ts[:, 0] + (-rcos)*ts[:, 1]
        ], dim=1)

    return M

def get_PM(theta, rot_range, eps_range=0.1, logger=None, inv=False):
    # theta: bsize, 6
    dev = theta.device

    theta = torch.tanh(theta) # [-1, 1]
    M = torch.zeros_like(theta).to(dev)
    t = theta[:, :2] 
    s = 1.0 + theta[:, 2:4] # [0, 2]
    rot = theta[:, 4] * rot_range
    rcos, rsin = torch.cos(rot), torch.sin(rot)
    sh = torch.tan(theta[:, 5] * rot_range)
    eps = theta[:, -2:] * eps_range # eps for perspective

    if logger:
        info = 'translation ({:.2f}/{:.2f}/{:.2f} scale ({:.2f}/{:.2f}/{:.2f}) rotation ({:.2f}/{:.2f}/{:.2f})'.format( \
            t.min(), t.max(), t.mean(),
            s.min(), s.max(), s.mean(),
            rot.min(), rot.max(), rot.mean()
        )
        logger.info(info)


    M = torch.stack([
        s[:, 0] * rcos, s[:, 0] * (rcos * sh + rsin), t[:, 0],
        s[:, 1] * (-rsin), s[:, 1] * (-rsin  * sh + rcos), t[:, 1],
        eps[:, 0], eps[:, 1], torch.ones_like(eps[:, 0]),
    ], dim=1)

    return M


     