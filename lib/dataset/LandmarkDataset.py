from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import copy
import logging
import random

import cv2
import kornia
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import transforms
from utils.transforms import fliplr_joints
from utils.utils import get_background_mask, normalize_coords

from PIL import Image
import torchvision.transforms.functional as tF

logger = logging.getLogger(__name__)

class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.num_joints = 0
        self.pixel_std = cfg.DATASET.PIXEL_STD
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT
        
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        # support affine & perspective transform
        self.aug_type = cfg.DATASET.AUG_TYPE
        if self.aug_type == 'affine':
            self.get_warp_func = lambda c,s,r,dst_size: \
                transforms.get_affine_transform(c, s, r, dst_size, pixel_std=self.pixel_std)
            self.warp_func = lambda img,trans,dst_size: \
                cv2.warpAffine(img, trans, tuple(dst_size), flags=cv2.INTER_LINEAR)
        elif self.aug_type == 'perspective':
            self.variation =  cfg.DATASET.VARIATION if self.is_train else 0.0
            self.get_warp_func = lambda c,s,r,dst_size: \
                transforms.get_perspective_transform(c, s, r, dst_size, self.variation, pixel_std=self.pixel_std)
            self.warp_func = lambda img,trans,dst_size: \
                cv2.warpPerspective(img, trans, tuple(dst_size), flags=cv2.INTER_LINEAR)
        else:
            assert 0, 'unknown augmentation type {}'.format(self.aug_type)
        
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.norm_func = lambda c: normalize_coords(c, self.heatmap_size[0], 
            self.heatmap_size[1], inv=False, type='[-1,1]')
        self.stride = self.image_size[0] // self.heatmap_size[0]
        
        self.transform = transform
        self.db = []

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def repeat(self, times):
        self.db = self.db * times 

    def __len__(self,):
        return len(self.db)

    def sample_aug_params(self):
        # flip or not 
        flip_flag = True if self.flip and random.random() <= 0.5 else False
        # sample scale & rotation factor 
        sf = self.scale_factor 
        rf = self.rotation_factor 
        scale = np.clip(np.random.randn()*sf + 1, 1-sf, 1+sf)
        rotation = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0 
        return scale, rotation, flip_flag

    def _getitem_sup(self, data_numpy, db_rec):
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']

        c, s, r = db_rec['center'], db_rec['scale'], 0 
        score = db_rec['score'] if 'score' in db_rec else 1 

        if self.is_train:
            sf, rf, flip_flag = self.sample_aug_params()
            s = s * sf 
            r = r + rf # r=0        
            if flip_flag:
                data_numpy = data_numpy[:, ::-1, :]
                origin_w = data_numpy.shape[1]
                joints, joints_vis = fliplr_joints(\
                    joints, joints_vis, origin_w, self.flip_pairs)
                c[0] = origin_w - c[0] - 1
        else: flip_flag = False

        trans = self.get_warp_func(c, s, r, self.image_size)
        if hasattr(self, 'return_mask') and self.return_mask:
            binary_mask = get_background_mask(trans, \
                (data_numpy.shape[1], data_numpy.shape[0]), self.image_size)

        input = self.warp_func(data_numpy, trans, self.image_size)
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, :2] = transforms.affine_transform(joints[i, 0:2], trans)

        if self.transform: input = self.transform(input)
        target, target_weight = self.generate_target(joints, joints_vis, \
            self.heatmap_size, self.image_size, self.sigma, self.target_type)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {}
        for tag in ['image_id', 'id', 'image', 'filename', 'imgnum']:
            meta[tag] = db_rec[tag] if tag in db_rec else None
        
        joints_norm = torch.from_numpy(joints[..., :2]) / self.stride 
        joints_norm = self.norm_func(joints_norm).float()
        meta.update({
            'joints': joints, 'joints_vis': joints_vis, 
            'center': c, 'scale': s, 'rotation': r, 'score': score, 
            'trans': trans, 'flip': flip_flag,
            'joints_norm': joints_norm,
        })
        if hasattr(self, 'return_mask') and self.return_mask: 
            meta['mask'] = binary_mask[None, ...] # 1 x h x w
        return input, target, target_weight, meta 

    def _getitem_semi(self, data_numpy, db_rec):
        joints = db_rec['joints_3d']
        joints_vis = db_rec['joints_3d_vis']
        
        # crop interested patch from original image
        c, s, r = db_rec['center'], db_rec['scale'], 0 
        score = db_rec['score'] if 'score' in db_rec else 1 
        flip_flag = True if self.is_train and self.flip and (random.random() < 0.5) else False
        if flip_flag: 
            data_numpy = data_numpy[:, ::-1, :]
            origin_w = data_numpy.shape[1]
            joints, joints_vis = fliplr_joints(\
                joints, joints_vis, origin_w, self.flip_pairs)
            c[0] = origin_w - c[0] - 1
        
        trans = self.get_warp_func(c, s, r, self.image_size)
        input = self.warp_func(data_numpy, trans, self.image_size)
        for i in range(self.num_joints):
            if joints_vis[i, 0] > 0.0:
                joints[i, :2] = transforms.affine_transform(joints[i, :2], trans)
        
        if self.transform: input = self.transform(input)
        target, target_weight = self.generate_target(joints, joints_vis, \
            self.heatmap_size, self.image_size, self.sigma, self.target_type)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)
        meta = {}
        for tag in ['image_id', 'id', 'image', 'filename', 'imgnum']:
            meta[tag] = db_rec[tag] if tag in db_rec else None 
        meta.update({
            'joints': joints, 'joints_vis': joints_vis, 
            'center': c, 'scale': s, 'rotation': r, 'score': score, 
            'flip': flip_flag, 'trans': trans,
        })
        if self.is_train: 
            # augment patch for semi-supervised learning
            sf, rf, _ = self.sample_aug_params()
            # warp img
            c_im = np.array(self.image_size) * 0.5
            s_im = np.array(self.image_size) / self.pixel_std 
            trans_im = self.get_warp_func(c_im, sf * s_im, rf, self.image_size)
            input_aug = kornia.warp_affine(\
                input.unsqueeze(0), torch.FloatTensor(trans_im).unsqueeze(0), 
            (self.image_size[1], self.image_size[0]), padding_mode='border').squeeze(0)# -> zeros: worse results(gray border)
            # warp heatmap 
            c_hm = np.array(self.heatmap_size) * 0.5 
            s_hm = np.array(self.heatmap_size) / self.pixel_std 
            trans_hm = self.get_warp_func(c_hm, sf * s_hm, rf, self.heatmap_size)
            target_aug = kornia.warp_affine(\
                target.unsqueeze(0), torch.FloatTensor(trans_hm).unsqueeze(0),
            (self.heatmap_size[1], self.heatmap_size[0])).squeeze(0)
            target_weight_aug = target_weight.clone() # same as origin
            # transform joints 
            joints_aug = np.zeros_like(joints)
            for i in range(self.num_joints):
                if joints_vis[i, 0] > 0.0: #(x,y)
                    joints_aug[i, :2] = transforms.affine_transform(joints[i, :2], trans_im)
            joints_aug[:, -1] = joints[:, -1] # z
            meta.update({'joints_aug': joints_aug})
            return input, target, target_weight, meta, \
                input_aug, target_aug, target_weight_aug, torch.from_numpy(trans_hm).type(input.dtype)
        else:
            return input, target, target_weight, meta


    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])
        # read img from files 
        image_file = db_rec['image']
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        if self.color_rgb: # default cv2.imread is BGR mode
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))
        
        item_type = self.item_type if hasattr(self, 'item_type') else 'sup'
        return getattr(self, '_getitem_{}'.format(item_type))(data_numpy, db_rec)

    @staticmethod
    def generate_target(joints, joints_vis, heatmap_size, image_size, sigma, target_type='gaussian'):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_joints = joints.shape[0]
        assert num_joints == joints_vis.shape[0]
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        assert target_type == 'gaussian', 'Only support gaussian map now!'

        if target_type == 'gaussian':
            target = np.zeros((num_joints,
                               heatmap_size[1],
                               heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = sigma * 3
            # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            for joint_id in range(num_joints):
                feat_stride = image_size / heatmap_size
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight
