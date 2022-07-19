from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
from os.path import join, exists, dirname, isfile, isdir
import pickle
from collections import defaultdict, OrderedDict
import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from dataset.LandmarkDataset import JointsDataset

logger = logging.getLogger(__name__)

class CEPHADataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None, subdir=None, **kwargs):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.cfg = cfg
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = cfg.DATASET.PIXEL_STD
        self.expand_ratio = cfg.DATASET.EXPAND_RATIO

        if 'ann_file_path' in kwargs:
            self.ann_file_path = join(cfg.DATASET.ROOT, kwargs['ann_file_path'])  
        else:
            if self.is_train:
                self.ann_file_path = join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_ANN)
            else:
                self.ann_file_path = join(cfg.DATASET.ROOT, cfg.DATASET.TEST_ANN)

        # add binary background mask
        if 'return_mask' in kwargs: self.return_mask = kwargs['return_mask']
        # add item_type (semi or sup)
        if 'item_type' in kwargs: self.item_type = kwargs['item_type']
        self.coco = COCO(self.ann_file_path)
        
        # deal with class names
        cats = [cat['name'] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index(subdir)            
        self.num_images = len(self.image_set_index)
        logger.info('=> num_images: {}'.format(self.num_images))

        # basic info for cephalometric dataset (fixed)
        self.num_joints = 19
        self.point_names = ['sella', 'nasion', 'orbitale', 'porion', 'subspinale', 'subpramentale', 
                            'pogonion', 'menton', 'gnathion', 'gonion', 'incision_inferius', 'incision_superius',
                            'upper_lip', 'lower_lip', 'subnasale', 'soft_tissue_pogonion', 'posterior_nasal_spine',
                            'anterior_nasal_spine', 'articulare']
        assert self.num_joints == cfg.MODEL.NUM_JOINTS 
        assert self.num_joints == len(self.point_names)
        
        self.line_pairs = [(0, 1), (0, 9), (1, 4), (1, 6), (2, 3), (4, 5), (7, 9), (8, 9), 
                           (10, 11), (16, 17), (1, 2), (3, 18), (4, 11), (4, 14), (8, 15),
                           (10, 13), (11, 12), (14, 17)]
        self.num_lines =  len(self.line_pairs)

        # no flip pairs in cephalometric
        self.flip_pairs = []
        self.db = self._get_db()
        logger.info('=> load {} samples'.format(len(self.db)))
        
    def _load_image_set_index(self, subdir_path):
        """ image id: int """
        # if self.is_train and subdir_path:
        if subdir_path is None or subdir_path == '':
            image_ids = self.coco.getImgIds()
        else:
            with open(subdir_path, 'r') as f:
                subdirs = f.readlines()
                image_ids = [int(i.strip()) for i in subdirs] 
            f.close()

        return image_ids

    def _get_db(self):
        gt_db = self._load_coco_keypoint_annotations()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db
    
    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        image_path = self.image_path_from_index(index)
        file_name = image_path.split('/')[-1]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1: # cls = 0 background
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0: # obj['keypoints'] all equals to 0
                continue

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            for ipt in range(self.num_joints):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
         
            result_dict = {
                'id': annIds,  
                'image_id': index, 
                'image': image_path, 
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': file_name,
                'imgnum': 0,
            }
            rec.append(result_dict)

        return rec
    
    def image_path_from_index(self, index):
        im_ann = self.coco.loadImgs(index)[0]
        return join(self.root, self.image_set, im_ann['file_name'])

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * self.expand_ratio
        return center, scale

    def evaluate(self, cfg, preds, output_dir, all_boxes, image_ids, 
                 *args, **kwargs):
        res_folder = join(output_dir, 'results')
        if not exists(res_folder):
            os.makedirs(res_folder)
        res_file = join(
            res_folder, 'keypoints_{}_{}_results.json'.format(
                self.__class__.__name__,
                self.image_set.replace('/', '_'))
            )

        # image x obj x (keypoints)
        kpts = defaultdict(list)
        for idx, (kpt, image_id) in enumerate(zip(preds, image_ids)):
            kpts[image_id].append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': image_id, 
            })

        self._write_coco_keypoint_results(
            list(kpts.values()), res_file)
        
        perf_name = kwargs.pop('perf_name', 'MRE')
        info_str = self._do_mre_keypoint_eval(res_file, res_folder)
        name_value = OrderedDict(info_str)
        return name_value, name_value[perf_name]

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [{'cat_id': self._class_to_coco_ind[cls],
                      'cls_ind': cls_ind,
                      'cls': cls,
                      'ann_type': 'keypoints',
                      'keypoints': keypoints
                      }
                     for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> Writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            if len(img_kpts) == 0:
                continue

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))])
            key_points = np.zeros(
                (_key_points.shape[0], self.num_joints * 3), dtype=np.float)

            for ipt in range(self.num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.
            
            result = [{'image_id': img_kpts[k]['image'],
                       'category_id': cat_id,
                       'keypoints': list(key_points[k]),
                       'score': img_kpts[k]['score'],
                       'center': list(img_kpts[k]['center']),
                       'scale': list(img_kpts[k]['scale'])
                       } for k in range(len(img_kpts))]
            cat_results.extend(result)

        return cat_results
        
    def _do_mre_keypoint_eval(self, res_file, res_folder):
        coco_dt = self.coco.loadRes(res_file)
        image_ids = self.coco.getImgIds()
        res_image_ids = coco_dt.getImgIds()
        assert image_ids == res_image_ids

        dist_dict = defaultdict(list)
        n = len(image_ids)
        for index in image_ids:
            gts = self.coco.loadAnns(self.coco.getAnnIds(imgIds=index))
            dts = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=index))
            for gt in gts:
                gt_kpt = np.array(gt['keypoints']).reshape(-1, 3).astype(np.int32)
                vis = np.where(gt_kpt[:, 2] > 1, 1, 0) # vis_flag = 2 (visible & labeled)
                for dt in dts:
                    dt_kpt = np.array(dt['keypoints']).reshape(-1, 3).astype(np.int32)
                    dist = np.sum(np.square(dt_kpt[:, :2] - gt_kpt[:, :2]) * vis[:, np.newaxis], axis=1)
                    dist = np.sqrt(dist)
                    dist_dict[index].append(dist)

        dists = np.concatenate([np.stack(v, axis=0) for v in dist_dict.values()], axis=0)
        mre = np.mean(dists, axis=0)
        sd = np.std(dists, axis=0)
        info_str =[('MRE', np.mean(mre)), ('SD', np.mean(sd))]

        sdr_thrs = [2.0, 2.5, 3.0, 4.0]
        for thr in sdr_thrs:
            mask = np.where(dists < (thr*10.0), 1, 0)
            sdr = float(np.sum(mask))/mask.flatten().shape[0]
            name = 'SDR_' + str(thr)
            info_str.append((name, sdr * 100.0)) 

        return info_str