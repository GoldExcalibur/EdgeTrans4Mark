from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.RESUME = ''
config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.DATA_DIR = ''
config.GPUS = ''
config.USE_GPU = True
config.WORKERS = 4
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True


MODEL_EXTRAS = {
    
}

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = ''
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = ''
config.MODEL.NUM_JOINTS = 17
config.MODEL.NUM_SKELETONS = 16
config.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
config.MODEL.NUM_FEAT = 0
config.MODEL.IN_CHANNEL = 3 

# common params for transformer
config.TRANSFORMER = edict()
config.TRANSFORMER.DIM_MODEL = 512
config.TRANSFORMER.NHEAD = 8
config.TRANSFORMER.NUM_ENCODER_LAYERS = 6
config.TRANSFORMER.NUM_DECODER_LAYERS = 6
config.TRANSFORMER.DIM_FEEDFORWARD = 2048
config.TRANSFORMER.DROPOUT = 0.1
config.TRANSFORMER.POS_EMBED_TYPE = 'sine'
config.TRANSFORMER.POS_EMBED_TEMP = 10000.0

# config.MODEL.EXTRA = MODEL_EXTRAS[config.MODEL.NAME]
config.MODEL.EXTRA = None
config.MODEL.STYLE = 'pytorch'

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.SUP_W = 1.0
config.LOSS.CONSIST_W = 0.0
config.LOSS.REC_GLOBAL = 'l1'
config.LOSS.REC_LOCAL = 'ssim'
config.LOSS.SMOOTH = 'grad'
config.LOSS.TEMPERATURE = 0.1
config.LOSS.COTEACH = False 
config.LOSS.RAMP_LEN = 250 

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = 'mpii'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TRAIN_ANN = ''
config.DATASET.TEST_SET = 'valid'
config.DATASET.TEST_ANN = ''
config.DATASET.VAL_SET = ''
config.DATASET.VAL_ANN = ''
config.DATASET.DATA_FORMAT = 'png'
config.DATASET.SELECT_DATA = False
config.DATASET.TRAIN_SUBDIR = None 
config.DATASET.POINT_NAMES = None
config.DATASET.LINE_PAIRS = None 
config.DATASET.PIXEL_STD = 200
config.DATASET.COLOR_RGB = False
config.DATASET.AUG_TYPE = 'affine'

config.DATASET.TEST_SUBDIR = None

# training data augmentation
config.DATASET.FLIP = True
config.DATASET.EXPAND_RATIO = 1.25
config.DATASET.SCALE_FACTOR = 0.25
config.DATASET.ROT_FACTOR = 30
config.DATASET.VARIATION = 0.0

# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.STEP_SIZE = 100
config.TRAIN.LR = 0.001
config.TRAIN.MIN_LR = 0.0

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.SCHEDULER = 'reduce'
config.TRAIN.PATIENCE = 50
config.TRAIN.THRESHOLD = 0.001

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140
config.TRAIN.GLOBAL_STEPS = 8000

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True
config.TRAIN.LOCAL_ITER  = 2 
config.TRAIN.DO_GLOBAL = True  
config.TRAIN.GLOBAL_TYPE = 'affine' 
config.TRAIN.ENABLE_PTS_EPOCH = 10000 
config.TRAIN.EMA_ALPHA = 0.9 
config.TRAIN.EPS_RANGE = 0.1

# ssl train
config.SSL = edict()
config.SSL.RAMP_TYPE = None
config.SSL.RAMP_LEN = 0

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
config.TEST.USE_EMA = False
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0
config.TEST.SOFT_NMS = False
config.TEST.LOCAL_ITER = 2 
config.TEST.INFER_TRAIN = False 

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = False
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False
config.DEBUG.VIS_ALL_IMAGES_PRED_GT = False

def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'MODEL':
        if 'EXTRA' in v and 'HEATMAP_SIZE' in v['EXTRA']:
            if isinstance(v['EXTRA']['HEATMAP_SIZE'], int):
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    [v['EXTRA']['HEATMAP_SIZE'], v['EXTRA']['HEATMAP_SIZE']])
            else:
                v['EXTRA']['HEATMAP_SIZE'] = np.array(
                    v['EXTRA']['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.COCO_BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.COCO_BBOX_FILE)

    config.MODEL.PRETRAINED = os.path.join(
            config.DATA_DIR, config.MODEL.PRETRAINED)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    if isinstance(name, list):
        name = '_'.join(name)
    elif isinstance(name, dict):
        name = '_'.join(name.values())
        
    extra = cfg.MODEL.EXTRA
    if 'resnet' in name:
        name = '{model}_{num_layers}'.format(
            model=name,
            num_layers=extra.NUM_LAYERS)
        deconv_suffix = ''.join(
            'd{}'.format(num_filters)
            for num_filters in extra.NUM_DECONV_FILTERS)
        full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name,
            deconv_suffix=deconv_suffix)
    elif 'hrnet' in name:
        model_width = cfg.MODEL.EXTRA.STAGE2.NUM_CHANNELS[0]
        name = '{model}_{w}'.format(
            model=name,
            w=model_width)
        full_name = '{height}x{width}_{name}'.format(
            height=cfg.MODEL.IMAGE_SIZE[1],
            width=cfg.MODEL.IMAGE_SIZE[0],
            name=name)
    else:
        # raise ValueError('Unkown model: {}'.format(cfg.MODEL))
        full_name = name

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])