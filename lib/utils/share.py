# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import glob
import argparse
import pickle
import os
from os.path import join, dirname, isdir, isfile, islink, realpath, normpath, abspath, exists
from scipy import interp
import SimpleITK as sitk
import cv2
import json
import shutil
# from . import sub_dir_util
from six import string_types
from collections import OrderedDict
from scipy.ndimage import gaussian_filter1d

def cv2_imwrite(path, img):
    if sys.version_info < (3, 0):
        cv2.imwrite(path.encode('utf-8'), img)
    else:
        cv2.imwrite(path, img)
        
def convert2unicode(pystr, encoding='utf-8'):
    if sys.version_info < (3, 0):
        if isinstance(pystr, str):
            pystr = pystr.decode(encoding)
    else:
        if isinstance(pystr, bytes):
            pystr = pystr.decode()
    return pystr

try:
    this_filename = __file__
    this_dir = dirname(realpath(this_filename))
except NameError:
    import sys
    this_filename = sys.argv[0]
    if sys.platform in ['win32', 'win64']:
        this_dir = join(dirname(realpath(this_filename)), "bone_age\\ba_utils")
    else:
        this_dir = join(dirname(realpath(this_filename)), "bone_age/ba_utils")

def save_pickle(data, fname, verbose=False):
    if sys.version_info < (3, 0):
        fmt = 'w'
    else:
        fmt = 'wb'
    with open(fname, fmt) as f:
        if verbose:
            print('saving pickle @ %s' % (fname))
        pickle.dump(data, f)

def load_pickle(fname, encoding='latin1', verbose=False):
    '''
    In py3, to load file originally pickled using py2, set  encoding="latin1"
    '''
    if not isfile(fname):
        print('loading pickle failed: not a file:', fname)
        raise ValueError
    if verbose:
        print('loading pickle @ %s' % fname)

    if sys.version_info < (3, 0):
        fmt = 'r'
        result = pickle.load(open(fname, fmt))
    else:
        fmt = 'rb'
        result = pickle.load(open(fname, fmt), encoding=encoding)
    return result

def is_equal_file(src_f, dst_f, ext_list=['py', 'c', 'h', 'cpp', 'pyx', 'sh', 'cu', 'yaml']):
    if not isfile(dst_f):
        return False
    if src_f.rsplit('.', 1)[-1] in ext_list and dst_f.rsplit('.', 1)[-1] in ext_list:
        src_str = open(src_f).read()
        dst_str = open(dst_f).read()
        if src_str == dst_str:
            return True

    return False

def copy_file(src_f, dst_f, symlink=False, force_overwrite=True, skip_if_equal=True, observe_only=False, verbose=True):
    if skip_if_equal:
        if is_equal_file(src_f, dst_f):
            if verbose:
                print('[skip] ' + dst_f)
            return
    if not isdir(dirname(dst_f)):
        os.makedirs(dirname(dst_f))
        
    if not symlink:
        if observe_only or verbose:
            print('copying => ' + dst_f)
        if not observe_only:
            shutil.copyfile(src_f, dst_f)
    else:
        if observe_only or verbose:
            print('linking => ' + dst_f)
        if not observe_only:
            if (isfile(dst_f) or islink(dst_f)) and force_overwrite:
                os.unlink(dst_f)
            os.symlink(src_f, dst_f)

def copy_by_ext(src_top, dst_top,
                ext_list=['py', 'pickle', 'xlsx', '.sh', '.c', '.h', '.cu'],
                ignore_list=['py~'],
                rm_dst_first=False,
                skip_if_equal=True,
                exclude_dirs=['Outputs', 'log', 'cfg', 'configs_bak', 'tmp'],
                reverse=False,
                observe_only=False,
                fname_filter=None,
                followlinks=True,
                verbose=False):
    ignore_file_list = ['raise_exp.py', 'clean.sh', 'web_main.py', 'web_shutdown.sh', 'web_startup.sh', 'cython_bbox.c', 'cython_nms.c', 'bone_age_predictor.py']

    if reverse:
        src_top, dst_top = dst_top, src_top
    src_top = normpath(src_top)
    dst_top = normpath(dst_top)
    src_file_list = []
    print('src top: %s' % (src_top))
    for dirpath, dirnames, filenames in os.walk(src_top, topdown=True, followlinks=followlinks):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for bname in filenames:
            fname = join(dirpath, bname)
            if bname in ignore_file_list:
                print('-- ignore %s' % fname)
            elif isfile(fname):
                ext = fname.rsplit('.', 1)[-1]
                if ext in ext_list and ext not in ignore_list:
                    if fname_filter is not None and fname_filter not in fname:
                        continue
                    src_file_list.append(fname)
                    # print('src: ' + fname)
    # return src_file_list
    print("Number of src files: %d" % (len(src_file_list)))
    dst_file_list = [fpath.replace(src_top, dst_top) for fpath in src_file_list]
    if rm_dst_first:
        if isdir(dst_top):
            shutil.rmtree(dst_top)
            print('rmtree %s' % (dst_top))
        
    for i in range(len(src_file_list)):
        copy_file(src_file_list[i], dst_file_list[i], symlink=False, skip_if_equal=skip_if_equal, observe_only=observe_only, verbose=verbose)

            
def float_array_2_str(float_array, decimal=1):
    s = '['
    for i, v in enumerate(float_array):
        if i != 0:
            s += ', '
        v = round(v, decimal)
        v_str = ("%4." + str(decimal) + "f") % v
        s += v_str
    s += ']'
    return s

def is_valid_date(date_str):
    if isinstance(date_str, string_types):
        date_str = str(date_str)
    else:
        return False
    date_str = date_str.strip()
    if len(date_str) != 8:
        return False
    if not date_str.isdigit():
        return False
    return True
    
def get_ymd(date_str):
    date_str = date_str.strip()
    if len(date_str) != 8:
        return -1,-1,-1
    y, m, d = date_str[:4], date_str[4:6], date_str[6:]
    return int(y),int(m),int(d)

def diff_date(date1, date2):
    """
    date1, date2: 'YYYYMMDD'

    return a float representing 'date2 - date1' in year
    """
    y1, m1, d1 = get_ymd(date1)
    y2, m2, d2 = get_ymd(date2)
    
    return (y2 - y1) + (m2 - m1)/12.0 + (d2 - d1)/365.0

def get_age_from_dcm_info(dcm_info, use_patient_age=False):
    warn_msg = '' 
    if use_patient_age:
        patient_age_str = dcm_info['patient_age']
        if patient_age_str[-1] not in ['D', 'W', 'M', 'Y']:
            warn_msg = 'warning.patient_age_format_invalid'
            return 0.0, warn_msg
        if not patient_age_str[:-1].isdigit():
            warn_msg = 'warning.patient_age_not_digit'
            return 0.0, warn_msg
        if patient_age_str[-1] == 'D':
            return float(patient_age_str[:-1])/365.0
        elif patient_age_str[-1] == 'W':
            return float(patient_age_str[:-1])*7/365.0
        elif patient_age_str[-1] == 'M':
            return float(patient_age_str[:-1])/12.0
        elif patient_age_str[-1] == 'Y':
            return float(patient_age_str[:-1])
        else:
            warn_msg = 'warning.patient_age_format_unknown'
            return 0.0, warn_msg
    # use study_date - birth_date
    birth_date_str, study_date_str, series_date_str = dcm_info['birth_date'], dcm_info['study_date'], dcm_info['series_date']
    if not is_valid_date(birth_date_str):
        warn_msg = 'W_BONEAGE_003' #'warning.birth_date_missing'
        return 0.0, warn_msg
    if (not is_valid_date(study_date_str)) and (not is_valid_date(series_date_str)):
        warn_msg = 'W_BONEAGE_004' #'warning.series_date_study_date_missing'
        return 0.0, warn_msg

    if study_date_str != None:
        image_date_str = study_date_str
    else:
        image_date_str = series_date_str
    
    return diff_date(birth_date_str, image_date_str), warn_msg # year age


def get_all_dcm_under(dcm_top_dirs):
    dcm_path_list = []
    for dcm_top_dir in dcm_top_dirs:
        dcm_path_list.extend(get_all_files_under(dcm_top_dir, with_ext=['.dcm', '.DCM', '.dicom', '.DICOM']))
    return dcm_path_list

def wc2mm(w, c):
    w, c = float(w), float(c)
    return c - w/2.0, c + w/2.0

def mm2wc(win_min, win_max):
    win_min, win_max = float(win_min), float(win_max)
    win_width = win_max - win_min
    win_center = (win_max + win_min) / 2.0
    return win_width, win_center

def get_continuous_index(index_list, weight=None, verbose=None):
    if weight is None:
        index_difference = index_list[1:] - index_list[:-1]
        continuous_index = index_list[np.where(index_difference == 1)]
        return np.min(continuous_index), np.max(continuous_index) + 1
    else:
        # by gongping
        island_list = [] # list of tuple (length, weight, start_idx, end_idx)
        start_idx = index_list[0]
        end_idx = index_list[0]
        for idx in index_list[1:]:
            if idx != end_idx + 1:
                # new island
                island_length = end_idx - start_idx + 1
                island_weight = np.sum(weight[start_idx:end_idx+1])
                island_list.append((island_length, island_weight, start_idx, end_idx))
                # init for next island
                start_idx = idx
                end_idx = idx
            else:
                # extend current island
                end_idx += 1
        island_length = end_idx - start_idx + 1
        island_weight = np.sum(weight[start_idx:end_idx+1])
        island_list.append((island_length, island_weight, start_idx, end_idx))
        # find the main land
        island_list = sorted(island_list, reverse=True)
        if verbose:
            for island in island_list:
                print('island: (length, weight, start, end)', island)
        main_land = island_list[0]
        return main_land[2], main_land[3]

def crop_bbox_from_img(img_np, win_bbox):
    x, y, w, h = map(int, win_bbox)
    if len(img_np.shape) == 3:
        crop_np = img_np[y:y+h, x:x+w, :]
    elif len(img_np.shape) == 2:
        crop_np = img_np[y:y+h, x:x+w]
    else:
        assert False, "Invalid image dimension"

    return crop_np

glb_default_cutoff_ratio = 0.05
def get_default_cutoff_ratio():
    global glb_default_cutoff_ratio
    return glb_default_cutoff_ratio

def set_default_cutoff_ratio(cutoff_ratio):
    global glb_default_cutoff_ratio
    glb_default_cutoff_ratio = cutoff_ratio

def calc_window_from_img(img_np, cutoff_ratio=None, min_win_energe=0.1, ext_win=0.05,
                         win_mode='alg', win_bbox=None, out_mode='wc', num_bins=100,
                         weight_sigma=None, weight_truncate=4,
                         verbose=False):
    """
    Calculate window width and window center from histogram of numpy image
    
    'win_mode':  'alg': 100bin connectivity guess
                 'mm': min/max window
    'out_mode':  'wc': window width/center
                 'mm': min/max window

    if win_mode == 'alg'
      100bins
      cutoff_ratio: bin threshold 1/100 * 0.05:
      min_win_energe: if within window energe < min_win_energe, fallback to (min, max) window
      ext_window by 5% per side: ext_win=0.05
    """
    if cutoff_ratio is None:
        cutoff_ratio = get_default_cutoff_ratio()
    
    if win_bbox is not None:
        img_np = crop_bbox_from_img(img_np, win_bbox)
    
    ori_min, ori_max = np.min(img_np), np.max(img_np)
    if win_mode == 'mm':
        return mm2wc(ori_min, ori_max)
    pixel_span = int(ori_max - ori_min)
    bin_num = num_bins
    if pixel_span < bin_num:
        bin_num = pixel_span
    cnt, bins = np.histogram(img_np, bins = bin_num)
    weight = cnt/float(np.sum(cnt))
    if weight_sigma:
        weight = gaussian_filter1d(weight, weight_sigma, mode='nearest', truncate=weight_truncate)
    intervals = np.array(list(zip(bins[:-1], bins[1:])))
    mean_weight = 1.0 / bin_num
    index  = np.where(weight >= mean_weight * cutoff_ratio)[0]
    min_index, max_index = get_continuous_index(index, weight=weight)
    within_window_energe = weight[min_index:max_index+1].sum()
    if within_window_energe < min_win_energe:
        vmin, vmax = ori_min, ori_max
    else:
        vmin = intervals[min_index][0]
        vmax = intervals[max_index][1]
        win_width = vmax - vmin
        if ext_win != 0 and verbose:
            print('ext window: %.1f, %.1f ->' % (vmin, vmax), end=' ')
        vmin = vmin - win_width * ext_win
        vmax = vmax + win_width * ext_win
        if ext_win != 0 and verbose:
            print('%.1f, %.1f' % (vmin, vmax))

    if out_mode == 'wc':
        return mm2wc(vmin, vmax)
    elif out_mode == 'mm':
        return vmin, vmax
    else:
        assert False
        
def extract_dcm_from_dirs(dicom_dir_list):
    dcm_path_list = []
    for dicom_dir in dicom_dir_list:
        path_list = glob.glob(os.path.join(dicom_dir, '*/*/*/*.dcm'))
        dcm_path_list += path_list
    print("Total Number of DICOMs within `dicom_dir_list`", len(dcm_path_list)        )
    return dcm_path_list

def apply_window(img_u16, win_wc):
    w, c = win_wc
    img_u16 = img_u16.copy()
    img_u16 = img_u16.astype(np.float)
    win_up = c + w / 2
    win_down = c - w / 2
    img_u16[img_u16 > win_up] = win_up
    img_u16[img_u16 < win_down] = win_down
    w = win_up - win_down
    img_u16 -= win_down
    img_8u = np.array(img_u16 * 255.0 / w, dtype='uint8')
    return img_8u

def extract_foreground_from_pure_background(img2d_np, verbose=False):
    assert img2d_np.ndim == 2, "expecting 2d image: HxW"
    pixel_max, pixel_min = img2d_np.max(), img2d_np.min()
    if verbose:
        print('pixel max: %d, pixel min: %d' % (pixel_max, pixel_min))
    assert pixel_max > pixel_min, "error.pure_graylevel_image"
    fg_idx = np.where((img2d_np != pixel_max) & (img2d_np != pixel_min))
    min_h, max_h = fg_idx[0].min(), fg_idx[0].max()
    min_w, max_w = fg_idx[1].min(), fg_idx[1].max()
    if verbose:
        print('Original shape: ', img2d_np.shape)
        print('Foreground height: ', min_h, max_h)
        print('Foreground  width: ', min_w, max_w)
    img_np_fg = img2d_np[min_h:max_h+1, min_w:max_w+1]
    return img_np_fg, min_h, max_h, min_w, max_w

def apply_window_to_dcm(dicom_path, extract_fg=True, win_mode='alg'):
    if sys.version_info < (3, 0):
        origin_image_itk = sitk.ReadImage(dicom_path.encode('utf-8'))
    else:
        origin_image_itk = sitk.ReadImage(dicom_path)
    origin_image_np16 = sitk.GetArrayFromImage(origin_image_itk)[0]
    dcm_info = get_dcm_info(origin_image_itk)
    dcm_info['dcm_height'] = origin_image_np16.shape[0]
    dcm_info['dcm_width'] = origin_image_np16.shape[1]
    if extract_fg:
        origin_image_np16, min_h, max_h, min_w, max_w = extract_foreground_from_pure_background(origin_image_np16)
        dcm_info['fg_height'] = max_h - min_h + 1 
        dcm_info['fg_width'] = max_w - min_w + 1
        dcm_info['fg_top_wh'] = min_w, min_h
        dcm_info['fg_bot_wh'] = max_w, max_h
    win_width, win_center = calc_window_from_img(origin_image_np16, win_mode=win_mode)
    dcm_info['cal_win_width'] = win_width
    dcm_info['cal_win_center'] = win_center
    image_np8 = apply_window(origin_image_np16, (win_width, win_center))
    return origin_image_itk, origin_image_np16, image_np8, dcm_info

def apply_window_to_img(img, **kwargs):
    win_width, win_center = calc_window_from_img(img, out_mode='wc', **kwargs)
    img_np8 = apply_window(img, (win_width, win_center))
    return img_np8


def mm_is_inside_wc(min_max, win_wc):
    w_min, w_max = wc2mm(*win_wc)
    if (w_min <= min_max[0] <= w_max) and (w_min <= min_max[1] <= w_max):
        return True, win_wc
    else:
        new_min = min(w_min, min_max[0])
        new_max = max(w_max, min_max[1])
        new_width, new_center = mm2wc(new_min, new_max)
        return False, (new_width, new_center)

dcm_meta_keys = OrderedDict([('patient_age', '0010|1010'), ('birth_date', '0010|0030'), ('series_date', '0008|0021'), ('study_date', '0008|0020'),
                             ('patient_name', '0010|0010'), ('sex', '0010|0040'), ('dcm_win_center', '0028|1050'), ('dcm_win_width', '0028|1051'),
                             ('patient_id', '0010|0020'), ('study_id', '0020|000d'), ('series_id', '0020|000e'), ('sop_id', '0008|0018'),
                             ('instance_number', '0020|0013'),
                             ('institution_name', '0008|0080'),
                             ('study_desc', '0008|1030'), ('series_desc', '0008|103e'),
                             ('body_part', '0018|0015'), ('modality', '0008|0060'), ('manufacturer', '0008|0070'), ('model_name', '0008|1090'),
                             ('KVP', '0018|0060'), ('Tube Current(mA)', '0018|1151'), ('Exposure Time(msec)', '0018|1150'), ('Exposure(mAs)', '0018|1152'), ('SourceImageDist(mm)', '0018|1110'), ('EntranceDose(dGy)', '0040|0302'), ('OrganDose(dGy)', '0040|0316')])

def get_dcm_info(image_sitk):
    image = image_sitk
    img_meta_data_keys = image.GetMetaDataKeys()
    info_dict = {}
    for k, v in dcm_meta_keys.items():
        dcm_val = None
        if v in img_meta_data_keys:
            if int(sys.version[0]) < 3:
                dcm_val = (image.GetMetaData(v.encode('utf-8'))).strip()
            else:
                dcm_val = (image.GetMetaData(v)).strip()
            if len(dcm_val) == 0:
                dcm_val = None
        if k in ['patient_name', 'patient_id', 'body_part'] and dcm_val is not None:
            decode_ok = False
            try:
                if int(sys.version[0]) < 3:
                    dcm_val = dcm_val.decode('utf-8')
                else:
                    dcm_val = dcm_val
                decode_ok = True
                # print('[utf-8] %s: %s' % (k, dcm_val))
            except UnicodeDecodeError as e:
                dcm_val = dcm_val.decode('gbk')
                decode_ok = True
                # print('[gbk] %s: %s' % (k, dcm_val))
            info_dict['%s_decode_ok' % k] = decode_ok
        info_dict[k] = dcm_val
    # for k, v in info_dict.items():
    #    print('\t%s: %s' % (k, v))
    info_dict['dcm_height'] = image.GetHeight()
    info_dict['dcm_width'] = image.GetWidth()
    return info_dict

def get_study_date(dcm_info):
    if 'study_date' in dcm_info:
        return dcm_info['study_date']
    elif 'series_date' in dcm_info:
        return dcm_info['series_date']
    else:
        return 'Unknown'

def get_sub_dir_from_dcm_info(dcm_info, use_sop_id=True):
    id_list = [dcm_info['patient_id'], dcm_info['study_id'], dcm_info['series_id']]
    if use_sop_id:
        if not id_list[0].startswith('ba_p1_'):
            id_list = id_list + [dcm_info['sop_id']]
    sub_dir = '/'.join(id_list)
    return sub_dir

def parse_dcm_list_by_sitk(input_dcm_path_list, use_sop_id=True, extract_fg=True):
    """
    Input:
        list of file paths
    Output:
        dcm_info_dict:    sub_dir -> {'patient_id': ..., 'modality': ...,}
        read_failed_list: list of read failed pathes
    """
    read_failed_list = []
    dcm_info_dict = {}
    total_file_num = len(input_dcm_path_list)
    for i, dcm_path in enumerate(input_dcm_path_list):
        print(u'[%4d/%4d] %s' % (i, total_file_num, dcm_path))
        try:
            origin_image_itk, origin_image_np16, image_np8, dcm_info = apply_window_to_dcm(dcm_path, extract_fg=extract_fg)
            sub_dir = get_sub_dir_from_dcm_info(dcm_info, use_sop_id=use_sop_id)
            dcm_info['path'] = dcm_path
            if sub_dir in dcm_info_dict:
                print('---- conflict with %s ----' % dcm_info_dict[sub_dir]['path'])
                print('')
            dcm_info_dict[sub_dir] = dcm_info
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print('sitk.ReadImage failed: ', dcm_path)
            read_failed_list.append(dcm_path)
            continue
    print('-' * 80)
    print('Failed: %d/%d' % (len(read_failed_list), total_file_num))
    return dcm_info_dict, read_failed_list
    
def count_body_part(dcm_info_dict):
    body_part_dict = {}
    for sub_dir in sorted(dcm_info_dict.keys()):
        dcm_info = dcm_info_dict[sub_dir]
        body_part = dcm_info['body_part']
        if body_part in body_part_dict:
            body_part_dict[body_part].append(sub_dir)
        else:
            body_part_dict[body_part] = [sub_dir]
    return body_part_dict

def show_body_part(part_dict):
    part_cnt_list = [(part_name, len(part_list)) for part_name, part_list in part_dict.items()]
    part_cnt_list = sorted(part_cnt_list, key=lambda tup: tup[1])
    for name, cnt in part_cnt_list:
        try:
            print("Body Part %20s: %d" % (name, cnt))
        except UnicodeEncodeError:
            continue

def count_dcm_attr(attr_name, dcm_info_dict):
    attr_sub_dir_dict = {}
    for sub_dir in sorted(dcm_info_dict.keys()):
        dcm_info = dcm_info_dict[sub_dir]
        attr_val = dcm_info[attr_name]
        if attr_val in attr_sub_dir_dict:
            attr_sub_dir_dict[attr_val].append(sub_dir)
        else:
            attr_sub_dir_dict[attr_val] = [sub_dir]
    return attr_sub_dir_dict

def show_dcm_attr(attr_name, attr_sub_dir_dict):
    attr_cnt_list = [(attr_val, len(sub_dir_list)) for attr_val, sub_dir_list in attr_sub_dir_dict.items()]
    attr_cnt_list = sorted(attr_cnt_list, key=lambda tup: tup[1])
    for attr_val, cnt in attr_cnt_list:
        print("%s %20s: %d" % (attr_name, attr_val, cnt))
    
    
def parse_all_dcm_by_sitk(input_root, save_dcm_info=True, save_failed_list=True, max_limit=None, use_sop_id=True):
    """
    Input:
        input_root: top level directory containing dcms
        
    Output:
        dcm_info_dict:    sub_dir -> {'patient_id': ..., 'modality': ...,}
        read_failed_list: list of read failed pathes
    """
    input_dcm_path_list = get_all_files_under(input_root, skip_ext=['json', 'txt', 'DICOMDIR', 'xlsx', 'xls'])
    if max_limit is not None:
        input_dcm_path_list = sorted(input_dcm_path_list)[:max_limit]
    dcm_info_dict, read_failed_list = parse_dcm_list_by_sitk(input_dcm_path_list, use_sop_id=use_sop_id)
    if save_dcm_info:
        save_fname = normpath(input_root) + '_dcm_info.pkl'
        save_pickle(dcm_info_dict, save_fname)
    if save_failed_list:
        save_fname = normpath(input_root) + '_failed.pkl'
        save_pickle(read_failed_list, save_fname)
    print('-'*80)
    print('-', input_root)
    part_dict = count_body_part(dcm_info_dict)
    show_body_part(part_dict)
    return dcm_info_dict, read_failed_list


def dcm_transformer(dcm_info_dict, npz_savedir=None, png_savedir=None, dcm_savedir=None, dcm_info_filter=None, img_transformer=None, extract_fg=False, win_mode='alg'):
    if not isinstance(dcm_info_dict, dict) and isfile(dcm_info_dict):
        dcm_info_dict = load_pickle(dcm_info_dict)
    total_dcm_num = len(dcm_info_dict)
    failed_sub_dir_list = []
    for i, sub_dir in enumerate(sorted(dcm_info_dict.keys())):
        dcm_info = dcm_info_dict[sub_dir]
        if dcm_info_filter is not None:
            if not dcm_info_filter(dcm_info):
                continue
        print("[%4d/%4d] %s" % (i, total_dcm_num, sub_dir))
        try:
            if npz_savedir is not None or png_savedir is not None:
                origin_image_itk, origin_image_np16, image_np8, _dcm_info = apply_window_to_dcm(dcm_info['path'], extract_fg=extract_fg, win_mode=win_mode)
                if img_transformer is not None:
                    image_np8 = img_transformer(image_np8)
        except:
            failed_sub_dir_list.append(sub_dir)
            import traceback
            traceback.print_exc()
            print('FAILED')
            continue
        if npz_savedir is not None:
            npz_savepath = join(npz_savedir, sub_dir)
            if os.path.exists(npz_savepath) is False:
                os.makedirs(npz_savepath)
                np.savez(join(npz_savepath, 'img.npz'), image_np8)
        # png
        if png_savedir is not None:
            png_savedir = os.path.normpath(png_savedir)
            png_series_path = join(png_savedir+'_' + win_mode, sub_dir)
            if os.path.exists(png_series_path) is False:
                os.makedirs(png_series_path)
            info_fname = join(png_series_path, 'info.pkl')
            save_pickle(_dcm_info, info_fname)
            # write transformed png to path
            png_fname = join(png_series_path, 'img.png')
            if os.path.isfile(png_fname):
                os.remove(png_fname)
            cv2.imwrite(png_fname, image_np8)
            # softlink transformed png
            png_link_dir = png_savedir + '_' + win_mode + '_view'
            if not os.path.exists(png_link_dir):
                os.makedirs(png_link_dir)
            png_link_fname = join(png_link_dir, sub_dir.replace('/', '|') + '.png')
            if not os.path.exists(png_link_fname):
                os.symlink(png_fname, png_link_fname)
        # dcm: softlink
        if dcm_savedir is not None:
            dcm_link_dir = join(dcm_savedir, sub_dir)
            if not exists(dcm_link_dir):
                os.makedirs(dcm_link_dir)
            dcm_link_fname = join(dcm_link_dir, 'img.dcm')
            if os.path.islink(dcm_link_fname):
                os.remove(dcm_link_fname)
            os.symlink(dcm_info['path'], dcm_link_fname)
    return failed_sub_dir_list

def reorganize_all_dcm_under(dcm_input_root, npz_savedir=None, png_savedir=None, dcm_savedir=None, use_sop_id=False, dcm_info_filter=None, img_transformer=None, extract_fg=False, **kwargs):
    dcm_info_dict, failed_read_list = parse_all_dcm_by_sitk(dcm_input_root,
                                                            save_dcm_info=True,
                                                            save_failed_list=True,
                                                            use_sop_id=use_sop_id)
    if npz_savedir is not None or png_savedir is not None or dcm_savedir is not None:
        dcm_transformer(dcm_info_dict, npz_savedir, png_savedir, dcm_savedir,
                        dcm_info_filter=dcm_info_filter,
                        img_transformer=img_transformer,
                        extract_fg=extract_fg, **kwargs)

