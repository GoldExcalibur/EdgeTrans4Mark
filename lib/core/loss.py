from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple

import math
import numpy as np
from einops import rearrange
import kornia 

from utils.utils import get_gaussian_maps, get_grid, normalize_coords

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
    
### Perceptual Loss ###########
class ConsistencyLoss(nn.Module):
    """
    Loss function for landmark prediction
    Input:
        loss_type: string, 'perceptual' or 'l2'
    """
    def __init__(self, loss_type='perceptual', vggnet=None):
        super(ConsistencyLoss, self).__init__()
        self.loss_type = loss_type
        self.vggnet = vggnet
        
    def forward(self, input_1, input_2):
        if self.loss_type == 'perceptual':
            loss = self.perceptual_loss(input_1, input_2)
        elif self.loss_type == 'mse':
            loss = F.mse_loss(input_1, input_2)
        elif self.loss_type == 'bce':
            loss = F.binary_cross_entropy_with_logits(input_1, input_2)
        elif self.loss_type == 'perceptual_color_hm':
            #Colorize heatmaps and apply perceptual loss
            loss = self.perceptual_loss(self.hm2color(input_1), self.hm2color(input_2))
        elif self.loss_type == 'perceptual_gray_hm':
            loss = self.perceptual_loss(self.hm2gray(input_1), self.hm2gray(input_2))
        else:
            raise ValueError('Incorrect loss_type for consistency loss', self.loss_type)

        return loss

    def perceptual_loss(self, pred_image, gt_image, ws=[50., 40., 6., 3., 3., 1.],
                        names=['input', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2']):

        #get features map from vgg
        feats_gt = self.vggnet(gt_image)
        feats_pred = self.vggnet(pred_image)

        losses = []
        for k, w in zip(names, ws):
            if k == 'input':
                loss = F.mse_loss(pred_image, gt_image, reduction='mean')
            else:
                loss = F.mse_loss(feats_pred[k], feats_gt[k], reduction='mean')
            #print('loss at layer {} is {}'.format(v, l))
            loss /= w
            losses.append(loss)
        loss = torch.stack(losses).sum()
        return loss

    def hm2gray(self, hm):
        #Convert heatmap to grayscale. Then stack 3 dimensions for input to VGG
        gray_hm = torch.sum(hm, dim=1).unsqueeze(1)
        gray_hm = torch.cat([gray_hm, gray_hm, gray_hm], dim=1)
        return gray_hm

# weighted L1 Loss
class WeightedL1Loss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target, weight=None):
        err = (input - target).abs()
        if not self.size_average: return err.sum()
        if weight is None: return err.mean()
        else: 
            weight_spatial_mean = weight / (weight.sum(dim=[-1, -2], keepdim=True)+1e-9)
            return (err * weight_spatial_mean).sum(dim=[-1,-2]).mean()

class WeightedL2Loss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average 

    def forward(self, input, target, **kwargs):
        weight = kwargs['weight'] if 'weight' in kwargs else None 
        err = (input - target)**2
        if not self.size_average: return err.sum()
        if weight is None: return err.mean()
        else: 
            weight_spatial_mean = weight / (weight.sum(dim=[-1,-2], keepdim=True)+1e-9)
            return (err * weight_spatial_mean).sum(dim=[-1, -2]).mean()
            
#########################################################
# pytorch implementation of ssim by https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
#########################################################
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float()[None, None, ...]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window 

class SSIM(nn.Module):
    def __init__(self, channel, window_size=11, mode='ssim', reduction=None):
        super().__init__()
        self.window_size = window_size 
        # assert channel in [1, 3]
        self.channel = channel 
        self.window = create_window(window_size, self.channel)
        ### load mode & reduction ############################
        assert mode in ['u', 'c', 's', 'ssim', 's3im']
        self.mode = mode 
        self.reduction = reduction # [mean, batch, None]

        
    def forward(self, img1, img2, **kwargs):
        weight = kwargs['weight'] if 'weight' in kwargs else None
        window = self.window.to(img1.device).type_as(img1)
        max_val = 255 if img1.max() > 128 else 1.
        min_val = -1 if img1.min() < -0.5 else 0. 
        L = max_val - min_val 
        C1 = (0.01 * L)**2; C2 = (0.03 * L)**2 
        
        ################## umap: mean divergence #####################################
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=self.channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=self.channel)
        mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1 * mu2 

        if self.mode == 'u': return (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        ################# cmap: standard deviation ###################################
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=self.channel) - mu1_sq 
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=self.channel) - mu2_sq
        sigma1_sq = F.relu(sigma1_sq); sigma2_sq = F.relu(sigma2_sq) # fixme: sometimes sigma_sq negative
        if self.mode == 'c': return (2*(sigma1_sq * sigma2_sq)**0.5 + C2)/(sigma1_sq + sigma2_sq + C2)

        ################# smap: correlation #########################################
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=self.channel) - mu1_mu2 
        if self.mode == 's': return (sigma12 + C2*0.5) / ((sigma1_sq * sigma2_sq)**0.5 + 0.5*C2)

        ################ structural similarity ######################################
        if self.mode == 'ssim':
            ssim_map = ((2 * mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        elif self.mode == 's3im':
            ssim_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        
        if self.reduction == 'mean': #
            if weight is None: return ssim_map.mean() 
            else: 
                weight_spatial_mean = weight / weight.sum(dim=[-1,-2], keepdim=True)
                out = (ssim_map * weight_spatial_mean).sum(dim=[-1, -2]).mean()
        elif self.reduction == 'batch': out = ssim_map.mean(dim=[1, 2, 3])
        elif self.reduction is None: out = ssim_map # b x c x h x w
        else: raise ValueError('Invalid reduction value {} !'.format(self.reduction))
        return 1.0 - out

def get_grad(img, norm=False):
    '''
    img: b x c x h x w (tensor)
    '''
    grad = kornia.filters.spatial_gradient(img, order=1, \
        normalized=True, mode='sobel') # b x c x 2 x h x w
    grad = grad.flatten(1,2) # b x (c x 2) x h x w
    if norm:
        bsize, ch, _, _ = grad.size()
        bmin = grad.view(bsize, ch, -1).min(dim=-1)[0][:, :, None, None]
        bmax = grad.view(bsize, ch, -1).max(dim=-1)[0][:, :, None, None]
        grad = (grad - bmin) / (bmax - bmin + 1e-9)
    return grad 
        
class EdgeSSIM(nn.Module):
    def __init__(self, channel, im_size, reduction='mean', **kwargs):
        super().__init__()
        self.channel = channel 
        self.im_w, self.im_h = im_size
        mode = kwargs.pop('mode', 'ssim')
        window_size = kwargs.pop('window_size', 11)
        self.ssim_im = SSIM(channel, window_size, mode=mode, reduction='mean')
        self.ssim_edge = SSIM(channel*2, window_size, mode='ssim', reduction=None)
     
    def forward(self, img1, img2, **kwargs):
        '''
        pts: b x njoint x 2; [-1,1]
        '''
        weight = kwargs.pop('weight', None)
        pts = kwargs.pop('pts', None)
        struct_loss = self.ssim_im(img1, img2, weight=weight)
        if pts is None: # no pseudo pts info provided
            return struct_loss 
        
        pts = torch.clamp(pts, -1.0, 1.0)
        edge1 = get_grad(img1[-pts.size(0):], norm=True)
        edge2 = get_grad(img2[-pts.size(0):], norm=True)
        pts_mask = get_gaussian_maps(pts, (self.im_w, self.im_h), 1/0.1, mode='rot')
        pts_mask = pts_mask.mean(dim=1, keepdim=True) # b x 1 x h x w
        edge_loss = self.ssim_edge(edge1, edge2)
        edge_loss = (edge_loss * pts_mask).sum(dim=[-1,-2], keepdim=True) / \
            (pts_mask.sum(dim=[-1,-2], keepdim=True) + 1e-9)
        return 0.5 * (struct_loss + edge_loss.mean())


# deformation field smooth loss 
class FlowGrad(nn.Module):
    def __init__(self, penalty='l1',):
        super().__init__()
        assert penalty in ['l1', 'l2']
        self.penalty = penalty 

    def forward(self, flow, **kwargs):
        '''
        flow: b x h x w x ndim. ndim=2
        '''
        ndim = len(flow.size())-2
        dy = torch.abs(flow[:, 1:, :, :] - flow[:, :-1, :, :])
        dx = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])

        if self.penalty == 'l2':
            dy = dy * dy 
            dx = dx * dx 

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / ndim
        return grad 

# smooth loss: decay weight on edges
class EdgeGrad(nn.Module):
    def __init__(self, im_size, penalty='l2',  **kwargs):
        super().__init__()
        self.penalty = penalty
        self.im_w, self.im_h = im_size
        self.T = kwargs['T'] if 'T' in kwargs else 0.1
        corner_flag = kwargs['corner_flag'] if 'corner_flag' in kwargs else True
        ygrid, xgrid = torch.meshgrid(torch.arange(self.im_h), torch.arange(self.im_w))
        self.ide_grid = torch.stack([xgrid, ygrid], dim=-1).unsqueeze(0).float() # 1 x h x w x 2
        self.norm_func = lambda g: normalize_coords(g, self.im_w, self.im_h, type='[-1,1]')
        self.trans_func = lambda f,g: F.grid_sample(f, g, mode='nearest', \
            padding_mode='border', align_corners=corner_flag)
    
    @torch.no_grad()
    def get_edge_mask(self, im):
        grad = kornia.filters.sobel(im, normalized=True, eps=1e-6) # b x c x h x w
        grad = grad.mean(dim=1, keepdim=True) # b x 1 x h x w
        return torch.exp(-grad / self.T)

    def forward(self, flow, **kwargs):
        '''
        flow: b x h x w x 2
        img: b x c x h x w
        '''
        bsize = flow.size(0); dev = flow.device
        edge_mask = self.get_edge_mask(kwargs['img']) # b x 1 x h x w

        offset = torch.randn_like(flow, device=dev)
        offset = F.normalize(offset, dim=-1)

        warp_grid = self.norm_func(self.ide_grid.to(dev) + offset)
        flow = flow.permute(0, 3, 1, 2) # b x 2 x h x w
        warp_flow = self.trans_func(flow, warp_grid)
        if self.penalty == 'l1': diff = (flow - warp_flow).abs()
        elif self.penalty == 'l2': diff = ((flow - warp_flow)**2)
        diff = (diff * edge_mask).sum(dim=[-2,-1]) / edge_mask.sum(dim=[-2, -1])
        return diff.mean()

        


