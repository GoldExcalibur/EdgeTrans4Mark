import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .Transformers import SelfAttnLayer, CrossAttnLayer

from utils.utils import num_trainable_params, init_weights
from utils.utils import transform_feats, resize_feats
from utils.utils import split_seq, merge_seq, get_stats
from utils.transforms import get_M, get_PM

logger = logging.getLogger(__name__)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PosEmbedLearned(nn.Module):
    def __init__(self, d_model, im_size):
        super().__init__()
        self.d_model = d_model
        dim = d_model // 2
        for name, max_len in zip(['pe_x', 'pe_y'], im_size):
            setattr(self, name, nn.Embedding(max_len, dim))

    def forward(self, x):
        # in: b x c x h x w
        # out: b x (h x w) x c
        bsize, _, h, w = x.size()
        dev = x.device
        pe_x = self.pe_x(torch.arange(w, device=dev)) # w x dim
        pe_y = self.pe_y(torch.arange(h, device=dev)) # h x dim

        pe = torch.cat([
            pe_x[None, None, :, :].repeat(bsize, h, 1, 1),
            pe_y[None, :, None, :].repeat(bsize, 1, w, 1)
        ], dim=-1) # b x h x w x c
        return rearrange(pe, 'b h w c -> b (h w) c')
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, im_size, need_glb_token=False):
        super().__init__()
        self.w, self.h = im_size
        self.pos_embed = PosEmbedLearned(d_model, im_size)
        self.glb_embed = nn.Embedding(1, d_model) if need_glb_token else None 
        
    def forward(self, feats):
        bsize = feats.size(0)
        dev = feats.device
        pos = self.pos_embed(feats)

        token = rearrange(feats, 'b c h w -> b (h w) c')
        seq = token + pos
        if self.glb_embed is not None:
            id0 = torch.tensor(0, device=dev).long()
            return torch.cat([
                self.glb_embed(id0).view(1, 1, -1).repeat(bsize, 1, 1),
                seq, 
            ], dim=1)
        else: return seq


# degree to transform matrix 
def deg2Affine(x, rot_range, **kwargs):
    M = get_M(x, rot_range).view(-1, 2, 3)
    Minv = get_M(x, rot_range, inv=True).view(-1, 2, 3)
    return M, Minv

def deg2Perspective(x, rot_range, eps_range, **kwargs):
    M = get_PM(x, rot_range, eps_range=eps_range).view(-1, 3, 3)
    Minv = torch.pinverse(M)
    return M, Minv

class AttnFuse(nn.Module):
    def __init__(self, feat_size, channels, dim_model, nhead, p, n_layers, \
        flow_stride=2, global_type='affine', rot_range=0.5*math.pi, eps_range=0.1, \
        offset_range=0.5, **kwargs):
        super().__init__()
        self.w, self.h = feat_size
        self.n_feats = len(channels)
        if channels[0] < channels[-1]:
            channels = channels[::-1]
        self.channels = channels # channels of multi-res feature maps
        logger.info('channels of multi-res feature maps: {}'.format(self.channels))
        # attention block hyper params  
        self.n_layers = n_layers
        self.dim_model = dim_model
        self.nhead = nhead
        self.p = p 

        self.flow_stride = flow_stride

        # build attention blocks
        self.feat_sizes = []
        reduces = []
        embeds = []
        blocks = []
        
        for i, dim_in in enumerate(self.channels): # low to high resolutions
            # compute feat_size of current stride 
            stride = (2 ** (self.n_feats-1-i))
            feat_size = (self.w//stride, self.h//stride)
            self.feat_sizes.append(feat_size)
            # reduce head: channel -> dim_model
            reduces.append(
                nn.Conv2d(dim_in, self.dim_model, kernel_size=1, stride=1, padding=0, bias=True)
            )
            # attention_block
            dim_hidden = 4 * self.dim_model
            attn_mode = 'Self' if i in [0] else 'Cross' # lowest resolution with self mode
            blocks.append(
                nn.ModuleList([
                    eval(f'{attn_mode}AttnLayer')(self.dim_model, dim_hidden, nhead, p=p)
                for _ in range(n_layers)])
            )
            # position encodings
            need_glb_token = True if i in [0,1] else False 
            embeds.append(
                PositionalEncoding(self.dim_model, feat_size, need_glb_token)
            )
          
        self.blocks = nn.ModuleList(blocks)
        self.embeds = nn.ModuleList(embeds)
        self.reduces = nn.ModuleList(reduces)
        
        # global head
        if global_type == 'affine':
            out_global_channel = 6 
            self.deg2M_func = lambda x: deg2Affine(x, rot_range)
        elif global_type == 'perspective':
            out_global_channel = 8 
            self.deg2M_func = lambda x: deg2Perspective(x, rot_range, eps_range)
        else: raise ValueError(f'Invalid globa alignment type {global_type}')

        self.global_head = nn.Sequential(
            nn.Linear(self.dim_model, self.dim_model),
            nn.BatchNorm1d(self.dim_model),
            nn.ReLU(),
            nn.Linear(self.dim_model, out_global_channel)
        )
        # local head
        out_local_channel = int(2 * (self.flow_stride ** 2))
        self.local_head = nn.Linear(self.dim_model, out_local_channel)
        self.offset_range = offset_range

    def forward(self, feats, stage, **kwargs):
        '''
        feats: list of several res feat maps of backbone
        hrnet18: 1/4, 1/8, 1/16, 1/32
        out: affine or flow ('global' or 'local')
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False 
        # low to high resolution
        n_res = len(feats) # num of resolutions
        outs = []; hlist = []
        for rid, f in enumerate(feats[::-1]):
            outs.append(
                self.embeds[rid](
                    self.reduces[rid](f)
                )
            )
            hlist.append(f.size(2))
       
        # compute global transform in low resolution
        for lid in range(self.n_layers):
            for rid in range(n_res):
                out = outs[rid]; cur_h = hlist[rid]
                # get memory for cross attention 
                if rid == 2: # 1st idx for high-res, discard glb token
                    memory = outs[rid-1][:, 1:] 
                elif rid > 0:
                    memory = outs[rid-1]
                else: memory = None 
                # high-res branch: b x (hw) x c
                if rid in [2,3]: # sub-window attn for high-res
                    out = split_seq(out, 4, cur_h)
                    memory = split_seq(memory, 4, cur_h//2)
                    out = self.blocks[rid][lid](out, memory=memory)
                    out = merge_seq(out, 4, cur_h//4)
                else:  # low-res branch: b x (1+hw) x c
                    out = self.blocks[rid][lid](out, memory=memory)
                outs[rid] = out # update fused feats (memory) to list
                # last ids for low-res branch
                if rid == 1 and stage == 'global': break 
        
        if stage == 'global':
            theta = self.global_head(outs[1][:, 0])
            M, Minv = self.deg2M_func(theta)
            return M, Minv
        elif stage == 'local':
            local_out = self.local_head(outs[3])
            local_out = rearrange(local_out, 'b (h w) (s1 s2 c) -> b (h s1) (w s2) c', h=self.h, s1=self.flow_stride, s2=self.flow_stride)
            flow = torch.tanh(local_out) * self.offset_range
            return flow 

def get_net(cfg):
    feat_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
    # params from backbone
    channels = cfg.MODEL.EXTRA.STAGE4.NUM_CHANNELS

    # self attention layer params
    dim_model = cfg.TRANSFORMER.DIM_MODEL
    nhead = cfg.TRANSFORMER.NHEAD 
    p = cfg.TRANSFORMER.DROPOUT
    nlayer = cfg.TRANSFORMER.NUM_ENCODER_LAYERS

    rot_range = 0.5 * math.pi 
    eps_range = cfg.TRAIN.EPS_RANGE 
    offset_range = 0.5 
    
    model = AttnFuse(
        feat_size, channels, dim_model, nhead, p, nlayer, 
        flow_stride=2, global_type=cfg.TRAIN.GLOBAL_TYPE,
        rot_range=rot_range, eps_range=eps_range, offset_range=offset_range,
    )

    model.apply(lambda m: init_weights(m, fc_std=0.01)) 
    logger.info('total num params of {}: {}'.format(model, num_trainable_params(model)))
    return model
