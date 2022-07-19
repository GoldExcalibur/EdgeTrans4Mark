import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

class PositionEmbedding1d(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: b x l x c
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size):
        # padding_idx=0, will return zero vector for idx 0
        super().__init__(3, embed_size, padding_idx=0)
        
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        dev = x.device
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=dev)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingLearned(nn.Module):
    def __init__(self, image_size, num_pos_feats):
        super().__init__()
        # num_pos_feats: here should be dim_model/2
        # since output: 2 * num_pos_feats
        self.w, self.h = image_size
        self.col_embed = nn.Embedding(self.w, num_pos_feats)
        self.row_embed = nn.Embedding(self.h, num_pos_feats)
        self.reset_parameters()


    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        # x: b, c, h, w
        # pos: b, (2 * pos_feats), h, w
        dev = x.device
        x_dim = len(x.size())
        if x_dim == 4:
            b, _, h, w = x.size()
            dev = x.device
            i = torch.arange(w, device=dev) # w x pos_feats
            j = torch.arange(h, device=dev) # h x pos_feats
            x_emb = self.col_embed(i)
            y_emb = self.row_embed(j)
            pos = torch.cat([
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ], dim=-1) # h x w x (2 * pos_feats)
            pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)
        elif x_dim == 3:
            b, n, c = x.size() # x, y
            assert c == 2, 'x is not coords'
            if x.dtype == torch.int32 or x.dtype == torch.int64:
                x_emb = self.col_embed(x[..., 0])
                y_emb = self.row_embed(x[..., 1])
                pos = torch.stack([x_emb, y_emb], dim=-1) # bs, n, num_pos_feats
            elif x.dtype == torch.float32 or x.dtype == torch.float64:
                col_grid = self.col_embed.weight.permute(1, 0).view(1, -1, self.w, 1).repeat(b, 1, 1, 1)
                row_grid = self.row_embed.weight.permute(1, 0).view(1, -1, self.h, 1).repeat(b, 1, 1, 1)
                norm_x = (x[..., 0]/(self.w - 1)) * 2 - 1
                norm_y = (x[..., 1]/(self.h - 1)) * 2 - 1
                norm_x = torch.stack([norm_x, torch.zeros(b, n).to(dev)], dim=-1).unsqueeze(2)
                norm_y = torch.stack([norm_y, torch.zeros(b, n).to(dev)], dim=-1).unsqueeze(2)
                x_emb = F.grid_sample(col_grid, norm_x, mode='bilinear', padding_mode='border')
                y_emb = F.grid_sample(row_grid, norm_y, mode='bilinear', padding_mode='border')
                pos = torch.cat([x_emb.squeeze(-1), y_emb.squeeze(-1)], dim=1).permute(0, 2, 1)
        else:
            assert 0, 'invalid input dimension for learned positional embedding !'
        return pos

# for case where coords is float: bilinear sample from parameters
class PositionEmbeddingBilinear(nn.Module):
    def __init__(self, image_size, num_pos_feats):
        super().__init__()
        self.w, self.h = image_size
        self.embed_table = nn.Parameter(torch.zeros(num_pos_feats, self.h, self.w))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed_table)

    def forward(self, coords):
        # warning: coords [x, y], but norm_coords [y, x]
        # coords: bs x out_h x out_w x 2 
        # output: bs x num_feats x out_h x out_w
        b = coords.size(0)
        norm_coords = torch.stack([
            coords[..., 1]/(self.h-1),
            coords[..., 0]/(self.w-1),
        ], dim=-1) * 2 - 1.0 
        batch_embed_table = self.embed_table.unsqueeze(0).repeat(b, 1, 1, 1)
        coords_pos_embed = F.grid_sample(batch_embed_table, coords, mode='bilinear', padding_mode='border')  
        return coords_pos_embed

class PositionEmbeddingLinear(nn.Module):
    def __init__(self, image_size, num_pos_feats):
        super().__init__()
        self.w, self.h = image_size
        self.num_pos_feats = num_pos_feats
        self.embed_fc = nn.Linear(2, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed_fc.weight, 0.0, 1.0)
        nn.init.constant_(self.embed_fc.bias, 0.0)

    def forward(self, coords):
        # coords: bsize * num_joints * 2
        norm_coords = torch.stack([
            coords[..., 0]/(self.w-1),
            coords[..., 1]/(self.h-1),
        ], dim=-1) * 2 - 1.0 
        coords_embed = self.embed_fc(norm_coords)
        return coords_embed


def build_position_embedding(config, **kwargs):
    pos_embed_type = kwargs['type'] if 'type' in kwargs else config.TRANSFORMER.POS_EMBED_TYPE 
    dim_model = kwargs['dim_model'] if 'dim_model' in kwargs else config.TRANSFORMER.DIM_MODEL
    if pos_embed_type == 'sine':
        return PositionEmbeddingSine(
            num_pos_feats=dim_model, 
            temperature=config.TRANSFORMER.POS_EMBED_TEMP)
    elif pos_embed_type == 'learned':
        return PositionEmbeddingLearned(
            config.MODEL.EXTRA.HEATMAP_SIZE, 
            dim_model//2)
    elif pos_embed_type == 'linear':
        return PositionEmbeddingLinear(
            config.MODEL.EXTRA.HEATMAP_SIZE,
            dim_model)
    elif pos_embed_type == 'bilinear':
        return PositionEmbeddingBilinear(
            config.MODEL.EXTRA.HEATMAP_SIZE,
            dim_model)
    else:
        assert 0, 'unknown position embedding type {} !'.format(pos_embed_type)
