import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_k = d_model//nhead
        self.nhead = nhead
        self.linears = _get_clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.eps = 1e-9 # masked value in attention matrix

    def attention_with_dropout(self, query, key, value, mask=None):
        # input: query [bs x head x l1 x c] key、value [bs x head x l2 x c]
        # mask: bs x 1 x 1 x l2
        # output: [bs x head x l1 x c]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5) # bs x head x l1 x l2
        if mask is not None:
            scores = scores.masked_fill(mask == 0, self.eps)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        # q, k, v: bs x len x c
        # mask: bs x len 
        if mask is not None: mask = mask.unsqueeze(1)
        bsize = query.size(0)
        query, key, value = \
            [l(x).view(bsize, -1, self.nhead, self.d_k).transpose(1, 2) \
            for l, x in zip(self.linears, (query, key, value))]
        x, attn = self.attention_with_dropout(query, key, value, mask=mask)
        # bs x head x l1 x c -> bs x l1 x (head x c)
        x = x.transpose(1, 2).contiguous().view(bsize, -1, self.nhead * self.d_k)
        return self.linears[-1](x), attn

class SelfAttnLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, nhead, p=0.1, normalize_before=False):
        super(SelfAttnLayer, self).__init__()
        self.self_attn = MultiHeadAttention(in_dim, nhead, dropout=p)
        self.in_dim = in_dim
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, in_dim)
        )
        
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.forward_type = 'pre' if normalize_before else 'post'

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
      
    def forward_post(self, src, pos=None, src_mask=None, return_attn=False):
        '''
        src: b x l x c1, pos: b x l x c2
        '''
        #layer 1
        q = k = self.with_pos_embed(src, pos)
        src2, self_attn = self.self_attn(q, k, src, mask=src_mask)
        src = src + self.drop1(src2)
        src = self.norm1(src)
        #layer 2
        src2 = self.mlp(src)
        src = src + self.drop2(src2)
        src = self.norm2(src)
        if return_attn: return src, self_attn
        else: return src

    def forward_pre(self, src, pos=None, src_mask=None, return_attn=False):
        '''
        src: b x l x c1, pos: b x l x c2
        '''
        # layer 1
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, self_attn = self.self_attn(q, k, src, mask=src_mask)
        src = src + self.drop1(src2)
        # layer 2
        src2 = self.norm2(src)
        src2 = self.mlp(src2)
        src = src + self.drop2(src2)

        if return_attn: return src, self_attn
        else: return src

    def forward(self, src, **kwargs):
        pos = kwargs['pos'] if 'pos' in kwargs else None 
        mask = kwargs['mask'] if 'mask' in kwargs else None 
        return_attn = kwargs['return_attn'] if 'return_attn' in kwargs else False
        return getattr(self, f'forward_{self.forward_type}')(src, pos, mask, return_attn=return_attn)

class CrossAttnLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, nhead, p=0.1, normalize_before=False):
        super(CrossAttnLayer, self).__init__()
        self.self_attn = MultiHeadAttention(in_dim, nhead, dropout=p)
        self.memory_attn = MultiHeadAttention(in_dim, nhead, dropout=p)
        self.in_dim = in_dim
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        # Implementation of Feedforward model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, in_dim)
        )

        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.drop1 = nn.Dropout(p)
        self.drop2 = nn.Dropout(p)
        self.drop3 = nn.Dropout(p)
        self.forward_type = 'pre' if normalize_before else 'post'

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_pos=None, memory_pos=None, return_attn=False):
        # layer 1
        q = k = self.with_pos_embed(tgt, tgt_pos)
        tgt2, self_attn = self.self_attn(q, k, tgt, mask=tgt_mask)
        tgt = tgt + self.drop1(tgt2)
        tgt = self.norm1(tgt)
        # layer 2
        memory_q = self.with_pos_embed(tgt, tgt_pos)
        memory_k = self.with_pos_embed(memory, memory_pos)
        tgt2, cross_attn = self.memory_attn(memory_q, memory_k, memory, mask=memory_mask)
        tgt = tgt + self.drop2(tgt2)
        tgt = self.norm2(tgt)
        # layer 3
        tgt2 = self.mlp(tgt)
        tgt = tgt + self.drop3(tgt2)
        tgt = self.norm3(tgt)

        if return_attn:
            return tgt, self_attn, cross_attn
        else:
            return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_pos=None, memory_pos=None, return_attn=False):
        # layer 1
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos)
        tgt2, self_attn = self.self_attn(q, k, tgt2, mask=tgt_mask)
        tgt = tgt + self.drop1(tgt2)
        # layer 2
        tgt2 = self.norm2(tgt)
        memory_q = self.with_pos_embed(tgt2, tgt_pos)
        memory_k = self.with_pos_embed(memory, memory_pos)
        tgt2, cross_attn = self.memory_attn(memory_q, memory_k, memory, mask=memory_mask)
        tgt = tgt + self.drop2(tgt2)
        # layer 3
        tgt2 = self.norm3(tgt)
        tgt2 = self.mlp(tgt2)
        tgt = tgt + self.drop3(tgt2)

        if return_attn:
            return tgt, self_attn, cross_attn
        else:
            return tgt

    def forward(self, tgt, **kwargs):
        memory = kwargs['memory']
        tgt_mask = kwargs['tgt_mask'] if 'tgt_mask' in kwargs else None 
        memory_mask = kwargs['memory_mask'] if 'memory_mask' in kwargs else None 
        tgt_pos = kwargs['tgt_pos'] if 'tgt_pos' in kwargs else None 
        memory_pos = kwargs['memory_pos'] if 'memory_pos' in kwargs else None
        return_attn = kwargs['return_attn'] if 'return_attn' in kwargs else False
        return getattr(self, 'forward_' + self.forward_type)(tgt, memory, 
            tgt_mask, memory_mask, tgt_pos, memory_pos, return_attn=return_attn)


# module proposed in cvpr2021: transformer tracking
# class CrossAttnLayer(nn.Module):
#     def __init__(self, in_dim, hidden_dim, nhead, p=0.1):
#         super(CrossAttnLayer, self).__init__()
#         self.mh_attns = nn.ModuleList([
#             nn.MultiheadAttetion(in_dim, nhead, dropout=p)
#         for _ in range(4)])

#         self.feedforwards = nn.ModuleList([
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(p),
#             nn.Linear(hidden_dim, in_dim),
#         ] for _ in range(2))

#         self.norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(6)])
#         self.drops = nn.ModuleList([nn.Dropout(p) for _ in range(6)])
    
#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos
    
#     def forward_sa(self, q_src, q_pos, kv_src, kv_pos, \
#         idx, kv_mask=None):
#         '''
#         query, key: with_pos_embed
#         value: no need to forward through with_pos_embed
#         '''
#         q = self.with_pos_embed(q_src, q_pos)
#         k = self.with_pos_embed(kv_src, kv_pos)
#         context = self.mh_attns[idx](q, k,  value=kv_src, attn_mask=kv_mask)[0]
#         return context
       
#     def forward_res(self, src, context, idx):
#         src = src + self.drops[idx](context)
#         src = self.norms[idx](src)
#         return src

#     def forward(self, src1, src2, src1_mask=None, src2_mask=None, pos_src1=None, pos_src2=None):
#         context = self.forward_sa(src1, pos_src1, src1, pos_src1, 0, src1_mask)
#         src1 = self.forward_res(src1, context, 0)

#         context = self.forward_sa(src2, pos_src2, src2, pos_src2, 1, src2_mask)
#         src2 = self.forward_res(src2, context, 1)
        
#         context1 = self.forward_sa(src1, pos_src1, src2, pos_src2, 2, src2_mask)
#         context2 = self.forward_sa(src2, pos_src2, src1, pos_src1, 3, src1_mask)
#         src1 = self.forward_res(src1, context1, 2)
#         src2 = self.forward_res(src2, context2, 3)
        
#         ff = self.feedforwards[0](src1)
#         src1 = self.foward_res(src1, ff, 4)

#         ff = self.feedforwards[1](src2)
#         src2 =self.forward_res(src2, ff, 5)
#         return src1, src2


class GlobalTransform(nn.Module):
    def __init__(self, n_layer, vis_dim, pos_dim, hidden_dim, nhead, p=0.1):
        super(GlobalTransform, self).__init__()
        self.n_layer = n_layer
        self.vis_dim = vis_dim
        self.pos_dim = pos_dim
        self.in_dim = vis_dim + pos_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.attn_layers = _get_clones(
            SelfAttnLayer(self.in_dim, hidden_dim, nhead, p), 
            n_layer)

        final_in_dim = n_layer * self.in_dim
        self.mlp = nn.Sequential(
            nn.Linear(final_in_dim, final_in_dim),
            nn.ReLU(),
            nn.Linear(final_in_dim, 8)
        )
        self.relu = nn.ReLU()
        # self.mlp = nn.Linear(final_in_dim, 8)

    def forward(self, src, pos):
        '''
        src: b x l x f1, pos: b x l x f2
        f1 + f2 = in_dim
        '''
        bsize = src.size(0)
        cats = []
        out = torch.cat([src, pos], dim=-1)
        for idx in range(self.n_layer):
            out = self.attn_layers[idx](out)
            cats.append(out.mean(dim=1)) # b x (f1 + f2)            
        cats = torch.cat(cats, dim=1)
        m = self.mlp(cats)
        return m

class LocalRefine(nn.Module):
    def __init__(self, n_layer, vis_dim, pos_dim, hidden_dim, nhead, p=0.1):
        super(LocalRefine, self).__init__()
        self.n_layer = n_layer
        self.vis_dim = vis_dim
        self.pos_dim = pos_dim
        self.in_dim = vis_dim + pos_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.attn_layers = _get_clones(
            SelfAttnLayer(self.in_dim, hidden_dim, nhead, p), 
            n_layer)

        self.fc = nn.Linear(self.in_dim, 2)

    def forward(self, src, pos):
        '''
        src: b x l x f1, pos: b x l x f2
        f1 + f2 = in_dim
        '''
        out = torch.cat([src, pos], dim=-1)
        for idx in range(self.n_layer):
            out = self.attn_layers[idx](out)
        out = self.fc(out)
        return out 

class GlobalTransform_v2(nn.Module):
    def __init__(self, n_layer, vis_dim, pos_dim, hidden_dim, nhead, p=0.1, out_dim=8):
        super(GlobalTransform_v2, self).__init__()
        self.n_layer = n_layer
        self.vis_dim = vis_dim
        self.pos_dim = pos_dim
        self.in_dim = vis_dim + pos_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.attn_layers = _get_clones(
            SelfAttnLayer(self.in_dim, hidden_dim, nhead, p), 
            n_layer)

        final_in_dim = n_layer * self.in_dim
        self.mlp = nn.Sequential(
            nn.Linear(final_in_dim, final_in_dim),
            nn.ReLU(),
            nn.Linear(final_in_dim, out_dim)
        )

    def forward(self, src, pos):
        '''
        src: b x l x f1, pos: b x l x f2
        f1 + f2 = in_dim
        '''
        bsize = src.size(0)
        cats = []
        out = torch.cat([src, pos], dim=-1)
        for idx in range(self.n_layer):
            out = self.attn_layers[idx](out)
            cats.append(out.sum(dim=1)) # b x (f1 + f2)            
        cats = torch.cat(cats, dim=1)
        m = self.mlp(cats)
        return m

class GlobalTransformv3(nn.Module):
    def __init__(self, n_layers, in_dim, hidden_dim, nhead, p=0.1):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            CrossAttnLayer(in_dim, hidden_dim, nhead, p=p)
        for _ in range(n_layers)])

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), 
            nn.ReLU(),
            nn.Linear(in_dim, 8)
        )

        self.tanh = nn.Tanh()

    def get_M(self, origin_out):
        # bsize x 7
        bsize = origin_out.size(0)
        dev = origin_out.device

        out = self.tanh(origin_out[:, :6])
        t = out[:, :2] # tx, ty
        s = 1.0 + out[:, 2:4] # sx, sy
        rot = out[:, 4] * 0.25 * math.pi
        rcos, rsin = torch.cos(rot), torch.sin(rot)
        sh = out[:, 5] * 0.25 * math.pi
        
        M = torch.zeros(bsize, 9, dtype=out.dtype, device=dev)
        # affine params: 6 degrees
        M[:, 0] = s[:, 0] * rcos
        M[:, 1] = s[:, 0] * rcos * sh + s[:, 0] * rsin
        M[:, 2] = t[:, 0]
        M[:, 3] = - s[:, 1] * rsin 
        M[:, 4] = - s[:, 1] * rsin * sh + s[:, 1] * rcos
        M[:, 5] = t[:, 1] 
        # perspect params: 2 degrees
        M[:, 6:8] = origin_out[:, 6:8]
        M[:, 8] = 1
        return M

    def forward(self, src1, src2, src1_mask=None, src2_mask=None, pos_src1=None, pos_src2=None):
        out1, out2 = src1, src2
        for i in range(self.n_layers):
            out1, out2 = self.layers[i](out1, out2, pos_src1=pos_src1, pos_src2=pos_src2)
        out1 = self.mlp(out1)
        out2 = self.mlp(out2)
        
        M1 = self.get_M(out1)
        M2 = self.get_M(out2)
        return M1, M2