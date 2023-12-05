# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .builder import MODELS
from .misc import Result, interpolate_pos_encoding
import diffdist.functional as diff_dist
import torch.distributed as dist


class ProjectMLP(nn.Module):

    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP, self).__init__()
        # hidden layers
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Conv1d(
            in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:

        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            # [B, C] -> [B, L, C]
            x = x.unsqueeze(1)
            add_dim = True

        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')

        if add_dim:
            x = x.squeeze(1)

        return x


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous() for _ in range(dist.get_world_size())]
    out_list = diff_dist.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)  # Gumbel softmax
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}  # hard for gumbel
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class NPR(nn.Module):
    """NPR: Generate the prototypes for the group tokens.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_output_group (int): Number of output groups.
        stage (int,default = 10): Gaussian learning iteration number.
        mom (float): momentum update parameters (gamma)

    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_output_group,
                 stage = 10,
                 mom = 0.9,
                 ):
        super(NPR, self).__init__()
        self.dim = dim
        self.num_output_group = num_output_group
        self.stage = stage
        self.mom = mom
        mu = torch.Tensor(1, num_output_group, out_dim)
        mu.normal_(0, math.sqrt((2. / out_dim)))
        mu = mu / (1e-6 + mu.norm(dim=2, keepdim=True))

        self.register_buffer('mu', mu)
        self.mapping_x = ProjectMLP(in_dim=dim, out_dim=out_dim)

    def forward(self, x, group_tokens, return_attn=False, is_training=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map
            is_training (bool): whether momentum update to the prototype

        Returns:
            mu (torch.Tensor): [B, S_1, C], Updated prototype
            attn_map (torch.Tensor): [B,L,S_1], Cross_attention map between image and prototype

        """

        stage_num = self.stage
        mom = self.mom
        N, G, C = group_tokens.shape
        mu = self.mu.repeat(N, 1, 1)
        x_id = x
        with torch.no_grad():
            for i in range(stage_num):
                z = torch.matmul(x_id, mu.permute(0, 2, 1))
                z = F.softmax(z, dim=-1)
                z = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.matmul(z.permute(0, 2, 1), x_id)
                mu = mu / (1e-6 + mu.norm(dim=2, keepdim=True))

        recon_x = torch.matmul(z, mu)
        recon_x = self.mapping_x(recon_x)
        new_z = torch.matmul(x_id, mu.permute(0, 2, 1))
        new_z = F.softmax(new_z, dim=-1)
        attn_map = new_z / (1e-6 + new_z.sum(dim=1, keepdim=True))

        # momentum update
        if is_training:
            self.mu = mom * self.mu + (1 - mom) * (dist_collect(mu).mean(dim=0, keepdim=True))


        return mu, attn_map.permute(0, 2, 1).unsqueeze(1), recon_x





class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens


    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """

        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)

        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens,
                                                      x)  # multi-head pre-assign, query=group, key=value=feature


        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += projected_group_tokens  # Residual part in Grouping block
        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))
        attn_dict['ori_center'] = projected_group_tokens
        attn_dict['ori_center_for_text'] = projected_group_tokens


        return new_x, attn_dict


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False,
                 return_attn=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attn = return_attn

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_return = attn

        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        if self.return_attn:
            return out, attn_return
        else:
            return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(
            self.attn(self.norm_q(query), self.norm_k(key), mask=mask))  # value = norm (key), multi-head=6
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class AttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 return_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True,
            return_attn=return_attn)
        self.return_attn = return_attn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if self.return_attn:
            out, attn = self.attn(self.norm1(x), mask=mask)
            x = x + self.drop_path(out)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.return_attn:
            return x, attn
        else:
            return x


class GroupingLayer(nn.Module):
    """A Transformer layer with Grouping Block for one stage.

    Args:
        dim (int): Number of input channels.
        num_input_token (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0

        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            In PGSeg setting, Grouping Block serves as the downsampling layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        group_projector (nn.Module | None, optional): Projector for the grouping layer. Default: None.
        zero_init_group_token (bool): Whether to initialize the grouping token to 0. Default: False.
        npr (nn.Module | None, optional): Genereate the prototype for group token.
    """

    def __init__(self,
                 dim,
                 num_input_token,
                 depth,
                 num_heads,
                 num_group_token,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 group_projector=None,
                 zero_init_group_token=False,
                 return_attn_o=True,
                 npr=None):

        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None

        # build blocks
        self.depth = depth
        self.return_attn = return_attn_o

        blocks = []
        for i in range(depth):
            blocks.append(
                AttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer,
                    return_attn=self.return_attn))
        self.blocks = nn.ModuleList(blocks)

        self.downsample = downsample
        self.downsample_spatial = npr
        self.input_resolution = num_input_token
        self.use_checkpoint = use_checkpoint

        self.group_projector = group_projector

    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'input_resolution={self.input_resolution}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None
        if self.return_attn:
            self.return_attn_list = []

        inp_x = x
        cat_x = self.concat_x(x, group_token)
        for blk_idx, blk in enumerate(self.blocks):
            if self.return_attn:
                if self.use_checkpoint:
                    cat_x, attn = checkpoint.checkpoint(blk, cat_x)
                else:
                    cat_x, attn = blk(cat_x)
            else:
                if self.use_checkpoint:
                    cat_x = checkpoint.checkpoint(blk, cat_x)
                else:
                    cat_x = blk(cat_x)

        ori_x, group_token = self.split_x(cat_x)

        attn_dict = None
        if self.downsample_spatial is not None:
            low_group, attn_map, re_x = self.downsample_spatial(inp_x, group_token, return_attn=False)
            self.return_attn_list.extend([low_group, attn_map])  # [torch.cat([new_x,group_token.detach()],dim=1)]

        if self.downsample is not None:
            ori_x, attn_dict = self.downsample(ori_x+re_x, group_token, return_attn=return_attn)  # GroupBlocking

        if self.return_attn:
            return ori_x, group_token, attn_dict, self.return_attn_list

        else:
            return x, group_token, attn_dict


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
            int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
            int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1),
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


@MODELS.register_module()
class PGSeg(nn.Module):
    r""" PGSeg
        A PyTorch impl of : `PGSeg`  - https://arxiv.org/abs/2310.19001

    Args:
        img_size (int | tuple[int]): Input image size. Default 224
        patch_size (int | tuple[int]): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 0
        embed_dim (int): Patch embedding dimension. Default: 384
        embed_factors (list[int]): Embedding dim multipliers for each stage.
        depths (list[int]): Depth of each stage
        num_heads (list[int]): Number of heads for each stage
        num_group_tokens (list[int]): Number of group tokens for each stage
        num_output_group (list[int]): Number of output groups for each stage
        hard_assignment (bool): Whether to use hard assignment or not. Default: True
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pos_embed_type (str): Type of positional embedding. Default: 'simple'
        freeze_patch_embed (bool): Whether to freeze patch embedding. Default: False
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=384,
                 embed_factors=[1, 1, 1],
                 depths=[6, 3, 3],
                 num_heads=[6, 6, 6],
                 num_group_tokens=[64, 8, 0],
                 num_output_groups=[64, 8],
                 hard_assignment=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 patch_norm=True,
                 use_checkpoint=False,
                 pos_embed_type='simple',
                 freeze_patch_embed=False,
                 return_attn_o=False, ):
        super().__init__()
        assert patch_size in [4, 8, 16]
        self.num_classes = num_classes
        assert len(embed_factors) == len(depths) == len(num_group_tokens)
        assert all(_ == 0 for _ in num_heads) or len(depths) == len(num_heads)
        # assert len(depths) - 1 == len(num_output_groups)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * embed_factors[len(depths) - 1])
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.pos_embed_type = pos_embed_type
        self.return_attn_o = return_attn_o
        assert pos_embed_type in ['simple', 'fourier']

        norm_layer = nn.LayerNorm

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        if pos_embed_type == 'simple':
            self.pos_embed = self.build_simple_position_embedding()
        elif pos_embed_type == 'fourier':
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            raise ValueError

        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        num_input_token = num_patches
        num_output_token = num_input_token

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            dim = int(embed_dim * embed_factors[i_layer])
            downsample = None
            downsample_npr = None
            if i_layer < self.num_layers - 1:
                out_dim = embed_dim * embed_factors[i_layer + 1]
                downsample = GroupingBlock(
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=num_heads[i_layer],
                    num_group_token=num_group_tokens[i_layer],
                    num_output_group=num_output_groups[i_layer],
                    norm_layer=norm_layer,
                    hard=hard_assignment,
                    gumbel=hard_assignment)

                downsample_npr = NPR(
                    dim=dim,
                    out_dim=out_dim,
                    num_output_group=num_output_groups[i_layer])

                num_output_token = num_output_groups[i_layer]

            if i_layer > 0 and num_group_tokens[i_layer] > 0:
                prev_dim = int(embed_dim * embed_factors[i_layer - 1])
                group_projector = nn.Sequential(
                    norm_layer(prev_dim),
                    MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))

                if dim != prev_dim:
                    group_projector = nn.Sequential(group_projector, norm_layer(prev_dim),
                                                    nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None
            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                # only zero init group token if we have a projection
                zero_init_group_token=group_projector is not None,
                return_attn_o=self.return_attn_o,
                npr=downsample_npr)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token


        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.mapping_text = ProjectMLP(in_dim=embed_dim, out_dim=256)
        self.apply(self._init_weights)
        contrast_t = 0.1
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / contrast_t))
        self.cross_entropy = nn.CrossEntropyLoss()

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        if self.pos_embed_type == 'simple' and 'pos_embed' in state_dict:
            load_pos_embed = state_dict['pos_embed']
            pos_embed = self.pos_embed
            if load_pos_embed.shape != pos_embed.shape:
                H_new = int(self.patch_embed.num_patches ** 0.5)
                W_new = H_new
                H_ori = int(load_pos_embed.shape[1] ** 0.5)
                W_ori = H_ori
                load_pos_embed = F.interpolate(
                    rearrange(load_pos_embed, 'b (h w) c -> b c h w', h=H_ori, w=W_ori, b=1),
                    size=(H_new, W_new),
                    mode='bicubic',
                    align_corners=False)
                load_pos_embed = rearrange(load_pos_embed, 'b c h w -> b (h w) c', h=H_new, w=W_new)
                state_dict['pos_embed'] = load_pos_embed
        return super().load_state_dict(state_dict, strict)

    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.patches_resolution
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False

        return pos_embed

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pos_embed(self, B, H, W):
        # 'simple' mode
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H,
                                             W)  # interpolate the position encoding to the request patch size H*W
        return pos_embed

    def forward_features(self, x, *, return_attn=False):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(
            x)  # 3*H*W -> ((H/P_H) * (W/P_W)) * PATCH_EMBEDDING dim, directly changing 3->patch_embedding with a conv
        # hw_shape -> patch size
        ori_img = x
        img_embed = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(img_embed)  # Drop_out
        group_token = None
        attn_dict_list = []
        if self.return_attn_o:
            self.return_attn_o_list = []

        for layer in self.layers:
            if self.return_attn_o:
                x, group_token, attn_dict, attn_o_list = layer(x, group_token, return_attn=return_attn)  # Grouping layer
                self.return_attn_o_list.append(attn_o_list)
            else:
                x, group_token, attn_dict = layer(x, group_token, return_attn=return_attn)

            attn_dict_list.append(attn_dict)

        x = self.norm(x)

        if self.return_attn_o:
            return x, group_token,attn_dict_list, self.return_attn_o_list
        else:
            return x, group_token, attn_dict_list, []


    @torch.no_grad()
    # HRS strategy, select some group-prototype pair
    def hrs(self,inp,label,threshold=0.1):
        B,G,_ = inp.shape
        ulabel = label.unsqueeze(-1)
        max_n,min_n = torch.max(inp,dim=1,keepdim=True)[0],torch.min(inp,dim=1,keepdim=True)[0]
        norm_inp = (inp - min_n) / (max_n - min_n + 1e-6)
        norm_inp = torch.gather(norm_inp,dim=1,index=ulabel)
        return (norm_inp > threshold).squeeze(-1)

    def closs(self, group, proto):

        batch_size, G, C = group.shape

        # get label globally
        labels = torch.arange(G, dtype=torch.long, device=group.device)  # + batch_size * dist.get_rank()
        labels = labels.unsqueeze(dim=0).repeat(batch_size, 1)
        # [B, C]
        group = F.normalize(group, dim=-1)
        proto = F.normalize(proto, dim=-1)

        logits_per_gp = group @ (proto).permute(0, 2, 1)
        threshold_ind_gp = self.hrs(logits_per_gp, labels)

        logits_per_pg = proto @ (group).permute(0, 2, 1)
        threshold_ind_pg = self.hrs(logits_per_pg, labels)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)

        logits_per_gp = logits_per_gp.permute(0, 2, 1)
        logits_per_pg = logits_per_pg.permute(0, 2, 1)


        loss_gp = self.cross_entropy(logits_per_gp[threshold_ind_gp] * logit_scale, labels[threshold_ind_gp])
        loss_pg = self.cross_entropy(logits_per_pg[threshold_ind_pg] * logit_scale, labels[threshold_ind_pg])

        loss = loss_gp + loss_pg

        return loss

    @torch.no_grad()
    def hug_match(self, inp, target):
        from scipy.optimize import linear_sum_assignment
        import numpy as np

        inp = F.normalize(inp, dim=-1)
        target = F.normalize(target, dim=-1)
        similarty_matrix = torch.matmul(inp, target.permute(0, 2, 1)).cpu()
        indice = np.array([linear_sum_assignment(similarty_matrix[i])[1] for i in range(inp.shape[0])])
        batch_indece = np.array([[i] * inp.shape[1] for i in range(inp.shape[0])])
        return batch_indece.flatten(),indice.flatten()


    def selected_center(self, center, low_center):
        B, G, C = center.shape
        # for i in range(B):
        selected_index = torch.randperm(G)
        center = center[:, selected_index, :]
        low_center = low_center[:, selected_index, :]
        return center, low_center

    def forward_with_NPR(self, x, attn_list, attn_o_list, text):

        attn_center1, attn_center2 = attn_list[0]['ori_center'], attn_list[1]['ori_center']
        low_center1, low_center2 = attn_o_list[0][0], attn_o_list[1][0]
        attn_text_center = attn_list[1]['ori_center_for_text']

        # Matched prototype-group pair in the stage 1 (64) I-NPR
        b1, t1 = self.hug_match(attn_center1, low_center1)
        match_center1 = low_center1[b1, t1, :].reshape(attn_center1.shape)

        # Matched prototype-group pair in the stage 2 (8) I-NPR
        b2, t2 = self.hug_match(attn_center2, low_center2)
        match_center2 = low_center2[b2, t2, :].reshape(attn_center2.shape)

        # Matched prototype-group pair in the stage 2 (8) T-NPR
        mapping_text_center2 = attn_text_center.mean(dim=-1, keepdim=True)
        b3, t3 = self.hug_match(mapping_text_center2, text)
        match_textcenter = text[b3, t3, :].reshape(text.shape)

        para = 0.1
        loss1 = self.closs(attn_center1, match_center1)
        loss2 = self.closs(attn_center2, match_center2)
        loss3 = self.closs(mapping_text_center2, match_textcenter)
        loss = para * (loss1 + loss2 + para * loss3)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x, loss


    def forward_image_head(self, x):
        """

        Args:
            x: shape [B, L, C]

        Returns:

        """
        # [B, L, C]
        x = self.avgpool(x.transpose(1, 2)[:, :, -self.num_group_tokens[0]:])  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x

    def forward(self, x, *, return_feat=False, return_attn=False, as_dict=False, return_pgloss=False, text=None):

        x, group_token, attn_dicts,attn_o_list = self.forward_features(x, return_attn=return_attn)

        x_feat = x if return_feat else None

        outs = Result(as_dict=as_dict)

        if return_pgloss:
            pred_x, loss_pg = self.forward_with_NPR(x, attn_list=attn_dicts, attn_o_list=attn_o_list, text=text)
            outs.append(loss_pg, name='loss_pg')
        else:
            pred_x = self.forward_image_head(x)

        outs.append(pred_x, name='x')

        if return_feat:
            outs.append(x_feat, name='feat')

        if return_attn:
            outs.append(attn_dicts, name='attn_dicts')

        return outs.as_return()