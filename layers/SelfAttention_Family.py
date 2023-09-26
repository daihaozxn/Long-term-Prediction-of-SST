import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
import os


class FullAttention(nn.Module):   ## Transformer用  Scaled Dot-Product Attention
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape  ## L: len_q, H: n_heads, E: d_k
        _, S, _, D = values.shape  ## S: len_k, D: d_v即E
        scale = self.scale or 1. / sqrt(E)

        ## 爱因斯坦求和约定  https://blog.csdn.net/mytzs123/article/details/125126430
        ## 实现  Q*K'   scores: [batch_size, n_heads, len_q, len_k]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:  ##  当mask_flag=True时，用于控制 Decoder 中的 subsequent mask
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)  ## Fills elements of self tensor with value where mask is one.

        A = self.dropout(torch.softmax(scale * scores, dim=-1))  ## 注意力权重系数 softmax((Q*K')/sqrt(d_k))
        V = torch.einsum("bhls,bshd->blhd", A, values)   ## 实现  softmax((Q*K')/sqrt(d_k))*V    V: [batch_size, len_q, n_heads, d_v]

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):   ## Informer用
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    # 计算每个query的稀疏性得分，并输出稀疏性得分最高的n_top个query
    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L_Q, D即d_v也就是E(d_k)]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # K_expand:[B,H,L_Q,L_K,E]
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # index_sample:[L_Q, sample_k]
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q   生成一个形状为(L_Q, sample_k)的随机整数二维张量，整数范围在0到L_K(不含)之间。
        # K_sample:[B,H,L_Q,sample_k,E]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # Q_K_sample:[B,H,L_Q,sample_k]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        # M:[B,H,L_Q]  Q_K_sample.max(-1)的输出是一个元组，元组第一个元素是最大值，第二个元素是最大值相应的序号
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)   #对应Informer论文公式(4)
        # M_top:[B,H,n_top]
        M_top = M.topk(n_top, sorted=False)[1]  #输出 M中排名前n_top个数的indices

        # use the reduced Q to calculate Q_K
        # Q_reduce:[B,H,n_top,D]
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        # Q_K: [B,H,n_top,L_K]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        # V:[B,H,L_V,D]
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            # V_sum: [B,H,D]
            V_sum = V.mean(dim=-2)
            # contex: [B, H, L_Q, D]
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            # contex: [B, H, L_V, D]
            contex = V.cumsum(dim=-2)  # cumsum返回dim维度上的逐个元素的累加和
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        # V:[B,H,L_V,D]
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        # attn:[B,H,u,L_Q]
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        # context_in:[B,H,L_Q,D]
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  #将attn和V乘积张量 转换为 张量context_in的类型
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # queries:[B,H,L_Q,D]  keys:[B,H,L_K,D]  values:[B,H,L_V,D]
        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)
        # Algo.(1)  Step 1
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        # Algo.(1) Step 2-5
        # scores_top: [B,H,u,L_K]  index:[B,H,u]
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # /sqrt(D)
        # get the context  Algo.(1) Step 7
        # context: [B, H, L_V, D]
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries  Algo.(1) Step 6
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):  ## Transformer/Informer中的 (Masked) Multi-Head Attention 和 Multi-Head Attention
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape  ## L: len_q
        _, S, _ = keys.shape  ## S: len_k
        H = self.n_heads  ## H: 分头数目

        queries = self.query_projection(queries).view(B, L, H, -1)  ##  queries: [batch_size, len_q, n_heads, d_k]
        keys = self.key_projection(keys).view(B, S, H, -1)          ##  keys: [batch_size, len_k, n_heads, d_k]
        values = self.value_projection(values).view(B, S, H, -1)    ##  values: [batch_size, len_k, n_heads, d_v]

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)  ## out: [batch_size, len_q, n_heads*d_v]  concat操作

        return self.out_projection(out), attn

