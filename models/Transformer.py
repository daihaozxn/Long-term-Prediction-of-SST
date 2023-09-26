import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np

# 定义了Model(nn.Module):的类
# 函数__init__(self, configs):
#
# 函数forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
#                 enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
#
class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x_enc的形状是[batch_size, seq_len, enc_in], x_mark_enc:[batch_size, seq_len, Embed.py中freq_map中freq对应的那个值]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  ## enc_out: [batch_size, seq_len, d_model]
        ## 注意，由于是用于时间序列预测，encoder的输入使用sliding windows去作采样，因而有相同的长度，所以无需padding mask，即enc_self_mask没有用到。同理，dec_self_mask也没有用到。
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)  ## 输出enc_out: [batch_size, seq_len, d_model]
        # x_dec的形状是[batch_size, label_len+pred_len, dec_in], x_mark_dec: [batch_size, label_len+pred_len, Embed.py中freq_map中freq对应的那个值]
        dec_out = self.dec_embedding(x_dec, x_mark_dec)  ## dec_out: [batch_size, label_len+pred_len, d_model]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)  ## 输出dec_out: [batch_size, label_len+pred_len, c_out]

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]   [batch_size, pred_len, c_out]
