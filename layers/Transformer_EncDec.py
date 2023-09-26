import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderStack(nn.Module):   ## 类InformerStack会调用这个EncoderStack，从论文里看是为增加distilling的鲁棒性
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s)
            attns.append(attn)
        # x_stack = torch.cat(x_stack, -2)
        x_stack = torch.cat((x_stack[0], x_stack[2]), -2)

        return x_stack, attns

class ConvLayer(nn.Module):  ## Informer会用到这个类 self-attention distilling
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,  #在Informer原始代码中，此处padding=1，这样才能做到 L严格减半
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        # self.activation = nn.GELU()
        # self.activation = nn.ReLU()
        # self.activation = nn.PReLU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention  ## encoder-self-attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )       ## Multi-Head Attention   应该是调用SelfAttention_Family.py中的AttentionLayer的forward函数
        x = x + self.dropout(new_x)  ## 残差网络的Add操作

        y = x = self.norm1(x)  ## LayerNorm
        ## 下面2行是实现 Position-wise Feed-Forward Networks
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):   ## EncoderLayer层的堆叠
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)  #attn_layers的数目比conv_layers多1层
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention    ## decoder-self-attention
        self.cross_attention = cross_attention  ## encoder-decoder-attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        ## x: [batch_size, label_len+pred_len, d_model]
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])   ##Masked Multi-Head Attention 和 残差网络的Add操作
        x = self.norm1(x)  ## LayerNorm操作

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])  ##Decoder中 Multi-Head Attention（使用了encoder-decoder-attention） 和 残差网络的Add操作

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))  ## y: [batch_size, d_ff, label_len+pred_len]
        y = self.dropout(self.conv2(y).transpose(-1, 1))  ## y: [batch_size, label_len+pred_len, d_model]

        return self.norm3(x + y)


class Decoder(nn.Module):  # DecoderLayer层的堆叠
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)  ## 输出x: [batch_size, label_len+pred_len, d_model]

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:  ## self.projection 对应于 Transformer.py中的初始化函数Decoder中 projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            x = self.projection(x)  ## 输出x: [batch_size, label_len+pred_len, c_out]
        return x
