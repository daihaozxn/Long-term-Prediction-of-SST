# 设置参数，使用exp文件夹中exp_main.py定义的类Exp_Main去定义对象Exp，并传递参数args实例化（初始化）对象exp
# 运行函数run.py单向传递实验参数args给实验主函数exp_main，初始化设置实验参数
import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2033
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
# os.environ['PYTHONHASHSEED'] =str(fix_seed)
# torch.cuda.manual_seed(fix_seed)
# torch.cuda.manual_seed_all(fix_seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic =True

# parse_args()详解 https://blog.csdn.net/zjc910997316/article/details/85319894
parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, InformerStack, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# enc_in: informer的encoder的输入维度
# dec_in: informer的decoder的输入维度
# c_out: informer的decoder的输出维度
# d_model: informer中self-attention的输入和输出向量维度
# n_heads: multi-head self-attention的head数
# e_layers: informer的encoder的层数
# d_layers: informer的decoder的层数
# d_ff: self-attention后面的FFN的中间向量表征维度
# factor: probsparse attention中设置的因子系数
# padding: decoder的输入中，作为占位的x_token是填0还是填1
# distil: informer的encoder是否使用注意力蒸馏
# attn: informer的encoder和decoder中使用的自注意力机制
# embed: 输入数据的时序编码方式
# activation: informer的encoder和decoder中的大部分激活函数
# output_attention: 是否选择让informer的encoder输出attention以便进行分析

parser.add_argument('--hidden_size', type=int, default=10, help='for LSTM')

# model define
parser.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
parser.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')  ## Autoformer使用涉及参数
parser.add_argument('--factor', type=int, default=1, help='attn factor')  ##Autoformer 和 Informer使用涉及参数
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)  ##Informer使用涉及参数
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
## 这三个选项是我们在实验过程中设计的三种不同的编码方式，
# 我们建议直接使用timeF编码，它是将时间戳拆解为月、日、周、小时、分钟等特征，然后将值缩放为[-0.5,0.5]的小数；
# fixed编码同样是拆解为各类特征，但是会将其对应的整数用positional encoding的方式进行编码，
# learned编码同样是拆解为各类特征，不过会将其对应的整数使用Embedding进行学习，在训练过程中动态调整
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
# 以下2行和注释行效果相同
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
# parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder', default=False)
# parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data', default=False)

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_ep{}_pati{}_act{}_eb{}_dt{}_mv{}_dp{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.train_epochs,
            args.patience,
            args.activation,
            args.embed,
            args.distil,
            args.moving_avg,
            args.dropout,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_ep{}_pati{}_act{}_eb{}_dt{}_mv{}_dp{}_{}_{}'.format(args.model_id,
                                                                                                                              args.model,
                                                                                                                              args.data,
                                                                                                                              args.features,
                                                                                                                              args.seq_len,
                                                                                                                              args.label_len,
                                                                                                                              args.pred_len,
                                                                                                                              args.d_model,
                                                                                                                              args.n_heads,
                                                                                                                              args.e_layers,
                                                                                                                              args.d_layers,
                                                                                                                              args.d_ff,
                                                                                                                              args.factor,
                                                                                                                              args.train_epochs,
                                                                                                                              args.patience,
                                                                                                                              args.activation,
                                                                                                                              args.embed,
                                                                                                                              args.distil,
                                                                                                                              args.moving_avg,
                                                                                                                              args.dropout,
                                                                                                                              args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
