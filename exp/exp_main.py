from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Transformer, TransformerStack
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric, RMSE

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

# 警告过滤器用于控制警告消息的行为，如忽略，显示或转换为错误（引发异常）。
# 警告过滤器维护着一个有序的过滤规则列表，匹配规则用于确定如何处理警告，任何特定警告都将依次与列表中的每个过滤规则匹配，直到找到匹配为止。
# 过滤规则类型为一个元组 (action，message，category，module，lineno)
# 具体可参考 https://www.cnblogs.com/hls-code/p/15711054.html
warnings.filterwarnings('ignore')

# 定义了一个Exp_Main(Exp_Basic)的类
# 函数__init__(self, args): super(Exp_Main, self).__init__(args)
# 函数_build_model(self): 根据输入参数中的模型名称（Transformer等）来建立模型
# 函数_get_data(self, flag): 根据flag状态（train/val/test/pred），调用data_factory.py中的data_provider函数，返回相应数据集和数据加载
# 函数_select_optimizer(self): 优化器选取
# 函数_select_criterion(self): 损失函数选取
# 函数vali(self, vali_data, vali_loader, criterion): 验证集（和测试集）的验证
# 函数train(self, setting): 训练集的训练
# 函数test(self, setting, test=0): 测试集的测试
# 函数predict(self, setting, load=False):类似函数test，但只有preds，没有trues，无法计算metrics

class Exp_Main(Exp_Basic):  #继承 类Exp_Basic
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)  #super().__init__()执行父类的构造函数，使得我们能够调用父类的属性。具体可参考https://blog.csdn.net/weixin_43702920/article/details/107802103

    def _build_model(self):
        model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'TransSt': TransformerStack,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.Data = data_set    # inverse-scale用到这行程序
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()#使用nn.MSELoss()创建一个对象criterion
        return criterion
    #在下面的函数train中被调用
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():  #  with torch.no_grad()的用法参考https://blog.csdn.net/weixin_46559271/article/details/105658654
            # with torch.no_grad():使用就像一个循环，其中循环内的每个张量都将requires_grad设置为False。这意味着当前与当前计算图相连的任何具有梯度的张量现在都与当前图分离。我们不再能够计算关于这个张量的梯度。
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                #添加inverse-scale代码
                # mean_X, std_X = self.Data.scaler.mean_, self.Data.scaler.scale_
                # pred = pred * std_X + mean_X
                # true = true * std_X + mean_X

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        # self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # batch_x:[batch_size, seq_len, enc_in]
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):#调用data_loader.py中的类Dataset_Custom中的函数__getitem__(self, index):
                iter_count += 1
                #  关于optimizer.zero_grad()：
                # https://blog.csdn.net/scut_salmon/article/details/82414730
                # https://blog.csdn.net/wanttifa/article/details/93972540
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # torch.zeros_like:生成和括号内变量维度一致的全是零的内容
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  ## 将batch_y中1到label_len长度的数据 和 pred_len个0 作连接

                # encoder - decoder
                if self.args.use_amp: #使用自动混合精度训练，暂且不管
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention: ## output_attention用于控制是否 输出每一个encoder layer的注意力权重系数
                        # models文件夹下各个模型的forward函数中至少需要4个输入参数，第1和第3分别是Encoder和Decoder的输入，
                        # 第2和第4应该是用于Embed.py中的temporal_embedding(由于一般选embed_type为timeF，所以temporal_embedding调用类TimeFeatureEmbedding)
                        # t = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        ## 由于无需补齐encoder和decoder的输入，所以，我觉得下面的batch_y可以不使用。
                        # outputs: [batch_size, seq_len, pred_len]
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    ## 如果预测模式是S，由于最后1个维度是1，所以f_dim=0，也还是意味着只有最后一个维度(效果等同于f_dim=-1)
                    ## 如果预测模式是M，则需要输出最后一个维度的所有值
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion) #这里调用函数vali，而非函数test，主要是考察test数据集的test_loss。函数vali和函数test前面部分基本相同。

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                #添加inverse-scale代码
                mean_X, std_X = self.Data.scaler.mean_, self.Data.scaler.scale_
                outputs = outputs * std_X + mean_X
                batch_y = batch_y * std_X + mean_X
                # pred: [batch_size, pred_len, c_out]
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:  #i是batch的数目（即：test中样本数目 / batch_size）的遍历
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds) #preds: [向下取整(test序列长度/batch_size), batch_size, pred_len, c_out]
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  #preds: [batch_size*向下取整(test序列长度/batch_size), pred_len, c_out]
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('rmse:{}, mse:{}, mae:{}'.format(rmse, mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('rmse:{}\tmse:{}\tmae:{}'.format(rmse, mse, mae))
        f.write('\n')
        # rmse_sigle = [RMSE(preds[:, 0, -1], trues[:, 0, -1])]
        # for i in range(self.args.pred_len//10):
        #     rmse_temp = RMSE(preds[:, i*10+9, -1], trues[:, i*10+9, -1])
        #     rmse_sigle.append(rmse_temp)
        f.write('{:.3f}\t'.format(RMSE(preds[:, 0, -1], trues[:, 0, -1])))
        for i in range(self.args.pred_len // 10):
            f.write('{:.3f}\t'.format(RMSE(preds[:, i*10+9, -1], trues[:, i*10+9, -1])))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        # 画图并保存 test中的观测和预测 到文件夹./results中，如果trues中间维度是-1，则画的是若干个连续的pred_len中最后一个点
        # plt.figure()
        # plt.plot(trues[:, -1, -1], label='Ground Truth')
        # plt.plot(preds[:, -1, -1], label='Prediction')
        # plt.legend()
        # plt.savefig(folder_path + 'timeseries.png')
        visual(trues[:, -1, -1], preds[:, -1, -1], os.path.join(folder_path, 'trues+preds.pdf'))
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
