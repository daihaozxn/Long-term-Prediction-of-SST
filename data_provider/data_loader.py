
import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

# 定义了各种应用条件下的类，均继承于 torch.utils.data中的类 Dataset
# 以定制的类Dataset_Custom为例（由于用于自己的研究问题，只需要关注这个类即可），包括：
# 函数__init__(self, root_path, flag='train', size=None,
#                  features='S', data_path='ETTh1.csv',
#                  target='OT', scale=True, timeenc=0, freq='h'):
# 函数__read_data__(self): 包含标准化/归一化函数，读取数据
# 函数__getitem__(self, index): 迭代器，在exp_main.py中train/vali/test函数，执行这句for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train/vali/test_loader):时就会调用
# 函数__len__(self): 在data_factory.py中执行到这句  print(flag, len(data_set))时会调用函数__len__(self)，，输出train/val/test序列的长度
# 函数inverse_transform(self, data): 数据逆变换

# 关于 定义函数名前后下划线可参考 https://blog.csdn.net/qq_34525916/article/details/111943575
# https://blog.csdn.net/mahoon411/article/details/125363880
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols 列的名称或叫 列标识
        cols = list(df_raw.columns)
        cols.remove(self.target)  # 移除'OT'
        cols.remove('date')   # 移除'date'
        df_raw = df_raw[['date'] + cols + [self.target]]  #这一步又把‘date’和'OT'加回来是什么意思？
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        # 以下2行应该是 确定train，val和test三种状态的开始序号和结束序号
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len] #减去 seq_len，可以在训练和测试时多几个batch
        # border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':#输入多变量，输出多变量('M') 或 输入多变量，输出单变量('MS')
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':#输入单变量，输出单变量
            df_data = df_raw[[self.target]]

        if self.scale:  #计算train_data数据的 均值 和 标准差，然后去标准化 train、val和test数据
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # 以下4行可参考https://blog.csdn.net/qq_42200107/article/details/126061079
            ## Series.apply(func, 1) = Series.apply(func, convert_dtype=True) = Series.apply(func)，convert_dtype的默认是True
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, convert_dtype=True)#使用它可以定义一个匿名函数。即当你需要一个函数，但又不想费神去命名一个函数，这时候，就可以使用 lambda了。
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            # t = df_stamp.drop(['date'], 1)
            ## 将df_stamp中的date列(axis=1来保证是列)drop掉，然后把values赋给data_stamp
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            ## freq的值决定data_stamp的第1个维度的值
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)#这里调用函数 time_features，时间特征编码
            data_stamp = data_stamp.transpose(1, 0) #交换0轴和1轴，因data_stamp只有两个轴，相当于转置

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    # 关于__getitem__可参考https://blog.csdn.net/weixin_42557907/article/details/81589574
    # https://blog.csdn.net/weixin_39645306/article/details/111221972
    def __getitem__(self, index):  #在exp_main.py中train/vali/test函数，执行这句for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train/vali/test_loader):时就会调用
        s_begin = index  #如果data_factory.py中函数data_provider中train/vali/test/pred状态的shuffle_flag=False时，这里的index从0开始。否则，index随机赋值。
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):  #在data_factory.py中执行到这句  print(flag, len(data_set))时会调用函数__len__(self)，输出train/val/test序列长度
        return len(self.data_x) - self.seq_len - self.pred_len + 1   #注意由于border1s从0开始，故num_train比train序列长度多seq_len+pred_len个数

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
