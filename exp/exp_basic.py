import os
import torch
import numpy as np
# 定义了一个Exp_Basic(object)的类
# 函数__init__(self, args):
# 函数_build_model(self): 只有两句 raise NotImplementedError 和 return None，
# 函数_acquire_device(self): 此函数主要是根据参数中使用 gpu(多个或单个) 或 cpu进行相应设置
# 函数_get_data(self)/vali(self)/train(self)/test(self)均只有一句命令pass

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError  #可参考https://blog.csdn.net/qq_40666620/article/details/105026716
        return None

    def _acquire_device(self):  #此函数主要是对于使用 gpu(多个或单个) 或 cpu的设置
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass   #pass 不做任何事情，一般用做占位语句。

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
