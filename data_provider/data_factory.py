# 主要定义了一个data_provider函数，
# 输入args和flag（分为train，validation, test, pred），
# 输出flag相应的数据集 data_set 和 data_loader(使用torch库函数Dataloader装载data_set)
# 导入了函数data_loader中的类Dataset_Custom
from data_provider.data_loader import Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    Data = data_dict[args.data] #用类Dataset_Custom去建立对象Data
    timeenc = 0 if args.embed != 'timeF' else 1  ##这句话的意思是如果不对时间进行编码，也就是args.embed='Fixed' or 'learned'，timeenc=0;否则，args.embed='timeF', timeenc=1

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # data_set是对象Data的实例
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
