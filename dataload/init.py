import os
from torch.utils.data import random_split
from .dataload_cdar import CloudRemovalWithMaskDataset

def getdata(config):
    data=config.datasets_dir
    if('RICE' in data):
        from .RICE import TrainDataset
    if(data=='T-Cloud'):
        from .Tcloud import TrainDataset
    if('My' in data):
        from .My import TrainDataset
    if(data=='WHU'):
        from .WHU import TrainDataset
    if (data == 'WHU'):
        from .WHU import TrainDataset
    if (data == 'Allclear'):
        from .Allclear import TrainDataset
        train = TrainDataset(config, json_path=config.allclear_train_json)
        test = TrainDataset(config, json_path=config.allclear_test_json)
        return train, test

    else:
        from .WHU import TrainDataset   #WHU


    #train = TrainDataset(config)
    #test = TrainDataset(config, isTrain=False)
    train = TrainDataset(config, isTrain=True)
    test = TrainDataset(config, isTrain=False)
    return train, test
