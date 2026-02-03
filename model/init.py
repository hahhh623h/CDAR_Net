
def init(model,config):
    if('My' in config.datasets_dir):
        config.in_ch=3
        config.out_ch=3
    if (model == 'CDAR_Net'):
        from .CDAR_Net.train_CDAR_Net import train
        print('model: CDAR_Net')
    train(config)