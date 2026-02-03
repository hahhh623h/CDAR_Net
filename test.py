import os
import shutil
import yaml
from attrdict import AttrMap
import utils.utils as utils
from utils.utils import gpu_manage, save_image, checkpoint
from dataload.init import getdata
from torch.utils.data import DataLoader
import torch
# from eval import test,testcomp
from eval import test3,test2,test
from model.CDAR_Net.train_CDAR_Net import train as train_cdar_net

#from thop import profile
import time
import torch.optim as optim
import numpy as np

'''if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    utils.make_manager()
    os.makedirs(config.out_dir, exist_ok=True)

    # 保存本次训练时的配置
    shutil.copyfile('config.yml', os.path.join(config.out_dir, 'config.yml'))

    gpu_manage(config)

    ### DATASET LOAD ###
    print('===> Loading datasets')
    train_dataset, validation_dataset = getdata(config)
    print('validation dataset:', len(validation_dataset))
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads, batch_size=config.validation_batchsize, shuffle=False)

    ### MODELS LOAD ###
    print('===> Loading models')
    if(config.model=='spagan'):
        from model.models_spagan.gen.SPANet import Generator,SPANet
        gen = Generator(gpu_ids=config.gpu_ids,channel =config.in_ch)
    if(config.model=='amgan'):
        from model.models_amgan_cr.gen.AMGAN import Generator,AMGAN
        gen = Generator(in_ch=config.in_ch, out_ch=config.out_ch, gpu_ids=config.gpu_ids)
    if(config.model=='mdsa'):
        from model.msda.model import Generate_quarter
        gen = Generate_quarter(in_channels=config.in_ch, height=3, width=6, num_dense_layer=4, growth_rate=16)
    if(config.model=='mn'):
        from model.mn.model import MPRNet
        gen = MPRNet(in_c=config.in_ch, out_c=config.in_ch)
    if(config.model=='cvae'):
        from model.cvae.network import VAE
        gen = VAE(in_channels=config.in_ch)
    if (config.model == 'mdsa_efem_MSE'):
        from model.msda.train_msda_efem_MSE import Generate_quarter_efem_01
        gen = Generate_quarter_efem_01(height=3, width=6, num_dense_layer=4, growth_rate=16, in_channels=config.in_ch)
    if (config.model == 'mdsa_efem'):
        from model.msda.train_msda_efem import Generate_quarter_efem_01
        gen = Generate_quarter_efem_01(height=3, width=6, num_dense_layer=4, growth_rate=16, in_channels=config.in_ch)
    if (config.model == 'mdsa_MSE'):
        from model.msda.train_msda_MSE import Generate_quarter_efem_01
        gen = Generate_quarter_efem_01(height=3, width=6, num_dense_layer=4, growth_rate=16, in_channels=config.in_ch)
    if (config.model == 'B_transformer'):
        from model.B_transformer.network import B_transformer
        gen = B_transformer()
    if (config.model == 'MSDA_CAFF_CADA_LOSS'):
        from model.msda.train_MSDA_CAFF_CADA_LOSS import Generate_quarter
        gen = Generate_quarter()
    if config.model == 'CDAR_Net':
        train_cdar_net(config)  # <--- 调用新的训练函数

    gen=gen.cuda()

    # input = torch.randn(2, 3, 256, 256).cuda()
    # all_time=[]
    # code to show the Computational complexity (speed, parameters,memory,complexity(GFLOPs))
    # for _ in range(100):
    #     time_start = time.time()
    #     predict = gen(input)
    #     time_end = time.time()
    #     all_time.append(time_end-time_start)
    # print('Speed: %.5f\n' % (1/np.mean(all_time)))
    #flops, params = profile(gen, inputs=(input, ))
    # print('Complexity: %.3fM' % (flops/1000000000), end=' GFLOPs\n')
    # optimizer = optim.SGD(gen.parameters(), lr=0.9, momentum=0.9, weight_decay=0.0005)
    # for _ in range(1000):
    #     optimizer.zero_grad()
    #     gen(input)

    # param = torch.load('./pre_train/'+os.listdir('./pre_train/')[0])
    with open(os.path.join(config.out_dir, 'config.yml'), 'r', encoding='UTF-8') as f:
        config1 = yaml.load(f, Loader=yaml.FullLoader)
    config1 = AttrMap(config1)
    param = torch.load(config1.gen_init)
    gen.load_state_dict(param, False)
    print('load {} as pretrained model'.format(config.gen_init))
    # print('load {} as pretrained model'.format(os.listdir('./pre_train/')[0]))
    criterionMSE = torch.nn.MSELoss()
    with torch.no_grad():
        log_validation = test2(config, validation_data_loader, gen, criterionMSE, 201)'''

# test.py
#模型测试
'''
import os
import shutil
import yaml
from attrdict import AttrMap
import utils.utils as utils
from utils.utils import gpu_manage
from dataload.init import getdata
from torch.utils.data import DataLoader
import torch

# 从新模型文件中导入 CDAR_Net_Generator
from model.msda.CDAR_Net import CDAR_Net_Generator

# 从我们新创建的评估文件中导入测试函数
from evaluation_CDAR import test_cdar_net

if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    # 确保输出目录存在
    os.makedirs(config.out_dir, exist_ok=True)

    # 管理GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    ### 数据集加载 ###
    print('===> Loading datasets')
    _, validation_dataset = getdata(config)
    print('Validation dataset size:', len(validation_dataset))
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=config.threads,
        batch_size=config.validation_batchsize,
        shuffle=False
    )

    ### 模型加载 ###
    print('===> Loading models')
    # 根据 config.model 初始化模型
    if config.model == 'CDAR_Net':
        gen = CDAR_Net_Generator(
            height=config.height,
            width=config.width,
            num_dense_layer=config.num_dense_layer,
            growth_rate=config.growth_rate,
            in_channels=config.in_ch
        )
        print("Model 'CDAR_Net' initialized.")
    else:
        # 这里可以保留或删除其他模型的加载逻辑
        # 为了清晰，我们假设如果不是 CDAR_Net 就报错
        raise ValueError(
            f"Model '{config.model}' is not supported by this test script. Please set model to 'CDAR_Net' in config.yml.")

    # 将模型移动到设备
    gen = gen.to(device)

    # 加载预训练的权重
    if not config.gen_init or not os.path.exists(config.gen_init):
        raise FileNotFoundError(
            f"Generator weights not found at path: {config.gen_init}. Please check 'gen_init' in config.yml.")

    param = torch.load(config.gen_init, map_location=device)
    gen.load_state_dict(param)
    print(f'Loaded pretrained generator from: {config.gen_init}')

    ### 执行测试 ###
    test_cdar_net(config, validation_data_loader, gen, device)'''

# 模型消融
import os
import yaml
from attrdict import AttrMap
from torch.utils.data import DataLoader
import torch
'''
# 从模型文件中导入所有可能的生成器
from model.msda.CDAR_Net import Generator_MCAM_Only, Generator_MCAM_CAFF, CDAR_Net_Generator

# 评估脚本保持不变
from evaluation_CDAR import test_cdar_net
from dataload.init import getdata

if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    os.makedirs(config.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")

    _, validation_dataset = getdata(config)
    validation_data_loader = DataLoader(dataset=validation_dataset, num_workers=config.threads,
                                        batch_size=config.validation_batchsize, shuffle=False)

    print(f"===> Loading Ablation Mode for Testing: {config.ablation_mode}")
    gen_args = {'height': config.height, 'width': config.width, 'num_dense_layer': config.num_dense_layer,
                'growth_rate': config.growth_rate, 'in_channels': config.in_ch}

    if config.ablation_mode == 'mcam_only':
        gen = Generator_MCAM_Only(**gen_args)
    elif config.ablation_mode == 'mcam_caff':
        gen = Generator_MCAM_CAFF(**gen_args)
    elif config.ablation_mode == 'full':
        gen = CDAR_Net_Generator(**gen_args)
    else:
        raise ValueError(f"Unknown ablation_mode: {config.ablation_mode}")

    gen = gen.to(device)

    if not config.gen_init or not os.path.exists(config.gen_init):
        raise FileNotFoundError(f"Generator weights not found at path: {config.gen_init}")

    param = torch.load(config.gen_init, map_location=device)
    gen.load_state_dict(param)
    print(f'Loaded pretrained generator from: {config.gen_init}')

    test_cdar_net(config, validation_data_loader, gen, device)'''

# test.py

import os
import yaml
from attrdict import AttrMap
from torch.utils.data import DataLoader
import torch

# 从模型文件中导入所有可能的生成器
from model.CDAR_Net.CDAR_Net import (
    Generator_Baseline, Generator_MCAM, Generator_CAFF, Generator_CCAR, Generator_CAFF_CCAR,
    Generator_MCAM_CAFF, Generator_MCAM_CCAR, CDAR_Net_Generator, CloudAwareFusion
)

# 评估脚本保持不变
from evaluation_CDAR import test_cdar_net
from dataload.init import getdata

if __name__ == '__main__':
    with open('config.yml', 'r', encoding='UTF-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttrMap(config)

    # 确保输出目录存在
    os.makedirs(config.out_dir, exist_ok=True)
    device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")

    # --- 数据加载 ---
    _, validation_dataset = getdata(config)
    validation_data_loader = DataLoader(
        dataset=validation_dataset,
        num_workers=config.threads,
        batch_size=config.validation_batchsize,
        shuffle=False
    )
    print(f"Validation dataset size: {len(validation_dataset)}")

    # --- 模型选择 ---
    ablation_models = {
        "baseline": Generator_Baseline,
        "baseline_mcam": Generator_MCAM,
        "baseline_caff": Generator_CAFF,
        "baseline_ccar": Generator_CCAR,
        "baseline_mcam_caff": Generator_MCAM_CAFF,
        "baseline_mcam_ccar": Generator_MCAM_CCAR,
        "baseline_caff_ccar": Generator_CAFF_CCAR,
        "full": CDAR_Net_Generator,
    }

    print(f"===> Loading Ablation Mode for Testing: {config.ablation_mode}")
    gen_class = ablation_models.get(config.ablation_mode)
    if not gen_class:
        raise ValueError(f"Unknown ablation_mode: {config.ablation_mode}")

    gen_args = {'height': config.height, 'width': config.width, 'num_dense_layer': config.num_dense_layer,
                'growth_rate': config.growth_rate, 'in_channels': config.in_ch}
    gen = gen_class(**gen_args)
    # 找到 gen 中的 caff 模块并设置阈值
    if hasattr(gen, 'caff') and isinstance(gen.caff, CloudAwareFusion):
        # 确保 config.caff_threshold 存在
        if not hasattr(config, 'caff_threshold'):
            print("Warning: caff_threshold not found in config.yml, using default 0.5 for CAFF.")
            gen.caff.threshold = 0.5
        else:
            gen.caff.threshold = config.caff_threshold
            print(f"Set CAFF threshold to: {config.caff_threshold}")
    gen = gen.to(device)

    # --- 加载权重 ---
    if not config.gen_init or not os.path.exists(config.gen_init):
        raise FileNotFoundError(
            f"Generator weights not found at path: {config.gen_init}. Please specify a trained model checkpoint in config.yml.")

    # 加载权重时需要注意，如果是在多GPU上训练的，模型参数的key会带有'module.'前缀
    state_dict = torch.load(config.gen_init, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        gen.load_state_dict(new_state_dict)
    else:
        gen.load_state_dict(state_dict)

    print(f'Loaded pretrained generator for evaluation from: {config.gen_init}')

    # --- 执行评估 ---
    # 假设 test_cdar_net 能够处理 gen(real_a) 的输出
    # evaluation_CDAR.py 中的 test_cdar_net 函数可能需要小幅调整
    # 确保它这样调用模型： fake_b, _, _, _, _ = gen(real_a)
    # 并且在测试时，即使是需要gt_mask的模型，也需要能在没有gt_mask的情况下运行（例如，内部使用一个全1的mask）
    # 在我们的新设计中，`gen`的forward都不需要`gt_mask`（CAFF-only情况在训练时传入，测试时可不传），所以调用是安全的
    test_cdar_net(config, validation_data_loader, gen, device)
