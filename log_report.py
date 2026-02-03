# import json
# import os
# import numpy as np
# import matplotlib as mpl
# mpl.use('Agg')
# from matplotlib import pyplot as plt
#
#
'''class LogReport():
    def __init__(self, log_dir, log_name='log'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def __call__(self, log):
        self.log_.append(log)
        with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump(self.log_, f, indent=4)

    def save_lossgraph(self):
        epoch = []
        gen_loss = []
        dis_loss = []

        for l in self.log_:
            epoch.append(l['epoch'])
            gen_loss.append(l['gen/loss'])
            dis_loss.append(l['dis/loss'])

        epoch = np.asarray(epoch)
        gen_loss = np.asarray(gen_loss)
        dis_loss = np.asarray(dis_loss)

        plt.plot(epoch, gen_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss_gen')
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_gen.pdf'))
        plt.close()

        plt.plot(epoch, dis_loss)
        plt.xlabel('epoch')
        plt.ylabel('loss_dis')
        plt.savefig(os.path.join(self.log_dir, 'lossgraph_dis.pdf'))
        plt.close()
#
#
# class TestReport():
#     def __init__(self, log_dir, log_name='log_test'):
#         self.log_dir = log_dir
#         self.log_name = log_name
#         self.log_ = []
#
#     def __call__(self, log):
#         self.log_.append(log)
#         with open(os.path.join(self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
#             json.dump(self.log_, f, indent=4)
#
#     def save_lossgraph(self):
#         epoch = []
#         ssim = []
#         psnr = []
#
#         for l in self.log_:
#             epoch.append(l['epoch'])
#             ssim.append(l['ssim'])
#             psnr.append(l['psnr'])
#
#         epoch = np.asarray(epoch)
#         ssim = np.asarray(ssim)
#         psnr = np.asarray(psnr)
#
#         plt.plot(epoch, ssim)
#         plt.xlabel('epoch')
#         plt.ylabel('ssim')
#         plt.savefig(os.path.join(self.log_dir, 'graph_ssim.pdf'))
#         plt.close()
#
#         plt.plot(epoch, psnr)
#         plt.xlabel('epoch')
#         plt.ylabel('psnr')
#         plt.savefig(os.path.join(self.log_dir, 'graph_psnr.pdf'))
#         plt.close()'''

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class LogReport:
    def __init__(self, log_dir, log_name='log.json'):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_name)
        self.log_data = []
        os.makedirs(log_dir, exist_ok=True)

    def __call__(self, log):
        self.log_data.append(log)
        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def save_lossgraph(self):
        if not self.log_data:
            print("No log data available to plot")
            return

        # 提取数据
        epochs = []
        loss_data = defaultdict(list)

        for entry in self.log_data:
            epochs.append(entry['epoch'])
            for key, value in entry.items():
                if key.startswith('loss/'):
                    loss_data[key].append(value)

        # 检查数据有效性
        if not epochs:
            print("No epoch data available")
            return

        # 创建图表
        plt.figure(figsize=(15, 10))
        plots_created = False  # 标记是否成功创建了任何子图

        # 总损失（添加检查）
        if 'loss/gen_total' in loss_data and 'loss/dis_total' in loss_data:
            plt.subplot(2, 3, 1)
            plt.plot(epochs, loss_data['loss/gen_total'], 'b-', label='Generator')
            plt.plot(epochs, loss_data['loss/dis_total'], 'r-', label='Discriminator')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Total Loss')
            plt.legend()
            plt.grid(True)
            plots_created = True

        # 任务损失（添加检查）
        if 'loss/task' in loss_data and len(loss_data['loss/task']) == len(epochs):
            plt.subplot(2, 3, 2)
            plt.plot(epochs, loss_data['loss/task'], 'g-')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Task Loss')
            plt.grid(True)
            plots_created = True

        # 域损失（添加检查）
        if 'loss/domain' in loss_data and len(loss_data['loss/domain']) == len(epochs):
            plt.subplot(2, 3, 3)
            plt.plot(epochs, loss_data['loss/domain'], 'm-')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Domain Loss')
            plt.grid(True)
            plots_created = True

        # 云感知损失（添加检查）
        if 'loss/cloud' in loss_data and len(loss_data['loss/cloud']) == len(epochs):
            plt.subplot(2, 3, 4)
            plt.plot(epochs, loss_data['loss/cloud'], 'c-')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Cloud Loss')
            plt.grid(True)
            plots_created = True

        # 云感知子损失（添加检查）
        subloss_plots = 0
        if 'loss/spectral' in loss_data or 'loss/consistency' in loss_data or 'loss/structural' in loss_data:
            plt.subplot(2, 3, 5)
            if 'loss/spectral' in loss_data and len(loss_data['loss/spectral']) == len(epochs):
                plt.plot(epochs, loss_data['loss/spectral'], 'b-', label='Spectral')
                subloss_plots += 1
            if 'loss/consistency' in loss_data and len(loss_data['loss/consistency']) == len(epochs):
                plt.plot(epochs, loss_data['loss/consistency'], 'g-', label='Consistency')
                subloss_plots += 1
            if 'loss/structural' in loss_data and len(loss_data['loss/structural']) == len(epochs):
                plt.plot(epochs, loss_data['loss/structural'], 'r-', label='Structural')
                subloss_plots += 1

            if subloss_plots > 0:
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Cloud Sub-Losses')
                plt.legend()
                plt.grid(True)
                plots_created = True

        # 只有成功创建了图表才保存
        if plots_created:
            plt.tight_layout()
            save_path = os.path.join(self.log_dir, 'loss_graphs.png')
            plt.savefig(save_path)
            plt.close()
            print(f'Saved loss graphs to {save_path}')
        else:
            print("No valid loss data available to plot")
            plt.close()


class TestReport:
    def __init__(self, log_dir, log_name='log_test.json'):
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_name)
        self.log_data = []
        os.makedirs(log_dir, exist_ok=True)

    def __call__(self, log):
        self.log_data.append(log)
        with open(self.log_path, 'w') as f:
            json.dump(self.log_data, f, indent=4)

    def save_lossgraph(self):
        if not self.log_data:
            return

        # 提取数据
        epochs = []
        losses = []
        ssims = []
        psnrs = []

        for entry in self.log_data:
            epochs.append(entry['epoch'])
            losses.append(entry.get('loss', 0))
            ssims.append(entry.get('ssim', 0))
            psnrs.append(entry.get('psnr', 0))

        # 创建图表
        plt.figure(figsize=(15, 5))

        # 损失
        plt.subplot(1, 3, 1)
        plt.plot(epochs, losses, 'b-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.grid(True)

        # SSIM
        plt.subplot(1, 3, 2)
        plt.plot(epochs, ssims, 'g-')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('Validation SSIM')
        plt.grid(True)

        # PSNR
        plt.subplot(1, 3, 3)
        plt.plot(epochs, psnrs, 'r-')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR (dB)')
        plt.title('Validation PSNR')
        plt.grid(True)

        # 保存图表
        plt.tight_layout()
        save_path = os.path.join(self.log_dir, 'validation_metrics.png')
        plt.savefig(save_path)
        plt.close()
        print(f'Saved validation graphs to {save_path}')


'''import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch'''


'''class TestReport ( ):
    def __init__(self, log_dir, log_name='log_test'):
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_ = []

    def _convert_tensor_to_serializable(self, data):
        """辅助方法：将Tensor转换为可序列化格式"""
        if isinstance (data, torch.Tensor):
            return data.tolist ( )  # 将Tensor转换为列表
        elif isinstance (data, dict):
            for key, value in data.items ( ):
                data[key] = self._convert_tensor_to_serializable (value)
        elif isinstance (data, list):
            return [self._convert_tensor_to_serializable (item) for item in data]
        return data

    def __call__(self, log):
        # 转换log中的Tensor
        log = self._convert_tensor_to_serializable (log)

        self.log_.append (log)
        with open (os.path.join (self.log_dir, self.log_name), 'w', encoding='UTF-8') as f:
            json.dump (self.log_, f, indent=4)

    def save_lossgraph(self):
        epoch = []
        ssim = []
        psnr = []
        for l in self.log_:
            # 添加键存在性检查
            if 'epoch' in l and 'ssim' in l and 'psnr' in l:
                epoch.append(l['epoch'])
                ssim.append(l['ssim'])
                psnr.append(l['psnr'])
            else:
                print(f"Warning: Missing keys in log entry: {l.keys()}")

        # 检查是否有足够的数据绘图
        if len(epoch) == 0:
            print("Warning: No valid validation data for graphs")
            return

        epoch = np.asarray(epoch)
        ssim = np.asarray(ssim)
        psnr = np.asarray(psnr)

        # 绘制SSIM图
        plt.figure()
        plt.plot(epoch, ssim)
        plt.xlabel('epoch')
        plt.ylabel('ssim')
        plt.savefig(os.path.join(self.log_dir, 'graph_ssim.pdf'))
        plt.close()

        # 绘制PSNR图
        plt.figure()
        plt.plot(epoch, psnr)
        plt.xlabel('epoch')
        plt.ylabel('psnr')
        plt.savefig(os.path.join(self.log_dir, 'graph_psnr.pdf'))
        plt.close()'''



'''
    def save_lossgraph(self): 
        epoch = []
        ssim = []
        psnr = []
        for l in self.log_:
            epoch.append (l['epoch'])
            ssim.append (l['ssim'])
            psnr.append (l['psnr'])

        epoch = np.asarray (epoch)
        ssim = np.asarray (ssim)
        psnr = np.asarray (psnr)

        plt.plot (epoch, ssim)
        plt.xlabel ('epoch')
        plt.ylabel ('ssim')
        plt.savefig (os.path.join (self.log_dir, 'graph_ssim.pdf'))
        plt.close ( )

        plt.plot (epoch, psnr)
        plt.xlabel ('epoch')
        plt.ylabel ('psnr')
        plt.savefig (os.path.join (self.log_dir, 'graph_psnr.pdf'))
        plt.close ( )
'''
