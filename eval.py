import numpy as np
from skimage.metrics import structural_similarity as SSIM
from torch.autograd import Variable
import lpips
from utils.utils import *
import torch
import time
from skimage.metrics import peak_signal_noise_ratio as PSNR
import torch.nn.functional as F
import math
from utils.utils import calculate_ssim  # 导入正确的函数
from tqdm import tqdm

loss_fn = lpips.LPIPS(net='alex', version=0.1)

def caculate_lpips(img0,img1):
    im1=np.copy(img0.cpu().numpy())
    im2=np.copy(img1.cpu().numpy())
    im1=torch.from_numpy(im1.astype(np.float32))
    im2 = torch.from_numpy(im2.astype(np.float32))
    im1.unsqueeze_(0)
    im2.unsqueeze_(0)
    current_lpips_distance  = loss_fn.forward(im1, im2)
    return current_lpips_distance 

def caculate_ssim(imgA, imgB):
    imgA1 = np.tensordot(imgA.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    imgB1 = np.tensordot(imgB.cpu().numpy().transpose(1, 2, 0), [0.298912, 0.586611, 0.114478], axes=1)
    score = SSIM(imgA1, imgB1, data_range=255)
    return score

def caculate_psnr( imgA, imgB):
    imgA1 = imgA.cpu().numpy().transpose(1, 2, 0)
    imgB1 = imgB.cpu().numpy().transpose(1, 2, 0)
    psnr = PSNR(imgA1, imgB1, data_range=255)
    return psnr

def get_image_arr(dataset):  #the id of the image you want to save 
    if(dataset=='RICE1'):
        return ['0','105','143','368','425','458','495']
    elif(dataset=='RICE2'):
        return ['49','17','185','209','309','619','408','630']
    elif(dataset=='T-Cloud'):
        return ['278','142','162','449','930','1261','1652']
    elif(dataset=='My14' or dataset=='My24'):
        return ['4','5','7','8','36','37','65']
    else:
        return []


def test3(config, test_data_loader, gen, criterionMSE, epoch):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    for i, batch in enumerate (test_data_loader):
        x, t, filename = Variable (batch[0]), Variable (batch[1]), batch[3]
        if config.cuda:
            x = x.cuda (0)
            t = t.cuda (0)

        # 检查模型输出并获取正确的部分
        out = gen (x)  # 如果 gen 返回的是一个元组（如 (att, out)），那么 out 就是第二个元素
        out = out[0]  # 如果返回的是 (att, out)，你只需要 out 部分

        # 保存图片等操作
        if epoch % config.snapshot_interval == 0 and epoch > 20 and filename[0] in get_image_arr (config.datasets_dir):
            if x.shape[1] == 3:
                save_image (config.out_dir, x, i, epoch, filename=filename[0] + 'LR')
                save_image (config.out_dir, t, i, epoch, filename=filename[0] + 'HR')
                save_image (config.out_dir, out, i, epoch, filename=filename[0] + 'SR')
            else:  # 处理四波段图像
                save_imagenir (config.out_dir, x, i, epoch, filename=filename[0] + 'LR')
                save_imagenir (config.out_dir, t, i, epoch, filename=filename[0] + 'HR')
                save_imagenir (config.out_dir, out, i, epoch, filename=filename[0] + 'SR')

        # 这里要确保 out 是一个 tensor，然后才可以调用 clamp 和其他操作
        # 打印 out[0] 的形状，确认它的维度
        print (out[0].shape)

        # 如果 out[0] 只有两个维度 [height, width]，可以将其转换为 [1, height, width]（假设只有一个通道）
        if len (out[0].shape) == 2:
            out_reshaped = out[0].unsqueeze (0)  # 增加一个通道维度
        else:
            out_reshaped = out[0]  # 如果已经是三维的，则直接使用

        imgA = (out[0] * 255).clamp (0, 255).to (torch.uint8)[:3, :, :]
        imgB = (t[0] * 255).clamp (0, 255).to (torch.uint8)[:3, :, :]

        # 计算 PSNR, SSIM, LPIPS 等
        psnr = caculate_psnr (imgA, imgB)
        c, w, h = imgA.shape
        if imgA.shape[0] == 4:
            lpips = 0
            ssim = 0
            for i in range (imgA.shape[0]):
                imA = imgA[i]
                imA = imA.expand (3, w, h)
                imB = imgB[i]
                imB = imB.expand (3, w, h)
                ssim1 = caculate_ssim (imA, imB)
                lpips1 = caculate_lpips (imA, imB)
                ssim += ssim1
                lpips += lpips1
            ssim = ssim / imgA.shape[0]
            lpips = lpips / imgA.shape[0]
        else:
            ssim = caculate_ssim (imgA, imgB)
            lpips = caculate_lpips (imgA, imgB)

        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips

    avg_psnr = avg_psnr / len (test_data_loader)
    avg_ssim = avg_ssim / len (test_data_loader)
    avg_lpips = avg_lpips / len (test_data_loader)

    print ("===> Avg. PSNR: {:.3f} dB".format (avg_psnr))
    print ("===> Avg. SSIM: {:.4f} dB".format (avg_ssim))
    print ("===> Avg. Lpips: {:.4f} dB".format (avg_lpips.item ( )))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test


'''def test(config, test_data_loader, gen, criterionMSE, epoch):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    for i, batch in enumerate(test_data_loader):
        x, t, filename = Variable(batch[0]), Variable(batch[1]),batch[3]
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)

        #out = gen (x)
        att, out = gen(x)
        if epoch % config.snapshot_interval == 0 and epoch > 20 and filename[0] in get_image_arr(config.datasets_dir):
            if(x.shape[1]==3):
                save_image(config.out_dir, x, i, epoch, filename=filename[0]+'LR')
                save_image(config.out_dir, t, i, epoch, filename=filename[0]+'HR')
                save_image(config.out_dir, out, i, epoch, filename=filename[0]+'SR')
            else:  #it handle the multispectral nir layer (the situation that image contain 4 band RGB and nir)
                save_imagenir(config.out_dir, x, i, epoch, filename=filename[0]+'LR')
                save_imagenir(config.out_dir, t, i, epoch, filename=filename[0]+'HR')
                save_imagenir(config.out_dir, out, i, epoch, filename=filename[0]+'SR')

        imgA = (out[0]*255).clamp(0, 255).to(torch.uint8)[:3,:,:]
        imgB = (t[0]*255).clamp(0, 255).to(torch.uint8)[:3,:,:]
        psnr = caculate_psnr(imgA, imgB)
        c,w,h=imgA.shape
        if(imgA.shape[0]==4):
            lpips=0
            ssim=0
            for i in range(imgA.shape[0]):
                imA = imgA[i]
                imA = imA.expand(3,w,h)
                imB = imgB[i]
                imB = imB.expand(3,w,h)
                ssim1 = caculate_ssim(imA, imB)
                lpips1 = caculate_lpips(imA, imB)
                ssim+=ssim1
                lpips+=lpips1
            ssim=ssim/imgA.shape[0]
            lpips=lpips/imgA.shape[0]
        else:
            ssim = caculate_ssim(imgA, imgB)
            lpips = caculate_lpips(imgA, imgB)
        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_lpips = avg_lpips / len(test_data_loader)

    print("===> Avg. PSNR: {:.3f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    print("===> Avg. Lpips: {:.4f} dB".format(avg_lpips.item()))
    
    log_test = {}
    log_test['epoch'] = epoch
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim

    return log_test'''


def test(config, test_data_loader, gen, epoch):
    # 设置模型为评估模式
    gen.eval()

    # 获取设备
    device = next(gen.parameters()).device

    # 初始化指标
    avg_psnr = 0
    avg_ssim = 0
    avg_l1 = 0
    count = 0

    # 禁用梯度计算
    with torch.no_grad():
        # 使用进度条
        pbar = tqdm(test_data_loader, desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 获取输入数据
            if len(batch) >= 3:
                # 数据集包含云掩膜
                real_a = batch[0].to(device)
                real_b = batch[1].to(device)
                cloud_mask = batch[2].to(device)
            else:
                # 数据集不包含云掩膜 - 创建全零掩膜
                real_a = batch[0].to(device)
                real_b = batch[1].to(device)
                cloud_mask = torch.zeros_like(real_a[:, :1, :, :]).to(device)

            # 检查图像尺寸
            if real_a.size(2) < 8 or real_a.size(3) < 8:
                print(f"Warning: Skipping small image (size: {real_a.size(2)}x{real_a.size(3)}) at batch {batch_idx}")
                continue

            # 生成去云图像
            #fake_b = gen(real_a, cloud_mask)

            # 确保输出在0-1范围内
            #fake_b = torch.clamp(fake_b, 0, 1)
            out = gen(real_a, cloud_mask)  # 如果你的 gen 需要 mask，就传 M_gt；否则用 gen(real_a)
            fake_b = out[0] if isinstance(out, (tuple, list)) else out
            fake_b = torch.clamp(fake_b, 0, 1)
            # 计算L1损失
            l1_loss = F.l1_loss(fake_b, real_b).item()
            avg_l1 += l1_loss * real_a.size(0)

            # 计算PSNR
            mse = F.mse_loss(fake_b, real_b).item()
            if mse < 1e-10:
                psnr = 100  # 完美匹配
            else:
                psnr = 10 * math.log10(1 / mse)
            avg_psnr += psnr * real_a.size(0)

            # 计算SSIM（如果需要）
            if config.calc_ssim:
                # 确保图像尺寸足够大
                if fake_b.size(2) < 8 or fake_b.size(3) < 8:
                    ssim_val = 0.0  # 小图像无法计算SSIM
                else:
                    try:
                        ssim_val = calculate_ssim(fake_b, real_b, data_range=1.0)
                    except Exception as e:
                        print(f"Error calculating SSIM: {e}")
                        ssim_val = 0.0
                avg_ssim += ssim_val * real_a.size(0)

            count += real_a.size(0)

            # 更新进度条
            pbar.set_postfix({
                'PSNR': f'{psnr:.2f} dB',
                'L1': f'{l1_loss:.4f}',
                'SSIM': f'{ssim_val:.4f}' if config.calc_ssim else 'N/A'
            })

    if count == 0:
        print("Warning: No valid images processed during validation")
        return {'psnr': 0, 'l1_loss': 0, 'ssim': 0}

        # 计算平均指标
    avg_l1 /= count
    avg_psnr /= count
    result = {'psnr': avg_psnr, 'l1_loss': avg_l1}

    if config.calc_ssim:
        avg_ssim /= count
        result['ssim'] = avg_ssim

    print(f"\nValidation Results - PSNR: {avg_psnr:.2f} dB, L1 Loss: {avg_l1:.4f}")
    if config.calc_ssim:
        print(f"SSIM: {avg_ssim:.4f}")

    return result

'''
def test2(config, test_data_loader, gen, criterionMSE, epoch):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    for i, batch in enumerate(test_data_loader):
        x, t, filename = Variable(batch[0]), Variable(batch[1]), batch[3]
        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)
            cloud_mask = cloud_mask.cuda(0)  # 确保云掩码也在正确的设备上

        att, out = gen(x)
        #out = gen (x)
        

        # 如果 out 是一个元组 (例如 (att, out))，只取 out 部分
        if isinstance (out, tuple):
            out = out[0]  # 取第一个元素（out）
            
        if epoch % config.snapshot_interval == 0 and epoch >= 200:
           save_image(config.out_dir, out, i, epoch, filename=filename[0])
            #   save_10tif(config.out_dir, out, i, epoch, filename=filename[0])


        imgA = (out[0] * 255).clamp(0, 255).to(torch.uint8)[:3, :, :]
        imgB = (t[0] * 255).clamp(0, 255).to(torch.uint8)[:3, :, :]
        psnr = caculate_psnr(imgA, imgB)
        c, w, h = imgA.shape
        if (imgA.shape[0] == 4):
            lpips = 0
            ssim = 0
            for i in range(imgA.shape[0]):
                imA = imgA[i]
                imA = imA.expand(3, w, h)
                imB = imgB[i]
                imB = imB.expand(3, w, h)
                ssim1 = caculate_ssim(imA, imB)
                lpips1 = caculate_lpips(imA, imB)
                ssim += ssim1
                lpips += lpips1
            ssim = ssim / imgA.shape[0]
            lpips = lpips / imgA.shape[0]
        else:
            ssim = caculate_ssim(imgA, imgB)
            lpips = caculate_lpips(imgA, imgB)
        avg_psnr += psnr
        avg_ssim += ssim
        avg_lpips += lpips
    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_lpips = avg_lpips / len(test_data_loader)

    print("===> Avg. PSNR: {:.3f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f} dB".format(avg_ssim))
    print("===> Avg. Lpips: {:.4f} dB".format(avg_lpips.item()))

    log_test = {}
    log_test['epoch'] = epoch
    log_test['psnr'] = avg_psnr
    log_test['ssim'] = avg_ssim
    log_test['LPIPS'] = avg_lpips
    return log_test'''


def test2(config, test_data_loader, gen, criterionMSE, epoch):
    avg_psnr = 0
    avg_ssim = 0
    avg_lpips = 0
    for i, batch in enumerate(test_data_loader):
        # 确保获取云掩码 (cloud_mask) - 假设它在 batch[2] 中
        if len(batch) >= 4:  # 有云掩码
            x, t, cloud_mask, filename = batch[0], batch[1], batch[2], batch[3]
        else:  # 没有云掩码
            x, t, filename = batch[0], batch[1], batch[2]
            cloud_mask = torch.ones_like(x[:, :1, :, :])  # 创建默认云掩码

        if config.cuda:
            x = x.cuda(0)
            t = t.cuda(0)
            cloud_mask = cloud_mask.cuda(0)  # 确保云掩码也在正确的设备上

        # 调用生成器时传入两个参数：x 和 cloud_mask
        with torch.no_grad():
            outputs = gen(x, cloud_mask)

        # 处理不同类型的输出
        if isinstance(outputs, tuple):
            # 如果输出是元组，取第二个元素作为输出图像
            out = outputs[1] if len(outputs) > 1 else outputs[0]
        else:
            out = outputs

        if epoch % config.snapshot_interval == 0 and epoch >= 200:
            save_image(config.out_dir, out, i, epoch, filename=filename[0])
            #   save_10tif(config.out_dir, out, i, epoch, filename=filename[0])

        # 确保图像在 [0, 255] 范围内
        imgA = (out[0] * 255).clamp(0, 255).to(torch.uint8)[:3, :, :]
        imgB = (t[0] * 255).clamp(0, 255).to(torch.uint8)[:3, :, :]

        psnr = caculate_psnr(imgA, imgB)
        c, w, h = imgA.shape

        # 处理多通道图像
        if imgA.shape[0] == 4:
            lpips_val = 0
            ssim_val = 0
            # 使用不同的索引变量名避免冲突
            for ch_idx in range(3):  # 只处理前3个通道 (RGB)
                imA = imgA[ch_idx:ch_idx + 1]  # 取单个通道
                imA = imA.expand(3, w, h)  # 扩展为三通道

                imB = imgB[ch_idx:ch_idx + 1]
                imB = imB.expand(3, w, h)

                ssim_val += caculate_ssim(imA, imB)
                lpips_val += caculate_lpips(imA, imB)

            ssim_val /= 3
            lpips_val /= 3
        else:
            ssim_val = caculate_ssim(imgA, imgB)
            lpips_val = caculate_lpips(imgA, imgB)

        # 确保将张量转换为 Python 数值
        if isinstance(lpips_val, torch.Tensor):
            lpips_val = lpips_val.item()

        avg_psnr += psnr
        avg_ssim += ssim_val
        avg_lpips += lpips_val

    avg_psnr = avg_psnr / len(test_data_loader)
    avg_ssim = avg_ssim / len(test_data_loader)
    avg_lpips = avg_lpips / len(test_data_loader)

    # 确保所有平均值都是 Python 数值
    if isinstance(avg_psnr, torch.Tensor):
        avg_psnr = avg_psnr.item()
    if isinstance(avg_ssim, torch.Tensor):
        avg_ssim = avg_ssim.item()
    if isinstance(avg_lpips, torch.Tensor):
        avg_lpips = avg_lpips.item()

    print("===> Avg. PSNR: {:.3f} dB".format(avg_psnr))
    print("===> Avg. SSIM: {:.4f}".format(avg_ssim))
    print("===> Avg. LPIPS: {:.4f}".format(avg_lpips))

    log_test = {
        'epoch': epoch,
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'LPIPS': avg_lpips
    }
    return log_test