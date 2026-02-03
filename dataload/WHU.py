'''import numpy as np
import os
from torch.utils import data
from utils.imgproc import imresize
import skimage.io as io
import blobfile as bf

class TrainDataset(data.Dataset):

    def __init__(self, config, isTrain=True):
        super().__init__()
        self.config = config
        if(isTrain):
            self.datasets_dir = config.datasets_dir+'/train'
        else:
            self.datasets_dir = config.datasets_dir+'/test'

        self.imlistl = sorted(bf.listdir(self.datasets_dir+'/clear'))

    def __getitem__(self, index):
        t = io.imread(os.path.join(self.datasets_dir, 'clear', str(self.imlistl[index]))).astype(np.float32)[:,:,:3]
        x = io.imread(os.path.join(self.datasets_dir, 'cloud', str(self.imlistl[index]))).astype(np.float32)[:,:,:3]
    

        M = np.clip((t-x).sum(axis=2), 0, 1).astype(np.float32)

        x = x / 255
        t = t / 255
        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)
        filename = self.imlistl[index].split('.')[0]

        return x, t, M, filename

    def __len__(self):
        return len(self.imlistl)'''

import numpy as np
import os
from torch.utils import data
import skimage.io as io
import blobfile as bf

def _stem(name: str):
    return os.path.splitext(name)[0]

class TrainDataset(data.Dataset):
    """
    目录结构：
      train/clear/xxx.png
      train/cloud/xxx.png
      train/cloud_image/ (mask files)
    test 同理

    返回：
      x: (C,H,W) float32 [0,1]
      t: (C,H,W) float32 [0,1]
      M: (H,W) float32 {0,1}   (DataLoader 后会变成 B,H,W，你训练里会 unsqueeze(1) -> B,1,H,W)
      filename
    """
    def __init__(self, config, isTrain=True):
        super().__init__()
        self.config = config

        root = os.path.join(config.datasets_dir, "train" if isTrain else "test")
        self.clear_dir = os.path.join(root, "clear")
        self.cloud_dir = os.path.join(root, "cloud")
        self.mask_dir  = os.path.join(root, "cloud_mask")

        self.imlist = sorted(bf.listdir(self.clear_dir))

        # mask 是否需要反转（云=0 无云=1 的话设 True）
        self.mask_invert = getattr(config, "mask_invert", False)

    def _find_mask(self, base_name: str) -> str:
        """
        在 cloud_image/ 中为 clear/cloud 的文件名 base_name 找到对应 mask。
        支持：
          xxx.png -> xxx.png / xxx_mask.png / xxxmask.png / xxx_M.png / xxx_m.png / xxx_cloudmask.png / xxx_cloud_mask.png ...
        也支持扩展名不同（png/jpg/tif...）
        """
        s = _stem(base_name)

        exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
        suffixes = ["", "_mask", "mask", "_M", "_m", "_cloudmask", "cloud_mask", "_cm", "_seg", "_label"]

        # 先按常见规则拼路径
        candidates = []
        for suf in suffixes:
            for ext in exts:
                candidates.append(os.path.join(self.mask_dir, s + suf + ext))

        # 再兜底：目录里扫一遍“包含 stem 的文件”（适合你 mask 命名很怪但仍包含原名）
        # （只做一次列目录缓存会更快；这里先简化直接扫）
        try:
            all_masks = bf.listdir(self.mask_dir)
            for fn in all_masks:
                low = fn.lower()
                if s.lower() in low and any(low.endswith(e) for e in exts):
                    candidates.append(os.path.join(self.mask_dir, fn))
        except Exception:
            pass

        # 去重 + 存在性检查
        seen = set()
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            if os.path.exists(p):
                return p

        raise FileNotFoundError(
            f"Mask not found for base '{base_name}'. "
            f"Expected in: {self.mask_dir} (e.g., {s}_mask.png / {s}_M.png / same-name)."
        )

    def _read_mask(self, mask_path: str, target_hw):
        m = io.imread(mask_path).astype(np.float32)
        if m.ndim == 3:
            m = m[:, :, 0]  # 取一个通道即可

        h, w = target_hw
        if m.shape[0] != h or m.shape[1] != w:
            # 兜底：裁剪到一致（更推荐你提前把三者做成同尺寸）
            m = m[:h, :w]

        if m.max() > 1.5:
            m = m / 255.0

        m = (m > 0.5).astype(np.float32)
        if self.mask_invert:
            m = 1.0 - m
        return m  # (H,W)

    def __getitem__(self, index):
        name = str(self.imlist[index])

        clear_path = os.path.join(self.clear_dir, name)
        cloud_path = os.path.join(self.cloud_dir, name)

        if not os.path.exists(clear_path):
            raise FileNotFoundError(f"Clear not found: {clear_path}")
        if not os.path.exists(cloud_path):
            raise FileNotFoundError(f"Cloud not found: {cloud_path}")

        mask_path = self._find_mask(name)

        t = io.imread(clear_path).astype(np.float32)[:, :, :3]
        x = io.imread(cloud_path).astype(np.float32)[:, :, :3]

        x = x / 255.0
        t = t / 255.0

        x = x.transpose(2, 0, 1)
        t = t.transpose(2, 0, 1)

        _, h, w = x.shape
        M = self._read_mask(mask_path, target_hw=(h, w))

        filename = name.split('.')[0]
        return x, t, M, filename

    def __len__(self):
        return len(self.imlist)
