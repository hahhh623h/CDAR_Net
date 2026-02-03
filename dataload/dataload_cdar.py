import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def _stem(path: str):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

class CloudRemovalWithMaskDataset(Dataset):
    """
    读取：
      cloudy_dir/xxx.(png/jpg/tif...)
      ground_truth_dir/xxx.(同名)
      cloudy_dir/xxx_mask.(png/jpg...)  或 xxxmask / xxx_M / xxx_cloudmask 等（同目录）
    返回：
      real_a (cloudy), real_b (gt), M_gt (mask, 0/1, shape: HxW or 1xHxW)
    """
    def __init__(self, cloudy_dir, gt_dir, in_ch=3, mask_invert=False, image_size=None):
        super().__init__()
        self.cloudy_dir = cloudy_dir
        self.gt_dir = gt_dir
        self.in_ch = in_ch
        self.mask_invert = mask_invert
        self.image_size = image_size  # (H,W) or None

        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
        self.cloudy_files = sorted([
            os.path.join(cloudy_dir, f) for f in os.listdir(cloudy_dir)
            if os.path.splitext(f)[1].lower() in exts
            # 避免把 mask 当成 cloudy
            and ("mask" not in f.lower())
        ])

        if len(self.cloudy_files) == 0:
            raise RuntimeError(f"No cloudy images found in: {cloudy_dir}")

        # transforms
        self.to_tensor = T.ToTensor()

    def _find_gt(self, cloudy_path):
        s = _stem(cloudy_path)
        # gt 同名匹配
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            p = os.path.join(self.gt_dir, s + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"GT not found for {cloudy_path} in {self.gt_dir}")

    def _find_mask(self, cloudy_path):
        """
        mask 和 cloudy 同目录，常见命名兜底：
          xxx.png -> xxx_mask.png / xxxmask.png / xxx_M.png / xxx_cloudmask.png / xxx_cloud_mask.png
        """
        s = _stem(cloudy_path)
        d = os.path.dirname(cloudy_path)

        candidates = []
        suffixes = ["_mask", "mask", "_M", "_m", "_cloudmask", "cloud_mask", "_cm", "_seg", "_label"]
        exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]

        # 1) 同名（不加后缀）
        for ext in exts:
            candidates.append(os.path.join(d, s + ext))

        # 2) 加后缀
        for suf in suffixes:
            for ext in exts:
                candidates.append(os.path.join(d, s + suf + ext))

        # 3) 也支持：原文件名里带 cloudy / haze 等替换成 mask（可选）
        low = os.path.basename(cloudy_path).lower()
        repl_pairs = [("cloudy", "mask"), ("cloud", "mask")]
        for a, b in repl_pairs:
            if a in low:
                cand = os.path.join(d, os.path.basename(cloudy_path).lower().replace(a, b))
                candidates.append(cand)

        # 去重
        seen = set()
        uniq = []
        for c in candidates:
            if c not in seen:
                uniq.append(c)
                seen.add(c)

        for p in uniq:
            if os.path.exists(p) and ("mask" in os.path.basename(p).lower() or p != cloudy_path):
                return p

        raise FileNotFoundError(
            f"Mask not found for {cloudy_path}. "
            f"Please put mask in same folder and name like {s}_mask.png (or similar)."
        )

    def _load_rgb_or_multiband(self, path):
        # 你如果是 13 通道数据（S2），这里需要用 rasterio 读取 tif 多波段。
        # 现在先按 3 通道常规图片做（与你现有保存png一致）。
        img = Image.open(path).convert("RGB")
        if self.image_size is not None:
            img = img.resize((self.image_size[1], self.image_size[0]), resample=Image.BILINEAR)
        return self.to_tensor(img)  # [0,1], shape (3,H,W)

    def _load_mask(self, path, target_hw):
        m = Image.open(path).convert("L")
        if self.image_size is not None:
            m = m.resize((self.image_size[1], self.image_size[0]), resample=Image.NEAREST)
        else:
            # 统一对齐到目标图尺寸
            m = m.resize((target_hw[1], target_hw[0]), resample=Image.NEAREST)

        m = self.to_tensor(m)  # (1,H,W), [0,1]
        # 二值化：>0.5 视为云
        m = (m > 0.5).float()
        if self.mask_invert:
            m = 1.0 - m
        return m  # (1,H,W)

    def __len__(self):
        return len(self.cloudy_files)

    def __getitem__(self, idx):
        cloudy_path = self.cloudy_files[idx]
        gt_path = self._find_gt(cloudy_path)
        mask_path = self._find_mask(cloudy_path)

        real_a = self._load_rgb_or_multiband(cloudy_path)
        real_b = self._load_rgb_or_multiband(gt_path)
        _, h, w = real_a.shape
        M = self._load_mask(mask_path, target_hw=(h, w))
        return real_a, real_b, M