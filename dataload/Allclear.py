import os
import json
import numpy as np
import torch
from torch.utils import data
import rasterio as rio
import rasterio.windows


def center_crop_read(path, band_ids, out_size=256):
    # band_ids: list like [4,3,2] (1-based for rasterio)
    with rio.open(path) as src:
        w, h = src.width, src.height
        x0 = w // 2 - out_size // 2
        y0 = h // 2 - out_size // 2
        win = rasterio.windows.Window(x0, y0, out_size, out_size)
        arr = src.read(band_ids, window=win).astype(np.float32)  # (C,H,W)
    return arr


def s2_to_rgb_tensor(s2_path, out_size=256):
    # Sentinel-2 常用 RGB = B4,B3,B2
    arr = center_crop_read(s2_path, [4, 3, 2], out_size=out_size)  # (3,H,W)
    # AllClear/S2 通常是 0~10000 缩放反射率
    arr = np.clip(arr, 0, 10000) / 10000.0
    arr = np.nan_to_num(arr, nan=0.0)
    return torch.from_numpy(arr)  # float32, (3,H,W)


def load_cloud_mask_from_cldshdw(cld_path, out_size=256):
    # 这里取 “二值云 mask” 通道（先假设 band2 是二值云）
    # 如果不确定，就先返回全零占位也能跑通训练流程
    try:
        m = center_crop_read(cld_path, [2], out_size=out_size)  # (1,H,W)
        m = np.nan_to_num(m, nan=0.0)
        m = (m > 0.5).astype(np.float32)  # (1,H,W)
        return torch.from_numpy(m)
    except Exception:
        return torch.zeros(1, out_size, out_size, dtype=torch.float32)


class TrainDataset(data.Dataset):
    """
    返回: x(cloudy RGB), t(clear RGB), M(mask), filename
    """

    def __init__(self, config, json_path=None, isTrain=True, choose_input="first", out_size=256):
        self.out_size = out_size
        self.choose_input = choose_input
        self.dst_root = getattr(config, "datasets_dir", None)
        self.src_root = getattr(config, "allclear_src_root", None)

        # 1) 优先用外部传入 json_path（你在 dataload/init.py 里手动传也行）
        if json_path is None:
            # 2) 其次根据 isTrain 自动选择你 config.yml 里的字段
            if isTrain:
                json_path = getattr(config, "allclear_train_json", None)
            else:
                json_path = getattr(config, "allclear_test_json", None)

        # 3) 兜底：兼容旧字段（可留可不留）
        if json_path is None:
            json_path = getattr(config, "allclear_json", None) or getattr(config, "json_path", None)

        if json_path is None:
            raise ValueError(
                "缺少 AllClear json 路径：请在 config.yml 里设置 allclear_train_json / allclear_test_json "
                "（或传入 json_path）"
            )

        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        if isinstance(self.samples, dict):
            self.keys = list(self.samples.keys())
        else:
            self.keys = None

        # 可选：快速自检一次根目录是否存在
        # print("dst_root:", self.dst_root, "exists:", os.path.exists(self.dst_root))
        # print("src_root:", self.src_root)

    def __len__(self):
        return len(self.keys) if self.keys is not None else len(self.samples)

    def _get_sample(self, idx):
        if self.keys is None:
            return self.samples[idx]
        return self.samples[self.keys[idx]]

    def _remap_path(self, p: str) -> str:
        """
        将 json 中的 Linux 路径(/scratch/...) 映射到本机 datasets_dir(E:\...)
        也兼容 json 里是相对路径的情况。
        """
        if p is None:
            return p

        # 统一成正斜杠方便判断前缀
        p_norm = str(p).replace("\\", "/")

        # 1) Linux 绝对路径：用 src_root 前缀替换为 dst_root
        if self.src_root:
            src_norm = str(self.src_root).replace("\\", "/").rstrip("/")
            if p_norm.startswith(src_norm):
                rel = p_norm[len(src_norm):].lstrip("/")  # 相对路径部分
                return os.path.join(self.dst_root, rel)

        # 2) 相对路径：直接拼到 dst_root
        is_windows_abs = (len(p_norm) > 1 and p_norm[1] == ":")
        is_linux_abs = p_norm.startswith("/")
        if (not is_windows_abs) and (not is_linux_abs):
            return os.path.join(self.dst_root, p_norm)

        # 3) 其它情况：原样返回
        return str(p)

    def __getitem__(self, idx):
        s = self._get_sample(idx)

        # 输入序列：s["s2_toa"] = [[timestamp, path], ...]
        s2_list = s["s2_toa"]
        if self.choose_input == "last":
            in_time, in_path = s2_list[-1][0], s2_list[-1][1]
        else:
            in_time, in_path = s2_list[0][0], s2_list[0][1]

        # 目标清晰图（训练一般需要这个字段）
        tgt_path = s["target"][0][1]

        # ---- 路径 remap：把 /scratch/... 映射到 E:\allclear_dataset\... ----
        in_path = self._remap_path(in_path)
        tgt_path = self._remap_path(tgt_path)

        # 云掩膜路径：先按字段替换，再 remap
        cld_path = in_path.replace("s2_toa", "cld_shdw")
        cld_path = self._remap_path(cld_path)

        # 可选：首次样本打印检查（建议你先 threads=0 跑通再关）
        # if idx == 0:
        #     print("in_path:", in_path, "exists:", os.path.exists(in_path))
        #     print("tgt_path:", tgt_path, "exists:", os.path.exists(tgt_path))
        #     print("cld_path:", cld_path, "exists:", os.path.exists(cld_path))

        x = s2_to_rgb_tensor(in_path, out_size=self.out_size)    # (3,H,W)
        t = s2_to_rgb_tensor(tgt_path, out_size=self.out_size)   # (3,H,W)
        M = load_cloud_mask_from_cldshdw(cld_path, out_size=self.out_size)  # (1,H,W)

        filename = f"{s['roi'][0]}_{in_time}".replace(" ", "_").replace(":", "")
        return x, t, M, filename