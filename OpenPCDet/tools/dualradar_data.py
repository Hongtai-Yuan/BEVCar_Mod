#!/usr/bin/env python3
# inspect_data_verbose.py
# ------------------------------------------------------------
# 功能：
#   * 不训练模型，只跑一遍 OpenPCDet 的数据 pipeline
#   * 详细打印首个 batch 的所有字段（形状 / dtype / 范围等）
#   * 如 batch 中包含相机图像，将：
#       - 打印第 0 张图像左上 16×16 像素的原始数值
#       - 将整个 images 张量保存为 .npy 方便离线查看
#       - 将前 4 张图像（或不足 4 张全部）恢复到 PNG 文件，便于人工目视检查
# ------------------------------------------------------------

import _init_path  # noqa: F401,  用于正确导入 OpenPCDet 模块 (修改 sys.path)
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

# ------------------------------------------------------------------
# 配置文件路径（请根据自己情况修改）
CFG_FILE = Path('/root/autodl-tmp/BEVCar/OpenPCDet/tools/cfgs/dual_radar_models/pointpillar_arbe.yaml')

# 批大小 & Dataloader 线程数
BATCH_SIZE = 4
NUM_WORKERS = 4
# ------------------------------------------------------------------


# -------------------------  打印辅助函数  ---------------------------

def print_batch_info(batch):
    """逐字段打印 batch 内容的摘要信息。"""
    print("\n>>> Batch 0\n")
    for k, v in batch.items():
        # ---- torch.Tensor ----
        if isinstance(v, torch.Tensor):
            shape = tuple(v.shape)
            dtype = v.dtype
            device = v.device
            try:
                _min = v.min().item() if v.numel() < 1_000_000 else "-"
                _max = v.max().item() if v.numel() < 1_000_000 else "-"
            except Exception:
                _min = _max = "-"
            print(f"{k:<20}: torch.Tensor {str(shape):<16} {str(dtype):<10} [min {_min}, max {_max}] (device {device})")

        # ---- numpy.ndarray ----
        elif isinstance(v, np.ndarray):
            shape = v.shape
            dtype = v.dtype
            try:
                _min = v.min() if v.size < 1_000_000 else "-"
                _max = v.max() if v.size < 1_000_000 else "-"
            except Exception:
                _min = _max = "-"
            print(f"{k:<20}: np.ndarray  {str(shape):<16} {str(dtype):<10} [min {_min}, max {_max}]")

        # ---- list / tuple ----
        elif isinstance(v, (list, tuple)):
            preview = v[:5] if len(v) <= 5 else v[:3] + ['...']
            print(f"{k:<20}: {type(v).__name__:<6} len={len(v):<5} example={preview}")

        # ---- other ----
        else:
            print(f"{k:<20}: {type(v).__name__:<20} {v}")


# -------------------------  图像处理函数  ---------------------------

def _get_image_tensor(batch):
    """尝试从 batch 中取出图像张量。

    * 约定常见字段名称： 'images' / 'images_left' 等。
    * 若均不存在，则返回 None。
    """
    for key in ("images", "images_left", "images_right"):
        if key in batch and isinstance(batch[key], torch.Tensor):
            return batch[key], key
    return None, None


def print_image_tensor(batch, crop_size: int = 16):
    """在终端打印第 0 张图像左上 crop_size×crop_size 的像素值。"""
    imgs, key = _get_image_tensor(batch)
    if imgs is None:
        # print("[Info] 当前 batch 中未找到图像字段 (images)。")
        return

    b, c, h, w = imgs.shape
    print(f"\n>>> Detected image tensor '{key}' with shape {imgs.shape}, dtype={imgs.dtype}\n")

    # 取第 0 张图像左上 crop_size×crop_size
    preview = imgs[0, :, :crop_size, :crop_size].cpu().numpy()
    np.set_printoptions(threshold=np.inf, linewidth=200, precision=3, suppress=True)
    print(f"First image (top-left {crop_size}×{crop_size} pixels):\n", preview)


def dump_images_npy(batch, out_dir: str = "img_dump"):
    """将 images 张量以 .npy 格式保存到磁盘。"""
    imgs, key = _get_image_tensor(batch)
    if imgs is None:
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / f"{key}_batch0.npy", imgs.cpu().numpy())
    print(f"[Saved] {key}_batch0.npy  ->  {out_path.resolve()}")


def save_images_png(batch, out_dir: str = "img_vis", max_save: int = 4):
    """将前 `max_save` 张图像恢复到 PNG 文件。"""
    imgs, key = _get_image_tensor(batch)
    if imgs is None:
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 如果张量是 float 且范围在 [0,1]，将其缩放到 [0,255]
    if imgs.dtype != torch.uint8:
        if imgs.max() <= 1.0:
            imgs = imgs * 255.0
        imgs = imgs.clamp(0, 255).to(torch.uint8)

    imgs = imgs.cpu()
    save_count = min(max_save, imgs.shape[0])
    for i in range(save_count):
        img = imgs[i]
        # 若为单通道，复制成 3 通道可视化
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        pil_img = Image.fromarray(img.permute(1, 2, 0).numpy())
        fname = out_path / f"{key}_batch0_img{i}.png"
        pil_img.save(fname)
        print(f"[Saved] {fname.relative_to(Path.cwd())}")


# -------------------------    主 程 序    ---------------------------

def main():
    # 加载配置
    if not CFG_FILE.exists():
        raise FileNotFoundError(f"找不到配置文件: {CFG_FILE.resolve()}")
    cfg_from_yaml_file(str(CFG_FILE), cfg)

    # 创建 dataloader
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=BATCH_SIZE,
        dist=False,
        workers=NUM_WORKERS,
        logger=None,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=1,
    )

    print(f"Dataset size : {len(dataset)} samples")
    print(f"Batch size    : {BATCH_SIZE}")
    print("---------------------------------------------")

    # 只遍历第一个 batch
    for b_idx, batch in enumerate(dataloader):
        print_batch_info(batch)
        print_image_tensor(batch, crop_size=16)
        dump_images_npy(batch, out_dir="img_dump")
        save_images_png(batch, out_dir="img_vis", max_save=4)
        break  # 只查看首批数据即可


if __name__ == "__main__":
    main()
