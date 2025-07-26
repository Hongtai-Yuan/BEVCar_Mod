#!/usr/bin/env python3
# compile_openpcdet_tuple.py  (修正版)
# ------------------------------------------------------------
# 把 OpenPCDet 数据集封装成 Lift-Splat-Shoot 风格 13-tuple DataLoader
# ------------------------------------------------------------
import _init_path                          # noqa
from pathlib import Path
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader as pcdet_build_dl
import torch.nn.functional as F

# ---------- 默认参数 ----------
CFG_FILE_DEFAULT = Path(
    '/root/autodl-tmp/BEVCar/OpenPCDet/tools/cfgs/dual_radar_models/pointpillar_arbe.yaml'
)
V_TARGET_DEFAULT = 12_000
# ------------------------------

# ---------- 工具函数 ----------
def _to_tensor(arr):
    return arr if isinstance(arr, torch.Tensor) else torch.from_numpy(arr)

def _get_imgs(sample, out_hw=(448, 896)):
    """
    返回 (1,1,3,448,896)、以及缩放系数 sx, sy，
    保证后续把内参 K 同步缩放。
    """
    if 'images' not in sample:
        return None, 1.0, 1.0
    img = _to_tensor(sample['images']).float()          # (H,W,3)
    H0, W0 = img.shape[0], img.shape[1]
    img = img.permute(2, 0, 1)                          # (3,H,W)
    img = F.interpolate(img.unsqueeze(0), out_hw,
                        mode='bilinear', align_corners=False).squeeze(0)
    sx, sy = out_hw[1] / W0, out_hw[0] / H0
    return img.unsqueeze(0).unsqueeze(0), sx, sy        # (1,1,3,H,W)

def _get_radar(sample, v_target):
    if 'points' not in sample:
        return None
    pts   = _to_tensor(sample['points']).float()        # (N,C)
    feats = pts.T                                       # (C,N)
    C, N  = feats.shape
    if N >= v_target:
        feats = feats[:, :v_target]
    else:
        pad = torch.zeros(C, v_target-N, dtype=feats.dtype)
        feats = torch.cat([feats, pad], 1)
    return feats.unsqueeze(0).unsqueeze(0)              # (1,1,C,V)

def _calib_to_rt_k(calib, sx=1.0, sy=1.0, to_homo=True):
    V2C = calib.V2C if hasattr(calib, 'V2C') else calib.tr_velo_to_cam
    R, t = V2C[:3, :3], V2C[:3, 3]
    R_inv, t_inv = R.T, -R.T @ t
    R_t = torch.from_numpy(R_inv).float().unsqueeze(0).unsqueeze(0)
    t_t = torch.from_numpy(t_inv).float().unsqueeze(0).unsqueeze(0)

    K = calib.P2[:3, :3].copy()
    K[0, 0] *= sx;  K[0, 2] *= sx      # fx, cx
    K[1, 1] *= sy;  K[1, 2] *= sy      # fy, cy
    if to_homo:
        K_h = torch.eye(4, dtype=torch.float32)
        K_h[:3, :3] = torch.from_numpy(K)
        K_t = K_h.unsqueeze(0).unsqueeze(0)
    else:
        K_t = torch.from_numpy(K).float().unsqueeze(0).unsqueeze(0)
    return R_t, t_t, K_t

def _split_boxes_labels(gt_boxes_np):
    t = torch.from_numpy(gt_boxes_np).float()           # (N,8)
    wlh   = t[:, [4, 3, 5]]
    zeros = torch.zeros((t.size(0), 2), dtype=t.dtype)
    boxes9 = torch.cat([t[:, :3], wlh, t[:, 6:7], zeros], 1)  # (N,9)
    labels = t[:, 7].long()
    return [boxes9], [labels]     # T = 1

def _dummy_bev():
    return (torch.zeros((1,1,7,200,200)),
            torch.zeros((1,1,3,200,200)),
            torch.zeros((1,1,1,200,200)))

# _voxel_triplet — 保持变长但不再 unsqueeze(0) 两次
def _voxel_triplet(sample):
    vox_np  = sample['voxels']          # (P,32,5)
    coor_np = sample['voxel_coords']    # (P,4)
    P = vox_np.shape[0]

    vox  = torch.from_numpy(vox_np).float()          # (P,32,5)
    coor = torch.from_numpy(coor_np[:,1:]).long()    # (P,3)
    num  = torch.tensor([P], dtype=torch.long)       # (1,)

    return vox, coor, num          # ← 不再多套 (1,1,…)


# ---------- Dataset ----------
class TupleWrapper(Dataset):
    def __init__(self, base_ds, v_target):
        self.base = base_ds
        self.v_target = v_target
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        s = self.base[idx]

        imgs, sx, sy = _get_imgs(s)                         # 0
        rots, trans, intrins = _calib_to_rt_k(s['calib'], sx, sy)  # 1,2,3
        radar  = _get_radar(s, self.v_target)               # 4
        vox_in, vox_coor, vox_num = _voxel_triplet(s)       # 8,9,10
        boxes_l, labels_l = _split_boxes_labels(s['gt_boxes'])     # 11,12
        map_mask, map_rgba, ego_bev = _dummy_bev()                 # 5,6,7

        return (imgs, rots, trans, intrins,
                radar, map_mask, map_rgba, ego_bev,
                vox_in, vox_coor, vox_num,
                boxes_l, labels_l)

# -------- collate_fn（动态 pad 8/9） --------
def tuple_collate(batch):
    transposed = list(zip(*batch))          # len == 13
    out = []
    for idx, elems in enumerate(transposed):
        first = elems[0]

        if first is None:                   # imgs / radar 可能 None
            out.append(None)

        elif torch.is_tensor(first):
            # ---- 特判 voxel_in / voxel_coor ----
            if idx in (8, 9):
                # P 在倒数第 3 维
                pad_dim = first.ndim - 3
                N_max = max(e.shape[pad_dim] for e in elems)

                padded = []
                for e in elems:
                    if e.shape[pad_dim] < N_max:
                        pad_size = list(e.shape)
                        pad_size[pad_dim] = N_max - e.shape[pad_dim]
                        pad = torch.zeros(*pad_size,
                                          dtype=e.dtype,
                                          device=e.device)
                        e = torch.cat([e, pad], dim=pad_dim)
                    padded.append(e)
                out.append(torch.stack(padded))
            else:
                out.append(torch.stack(elems))

        elif isinstance(first, list):       # gt_boxes / labels
            out.append([e for e in elems])

        else:
            raise TypeError(type(first))
    return tuple(out)



# ---------- compile_data ----------
def compile_data(cfg_file: str | Path = CFG_FILE_DEFAULT,
                 batch_size: int = 1,
                 num_workers: int = 4,
                 v_target: int = V_TARGET_DEFAULT,
                 shuffle: bool = True):
    cfg_from_yaml_file(str(cfg_file), cfg)

    train_raw, val_raw, _ = pcdet_build_dl(
        dataset_cfg   = cfg.DATA_CONFIG,
        class_names   = cfg.CLASS_NAMES,
        batch_size    = 1,   # 单样本
        dist          = False,
        workers       = 4,
        logger        = None,
        training      = True,
        merge_all_iters_to_one_epoch = False,
        total_epochs  = 1,
    )

    train_ds = TupleWrapper(train_raw, v_target)
    val_ds   = TupleWrapper(val_raw, v_target)

    trainloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=True,
        collate_fn=tuple_collate, pin_memory=False
    )
    valloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        collate_fn=tuple_collate, pin_memory=False
    )
    return trainloader, valloader


# ------------ quick demo ------------
# if __name__ == '__main__':
#     tl, _ = compile_data(batch_size=2)
#     batch = next(iter(tl))
#     for i, e in enumerate(batch):
#         if torch.is_tensor(e):
#             print(f"[{i:2}] tensor       shape {tuple(e.shape)}")
#         elif e is None:
#             print(f"[{i:2}] None")
#         else:
#             shapes = [tuple(t.shape) for t in e]
#             print(f"[{i:2}] list(len={len(e)}) elem_shapes {shapes}")
