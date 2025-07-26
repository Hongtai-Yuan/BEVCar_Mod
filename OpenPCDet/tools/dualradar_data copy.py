# inspect_data.py
# 运行方式：python inspect_data.py
# 功能：不训练，只把数据 pipeline 跑一遍并打印首个 batch 的内容

import _init_path
from pathlib import Path
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader

# ------------------------------------------------------------------
# 1. 固定使用的 YAML 配置（自行按需要改为你那份）
CFG_FILE = Path('/root/autodl-tmp/BEVCar/OpenPCDet/tools/cfgs/dual_radar_models/pointpillar_lidar.yaml')

# 2. DataLoader 线程数和一次拿多少张样本
BATCH_SIZE = 4        # 可改大
NUM_WORKERS = 4
# ------------------------------------------------------------------


def main():
    # 读取 yaml
    if not CFG_FILE.exists():
        raise FileNotFoundError(f'找不到配置文件: {CFG_FILE.resolve()}')
    cfg_from_yaml_file(str(CFG_FILE), cfg)

    # 如果想覆盖 cfg 里的 batch_size，这里改即可
    batch_size = BATCH_SIZE or cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    # 创建 dataloader（只看训练 split，dist=False）
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=NUM_WORKERS,
        logger=None,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=1
    )

    print(f'Dataset size : {len(dataset)} samples')
    print(f'Batch size    : {batch_size}')
    print('---------------------------------------------')

    # 取一个 batch 打印
    for b_idx, batch in enumerate(dataloader):
        print(f'>>> Batch {b_idx}')
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f'  {k:<15}: Tensor {tuple(v.shape)}  {v.dtype}')
            elif isinstance(v, (list, tuple)):
                print(f'  {k:<15}: list/tuple len={len(v)}')
            else:
                print(f'  {k:<15}: {type(v)}  {v}')
        break  # 只看第一批


if __name__ == '__main__':
    main()
