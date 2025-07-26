# set seed in the beginning
import argparse
import os
import random
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # may help for debugging
# print("FIXED CUDA DEVICE: " + os.environ['CUDA_VISIBLE_DEVICES'])  # debug-only
import time
import warnings

import numpy as np
import torch
import torch.multiprocessing
import torch.nn.functional as F
import yaml
from shapely.errors import ShapelyDeprecationWarning
from tensorboardX import SummaryWriter
import logging
import sys    
import nuscenes_data
import saverloader
import utils.basic
import utils.geom
import utils.improc
import utils.misc
import utils.vox
from nets.segnet_simple_lift_fuse_ablation_new_decoders import (
    SegnetSimpleLiftFuse,
)
from nets.segnet_transformer_lift_fuse_new_decoders import (
    SegnetTransformerLiftFuse,
)
# from pcdet.models.dense_heads import TransFusionHead


# Suppress deprecation warnings from shapely regarding the nuscenes map api
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning, module="nuscenes.map_expansion.map_api")

torch.multiprocessing.set_sharing_strategy('file_system')

random.seed(125)
np.random.seed(125)
torch.manual_seed(125)

# the scene centroid is defined wrt a reference camera,
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0  # down 1 meter
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200




def requires_grad(parameters: iter, flag: bool = True) -> None:
    """
    Sets the `requires_grad` attribute of the given parameters.
    Args:
        parameters (iterable): An iterable of parameter tensors whose `requires_grad` attribute will be set.
        flag (bool, optional): If True, sets `requires_grad` to True. If False, sets it to False.
            Default is True.

    Returns:
        None
    """
    for p in parameters:
        p.requires_grad = flag


def fetch_optimizer(lr: float, wdecay: float, epsilon: float, num_steps: int, params: iter) \
        -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.OneCycleLR]:
    """
    Fetches an AdamW optimizer and a OneCycleLR scheduler.
    Args:
        lr (float): Learning rate for the optimizer.
        wdecay (float): Weight decay (L2 penalty) for the optimizer.
        epsilon (float): Term added to the denominator to improve numerical stability in the optimizer.
        num_steps (int): Number of steps for the learning rate scheduler.
        params (iter): Iterable of parameters to optimize or dictionaries defining parameter groups.

    Returns:
        tuple: A tuple containing the optimizer and the learning rate scheduler.
            - optimizer (torch.optim.AdamW): The AdamW optimizer.
            - scheduler (torch.optim.lr_scheduler.OneCycleLR): The OneCycleLR learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps + 100, pct_start=0.05,
                                                    cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


# --- STEP 1-1 BEGIN (覆盖原函数) ---
def create_train_pool_dict(name: str, n_pool: int):
    """
    只保留 total loss 和 耗时两个滑动窗口；检测任务其他指标由 tb_dict 直接写 tensorboard
    """
    return {
        f'loss_pool_{name}': utils.misc.SimplePool(n_pool, version='np'),
        f'time_pool_{name}': utils.misc.SimplePool(n_pool, version='np'),
    }, name
# --- STEP 1-1 END ---


# -----------------------------------------------------------------------------  
# RUN_MODEL ― 端到端 (兼容 voxel_net / 占位雷达)  
# -----------------------------------------------------------------------------
def run_model(model, batch, vox_util, device='cuda:0'):
    """
    适配 SegnetTransformerLiftFuse(DataParallel)：
        - 自动处理 dataloader 返回的各种 list/array → Tensor
        - 始终提供 rad_occ_mem0：
            • 若 radar_encoder_type == 'voxel_net' → 三元组 (feats, coords, nums)
            • 否则           → 单张量 (B,1,Z,Y,X)
    """
    # === ❶ 基础搬到 GPU -------------------------------------------------------
    imgs, rots, trans, intrins = batch[:4]
    radar_data                 = batch[6]            # NuScenes dataloader 第 7 位
    gt_boxes_l, gt_labels_l    = batch[-2:]

    imgs     = imgs[:, 0].to(device)                 # (B,S,C,H,W)
    rots     = rots[:, 0].to(device)
    trans    = trans[:, 0].to(device)
    intrins  = intrins[:, 0].to(device)
    rad_data = radar_data[:, 0].to(device)           # (B,R,19)

    B, S, _, H, W = imgs.shape
    cam0_T_camXs = torch.eye(4, device=device).view(1,1,4,4).repeat(B, S, 1, 1)

    # === ❷ GT padding --------------------------------------------------------
    def _norm(x):
        while isinstance(x, (list, tuple)):
            x = x[0] if len(x) else np.zeros((0, 9), np.float32)
        t = torch.as_tensor(x, dtype=torch.float32, device=device)
        if t.ndim == 1:  t = t[None]
        elif t.ndim > 2: t = t.reshape(-1, 9)
        return t
    flat_bxs  = [_norm(b) for b in gt_boxes_l]
    flat_labs = [_norm(l).view(-1) for l in gt_labels_l]

    max_n = max(t.size(0) for t in flat_bxs) or 1
    pad = imgs.new_zeros((B, max_n, 10))
    for b, (bx, lb) in enumerate(zip(flat_bxs, flat_labs)):
        n = bx.size(0)
        if n:                                     # 有真 GT，正常拷贝
            pad[b, :n, :9] = bx
            pad[b, :n,  9] = lb + 1
        else:                                     # 没有 GT → 塞 1 个哑目标
            pad[b, 0, :7] = torch.tensor([0.,0.,0., 0.1,0.1,0.1, 0.], device=device)  # xyz lwh θ
            pad[b, 0,  9] = 1.   # cls=1 对应 'car'

    # === ❸ 生成 / 占位 rad_occ_mem0 ------------------------------------------
    Z, Y, X = vox_util.Z, vox_util.Y, vox_util.X
    radar_type   = getattr(model.module, 'radar_encoder_type', '')
    use_voxnet  = radar_type == 'voxel_net'

    try:
        # 若你后续真的想用 voxel_net + 雷达点，请将下行 `raise` 删除，
        # 并把 voxelize_xyz_and_feats 等真实三元组生成逻辑写在这里。
        raise RuntimeError                           # 强制走 except 的占位分支
    except Exception:
        if use_voxnet:
            # --------- VoxelNet 占位三元组 ------------------------------------
            # feats shape: (B, num_vox=1, max_pts=1, num_feat=7)
            dummy_feats  = torch.zeros((B, 2, 2, 7), device=device)  # (B, vox=2, pts=2, feat=7)
            dummy_coords = torch.zeros((B, 2, 4), dtype=torch.int32, device=device)
            dummy_nums   = torch.full((B,), 2, dtype=torch.int32, device=device)  # 每批 2 voxel
            rad_occ_mem0 = (dummy_feats, dummy_coords, dummy_nums)
        else:
            # --------- 非 VoxelNet / RGB-only：单张量 -------------------------
            rad_occ_mem0 = torch.zeros((B, 1, Z, Y, X), device=device)

    # === ❹ 前向 --------------------------------------------------------------
    out = model(rgb_camXs    = imgs,
                pix_T_cams   = intrins,
                cam0_T_camXs = cam0_T_camXs,
                vox_util     = vox_util,
                rad_occ_mem0 = rad_occ_mem0,
                gt_boxes     = pad)

    return out if isinstance(out, tuple) else (0.*imgs.sum(), {})
# -----------------------------------------------------------------------------






def main(
        exp_name='bevcar_debug',
        # training
        max_iters=75000,
        log_freq=1000,
        shuffle=True,
        dset='trainval',
        save_freq=1000,
        batch_size=8,
        grad_acc=5,
        lr=3e-4,
        use_scheduler=True,
        weight_decay=1e-7,
        nworkers=12,
        # data/log/save/load directories
        data_dir='../nuscenes/',
        custom_dataroot='../../../nuscenes/scaled_images',
        log_dir='logs_nuscenes_bevcar',
        ckpt_dir='checkpoints/',
        keep_latest=1,
        init_dir='',
        val_freq=1000,
        ignore_load=None,
        load_step=False,
        load_optimizer=False,
        load_scheduler=False,
        # data
        final_dim=[448, 896],  # to match //8, //14, //16 and //32 in Vit
        rand_flip=True,
        rand_crop_and_resize=True,
        ncams=1,
        nsweeps=5,
        # model
        encoder_type='dino_v2',
        radar_encoder_type='voxel_net',
        use_rpn_radar=False,
        train_task='both',
        use_radar=False,
        use_radar_filters=False,
        use_radar_encoder=False,
        use_metaradar=False,
        use_shallow_metadata=False,
        use_pre_scaled_imgs=False,
        use_obj_layer_only_on_map=False,
        init_query_with_image_feats=True,
        do_rgbcompress=True,
        do_shuffle_cams=True,
        use_multi_scale_img_feats=False,
        num_layers=6,
        # cuda
        device_ids=[0, 1],
        freeze_dino=True,
        do_feat_enc_dec=True,
        combine_feat_init_w_learned_q=True,
        model_type='transformer',
        use_radar_occupancy_map=False,
        learnable_fuse_query=True,
):
    assert (model_type in ['transformer', 'simple_lift_fuse', 'SimpleBEV_map'])
    B = batch_size
    assert (B % len(device_ids) == 0)  # batch size must be divisible by number of gpus
    if grad_acc > 1:
        print('effective batch size:', B * grad_acc)
    device = 'cuda:%d' % device_ids[0]

    # debug only
    if torch.cuda.is_available():
        print("CUDA is available")
        print("Devices available: %d " % torch.cuda.device_count())
        print("Current CUDA device ID: %d" % torch.cuda.current_device())
        # device_ids[0])  # torch.cuda.current_device())
    else:
        print("CUDA is --- NOT --- available")

    # autogen a name
    model_name = "%d" % B
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    lrn = "%.1e" % lr  # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1]  # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_scheduler:
        model_name += "s"

    import datetime
    model_date = datetime.datetime.now().strftime('%H-%M-%S')
    model_name = model_name + '_' + model_date

    model_name = exp_name + '_' + model_name
    print('model_name', model_name)


    # ───── 在这里（model_name 和 log_dir 已经可用）初始化 logging ─────
    os.makedirs(log_dir, exist_ok=True)          # 放在 log_txt 之前
    log_txt = os.path.join(log_dir, f"{model_name}.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_txt, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
        force=True            # 若外部库已设置 logging，可强制覆盖
    )
    # set up ckpt and logging
    ckpt_dir = os.path.join(ckpt_dir, model_name)
    writer_t = SummaryWriter(os.path.join(log_dir, model_name + '/t'), max_queue=10, flush_secs=60)

    print('resolution:', final_dim)

    if use_radar_encoder:
        print("Radar encoder: ", radar_encoder_type)
    else:
        print("NO RADAR ENCODER")



    if rand_crop_and_resize:
        resize_lim = [0.8, 1.2]
        crop_offset = int(final_dim[0] * (1 - resize_lim[0]))
    else:
        resize_lim = [1.0, 1.0]
        crop_offset = 0

    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT'],
        'ncams': ncams,
    }
    # train_dataloader, val_dataloader = nuscenes_data.compile_data(
    #     dset,
    #     data_dir,
    #     data_aug_conf=data_aug_conf,
    #     centroid=scene_centroid_py,
    #     bounds=bounds,
    #     res_3d=(Z, Y, X),
    #     bsz=B,
    #     nworkers=nworkers,
    #     shuffle=shuffle,
    #     use_radar_filters=use_radar_filters,
    #     seqlen=1,
    #     nsweeps=nsweeps,
    #     do_shuffle_cams=do_shuffle_cams,
    #     radar_encoder_type=radar_encoder_type,
    #     use_shallow_metadata=use_shallow_metadata,
    #     use_pre_scaled_imgs=use_pre_scaled_imgs,
    #     custom_dataroot=custom_dataroot,
    #     use_obj_layer_only_on_map=use_obj_layer_only_on_map,
    #     use_radar_occupancy_map=use_radar_occupancy_map,
    # )
    from OpenPCDet.tools.mudata import compile_data
    train_dataloader, val_dataloader = compile_data()   # 全走默认
    train_iterloader = iter(train_dataloader)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)


    # Transformer based lifting and fusion
    if model_type == 'transformer':
        model = SegnetTransformerLiftFuse(Z_cam=200, Y_cam=8, X_cam=200, Z_rad=Z, Y_rad=Y, X_rad=X, vox_util=vox_util,
                                          use_radar=use_radar, use_metaradar=use_metaradar,
                                          use_shallow_metadata=use_shallow_metadata,
                                          use_radar_encoder=use_radar_encoder, do_rgbcompress=do_rgbcompress,
                                          encoder_type=encoder_type, radar_encoder_type=radar_encoder_type,
                                          rand_flip=rand_flip, train_task=train_task,
                                          init_query_with_image_feats=init_query_with_image_feats,
                                          use_obj_layer_only_on_map=use_obj_layer_only_on_map,
                                          do_feat_enc_dec=do_feat_enc_dec,
                                          use_multi_scale_img_feats=use_multi_scale_img_feats, num_layers=num_layers,
                                          latent_dim=128,
                                          combine_feat_init_w_learned_q=combine_feat_init_w_learned_q,
                                          use_rpn_radar=use_rpn_radar, use_radar_occupancy_map=use_radar_occupancy_map,
                                          freeze_dino=freeze_dino, learnable_fuse_query=learnable_fuse_query)

    elif model_type == 'simple_lift_fuse':
        # our net with replaced lifting and fusion from SimpleBEV
        model = SegnetSimpleLiftFuse(Z_cam=200, Y_cam=8, X_cam=200, Z_rad=Z, Y_rad=Y, X_rad=X, vox_util=vox_util,
                                     use_radar=use_radar, use_metaradar=use_metaradar,
                                     use_shallow_metadata=use_shallow_metadata, use_radar_encoder=use_radar_encoder,
                                     do_rgbcompress=do_rgbcompress, encoder_type=encoder_type,
                                     radar_encoder_type=radar_encoder_type, rand_flip=rand_flip, train_task=train_task,
                                     use_obj_layer_only_on_map=use_obj_layer_only_on_map,
                                     do_feat_enc_dec=do_feat_enc_dec,
                                     use_multi_scale_img_feats=use_multi_scale_img_feats, num_layers=num_layers,
                                     latent_dim=128, use_rpn_radar=use_rpn_radar,
                                     use_radar_occupancy_map=use_radar_occupancy_map,
                                     freeze_dino=freeze_dino)


    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    parameters = list(model.parameters())

    if use_scheduler:
        optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, max_iters, parameters)
    else:
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        scheduler = None

    # Counting trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params}')
    # Counting non-trainable parameters
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'Non-trainable parameters: {non_trainable_params}')
    # Overall parameters
    total_params = trainable_params + non_trainable_params
    print('Total parameters (trainable + fixed)', total_params)


    # load checkpoint
    global_step = 0
    if init_dir:
        if load_step and load_optimizer and load_scheduler:
            global_step = saverloader.load(init_dir, model, optimizer, scheduler=scheduler,
                                           ignore_load=ignore_load, device_ids=device_ids, is_DP=True)

        elif load_step and load_optimizer:
            global_step = saverloader.load(init_dir, model.module, optimizer, ignore_load=ignore_load,
                                           device_ids=device_ids)
            print("global_step: ", global_step)
        elif load_step:
            global_step = saverloader.load(init_dir, model.module, ignore_load=ignore_load, device_ids=device_ids)
        else:
            _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
            global_step = 0
        print('checkpoint loaded...')

    model.train()

    # set up running logging pool
    n_pool_train = 10

    sw_t = None

    # training loop
    while global_step < max_iters:
        global_step += 1

        iter_start_time = time.time()
        iter_read_time = 0.0

        metrics = {}
        metrics_mean_grad_acc = {}
        total_loss = 0.0

        for internal_step in range(grad_acc):
            # read sample
            read_start_time = time.time()

            if internal_step == grad_acc - 1:
                sw_t = utils.improc.Summ_writer(  # only save last sample of gradient accumulation
                    writer=writer_t,
                    global_step=global_step,
                    log_freq=log_freq,
                    fps=2,
                    scalar_freq=int(log_freq / 2),
                    just_gif=True)
            else:
                sw_t = None

            try:
                sample = next(train_iterloader)
            except StopIteration:
                train_iterloader = iter(train_dataloader)
                sample = next(train_iterloader)

            # === 只在第 1 步打印一次标签分布 =========================================
            if global_step == 1:
                def _norm(x):
                    """跟 run_model 里完全一样的递归展开 + to tensor"""
                    while isinstance(x, (list, tuple)):
                        x = x[0] if len(x) else np.zeros((0, 9), np.float32)
                    return torch.as_tensor(x, dtype=torch.float32)

                gt_boxes_nested = sample[-2]          # 倒数第 2 项
                flat_boxes = [_norm(b) for b in gt_boxes_nested]

                print("📦 标签分布（batch 内逐样本）：")
                for idx, bx in enumerate(flat_boxes):
                    if bx.ndim != 2 or bx.size(0) == 0:
                        print(f"  sample[{idx}] : 无 gt 或 shape={bx.shape}")
                        continue
                    labels = bx[:, -1].long()              # 最后一列就是 label
                    uniq, cnt = torch.unique(labels, return_counts=True)
                    txt = ", ".join([f"{u.item()}:{c.item()}" for u, c in zip(uniq, cnt)])
                    print(f"  sample[{idx}] : {txt}")




            read_time = time.time() - read_start_time
            iter_read_time += read_time

            # run training iteration
            total_loss_, tb_dict = run_model(model, sample, vox_util, device)


            (total_loss_ / grad_acc).backward()

            # collect total loss and metrics over grad_acc steps
            if internal_step == 0:
                total_loss = total_loss_
            else:
                total_loss += total_loss_

            if internal_step == grad_acc - 1:
                total_loss = total_loss / grad_acc  # 12.9796

        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if use_scheduler:
            scheduler.step()
        optimizer.zero_grad()


        # --- 4. 按 save_freq 保存模型 ---
        if global_step % save_freq == 0:
            saverloader.save(
                ckpt_dir, optimizer, model.module, global_step,
                scheduler=scheduler, keep_latest=keep_latest
            )

        model.train()

        # log lr and time
        current_lr = optimizer.param_groups[0]['lr']
        sw_t.summ_scalar('_/current_lr', current_lr)

        iter_time = time.time() - iter_start_time


        # --- STEP 1-2 BEGIN (替换三处 print) ---
        print(f'{model_name}; step {global_step:06d}/{max_iters}; '
            f'time {iter_time:.2f}s; loss {total_loss.item():.4f}')
        logging.info(
            f"{model_name}; step {global_step:06d}/{max_iters}; "
            f"time {iter_time:.2f}s; loss {total_loss.item():.4f}"
        )
        # --- STEP 1-2 END ---





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training with model-specific config.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')

    args = parser.parse_args()

    # Load the config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    main(**config)
