"""
Offline evaluation script (mAP) for Segnet* models on NuScenes val split.
This is a **stand‑alone replacement** for your original eval.py, trimmed down to
focus purely on *3‑D object detection* (mAP@0.5 BEV).  It

* loads a trained checkpoint (SegnetTransformerLiftFuse / SegnetSimpleLiftFuse …)
* runs it in inference mode on the val dataloader
* collects predictions & ground‑truth boxes
* computes class‑agnostic AP with IoU=0.5 in BEV
* prints mAP and saves it to TensorBoard + CSV

Dependencies: numpy, torch, tensorboardX, shapely, your utils/, nets/, ops/iou3d_nms

Author: ChatGPT – 2025‑06‑17
"""

# -----------------------------------------------------------------------------
# 1. Imports & misc
# -----------------------------------------------------------------------------
import argparse, os, random, time, warnings, yaml, csv
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from shapely.errors import ShapelyDeprecationWarning

# project‑local lib
import nuscenes_data, saverloader
import utils.basic, utils.geom, utils.vox
from nets.segnet_transformer_lift_fuse_new_decoders import SegnetTransformerLiftFuse
from nets.segnet_simple_lift_fuse_ablation_new_decoders import SegnetSimpleLiftFuse

from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou_bev  # CUDA BEV IoU kernel

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

random.seed(125); np.random.seed(125); torch.manual_seed(125)

# -----------------------------------------------------------------------------
# 2. Helper – simple AP calculator (class‑agnostic, IoU@0.5, Pascal style)
# -----------------------------------------------------------------------------

def compute_map(preds, gts, iou_thr=0.5):
    """preds / gts : list[ dict{boxes(N,7),scores(N),labels(N)} ]"""
    tp, fp, conf = [], [], []
    num_gt = 0
    for pred, gt in zip(preds, gts):
        boxes_p = pred['boxes3d'].cuda(); scores_p = pred['scores']
        boxes_g = gt['boxes3d'].cuda();   num_gt += boxes_g.size(0)
        matched = set()
        for b, score in zip(boxes_p, scores_p):
            ious = boxes_iou_bev(b[None], boxes_g).squeeze(0)  # (M,)
            max_iou, idx = ious.max(0)
            if max_iou >= iou_thr and idx.item() not in matched:
                tp.append(1); fp.append(0); matched.add(idx.item())
            else:
                tp.append(0); fp.append(1)
            conf.append(score.item())
    if num_gt == 0:
        return 0.0
    # sort by confidence desc
    order = np.argsort(-np.array(conf))
    tp = np.cumsum(np.array(tp)[order])
    fp = np.cumsum(np.array(fp)[order])
    recall    = tp / num_gt
    precision = tp / np.maximum(tp+fp, 1e-9)
    # integrate PR curve (11‑point)
    recall_levels = np.linspace(0,1,11)
    precisions = [precision[recall>=r].max() if (recall>=r).any() else 0.0 for r in recall_levels]
    ap = np.mean(precisions)
    return ap

# -----------------------------------------------------------------------------
# 3. Vox/grid defaults (same as original)
# -----------------------------------------------------------------------------
XMIN, XMAX = -50, 50; ZMIN, ZMAX = -50, 50; YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)
Z,Y,X = 200,8,200
scene_centroid = torch.tensor([[0.0,1.0,0.0]], dtype=torch.float32)

# -----------------------------------------------------------------------------
# 4. run_model – now returns pred_dicts in eval mode
# -----------------------------------------------------------------------------

def run_model(model, batch, vox_util, device='cuda:0', return_pred=False):
    imgs, rots, trans, intrins = batch[:4]
    gt_boxes_l, gt_labels_l    = batch[-2:]

    imgs    = imgs[:,0].to(device)
    intrins = intrins[:,0].to(device)
    B,S = imgs.shape[:2]
    cam0_T_camXs = torch.eye(4, device=device).view(1,1,4,4).repeat(B,S,1,1)

    out = model(rgb_camXs=imgs,
                pix_T_cams=intrins,
                cam0_T_camXs=cam0_T_camXs,
                vox_util=vox_util,
                rad_occ_mem0=None)

    # training=False, forward returns list[dict]
    if isinstance(out, list):
        if return_pred:
            pred_dicts = []
            for d in out:
                pred_dicts.append({'boxes3d': d['pred_boxes'],
                                   'scores':  d['pred_scores'],
                                   'labels':  d['pred_labels']})
            # dummy loss (not used) & empty metrics
            return torch.tensor(0.0, device=device), {}, pred_dicts
        else:
            return torch.tensor(0.0, device=device), {}

    # seg branch not used in detection eval
    loss, metrics = out
    return loss, metrics, None

# -----------------------------------------------------------------------------
# 5. Main eval loop
# -----------------------------------------------------------------------------

def main(config):
    device_ids = config.get('device_ids',[0])
    device = f'cuda:{device_ids[0]}'

    # ---------------- dataloader ----------------
    data_aug_conf = {'final_dim':config['final_dim'],
                     'cams':['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT'],
                     'ncams':config['ncams']}
    _, val_dl = nuscenes_data.compile_data(
        'val', config['data_dir'], data_aug_conf=data_aug_conf,
        centroid=scene_centroid.numpy(), bounds=bounds, res_3d=(Z,Y,X),
        bsz=config['batch_size'], nworkers=config['nworkers'], shuffle=False,
        use_pre_scaled_imgs=True)

    vox_util = utils.vox.Vox_util(Z,Y,X, scene_centroid=scene_centroid.to(device), bounds=bounds)

    # ---------------- model ----------------
    if config['model_type']=='transformer':
        net = SegnetTransformerLiftFuse(Z_cam=200,Y_cam=8,X_cam=200,Z_rad=Z,Y_rad=Y,X_rad=X,vox_util=None,
                                         use_radar=False, train_task='object', rand_flip=False,
                                         encoder_type=config['encoder_type'],
                                         freeze_dino=True)
    else:
        net = SegnetSimpleLiftFuse(Z_cam=200,Y_cam=8,X_cam=200,Z_rad=Z,Y_rad=Y,X_rad=X,vox_util=None,
                                    use_radar=False, train_task='object', rand_flip=False,
                                    encoder_type=config['encoder_type'])
    net = torch.nn.DataParallel(net, device_ids=device_ids).to(device)

    # load ckpt
    saverloader.load(config['ckpt'], net.module, ignore_load=None, is_DP=True, step=config.get('load_step'))
    net.eval(); print('Loaded model ↗')

    # ---------------- evaluation loop ----------------
    writer = SummaryWriter(os.path.join(config['log_dir'], 'eval_det'))
    all_preds, all_gts = [], []
    t0=time.time()
    for step,batch in enumerate(val_dl):
        with torch.no_grad():
            _, _, pred_dicts = run_model(net, batch, vox_util, device, return_pred=True)
        all_preds.extend(pred_dicts)
        gt_boxes_l, gt_labels_l = batch[-2:]
        for b in range(len(gt_boxes_l)):
            all_gts.append({'boxes3d':gt_boxes_l[b].to(device), 'labels':gt_labels_l[b]})
        if (step+1)%config['log_freq']==0:
            print(f'processed {step+1}/{len(val_dl)} batches...')
    runtime = time.time()-t0

    mAP = compute_map(all_preds, all_gts, iou_thr=0.5)
    print(f'\n==============================\n  mAP@0.5 (BEV): {mAP*100:.2f}%\n  Runtime   : {runtime/len(val_dl):.3f} s / batch\n==============================')

    # log
    writer.add_scalar('val/mAP_0.5', mAP, 0)
    csv_path = os.path.join(config['log_dir'], 'val_det_log.csv')
    os.makedirs(config['log_dir'], exist_ok=True)
    with open(csv_path,'a',newline='') as f:
        csv.writer(f).writerow([config['ckpt'], mAP])
    writer.close()

# -----------------------------------------------------------------------------
# 6. CLI
# -----------------------------------------------------------------------------
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True, help='eval_det.yaml')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
