"""
code adapted from https://github.com/nv-tlabs/lift-splat-shoot
and also https://github.com/wayveai/fiery/blob/master/fiery/data.py
"""
import os
import random
from functools import reduce
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import NuScenesEval
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, PointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from PIL import Image
from pyquaternion import Quaternion
from scipy import ndimage
# distributed dataloader
from torch.utils.data.distributed import DistributedSampler

import utils.geom
import utils.py
import utils.vox
# custom eval split
from custom_nuscenes_splits import create_drn_eval_split_scenes

discard_invisible = False


def convert_egopose_to_matrix_numpy(egopose):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = Quaternion(egopose['rotation']).rotation_matrix
    translation = np.array(egopose['translation'])
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def get_radar_data(nusc, sample_rec, nsweeps, min_distance, use_radar_filters, dataroot):
    """
    Returns at most nsweeps of radar in the ego frame.
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    # import ipdb; ipdb.set_trace()

    # points = np.zeros((5, 0))
    points = np.zeros((19, 0))  # 18 plus one for time

    # Get reference pose and timestamp.
    ref_sd_token = sample_rec['data']['RADAR_FRONT']
    ref_sd_rec = nusc.get('sample_data', ref_sd_token)
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    ref_time = 1e-6 * ref_sd_rec['timestamp']

    # Homogeneous transformation matrix from global to _current_ ego car frame.
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    if use_radar_filters:
        RadarPointCloud.default_filters()
    else:
        RadarPointCloud.disable_filters()

    # Aggregate current and previous sweeps.
    # from all radars
    radar_chan_list = ["RADAR_BACK_RIGHT", "RADAR_BACK_LEFT", "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT"]
    for radar_name in radar_chan_list:
        sample_data_token = sample_rec['data'][radar_name]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = RadarPointCloud.from_file(os.path.join(dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
            times = time_lag * np.ones((1, current_pc.nbr_points()))

            new_points = np.concatenate((current_pc.points, times), 0)
            points = np.concatenate((points, new_points), 1)

            # print('time_lag', time_lag)
            # print('new_points', new_points.shape)

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points


def ego_to_cam(points, rot, trans, intrins=None):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    if intrins is not None:
        points = intrins.matmul(points)
        points[:2] /= points[2:3]

    return points


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) & \
        (pts[0] > 1) & (pts[0] < W - 1) & \
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


def img_transform(img, resize_dims, crop):
    img = img.resize(resize_dims, Image.NEAREST)
    img = img.crop(crop)
    return img


def img_crop(img, resize_dims, crop):
    img = img.crop(crop)
    return img


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    torchvision.transforms.ToPILImage(),
))

denormalize_img_torch = torchvision.transforms.Compose((
    NormalizeInverse(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
))

normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))
totorch_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
))
normalize_img_torch = torchvision.transforms.Compose((
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([int((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084 / 2. + 0.5, W / 2.],
        [4.084 / 2. + 0.5, W / 2.],
        [4.084 / 2. + 0.5, -W / 2.],
        [-4.084 / 2. + 0.5, -W / 2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0, 1]] = pts[:, [1, 0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def add_ego2(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084 / 2. + 1, W / 2.],
        [4.084 / 2. + 1, W / 2.],
        [4.084 / 2. + 1, -W / 2.],
        [-4.084 / 2. + 1, -W / 2.],
    ])
    pts = (pts - bx) / dx
    # pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                       map_name=map_name) for map_name in [
                     "singapore-hollandvillage",
                     "singapore-queenstown",
                     "boston-seaport",
                     "singapore-onenorth",
                 ]}
    return nusc_maps


def fetch_nusc_map2(rec, nusc_maps, nusc, scene2map, car_from_current):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])

    global_from_car = transform_matrix(egopose['translation'],
                                       Quaternion(egopose['rotation']), inverse=False)

    trans_matrix = reduce(np.dot, [global_from_car, car_from_current])

    rot = np.arctan2(trans_matrix[1, 0], trans_matrix[0, 0])
    center = np.array([trans_matrix[0, 3], trans_matrix[1, 3], np.cos(rot), np.sin(rot)])

    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    # Debug only:
    # print("Current map: ", map_name)

    poly_names = ['drivable_area', 'carpark_area', 'ped_crossing', 'walkway', 'stop_line']
    line_names = ['road_divider', 'lane_divider']

    # lmap = get_local_map(nusc_maps[map_name], center, 50.0, poly_names, line_names)

    # own labels handling
    # use build in nuscenes function to get binary map mask
    patch_box = (center[0], center[1], 100, 100)  # patch of the map (center_x, center_y, height, width)
    rot_angle = rot * 180 / np.pi  # rotation of the patch -> should match the model output
    # print for console debug
    # print("Car rot angle: ", rot_angle)
    patch_angle = rot_angle  # angle in degrees
    # patch_angle = 0.0  # for debugging only
    canvas_size = (200, 200)  # size of generated output mask -> 200x200
    layer_names = poly_names + line_names
    map_mask = nusc_maps[map_name].get_map_mask(patch_box=patch_box,
                                                patch_angle=patch_angle,
                                                layer_names=layer_names,
                                                canvas_size=canvas_size)

    """ Debug: test if mask is correct:
    figsize = (12, 4)
    fig, ax = nusc_maps[map_name].render_map_mask(patch_box=patch_box,
                                                  patch_angle=patch_angle,
                                                  layer_names=layer_names,
                                                  canvas_size=canvas_size,
                                                  figsize=figsize,
                                                  n_row=1)
    """

    # generate bev view of the ego car for the current patch
    ego_in_bev = np.zeros_like(map_mask[0])

    W = 1.85
    bx = -49.75
    dx = 0.5
    car_points = np.array([[-4.084 / 2. + 1, W / 2.],
                           [4.084 / 2. + 1, W / 2.],
                           [4.084 / 2. + 1, -W / 2.],
                           [-4.084 / 2. + 1, -W / 2.]])

    car_points_scale_shift = (car_points - bx) / dx

    round_car_approx = np.round(car_points_scale_shift)
    x = abs((round_car_approx[0][1] + 1) - round_car_approx[-1][1])
    y = abs(round_car_approx[0][0] - (round_car_approx[1][0] - 1))
    car_mask = np.ones((int(y), int(x)))
    ego_in_bev[int(round_car_approx[0][0]): int(round_car_approx[1][0] - 1),
    int(round_car_approx[-1][1]): int(round_car_approx[0][1] + 1)] = car_mask

    rot_angle = 0.0
    ego_angle = np.arctan2(car_from_current[1, 0], car_from_current[0, 0])
    ego_angle = ego_angle * 180 / np.pi

    bev_plane_rot = ndimage.rotate(ego_in_bev, angle=ego_angle + 90.0, reshape=False)
    egocar_bev = (bev_plane_rot > 0.1).astype(int)

    return map_mask, egocar_bev


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch,
        center[1] - stretch,
        center[0] + stretch,
        center[1] + stretch,
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for record in getattr(nmap, layer_name):
            token = record['token']

            line = nmap.extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
            )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys


# BEVCar colors
masks_colors = np.zeros((7, 3))
masks_colors[0] = (255. / 255., 192. / 255., 203. / 255.)  # pink
masks_colors[1] = (255. / 255., 165. / 255., 0. / 255.)  # orange
masks_colors[2] = (0. / 255., 200. / 255., 0. / 255.)  # dimmer green
masks_colors[3] = (154. / 255., 230. / 255., 198. / 255.)  # teal
masks_colors[4] = (255. / 255., 0. / 255., 0. / 255.)  # red
masks_colors[5] = (255. / 255., 255. / 255., 0. / 255.)  # yellow
masks_colors[6] = (128. / 255., 0. / 255., 128. / 255.)  # purple


def get_rgba_map_from_mask2(map_masks, threshold=0.8, a=0.4):
    """
    class 0:    'drivable_area' --- color in rbg: (1.00, 0.50, 0.31)\n
    class 1:    'carpark_area'  --- color '#FFD700' in rbg: (255./255., 215./255., 0./255)\n
    class 2:    'ped_crossing'  --- color '#069AF3' in rbg: (6./255., 154/255., 243/255.) \n
    class 3:    'walkway'       --- color '#FF00FF' in rbg: (255./255., 0./255., 255./255.) \n
    class 4:    'stop_line'     --- color '#FF0000' in rbg: (255./255., 0./255., 0./255.) \n
    class 5:    'road_divider'  --- color in rbg: (0.0, 0.0, 1.0)\n
    class 6:    'lane_divider'  --- color in rbg: (159./255., 0.0, 1.0)\n

    threshold: defines at which probability the pixel belongs to a certain class
    a: sets the opacity of the mask layers; if alpha=1.0 no overlapping visible; default: alpha=0.4
    """
    masks_shape = map_masks.shape
    rgba_image = np.ones((4, masks_shape[1], masks_shape[2]), dtype=np.float32)

    for layer, (mask, color) in enumerate(zip(map_masks, masks_colors)):
        alpha = a * (mask > threshold).astype(np.float32)  # a = 0.4
        # Assign RGB values based on the color
        rgb = np.full((3, 200, 200), color[..., np.newaxis, np.newaxis])

        # Combine the RGB and alpha channels
        rgba = np.concatenate([rgb, np.expand_dims(alpha, axis=0)], axis=0)

        # Update the RGBA image with the values of the current mask
        rgba_image = rgba_image * (1 - alpha) + rgba * alpha

    bev_map = rgba_image[:3]
    if bev_map.max() > 1.0:
        bev_map = np.clip(bev_map, 0.0, 1.0)
    return torch.Tensor(bev_map)


def get_rgba_map_from_mask2_on_batch(map_masks: torch.Tensor, threshold: float = 0.8, a: float = 0.4) -> torch.Tensor:
    """
    class 0:    'drivable_area' --- color in rbg: (1.00, 0.50, 0.31)\n
    class 1:    'carpark_area'  --- color '#FFD700' in rbg: (255./255., 215./255., 0./255)\n
    class 2:    'ped_crossing'  --- color '#069AF3' in rbg: (6./255., 154/255., 243/255.) \n
    class 3:    'walkway'       --- color '#FF00FF' in rbg: (255./255., 0./255., 255./255.) \n
    class 4:    'stop_line'     --- color '#FF0000' in rbg: (255./255., 0./255., 0./255.) \n
    class 5:    'road_divider'  --- color in rbg: (0.0, 0.0, 1.0)\n
    class 6:    'lane_divider'  --- color in rbg: (159./255., 0.0, 1.0)\n

    threshold: defines at which probability the pixel belongs to a certain class
    a: sets the opacity of the mask layers; if alpha=1.0 no overlapping visible; default: alpha=0.4
    """
    B, classes, length_z, length_x = map_masks.shape  # (B,7,200,200)
    # (7, 200, 200)
    bev_map = np.ones((B, 3, 200, 200))  # (B, 3, 200, 200)
    rgba_image = np.ones((4, length_z, length_x), dtype=np.float32)  # (B, 4, 200, 200)

    for b in range(B):
        for layer, (mask, color) in enumerate(zip(map_masks[b], masks_colors)):
            alpha = a * (mask > threshold).astype(np.float32)  # a = 0.4   # (200, 200)
            # Assign RGB values based on the color
            rgb = np.full((3, 200, 200), color[..., np.newaxis, np.newaxis])  # (3, 200, 200)

            # Combine the RGB and alpha channels
            rgba = np.concatenate([rgb, np.expand_dims(alpha, axis=0)], axis=0)  # (4, 200, 200)

            # Update the RGBA image with the values of the current mask
            rgba_image = rgba_image * (1 - alpha) + rgba * alpha  # (4, 200, 200)

        bev_map[b] = rgba_image[:3]

    # clip values
    if bev_map.max() > 1.0:
        bev_map = np.clip(bev_map, 0.0, 1.0)
    return torch.Tensor(bev_map)


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc: NuScenes, nusc_maps: dict, is_train: bool, data_aug_conf: dict,
                 centroid: torch.Tensor = None, bounds: tuple = None, res_3d=None, nsweeps: int = 1, seqlen: int = 1,
                 refcam_id: int = 1, get_tids: bool = False, temporal_aug: bool = False,
                 use_radar_filters: bool = False, do_shuffle_cams: bool = True, radar_encoder_type: bool = None,
                 use_shallow_metadata: bool = True, use_pre_scaled_imgs: bool = True, custom_dataroot: str = None,
                 use_obj_layer_only_on_map: bool = False, vis_full_scenes: bool = False,
                 use_radar_occupancy_map: bool = False, do_drn_val_split: bool = False, get_val_day: bool = False,
                 get_val_rain: bool = False, get_val_night: bool = False, print_details: bool = False):
        self.nusc = nusc
        self.nusc_maps = nusc_maps
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        # self.grid_conf = grid_conf
        self.nsweeps = nsweeps
        self.use_radar_filters = use_radar_filters
        self.do_shuffle_cams = do_shuffle_cams
        self.res_3d = res_3d
        self.bounds = bounds
        self.centroid = centroid

        self.seqlen = seqlen
        self.refcam_id = refcam_id
        self.vis_full_scenes = vis_full_scenes
        self.use_radar_occupancy_map = use_radar_occupancy_map

        self.do_drn_val_split = do_drn_val_split
        self.get_val_day = get_val_day
        self.get_val_rain = get_val_rain
        self.get_val_night = get_val_night

        self.print_details = print_details

        self.dataroot = self.nusc.dataroot

        self.scenes = self.get_scenes()

        if self.do_drn_val_split and not self.is_train:
            # returns samples in the correct "DAY/RAIN/NIGHT" order
            self.ixes = self.get_ordered_drn_samples()
        else:
            self.ixes = self.prepro()
        if temporal_aug:
            self.indices = self.get_indices_tempaug()
        else:
            self.indices = self.get_indices()

        self.get_tids = get_tids

        self.radar_encoder_type = radar_encoder_type
        self.use_shallow_metadata = use_shallow_metadata

        self.use_pre_scaled_imgs = use_pre_scaled_imgs
        self.custom_dataroot = custom_dataroot

        self.use_obj_layer_only_on_map = use_obj_layer_only_on_map

        XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX = self.bounds
        Z, Y, X = self.res_3d
        self.Z = Z
        self.Y = Y
        self.X = X

        self.vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=torch.from_numpy(self.centroid).float().cuda(),
            bounds=self.bounds,
            assert_cube=False)

        grid_conf = {  # note the downstream util uses a different XYZ ordering
            'xbound': [XMIN, XMAX, (XMAX - XMIN) / float(X)],
            'ybound': [ZMIN, ZMAX, (ZMAX - ZMIN) / float(Z)],
            'zbound': [YMIN, YMAX, (YMAX - YMIN) / float(Y)],
        }
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        if self.print_details:
            # print('ixes', self.ixes.shape)
            print('indices', self.indices.shape)
            print(self)

    def get_scenes(self):
        if self.do_drn_val_split and not self.is_train:  # only for eval...
            if self.get_val_day:
                split = 'val_day'
            elif self.get_val_rain:
                split = 'val_rain'
            elif self.get_val_night:
                split = 'val_night'
            else:  # get all val data, but in: day, rain, night order
                split = 'val_all'
            scenes = create_drn_eval_split_scenes()[split]
        else:
            # filter by scene split
            split = {
                'v1.0-trainval': {True: 'train', False: 'val'},
                'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
                'v1.0-test': {True: 'test', False: 'test'},
            }[self.nusc.version][self.is_train]
            scenes = create_splits_scenes()[split]
        return scenes




    def evaluate_nusc_detection(self, det_annos, output_dir):
        # det_annos: list of dict, 每个 dict 包含 'sample_token', 'translation', 'size', 'rotation', 'velocity', 'detection_name', 'detection_score'
        nusc = NuScenes(version='v1.0-{}'.format(self.nusc.version.split('-')[-1]),
                    dataroot=self.nusc.dataroot, verbose=False)

        # 转为 NuScenes 官方格式
        nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)  
        nusc_annos['meta'] = {
            'use_camera': True,   # 根据你模型输入决定
            'use_lidar': False,
            'use_radar': True,
            'use_map': True,
            'use_external': False,
        }

        res_path = os.path.join(output_dir, 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        if self.nusc.version.endswith('test'):
            print("No GT for test split.")
            return

        eval_set = 'val' if 'trainval' in self.nusc.version else 'mini_val'
        cfg = config_factory('detection_cvpr_2019')
        nusc_eval = NuScenesEval(
            nusc,
            config=cfg,
            result_path=res_path,
            eval_set=eval_set,
            output_dir=output_dir,
            verbose=True,
        )
        metrics = nusc_eval.main(plot_examples=0, render_curves=False)
        print("NuScenes Detection results:\n", metrics)


    def get_drn_scene_tokens(self):
        # Create a mapping from scene names to scene tokens
        scene_name_to_token = {scene['name']: scene['token'] for scene in self.nusc.scene}

        # Retrieve scene tokens in the order of scene names in self.scenes
        drn_scene_tokens = []
        for scene_name in self.scenes:
            if scene_name in scene_name_to_token:
                drn_scene_tokens.append(scene_name_to_token[scene_name])

        return drn_scene_tokens

    def get_ordered_drn_samples(self):
        """
        Returns:
            all samples in the dataset sorted for scene ids
        """
        # Create a mapping from scene names to scene tokens
        scene_name_to_token = {scene['name']: scene['token'] for scene in self.nusc.scene}
        drn_samples = []

        for scene_name in self.scenes:
            if scene_name in scene_name_to_token:
                scene_token = scene_name_to_token[scene_name]
                # Retrieve the first sample token from the scene
                scene = self.nusc.get('scene', scene_token)
                first_sample_token = scene['first_sample_token']

                # Start with the first sample
                current_sample_token = first_sample_token
                while current_sample_token:
                    # Retrieve the current sample
                    sample = self.nusc.get('sample', current_sample_token)
                    drn_samples.append(sample)

                    # Move to the next sample in the scene
                    current_sample_token = sample['next']

        return drn_samples

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]
        # remove samples that aren't in this split
        samples = [samp for samp in samples if self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]
        return samples

    def get_indices(self):
        """
        Note: if used for sequence collection -> e.g. get 40 samples of a scene for visualization
        the following will happen: scenes with under (<)40 samples will be skipped and scenes with
        over (>)40 samples  will occur essentially be stored multiple times
        this behaviour does not affect single sample sequences, used in training and eval.
        For scene - based visualization, please set self.vis_full_scenes to True.
        """
        indices = []
        if self.vis_full_scenes:
            # --> visualize the whole scene and avoid skipped / doubled scenes --> set variable
            valid = True
            index = 0
            while index < len(self.ixes):  # as long as we are not over every sample in the current split...
                # start with first index, go as long as "valid"
                current_indices = []
                while valid:
                    rec = self.ixes[index]
                    current_indices.append(index)
                    # check if next sample is in the current scene --> if not -> close scene here
                    if rec['next'] == '':
                        valid = False
                    index += 1

                # add length of valid indices
                current_indices.insert(0, len(current_indices))
                # check if we need to fill up to 42 values
                while len(current_indices) != 42:
                    current_indices.append(0)

                indices.append(current_indices)
                valid = True

        else:  # single sample or incomplete scenes
            for index in range(len(self.ixes)):
                is_valid_data = True
                previous_rec = None
                current_indices = []
                for t in range(self.seqlen):
                    index_t = index + t
                    # Going over the dataset size limit.
                    if index_t >= len(self.ixes):
                        is_valid_data = False
                        break
                    rec = self.ixes[index_t]
                    # Check if scene is the same
                    if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                        is_valid_data = False
                        break

                    current_indices.append(index_t)
                    previous_rec = rec

                if is_valid_data:
                    index += 38
                    indices.append(current_indices)

        return np.asarray(indices)

    def get_indices_tempaug(self):
        indices = []
        t_patterns = None
        if self.seqlen == 1:
            return self.get_indices()
        elif self.seqlen == 2:
            # seq options: (t, t+1), (t, t+2)
            t_patterns = [[0, 1], [0, 2]]
        elif self.seqlen == 3:
            # seq options: (t, t+1, t+2), (t, t+1, t+3), (t, t+2, t+3)
            t_patterns = [[0, 1, 2], [0, 1, 3], [0, 2, 3]]
        elif self.seqlen == 5:
            t_patterns = [
                [0, 1, 2, 3, 4],  # normal
                [0, 1, 2, 3, 5], [0, 1, 2, 4, 5], [0, 1, 3, 4, 5], [0, 2, 3, 4, 5],  # 1 skip
                # [1,0,2,3,4], [0,2,1,3,4], [0,1,3,2,4], [0,1,2,4,3], # 1 reverse
            ]
        else:
            raise NotImplementedError("timestep not implemented")

        for index in range(len(self.ixes)):
            for t_pattern in t_patterns:
                is_valid_data = True
                previous_rec = None
                current_indices = []
                for t in t_pattern:
                    index_t = index + t
                    # going over the dataset size limit
                    if index_t >= len(self.ixes):
                        is_valid_data = False
                        break
                    rec = self.ixes[index_t]
                    # check if scene is the same
                    if (previous_rec is not None) and (rec['scene_token'] != previous_rec['scene_token']):
                        is_valid_data = False
                        break

                    current_indices.append(index_t)
                    previous_rec = rec

                if is_valid_data:
                    indices.append(current_indices)
                    # indices.append(list(reversed(current_indices)))
                    # indices += list(itertools.permutations(current_indices))

        return np.asarray(indices)

    def sample_augmentation(self) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
        """
        Applies augmentation on inout image data by resizing and cropping
        Returns:
            tuple[tuple,tuple]: resize_dims, crop
        """
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            if 'resize_lim' in self.data_aug_conf and self.data_aug_conf['resize_lim'] is not None:
                resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            else:
                resize = self.data_aug_conf['resize_scale']

            resize_dims = (int(fW * resize), int(fH * resize))

            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = self.data_aug_conf['crop_offset']
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        for cam in cams:
            resize_dims, crop = self.sample_augmentation()

            samp = self.nusc.get('sample_data', rec['data'][cam])

            if not self.use_pre_scaled_imgs:
                imgname = os.path.join(self.dataroot, samp['filename'])
                img = Image.open(imgname)
                W, H = img.size
            else:
                custom_path = os.path.join(self.custom_dataroot, '#CUSTOM_RES#')  # TODO: adapt to rescaled imgs folder
                imgname = os.path.join(custom_path, samp['filename'])
                img = Image.open(imgname)
                W, H = (1600, 900)  # img.size must be fixed to keep the rest of the code working

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)
            tran = torch.Tensor(sens['translation'])

            # W and H: original image size to scale intrinsics correctly
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            intrin = utils.geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = utils.geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = utils.geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)

            img = img_transform(img, resize_dims, crop)

            imgs.append(totorch_img(img))

            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)

        return torch.stack(imgs), torch.stack(rots), torch.stack(trans), torch.stack(intrins)

    def get_radar_data(self, rec, nsweeps):
        pts = get_radar_data(self.nusc, rec, nsweeps=nsweeps, min_distance=2.2,
                             use_radar_filters=self.use_radar_filters, dataroot=self.dataroot)
        return torch.Tensor(pts)

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for ii, tok in enumerate(rec['anns']):
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if discard_invisible and int(inst['visibility_token']) == 1:
                # filter invisible vehicles
                continue

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            # noinspection PyUnresolvedReferences
            cv2.fillPoly(img, [pts], ii + 1.0)

        return torch.Tensor(img).unsqueeze(0), torch.Tensor(convert_egopose_to_matrix_numpy(egopose))

    def get_seg_bev(self, lrtlist_cam, vislist):
        B, N, D = lrtlist_cam.shape
        assert (B == 1)

        seg = np.zeros((self.Z, self.X))
        val = np.ones((self.Z, self.X))

        corners_cam = utils.geom.get_xyzlist_from_lrtlist(lrtlist_cam)  # B, N, 8, 3
        y_cam = corners_cam[:, :, :, 1]  # y part; B, N, 8
        corners_mem = self.vox_util.Ref2Mem(corners_cam.reshape(B, N * 8, 3), self.Z, self.Y, self.X).reshape(B, N, 8,
                                                                                                              3)

        # take the xz part
        corners_mem = torch.stack([corners_mem[:, :, :, 0], corners_mem[:, :, :, 2]], dim=3)  # B, N, 8, 2
        # corners_mem = corners_mem[:,:,:4] # take the bottom four

        for n in range(N):
            _, inds = torch.topk(y_cam[0, n], 4, largest=False)  # returns indices of the 4 smallest values
            pts = corners_mem[0, n, inds].numpy().astype(np.int32)  # 4, 2

            # if this messes in some later conditions,
            # the solution is to draw all combos
            pts = np.stack([pts[0], pts[1], pts[3], pts[2]])

            # pts[:, [1, 0]] = pts[:, [0, 1]]
            # noinspection PyUnresolvedReferences
            cv2.fillPoly(seg, [pts], n + 1.0)

            if vislist[n] == 0:
                # draw a black rectangle if it's invisible
                # noinspection PyUnresolvedReferences
                cv2.fillPoly(val, [pts], 0.0)

        return torch.Tensor(seg).unsqueeze(0), torch.Tensor(val).unsqueeze(0)  # 1, Z, X

    def get_lrtlist(self, rec):
        egopose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        lrtlist = []
        boxlist = []
        vislist = []
        tidlist = []
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)

            # NuScenes filter
            if 'vehicle' not in inst['category_name']:
                continue
            if int(inst['visibility_token']) == 1:
                vislist.append(torch.tensor(0.0))  # invisible
            else:
                vislist.append(torch.tensor(1.0))  # visible

            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            tidlist.append(inst['instance_token'])

            # print('rotation', inst['rotation'])
            r = box.rotation_matrix
            t = box.center
            l = box.wlh
            l = np.stack([l[1], l[0], l[2]])
            lrt = utils.py.merge_lrt(l, utils.py.merge_rt(r, t))
            lrt = torch.Tensor(lrt)
            lrtlist.append(lrt)
            ry, _, _ = Quaternion(inst['rotation']).yaw_pitch_roll
            rs = np.stack([ry * 0, ry, ry * 0])
            box_ = torch.from_numpy(np.stack([t, l, rs])).reshape(9)

            boxlist.append(box_)
        if len(lrtlist):
            lrtlist = torch.stack(lrtlist, dim=0)
            boxlist = torch.stack(boxlist, dim=0)
            vislist = torch.stack(vislist, dim=0)
        else:
            lrtlist = torch.zeros((0, 19))
            boxlist = torch.zeros((0, 9))
            vislist = torch.zeros(0)
            tidlist = []

        return lrtlist, boxlist, vislist, tidlist

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.indices)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)

        Z, Y, X = self.res_3d
        self.vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=torch.from_numpy(self.centroid).float().cuda(),
            bounds=self.bounds,
            assert_cube=False)
        self.Z, self.Y, self.X = Z, Y, X

    def get_single_item(self, index, cams, refcam_id: int = None):
        rec = self.ixes[index]

        # ---------- 图像 ----------
        imgs, rots, trans, intrins = self.get_image_data(rec, cams)

        # ---------- 交换 ref-cam ----------
        if refcam_id is None:
            refcam_id = np.random.randint(1, len(cams)) if self.is_train else self.refcam_id
        for tensor in (imgs, rots, trans, intrins):
            tensor[0], tensor[refcam_id] = tensor[refcam_id].clone(), tensor[0].clone()

        # ---------- 地图 patch（可选，可按需删除） ----------
        scene  = self.nusc.get('scene', rec['scene_token'])
        log    = self.nusc.get('log', scene['log_token'])
        scene2map = {scene['name']: log['location']}

        car_from_current         = np.eye(4)
        car_from_current[:3, :3] = rots[0].numpy()
        car_from_current[:3, 3]  = trans[0].numpy()

        map_mask, egocar_bev = fetch_nusc_map2(
            rec, self.nusc_maps, self.nusc, scene2map, car_from_current
        )
        bev_map_mask     = torch.tensor(map_mask)                 # (7,200,200)
        bev_map_rgba     = get_rgba_map_from_mask2(map_mask, 0.8, 1.0)
        egocar_bev_tensor = torch.tensor(egocar_bev).unsqueeze(0) # (1,200,200)

        # ---------- 雷达 ----------
        radar_np = self.get_radar_data(rec, nsweeps=self.nsweeps)  # (19,V) np
        radar_np = radar_np.T                                      # (V,19)
        V_target = 700 * self.nsweeps
        radar_np = (radar_np[:V_target]
                    if radar_np.shape[0] > V_target
                    else np.pad(radar_np, [(0, V_target - radar_np.shape[0]), (0, 0)]))
        radar_data = torch.from_numpy(radar_np.T).float()          # (19,V_target)

        # ---------- GT Boxes & Labels ----------
        NAME2CLSID = {
            'car':0,'truck':1,'bus':2,'trailer':3,'construction':4,
            'pedestrian':5,'motorcycle':6,'bicycle':7,'traffic_cone':8,'barrier':9
        }
        gt_boxes, gt_labels = [], []
        for tok in rec['anns']:
            ann   = self.nusc.get('sample_annotation', tok)
            cname = ann['category_name'].split('.')[0]
            if cname not in NAME2CLSID:          # 跳过不关心类别
                continue
            # box 格式 [x,y,z,w,l,h,ry] + 额外 vx,vy=0
            box = np.array(ann['translation'] + ann['size'] + [Quaternion(ann['rotation']).yaw] + [0.,0.],
                           dtype=np.float32)
            gt_boxes.append(box)
            gt_labels.append(NAME2CLSID[cname])
        if len(gt_boxes) > 0:
            gt_boxes  = torch.tensor(gt_boxes)          # (N,9)  这里只有 9 维，用 TransFusionHead 的 encode_box 时再补 vx,vy,vz
            gt_labels = torch.tensor(gt_labels).long()  # (N,)
        else:
            gt_boxes  = torch.zeros((0,9), dtype=torch.float32)
            gt_labels = torch.zeros((0,),  dtype=torch.long)

        # ---------- (可选) Radar-VoxelNet ----------
        if self.radar_encoder_type == "voxel_net":
            rad_T   = radar_data.permute(1, 0)             # (V,19)
            xyz_rad = rad_T[:, :3]
            meta_rad = rad_T[:, 3:] if not self.use_shallow_metadata else rad_T[:, 5:8]

            velo_T_cams = utils.geom.merge_rt(rots, trans)         # (S,4,4)
            cams_T_velo = utils.geom.safe_inverse(velo_T_cams)     # (S,4,4)
            rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[0:1], xyz_rad.unsqueeze(0))

            voxel_in, voxel_coor, num_occ = self.vox_util.voxelize_xyz_and_feats_voxelnet(
                rad_xyz_cam0, meta_rad.unsqueeze(0), self.Z, self.Y, self.X,
                assert_cube=False, use_radar_occupancy_map=self.use_radar_occupancy_map
            )
            return (imgs, rots, trans, intrins,
                    radar_data,
                    bev_map_mask, bev_map_rgba, egocar_bev_tensor,
                    gt_boxes, gt_labels,
                    voxel_in.squeeze(0), voxel_coor.squeeze(0), num_occ.squeeze(0))

        # ---------- 默认返回（无 voxel_net） ----------
        return (imgs, rots, trans, intrins,
                radar_data,
                bev_map_mask, bev_map_rgba, egocar_bev_tensor,
                gt_boxes, gt_labels)

    def __getitem__(self, index):
        cams = self.choose_cams()
        if self.is_train and self.do_shuffle_cams and len(cams) > 1:
            refcam_id = np.random.randint(1, len(cams))
        else:
            refcam_id = self.refcam_id

        # ---------- 预分配存放 ----------
        imgs_l, rots_l, trans_l, intrins_l = [], [], [], []
        radar_l, mapmask_l, maprgba_l, ego_l = [], [], [], []
        gt_boxes_l, gt_labels_l = [], []

        vox_in_l, vox_coor_l, vox_num_l = [], [], []      # 仅 voxel_net 时用

        sample_indices = (self.indices[index][1:self.indices[index][0]+1]
                          if self.vis_full_scenes else self.indices[index])

        for idx_t in sample_indices:
            if self.radar_encoder_type == "voxel_net":
                (imgs, rots, trans, intrins,
                 radar_data, map_mask, map_rgba, ego_bev,
                 gt_boxes, gt_labels,
                 vox_in, vox_coor, vox_num) = self.get_single_item(idx_t, cams, refcam_id)

                vox_in_l.append(vox_in)
                vox_coor_l.append(vox_coor)
                vox_num_l.append(vox_num)
            else:
                (imgs, rots, trans, intrins,
                 radar_data, map_mask, map_rgba, ego_bev,
                 gt_boxes, gt_labels) = self.get_single_item(idx_t, cams, refcam_id)

            imgs_l.append(imgs);        rots_l.append(rots);       trans_l.append(trans);     intrins_l.append(intrins)
            radar_l.append(radar_data); mapmask_l.append(map_mask); maprgba_l.append(map_rgba); ego_l.append(ego_bev)
            gt_boxes_l.append(gt_boxes);gt_labels_l.append(gt_labels)

        # ---------- stack ----------
        all_imgs     = torch.stack(imgs_l)
        all_rots     = torch.stack(rots_l)
        all_trans    = torch.stack(trans_l)
        all_intrins  = torch.stack(intrins_l)
        all_radar    = torch.stack(radar_l)
        all_mapmask  = torch.stack(mapmask_l)
        all_maprgba  = torch.stack(maprgba_l)
        all_ego      = torch.stack(ego_l)

        # gt_boxes / labels 用 list 保持变长，不 stack
        if self.radar_encoder_type == "voxel_net":
            return (all_imgs, all_rots, all_trans, all_intrins,
                all_radar, all_mapmask, all_maprgba, all_ego,
                torch.stack(vox_in_l), torch.stack(vox_coor_l), torch.stack(vox_num_l),
                gt_boxes_l, gt_labels_l)            # ← 挪到最后
        else:
            return (all_imgs, all_rots, all_trans, all_intrins,
                all_radar, all_mapmask, all_maprgba, all_ego,
                gt_boxes_l, gt_labels_l)


def worker_rnd_init(x):
    np.random.seed(13 + x)

def describe_batch_structure(loader, max_items=1):
    """
    打印 DataLoader 中前 max_items 个 batch 的元素顺序、类型与形状。
    不需要知道 Dataset __getitem__ 的返回格式。
    """
    it = iter(loader)
    for b in range(max_items):
        try:
            batch = next(it)
        except StopIteration:
            print("DataLoader 里只有", b, "个 batch")
            return

        print("\n", "="*50)
        print(f"Batch {b}  (batch_size = {loader.batch_size})")
        print("="*50)

        # 如果 Dataset 返回 dict，可转成 list 保证顺序稳定（Python3.7+ dict 保序）
        if isinstance(batch, dict):
            items = list(batch.items())
        else:
            items = list(enumerate(batch))   # 元组 / 列表

        for key, value in items:
            # key 可能是 str 也可能是 int，这里都转成 str 打印
            label = str(key)
            if torch.is_tensor(value):
                print(f"[{label:>2}] tensor       shape {tuple(value.shape)}")
            elif isinstance(value, list):
                # 如果 list 内含 tensor，顺便列出子元素形状
                elem_shapes = [tuple(x.shape) if torch.is_tensor(x) else type(x)
                               for x in value]
                print(f"[{label:>2}] list(len={len(value)})  elem_shapes {elem_shapes}")
            elif isinstance(value, dict):
                print(f"[{label:>2}] dict(len={len(value)})")
            else:
                print(f"[{label:>2}] {type(value)}")

# ---------- 运行 ----------


def my_seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    # print("Worker seed: ", worker_seed, " Worker_id: ", worker_id)

    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def compile_data(version, dataroot, data_aug_conf, centroid, bounds, res_3d, bsz,
                 nworkers, radar_encoder_type=None, shuffle=True, nsweeps=1, nworkers_val=1, seqlen=1, refcam_id=1,
                 get_tids=False, temporal_aug=False, use_radar_filters=False, use_shallow_metadata=True,
                 do_shuffle_cams=True, distributed_sampler=False, rank=None, use_pre_scaled_imgs=True,
                 custom_dataroot=None, use_obj_layer_only_on_map=False, vis_full_scenes=False,
                 use_radar_occupancy_map=False, do_drn_val_split=False, get_val_day=False, get_val_rain=False,
                 get_val_night=False):

    print_details = False
    if rank == 0 or rank is None:
        print_details = True
        print('loading nuscenes...')
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=dataroot,
                    verbose=print_details)
    if print_details:
        print('loading Maps...')
    nusc_maps = get_nusc_maps(map_folder=dataroot)

    traindata = VizData(
        nusc,
        nusc_maps=nusc_maps,
        is_train=True,
        data_aug_conf=data_aug_conf,
        nsweeps=nsweeps,
        centroid=centroid,
        bounds=bounds,
        res_3d=res_3d,
        seqlen=seqlen,
        refcam_id=refcam_id,
        get_tids=get_tids,
        temporal_aug=temporal_aug,
        use_radar_filters=use_radar_filters,
        do_shuffle_cams=do_shuffle_cams,
        radar_encoder_type=radar_encoder_type,
        use_shallow_metadata=use_shallow_metadata,
        use_pre_scaled_imgs=use_pre_scaled_imgs,
        custom_dataroot=custom_dataroot,
        use_obj_layer_only_on_map=use_obj_layer_only_on_map,
        vis_full_scenes=vis_full_scenes,
        use_radar_occupancy_map=use_radar_occupancy_map,
        print_details=print_details
    )
    valdata = VizData(
        nusc,
        nusc_maps=nusc_maps,
        is_train=False,
        data_aug_conf=data_aug_conf,
        nsweeps=nsweeps,
        centroid=centroid,
        bounds=bounds,
        res_3d=res_3d,
        seqlen=seqlen,
        refcam_id=refcam_id,
        get_tids=get_tids,
        temporal_aug=False,
        use_radar_filters=use_radar_filters,
        do_shuffle_cams=False,
        radar_encoder_type=radar_encoder_type,
        use_shallow_metadata=use_shallow_metadata,
        use_pre_scaled_imgs=use_pre_scaled_imgs,
        custom_dataroot=custom_dataroot,
        use_obj_layer_only_on_map=use_obj_layer_only_on_map,
        vis_full_scenes=vis_full_scenes,
        use_radar_occupancy_map=use_radar_occupancy_map,
        do_drn_val_split=do_drn_val_split,
        get_val_day=get_val_day,
        get_val_rain=get_val_rain,
        get_val_night=get_val_night,
        print_details=print_details
    )

    # for distributed data loading
    if distributed_sampler:

        g = torch.Generator()
        g.manual_seed(125 + rank)

        distributed_train_sampler = DistributedSampler(dataset=traindata, shuffle=shuffle)
        distributed_val_sampler = DistributedSampler(dataset=valdata, shuffle=False)

        trainloader = torch.utils.data.DataLoader(
            traindata,
            batch_size=bsz,
            shuffle=False,  # formerly true  --> now handled by distributed sampler
            sampler=distributed_train_sampler,
            num_workers=nworkers,
            drop_last=True,
            worker_init_fn=my_seed_worker,
            generator=g,
            pin_memory=False)
        valloader = torch.utils.data.DataLoader(
            valdata,
            batch_size=bsz,
            shuffle=False,  # formerly true
            sampler=distributed_val_sampler,
            num_workers=nworkers_val,
            drop_last=True,
            pin_memory=False)
    else:
        trainloader = torch.utils.data.DataLoader(
            traindata,
            batch_size=bsz,
            shuffle=shuffle,  # formerly true
            num_workers=nworkers,
            drop_last=True,
            worker_init_fn=worker_rnd_init,
            pin_memory=False)
        valloader = torch.utils.data.DataLoader(
            valdata,
            batch_size=bsz,
            shuffle=shuffle,  # formerly true
            num_workers=nworkers_val,
            drop_last=True,
            pin_memory=False)

    if print_details:
        print('data ready')
    print('trainloader', trainloader)
    describe_batch_structure(trainloader)
    return trainloader, valloader
