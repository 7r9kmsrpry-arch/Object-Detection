# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadOccGTFromFile(object):
    def __call__(self, results):
        occ_gt_path = results['occ_gt_path']
        occ_gt_path = os.path.join(occ_gt_path, "labels.npz")

        occ_labels = np.load(occ_gt_path)
        semantics = occ_labels['semantics']
        mask_lidar = occ_labels['mask_lidar']
        mask_camera = occ_labels['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera
        return results


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(LoadImageFromFile):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """複数 sweep の点群を読み込む。

    主にnuScenesデータセットで、過去のsweepを活用するために使用される。

    Args:
        sweeps_num (int, optional): 使用するsweep数。デフォルトは10。
        load_dim (int, optional): 読み込む点群の次元数。デフォルトは5。
        use_dim (list[int], optional): 使用する次元。デフォルトは [0, 1, 2, 4]。
        time_dim (int, optional): 各点のタイムスタンプを表す次元番号。デフォルトは4。
        file_client_args (dict, optional): file client の設定辞書。
            詳細はhttps://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.pyを参照。
            デフォルトは dict(backend='disk')。
        pad_empty_sweeps (bool, optional): sweepが空の場合に、キーフレームを繰り返して埋めるかどうか。デフォルトはFalse。
        remove_close (bool, optional): 近距離の点を削除するかどうか。デフォルトは False。
        test_mode (bool, optional): `test_mode=True` の場合、sweepをランダムにサンプリングせず、最も近いNフレームを選択する。デフォルトは False。
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """原点から一定半径以内にある近すぎる点を削除する。

        Args:
            points (np.ndarray | :obj:`BasePoints`): sweep の点群。
            radius (float, optional): この半径未満の点を削除するための閾値。デフォルトは 1.0。

        Returns:
            np.ndarray: 近距離点を削除した後の点群。
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """複数sweepの点群ファイルを読み込み、結合する処理を実行する。

        Args:
            results (dict): 複数sweepの点群ファイル名を含む結果辞書。

        Returns:
            dict: 複数sweepを結合した点群データを含む結果辞書。追加・更新されるキーと値は以下の通り。
                - points (np.ndarray | :obj:`BasePoints`): 複数 sweep を結合した点群配列
        """

        # 現在フレームの点群に対する処理
        points = results['points'] # 現在フレームの点群を取得
        points.tensor[:, self.time_dim] = 0 # タイムスタンプを０に設定

        # 結合用リストを初期化
        sweep_points_list = [points]
        # 現在フレームの時間(秒)を取得
        ts = results['timestamp']

        # sweepの処理
        if self.pad_empty_sweeps and len(results['sweeps']) == 0: # sweepsがない場合、現在フレームの結果で埋める
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            # sweepの選択
            if len(results['sweeps']) <= self.sweeps_num: # sweeps数が少ない場合、全て使用
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode: # テストモード時=sweep_num個を決定論的に使用
                choices = np.arange(self.sweeps_num)
            else: # 学習時：ランダムに取得
                choices = np.random.choice(len(results['sweeps']), self.sweeps_num, replace=False)
            
            # 各sweepを処理
            for idx in choices:
                # sweep点群の読み込み
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                # 近距離点群の削除
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                # タイムスタンプの取得
                sweep_ts = sweep['timestamp'] / 1e6
                # 現在フレーム基準に座標変換
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                # 時間差を格納
                points_sweep[:, self.time_dim] = ts - sweep_ts
                # 点群オブジェクトに変換し、リストに追加
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        # 全結合
        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """ファイルから点群を読み込む。

    点群ファイルからデータを読み込む処理クラス。

    Args:
        coord_type (str): 点群の座標系の種類。
            利用可能な選択肢は以下の通り。
            - 'LIDAR': LiDAR座標系の点群
            - 'DEPTH': Depth座標系の点群（通常は屋内データセット向け）
            - 'CAMERA': カメラ座標系の点群
        load_dim (int, optional): 読み込む点群の次元数。デフォルトは 6。
        use_dim (list[int], optional): 点群のうち実際に使用する次元。
            デフォルトは [0, 1, 2]。
            KITTI データセットでは、intensity次元を使うために、use_dim=4 or use_dim=[0, 1, 2, 3]を設定する。
        shift_height (bool, optional): 高さをシフトした特徴を使うかどうか。デフォルトは False。
        use_color (bool, optional): 色特徴を使うかどうか。デフォルトは False。
        file_client_args (dict, optional): file client の設定辞書。
            詳細はhttps://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.pyを参照。
            デフォルトは dict(backend='disk')。
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        
        # メンバ変数の設定
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        """点群データを読み込む内部関数。

        Args:
            pts_filename (str): 点群データファイルのファイル名。

        Returns:
            np.ndarray: 点群データを含む配列。
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points

    def __call__(self, results):
        """点群ファイルから点群データを読み込む処理を実行する。

        Args:
            results (dict): 点群データに関する情報を含む結果辞書。

        Returns:
            dict: 点群データを含む結果辞書。追加されるキーと値は以下の通り。
                - points (:obj:`BasePoints`): 点群データ
        """

        # 点群データの読み込み
        pts_filename = results['pts_filename'] # 点群ファイルパスの取得
        points = self._load_points(pts_filename)

        # 読み込んだ点群データを整形(N, use_dim)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]
        attribute_dims = None

        # 高さ情報の追加(skip)
        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        # 色情報の追加(skip)
        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        # info(result)にpointsデータを追加
        points_class = get_points_type(self.coord_type) # LiDARPoints
        points = points_class(points, points_dim=points.shape[-1], attribute_dims=attribute_dims) # ndarray→Pointsオブジェクトに変換
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)

            lidarego2global = np.eye(4, dtype=np.float32)
            lidarego2global[:3, :3] = Quaternion(
                results['curr']['ego2global_rotation']).rotation_matrix
            lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            lidarego2global = torch.from_numpy(lidarego2global)

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            camego2global = np.eye(4, dtype=np.float32)
            camego2global[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['ego2global_rotation']).rotation_matrix
            camego2global[:3, 3] = results['curr']['cams'][cam_name][
                'ego2global_translation']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
                lidarego2global.matmul(lidar2lidarego))
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results


@PIPELINES.register_module()
class PointToMultiViewDepthFusion(PointToMultiViewDepth):
    def __call__(self, results):
        points_camego_aug = results['points'].tensor[:, :3]
        # print(points_lidar.shape)
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        points_camego = points_camego_aug - bda[:3, 3].view(1,3)
        points_camego = points_camego.matmul(torch.inverse(bda[:3,:3]).T)

        depth_map_list = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]

            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            camego2img = cam2img.matmul(torch.inverse(cam2camego))

            points_img = points_camego.matmul(
                camego2img[:3, :3].T) + camego2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map = self.points2depthmap(points_img, imgs.shape[2],
                                             imgs.shape[3])
            depth_map_list.append(depth_map)
        depth_map = torch.stack(depth_map_list)
        results['gt_depth'] = depth_map
        return results

def mmlabNormalize(img):
    """入力画像を正規化し、CHWに順番を入れ替える。"""

    from mmcv.image.photometric import imnormalize
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32) # ImageNet事前学習系の正規化値
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)    # ImageNet事前学習系の正規化値
    to_rgb = True # OpenCV経由の場合、デフォルトでBGRになっているが、PIlowの場合は不要
    img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img


@PIPELINES.register_module()
class PrepareImageInputs(object):
    """複数チャネル画像入力を、個別ファイルのリストから読み込んで準備する。

    `results['img_filename']` には、ファイル名のリストが入っていることを想定する。

    Args:
        to_float32 (bool): 画像をfloat32に変換するかどうか。デフォルトは False。
        color_type (str): 画像ファイルの色形式。デフォルトは'unchanged'。
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        opencv_pp=False,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.opencv_pp = opencv_pp

        # debug
        print(f"opencv_pp:{self.opencv_pp}")

    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        """画像変換を行い、対応する post-homography 変換も更新する。

        画像に対してリサイズ・クロップ・反転・回転を適用し、
        それに対応する2D平面上の変換行列(元画像上の点がオーグメンテーション後画像のどこに移るかを表す行列)`post_rot`と並進`post_tran`もあわせて更新する。

        Args:
            img: 入力画像。
            post_rot (torch.Tensor): 画像平面上の後段変換用2x2回転・スケール行列。
            post_tran (torch.Tensor): 画像平面上の後段変換用2次元並進ベクトル。
            resize (float): リサイズ倍率。
            resize_dims (tuple[int, int]): リサイズ後画像サイズ (W, H)。
            crop (tuple[int, int, int, int]): クロップ領域 (left, top, right, bottom)。
            flip (bool): 左右反転を行うかどうか。
            rotate (float): 回転角度（度）。

        Returns:
            tuple:
                - img: 変換後画像
                - post_rot: 更新後の 2x2 変換行列
                - post_tran: 更新後の 2次元並進ベクトル
        """

        # 画像オーグメンテーションの実施
        if not self.opencv_pp: # デフォルトでここに入る
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # オーグメンテーションの内容を行列に反映
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
        if self.opencv_pp:
            img = self.img_transform_core_opencv(img, post_rot, post_tran, crop)
        return img, post_rot, post_tran

    def img_transform_core_opencv(self, img, post_rot, post_tran,
                                  crop):
        img = np.array(img).astype(np.float32)
        img = cv2.warpAffine(img,
                             np.concatenate([post_rot,
                                            post_tran.reshape(2,1)],
                                            axis=1),
                             (crop[2]-crop[0], crop[3]-crop[1]),
                             flags=cv2.INTER_LINEAR)
        return img

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        """画像に対して基本的な幾何変換を適用する。

        画像を以下の順で変換する。
        1. リサイズ
        2. クロップ
        3. 左右反転
        4. 回転

        Args:
            img: 入力画像。
            resize_dims (tuple[int, int]): リサイズ後画像サイズ (W, H)。
            crop (tuple[int, int, int, int]): クロップ領域 (left, top, right, bottom)。
            flip (bool): 左右反転するかどうか。
            rotate (float): 回転角度（度）。

        Returns:
            変換後の画像。
        """

        # リサイズ
        img = img.resize(resize_dims)
        # クロップ
        img = img.crop(crop)
        # 反転
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        # 回転
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        """
        使用するカメラを選択する。

        Returns:
            cam_names (list[str]): 使用するカメラ名のリスト。
        """

        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']

        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        """画像のリサイズ・クロップ・反転・回転の augmentation パラメータを決定する。

        学習時はランダムaugmentationをサンプリングし、テスト時は決定的な前処理パラメータを返す。

        Args:
            H (int): 元画像の高さ。
            W (int): 元画像の幅。
            flip (bool, optional): テスト時に明示的に左右反転を指定するかどうか。デフォルトは None。
            scale (float, optional): テスト時に追加のリサイズ量を指定する値。デフォルトは None。

        Returns:
            tuple:
                - resize (float): リサイズ倍率
                - resize_dims (tuple[int, int]): リサイズ後の画像サイズ (W, H)
                - crop (tuple[int, int, int, int]): クロップ領域 (left, top, right, bottom)
                - flip (bool): 左右反転するかどうか
                - rotate (float): 回転角度
        """

        # モデルへの最終的な入力画像サイズを取得
        fH, fW = self.data_config['input_size'] 

        # オーグメンテーションの適用
        # アスペクト比を保ってリサイズ→必要な領域をクロップする(学習時には余分に大きくリサイズしてクロップする)
        if self.is_train: # 学習時
            # リサイズ倍率の決定
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            # リサイズ後の画像サイズの決定
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            # クロップ高さの決定
            random_crop_height = self.data_config.get('random_crop_height', False) # False
            if random_crop_height:
                crop_h = int(np.random.uniform(max(0.3*newH, newH-fH), newH-fH))
            else:
                crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH # 常に下側基準で採用
            # クロップ幅の決定
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            # クロップ領域の決定
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # 左右反転フラグの設定(50%で実施)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            # 回転角度のサンプリング
            rotate = np.random.uniform(*self.data_config['rot'])
            # 垂直反転(デフォルトでは実施なし)
            if self.data_config.get('vflip', False) and np.random.choice([0, 1]):
                rotate += 180
        else: # テスト時
            # リサイズ率の設定
            resize = float(fW) / float(W)
            if scale is not None:
                resize += scale
            else:
                resize += self.data_config.get('resize_test', 0.0)
            # リサイズ後の画像サイズの決定
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            # クロップ領域の設定
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            # フリップの設定
            flip = False if flip is None else flip
            # 回転の設定
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor_transforms(self, cam_info, cam_name):
        """指定したカメラの座標変換行列を取得する。

        `cam_info` に含まれるカメラ情報から、指定したカメラ`cam_name`について、以下の4x4同次変換行列を作成して返す。
        - sensor座標系 → ego座標系
        - ego座標系 → global座標系

        Args:
            cam_info (dict): カメラ情報を含む辞書。
            cam_name (str): 対象カメラ名。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - sensor2ego: sensor座標系からego座標系への4x4変換行列
                - ego2global: ego座標系からglobal座標系への4x4変換行列
        """

        w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
        # sweep sensor to sweep ego
        sensor2ego_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(
            cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(
            Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(
            cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        return sensor2ego, ego2global

    def photo_metric_distortion(self, img, pmd):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand()>pmd.get('rate', 1.0):
            return img

        img = np.array(img).astype(np.float32)
        assert img.dtype == np.float32, \
            'PhotoMetricDistortion needs the input image of dtype np.float32,' \
            ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-pmd['brightness_delta'],
                                   pmd['brightness_delta'])
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if np.random.randint(2):
            img[..., 1] *= np.random.uniform(pmd['saturation_lower'],
                                          pmd['saturation_upper'])

        # random hue
        if np.random.randint(2):
            img[..., 0] += np.random.uniform(-pmd['hue_delta'], pmd['hue_delta'])
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if np.random.randint(2):
                alpha = np.random.uniform(pmd['contrast_lower'],
                                       pmd['contrast_upper'])
                img *= alpha

        # randomly swap channels
        if np.random.randint(2):
            img = img[..., np.random.permutation(3)]
        return Image.fromarray(img.astype(np.uint8))

    def get_inputs(self, results, flip=None, scale=None):
        """画像に対する前処理入力を作成する関数。

        Args:
            results (dict): 読み込みパイプラインから渡される結果辞書。

        Returns:
            dict: 前処理後の画像情報を含む結果辞書。
        """

        # 出力用の配列を作成
        imgs = []        # 画像
        sensor2egos = [] # センサー→自車
        ego2globals = [] # 自車→グローバル
        intrins = []     # 内部パラメータ
        # 元画像上の点が、オーグメンテーション後画像のどこに移るかを表す行列
        post_rots = []   # 回転
        post_trans = []  # 平行移動

        # 使用するカメラ名を選択する
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names

        # 可視化用の生画像の配列
        canvas = []

        for cam_name in cam_names:
            # 現在フレームの画像を取得
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 回転と並進情報を格納する行列を作成
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # 内部パラメータの取得
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # センサー→ego、ego→globalの変換行列を取得
            sensor2ego, ego2global = self.get_sensor_transforms(results['curr'], cam_name)

            # オーグメンテーションの適用
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # 回転と並進行列を扱いやすいように3×3行列に変換
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # 画像の明るさなどの見た目の変換(デフォルトでは実施なし)
            if self.is_train and self.data_config.get('pmd', None) is not None:
                img = self.photo_metric_distortion(img, self.data_config['pmd'])

            # 可視化用の画像配列に保存
            canvas.append(np.array(img))
            # 正規化を行い、学習用の画像配列に保存
            imgs.append(self.normalize_img(img))

            # 時系列処理
            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    # 隣接フレームの取得
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    # オーグメンテーションの実施
                    if self.opencv_pp:
                        img_adjacent = \
                            self.img_transform_core_opencv(
                                img_adjacent,
                                post_rot[:2, :2],
                                post_tran[:2],
                                crop)
                    else:
                        img_adjacent = self.img_transform_core(
                            img_adjacent,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=rotate)
                    # 隣接フレームを追加
                    imgs.append(self.normalize_img(img_adjacent))
            
            # 内部パラメータ行列などを格納
            intrins.append(intrin)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            for adj_info in results['adjacent']:
                # 過去フレーム分も各行列を複製
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # 過去フレームを整列させるための行列を取得(過去のフレームのegoやglobal基準)
                for cam_name in cam_names:
                    sensor2ego, ego2global = self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        # 各情報をスタックする
        imgs = torch.stack(imgs)

        sensor2egos = torch.stack(sensor2egos)
        ego2globals = torch.stack(ego2globals)
        intrins = torch.stack(intrins)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)

        # 描画用画像(現在フレームのオーグメンテーション後の各カメラ)を保存
        results['canvas'] = canvas
        
        return (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans)

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results


@PIPELINES.register_module()
class LoadAnnotations(object):
    """アノテーション情報を読み込む。"""

    def __call__(self, results):
        gt_boxes, gt_labels = results['ann_infos']
        gt_boxes, gt_labels = torch.Tensor(gt_boxes), torch.tensor(gt_labels)
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        # gt_bboxesのテンソルを3D box専用オブジェクトに変換
        # gt_boxesの中心は(x,y,z)の中心だが、LiDARインスタンスにした際に内部で中心zを底面(0)に変換される
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5)) 
        results['gt_labels_3d'] = gt_labels
        
        return results


@PIPELINES.register_module()
class BEVAug(object):
    """BEVオーグメンテーションを実施する。"""

    def __init__(self, bda_aug_conf, classes, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes

    def sample_bda_augmentation(self):
        """BEV Data Augmentation(BDA)の変換パラメータをサンプリングする。

        学習時は、設定された範囲に基づいて回転・拡大縮小・反転・並進のオーグメンテーション量をランダムに生成する。
        推論時はオーグメンテーションを行わないため、恒等変換に対応する値を返す。

        Returns:
            tuple:
                rotate_bda (float): BEV平面上での回転角。
                scale_bda (float): BEV平面上でのスケール倍率。
                flip_dx (bool): x軸方向に反転するかどうか。
                flip_dy (bool): y軸方向に反転するかどうか。
                tran_bda (ndarray): x, y, z方向の並進量。
        """

        # オーグメンテーションの値の決定
        if self.is_train: # 学習時
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])        # 回転
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])       # スケール
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']   # 水平反転
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']   # 垂直反転
            translation_std = self.bda_aug_conf.get('tran_lim', [0.0, 0.0, 0.0]) # 平行移動量(実施なし)
            tran_bda = np.random.normal(scale=translation_std, size=3).T         # 平行移動量(実施なし)
        else: # 推論時(恒等変換)
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
            tran_bda = np.zeros((1, 3), dtype=np.float32)
        return rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda

    def bev_transform(self, gt_boxes, rotate_angle, scale_ratio, flip_dx,
                      flip_dy, tran_bda):
        """BEVオーグメンテーションを3D GT bboxに適用する。

            回転・拡大縮小・反転・平行移動のパラメータに基づいて、GT bbox の位置、サイズ、yaw角、速度成分を更新する。
            あわせて、点群などにも共通で適用できる3✕3の変換行列を返す。

            Args:
                gt_boxes (Tensor): 3D GT bbox群。一般に各 bbox は [x, y, z, dx, dy, dz, yaw, vx, vy] の形式を想定する。
                rotate_angle (float): BEV平面上での回転角[deg]。
                scale_ratio (float): BEV平面上でのスケール倍率。
                flip_dx (bool): x軸方向に反転するかどうか。
                flip_dy (bool): y軸方向に反転するかどうか。
                tran_bda (ndarray or Tensor): x, y, z方向の平行移動量。

            Returns:
                tuple:
                    gt_boxes (Tensor): BEVオーグメンテーション適用後のGT bbox。
                    rot_mat (Tensor): 回転・拡大縮小・反転をまとめた3✕3変換行列。
        """

        # 回転行列の作成
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi) # 回転角度をラジアンに変換
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], 
                                [rot_sin,  rot_cos, 0],
                                [0,              0, 1]])
        
        # スケール行列の作成
        scale_mat = torch.Tensor([[scale_ratio,           0,           0], 
                                  [0,           scale_ratio,           0],
                                  [0,                     0, scale_ratio]])
        
        # 反転行列の作成
        flip_mat = torch.Tensor([[1, 0, 0], 
                                 [0, 1, 0], 
                                 [0, 0, 1]])

        if flip_dx: # 水平反転
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], 
                                                [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy: # 垂直反転
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], 
                                                [0, -1, 0],
                                                [0, 0, 1]])

        # 回転・拡大縮小・反転を合成
        rot_mat = flip_mat @ (scale_mat @ rot_mat)

        # GTboxに適用
        if gt_boxes.shape[0] > 0:
            gt_boxes[:, :3] = (rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1) # 中心座標の変換
            gt_boxes[:, 3:6] *= scale_ratio # サイズの変換
            gt_boxes[:, 6] += rotate_angle  # yaw角の更新
            # 反転時のyaw角補正
            if flip_dx:
                gt_boxes[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:, 6]
            if flip_dy:
                gt_boxes[:, 6] = -gt_boxes[:, 6]
            gt_boxes[:, 7:] = (rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1) # 速度ベクトルの変換
            gt_boxes[:, :3] = gt_boxes[:, :3] + tran_bda # 中心座標に平行移動を適用

        return gt_boxes, rot_mat

    def __call__(self, results):
        """BEV方向のデータ拡張をGT bbox・点群・関連情報に適用する。

        Args:
            results (dict): 学習用サンプルの情報を格納した辞書。
                主に以下のキーを含む。
                - gt_bboxes_3d: 3D GT bbox
                - points: 点群
                - img_inputs: 画像入力と各種変換情報
                - voxel_semantics, mask_lidar, mask_camera: ボクセル関連情報

        Returns:
            dict: BEVオーグメンテーション適用後の results。
                GT bbox、点群、変換行列、必要に応じて voxel_semanticsなどが更新された状態で返される。
        """
        # GT bbox情報の抽出
        gt_boxes = results['gt_bboxes_3d'].tensor
        gt_boxes[:,2] = gt_boxes[:,2] + 0.5*gt_boxes[:,5] # 中心のZを物体中心に補正

        # BEVオーグメンテーションのパラメータを取得
        rotate_bda, scale_bda, flip_dx, flip_dy, tran_bda = self.sample_bda_augmentation()

        # BEVオーグメンテーションの4×4の同次行列の作成
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1

        # GTbboxにBEVオーグメンテーションを適用
        gt_boxes, bda_rot = self.bev_transform(gt_boxes, rotate_bda, scale_bda,
                                               flip_dx, flip_dy, tran_bda)
        
        # 点群にもBEVオーグメンテーションを適用
        if 'points' in results:
            points = results['points'].tensor
            points_aug = (bda_rot @ points[:, :3].unsqueeze(-1)).squeeze(-1)
            points[:,:3] = points_aug + tran_bda
            points = results['points'].new_point(points)
            results['points'] = points
        
        # 同次行列に変換行列を保存
        bda_mat[:3, :3] = bda_rot
        bda_mat[:3, 3] = torch.from_numpy(tran_bda)

        # 変換後のGTboxを保存
        if len(gt_boxes) == 0:
            gt_boxes = torch.zeros(0, 9)
        results['gt_bboxes_3d'] = LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1], origin=(0.5, 0.5, 0.5))
        
        # img_inputsにbda行列を追加
        if 'img_inputs' in results:
            imgs, rots, trans, intrins = results['img_inputs'][:4]
            post_rots, post_trans = results['img_inputs'][4:]
            results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_mat)
        
        # voxel_semanticsは使用なしのため、skip
        if 'voxel_semantics' in results:
            if flip_dx:
                results['voxel_semantics'] = results['voxel_semantics'][::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][::-1,...].copy()
            if flip_dy:
                results['voxel_semantics'] = results['voxel_semantics'][:,::-1,...].copy()
                results['mask_lidar'] = results['mask_lidar'][:,::-1,...].copy()
                results['mask_camera'] = results['mask_camera'][:,::-1,...].copy()

        return results
