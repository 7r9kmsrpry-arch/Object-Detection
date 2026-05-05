# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from copy import deepcopy

import mmcv

from ..builder import PIPELINES
from .compose import Compose


@PIPELINES.register_module()
class MultiScaleFlipAug:
    """Test-time augmentation with multiple scales and flipping. An example
    configuration is as followed:

    .. code-block::
        img_scale=[(1333, 400), (1333, 800)],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:
    .. code-block::
        dict(
            img=[...],
            img_shape=[...],
            scale=[(1333, 400), (1333, 400), (1333, 800), (1333, 800)]
            flip=[False, True, False, True]
            ...
        )
    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float | list[float] | None): Scale factors for resizing.
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Default:
            "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale=None,
                 scale_factor=None,
                 flip=False,
                 flip_direction='horizontal'):
        self.transforms = Compose(transforms)
        assert (img_scale is None) ^ (scale_factor is None), (
            'Must have but only one variable can be set')
        if img_scale is not None:
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]
            self.scale_key = 'scale'
            assert mmcv.is_list_of(self.img_scale, tuple)
        else:
            self.img_scale = scale_factor if isinstance(
                scale_factor, list) else [scale_factor]
            self.scale_key = 'scale_factor'

        self.flip = flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([t['type'] == 'RandomFlip' for t in transforms])):
            warnings.warn(
                'flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.
        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """

        aug_data = []
        flip_args = [(False, None)]
        if self.flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.img_scale:
            for flip, direction in flip_args:
                _results = results.copy()
                _results[self.scale_key] = scale
                _results['flip'] = flip
                _results['flip_direction'] = direction
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str


@PIPELINES.register_module()
class MultiScaleFlipAug3D(object):
    """複数スケールおよび反転を用いたテスト時オーグメンテーション。

    Args:
        transforms (list[dict]): 各オーグメンテーションで適用する変換処理のリスト。
        img_scale (tuple | list[tuple]): リサイズ時に使用する画像スケール。
        pts_scale_ratio (float | list[float]): 点群をリサイズする際のスケール比。
        flip (bool, optional): 反転オーグメンテーションを適用するかどうか。デフォルトは False。
        flip_direction (str | list[str], optional): 画像に適用する反転方向。
            "horizontal" と "vertical" を指定可能。
            list を指定した場合は、複数の反転オーグメンテーションを適用する。
            ``flip == False`` の場合、この設定は無効。
            デフォルトは "horizontal"。
        pcd_horizontal_flip (bool, optional): 点群に水平方向の反転を適用するかどうか。
            デフォルトは True。
            ただし、'flip' が有効な場合のみ動作する。
        pcd_vertical_flip (bool, optional): 点群に垂直方向の反転を適用するかどうか。
            デフォルトは True。
            ただし、'flip' が有効な場合のみ動作する。
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 pts_scale_ratio,
                 flip=False,
                 flip_direction='horizontal',
                 pcd_horizontal_flip=False,
                 pcd_vertical_flip=False):
        
        # 処理パイプラインの作成
        self.transforms = Compose(transforms)

        # メンバ変数の初期化
        # img_scaleとpts_scaleをリストにする(複数対応のため)
        self.img_scale = img_scale if isinstance(img_scale, list) else [img_scale]
        self.pts_scale_ratio = pts_scale_ratio if isinstance(pts_scale_ratio, list) else [float(pts_scale_ratio)]
        # 型チェック
        assert mmcv.is_list_of(self.img_scale, tuple)
        assert mmcv.is_list_of(self.pts_scale_ratio, float)

        # flip関連のフラグを保存(デフォルトでflipの実施なし)
        self.flip = flip
        self.pcd_horizontal_flip = pcd_horizontal_flip
        self.pcd_vertical_flip = pcd_vertical_flip
        # flip_directionを必ずリストに設定(複数対応のため)
        self.flip_direction = flip_direction if isinstance(flip_direction, list) else [flip_direction]
        assert mmcv.is_list_of(self.flip_direction, str)
        if not self.flip and self.flip_direction != ['horizontal']:
            warnings.warn('flip_direction has no effect when flip is set to False')
        if (self.flip and not any([(t['type'] == 'RandomFlip3D' or t['type'] == 'RandomFlip') for t in transforms])):
            warnings.warn('flip has no effect when RandomFlip is not in transforms')

    def __call__(self, results):
        """results内の共通フィールドに対してテスト時オーグメンテーションを適用する。

        Args:
            results (dict): オーグメンテーション対象のデータを格納した辞書。

        Returns:
            dict: 異なるスケール・反転条件でオーグメンテーションされたデータを、各キーごとにリスト化してまとめた辞書。
        """

        # 各オーグメンテーションの結果を格納するリストの初期化
        aug_data = []

        # modified from `flip_aug = [False, True] if self.flip else [False]`
        # to reduce unnecessary scenes when using double flip augmentation
        # during test time
        # flip関連の設定
        flip_aug = [True] if self.flip else [False]
        pcd_horizontal_flip_aug = [False, True] if self.flip and self.pcd_horizontal_flip else [False]
        pcd_vertical_flip_aug = [False, True] if self.flip and self.pcd_vertical_flip else [False]
        
        # 全組み合わせを総当りで実施
        for scale in self.img_scale:
            for pts_scale_ratio in self.pts_scale_ratio:
                for flip in flip_aug:
                    for pcd_horizontal_flip in pcd_horizontal_flip_aug:
                        for pcd_vertical_flip in pcd_vertical_flip_aug:
                            for direction in self.flip_direction:
                                # results.copy will cause bug
                                # since it is shallow copy
                                _results = deepcopy(results) 
                                _results['scale'] = scale
                                _results['flip'] = flip
                                _results['pcd_scale_factor'] = pts_scale_ratio
                                _results['flip_direction'] = direction
                                _results['pcd_horizontal_flip'] = pcd_horizontal_flip
                                _results['pcd_vertical_flip'] = pcd_vertical_flip
                                data = self.transforms(_results)
                                aug_data.append(data)

        # list of dict to dict of list
        # 出力の整形
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
                
        return aug_data_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}, '
        repr_str += f'img_scale={self.img_scale}, flip={self.flip}, '
        repr_str += f'pts_scale_ratio={self.pts_scale_ratio}, '
        repr_str += f'flip_direction={self.flip_direction})'
        return repr_str
