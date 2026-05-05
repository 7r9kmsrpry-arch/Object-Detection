# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.pipelines import to_tensor
from ..builder import PIPELINES


@PIPELINES.register_module()
class DefaultFormatBundle(object):
    """デフォルトの整形処理をまとめたクラス。

    "img"、"proposals"、"gt_bboxes"、"gt_labels"、"gt_masks"、
    "gt_semantic_seg" などの共通フィールドに対する整形処理を簡略化する。

    resultの中には色んなデータ形式が混じっているので、モデルが扱いやすいように以下の処理を行う。
    tensor にできるものは tensor にする。
    CPU 上のまま持つべきものは cpu_only=True
    バッチ化時に stack したいものは stack=True

    これらのフィールドは、以下のように整形される。

    - img: (1) 軸を入れ替える, (2) tensor に変換する, (3) DataContainer(stack=True) に変換する
    - proposals: (1) tensor に変換する, (2) DataContainer に変換する
    - gt_bboxes: (1) tensor に変換する, (2) DataContainer に変換する
    - gt_bboxes_ignore: (1) tensor に変換する, (2) DataContainer に変換する
    - gt_labels: (1) tensor に変換する, (2) DataContainer に変換する
    - gt_masks: (1) tensor に変換する, (2) DataContainer(cpu_only=True) に変換する
    - gt_semantic_seg: (1) 0次元目を追加する, (2) tensor に変換する,
                       (3) DataContainer(stack=True) に変換する
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """results 内の共通フィールドを変換し、既定の形式に整形する。

        Args:
            results (dict): 変換対象のデータを格納した辞書。

        Returns:
            dict: 既定の bundle 形式に整形されたデータを格納した辞書。
        """

        # 画像の整形
        if 'img' in results:
            if isinstance(results['img'], list): # 1サンプル内に複数画像(FRONT, Rightなど)がある場合
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']] # HWC→CHWに変換
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))       # 画像郡を1つにまとめる
                results['img'] = DC(to_tensor(imgs), stack=True)          # テンソル化して、DataContainerに入れる
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=True)

        # 共通フィールドの整形
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list): # listの場合
                results[key] = DC([to_tensor(res) for res in results[key]]) # 各要素をTensor化して、listのままDataContainerに入れる
            else:
                results[key] = DC(to_tensor(results[key]))
        
        # GTbboxの整形
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes): # 専用クラスの場合
                results['gt_bboxes_3d'] = DC(results['gt_bboxes_3d'], cpu_only=True) # tensor化せずに、cpuで保持
            else:
                results['gt_bboxes_3d'] = DC(to_tensor(results['gt_bboxes_3d'])) # tensor化

        # 関係ないのでskip
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class Collect3D(object):
    """特定タスクに必要なデータを loader の出力から収集する。

        通常、これはデータローダpipelineの最後の段階で使用される。
        一般に keys には、"img"、"proposals"、"gt_bboxes"、"gt_bboxes_ignore"、"gt_labels"、"gt_masks" などの一部を指定する。

        "img_meta" 項目は常に生成される。
        "img_meta" 辞書の中身は "meta_keys" に依存する。
        デフォルトでは以下の情報を含む。

            - 'img_shape': ネットワーク入力画像の形状。
                (h, w, c) のタプルで表される。
                バッチテンソルのサイズがこれより大きい場合、画像は右端や下端にゼロパディングされることがある。
            - 'scale_factor': 前処理で適用されたスケール倍率を表す float 値
            - 'flip': 画像反転変換が適用されたかどうかを示す bool 値
            - 'filename': 画像ファイルのパス
            - 'ori_shape': 元画像の形状を表す (h, w, c) のタプル
            - 'pad_shape': パディング後の画像形状
            - 'lidar2img': LiDAR座標から画像座標への変換
            - 'depth2img': depth座標から画像座標への変換
            - 'cam2img': カメラ座標から画像座標への変換
            - 'pcd_horizontal_flip': 点群が水平方向に反転されたかを示す bool 値
            - 'pcd_vertical_flip': 点群が垂直方向に反転されたかを示す bool 値
            - 'box_mode_3d': 3D bbox の表現モード
            - 'box_type_3d': 3D bbox の型
            - 'img_norm_cfg': 正規化情報を格納した辞書
                - mean: チャネルごとの平均減算値
                - std: チャネルごとの標準偏差による除算値
                - to_rgb: bgr から rgb に変換したかどうかを示す bool 値
            - 'pcd_trans': 点群に適用された変換情報
            - 'sample_idx': サンプルのインデックス
            - 'pcd_scale_factor': 点群のスケール倍率
            - 'pcd_rotation': 点群に適用された回転
            - 'pts_filename': 点群ファイルのパス

        Args:
            keys (Sequence[str]): ``data`` に収集する results 内のキー。
            meta_keys (Sequence[str], optional): ``mmcv.DataContainer`` に変換して
                ``data['img_metas']`` に格納するメタ情報のキー。
                デフォルトは
                ('filename', 'ori_shape', 'img_shape', 'lidar2img',
                'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """

    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug')):
        

        self.keys = keys           # モデルに直接渡すデータ
        self.meta_keys = meta_keys # img_metasにまとめたいデータ

    def __call__(self, results):
        """results から必要なキーを収集する。

        ``meta_keys`` に含まれるキーは :obj:`mmcv.DataContainer` に変換される。

        Args:
            results (dict): 収集対象のデータを格納した辞書。

        Returns:
            dict: 以下のキーを含む辞書を返す。
                - ``self.keys`` で指定されたキー
                - ``img_metas``
        """

        # 空の辞書の作成
        data = {}
        img_metas = {}

        # meta_keysにある情報を集める
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        # img_metasをDCで包む
        # DataLoaderにCPUで扱うことと、バッチテンソルとして扱わなくていいいことを明示
        data['img_metas'] = DC(img_metas, cpu_only=True)

        # keysのデータを集める
        for key in self.keys:
            data[key] = results[key]

        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


@PIPELINES.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):
    """デフォルトの整形処理をまとめたクラス。

    voxels に関連する共通フィールド
    "proposals"、"gt_bboxes"、"gt_labels"、"gt_masks"、"gt_semantic_seg" などの整形処理を簡略化する。

    これらのフィールドは、以下のように整形される。

    - img: (1) 軸を入れ替える, (2) tensor に変換する, (3) DataContainer(stack=True) に変換する
    - proposals: (1) tensor に変換する, (2) DataContainer に変換する
    - gt_bboxes: (1) tensor に変換する, (2) DataContainer に変換する
    - gt_bboxes_ignore: (1) tensor に変換する, (2) DataContainer に変換する
    - gt_labels: (1) tensor に変換する, (2) DataContainer に変換する
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """results 内の共通フィールドを変換し、3Dデータを含む既定の形式に整形する。

        Args:
            results (dict): 変換対象のデータを格納した辞書。

        Returns:
            dict: 既定の bundle 形式に整形されたデータを格納した辞書。
        """

        # 点群データの整形
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        # Voxel関連データの整形
        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        # GTがある場合の処理
        if self.with_gt:
            # 不要なGT情報の削除
            if 'gt_bboxes_3d_mask' in results: # 3D GT-box
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results: # 2D GT-box
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            
            # クラス名からラベル名に変換
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0: # GTが空の場合→空配列を作成する
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(results['gt_names'][0], list): # GTがカメラごとにある場合→カメラごとに名前をIDに変換する
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [np.array([self.class_names.index(n) for n in res], dtype=np.int64) for res in results['gt_names']]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([self.class_names.index(n) for n in results['gt_names']], dtype=np.int64) # 通常のリストの場合→普通に数字に変換する
                
                # 3D bbox用
                # we still assume one pipeline for one frame LiDAR thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([self.class_names.index(n) for n in results['gt_names_3d']], dtype=np.int64)
        
        # 3D以外のデータの処理を親クラスのメソッドで実行
        results = super(DefaultFormatBundle3D, self).__call__(results)
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
