# Copyright (c) OpenMMLab. All rights reserved.
import platform

from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS as MMDET_DATASETS
from mmdet.datasets.builder import _concat_dataset

# Linux系でのresource設定(同時に開けるファイル数の上限を増やす)
if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

# Registryの定義
OBJECTSAMPLERS = Registry('Object sampler') # GTデータベースからのサンプリング用
DATASETS = Registry('dataset')              # Datasetクラス
PIPELINES = Registry('pipeline')            # 前処理パイプライン


def build_dataset(cfg, default_args=None):
    """
    Configに記載のdataset設定を読み込んで、対応するDatasetオブジェクトを作成する。

    Args:
        cfg (dict): datasetの設定辞書
        default_args(): 

    Returns:
        dataset(): 
    """

    # ラッパー系クラスのimport
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset, ConcatDataset, RepeatDataset)

    # datasetの作成
    if isinstance(cfg, (list, tuple)): 
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset': # 今回はこれ
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args)) # CBGSデータセットを作成
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    elif cfg['type'] in DATASETS._module_dict.keys(): # gt_databaseではここに入る
        dataset = build_from_cfg(cfg, DATASETS, default_args)
    else:
        dataset = build_from_cfg(cfg, MMDET_DATASETS, default_args)
    return dataset
