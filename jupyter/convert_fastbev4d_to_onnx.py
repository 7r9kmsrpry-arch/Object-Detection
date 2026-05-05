# # ========================================
# # ライブラリの import
# # ========================================
# # TRT系はコメントアウトされている
# import argparse

# import torch.onnx
# from onnxsim import simplify # onnxグラフの簡略化

# from mmcv import Config # Configファイルの読み込み
# # from mmdeploy.backend.tensorrt.utils import save, search_cuda_version

# # mmdetection/mmdet3d系の読み込み
# try:
#     # If mmdet version > 2.23.0, compat_cfg would be imported and
#     # used from mmdet instead of mmdet3d.
#     from mmdet.utils import compat_cfg
# except ImportError:
#     from mmdet3d.utils import compat_cfg

# import os
# from typing import Dict, Optional, Sequence, Union

# import h5py
# import mmcv
# import numpy as np
# import onnx
# # import pycuda.driver as cuda
# # import tensorrt as trt
# import torch
# import tqdm
# from mmcv.runner import load_checkpoint
# # from mmdeploy.apis.core import no_mp
# # from mmdeploy.backend.tensorrt.calib_utils import HDF5Calibrator
# # from mmdeploy.backend.tensorrt.init_plugins import load_tensorrt_plugin
# # from mmdeploy.utils import load_config
# from packaging import version
# from torch.utils.data import DataLoader

# from mmdet3d.datasets import build_dataloader, build_dataset
# from mmdet3d.models import build_model
# from mmdet.datasets import replace_ImageToTensor
# from tools.misc.fuse_conv_bn import fuse_module


# import torch.nn as nn
# class FastBEV4DExportWrapper(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model

#     def forward(self, bev_feat):
#         all_cls_scores, all_bbox_preds = self.model(bev_feat)

#         return all_cls_scores, all_bbox_preds

# class BevEncoderOnly(nn.Module):
#     def __init__(self, bev_encoder):
#         super().__init__()
#         self.bev_encoder = bev_encoder

#     def forward(self, bev_feat):
#         return self.bev_encoder(bev_feat)

# import traceback

# # ========================================
# # 関数定義
# # ========================================
# def parse_args():
#     """
#     コマンドライン引数の読み込み
#     """
#     parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
#     parser.add_argument('config', help='deploy config file path')
#     parser.add_argument('checkpoint', help='checkpoint file')
#     parser.add_argument('work_dir', help='work dir to save file')
#     parser.add_argument(
#         '--prefix', default='fastbev', help='prefix of the save file name')
#     parser.add_argument(
#         '--fp16', action='store_true', help='Whether to use tensorrt fp16')
#     parser.add_argument(
#         '--int8', action='store_true', help='Whether to use tensorrt int8')
#     parser.add_argument(
#         '--fuse-conv-bn',
#         action='store_true',
#         help='Whether to fuse conv and bn, this will slightly increase'
#         'the inference speed')
#     args = parser.parse_args()
#     return args


# def get_plugin_names():
#     return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


# def create_calib_input_data_impl(calib_file: str,
#                                  dataloader: DataLoader,
#                                  model_partition: bool = False,
#                                  metas: list = []) -> None:
#     """
#     2つの数値を加算する。

#     Args:
#         a: 1つ目の数値
#         b: 2つ目の数値

#     Returns:
#         加算結果
#     """
#     with h5py.File(calib_file, mode='w') as file:
#         calib_data_group = file.create_group('calib_data')
#         assert not model_partition
#         # create end2end group
#         input_data_group = calib_data_group.create_group('end2end')
#         input_group_img = input_data_group.create_group('img')
#         input_keys = [
#             'ranks_bev', 'ranks_depth', 'ranks_feat', 'interval_starts',
#             'interval_lengths'
#         ]
#         input_groups = []
#         for input_key in input_keys:
#             input_groups.append(input_data_group.create_group(input_key))
#         metas = [
#             metas[i].int().detach().cpu().numpy() for i in range(len(metas))
#         ]
#         for data_id, input_data in enumerate(tqdm.tqdm(dataloader)):
#             # save end2end data
#             input_tensor = input_data['img_inputs'][0][0]
#             input_ndarray = input_tensor.squeeze(0).detach().cpu().numpy()
#             # print(input_ndarray.shape, input_ndarray.dtype)
#             input_group_img.create_dataset(
#                 str(data_id),
#                 shape=input_ndarray.shape,
#                 compression='gzip',
#                 compression_opts=4,
#                 data=input_ndarray)
#             for kid, input_key in enumerate(input_keys):
#                 input_groups[kid].create_dataset(
#                     str(data_id),
#                     shape=metas[kid].shape,
#                     compression='gzip',
#                     compression_opts=4,
#                     data=metas[kid])
#             file.flush()


# def create_calib_input_data(calib_file: str,
#                             deploy_cfg: Union[str, mmcv.Config],
#                             model_cfg: Union[str, mmcv.Config],
#                             model_checkpoint: Optional[str] = None,
#                             dataset_cfg: Optional[Union[str,
#                                                         mmcv.Config]] = None,
#                             dataset_type: str = 'val',
#                             device: str = 'cpu',
#                             metas: list = [None]) -> None:
#     """Create dataset for post-training quantization.

#     Args:
#         calib_file (str): The output calibration data file.
#         deploy_cfg (str | mmcv.Config): Deployment config file or
#             Config object.
#         model_cfg (str | mmcv.Config): Model config file or Config object.
#         model_checkpoint (str): A checkpoint path of PyTorch model,
#             defaults to `None`.
#         dataset_cfg (Optional[Union[str, mmcv.Config]], optional): Model
#             config to provide calibration dataset. If none, use `model_cfg`
#             as the dataset config. Defaults to None.
#         dataset_type (str, optional): The dataset type. Defaults to 'val'.
#         device (str, optional): Device to create dataset. Defaults to 'cpu'.
#     """
#     with no_mp():
#         if dataset_cfg is None:
#             dataset_cfg = model_cfg

#         # load cfg if necessary
#         deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

#         if dataset_cfg is None:
#             dataset_cfg = model_cfg

#         # load dataset_cfg if necessary
#         dataset_cfg = load_config(dataset_cfg)[0]

#         from mmdeploy.apis.utils import build_task_processor
#         task_processor = build_task_processor(model_cfg, deploy_cfg, device)

#         dataset = task_processor.build_dataset(dataset_cfg, dataset_type)

#         dataloader = task_processor.build_dataloader(
#             dataset, 1, 1, dist=False, shuffle=False)

#         create_calib_input_data_impl(
#             calib_file, dataloader, model_partition=False, metas=metas)

# def main():
#     # コマンドライン引数を受け取る
#     args = parse_args()

#     # onnxファイルの保存先ディレクトリの作成
#     if not os.path.exists(args.work_dir):
#         os.makedirs(args.work_dir)

#     if args.int8:
#         assert args.fp16

#     # 出力モデル名の接頭辞を設定
#     model_prefix = args.prefix
#     if args.int8:
#         model_prefix = model_prefix + '_int8'
#     elif args.fp16:
#         model_prefix = model_prefix + '_fp16'
    
#     # configファイルの読み込み
#     cfg = Config.fromfile(args.config)
    
#     # 事前学習済み重みを読み込まない(後で学習済み重みをloadする)
#     cfg.model.pretrained = None
    
#     # モデル名をTRT用に変更
#     cfg.model.type = "FastBEV4DTRT"
    
#     # MMCV系のバージョン互換処理
#     cfg = compat_cfg(cfg)
    
#     # 使用GPUを設定
#     cfg.gpu_ids = [0]
    
#     # データローダーのデフォルト設定
#     test_dataloader_default_args = dict(samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)
    
#     # データローダーの処理の変更
#     if isinstance(cfg.data.test, dict):
#         cfg.data.test.test_mode = True
#         if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
#             # Replace 'ImageToTensor' to 'DefaultFormatBundle'
#             cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
#     elif isinstance(cfg.data.test, list):
#         for ds_cfg in cfg.data.test:
#             ds_cfg.test_mode = True
#         if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
#             for ds_cfg in cfg.data.test:
#                 ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    
#     # データローダーの作成
#     test_loader_cfg = {
#         **test_dataloader_default_args,
#         **cfg.data.get('test_dataloader', {})
#     }
#     dataset = build_dataset(cfg.data.test)
#     data_loader = build_dataloader(dataset, **test_loader_cfg)
    
#     # モデルの作成＆重みのロード
#     cfg.model.train_cfg = None
#     cfg.model.img_view_transformer.accelerate = True # View変換をTRT用に高速化する
#     model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
#     load_checkpoint(model, args.checkpoint, map_location='cpu')
    
#     # fuse設定
#     if args.fuse_conv_bn:
#         model_prefix = model_prefix + '_fuse'
#         model = fuse_module(model)
    
#     # 評価モードに設定
#     model.cuda()
#     model.eval()

#     # import ipdb; ipdb.set_trace()
#     for i, data in enumerate(data_loader):
#         if i == 0:
#             continue
        
#         # FastBEV4DTRTの入力情報の作成
#         # 入力をフレームごとのlistに分解
#         # sensor2keyegos[0] : 現在フレーム、sensor2keyegos[1] : 過去1フレーム
#         inputs = [d.cuda() for d in data['img_inputs'][0]]
#         imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, _ = model.prepare_inputs(inputs)

#         # 出力格納用リストの作成
#         bev_feat_list = []

#         # 各フレームを順に処理
#         for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
#             # img_view_transformerへの補助入力の作成(Noneが返ってくる)
#             mlp_input = None
#             # 現フレーム分の入力をまとめる
#             inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input)
#             # BEV特徴量の作成
#             bev_feat, depth = model.prepare_bev_feat(*inputs_curr) 
#             # リストへ追加
#             bev_feat_list.append(bev_feat)

#         bev_feat_curr = bev_feat_list[0]
#         bev_feat_prev = bev_feat_list[1]

#         _, C, H, W = bev_feat_prev.shape
#         bev_feat_prev = model.shift_feature(bev_feat_prev, [sensor2keyegos[0], sensor2keyegos[1]], bda)
#         bev_feat_list = [bev_feat_curr]
#         num_frame = 2
#         bev_feat_list.append(bev_feat_prev.view(1, (model.num_frame - 1) * C, H, W))
#         bev_feats = torch.cat(bev_feat_list, dim=1)

#         # wrapper = FastBEV4DExportWrapper(model).cuda().eval()
#         wrapper_bev = BevEncoderOnly(model.bev_encoder).cuda().eval()
#         bev_feats = bev_feats.detach().float().contiguous()

#         print("bev_feat_curr.shape =", bev_feat_curr.shape)
#         print("bev_feat_prev.shape =", bev_feat_prev.shape)
#         print("bev_feats.shape     =", bev_feats.shape)

#         with torch.no_grad():
#             y = wrapper_bev(bev_feats)
#             print("wrapper_bev output shape =", y.shape)

#         # # ONNX export(FastBEVTRT)
#         # with torch.no_grad():
#         #     try:
#         #         torch.onnx.export(
#         #             wrapper_bev,
#         #             (bev_feats,),
#         #             "./onnx/bev_encoder_only.onnx",
#         #             opset_version=16,
#         #             input_names=["bev_feat"],
#         #             output_names=["bev_encoded"],
#         #             do_constant_folding=False,
#         #         )
#         #         print("export ok")
#         #     except Exception as e:
#         #         print("export failed:")
#         #         print(type(e).__name__, e)
#         #         traceback.print_exc()
#         break

#     # # ONNXモデルの検証(FastBEV4DTRT)
#     # #onnx_model_4d = onnx.load(args.work_dir + model_prefix + '_4d.onnx')
#     # onnx_model_4d = onnx.load("./onnx/bev_encoder_only.onnx")
#     # try:
#     #     onnx.checker.check_model(onnx_model_4d)
#     # except Exception:
#     #     print('ONNX Model(FastBEV4DTRT) Incorrect')
#     # else:
#     #     print('ONNX Model(FastBEV4DTRT) Correct')
#     # # # 計算グラフの簡素化
#     # # onnx_simp_4d, check_4d = simplify(onnx_model_4d)
#     # # assert check_4d, "Simplified ONNX model could not be validated"
#     # # # onnxファイルを上書き保存
#     # # onnx.save(onnx_simp_4d, args.work_dir + model_prefix + '_4d.onnx')
#     # print(f"🚀 The export is completed. ONNX save as {args.work_dir + model_prefix + '_4d.onnx'} 🤗, Have a nice day~")

# if __name__ == '__main__':

#     main()


# ========================================
# ライブラリの import
# ========================================
import argparse
import os
import traceback

import torch
import torch.onnx
import onnx

from mmcv import Config

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module

import torch.nn as nn


class BevEncoderOnly(nn.Module):
    def __init__(self, img_bev_encoder_backbone, img_bev_encoder_neck):
        super().__init__()
        self.img_bev_encoder_backbone = img_bev_encoder_backbone
        self.img_bev_encoder_neck = img_bev_encoder_neck

    def forward(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

import torch
import torch.nn as nn

class FastBEV4DExportWrapper(nn.Module):
    def __init__(self, img_bev_encoder_backbone, img_bev_encoder_neck, pts_bbox_head):
        super().__init__()
        self.img_bev_encoder_backbone = img_bev_encoder_backbone
        self.img_bev_encoder_neck = img_bev_encoder_neck
        self.pts_bbox_head = pts_bbox_head

    def forward(self, bev_feat):
        x = self.img_bev_encoder_backbone(bev_feat)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]

        outs = self.pts_bbox_head([x])

        cls_out = outs['all_cls_scores'][-1]
        bbox_out = outs['all_bbox_preds'][-1]

        # 入力依存を明示的に残すための診断用コード
        dummy = bev_feat.sum() * 0.0
        cls_out = cls_out + dummy
        bbox_out = bbox_out + dummy

        return cls_out, bbox_out


def parse_args():
    parser = argparse.ArgumentParser(description='Export BEV encoder to ONNX')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument('--prefix', default='fastbev', help='prefix of the save file name')
    parser.add_argument('--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument('--int8', action='store_true', help='Whether to use tensorrt int8')
    parser.add_argument('--fuse-conv-bn', action='store_true', help='Whether to fuse conv and bn')
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    if args.int8:
        assert args.fp16

    model_prefix = args.prefix
    if args.int8:
        model_prefix += '_int8'
    elif args.fp16:
        model_prefix += '_fp16'

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = "FastBEV4DTRT"
    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    test_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False
    )

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    cfg.model.img_view_transformer.accelerate = True

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model_prefix += '_fuse'
        model = fuse_module(model)

    model.cuda()
    model.eval()

    out_path = os.path.join(args.work_dir, "bev_encoder_only.onnx")
    if os.path.exists(out_path):
        os.remove(out_path)

    export_ok = False

    for i, data in enumerate(data_loader):
        if i == 0:
            continue

        inputs = [d.cuda() for d in data['img_inputs'][0]]
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, _ = model.prepare_inputs(inputs)

        bev_feat_list = []
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(
            imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans
        ):
            mlp_input = None
            inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input)
            bev_feat, _ = model.prepare_bev_feat(*inputs_curr)
            bev_feat_list.append(bev_feat)

        bev_feat_curr = bev_feat_list[0]
        bev_feat_prev = bev_feat_list[1]

        _, C, H, W = bev_feat_prev.shape
        bev_feat_prev = model.shift_feature(
            bev_feat_prev,
            [sensor2keyegos[0], sensor2keyegos[1]],
            bda
        )

        bev_feats = torch.cat(
            [
                bev_feat_curr,
                bev_feat_prev.view(1, (model.num_frame - 1) * C, H, W)
            ],
            dim=1
        ).detach().float().contiguous().cuda()

        #wrapper_bev = BevEncoderOnly(model.img_bev_encoder_backbone, model.img_bev_encoder_neck).cuda().eval()
        wrapper_4d = FastBEV4DExportWrapper(
            model.img_bev_encoder_backbone,
            model.img_bev_encoder_neck,
            model.pts_bbox_head,
        ).cuda().eval()

        out_path_4d = os.path.join(args.work_dir, "fastbev_4d.onnx")
        if os.path.exists(out_path_4d):
            os.remove(out_path_4d)

        with torch.no_grad():
            cls_out, bbox_out = wrapper_4d(bev_feats)
            print("PyTorch forward ok")
            print("cls_out.shape  =", cls_out.shape)
            print("bbox_out.shape =", bbox_out.shape)

        torch.onnx.export(
            model,
            (bev_feats,),
            out_path_4d,
            opset_version=16,
            input_names=["bev_feat"],
            output_names=["all_cls_scores", "all_bbox_preds"],
            do_constant_folding=False,
            verbose=False,
        )

        print("export ok")
        export_ok = True

        # print("bev_feat_curr.shape =", bev_feat_curr.shape)
        # print("bev_feat_prev.shape =", bev_feat_prev.shape)
        # print("bev_feats.shape     =", bev_feats.shape)

        # with torch.no_grad():
        #     y = wrapper_bev(bev_feats)
        #     print("PyTorch forward ok")
        #     print("wrapper_bev output type =", type(y))
        #     if isinstance(y, (list, tuple)):
        #         print("len(y) =", len(y))
        #         for j, v in enumerate(y):
        #             print(f"  [{j}] type={type(v)}, shape={getattr(v, 'shape', None)}")
        #     else:
        #         print("wrapper_bev output shape =", y.shape)

        # try:
        #     with torch.no_grad():
        #         torch.onnx.export(
        #             wrapper_bev,
        #             (bev_feats,),
        #             out_path,
        #             opset_version=16,
        #             input_names=["bev_feat"],
        #             output_names=["bev_encoded"],
        #             do_constant_folding=False,
        #             verbose=False,
        #         )
        #     export_ok = True
        #     print("export ok")
        # except Exception as e:
        #     print("export failed")
        #     print(type(e).__name__, e)
        #     traceback.print_exc()

        break

    # if not export_ok:
    #     print("ONNX was not generated.")
    #     return

    # onnx_model = onnx.load(out_path)
    # onnx.checker.check_model(onnx_model)
    # print("ONNX Model Correct")

    # print("graph inputs:")
    # for x in onnx_model.graph.input:
    #     print(" ", x.name)

    # print("graph outputs:")
    # for x in onnx_model.graph.output:
    #     print(" ", x.name)

    # print(f"saved: {out_path}")

    if not export_ok:
        print("ONNX was not generated.")
        return

    onnx_model = onnx.load(out_path_4d)
    onnx.checker.check_model(onnx_model)
    print("ONNX Model Correct")

    print("graph inputs:")
    for x in onnx_model.graph.input:
        print(" ", x.name)

    print("graph outputs:")
    for x in onnx_model.graph.output:
        print(" ", x.name)

    print(f"saved: {out_path_4d}")


if __name__ == '__main__':
    main()