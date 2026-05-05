# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from mmdet3d.models.utils.grid_mask import GridMask
from mmdet.models.backbones.resnet import ResNet

# debug
import time
import torch
import json
import numpy as np

def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


@DETECTORS.register_module()
class FastBEV(CenterPoint):
    r"""FastBEV paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self,
                 img_view_transformer,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 use_grid_mask=False,
                 use_depth=False,
                 **kwargs):
        super(FastBEV, self).__init__(**kwargs)
        self.grid_mask = None if not use_grid_mask else \
            GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
                     prob=0.7)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        if img_bev_encoder_neck and img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = \
                builder.build_backbone(img_bev_encoder_backbone)
            self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck)
        self.use_depth = use_depth

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        if self.grid_mask is not None:
            imgs = self.grid_mask(imgs)
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]
        return x

    def prepare_inputs(self, inputs):
        # split the inputs into each frame
        assert len(inputs) == 7
        B, N, C, H, W = inputs[0].shape
        imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs

        sensor2egos = sensor2egos.view(B, N, 4, 4)
        ego2globals = ego2globals.view(B, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0,  ...].unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        return [imgs, sensor2keyegos, ego2globals, intrins,
                post_rots, post_trans, bda]

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        img = self.prepare_inputs(img)
        x, _ = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:8])
        x = self.bev_encoder(x)
        return [x], depth

    def extract_feat(self, points, img, img_metas, **kwargs):
        """
        画像および点群入力から特徴量を抽出する。

        Args:
            points:
                点群入力。現在の実装では未使用の場合がある。
            img:
                画像入力、または画像と幾何情報をまとめた入力。
            img_metas:
                各サンプルに対応するメタ情報。
            **kwargs:
                追加の引数。

        Returns:
            tuple:
                (img_feats, pts_feats, depth) を返す。
                - img_feats: 画像由来の特徴量
                - pts_feats: 点群由来の特徴量。ここでは None
                - depth: 深度関連の出力
        """

        # headに入力する特徴量の抽出
        img_feats, depth = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        if self.use_depth:
            gt_depth = kwargs['gt_depth']
            loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
            losses = dict(loss_depth=loss_depth)
        else:
            losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]):
                外側の list はテスト時拡張（augmentation）ごとの入れ物。
                内側の torch.Tensor は NxC 形状で、バッチ内の全点群を含む。

            img_metas (list[list[dict]]):  画像の前処理内容などの補助的な情報
                外側のlistはテスト時拡張（マルチスケール、フリップなど）ごとの入れ物。
                内側のlistはバッチ内画像ごとのメタ情報を表す。

            img_inputs (list[torch.Tensor], optional): 画像や変換行列
                外側の list はテスト時拡張ごとの入れ物。
                内側の torch.Tensor は NxCxHxW 形状で、バッチ内の全画像を含む。デフォルトは None。
                img_inputs = [[img, sensor2ego, ego2global, intrins, post_rots, post_trans, bda]]
        """

        # img_inputsとimg_metasがlistであることを確認する
        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))

        # テスト時拡張の個数を取得する
        num_augs = len(img_inputs)

        # img_inputsとimg_metasの拡張数が一致しているか確認する
        if num_augs != len(img_metas):
            raise ValueError('num of augmentations ({}) != num of image meta ({})'.format(len(img_inputs), len(img_metas)))

        # 1段だけの通常入力かどうかを確認する
        # → 通常推論の場合、simple_test
        # → 多段の拡張入力の場合、aug_test
        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0], **kwargs)
        else:
            return self.aug_test(None, img_metas[0], img_inputs[0], **kwargs)

    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        assert False

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """
        テスト時拡張を行わない通常推論を実行する。

        Args:
            points:
                点群入力。モデル構成によっては補助的に用いられる。
            img_metas:
                各サンプルに対応するメタ情報のリスト。
                画像サイズ、前処理情報、座標変換情報などを含む。
            img:
                画像入力、または画像と幾何情報をまとめた入力。
                実際の中身は pipeline / モデル実装に依存する。
            rescale:
                推論結果を元スケールに戻すかどうか。
            **kwargs:
                追加の引数。

        Returns:
            list[dict]:
                バッチ内各サンプルの推論結果を格納したリスト。
                各要素は辞書で、`pts_bbox` キーに3D物体検出結果を持つ。
        """

        # headに入力するための特徴マップを作成
        img_feats, _, _ = self.extract_feat(points, img=img, img_metas=img_metas, **kwargs)

        # 結果格納用の空の辞書を作成
        bbox_list = [dict() for _ in range(len(img_metas))]
        #print(f"len_ing_feats:{len(img_feats)}")
        #print(f"img_feats.shape:{img_feats[0].shape}")

        if self.debug:
            self.head_cnt += 1
            torch.cuda.synchronize()
            head_start = time.perf_counter()

        # head処理
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        
        if self.debug and self.head_cnt >= 10:
            torch.cuda.synchronize()
            self.head_time += (time.perf_counter() - head_start)*1000
            if self.head_cnt == 50:
                print(f"head_time:{self.head_time/self.head_cnt:.2f} ms")
        
        # 各バッチの検出結果を格納
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        img_feats, _, _ = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(img_feats)
        return outs



@DETECTORS.register_module()
class FastBEV4D(FastBEV):
    r"""FastBEV4D paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2203.17054>`_

    Args:
        pre_process (dict | None): Configuration dict of BEV pre-process net.
        align_after_view_transfromation (bool): Whether to align the BEV
            Feature after view transformation. By default, the BEV feature of
            the previous frame is aligned during the view transformation.
        num_adj (int): Number of adjacent frames.
        with_prev (bool): Whether to set the BEV feature of previous frame as
            all zero. By default, False.
    """
    def __init__(self,
                 pre_process=None,
                 align_after_view_transfromation=False,
                 num_adj=1,
                 with_prev=True,
                 pre_grad=False,
                 **kwargs):
        super(FastBEV4D, self).__init__(**kwargs)
        self.pre_process = pre_process is not None
        if self.pre_process:
            self.pre_process_net = builder.build_backbone(pre_process)
        self.align_after_view_transfromation = align_after_view_transfromation
        self.num_frame = num_adj + 1

        self.with_prev = with_prev
        self.pre_grad = pre_grad
        self.grid = None

        # debug用
        self.debug = True
        self.view_cnt   = 0
        self.img_cnt = 0
        self.head_cnt = 0
        self.bev_encoder_cnt = 0
        self.view_transform_time = 0.0
        self.bev_encoder_time    = 0.0
        self.img_encoder_time    = 0.0
        self.head_time           = 0.0
        self.first_frame = True
        self.frame_no = 0

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        n, c, h, w = input.shape
        _, v, _, _ = sensor2keyegos[0].shape
        if self.grid is None:
            # generate grid
            xs = torch.linspace(
                0, w - 1, w, dtype=input.dtype,
                device=input.device).view(1, w).expand(h, w)
            ys = torch.linspace(
                0, h - 1, h, dtype=input.dtype,
                device=input.device).view(h, 1).expand(h, w)
            grid = torch.stack((xs, ys, torch.ones_like(xs)), -1)
            self.grid = grid
        else:
            grid = self.grid
        grid = grid.view(1, h, w, 3).expand(n, h, w, 3).view(n, h, w, 3, 1)

        # get transformation from current ego frame to adjacent ego frame
        # transformation from current camera frame to current ego frame
        c02l0 = sensor2keyegos[0][:, 0:1, :, :]

        # transformation from adjacent camera frame to current ego frame
        c12l0 = sensor2keyegos[1][:, 0:1, :, :]

        # add bev data augmentation
        bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
        # bda_[:, :, :3, :3] = bda.unsqueeze(1)
        bda_[:, :, :, :] = bda.unsqueeze(1)
        bda_[:, :, 3, 3] = 1
        c02l0 = bda_.matmul(c02l0)
        if bda_adj is not None:
            bda_ = torch.zeros((n, 1, 4, 4), dtype=grid.dtype).to(grid)
            bda_[:, :, :3, :3] = bda_adj.unsqueeze(1)
            bda_[:, :, 3, 3] = 1
        c12l0 = bda_.matmul(c12l0)

        # transformation from current ego frame to adjacent ego frame
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(
            n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,
                      [True, True, False, True], :][:, :, :, :,
                                                    [True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.img_view_transformer.grid_interval[0]
        feat2bev[1, 1] = self.img_view_transformer.grid_interval[1]
        feat2bev[0, 2] = self.img_view_transformer.grid_lower_bound[0]
        feat2bev[1, 2] = self.img_view_transformer.grid_lower_bound[1]
        feat2bev[2, 2] = 1
        feat2bev = feat2bev.view(1, 3, 3)
        tf = torch.inverse(feat2bev).matmul(l02l1).matmul(feat2bev)

        # transform and normalize
        grid = tf.matmul(grid)
        normalize_factor = torch.tensor([w - 1.0, h - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        grid = grid[:, :, :, :2, 0] / normalize_factor.view(1, 1, 1, 2) * 2.0 - 1.0

        return grid

    @force_fp32()
    def shift_feature(self, input, sensor2keyegos, bda, bda_adj=None):
        grid = self.gen_grid(input, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(input, grid.to(input.dtype), align_corners=True)
        return output

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran, bda, mlp_input):
        """
        1フレーム分の画像と幾何情報から BEV 特徴および深度出力を生成する。

        Args:
            img:
                1フレーム分のマルチカメラ画像テンソル。
            rot:
                センサ座標から基準座標系への回転を含む変換情報。
            tran:
                センサ座標から基準座標系への並進を含む変換情報。
            intrin:
                カメラ内部パラメータ。
            post_rot:
                画像前処理後の回転補正。
            post_tran:
                画像前処理後の平行移動補正。
            bda:
                BEV augmentation 行列。
            mlp_input:
                view transformer 内部で使う幾何条件入力。

        Returns:
            tuple:
                - bev_feat: 1フレーム分の BEV 特徴
                - depth: 深度関連の出力
        """

        # if self.debug:
        #     self.img_cnt += 1
        #     torch.cuda.synchronize()
        #     img_start = time.perf_counter() 
        
        # 画像特徴量の抽出
        x, _ = self.image_encoder(img)

        # if self.debug and self.img_cnt >= 10:
        #     torch.cuda.synchronize()
        #     self.img_encoder_time += (time.perf_counter() - img_start)*1000
        #     if self.img_cnt == 50:
        #         print(f"img_encoder_time:{self.img_encoder_time/self.img_cnt:.2f} ms")

        #print("start img_view_transformer")
        # if self.debug:
        #     self.view_cnt += 1
        #     torch.cuda.synchronize()
        #     view_start = time.perf_counter() 

        # BEV特徴マップの生成
        bev_feat, depth = self.img_view_transformer([x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input])
        
        # BEV特徴量の整形
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        
        # if self.debug and self.view_cnt >= 10:
        #     torch.cuda.synchronize()
        #     self.view_transform_time += (time.perf_counter() - view_start)*1000
        #     if self.view_cnt == 50:
        #         print(f"view_transform_time:{self.view_transform_time/self.view_cnt:.2f} ms")

        return bev_feat, depth

    def extract_img_feat_sequential(self, inputs, feat_prev):
        """
        逐次処理モードで画像特徴を抽出する。

        現在フレームの画像から新しい BEV 特徴を生成し、
        過去フレームの BEV 特徴 `feat_prev` を現在フレーム基準へ整列したうえで結合し、
        最終的な BEV 特徴を生成する。

        Args:
            inputs:
                現在フレーム処理に必要な入力一式。(逐次処理用の中身になっている)
                通常は以下を含む。
                - 現在フレーム画像
                - 現在フレームの sensor2keyego
                - 現在フレームの ego2global
                - 内部パラメータ
                - 過去フレームの sensor2keyego
                - post_rots
                - post_trans
                - bda
            feat_prev:
                過去フレーム分の BEV 特徴。
                shape は通常 [(num_frame - 1), C, H, W] 相当。

        Returns:
            tuple:
                - [x]&#58; bev_encoder 後の最終特徴
                - depth: 現在フレームの深度出力
        """

        # 入力の分解
        imgs, sensor2keyegos_curr, ego2globals_curr, intrins = inputs[:4]
        sensor2keyegos_prev, _, post_rots, post_trans, bda = inputs[4:]

        # BEV特徴量の格納用リストの作成
        bev_feat_list = []
        
        # 現在フレーム用のmlp_inputを作成(None)
        mlp_input = self.img_view_transformer.get_mlp_input(sensor2keyegos_curr[0:1, ...], ego2globals_curr[0:1, ...], intrins, post_rots, post_trans, bda[0:1, ...])
        
        # 現在フレームの情報をまとめる
        # currも中身は全部現在フレームの情報だけど、過去フレーム数分並んでいる
        inputs_curr = (imgs, sensor2keyegos_curr[0:1, ...],
                       ego2globals_curr[0:1, ...], intrins, post_rots,
                       post_trans, bda[0:1, ...], mlp_input)

        # 現在フレームのみBEV化
        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
        bev_feat_list.append(bev_feat)

        # 過去の特徴マップを現在フレームにアライン
        _, C, H, W = feat_prev.shape
        feat_prev = self.shift_feature(feat_prev, [sensor2keyegos_curr, sensor2keyegos_prev], bda)
        bev_feat_list.append(feat_prev.view(1, (self.num_frame - 1) * C, H, W))

        # 全フレームをチャネル方向に結合し、BEVエンコーダーに入力
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)

        return [x], depth

    def prepare_inputs(self, inputs, stereo=False):
        """
        入力テンソル群をフレームごとに分割し、各時刻の画像・幾何情報をモデル内部で扱いやすい形に整形する。

        Args:
            inputs:
                pipeline から渡される画像系入力一式。
                通常は以下を含む。
                - inputs[0]: 画像テンソル
                - inputs[1]: sensor2ego
                - inputs[2]: ego2global
                - inputs[3]: intrins
                - inputs[4]: post_rots
                - inputs[5]: post_trans
                - inputs[6]: bda
            stereo:
                ステレオ／時系列対応用の追加変換
                `curr2adjsensor` を計算するかどうか。

        Returns:
            tuple:
                - imgs:
                    各フレームごとに分割した画像テンソルのリスト
                - sensor2keyegos:
                    各フレーム・各カメラの sensor → key ego 変換
                - ego2globals:
                    各フレーム・各カメラの ego → global 変換
                - intrins:
                    各フレーム・各カメラの内部パラメータ
                - post_rots:
                    各フレーム・各カメラの画像後処理回転
                - post_trans:
                    各フレーム・各カメラの画像後処理平行移動
                - bda:
                    BEV augmentation 行列
                - curr2adjsensor:
                    stereo=True のときに計算される、現在フレームから隣接フレームのセンサ座標系への変換。
                    stereo=False のときは None。
        """

        # 画像テンソルをフレームごとに分割
        B, N, C, H, W = inputs[0].shape # 画像のshapeの取得
        N = N // self.num_frame         # 1フレームあたりのカメラ数に戻す
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W) # 画像を[B, N, num_frame, C, H, W] に並べ替える
        imgs = torch.split(imgs, 1, 2)       # フレームごとに分割
        imgs = [t.squeeze(2) for t in imgs]  # リスト化(list[B, N, C, H, W])

        # 幾何的情報を抽出
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = inputs[1:7]

        # フレームごとに処理できるように、整形
        # sensor2egos[:, 0, ...] : 現在フレーム
        # sensor2egos[:, 1, ...] : 過去1フレーム
        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # sensor→key egoの変換行列を計算
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)          # keyフレーム(現在フレーム)のego→globalを計算 (実車環境時は、key=globalとして、前回フレームとの差分はcanを使うように変更する必要がある！)
        global2keyego = torch.inverse(keyego2global.double())                        # key→globalの逆行列(global→key)
        sensor2keyegos = global2keyego @ ego2globals.double() @ sensor2egos.double() # 各フレームのsensor→key ego変換行列を作成
        sensor2keyegos = sensor2keyegos.float()

        # ステレオ用の処理(skip)
        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = torch.inverse(ego2globals_adj @ sensor2egos_adj) @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        # 他の値もフレーム単位で操作しやすいように整形
        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3),
        ]

        # フレームごとのlistに分解
        # sensor2keyegos[0] : 現在フレーム、sensor2keyegos[1] : 過去1フレーム
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra

        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, curr2adjsensor

    def extract_img_feat(self,
                         img,
                         img_metas,
                         pred_prev=False,
                         sequential=False,
                         **kwargs):
        """
        画像入力から BEV 特徴を抽出する。

        Args:
            img:
                画像入力、または画像と幾何情報をまとめた入力。
            img_metas:
                各サンプルのメタ情報。
            pred_prev:
                過去フレーム特徴を別経路で返すかどうか。
                True のとき、過去フレーム特徴と、その整列に必要な情報を返す。
            sequential:
                時系列入力を逐次処理する専用経路を使うかどうか。
            **kwargs:
                追加引数。sequential=True のときは `feat_prev` などを含む。

        Returns:
            tuple:
                通常時は `([x], depth)` を返す。
                - [x]&#58; BEV encoder 後の最終画像特徴
                - depth: キーフレームに対応する深度出力

                pred_prev=True のときは、
                - 過去フレーム特徴
                - その整列に必要な情報一式
                を返す。
        """
        print(f"sequential:{sequential}")
        print(f"align:{self.align_after_view_transfromation}")
        print(f"pred_prev:{pred_prev}")

        # 逐次処理モード(前回の特徴を再利用する)
        if sequential:
            return self.extract_img_feat_sequential(img, kwargs['feat_prev'])

        # 入力をフレームごとのlistに分解
        # sensor2keyegos[0] : 現在フレーム、sensor2keyegos[1] : 過去1フレーム
        imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda, _ = self.prepare_inputs(img)

        # # debug
        # if self.first_frame:
        #     dump_meta = {
        #         "frame_0_current": {
        #             "sensor2keyegos": to_jsonable(sensor2keyegos[0]),
        #             "ego2globals": to_jsonable(ego2globals[0]),
        #             "intrins": to_jsonable(intrins[0]),
        #             "post_rots": to_jsonable(post_rots[0]),
        #             "post_trans": to_jsonable(post_trans[0]),
        #         },
        #         "frame_1_prev": {
        #             "sensor2keyegos": to_jsonable(sensor2keyegos[1]),
        #             "ego2globals": to_jsonable(ego2globals[1]),
        #             "intrins": to_jsonable(intrins[1]),
        #             "post_rots": to_jsonable(post_rots[1]),
        #             "post_trans": to_jsonable(post_trans[1]),
        #         },
        #         "bda": to_jsonable(bda),
        #     }
        #     with open("prepare_inputs_meta.json", "w", encoding="utf-8") as f:
        #         json.dump(dump_meta, f, ensure_ascii=False, indent=2)
        #     print("dump json")

        # 出力格納用リストの作成
        bev_feat_list = []
        depth_list = []

        # 各フレームを順に処理
        key_frame = True  # 学習時はキーフレーム(現在フレーム)のみを勾配対象とする
        for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
            if key_frame or self.with_prev: # 現在のみ、現在+過去フレームの両方に対応
                # 現在の姿勢でBEVを作り、後で整列させる
                if self.align_after_view_transfromation:
                    sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                
                # img_view_transformerへの補助入力の作成(Noneが返ってくる)
                mlp_input = self.img_view_transformer.get_mlp_input(sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                
                # 現フレーム分の入力をまとめる
                inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input)

                # BEV特徴量の作成
                if key_frame: # 現在フレーム
                    bev_feat, depth = self.prepare_bev_feat(*inputs_curr) 
                    #print(bev_feat)
                    # debug
                    #np.savetxt(f"{self.frame_no}_bev_feat_curr_origin.txt", bev_feat.detach().cpu().numpy().reshape(-1))
                else: # 過去フレーム
                    # TODO: bug here, need fixing
                    if self.pre_grad: # 過去フレームにも勾配を流す
                        bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                    else: # 過去フレームには勾配を流さない
                        with torch.no_grad():
                            bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
                    # debug
                    #np.savetxt(f"{self.frame_no}_bev_feat_prev_origin.txt", bev_feat.detach().cpu().numpy().reshape(-1))
                    
            else: # 過去フレームを使わない場合
                # 過去フレームをゼロ特徴で模擬
                bev_feat = torch.zeros_like(bev_feat_list[0])
                depth = None
            
            # リストへ追加
            bev_feat_list.append(bev_feat)
            depth_list.append(depth)
            key_frame = False
        
        # 過去フレーム特徴だけを返すモード(skip)
        if pred_prev:
            assert self.align_after_view_transfromation
            assert sensor2keyegos[0].shape[0] == 1
            feat_prev = torch.cat(bev_feat_list[1:], dim=0)
            ego2globals_curr = \
                ego2globals[0].repeat(self.num_frame - 1, 1, 1, 1)
            sensor2keyegos_curr = \
                sensor2keyegos[0].repeat(self.num_frame - 1, 1, 1, 1)
            ego2globals_prev = torch.cat(ego2globals[1:], dim=0)
            sensor2keyegos_prev = torch.cat(sensor2keyegos[1:], dim=0)
            bda_curr = bda.repeat(self.num_frame - 1, 1, 1)
            return feat_prev, [imgs[0],
                               sensor2keyegos_curr, ego2globals_curr,
                               intrins[0],
                               sensor2keyegos_prev, ego2globals_prev,
                               post_rots[0], post_trans[0],
                               bda_curr]
        
        # 過去フレームのBEV特徴を現在フレーム基準に整列
        if self.align_after_view_transfromation:
            for adj_id in range(1, self.num_frame):
                bev_feat_list[adj_id] = self.shift_feature(bev_feat_list[adj_id], [sensor2keyegos[0], sensor2keyegos[adj_id]], bda)
        
        # # debug
        # if self.debug:
        #     self.bev_encoder_cnt += 1
        #     torch.cuda.synchronize()
        #     bev_encoder_start = time.perf_counter() 

        # 全フレームをチャネル方向に結合し、BEVエンコーダーに入力
        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)

        # debug
        #print("bev_feat", x)

        # if self.debug and self.bev_encoder_cnt >= 10:
        #     torch.cuda.synchronize()
        #     self.bev_encoder_time += (time.perf_counter() - bev_encoder_start)*1000
        #     if self.bev_encoder_cnt == 50:
        #         print(f"bev_encoder_time:{self.bev_encoder_time/self.bev_encoder_cnt:.2f} ms")

        # debug
        self.first_frame = False
        self.frame_no += 1

        return [x], depth_list[0]


# @DETECTORS.register_module()
# class FastBEVTRT(FastBEV):

#     def result_serialize(self, outs):

#         # 出力用の空リストを作成
#         outs_ = []
#         for out in outs:
#             for key in ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']:
#                 outs_.append(out[0][key])
#         return outs_

#     def result_deserialize(self, outs):
#         outs_ = []
#         keys = ['reg', 'height', 'dim', 'rot', 'vel', 'heatmap']
#         for head_id in range(len(outs) // 6):
#             outs_head = [dict()]
#             for kid, key in enumerate(keys):
#                 outs_head[0][key] = outs[head_id * 6 + kid]
#             outs_.append(outs_head)
#         return outs_

#     def forward(
#         self,
#         img,
#         coors_img,
#         coors_depth=None
#     ):
#         """画像入力から BEV 特徴を生成し、3D物体検出 Head の出力を返す。

#         複数カメラ画像から画像特徴を抽出し、事前計算した`coors_img` / `coors_depth` を用いて 3D voxel 空間へ高速に再配置する。
#         その後、高さ方向Zを融合して2DのBEV特徴へ変換し、BEV encoderとdetection headを通して最終出力を得る。

#         処理の流れは以下の通り。
#             1. Backbone / Neck / depth_net により画像特徴を生成する
#             2. 特徴マップを `[B, C, H, W]` から `[B*H*W, C]` の表形式へ変換する
#             （ここでの B は実質的にカメラ台数）
#             3. どのカメラにも写らない 3D 座標用に、先頭要素をゼロ特徴へ置き換える
#             4. `coors_img` と `coors_depth` を用いて、各 voxel に対応する
#             画像特徴および深度重みを抽出する
#             5. `[num_voxel, C]` 形式の特徴列を `[1, X, Y, Z, C]` の voxel 格子へ整形する
#             6. 高さ方向 Z を `sum` / `max` / `s2c` のいずれかで融合し、
#             2D BEV 特徴 `[N, C, H, W]` へ変換する
#             7. downsample, bev_encoder, pts_bbox_head を通して検出出力を得る
#             8. Head の出力辞書を ONNX / TensorRT で扱いやすいテンソル列へ整形する

#         Args:
#             img (torch.Tensor):
#                 入力画像テンソル。
#                 形状は通常 `[N, C, H, W]` を想定する。
#                 ここで `N` はバッチサイズではなく、実質的にはカメラ台数として扱われる。
#             coors_img (torch.Tensor):
#                 各 3D voxel が参照する画像特徴位置を表す 1 次元 index 配列。
#                 `x = x[coors_img]` の形で用いられ、voxel 順に対応する画像特徴を取得する。
#             coors_depth (torch.Tensor, optional):
#                 各 3D voxel が参照する深度 bin の位置を表す 1 次元 index 配列。
#                 depth を用いる構成で使用する。
#                 depth を使わない構成では `None` でもよい。

#         Returns:
#             list[torch.Tensor]:
#                 Detection head の出力を直列化したテンソルのリスト。
#                 各 task について
#                 `reg`, `height`, `dim`, `rot`, `vel`, `heatmap`
#                 の順で格納される。

#         Notes:
#             - `img_view_transformer.depth_net` の出力チャネル数が
#             `D + out_channels` の場合は、前半を深度情報、後半を画像特徴として扱う。
#             - 先頭要素をゼロ特徴に置き換えることで、どのカメラにも写らない voxel に対して
#             ダミーの index 0 を安全に割り当てられるようにしている。
#             - `is_transpose` により、BEV 空間の `X/Y` と 2D 特徴マップの `H/W` の対応が変わる。
#             - `fuse_type` により、高さ方向 Z の融合方法が変わる。
#                 * `sum`: 高さ方向の特徴を総和
#                 * `max`: 高さ方向の特徴の最大値を使用
#                 * `s2c`: 高さ方向をチャネル方向へ並べて学習的に融合
#         """

#         # 画像特徴量の作成
#         img = self.img_backbone(img)
#         img = self.img_neck(img)
#         img = self.img_view_transformer.depth_net(img) # D+out_channels

#         # 特徴マップを画素ごとの表に変換
#         channel = img.shape[1]
#         img = img.permute(0, 2, 3, 1).reshape(-1, channel) # [N,C,H,W]→[N,H,W,C]→[N*H*W, C] (Bはカメラ台数)

#         # どのカメラにも映らない3D座標はダミーの0特徴に設定
#         img_0 = torch.zeros([1, img.shape[1]]).to(img.device) # [1, C(値0)] ←先頭カメラの左上画素の特徴量を0に設定(どのカメラにも映らない座標は先頭カメラの左隅の特徴を取るように設定してるので)
#         img_rest = img[1:] # 残りの要素
#         img = torch.cat([img_0, img_rest]) # ダミー+1要素目以降を結合

#         # depth情報の取り扱い        
#         if channel != self.img_view_transformer.out_channels: # depthありの場合
#             depth = img[:, :self.img_view_transformer.D].reshape(-1) # 前半Dチャネル(各画素の深度推定情報)を取り出す [B✕H✕W✕D]
#             x = img[:, self.img_view_transformer.D:(self.img_view_transformer.D + self.img_view_transformer.out_channels)] # 特徴マップ([B,C,H,W])
#             x = x[coors_img]                        # 3D座標の並びに対応した特徴マップを抽出
#             depth = depth[coors_depth].unsqueeze(1) # 3D座標の並びに対応した深度推定結果を抽出
#             x = x * depth                           # 対応する深度binの値で重み付け(深度を使わない場合、ここをコメントアウトする)
#         else:
#             x = img[coors_img] # [num_voxel, out_channels]

#         # 整形 ([num_voxel, out_channels]→[1, X, Y, Z, C])
#         x = x.view(1, *self.img_view_transformer.grid_size.int().tolist(), self.img_view_transformer.out_channels)
#         N, X, Y, Z, C = x.shape

#         # BEV座標系の向きによって、後段でX/YとH/Wの対応を変える
#         if self.img_view_transformer.is_transpose: # defaultでTrue
#             permute = [0, 3, 2, 1]
#         else:
#             permute = [0, 3, 1, 2]
        
#         # 高さ方向の結合
#         if self.img_view_transformer.fuse_type is not None:
#             if self.img_view_transformer.fuse_type == 's2c':
#                 x = x.reshape(N, X, Y, Z*C).permute(permute)
#                 x = self.img_view_transformer.fuse(x)
#             elif self.img_view_transformer.fuse_type == 'sum': # default
#                 x = x.sum(dim=-2).permute(permute) # 高さ方向の全特徴量を足し合わせて、並び替え [N,X,Y,C]→[N,C,Y,X]
#             elif self.img_view_transformer.fuse_type == 'max':
#                 x = x.max(dim=-2)[0].permute(permute)
#             else:
#                 raise NotImplemented
            
#             # ダウンサンプル
#             x = self.img_view_transformer.downsample(x) # [N, C', H', W']

#         # BEV特徴量の生成
#         bev_feat = self.bev_encoder(x)

#         # Head処理
#         outs = self.pts_bbox_head([bev_feat])

#         # Headの内容を抽出
#         outs = self.result_serialize(outs)

#         return outs

#     def get_bev_pool_input(self, input):
#         input = self.prepare_inputs(input)
#         coor = self.img_view_transformer.get_lidar_coor(*input[1:7])
#         return self.img_view_transformer.voxel_pooling_prepare_v2(coor)

