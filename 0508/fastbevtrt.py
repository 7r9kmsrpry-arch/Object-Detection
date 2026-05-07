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
from .fastbev import FastBEV4D

@DETECTORS.register_module()
class FastBEVTRT(FastBEV4D):
    """現在フレーム1枚からBEV特徴を生成する。

    前提:
        - batch size = 1
        - img.shape = [Ncam, C, H, W]
        - coors_img / coors_depth は get_fastray_input(..., accelerate=True) で事前計算済みの1次元index
    """

    def forward(self, img, coors_img, coors_depth):
        """画像入力から BEV 特徴を生成し、3D物体検出 Head の出力を返す。

        複数カメラ画像から画像特徴を抽出し、事前計算した`coors_img` / `coors_depth` を用いて 3D voxel 空間へ高速に再配置する。
        その後、高さ方向Zを融合して2DのBEV特徴へ変換し、BEV encoderとdetection headを通して最終出力を得る。

        Args:
            img (torch.Tensor):
                入力画像テンソル。
                形状は通常 `[N, C, H, W]` を想定する。
                ここで `N` はバッチサイズではなく、実質的にはカメラ台数として扱われる。
            coors_img (torch.Tensor):
                各3Dvoxelが参照する画像特徴位置を表す1次元index配列。
                `x = x[coors_img]` の形で用いられ、voxel 順に対応する画像特徴を取得する。
            coors_depth (torch.Tensor, optional):
                各3Dvoxelが参照する深度binの位置を表す1次元index配列。

        Returns:
            list[torch.Tensor]:
                Detection head の出力を直列化したテンソルのリスト。
                各 task について
                `reg`, `height`, `dim`, `rot`, `vel`, `heatmap`
                の順で格納される。

        Notes:
            - `img_view_transformer.depth_net` の出力チャネル数が
            `D + out_channels` の場合は、前半を深度情報、後半を画像特徴として扱う。
            - 先頭要素をゼロ特徴に置き換えることで、どのカメラにも写らない voxel に対して
            ダミーの index 0 を安全に割り当てられるようにしている。
            - `is_transpose` により、BEV 空間の `X/Y` と 2D 特徴マップの `H/W` の対応が変わる。
            - `fuse_type` により、高さ方向 Z の融合方法が変わる。
                * `sum`: 高さ方向の特徴を総和
                * `max`: 高さ方向の特徴の最大値を使用
                * `s2c`: 高さ方向をチャネル方向へ並べて学習的に融合
        """
        # 画像特徴量の抽出
        # img: [Ncam, C, H, W]
        x = self.img_backbone(img)
        x = self.img_neck(x)

        N, C, H, W = x.shape

        # 深度分布+BEV生成用特徴マップの作成
        x = self.img_view_transformer.depth_net(x) # [B*N, D + out_channels, H, W]　use_depth＝Falseの場合、Dの部分は後段で使用しないので、その部分は勾配流れない
        x = x.view(1, N, self.img_view_transformer.D + self.img_view_transformer.out_channels, H, W).permute(0, 1, 3, 4, 2) # [B, N, H, W, C]
        # warning: make sure not been sampled
        x[:, 0, 0, 0] = 0.0 # ダミー用の特徴マップを作成(要素全部0)

        # 深度推定結果(B,N,H,W,D)の抽出
        if self.img_view_transformer.depth_act == 'sigmoid':
            depth = x[..., :self.img_view_transformer.D].sigmoid() # 生値を0~1に変換
        else:
            depth = x[..., :self.img_view_transformer.D].softmax(dim=-1) # 生値を確率分布に変換

        # BEV生成用特徴マップを抽出
        x = x[..., self.img_view_transformer.D:(self.img_view_transformer.D + self.img_view_transformer.out_channels)]

        # 画像特徴量→3Dボクセル特徴量への変換
        x = x.reshape(-1, self.img_view_transformer.out_channels) # [B, N, H, W, out_channels]→[B*N*H*W, out_channels]
        depth = depth.reshape(-1)            # [B, N, H, W, D]→[B*N*H*W*D]
        x = x[coors_img]                        # バッチサイズ1前提 [num_voxel, out_channels]
        depth = depth[coors_depth].unsqueeze(1) # [num_voxel, 1]
        x = x * depth                                       # 画像特徴に深度重みを掛ける
        x = x.view(1, *self.img_view_transformer.grid_size.int().tolist(), self.img_view_transformer.out_channels) # [B, X, Y, Z, out_channels]

        # # 深度推定
        # img = self.img_view_transformer.depth_net(img)

        # # [Ncam, C, H, W] -> [Ncam*H*W, C]
        # channel = img.shape[1]
        # img = img.permute(0, 2, 3, 1).reshape(-1, channel)

        # # index=0 をダミー0特徴にする
        # img_0 = torch.zeros([1, img.shape[1]], device=img.device, dtype=img.dtype)
        # img_rest = img[1:] # 残りの要素
        # img = torch.cat([img_0, img_rest]) # ダミー+1要素目以降を結合
        # # img = torch.cat([img_0, img[1:]], dim=0)

        # x = x.view(B, N, self.D + self.out_channels, H, W).permute(0, 1, 3, 4, 2) # [B, N, H, W, C]
        # # warning: make sure not been sampled
        # x[:, 0, 0, 0] = 0.0 # ダミー用の特徴マップを作成(要素全部0)
        # # depth情報の取り扱い        
        # if channel != self.img_view_transformer.out_channels: # depthありの場合
        #     depth = img[:, :self.img_view_transformer.D].reshape(-1) # 前半Dチャネル(各画素の深度推定情報)を取り出す [B✕H✕W✕D]
        #     x = img[:, self.img_view_transformer.D:(self.img_view_transformer.D + self.img_view_transformer.out_channels)] # 特徴マップ([B,C,H,W])
        #     x = x[coors_img]                        # 3D座標の並びに対応した特徴マップを抽出
        #     depth = depth[coors_depth].unsqueeze(1) # 3D座標の並びに対応した深度推定結果を抽出
        #     x = x * depth                           # 対応する深度binの値で重み付け(深度を使わない場合、ここをコメントアウトする)
        # else:
        #     x = img[coors_img] # [num_voxel, out_channels]

        # # # depthは使用しない
        # # x = img[:, self.img_view_transformer.D:(self.img_view_transformer.D + self.img_view_transformer.out_channels)]

        # # # Voxelに対応する画像特徴の抽出
        # # x = x[coors_img]

        # # [num_voxel, C] -> [1, X, Y, Z, C]
        # x = x.view(1,*self.img_view_transformer.grid_size.int().tolist(), self.img_view_transformer.out_channels)
        # #N, X, Y, Z, C = x.shape

        # BEV座標系の向きによって、後段でX/YとH/Wの対応を変える
        if self.img_view_transformer.is_transpose:
            permute = [0, 3, 2, 1]
        else:
            permute = [0, 3, 1, 2]

        # 高さ方向の結合
        if self.img_view_transformer.fuse_type is not None:
            if self.img_view_transformer.fuse_type == 's2c':
                x = x.reshape(N, X, Y, Z * C).permute(permute)
                x = self.img_view_transformer.fuse(x)
            elif self.img_view_transformer.fuse_type == 'sum':
                x = x.sum(dim=-2).permute(permute)
            elif self.img_view_transformer.fuse_type == 'max':
                x = x.max(dim=-2)[0].permute(permute)
            else:
                raise NotImplementedError

        # ダウンサンプルの実施
        x = self.img_view_transformer.downsample(x)

        # 後処理
        if self.pre_process:
            x = self.pre_process_net(x)[0]

        # 出力: [1, Cbev, Hbev, Wbev]
        return x

#         # 画像特徴量の作成
#         img = self.img_backbone(img)
#         img = self.img_neck(img)
#         img = self.img_view_transformer.depth_net(img) # D+out_channels

#         # 特徴マップを画素ごとの表に変換
#         channel = img.shape[1]
#         img = img.permute(0, 2, 3, 1).reshape(-1, channel) # [B,C,H,W]→[B,H,W,C]→[B*H*W, C] (Bはカメラ台数)

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

@DETECTORS.register_module()
class FastBEV4DTRT(FastBEV4D):
    """過去+現在のBEV特徴量を受取り、物体認識まで行う。

    前提:
        - batch size = 1
        - 過去のBEV特徴量=1
        - gridは計算済み
    """

    def result_serialize(self, outs):
        """VADHeadのdict出力をONNX/TRT向けに直列化する。

        Args:
            outs (dict): VAD_HEADの出力
                - bev_embded: BEV特徴量
                - all_cls_scores: 物体検出のクラスlogitsを層ごとにまとめたもの([L, B, A, num_cls])
                - all_bbox_preds: 物体検出のボックス回帰（物理座標に変換済）  ([L, B, A, box_dim])
                - enc_cls_scores
                - enc_bbox_preds
        """

        # ONNX/TRT向けにリストにする
        return [outs['all_cls_scores'][-1], outs['all_bbox_preds'][-1]] # 最終層の出力のみを抽出

    def forward(self, bev_feat):
        """BEV特徴量を受取り、3D物体検出Headの出力を返す。

        Args:
            bev_feat (torch.tensor):
                過去フレームと現在フレームのBEVをチャネル方向に結合したもの
        """

        # # BEV特徴量の格納用リストの作成
        # bev_feat_list = []
        # bev_feat_list.append(bev_feat_curr)

        # # 過去のBEV特徴量を現在のBEV特徴量に整列させる
        # _, C, H, W = bev_feat_prev.shape
        # bev_feat_prev = F.grid_sample(bev_feat_prev, grid.to(bev_feat_prev.dtype), align_corners=True)
        # bev_feat_list.append(bev_feat_prev.view(1, (self.num_frame - 1) * C, H, W))
        # bev_feat = torch.cat(bev_feat_list, dim=1)

        # BEVエンコーダー
        x = self.bev_encoder(bev_feat)

        # Head処理
        bbox_pts = self.pts_bbox_head([x])

        cls_out = bbox_pts['all_cls_scores'][-1]
        bbox_out = bbox_pts['all_bbox_preds'][-1]

        # dummy = bev_feat.sum() * 0.0
        # cls_out = cls_out + dummy
        # bbox_out = bbox_out + dummy

        return [cls_out, bbox_out]

        # # Headの内容を抽出
        # outs = self.result_serialize(bbox_pts)

        # return outs


# @DETECTORS.register_module()
# class FastBEV4DTRT(FastBEV4D):
#     """過去+現在のBEV特徴量を受取り、物体認識まで行う。

#     前提:
#         - batch size = 1
#         - 過去のBEV特徴量=1
#         - gridは計算済み
#     """

#     def extract_feat(self, inputs):
#         #print(f"sequential:{sequential}")
#         print(f"align:{self.align_after_view_transfromation}")

#         # 入力をフレームごとのlistに分解
#         # sensor2keyegos[0] : 現在フレーム、sensor2keyegos[1] : 過去1フレーム
#         imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, bda= inputs

#         # 出力格納用リストの作成
#         bev_feat_list = []
#         depth_list = []

#         # 各フレームを順に処理
#         key_frame = True  # 学習時はキーフレーム(現在フレーム)のみを勾配対象とする
#         for img, sensor2keyego, ego2global, intrin, post_rot, post_tran in zip(imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans):
#             if key_frame or self.with_prev: # 現在のみ、現在+過去フレームの両方に対応
#                 # 現在の姿勢でBEVを作り、後で整列させる
#                 if self.align_after_view_transfromation:
#                     sensor2keyego, ego2global = sensor2keyegos[0], ego2globals[0]
                
#                 # img_view_transformerへの補助入力の作成(Noneが返ってくる)
#                 mlp_input = self.img_view_transformer.get_mlp_input(sensor2keyegos[0], ego2globals[0], intrin, post_rot, post_tran, bda)
                
#                 # 現フレーム分の入力をまとめる
#                 inputs_curr = (img, sensor2keyego, ego2global, intrin, post_rot, post_tran, bda, mlp_input)

#                 # BEV特徴量の作成
#                 if key_frame: # 現在フレーム
#                     bev_feat, depth = self.prepare_bev_feat(*inputs_curr) 
#                     #print(bev_feat)
#                 else: # 過去フレーム
#                     # TODO: bug here, need fixing
#                     if self.pre_grad: # 過去フレームにも勾配を流す
#                         bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
#                     else: # 過去フレームには勾配を流さない
#                         with torch.no_grad():
#                             bev_feat, depth = self.prepare_bev_feat(*inputs_curr)
#             else: # 過去フレームを使わない場合
#                 # 過去フレームをゼロ特徴で模擬
#                 bev_feat = torch.zeros_like(bev_feat_list[0])
#                 depth = None
            
#             # リストへ追加
#             bev_feat_list.append(bev_feat)
#             depth_list.append(depth)
#             key_frame = False
             
#         # 過去フレームのBEV特徴を現在フレーム基準に整列
#         if self.align_after_view_transfromation:
#             for adj_id in range(1, self.num_frame):
#                 bev_feat_list[adj_id] = self.shift_feature(bev_feat_list[adj_id], [sensor2keyegos[0], sensor2keyegos[adj_id]], bda)
        
#         # 全フレームをチャネル方向に結合し、BEVエンコーダーに入力
#         bev_feat = torch.cat(bev_feat_list, dim=1)
#         x = self.bev_encoder(bev_feat)

#         return [x], depth_list[0]

#     def forward(self, inputs):
#         """画像を受取り、3D物体検出Headの出力を返す。

#         Args:
#             bev_feat (torch.tensor):
#                 過去フレームと現在フレームのBEVをチャネル方向に結合したもの
#         """

#         # # BEV特徴量の格納用リストの作成
#         # bev_feat_list = []
#         # bev_feat_list.append(bev_feat_curr)

#         # # 過去のBEV特徴量を現在のBEV特徴量に整列させる
#         # _, C, H, W = bev_feat_prev.shape
#         # bev_feat_prev = F.grid_sample(bev_feat_prev, grid.to(bev_feat_prev.dtype), align_corners=True)
#         # bev_feat_list.append(bev_feat_prev.view(1, (self.num_frame - 1) * C, H, W))
#         # bev_feat = torch.cat(bev_feat_list, dim=1)

#         # BEV特徴量の作成

#         # BEVエンコーダー
#         bev_feat, _ = self.extract_feat(inputs)

#         # Head処理
#         bbox_pts = self.pts_bbox_head(bev_feat)

#         cls_out = bbox_pts['all_cls_scores'][-1]
#         bbox_out = bbox_pts['all_bbox_preds'][-1]

#         return cls_out, bbox_out

#         # # Headの内容を抽出
#         # outs = self.result_serialize(bbox_pts)

#         # return outs