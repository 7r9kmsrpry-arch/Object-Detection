from typing import Tuple
import os

import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast

from ..builder import NECKS


@NECKS.register_module()
class FastrayTransformer(BaseModule):
    
    def __init__(
        self,
        grid_config,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        downsample: int = 1,
        stride: int = 8,
        fuse = None,
        use_depth = True,
        loss_depth_weight = 0.125,
        depth_act='sigmoid',
        sid = False,
        is_transpose = True,
        accelerate = False,
    ) -> None:
        """画像特徴をBEV 空間へ変換するためのView Transformerモジュール。

        画像特徴マップから深度方向の情報とBEV用特徴を生成し、事前に定義した3Dグリッド（x, y, z, depth）に基づいて
        画像特徴をBEV空間へ対応付けるための設定を保持する。

        主な役割は以下の通り。
            1. BEV 空間のグリッド情報を作成する
            2. voxel 座標を生成する
            3. 深度推定用の `depth_net` を構築する
            4. 高さ方向 Z の融合方法（sum / max / s2c）を設定する
            5. 必要に応じて BEV 特徴のダウンサンプル層を構築する
            6. 高速化モード（accelerate）用の設定を保持する

        Args:
            grid_config (dict):
                BEV グリッドの設定辞書。
                通常は `x`, `y`, `z`, `depth` を含み、それぞれ`[min, max, resolution]` の形式で与える。
            in_channels (int):
                `depth_net` に入力する画像特徴マップのチャネル数。
            out_channels (int):
                BEV 空間へ持ち上げた後の特徴次元数。
            image_size (Tuple[int, int]):
                入力画像サイズ `(H, W)`。
            feature_size (Tuple[int, int]):
                Backbone / Neck 出力の特徴マップサイズ `(H, W)`。
            downsample (int, optional):
                BEV 特徴を追加でダウンサンプルする倍率。
                `1` の場合はダウンサンプルなし、`2` の場合は 2 倍縮小する。
                デフォルトは `1`。
            stride (int, optional):
                入力画像に対する特徴マップの stride。
                デフォルトは `8`。
            fuse (dict, optional):
                高さ方向 Z の融合方法を指定する設定。
                例えば `{'type': 's2c'}` のように与える。
                `None` の場合は融合層を追加しない。
            use_depth (bool, optional):
                深度推定を用いるかどうか。
                デフォルトは `True`。
            loss_depth_weight (float, optional):
                深度損失の重み。
                デフォルトは `0.125`。
            depth_act (str, optional):
                深度方向の活性化関数。
                `'sigmoid'` または `'softmax'` を指定する。
                デフォルトは `'sigmoid'`。
            sid (bool, optional):
                深度離散化で SID（Spacing Increasing Discretization）を使うかどうか。
                Trueの場合、深度ビンを等間隔ではない方法で離散化する。
                デフォルトは `False`。
            is_transpose (bool, optional):
                BEV 特徴を 2D 特徴マップへ並べ替える際に、X/Yの並びを入れ替えるかどうか。
                デフォルトは `True`。(H：Y、W：X)
            accelerate (bool, optional):
                高速化モードを使うかどうか。
                デフォルトは `False`。
        """

        # メンバ変数の初期化
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.grid_config = grid_config
        self.out_channels = out_channels
        self.stride = stride
        self.is_transpose = is_transpose

        # ダウンサンプル層の構築
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else: # デフォルト
            self.downsample = nn.Identity() # 恒等関数

        # グリッド情報(下限値、解像度、サイズ)の作成
        self.create_grid_infos(**grid_config)

        # 高さ方向の融合方法設定
        self.fuse = self.fuse_type = None
        if fuse is not None:
            if fuse['type'] == 's2c':
                self.fuse = nn.Conv2d(self.out_channels*self.grid_size[2].int(), self.out_channels, kernel_size=1) # チャネル方向に結合して、out_channelsに戻す
            self.fuse_type = fuse['type']
        
        # 3Dボクセルグリッドの作成
        self.voxel_coords = self.create_voxel_coords()

        # 高速化モードの設定(model deploy時はTrue)
        self.accelerate = accelerate # バッチサイズ=1にする必要ある！

        # 深度推定関連の設定
        assert depth_act in ['sigmoid', 'softmax']
        self.depth_act = depth_act
        self.sid = sid
        self.use_depth = use_depth
        self.loss_depth_weight = loss_depth_weight
        self.D = int((self.grid_config['depth'][1] - self.grid_config['depth'][0]) / self.grid_config['depth'][2]) # 深度のビンの数
        self.depth_net = nn.Conv2d(
            self.in_channels, 
            self.D + self.out_channels, # 深度推定結果+BEVへの特徴マップを作成
            kernel_size=1, 
            padding=0
        )

        # debug
        self.first_frame = True

    def create_grid_infos(self, x, y, z, **kwargs):
        """グリッド空間の下限・解像度・サイズを設定する。

        x, y, z 各軸について、
        - 下限値
        - 刻み幅
        - セル数
        を計算し、メンバ変数として保存する。

        Args:
            x (tuple[float]): x軸方向の設定。
                (下限, 上限, 刻み幅) の形式で与える。
            y (tuple[float]): y軸方向の設定。
                (下限, 上限, 刻み幅) の形式で与える。
            z (tuple[float]): z軸方向の設定。
                (下限, 上限, 刻み幅) の形式で与える。
            **kwargs: 将来拡張用の追加引数。
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])              # グリッド下限
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])                 # グリッド解像度
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2] for cfg in [x, y, z]]) # グリッドサイズ(セル数)

    def create_voxel_coords(self):
        """ボクセルグリッド内の各セル中心座標を生成する。

        `create_grid_infos()` で作成した
        - `self.grid_size`
        - `self.grid_interval`
        - `self.grid_lower_bound`
        を用いて、x, y, z 各軸方向の全ボクセルについて3次元座標を生成する。

        生成される座標は、各ボクセルのインデックスではなく、実空間上の座標値を表す。
        最後に `(N, 3)` 形状へ変換し、学習対象ではない固定パラメータとして返す。

        Returns:
            nn.Parameter:
                全ボクセルの座標を格納したテンソル。
                shape は `(num_voxels, 3)`。
                各行は 1 個のボクセルに対応し、`[x, y, z]` の実座標を表す。
        """
        # 各軸方向のインデックスを作成
        x = torch.arange(int(self.grid_size[0])).view(-1, 1, 1).expand(-1, int(self.grid_size[1]), int(self.grid_size[2])) # [Nx, Ny, Nz]
        y = torch.arange(int(self.grid_size[1])).view(1, -1, 1).expand(int(self.grid_size[0]), -1, int(self.grid_size[2])) # [Nx, Ny, Nz]
        z = torch.arange(int(self.grid_size[2])).view(1, 1, -1).expand(int(self.grid_size[0]), int(self.grid_size[1]), -1) # [Nx, Ny, Nz]
        
        # (x,y,z)をまとめる
        coords = torch.stack((x, y, z), dim=3) # [Nx, Ny, Nz, 3]
        
        # グリッドインデックスを実座標へ変換する
        coords = coords * self.grid_interval + (self.grid_lower_bound - self.grid_interval / 2.0) # ÷2してるので、グリッド境界ではなく中心点になる
        
        # 2次元に整形
        coords = coords.reshape(-1, 3) # [Nx, Ny, Nz, 3]→[Nx✕Ny✕Nz, 1]

        return nn.Parameter(coords, requires_grad=False) # 学習なしパラメータに設定

    def get_fastray_input(self, input):
        """
        FastRay 用の事前対応表を作成する。

        入力として与えられた画像・座標変換行列・画像拡張情報などを用いて、
        あらかじめ定義された 3D ボクセル座標 `self.voxel_coords` を各カメラ画像へ投影し、以下の対応関係をバッチ単位で生成する。

        1. 各ボクセルが対応するボクセルインデックス
        2. 各ボクセルが対応する画像上の画素座標（カメラ番号, h, w）
        3. 各ボクセルが対応する深度インデックス

        主な用途は、3D 空間上の各ボクセルと 2D 画像特徴 / 深度特徴を高速に対応付けるための前処理である。

        Args:
            input (list or tuple):
                少なくとも先頭 7 要素として以下を含む入力。
                - img: 画像特徴マップ
                    shape = (B, N, C, H, W)
                - sensor2ego: カメラ座標系 → ego座標系 の変換行列
                    shape = (B, N, 4, 4) を想定
                - ego2global: ego座標系 → global座標系 の変換行列
                    ※ この関数内では未使用
                - cam2imgs: カメラ内部パラメータ行列
                    shape = (B, N, 3, 3) を想定
                - post_rots: 画像拡張後の回転変換
                - post_trans: 画像拡張後の平行移動変換
                - bda: BEV データ拡張行列（LiDAR/BEV側の拡張）

        Returns:
            tuple:
                - batch_pre_voxel_coors_list:
                    各バッチにおけるボクセルインデックスのリスト
                - batch_pre_img_coors_list:
                    各バッチにおける画像座標のリスト
                    accelerate=False のときは (cam, h, w)
                    accelerate=True のときは 1 次元インデックス
                - batch_pre_depth_coors_list:
                    各バッチにおける深度インデックスのリスト
                    accelerate=False のときは depth bin
                    accelerate=True のときは 1 次元インデックス
        """

        # 入力を分解
        img, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda = input[:7]
        # 画像特徴マップのshapeを取得
        batch_size, n_images, n_channels, height, width = img.shape # {B, N, C, H, W]}
        # bdaを別名で保持
        lidar_aug_matrix = bda
        
        # 特に何もしてない
        post_rots = post_rots
        post_trans = post_trans
        # センサー→egoへの回転と並進を抽出(未使用)
        camera2lidar_rots = sensor2ego[..., :3, :3] # 回転
        camera2lidar_trans = sensor2ego[..., :3, 3] # 並進
        # bdaから回転と並進を取り出す(未使用)
        extra_rots = bda[..., :3, :3]
        extra_trans = bda[..., :3, 3]

        # ego→imgsの変換行列を作成
        # cam2imgを4✕4に拡張
        new_cam2imgs = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(*sensor2ego.shape[:2], 1, 1).to(sensor2ego.device) # [B,N,4,4]
        # 内部パラメータを代入
        new_cam2imgs[:, :, :3, :3] = cam2imgs
        # ego→sensor✕sensor→imgs = ego→imgs
        camego2imgs = new_cam2imgs.matmul(torch.inverse(sensor2ego))

        # 出力用リストを初期化
        batch_pre_voxel_coors_list = []
        batch_pre_img_coors_list = []
        batch_pre_depth_coors_list = []

        # 各ボクセルの3D座標を各カメラ画像へ投影して、最終的にfeature map上の整数座標へ変換する
        for b in range(batch_size):
            # バッチごとの情報を取り出す
            cur_lidar_aug_matrix = lidar_aug_matrix[b] # bda行列
            cur_camego2img = camego2imgs[b]            # ego→imgsの変換行列
            curr_post_rots = post_rots[b]              # augmentationの座標変換
            curr_post_trans = post_trans[b]            # augmentationの座標変換

            # inverse aug(test時は恒等変換で何もなし)
            # BEVaug後に各座標に対応する点が、もともとどの座標の点なのかを算出
            # 2D→3Dにするのではなく、直接BEVaug後の3Dを作成する
            cur_coords = self.voxel_coords - cur_lidar_aug_matrix[:3, 3].view(1,3)                      # 並進成分を引く
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(cur_coords.transpose(1, 0)) # 回転成分を打ち消す

            # camego2image
            # ego座標系の各3Dボクセル座標を、各カメラの画像射影用座標へ変換
            # (内部パラメータを含む線形成分 + 並進)
            cur_coords = cur_camego2img[:, :3, :3].matmul(cur_coords) # 回転
            cur_coords += cur_camego2img[:, :3, 3].reshape(-1, 3, 1)  # +並進 
            
            # z成分を奥行きとして保存し、透視投影により (u, v) 座標へ変換
            dist = cur_coords[:, 2, :] 
            cur_coords[:, 2, :][cur_coords[:, 2, :] <= 0.0] = torch.inf # カメラから見て後側の点を無効化
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :] 

            # imgaug
            # 画像オーグメンテーション後のどのピクセルに対応してるかを算出
            cur_coords = curr_post_rots.matmul(cur_coords)    # 回転
            cur_coords += curr_post_trans.reshape(-1, 3, 1)   # 並進
            cur_coords = cur_coords[:, :2, :].transpose(1, 2) # [Num_camera, 2, num_voxels]→[Num_camera, num_voxels, 2]

            # normalize coords for grid sample
            # 特徴マップのどのピクセルに該当するかを算出
            cur_coords = cur_coords[..., [1, 0]] / self.stride # テンソルは[h,w]なので、[x, y]→[y, x]にする
            cur_coords = cur_coords.long()                     # 整数インデックスに変換

            # 画像特徴マップの対応先があるかどうかの確認([num_cams, num_voxels])
            on_img = ((cur_coords[..., 0] < (self.image_size[0] / self.stride)) # 縦方向が上端を超えていない
                    & (cur_coords[..., 0] >= 0)                                 # 縦方向が負の値ではない
                    & (cur_coords[..., 1] < (self.image_size[1] / self.stride)) # 横方向が右端を超えていない
                    & (cur_coords[..., 1] >= 0)                                 # 横方向が負ではない
                    & (dist >= self.grid_config['depth'][0])                    # 近すぎる点(default=1m)を除外
                    & (dist < self.grid_config['depth'][1]))                    # 遠すぎる点(default=60m)を除外
            
            # 複数カメラで同じ3D座標に対応する場合、先に処理したカメラのみ選択する
            for valid_i in range(1, len(on_img)):
                for valid_j in range(0, valid_i):
                    on_img[valid_i][on_img[valid_j] == True] = False # [on_img[valid_j]==True]はbool判定結果

            # 有効と判定されたvoxelについて、後で使いやすい対応表を作成
            # 空リストの作成
            pre_img_coors_list = []
            pre_depth_coors_list = []
            pre_voxel_coors_list = []

            # カメラごとに処理
            for c in range(on_img.shape[0]):
                # 有効な画像座標のみを抽出 [num_valid, 2]
                masked_coords = cur_coords[c, on_img[c]] 
                # assert masked_coords[(masked_coords[:, 0] == 0) & (masked_coords[:, 1] == 0)].shape[0] == 0
                
                # [num_valid, 3(c, h, w)]に変換
                pre_img_coors_list.append(torch.cat([
                    masked_coords.new(masked_coords[:, 0:1].shape).zero_()+c, # カメラ番号
                    masked_coords[:, 0:1], # h
                    masked_coords[:, 1:2]  # w
                ], dim=1))

                # 有効なvoxelについて、「元のvoxel番号」と「depth bin番号」を対応付ける
                pre_voxel_coors_list.append(torch.nonzero(on_img[c])[:, 0]) # vowelの何番目を使用したかを保存([1, 3, 8 ,...]など)
                # 各vowelの深度を対応するbin番号に変換
                depth_idx = ((dist[c, on_img[c]] - (self.grid_config['depth'][0] - self.grid_config['depth'][2])) / self.grid_config['depth'][2]).long() - 1 # depth_min=1なので、depth_minが0番ビンになるように-1してる
                # 有効な要素が0出ないときのみ、binが有効かどうかをチェック
                if len(depth_idx) != 0:
                    assert depth_idx.min() >= 0
                    assert depth_idx.max() < self.D
                # 対応する深度のbin番号を追加
                pre_depth_coors_list.append(depth_idx)

            # どのカメラにも写っていないvoxel番号を取得
            pre_voxel_coors_index = torch.nonzero(~on_img.sum(0).bool())[:, 0] # [num_true←どのカメラにも写ってない個数,]
            # どのカメラにも写ってないボクセルに対して、ダミーの画像対応表を作成[num_true, 3(0,0,0)←0番目のカメラの(h=0,w=0)]の行列を作成
            pre_img_coors_index = torch.zeros(pre_voxel_coors_index.shape[0], 3).long().to(pre_voxel_coors_index.device)
            # どのカメラにも写ってないボクセルに対して、ダミー深度ビン番号(0)を作成[num_true,]
            pre_depth_coors_index = torch.zeros(pre_voxel_coors_index.shape[0]).long().to(pre_voxel_coors_index.device)

            # カメラに写ってない情報も追加
            pre_voxel_coors_list.append(pre_voxel_coors_index) 
            pre_img_coors_list.append(pre_img_coors_index)     
            pre_depth_coors_list.append(pre_depth_coors_index)

            # 各カメラごとの対応表を連結
            pre_voxel_coors_list = torch.cat(pre_voxel_coors_list, dim=0) # [num_voxels,]   例：[2, 5, 7, 9, 10]
            pre_img_coors_list = torch.cat(pre_img_coors_list, dim=0)     # [num_voxels, 3] 例：[[0,1,3],[0,2,4],[1,5,6],[0,0,0],[0,0,0]]
            pre_depth_coors_list = torch.cat(pre_depth_coors_list, dim=0) # [num_voxels,]  例：[4, 8, 2, 0, 0]

            # [c,h,w]やdepth_idxを1次元indexに潰して、voxel順に並べ直す
            if self.accelerate: # model deploy時は、デフォルトでTrue
                assert batch_size == 1, batch_size    # バッチサイズ1専用
                assert self.use_depth, self.use_depth # depthを使う前提
                
                # 新しいindex用リストを作成
                # 後段で毎回、カメラ番号を見る→h,wを使って参照する→depthを組み合わせるをしてると遅いので、１個の整数indexにまとめる
                new_img_indices   = [] # c,h,wを潰した1次元index
                new_depth_indices = [] # [c,h,w,d]を潰した1次元index
                N = pre_img_coors_list.shape[0] # 総voxel数
                for idx in range(N):
                    fc, fh, fw = pre_img_coors_list[idx] # [c,h,w]
                    fd = pre_depth_coors_list[idx]       # 深度のビン番号
                    new_img_indices.append(fc * (height * width) + fh * width + fw) # 1次元に潰す
                    new_depth_indices.append((fc * (height * width) + fh * width + fw) * self.D + fd) # Depthのビンの数分拡張して1次元で管理(1画素にD分だけbinを持つ)
                
                # リストをTensorに変換する
                pre_img_coors_list = torch.Tensor(new_img_indices).long().to(pre_img_coors_list.device)     # [C✕H✕W]
                pre_depth_coors_list = torch.Tensor(new_depth_indices).long().to(pre_img_coors_list.device) # [C✕H✕W✕D]

                # sort by voxel coor
                # 各カメラの結果が入り混じってるので、voxel番号順に並び替え
                pre_voxel_coors_list, sort_idx = pre_voxel_coors_list.sort() # [0,1,...,num_voxels-1]
                # 画像テンソルと深度ビンも座標のソート結果に対応させる
                pre_img_coors_list = pre_img_coors_list[sort_idx]
                pre_depth_coors_list = pre_depth_coors_list[sort_idx]
                assert len(pre_voxel_coors_list) == len(self.voxel_coords)

            # 今回のバッチの結果をリストに追加
            batch_pre_voxel_coors_list.append(pre_voxel_coors_list)
            batch_pre_img_coors_list.append(pre_img_coors_list)
            batch_pre_depth_coors_list.append(pre_depth_coors_list)

        return batch_pre_voxel_coors_list, batch_pre_img_coors_list, batch_pre_depth_coors_list

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.stride,
                                   self.stride, W // self.stride,
                                   self.stride, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.stride * self.stride)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    torch.inf * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.stride,
                                   W // self.stride)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            raise NotImplemented
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss

    def forward(self, input, depth_from_lidar=None):
        """画像特徴を 3D voxel 空間へ配置し、2D の BEV 特徴へ変換する。

        この関数は、複数カメラの画像特徴から深度分布と BEV 用特徴を生成し、事前計算した voxel-画像対応表を用いて 3D voxel グリッドへ再配置する。
        その後、高さ方向Zを融合して2DのBEV特徴へ変換し、必要に応じてダウンサンプルした結果を返す。

        処理の流れは以下の通り。
            1. 入力画像特徴 `[B, N, C, H, W]` を `[B*N, C, H, W]` に変形して`depth_net` へ入力する
            2. 各画素について、深度分布 `D` と BEV 用特徴 `out_channels` を生成する
            3. `get_fastray_input()` により、各voxelが参照する画像位置と深度binの対応表を取得する
            4. `accelerate=True` の場合は1次元indexによる高速参照でvoxel特徴を作る
            5. `accelerate=False` の場合はバッチごとに明示的にvoxel特徴を配置する
            6. `[B, X, Y, Z, C]` のvoxel格子を `sum` / `max` / `s2c` で高さ方向に融合し、2D の BEV 特徴へ変換する
            7. downsample を適用し、BEV 特徴と深度分布を返す

        Args:
            input (list or tuple):
                View Transformer 用の入力一式。
                先頭要素 `input[0]` は画像特徴テンソルで、形状は
                `[B, N, C, H, W]` を想定する。
                ここで
                    - `B`: バッチサイズ
                    - `N`: カメラ台数
                    - `C`: 入力特徴チャネル数
                    - `H, W`: 特徴マップの高さ・幅
                を表す。
                そのほかの要素にはカメラパラメータや幾何変換情報が含まれ、
                `get_fastray_input()` 内で使用される。
            depth_from_lidar (torch.Tensor, optional):
                LiDAR 由来の深度情報。
                この関数内では使用していないが、拡張用の引数として保持されている。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x (torch.Tensor):
                    2D の BEV 特徴マップ。
                    形状は通常 `[B, C', H_bev, W_bev]`。
                - depth (torch.Tensor):
                    各画素の深度分布。
                    形状は `[B, N, H, W, D]`。

        Notes:
            - `depth_net` の出力は各画素について
            「深度分布 `D`」と「BEV 用特徴 `out_channels`」を連結したもの。
            - `x[:, 0, 0, 0] = 0.0` により、どのカメラにも写らない voxel 用の
            ダミー参照先をゼロ特徴にしている。
            - `accelerate=True` では、事前に 1 次元 index 化した
            `pre_img_coors_list` / `pre_depth_coors_list` を使って高速に voxel 特徴を作る。
            - `accelerate=False` では、voxel 番号と画像座標の対応表を用いて
            明示的に voxel 特徴を埋める。
            - `fuse_type` により、高さ方向 Z の融合方法が決まる。
                * `sum`: 高さ方向の特徴を総和
                * `max`: 高さ方向の特徴の最大値を使用
                * `s2c`: 高さ方向をチャネル方向へ並べて学習的に融合
        """

        # 入力画像特徴量を抽出
        x = input[0]
        # debug
        #print('x: ', x)
        print('use_depth', self.use_depth)

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W) # バッチとカメラ台数の次元を統合

        # 深度分布+BEV生成用特徴マップの作成
        x = self.depth_net(x) # [B*N, D + out_channels, H, W]　use_depth＝Falseの場合、Dの部分は後段で使用しないので、その部分は勾配流れない
        x = x.view(B, N, self.D + self.out_channels, H, W).permute(0, 1, 3, 4, 2) # [B, N, H, W, C]
        # warning: make sure not been sampled
        x[:, 0, 0, 0] = 0.0 # ダミー用の特徴マップを作成(要素全部0)

        # 深度推定結果(B,N,H,W,D)の抽出
        if self.depth_act == 'sigmoid':
            depth = x[..., :self.D].sigmoid() # 生値を0~1に変換
        else:
            depth = x[..., :self.D].softmax(dim=-1) # 生値を確率分布に変換

        # BEV生成用特徴マップを抽出
        x = x[..., self.D:(self.D + self.out_channels)]

        # FastRay用の事前対応表を作成
        pre_voxel_coors_list, pre_img_coors_list, pre_depth_coors_list = self.get_fastray_input(input)

        # debug
        if self.first_frame:
            print('img_coord: ', pre_img_coors_list)
            self.first_frame = False

        # 画像特徴量→3Dボクセル特徴量への変換
        if self.accelerate: # 高速版(バッチサイズ1限定！)
            assert self.use_depth, self.use_depth
            x = x.reshape(-1, self.out_channels) # [B, N, H, W, out_channels]→[B*N*H*W, out_channels]
            depth = depth.reshape(-1)            # [B, N, H, W, D]→[B*N*H*W*D]
            x = x[pre_img_coors_list[0]]                        # バッチサイズ1前提 [num_voxel, out_channels]
            depth = depth[pre_depth_coors_list[0]].unsqueeze(1) # [num_voxel, 1]
            x = x * depth                                       # 画像特徴に深度重みを掛ける
            x = x.view(B, *self.grid_size.int().tolist(), self.out_channels) # [B, X, Y, Z, out_channels]
        else: # 非高速版(バッチサイズ1以外でもOK。内容は同じ)
            # 空のvoxel配列を作成
            voxel_feature = torch.zeros(
                (B, int(self.grid_size[0]) * int(self.grid_size[1]) * int(self.grid_size[2]), self.out_channels), device=x.device
            ).type_as(x) # [B, X*Y*Z, out_channels]
            for i in range(B):
                if self.use_depth:
                    voxel_feature[i][pre_voxel_coors_list[i]] = \
                        x[i][pre_img_coors_list[i][:, 0], pre_img_coors_list[i][:, 1], pre_img_coors_list[i][:, 2]] * \
                        depth[i][pre_img_coors_list[i][:, 0], pre_img_coors_list[i][:, 1], pre_img_coors_list[i][:, 2], pre_depth_coors_list[i]].unsqueeze(-1)
                else:
                    voxel_feature[i][pre_voxel_coors_list[i]] = \
                        x[i][pre_img_coors_list[i][:, 0], pre_img_coors_list[i][:, 1], pre_img_coors_list[i][:, 2]]
            
            # 整形
            x = voxel_feature.view(B, *self.grid_size.int().tolist(), self.out_channels)
            N, X, Y, Z, C = x.shape
        
        # Z方向の結合
        permute = [0, 3, 2, 1] if self.is_transpose else [0, 3, 1, 2]
        if self.fuse_type is not None:
            if self.fuse_type == 's2c':
                x = x.reshape(N, X, Y, Z*C).permute(permute)
                x = self.fuse(x)
            elif self.fuse_type == 'sum':
                x = x.sum(dim=-2).permute(permute)
            elif self.fuse_type == 'max':
                x = x.max(dim=-2)[0].permute(permute)
            else:
                raise NotImplemented
        
        # ダウンサンプルの実施
        x = self.downsample(x)

        return x, depth
    
    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None

