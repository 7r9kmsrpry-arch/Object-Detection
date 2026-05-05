import torch
import torch.nn as nn
import torch.nn.functional as F

class OnnxInputBuilder():
    
    def __init__(self, grid_config, image_size, stride=16, accelerate=True, img_shape=[1,6,256,16,44]):
        """ONNXモデルへの入力情報を作成する

        Args:
            grid_config (dict):
                BEV グリッドの設定辞書。
                通常は `x`, `y`, `z`, `depth` を含み、それぞれ`[min, max, resolution]` の形式で与える。
            image_size (Tuple[int, int]):
                入力画像サイズ `(H, W)`。
            stride (int, optional):
                入力画像に対する特徴マップの stride。
                デフォルトは `8`。
            accelerate (bool, optional):
                高速化モードを使うかどうか。
                デフォルトは `False`。
        """

        # メンバ変数の初期化
        self.image_size = image_size
        self.grid_config = grid_config
        self.stride = stride

        # グリッド情報(下限値、解像度、サイズ)の作成
        self.create_grid_infos(**grid_config)
        
        # 3Dボクセルグリッドの作成
        self.voxel_coords = self.create_voxel_coords()

        # 高速化モードの設定(model deploy時はTrue)
        self.accelerate = accelerate # バッチサイズ=1にする必要ある！

        # 深度ビンの設定
        self.D = int((self.grid_config['depth'][1] - self.grid_config['depth'][0]) / self.grid_config['depth'][2]) # 深度のビンの数
        self.use_depth = True

        # gridの設定
        self.grid = None

        # 画像特徴量のサイズ
        self.batch_size, _, _, self.height, self.width = img_shape

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
                現在フレームの情報として、以下を含む入力。
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
        for b in range(self.batch_size):
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
                assert self.batch_size == 1, self.batch_size    # バッチサイズ1専用
                assert self.use_depth, self.use_depth # depthを使う前提
                
                # 新しいindex用リストを作成
                # 後段で毎回、カメラ番号を見る→h,wを使って参照する→depthを組み合わせるをしてると遅いので、１個の整数indexにまとめる
                new_img_indices   = [] # c,h,wを潰した1次元index
                new_depth_indices = [] # [c,h,w,d]を潰した1次元index
                N = pre_img_coors_list.shape[0] # 総voxel数
                for idx in range(N):
                    fc, fh, fw = pre_img_coors_list[idx] # [c,h,w]
                    fd = pre_depth_coors_list[idx]       # 深度のビン番号
                    new_img_indices.append(fc * (self.height * self.width) + fh * self.width + fw) # 1次元に潰す
                    new_depth_indices.append((fc * (self.height * self.width) + fh * self.width + fw) * self.D + fd) # Depthのビンの数分拡張して1次元で管理(1画素にD分だけbinを持つ)
                
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

    def gen_grid(self, input, sensor2keyegos, bda, bda_adj=None):
        """過去フレームから現在フレームへのサンプリング用gridの作成"""

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
        l02l1 = c02l0.matmul(torch.inverse(c12l0))[:, 0, :, :].view(n, 1, 1, 4, 4)
        '''
          c02l0 * inv(c12l0)
        = c02l0 * inv(l12l0 * c12l1)
        = c02l0 * inv(c12l1) * inv(l12l0)
        = l02l1 # c02l0==c12l1
        '''

        l02l1 = l02l1[:, :, :,[True, True, False, True], :][:, :, :, :,[True, True, False, True]]

        feat2bev = torch.zeros((3, 3), dtype=grid.dtype).to(grid)
        feat2bev[0, 0] = self.grid_interval[0]
        feat2bev[1, 1] = self.grid_interval[1]
        feat2bev[0, 2] = self.grid_lower_bound[0]
        feat2bev[1, 2] = self.grid_lower_bound[1]
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
    
    def shift_feature(self, bev_feat_prev, sensor2keyegos, bda, bda_adj=None):
        """過去から現在の車両姿勢の変化を過去のBEVに適応"""

        grid = self.gen_grid(bev_feat_prev, sensor2keyegos, bda, bda_adj=bda_adj)
        output = F.grid_sample(bev_feat_prev, grid.to(bev_feat_prev.dtype), align_corners=True)

        return output
        