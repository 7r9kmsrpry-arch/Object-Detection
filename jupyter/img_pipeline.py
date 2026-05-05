# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion

# 画像の正規化

def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)

def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img

def image_normalize_core(img):
    """入力画像を正規化し、CHWに順番を入れ替える。"""

    # 正規化用パラメータの設定
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32) # ImageNet事前学習系の正規化値
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)    # ImageNet事前学習系の正規化値
    to_rgb = True # OpenCV経由の場合、デフォルトでBGRになっているが、PIlowの場合は不要

    # 画像の正規化
    img = imnormalize(np.array(img), mean, std, to_rgb)
    
    # [H,W,C]→[C,H,W]に入れ替え
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()

    return img

# 画像の読み込み
class PrepareImageInputs():
    """複数チャネル画像入力を、個別ファイルのリストから読み込んで準備する。

    `input_data['img_filename']` には、ファイル名のリストが入っていることを想定する。

    Args:
        to_float32 (bool): 画像をfloat32に変換するかどうか。デフォルトは False。
        color_type (str): 画像ファイルの色形式。デフォルトは'unchanged'。
    """

    def __init__(
        self,
        data_config,
        sequential=True,
        opencv_pp=False,
        num_frame=2
    ):
        self.data_config = data_config
        self.normalize_img = image_normalize_core
        self.sequential = sequential
        self.opencv_pp = opencv_pp
        self.num_frame = num_frame

    def get_rot(self, h):
        """回転行列の作成"""

        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims, crop, flip, rotate):
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

        # オーグメンテーションの内容を変換行列に反映
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

        # OpenCVの関数を使った、オーグメンテーションの実施
        if self.opencv_pp:
            img = self.img_transform_core_opencv(img, post_rot, post_tran, crop)
            
        return img, post_rot, post_tran

    def img_transform_core_opencv(self, img, post_rot, post_tran, crop):
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
        sensor2ego_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        sensor2ego_tran = torch.Tensor(cam_info['cams'][cam_name]['sensor2ego_translation'])
        sensor2ego = sensor2ego_rot.new_zeros((4, 4))
        sensor2ego[3, 3] = 1
        sensor2ego[:3, :3] = sensor2ego_rot
        sensor2ego[:3, -1] = sensor2ego_tran
        
        # sweep ego to global
        w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
        ego2global_rot = torch.Tensor(Quaternion(w, x, y, z).rotation_matrix)
        ego2global_tran = torch.Tensor(cam_info['cams'][cam_name]['ego2global_translation'])
        ego2global = ego2global_rot.new_zeros((4, 4))
        ego2global[3, 3] = 1
        ego2global[:3, :3] = ego2global_rot
        ego2global[:3, -1] = ego2global_tran
        
        return sensor2ego, ego2global

    def get_inputs(self, input_data, flip=None, scale=None):
        """画像に対する前処理入力を作成する関数。

        Args:
            input_data (dict): 今回フレームと隣接フレームのメタ情報

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
        cam_names = self.data_config['cams']
        input_data['cam_names'] = cam_names

        # 可視化用の生画像の配列
        canvas = []

        for cam_name in cam_names:
            # 現在フレームの画像を取得
            cam_data = input_data['curr']['cams'][cam_name]
            filename = cam_data['data_path']
            img = Image.open(filename)

            # 回転と並進情報を格納する行列を作成
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # 内部パラメータの取得
            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            # センサー→ego、ego→globalの変換行列を取得
            sensor2ego, ego2global = self.get_sensor_transforms(input_data['curr'], cam_name)

            # オーグメンテーションの適用
            img_augs = self.sample_augmentation(H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            img, post_rot2, post_tran2 = self.img_transform(img, 
                                                            post_rot, 
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

            # 可視化用の画像配列に保存
            canvas.append(np.array(img))
            # 正規化を行い、学習用の画像配列に保存
            imgs.append(self.normalize_img(img))

            # 時系列処理
            if self.sequential:
                assert 'adjacent' in input_data
                for adj_info in input_data['adjacent']:
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
            for adj_info in input_data['adjacent']:
                # 過去フレーム分も各行列を複製
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # 過去フレームを整列させるための行列を取得(過去のフレームのegoやglobal基準)
                for cam_name in cam_names:
                    sensor2ego, ego2global = self.get_sensor_transforms(adj_info, cam_name)
                    sensor2egos.append(sensor2ego)
                    ego2globals.append(ego2global)

        # 各情報をスタックして、バッチ次元を追加する
        imgs = torch.stack(imgs).unsqueeze(0)
        sensor2egos = torch.stack(sensor2egos).unsqueeze(0)
        ego2globals = torch.stack(ego2globals).unsqueeze(0)
        intrins = torch.stack(intrins).unsqueeze(0)
        post_rots = torch.stack(post_rots).unsqueeze(0)
        post_trans = torch.stack(post_trans).unsqueeze(0)

        # 描画用画像(現在フレームのオーグメンテーション後の各カメラ)を保存
        input_data['canvas'] = canvas

        # BEV Augmentation行列(恒等変換)
        bda_mat = torch.eye(4).unsqueeze(0)

        # 各情報を時間方向に扱いやすいように分解する(list[B, N, ...])
        inputs = (imgs, sensor2egos, ego2globals, intrins, post_rots, post_trans, bda_mat)
        inputs = self.prepare_inputs(inputs)

        return inputs

    def prepare_inputs(self, inputs):
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

        # 現在フレーム
        img_inputs_curr = [imgs[0], sensor2keyegos[0], ego2globals[0], intrins[0], post_rots[0], post_trans[0], bda]
        # 過去フレーム
        img_inputs_prev = [imgs[1], sensor2keyegos[1], ego2globals[1], intrins[1], post_rots[1], post_trans[1], bda]

        return img_inputs_curr, img_inputs_prev

    def __call__(self, input_data):
        img_inputs_curr, img_inputs_prev = self.get_inputs(input_data)
        input_data['img_inputs_curr'] = img_inputs_curr
        input_data['img_inputs_prev'] = img_inputs_prev

        return input_data