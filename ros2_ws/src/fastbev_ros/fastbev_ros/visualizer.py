import torch
import cv2
import numpy as np
import os
from pyquaternion.quaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from .lidar_box3d import LiDARInstance3DBoxes as LB
import pyquaternion

class Visualizer():
    """
    認識結果のバウンディングボックスを重畳した画像を作成する。

    Args:
        scale_factor (int): 描画時の画像の縮小率
        color_map (touple): 描画時の線の色
        save_format (str):  描画結果の保存フォーマット('video' or 'image')
        save_path (str):    描画結果の保存先パス
    """

    def __init__(self, score_thresh, save_path, save_format='video', save_prefix='result', fps=10, scale_factor=3, color_map=(0, 255, 255)):
        # メンバ変数の設定
        # 描画しきい値
        self.score_thresh = score_thresh 

        # 保存設定
        self.save_path   = save_path     # 描画結果の保存先
        self.save_format = save_format # 描画結果の保存フォーマット
        if self.save_format == 'video':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(1600 / scale_factor * 3)
            height = int(900 / scale_factor * 2)
            self.video_writer = cv2.VideoWriter(os.path.join(save_path, f'{save_prefix}.mp4'), fourcc, fps, (width, height))
        
        # 描画設定
        self.scale_factor = scale_factor # 描画時の画像の縮小率
        self.color_map = color_map       # 描画時の線の色
        self.draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), 
                                            (4, 5), (5, 6), (6, 7), (7, 4), 
                                            (0, 4), (1, 5), (2, 6), (3, 7)] # 画像に描画するbboxの辺の順番

        # 使用するカメラ一覧
        self.views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT']

        # 認識クラス一覧
        self.classes = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]

        # ego→globalへの変換行列の基準カメラ(どのカメラでも同じ値が入ってる)
        self.ego_cam = 'CAM_FRONT'

    def format_bbox(self, bboxes, scores, labels, data_info):
        """推論結果をnuScenesの標準提出形式に変換する。

        Args:
            bboxes (list(LiDARInstance3DBoxes)): 3Dバウンディングボックス
            scores (list(tensor.torch)): 各ボックスの信頼度
            labels (list(tensor.torch)): 各ボックスのクラスラベル
            data_info (dict):  今回フレームのメタ情報

        Returns:
            nusc_results (list[dict]): nuScenesの標準形式にformatされた認識結果
        """

        # 初期化
        mapped_class_names = self.classes
        nusc_results = list()

        # numpyに変換
        bboxes_np = bboxes.tensor.detach().numpy()
        scores_np = scores.detach().numpy()
        labels_np = labels.detach().numpy()

        # debug
        sample_token = data_info['token']
        nusc_annos = {}

        # 認識結果の座標系をego座標系→global座標系に変換する情報を取得
        trans = data_info['cams'][self.ego_cam]['ego2global_translation']
        rot   = data_info['cams'][self.ego_cam]['ego2global_rotation']
        print(f"rot: {rot}")
        rot = pyquaternion.Quaternion(rot)

        print(f"trans: {trans}")
        print(f"rot_quaternion: {rot}")

        # bboxごとに処理を実施
        for bbox, score, label in zip(bboxes_np, scores_np, labels_np):

            # クラスラベルをクラス名に変換
            name = mapped_class_names[label]

            # bbox情報を取得 
            center = bbox[:3]           # 中心座標
            wlh = bbox[[4, 3, 5]]       # 大きさ
            box_yaw = bbox[6]           # yaw
            box_vel = bbox[7:].tolist() # 速度
            box_vel.append(0)           # z成分の速度(nuScnesBoxが3次元を想定)
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw) # Quaternionに変換

            # NuScenesBoxを作成
            nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)

            # ego→globalに変換
            nusc_box.rotate(rot)
            nusc_box.translate(trans)
            
            # 1サンプル分の結果を格納
            nusc_result = dict(
                translation=nusc_box.center.tolist(),
                size=nusc_box.wlh.tolist(),
                rotation=nusc_box.orientation.elements.tolist(),
                detection_name=name,
                detection_score=float(score),
            )

            # 最終的な結果に追加
            nusc_results.append(nusc_result)
        
        # debug
        if sample_token in nusc_annos:
            nusc_annos[sample_token].extend(nusc_results)
        else:
            nusc_annos[sample_token] = nusc_results


        return nusc_results, nusc_annos

    def check_point_in_img(self, points, height, width):
        """
        点群の各点が画像範囲内に存在するかを判定する。

        Args:
            points (numpy.ndarray): 各点の画像座標を格納した配列。
                形状は (N, 2) または少なくとも先頭2列に x, y を含むことを想定する。
                `points[:, 0]` を x 座標、`points[:, 1]` を y 座標として扱う。
            height (int): 画像の高さ（ピクセル数）。
            width (int): 画像の幅（ピクセル数）。

        Returns:
            numpy.ndarray: 各点が画像範囲内にあるかどうかを表す真偽値配列。形状は (N,)。
                        各要素は、対応する点が`0 <= x < width` かつ `0 <= y < height`を満たす場合に True、それ以外は False となる。

        Notes:
            - 画像の左上を原点 (0, 0) とする画像座標系を前提としている。
            - x 座標は幅方向、y 座標は高さ方向として判定する。
            - 境界上のうち、左端と上端は有効、右端 (`x == width`) と下端 (`y == height`) は画像外として扱う。
        """

        # 画像内に収まっているかどうかを確認
        valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
        valid = np.logical_and(valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))

        return valid

    def lidar2img(self, points_lidar, camrera_info):
        points_lidar_homogeneous = \
            np.concatenate([points_lidar,
                            np.ones((points_lidar.shape[0], 1),
                                    dtype=points_lidar.dtype)], axis=1)
        camera2lidar = np.eye(4, dtype=np.float32)
        camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
        camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
        lidar2camera = np.linalg.inv(camera2lidar)
        points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
        points_camera = points_camera_homogeneous[:, :3]
        valid = np.ones((points_camera.shape[0]), dtype=bool)
        valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
        points_camera = points_camera / points_camera[:, 2:3]
        camera2img = np.array(camrera_info['cam_intrinsic'], dtype=np.float32)
        points_img = points_camera @ camera2img.T
        points_img = points_img[:, :2]
        return points_img, valid

    def get_lidar2global(self, infos):
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = infos['lidar2ego_translation']
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = Quaternion(
            infos['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = infos['ego2global_translation']
        return ego2global @ lidar2ego
    
    def draw_bbox(self, nusc_results, data_info, frame_idx):
        """bboxを画像に描画する
    
        Args:
            data_info (dict):  今回フレームのメタ情報
            frame_idx (int): 描画するフレーム番号 
            nusc_results (list[dict]): nuScenesの標準形式にformatされた認識結果     

        Returns:
            drawed_img (np.array): 認識結果を描画した画像
        """

        # nuscenesの標準形式のbboxから[x,y,z,dx,dy,dz,yaw]を抽出
        pred_boxes = [
            nusc_results[rid]['translation'] + 
            nusc_results[rid]['size'] + 
            [Quaternion(nusc_results[rid]['rotation']).yaw_pitch_roll[0] + np.pi / 2] 
            for rid in range(len(nusc_results))
        ]

        if len(pred_boxes) == 0: # 予測結果が0の場合、空の配列を作成
            corners_lidar = np.zeros((0, 3), dtype=np.float32) 
        else: 
            # globak座標系(nuscenesの標準形)の各予測結果をLiDAR座標系に変換
            pred_boxes = np.array(pred_boxes, dtype=np.float32)   # np配列化
            boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))        # LiDAR Object化
            corners_global = boxes.corners.numpy().reshape(-1, 3) # 全ボックスの頂点座標を取得([N*8, 3])
            corners_global = np.concatenate(
                [corners_global,
                 np.ones([corners_global.shape[0], 1])],
                axis=1) # 同次座標系に変換[N*8, 4]
            # LiDAR→globalの変換行列を取得
            l2g = self.get_lidar2global(data_info)
            # global→LiDARに戻す
            corners_lidar = corners_global @ np.linalg.inv(l2g).T
            # [x,y,z]部分のみ抽出
            corners_lidar = corners_lidar[:, :3]
        
        # 予測スコアを取得
        scores = [nusc_results[rid]['detection_score'] for rid in range(len(nusc_results))]
        scores = np.array(scores)

        # 画像上にbboxを投影
        imgs = [] # 描画後の画像格納用リスト
        for view in self.views:
            img = cv2.imread(data_info['cams'][view]['data_path'])
            # draw instances
            # LiDAR座標系の頂点を画像座標へ投影
            corners_img, valid = self.lidar2img(corners_lidar, data_info['cams'][view])
            # 画像内に入っている頂点のみを有効とする
            valid = np.logical_and(
                valid,
                self.check_point_in_img(corners_img, img.shape[0], img.shape[1]))

            # 有効かどうかの判断結果と頂点座標をboxごと8点に並べ直す
            valid = valid.reshape(-1, 8) 
            corners_img = corners_img.reshape(-1, 8, 2).astype(np.int32)

            # スコアしきい値を反映
            score_mask = scores > self.score_thresh

            # 有効な点のみで構成される辺を描画
            for aid in range(valid.shape[0]):
                if not score_mask[aid]:
                    continue
                for index in self.draw_boxes_indexes_img_view:
                    if valid[aid, index[0]] and valid[aid, index[1]]:
                        cv2.line(
                            img,
                            corners_img[aid, index[0]],
                            corners_img[aid, index[1]],
                            color=self.color_map,
                            thickness=self.scale_factor)

            # 描画結果を格納
            imgs.append(img)
        
        # 全カメラの画像を融合
        # 空画像の作成
        img = np.zeros((900 * 2, 1600 * 3, 3), dtype=np.uint8)
        # 前方3カメラ画像を横方向に結合し、空画像の上段に配置する
        img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
        # 後方3カメラ画像を左右反転して、横方向に結合する
        img_back = np.concatenate([imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]], axis=1)
        # 後方カメラ画像をから画像に配置する
        img[900:, :, :] = img_back
        # 画像を縮小する
        img = cv2.resize(img, (int(1600 / self.scale_factor * 3), int(900 / self.scale_factor * 2)))
        
        # 保存
        if self.save_format == 'image':
            cv2.imwrite(os.path.join(self.save_path, f"{frame_idx:06d}.jpg"), img)
        elif self.save_format == 'video':
            self.video_writer.write(img)

        return img

    # def get_corners(self, bboxes):
    #     """bboxの8頂点座標を算出する
    
    #     Args:
    #         bboxes (list): 物体認識結果(list[x,y,z,dx,dy,dz,yaw])
        
    #     Returns:
    #         corners (torch.tensor): 認識結果を描画した画像
    #     """

    #     # bboxの大きさを抽出(扱いやすいようにtensorにしてる)
    #     bboxes_tensor = torch.tensor(bboxes)
    #     bboxes_tensor[:, 2] = bboxes_tensor[:, 2] - bboxes_tensor[:, 5] * 0.5 # 重心zからhの半分を引いて、底面をzに設定
    #     dims = bboxes_tensor[:, 3:6] # 大きさ

    #     # 頂点座標の作成
    #     corners_norm = torch.from_numpy(np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(dtype=dims.dtype)
        
    #     # 描画しやすいように頂点順を並び替える
    #     corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
        
    #     # 中心座標分ずらす
    #     corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])

    #     # 大きさを反映
    #     corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
        
    #     # Z座標で回転させる
    #     corners = self.rotate_corners(corners, bboxes_tensor[:, 6])
        
    #     # bboxの中心座標を足して、絶対座標(LiDAR座標系)に変換する
    #     corners += bboxes_tensor[:, :3].view(-1, 1, 3)
        
    #     return corners

    # def rotate_corners(self, corners, yaws):
    #     """指定した軸を中心として、bboxの頂点をを角度`angles`だけ回転する。

    #     Args:
    #         corners (torch.Tensor): bboxの頂点座標。[N, 8, 3]

    #         yaws (torch.Tensor): bboxのyaw角。[N,]

    #     Returns:
    #         rotated_corners (torch.Tensor): 回転後のbboxの頂点。[N, 8, 3]
    #     """

    #     # Z軸回りの回転行列を作成
    #     rot_sin = torch.sin(yaws) 
    #     rot_cos = torch.cos(yaws)
    #     ones = torch.ones_like(rot_cos)
    #     zeros = torch.zeros_like(rot_cos)

    #     rot_mat_T = torch.stack([
    #                 torch.stack([rot_cos, rot_sin, zeros]),
    #                 torch.stack([-rot_sin, rot_cos, zeros]),
    #                 torch.stack([zeros, zeros, ones])
    #                 ]) # (3, 3, N)
        
    #     # Z軸回りにyaw角の分だけ回転させる
    #     rotated_corners = torch.einsum('aij,jka->aik', corners, rot_mat_T)

    #     return rotated_corners
