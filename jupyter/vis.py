# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion

from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB


class Visualizer():
    """
    認識結果のバウンディングボックスを重畳した画像を作成する。

    Args:
        scale_factor (int): 描画時の画像の縮小率
        color_map (touple): 描画時の線の色
    """

    def __init__(self, scale_factor=3, color_map=(0, 255, 255)):
        # メンバ変数の設定
        self.scale_factor = scale_factor # 描画時の画像の縮小率
        self.color_map = color_map       # 描画時の線の色
        self.draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                            (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                            (2, 6), (3, 7)] # 画像に描画するbboxの辺の順番
        self.views = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'] # 使用するカメラ一覧

    def check_point_in_img(points, height, width):
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

    def lidar2img(points_lidar, camrera_info):
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
        camera2img = camrera_info['cam_intrinsic']
        points_img = points_camera @ camera2img.T
        points_img = points_img[:, :2]
        return points_img, valid

    def get_lidar2global(infos):
        lidar2ego = np.eye(4, dtype=np.float32)
        lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = infos['lidar2ego_translation']
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = Quaternion(
            infos['ego2global_rotation']).rotation_matrix
        ego2global[:3, 3] = infos['ego2global_translation']
        return ego2global @ lidar2ego
    
    def draw_bbox(self, data_info, detections):
        """現在フレームと過去フレームの情報を生成する
    
        Args:
            data_info (dict):  今回フレームのメタ情報
            detections (list): 物体認識結果(list[x,y,z,dx,dy,dx,yaw])
        
        Returns:
            drawed_img (np.array): 認識結果を描画した画像
        """

        if len(pred_boxes) == 0: # 予測結果が0の場合、空の配列を作成
            corners_lidar = np.zeros((0, 3), dtype=np.float32) 
        else: 
            # 各予測結果をLiDAR座標系に変換
            pred_boxes = np.array(detections, dtype=np.float32)   # np配列化
            corners_global = # bboxの頂点座標を算出
            corners_global = boxes.corners.numpy().reshape(-1, 3) # 全ボックスの頂点座標を取得([N*8, 3])
            corners_global = np.concatenate(
                [corners_global,
                np.ones([corners_global.shape[0], 1])],
                axis=1) # 同次座標系に変換[N*8, 4]
            # LiDAR→globalの変換行列を取得
            l2g = get_lidar2global(infos)
            # global→LiDARに戻す
            corners_lidar = corners_global @ np.linalg.inv(l2g).T
            # [x,y,z]部分のみ抽出
            corners_lidar = corners_lidar[:, :3]
        
        # 予測結果の有効フラグを作成
        pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=np.bool)
        # 予測スコアを取得
        scores = [
            pred_res[rid]['detection_score'] for rid in range(len(pred_res))
        ]

    def main():
        # コマンドライン引数の読み込み
        args = parse_args()

        # load predicted results
        res = json.load(open(args.res, 'r'))
        
        # load dataset information
        info_path = args.root_path + '/bevdetv3-nuscenes_infos_%s.pkl' % args.version
        dataset = pickle.load(open(info_path, 'rb'))
        
        # prepare save path and medium
        vis_dir = args.save_path
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        print('saving visualized result to %s' % vis_dir)
        
        # 可視化用の設定
        scale_factor = args.scale_factor # 画像の描画時の比率

        # 動画保存時の設定
        if args.format == 'video':
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            vout = cv2.VideoWriter(
                os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
                args.fps, (int(1600 / scale_factor * 3),
                        int(900 / scale_factor * 2 + canva_size)))

        # BEVでboxを描く辺の定義
        draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # 画像上で3Dbboxを描く辺の定義
        draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                    (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                    (2, 6), (3, 7)]
        
        # 使用するカメラ名一覧
        views = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
            'CAM_BACK_LEFT',  'CAM_BACK',  'CAM_BACK_RIGHT'
        ]
        print('start visualizing results')
        
        # 描画処理
        for cnt, infos in enumerate(dataset['infos'][:min(args.vis_frames, len(dataset['infos']))]): # 先頭フレーム〜設定フレーム数だけ可視化
            # 10フレームごとに進捗表示
            if cnt % 10 == 0:
                print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['infos']))))
            
            # 現在フレームの予測結果を取得
            # collect instances
            pred_res = res['results'][infos['token']]

            # 各予測を[x,y,z,dx,dy,dz,yaw]に合わせる
            pred_boxes = [
                pred_res[rid]['translation'] + 
                pred_res[rid]['size'] + 
                [Quaternion(pred_res[rid]['rotation']).yaw_pitch_roll[0] + np.pi / 2] 
                for rid in range(len(pred_res))
            ]

            if len(pred_boxes) == 0: # 予測結果が0の場合、空の配列を作成
                corners_lidar = np.zeros((0, 3), dtype=np.float32) 
            else: 
                # 各予測結果をLiDAR座標系に変換
                pred_boxes = np.array(pred_boxes, dtype=np.float32)   # np配列化
                boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))        # LiDAR Object化
                corners_global = boxes.corners.numpy().reshape(-1, 3) # 全ボックスの頂点座標を取得([N*8, 3])
                corners_global = np.concatenate(
                    [corners_global,
                    np.ones([corners_global.shape[0], 1])],
                    axis=1) # 同次座標系に変換[N*8, 4]
                # LiDAR→globalの変換行列を取得
                l2g = get_lidar2global(infos)
                # global→LiDARに戻す
                corners_lidar = corners_global @ np.linalg.inv(l2g).T
                # [x,y,z]部分のみ抽出
                corners_lidar = corners_lidar[:, :3]
            
            # 予測結果の有効フラグを作成
            pred_flag = np.ones((corners_lidar.shape[0] // 8, ), dtype=np.bool)
            # 予測スコアを取得
            scores = [
                pred_res[rid]['detection_score'] for rid in range(len(pred_res))
            ]

            # GTの描画用設定
            if args.draw_gt:
                gt_boxes = infos['gt_boxes']
                gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
                width = gt_boxes[:, 4].copy()
                gt_boxes[:, 4] = gt_boxes[:, 3]
                gt_boxes[:, 3] = width
                corners_lidar_gt = \
                    LB(infos['gt_boxes'],
                    origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
                corners_lidar = np.concatenate([corners_lidar, corners_lidar_gt],
                                            axis=0)
                gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
                pred_flag = np.concatenate(
                    [pred_flag, np.logical_not(gt_flag)], axis=0)
                scores = scores + [0 for _ in range(infos['gt_boxes'].shape[0])]
            
            # スコアをソート
            scores = np.array(scores, dtype=np.float32)
            sort_ids = np.argsort(scores)

            # 画像上にbboxを投影
            imgs = [] # 描画後の画像格納用リスト
            for view in views:
                img = cv2.imread(infos['cams'][view]['data_path'])
                # draw instances
                # LiDAR座標系の頂点を画像座標へ投影
                corners_img, valid = lidar2img(corners_lidar, infos['cams'][view])
                # 画像内に入っている頂点のみを有効とする
                valid = np.logical_and(
                    valid,
                    check_point_in_img(corners_img, img.shape[0], img.shape[1]))

                # 有効かどうかの判断結果と頂点座標をboxごと8点に並べ直す
                valid = valid.reshape(-1, 8) 
                corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)

                # 有効な点のみで構成される辺を描画
                for aid in range(valid.shape[0]):
                    for index in draw_boxes_indexes_img_view:
                        if valid[aid, index[0]] and valid[aid, index[1]]:
                            cv2.line(
                                img,
                                corners_img[aid, index[0]],
                                corners_img[aid, index[1]],
                                color=color_map[int(pred_flag[aid])],
                                thickness=scale_factor)

                # 描画結果を格納
                imgs.append(img)

            # bird-eye-view
            canvas = np.zeros((int(canva_size), int(canva_size), 3),
                            dtype=np.uint8)
            # draw lidar points
            lidar_points = np.fromfile(infos['lidar_path'], dtype=np.float32)
            lidar_points = lidar_points.reshape(-1, 5)[:, :3]
            lidar_points[:, 1] = -lidar_points[:, 1]
            lidar_points[:, :2] = \
                (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
            for p in lidar_points:
                if check_point_in_img(
                        p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                    color = depth2color(p[2])
                    cv2.circle(
                        canvas, (int(p[0]), int(p[1])),
                        radius=0,
                        color=color,
                        thickness=1)

            # draw instances
            corners_lidar = corners_lidar.reshape(-1, 8, 3)
            corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
            bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
            bottom_corners_bev = \
                (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
            bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
            center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
            head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
            canter_canvas = \
                (center_bev + show_range) / show_range / 2.0 * canva_size
            center_canvas = canter_canvas.astype(np.int32)
            head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
            head_canvas = head_canvas.astype(np.int32)

            for rid in sort_ids:
                score = scores[rid]
                if score < args.vis_thred and pred_flag[rid]:
                    continue
                score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
                color = color_map[int(pred_flag[rid])]
                for index in draw_boxes_indexes_bev:
                    cv2.line(
                        canvas,
                        bottom_corners_bev[rid, index[0]],
                        bottom_corners_bev[rid, index[1]],
                        [color[0] * score, color[1] * score, color[2] * score],
                        thickness=1)
                cv2.line(
                    canvas,
                    center_canvas[rid],
                    head_canvas[rid],
                    [color[0] * score, color[1] * score, color[2] * score],
                    1,
                    lineType=8)

            # 画像6枚とBEV結果を融合
            # 空画像の作成
            img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3), dtype=np.uint8)
            # 前方3カメラ画像を横方向に結合し、空画像の上段に配置する
            img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
            # 後方3カメラ画像を左右反転して、横方向に結合する
            img_back = np.concatenate(
                [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
                axis=1)
            # 後方カメラ画像をから画像に配置する
            img[900 + canva_size * scale_factor:, :, :] = img_back
            # 画像を縮小する
            img = cv2.resize(img, (int(1600 / scale_factor * 3),
                                int(900 / scale_factor * 2 + canva_size)))
            # BEV画像を配置
            w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
            img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
                w_begin:w_begin + canva_size, :] = canvas
            
            # 保存
            if args.format == 'image':
                cv2.imwrite(os.path.join(vis_dir, '%s.jpg' % infos['token']), img)
            elif args.format == 'video':
                vout.write(img)
        
        # 終了時の処理
        if args.format == 'video':
            vout.release()


    if __name__ == '__main__':
        main()
