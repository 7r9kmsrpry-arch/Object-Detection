import torch
import numpy as np

from .lidar_box3d import LiDARInstance3DBoxes


class BboxDecoder():
    """Headの出力からbboxを生成する
    
    Args:
        post_center_range (list[float]): 中心座標の有効範囲
        max_num (int): 最終的に残すbboxの個数
        num_classes (int): 認識クラス数
        score_threshold (float): bboxのスコアしきい値
    """
    
    def __init__(self, post_center_range, max_num=100, num_classes=10, score_threshold=None):
        
        # メンバ変数の初期化
        self.post_center_range = post_center_range 
        self.max_num = max_num                     
        self.num_classes = num_classes             
        self.score_threshold = score_threshold     

    def get_bbox(self, preds_dicts):
        """Headの出力からbboxを生成する。

        Args:
            preds_dicts (list): スコアと認識結果のリスト [[B, A, num_cls], [B, A, box_dims]]

        Returns:
            bboxes (list): bboxのリスト
            scores (list): scoreのリスト
            labels (list): labelのリスト
        """

        # bboxを取得
        det_preds_dicts = self.decode(preds_dicts) # [B, dict(key='bboxes', 'scores', 'labels')]←同じbboxが重複している可能性ある

        # bbox情報を抽出
        bboxes = det_preds_dicts['bboxes']   # bbox情報を取得
        # Zの定義を中心座標から底面座標に変換する(評価APIや可視化の座標系と一致させる)
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5 # 重心zからhの半分を引いて、底面をzに設定
        # bboxを指定した座標系で宣言(LiDAR)
        code_size = bboxes.shape[-1] # bboxの次元数
        bboxes = LiDARInstance3DBoxes(bboxes, code_size) # img_metas[i]['box_type_3d'] には、LiDARを指定
                                                         # MMDet3D系ではこの指定により、生の[x,y,z,w,l,h,yaw,…]配列をLiDARInstance3DBoxesという専用クラスに包む（callable）
                                                         # 以降はこの型のコンベンション（座標軸・回転軸・zの定義など）**に従って、回転・平行移動・BEV投影・評価などのユーティリティを安全に使えます
        scores = det_preds_dicts['scores'] # クラススコア(最大のクラススコア)
        labels = det_preds_dicts['labels'] # クラスラベル(クラススコアが最大のラベル)

        # cpuの転送
        bboxes_cpu = bboxes.to('cpu')
        scores_cpu = scores.cpu()
        labels_cpu = labels.cpu()

        return bboxes_cpu, scores_cpu, labels_cpu

    def decode(self, preds_dicts):
        """予測結果から3Dバウンディングボックスをデコードする。

        Args:
            preds_dicts (list):
                Headのforward出力を格納したリスト。[all_cls_scores, all_bbox_preds]
                - all_cls_scores (Tensor):
                    分類ヘッドの出力。
                    [バッチサイズ, クエリ数, クラス数]。
                
                - all_bbox_preds (Tensor):
                    回帰ヘッドの出力。
                    [バッチサイズ, クエリ数, bbox次元数]。
                    bbox は正規化表現(cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)で表される。

        Returns:
            prediction (dict):
                デコード後の予測結果を格納した辞書。
                主に以下のキーを含む。
                - 'bboxes':
                    最終的に残した3Dバウンディングボックス。
                - 'scores':
                    各BBoxのスコア。
                - 'labels':
                    各BBoxのクラスラベル。
        """

        # 最終層の結果を取り出す
        all_cls_scores = preds_dicts[0] # [B, Q, num_cls]
        all_bbox_preds = preds_dicts[1] # [B, Q, box_dims]
        
        # バッチサイズを取得
        batch_size = all_cls_scores.size()[0]
        assert batch_size == 1

        # decode
        prediction = self.decode_single(all_cls_scores[0], all_bbox_preds[0])

        return prediction

    def decode_single(self, cls_scores, bbox_preds):
        """1サンプル分の分類結果とBBox回帰結果から、最終的な3Dバウンディングボックス候補を生成する。

            Args:
                cls_scores (Tensor):
                    分類ヘッドの出力。
                    [num_query, cls_out_channels]。
                    各queryに対する各クラスのスコアを表す。

                bbox_preds (Tensor):
                    回帰ヘッドの出力。
                    [num_query, box_dim]。
                    各queryに対応する正規化済みBBoxパラメータ
                    (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy)を表す。

            Returns:
                prediction (dict):
                    デコード後の予測結果を格納した辞書。
                    主に以下のキーを含む。
                    - 'bboxes':
                        最終的に残した3Dバウンディングボックス。
                    - 'scores':
                        各BBoxのスコア。
                    - 'labels':
                        各BBoxのクラスラベル。
        """ 

        # 最終的に候補として残す上限を取得
        max_num = self.max_num

        # 分類スコアをlogitsから確率に変換
        cls_scores = cls_scores.sigmoid()

        # 全クエリの全クラスの候補の中からスコア上位max_num個を抽出
        scores, indexs = cls_scores.view(-1).topk(max_num)
        # どのクラスに属するかを取得
        labels = indexs % self.num_classes
        # どのクエリのboxかを取得
        bbox_index = indexs // self.num_classes
        # スコア上位top_kのbboxを抽出(同じbboxが複数回選ばれる可能性がある)
        bbox_preds = bbox_preds[bbox_index]
       
        # top-kで選んだ候補bboxを実際のbox形式に直して、さらにスコアで候補を絞る
        final_box_preds = self.denormalize_bbox(bbox_preds) # bboxを[cx, cy, cz, w, l, h, rot, vx, vy]の形式に変換
        final_scores = scores 
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None: # defaultでNoneのため、skip
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        # 中心座標での選別
        if self.post_center_range is not None:
            # (x,y,z)が指定範囲内かどうかのmaskを作成
            self.post_center_range = torch.as_tensor(self.post_center_range, device=scores.device, dtype=torch.float32)
            mask = (final_box_preds[..., :3] >= self.post_center_range[:3]).all(1)  # 下限のmask
            mask &= (final_box_preds[..., :3] <= self.post_center_range[3:]).all(1) # 上限のmask

            # スコア条件のmask(現状なし)
            if self.score_threshold:
                mask &= thresh_mask

            # 有効な結果のみを抽出
            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]

            # 最終的な結果を保存
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
            }
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')

        return predictions_dict

    def denormalize_bbox(self, normalized_bboxes):
        """Headが出力した正規化表現のBBoxパラメータを、実際に使うBBox表現へ変換する。

        Args:
            normalized_bboxes (torch.Tensor):
                正規化済みのBBoxテンソル。
                末尾次元は通常、以下の順を想定する。

                - x中心
                - y中心
                - log(w)
                - log(l)
                - z中心
                - log(h)
                - sin(yaw)
                - cos(yaw)
                - vx (任意)
                - vy (任意)

                形状は `[..., box_dim]`。

        Returns:
            torch.Tensor:
                復元後のBBoxテンソル。
                末尾次元は通常、以下の順になる。

                - x中心
                - y中心
                - z中心
                - w
                - l
                - h
                - yaw
                - vx (存在する場合)
                - vy (存在する場合)

                形状は `[..., out_dim]`。
        """

        # yaw角の作成
        rot_sine = normalized_bboxes[..., 6:7]   # sin成分
        rot_cosine = normalized_bboxes[..., 7:8] # con成分
        rot = torch.atan2(rot_sine, rot_cosine)  # yawに変換

        # 中心座標を取り出す
        cx = normalized_bboxes[..., 0:1]
        cy = normalized_bboxes[..., 1:2]
        cz = normalized_bboxes[..., 4:5]
    
        # サイズ成分を取り出す
        w = normalized_bboxes[..., 2:3]
        l = normalized_bboxes[..., 3:4]
        h = normalized_bboxes[..., 5:6]

        # log出力を元に戻す
        w = w.exp() 
        l = l.exp() 
        h = h.exp() 

        # 速度がある場合、速度情報を追加
        if normalized_bboxes.size(-1) > 8:
            # velocity 
            vx = normalized_bboxes[:, 8:9]
            vy = normalized_bboxes[:, 9:10]
            denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
        else:
            denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)

        return denormalized_bboxes

    


