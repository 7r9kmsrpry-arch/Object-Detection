import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32, auto_fp16
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
# from projects.mmdet3d_plugin.VAD.utils.traj_lr_warmup import get_traj_warmup_loss_weight
# from projects.mmdet3d_plugin.VAD.utils.map_utils import (
#     normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
# )


# VADのHead処理部分
@HEADS.register_module()
class VADHead(DETRHead):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False, # デコーダ各層で参照点(ボックス)を段階的に更新するか。True だと各層で回帰精度が上がる反面、計算増
                 as_two_stage=False,    # 2段階検出(Trueの場合、encoder側で提案領域を生成)
                 transformer=None,      # DETR系の Encoder/Decoder 構成（層数、ヘッド数、hidden dim など）
                 bbox_coder=None,       # ボックス表現のエンコード/デコード（cx,cy,w,h, yaw 等→ネット出力/ターゲットに合わせる）
                 num_cls_fcs=2,         # クラス分類ヘッドの全結合層数(中間FCの数 + 最後の出力FC)
                 code_weights=None,     # 回帰コード（ボックス成分やベクトル成分）ごとの損失重み。L1/GIoU等に掛ける係数
                 bev_h=30,              # BEVの範囲
                 bev_w=30,              # BEVの範囲
                 **kwargs):

        # メンバ変数の設定
        self.bev_h = bev_h        # BEVの範囲(奥行き)
        self.bev_w = bev_w        # BEVの範囲(幅)
        self.fp16_enabled = False # FP16の有効化フラグ

        # DETR系フラグの反映
        self.with_box_refine = with_box_refine # デコーダ各層で参照点を段階的に更新するか(デフォルト：True)
        self.as_two_stage    = as_two_stage    # 2段階検出(デフォルト：False)
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage

        # 回帰コードの次元の設定(bbox_coderの出力次元と合わせる必要ある！)
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10 # 例：[x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]

        # 回帰の各次元に掛ける損失の重みの設定
        if code_weights is not None:
            self.code_weights = code_weights
        else: # デフォルトでここに入る
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2] # 速度はノイズが乗りやすく不安定なので弱めている

        # bbox_coderの設定を反映
        self.bbox_coder = build_bbox_coder(bbox_coder)    # bbox_coderを構築
        self.pc_range = self.bbox_coder.pc_range          # 認識範囲
        self.real_w = self.pc_range[3] - self.pc_range[0] # BEV平面の実世界のサイズ(幅)
        self.real_h = self.pc_range[4] - self.pc_range[1] # BEV平面の実世界のサイズ(奥行き)
        self.num_cls_fcs = num_cls_fcs - 1                # 中間のfcsの総数を取得(出力層のfc層を省いて、ループで構築できるようにしてる)
        
        # 親クラス(DETRHeadの初期化)
        super(VADHead, self).__init__(*args, transformer=transformer, **kwargs)

        # bboxの重みををnn.Parameter（非学習：固定の重み）に設定
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        
    # Headの層を構築(親クラス(DETRHead)の初期時に呼ばれる)
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        # 物体検出のクラス分類ヘッド
        cls_branch = []
        # 中間層の生成
        for _ in range(self.num_reg_fcs): # num_reg_fcsはDETRHeadで定義されたデフォルト値を使ってる！
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        # 出力層の生成
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        # 物体検出の位置の回帰ヘッド
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        # 同じ層構成のヘッド（またはブロック）をN個、独立パラメータで複製して、ModuleListに入れる(形は同じ、重みは別な複数ヘッドを手早く作る)
        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # 各ブランチで何回（何層分）予測ヘッドを用意するかを設定
        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        # ↑two-Stage モードのとき（as_two_stage=True）、Encoder出力特徴から「一次予測box（proposal）」を出す処理を行う。
        # そのとき、回帰ヘッド群（reg_branch）のうち最後の1個を使ってproposalを作る。
        # 後々、self.reg_branches = _get_clones(reg_branch, num_pred)で「Encoderでのproposal予測」に使うために +1 して (num_decoder_layers + 1) 個作る

        # デコーダー総数の取得
        num_decoder_layers = 1     # 物体検出用のdecoder層数
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers

        # num_motion_decoder_layers = 1 # 物体の将来軌跡用のdecoder総数(1層固定)
        
        # 各層で予測する回数の決定（two-stage対応）
        # one-stage(標準DETR)：各ブランチは 「デコーダ層の回数」だけ予測ヘッドを用意（＝各層で出力/リファイン）
        # two-stage：「デコーダ層の回数」 + 1 にする。(+1 はEncoder段での提案（proposal）出力分。Deformable-DETR系の流儀で、Encoder出力からの初期予測と、各Decoder層でのリファイン予測を足し合わせた回数になる)
        # この回数が、ヘッド複製数と出力回数になる。
        num_pred = (num_decoder_layers + 1) if self.as_two_stage else num_decoder_layers

        # motion_num_pred = (num_motion_decoder_layers + 1) if \
        #     self.as_two_stage else num_motion_decoder_layers

        # 各ブランチ用の予測ヘッドを「層ごとに別パラメータで持つか（複製）」or「全層で同じパラメータを共有するか（共有）」を設定
        if self.with_box_refine: # 層ごとに別パラメータを設定(層を追うごとに出力を段階的にリファインできるように、層ごとに最適なパラメータを学習させるため)
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            # self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            # self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
        else: # 全層で重みを共有(各層で適用される特徴は違うので出力は変わるが、予測ヘッドの重みは全層で同一)
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            # self.traj_branches = nn.ModuleList(
            #     [traj_branch for _ in range(motion_num_pred)])
            # self.traj_cls_branches = nn.ModuleList(
            #     [traj_cls_branch for _ in range(motion_num_pred)])

        # one-stageのときに使用する、学習クエリ埋め込みを用意
        # two-stageでは、Encoder側でproposal（初期参照点）を作ってDecoderに渡すが、
        # one-stageでは、proposalが無いので、学習可能な埋め込み(learnable queries) を自前で用意する
        if not self.as_two_stage:            
            # 物体検出用の埋め込みベクトルを生成
            # 出力次元が 2 * embed_dims なのは、前半と後半を分けて使う前提(DETR系ではデコーダに渡すときに2本のベクトルを入力する！)
            # 前半：クエリの内容埋め込み(query content:何を探すかの内容ベクトル)、後半：位置/参照点用の埋め込み (query pos / reference init:どこを見るかの“位置・参照点”の手がかり)
            # forward() で torch.chunk(..., 2, dim=-1) しているはず
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        
    # 重みの初期化(DETRHeadの初期化時に呼ばれる)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        # エンコーダー・デコーダー層の重みを初期化
        self.transformer.init_weights()

        # クラス予測ブランチの重みを初期化
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    # @auto_fp16(apply_to=('mlvl_feats'))
    #@force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                bev_embdeds # bev埋め込みlist of [bs, c, H, W]
                #mlvl_feats,      # 画像特徴量([[T、N、C、H、W]],一番外側はマルチスケール)
                #img_metas,        # 画像や幾何行列など“各フレームのメタ情報”（[T] 構造のリスト）
                #prev_bev=None,   # 前回時刻までのBEVのリスト
                #only_bev=False,  # Trueなら、BEVエンコーダ部分のみを実行して、デコーダ以降（検出など）はスキップ
                #test_mode=False, # 推論モード(Trueの場合、BEV生成部分をC++/CUDAで高速化)
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
                各デコーダ層の物体クラス確率
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
                各層での物体ボックス (正規化済み cx, cy, w, l, cz, h, theta, vx, vy)
        """
        
        #bs, num_cam, _, _, _ = mlvl_feats[0].shape                  # バッチ(時系列数)、カメラ数の取得(マルチスケールなのでH,Wは異なるが、バッチとカメラ個数は同じ)
        bev_embded = bev_embdeds[0]
        dtype = bev_embded.dtype                                 # 特徴マップのdtypeの取得
        object_query_embeds = self.query_embedding.weight.to(dtype) # 物体検出用クエリの初期埋め込みベクトルを取得
                
        # 物体検出の実行
        outputs = self.transformer(
                bev_embded,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501 box refineありなら、各デコーダ層で回帰ブランチを共有/クローンして逐次洗練
                cls_branches=self.cls_branches if self.as_two_stage else None,     # two-stageなら、中間の候補を使った2段目分類ブランチを使用
        )

        # bev_embed：BEV特徴[H*W, B, C], 
        # hs：各層の物体クエリ出力[L, B, N_obj, C]  (Deep Supervision用に、各層の結果を出力)
        # init_reference：物体検出クエリの初期参照点（x,y,z,wなど）[B, N_obj, 4]
        # inter_references：各層の参照点[L, B, N_obj, 4]
        bev_embed, hs, init_reference, inter_references = outputs 

        # 物体検出の出力用設定
        hs = hs.permute(0, 2, 1, 3) # [L, B, N_obj, C] -> [L, N_obj, B, C] 後段のヘッド（分類・回帰）が「バッチ次元が手前」にある前提
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []

        # 物体検出のデコード
        # 各デコーダ層の出力から「分類」と「回帰（座標）」を作り、参照点で逐次リファイン（iterative refinement）して最終座標に落とす
        for lvl in range(hs.shape[0]):
            # 参照点を決定
            if lvl == 0:
                reference = init_reference            # 初期参照点（学習された初期位置）
            else:
                reference = inter_references[lvl - 1] # 1つ前の層で更新された参照点(層目以降は前層の参照点を使って位置を更新していく（= iterative refinement)
            reference = inverse_sigmoid(reference)    # logit）空間にいったん戻してから残差を足すのが DETR/Deformable-DETR 系の定番

            # 分類・回帰の生出力
            outputs_class = self.cls_branches[lvl](hs[lvl]) # [B, Q, num_classes]で、クラスlogitsを予測 , hs[lvl] はこの層のオブジェクトクエリ埋め込み（[B, Q, D]）
            tmp = self.reg_branches[lvl](hs[lvl])           # reg_branches はバウンディング要素（中心・大きさ・向きなど）を予測

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3 # 参照点は[x,y,z]の3要素の想定

            # 参照点に対する残差を加算し、sigmoid で [0,1] 範囲に正規化
            # bboxは(cx, cy, w, l, cz, h, theta, vx, vy)の順番
            # 中心(x, y)の残差をlogit空間で足してから sigmoid → [0,1] 正規化 BEV 座標へ(sigmoid 空間で直接足すより、logit 空間（inverse_sigmoid）で残差を足す方が学習が安定し、境界付近（0や1に近い）でも勾配が扱いやすくなる)
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach()) # (x, y)の正規化値を保存
                                                                      # outputs_coords_bevは可視化/ログ/評価で使う中間結果の保管所のため、clone().detach()でこの保存物に勾配が流れないようにする
                                                                      # .detach()：計算グラフ（autograd）から切り離す、clone()：参照を切って副作用防止
            # 中心(z)の残差をlogit空間で足してから sigmoid → [0,1] 正規化
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            # 正規化座標 → 物理座標（メートル系）へ変換
            tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]) # x(倍率を範囲でかけて、最小値に足す)
            tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]) # y
            tmp[..., 4:5] = (tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]) # z

            # この層の出力を蓄積(後で層ごとにlossを計算(Deep Supervision）するため)
            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class) # 分類
            outputs_coords.append(outputs_coord)  # メートル系の座標

        # 各層ごとにリストへ貯めていた出力を、最終的に層次元Lを先頭に持つ1個のテンソルへまとめる(stackは新軸を作る点でcatと異なる（cat は既存軸方向に連結）)
        # 直前に clone().detach() して保存したものは 別のリスト（例：outputs_coords_bev）で、ここは学習対象（勾配付き）の主経路をまとめている
        # Deep Supervision（層ごとの損失）をfor文で処理しやすくする
        outputs_classes = torch.stack(outputs_classes)             # [L, B, A, num_cls]
        outputs_coords = torch.stack(outputs_coords)               # [L, B, A, box_dim]

        # 出力の作成
        outs = {
            'bev_embed': bev_embed, # BEV特徴量
            'all_cls_scores': outputs_classes, # 物体検出のクラスlogitsを層ごとにまとめたもの([L, B, A, num_cls])
            'all_bbox_preds': outputs_coords,  # 物体検出のボックス回帰（物理座標に変換済）  ([L, B, A, box_dim])
            'enc_cls_scores': None,      # enc_~は、as_two_stage 系のエンコーダ提案が有効な場合の予測
            'enc_bbox_preds': None,
        }

        return outs

    # １枚の画像に対する、1層分のターゲットを生成(multi_applyで勝手に1枚ずつになって入力される)
    def _get_target_single(self,
                           cls_score, # 各クエリのクラス Logits [num_query, num_classes]
                           bbox_pred, # ボックスの座標
                           gt_labels, # 正解ラベル
                           gt_bboxes, # [num_gts, 9] で BEV 3D+速度（[x, y, z, w, l, h, yaw, vx, vy]）
                           gt_bboxes_ignore=None): # 無視 GT（この実装は基本未対応）
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        
        # クエリ数Aを取得
        num_bboxes = bbox_pred.size(0)


        # assigner and sampler
        gt_bbox_c   = gt_bboxes.shape[-1] # GTの回帰ターゲットに使う次元幅（予測 bbox_predは10次元でも、教師は9次元まで等、実装に合わせる）
        num_gt_bbox = gt_bboxes.shape[0]  # GTの数を取得

        # Hungarianマッチング
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore)

        # 対応付け結果から、正例/負例のインデックスを取り出す
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # 分類ラベル設定
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long) # 背景クラスIDをnum_classesとして全クエリ初期化
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]             # 正例位置だけGTクラスに置換
        label_weights = gt_bboxes.new_ones(num_bboxes)                                 # label_weights は全1（必要なら後段でFocal等の重み付け）

        # ボックス回帰ターゲット
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # 正例に割り当てられた GT ボックスを、そのクエリ位置の教師ターゲットに書き込む
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        # return (
        #     labels, label_weights, bbox_targets, bbox_weights, traj_targets,
        #     traj_weights, traj_masks.view(-1, self.fut_ts, 2)[..., 0],
        #     pos_inds, neg_inds
        # )

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    # 各画像ごとに「予測 ↔ GT」を対応付け（assign）して、分類・回帰・将来軌跡用の教師ターゲットを作る
    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """

        # ignore 非対応を明示
        assert gt_bboxes_ignore_list is None, 'Only supports for gt_bboxes_ignore setting to None.'
        
        # バッチサイズ（=画像枚数）を取得し、ignoreのダミーを複製(中身は全部None)
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        # 画像ごとのターゲット生成を一括適用
        # multi_applyは、対応するリストのi番目を取り出して
        # self._get_target_single( cls_scores[i], bbox_preds[i], gt_labels[i], gt_bboxes[i], gt_attr_labels[i], gt_ignore[i] )を全画像分実行し、
        # 各返り値を“画像ごとのリスト”にまとめて返す
        (labels_list, label_weights_list, bbox_targets_list, 
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list
         )

        # 正負サンプル数の総計(損失の正規化に使用)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg)
    
    # 物体検出＋将来軌跡用の1デコーダ層分の損失を計算
    def loss_single(self,
                    cls_scores,                  # その層のクエリ(A個)に対するクラスlogits [B, A, C_cls]
                    bbox_preds,                  # その層のボックス回帰（sigmoid済みの正規化 (cx, cy, w, h)) bbox_preds: [B, A, 4]
                    gt_bboxes_list,              # バッチ各画像の GT ボックス。上位で [cx, cy, cz, …] 形式に整形済み
                    gt_labels_list,              # 各 GT のクラス ID
                    gt_bboxes_ignore_list=None): # 無視 GT。上位で assert None なので通常 None
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        
        # 各バッチ画像ごとに予測を分割し、対応付けして教師ターゲットを作成
        num_imgs = cls_scores.size(0) # バッチサイズの取得
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)] # テンソルを画像単位のリストに分割
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)] # テンソルを画像単位のリストに分割
        # 各バッチ画像に対して、予測クエリ（A本）とGTの対応付け（Hungarianなど）を行い、分類・回帰（＋属性）の教師を作る
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        # 
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    # 損失計算
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list, # 画像(バッチ)ごとのGT-2D/BEVボックス [tl_x, tl_y, br_x, br_y] (docstring上は左の形式だが、Headは9次元出力なので要確認)
             gt_labels_list, # 各GTのクラスID
             preds_dicts,    # 辞書形式の予測結果(outs)
             gt_bboxes_ignore=None,     # VADの呼び出しでは引数に入れてない
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # 無視クラスが定義されている場合、エラーとする
        assert gt_bboxes_ignore is None, f'{self.__class__.__name__} only supports ' f'for gt_bboxes_ignore setting to None.'

        # 予測結果の抽出
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        # enc_cls_scores = preds_dicts['enc_cls_scores'] # two-stage用
        # enc_bbox_preds = preds_dicts['enc_bbox_preds'] # two-stage用
        
        # 物体検出の損失算出
        # デコーダーの層数とデバイスの取得(以降で、GTテンソルを同じdeviceに載せ替える)
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        # GTボックスのフォーマット整形([cx, cy, cz, w, l, h, yaw, ...]形式に揃える)
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        # Deep Supervision用にGTを層数ぶん複製
        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        # multi_applyで層ごとにloss_singleを一気に適用
        # mmcv.runner.multi_apply：同じ関数を、対応する引数列の要素ごとに適用して、結果をタプルのリストで返す
        # losses_cls = [loss_cls_l0, loss_cls_l1, ..., loss_cls_l{L-1}] のような層ごとの損失リストが得られる(後段で加重和や平均をとって最終lossにするのが定石)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_gt_bboxes_list, all_gt_labels_list, all_gt_bboxes_ignore_list)
        
        # デコーダーの最終層の損失を格納
        loss_dict = dict()
        loss_dict['loss_cls']      = losses_cls[-1]
        loss_dict['loss_bbox']     = losses_bbox[-1]

        # 最終層以外の損失も追加(Deep Supervision用)
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        # as_two_stage 系のエンコーダ提案が有効な場合の損失
        # loss of proposal generated from encode feature map.
        # if enc_cls_scores is not None:
        #     binary_labels_list = [
        #         torch.zeros_like(gt_labels_list[i])
        #         for i in range(len(all_gt_labels_list))
        #     ]
        #     enc_loss_cls, enc_losses_bbox = \
        #         self.loss_single(enc_cls_scores, enc_bbox_preds,
        #                          gt_bboxes_list, binary_labels_list,
        #                          gt_bboxes_ignore)
        #     loss_dict['enc_loss_cls'] = enc_loss_cls
        #     loss_dict['enc_loss_bbox'] = enc_losses_bbox

        return loss_dict

    # デコーダ出力を最終の可読な結果(3D bbox・スコア・ラベル・軌跡など)に整形
    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, 
                   preds_dicts, # 予測結果(forwardの出力) 
                   img_metas, 
                   rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        # bboxを取得
        det_preds_dicts = self.bbox_coder.decode(preds_dicts) # [B, dict(key='bboxes', 'scores', 'labels')]←同じbboxが重複している可能性ある
        
        # バッチ数を取得
        num_samples = len(det_preds_dicts)

        # 出力用のリストを作成
        ret_list = []

        # バッチごとに処理
        for i in range(num_samples):
            preds = det_preds_dicts[i] # i番目のバッチの物体情報を取得
            bboxes = preds['bboxes']   # bbox情報を取得
            # Zの定義を中心座標から底面座標に変換する(評価APIや可視化の座標系と一致させる)
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5 # 重心zからhの半分を引いて、底面をzに設定
            # bboxを指定した座標系で宣言(LiDAR)
            code_size = bboxes.shape[-1] # bboxの次元数
            bboxes = img_metas[i]['box_type_3d'](bboxes, code_size) # img_metas[i]['box_type_3d'] には、LiDARを指定
                                                                    # MMDet3D系ではこの指定により、生の[x,y,z,w,l,h,yaw,…]配列をLiDARInstance3DBoxesという専用クラスに包む（callable）
                                                                    # 以降はこの型のコンベンション（座標軸・回転軸・zの定義など）**に従って、回転・平行移動・BEV投影・評価などのユーティリティを安全に使えます
            scores = preds['scores'] # クラススコア(最大のクラススコア)
            labels = preds['labels'] # クラスラベル(クラススコアが最大のラベル)

            # 出力に追加
            ret_list.append([bboxes, scores, labels])
        
        return ret_list

