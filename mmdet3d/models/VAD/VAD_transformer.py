import torch
import numpy as np
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.utils import ext_loader
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from mmdet.models.utils.builder import TRANSFORMER
from torchvision.transforms.functional import rotate
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from .modules.decoder import CustomMSDeformableAttention
from .modules.temporal_self_attention import TemporalSelfAttention
from .modules.spatial_cross_attention import MSDeformableAttention3D

# カスタムクラス
#from projects.mmdet3d_plugin.models.encoders import build_bev_encoder
import time

# Deformable AttentionのC++実装を使用する関数バインド設定
ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

# VADの物体検出・Map検出のTransformer定義
@TRANSFORMER.register_module() #MMCVのTRANSFORMERレジストリに登録(configでtype='VADPerceptionTransformer'と記載)
class VADPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 #num_feature_levels=4,         # FPNのマルチスケール段数(config設定なし→デフォルトの4が設定される)
                 #num_cams=6,                   # カメラ台数
                 two_stage_num_proposals=300,  # DETR系のTwo-Stage運用時、エンコーダ特徴から作る仮クエリ数：300
                 encoder=None,                 # Fast-BEV+BEVDet4Dのbev埋め込み
                 decoder=None,                 # DetectionTransformerDecoder
                 embed_dims=256,               # 埋め込み次元
                 **kwargs):                    # 名前付き引数として宣言していない追加の引数を辞書として受け取る
        
        # 親クラスの初期化
        super(VADPerceptionTransformer, self).__init__(**kwargs)
                
        # 物体検出のDecoderの構築
        if decoder is not None:
            self.decoder = build_transformer_layer_sequence(decoder)
        else:
            self.decoder = None
            
        # メンバ変数定義
        self.embed_dims = embed_dims
        #self.num_cams = num_cams
        self.fp16_enabled = False
        self.two_stage_num_proposals = two_stage_num_proposals

        # 内部の層を構築
        self.init_layers()

        # # debug用
        # self.debug = False 
        # self.cnt   = 0
        # self.bev_encoder_time = 0.0

    # 後段でQ/K/Vに投影される入力特徴に加算する埋め込みの設定(nn.Parameterは学習対象)
    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""

        # 参照点の初期化用のベクトル
        self.reference_points = nn.Linear(self.embed_dims, 3)   # 物体検出用の初期参照点。検出クエリに対して、初期座標のガイドを作る(D次元→3次元(x,y,z))
                                                                # あとでこの座標をsigomidで0~1に正規化し、BEV領域内に収める(アンカーfree版のアンカー的な役割を持つ)
                                                                # 各層でボックスを参照点に対するオフセット(残差）として予測(最初の参照点を中心に微調整する学習になり、学習が安定化する！)
        
    # 重みの初期化
    def init_weights(self):
        """Initialize the transformer weights."""
        
        # 次元>1(行列)の重みはxavierで初期化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Attentionの専用初期化(3D可変サンプリング注意 (MSDeformableAttention3D)、時系列自己注意などはサンプリングオフセットや参照点バイアスに専用の初期化が必要)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) or isinstance(m, CustomMSDeformableAttention):
                try: # メソッド名の違いに対応
                    m.init_weight() # attention系は固有の初期化を実施
                except AttributeError:
                    m.init_weights()
                    
        # 埋め込み・線形層の追加初期化(Xavier Uniform、bias=0を適用)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                #mlvl_feats,         # 画像バックボーンから得られたマルチスケール特徴(list of [B, N_cam, C, H, W])
                bev_embed,          # Fast-BEVで生成したBEV埋め込み
                object_query_embed, # 物体用クエリ埋め込み（位置＋内容）[N_obj_query, 2*C]
                bev_h,              # BEVの奥行き
                bev_w,              # BEVの幅
                reg_branches=None,  # 物体デコーダ用の各層ごとの回帰ヘッド
                cls_branches=None,  # 物体デコーダ用の各層ごとの分類ヘッド
                #img_metas=None,     # 画像や幾何行列など“各フレームのメタ情報”（[T] 構造のリスト）
                #prev_bev=None,      # 前フレームのBEV特徴（時系列融合用）[B, H_bev*W_bev, C] or None
                #test_mode=False,    # 推論モード
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        # バッチサイズの取得
        bs = bev_embed.size(0)

        # BEV埋め込みの形状変更
        B, C, H, W = bev_embed.shape
        bev_embed = bev_embed.view(B, C, H*W)

        # # debug(時間計測用)
        # if self.debug:
        #     self.cnt += 1
        #     t0 = time.time()

        # # BEV埋め込みの生成
        # bev_embed = self.bev_encoder(mlvl_feats, img_metas, prev_bev, test_mode)

        # # debug(時間計測用)
        # if self.debug:
        #    t1 = time.time()
        #    self.bev_encoder_time += (t1 - t0)*1000
        #    if self.cnt == 100:
        #         print(f"bev_encoder time: {(self.bev_encoder_time)/self.cnt:.2f} ms")

        # オブジェクトQueryの初期化
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1) # 位置埋め込み(各クエリの位置識別子(クエリの識別ID)←peとは別！)と内容埋め込み(各クエリの初期内容ベクトル)に分解
        # バッチ方向に拡張
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        # 初期参照点の生成
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # 次元の並び替え(PyTorchのTransformerは、[L, B, D] 形式を想定)
        query = query.permute(1, 0, 2)          # [num_query, B, C]
        query_pos = query_pos.permute(1, 0, 2)  # [num_query, B, C]
        # bev_embed = bev_embed.permute(1, 0, 2)  # [H*W, B, C]
        bev_embed = bev_embed.permute(2, 0, 1)  # [H*W, B, C]

        # 物体検出(各object queryがBEV埋め込みを参照しながらオブジェクト（車、人など）を推論)
        if self.decoder is not None:
            # ([L, Q, B, D], [L, B, Q, D])を返す←L: デコーダ層数, Q: オブジェクトクエリ数, B: バッチサイズ, D: 埋め込み次元
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                **kwargs)
            inter_references_out = inter_references
        else:
            inter_states = query.unsqueeze(0)
            inter_references_out = reference_points.unsqueeze(0)

        # 出力
        return (
            bev_embed,                # BEV特徴[H*W, B, C]
            inter_states,             # 各層の物体クエリ出力[L, B, N_obj, C]  (Deep Supervision用に、各層の結果を出力)
            init_reference_out,       # 初期参照点（x,y,z,wなど）[B, N_obj, 4]
            inter_references_out,     # 各層で更新された参照点 [L, B, N_obj, 4](Deep Supervision用に、各層の結果を出力)
            ) 
