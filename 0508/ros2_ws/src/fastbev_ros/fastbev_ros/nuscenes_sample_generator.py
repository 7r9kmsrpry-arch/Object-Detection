class NuScenesSampleGenerator():
    """現在フレームと過去フレームの情報を生成する
    
    Args:
        data_infos (list[dict]): データセットの情報(フレーム番号ごとのdict)
        num_adj_frame (int): 使用する過去フレームの枚数
    """

    def __init__(self, data_infos, num_adj_frame=1):
        # メンバ変数の設定
        self.data_infos = data_infos
        self.num_adj_frame = num_adj_frame

    def get_data_info(self, index):
        """指定した index に対応するデータ情報を取得する。

        Args:
            index (int): 取得したいサンプルデータのインデックス。

        Returns:
            dict: 前処理パイプラインに渡されるデータ情報。主に以下のキーを含む。
                - sample_idx (str): サンプルID
                - timestamp (float): サンプルのタイムスタンプ
                - curr (dict): 現在フレームの情報
                - adjacement (dict): 隣接フレームの情報
        """

        # １サンプルのinfoの抽出
        info = self.data_infos[index]
        
        # パイプラインに渡す基本情報を生成
        input_dict = dict(
            sample_idx=info['token'],          # サンプル識別子
            timestamp=info['timestamp'] / 1e6, # タイムスタンプ(マイクロ秒→秒に変換)
        )

        # 現在フレームの生情報をcurrに格納
        input_dict.update(dict(curr=info))
        # 隣接フレームの情報を追加
        info_adj_list = self.get_adj_info(info, index)
        input_dict.update(dict(adjacent=info_adj_list))

        return input_dict

    def get_adj_info(self, info, index):
        """指定したサンプルに対する隣接フレーム情報を取得する。

        現在サンプル `info` とそのインデックス `index` をもとに、`multi_adj_frame_id_cfg` で指定されたオフセットの過去フレーム情報を`self.data_infos` から収集して返す。
        同一scene内に存在しないフレームを参照しようとした場合は、代わりに現在フレーム`info`自身を追加する。

        Args:
            info (dict): 現在サンプルの情報辞書。
            index (int): 現在サンプルのインデックス。

        Returns:
            list[dict]: 隣接フレーム情報のリスト。
        """

        # 隣接フレーム情報を格納するリストを初期化
        info_adj_list = []

        # 隣接フレーム番号の取得([1])
        adj_id_list = list(range(1, self.num_adj_frame+1)) 

        # 隣接シーンの追加
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0) # 最初のフレームの「-」防止
            # シーンをまたぐ場合、現在のinfoを追加
            if not self.data_infos[select_id]['scene_token'] == info['scene_token']: 
                info_adj_list.append(info) 
            else: 
                info_adj_list.append(self.data_infos[select_id]) 
                
        return info_adj_list
