import os
import json

import numpy as np
import torch
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

from .nuscenes_sample_generator import NuScenesSampleGenerator
from .img_pipeline import PrepareImageInputs
from .onnx_input_builder import OnnxInputBuilder
from .bbox_decoder import BboxDecoder
from .visualizer import Visualizer
from .utils import load_config, load_json


class FastBEVNode(Node):
    """
    FastBEVの一連の処理を1つのROS2ノードで実行する。

    Publish:
        /fastbev/visualized_image : sensor_msgs/Image
    """

    def __init__(self):
        super().__init__("fastbev")

        # パスの設定
        package_share_dir = get_package_share_directory("fastbev_ros")

        default_config_path = os.path.join(
            package_share_dir,
            "config",
            "config.yaml"
        )

        default_json_path = os.path.join(
            package_share_dir,
            "data",
            "nuscenes",
            "bevdetv3-nuscenes_infos_val.json"
        )

        default_fastbev_onnx_path = os.path.join(
            package_share_dir,
            "models",
            "onnx",
            "fastbev.onnx"
        )

        default_fastbev4d_onnx_path = os.path.join(
            package_share_dir,
            "models",
            "onnx",
            "fastbev_4d.onnx"
        )

        # パラメータ定義
        self.declare_parameter("config_path", default_config_path)
        self.declare_parameter("json_path", default_json_path)
        self.declare_parameter("fastbev_onnx_path", default_fastbev_onnx_path)
        self.declare_parameter("fastbev4d_onnx_path", default_fastbev4d_onnx_path)

        self.declare_parameter("start_index", 0)
        self.declare_parameter("end_index", 5)
        self.declare_parameter("period_sec", 0.1)
        self.declare_parameter("num_adj_frame", 1)

        self.declare_parameter("use_cuda", True)
        self.declare_parameter("image_encoding", "rgb8")

        # パラメータ取得
        self.config_path = self.get_parameter("config_path").value
        self.json_path = self.get_parameter("json_path").value
        self.fastbev_onnx_path = self.get_parameter("fastbev_onnx_path").value
        self.fastbev4d_onnx_path = self.get_parameter("fastbev4d_onnx_path").value

        self.index = self.get_parameter("start_index").value
        self.end_index = self.get_parameter("end_index").value
        period_sec = self.get_parameter("period_sec").value
        num_adj_frame = self.get_parameter("num_adj_frame").value

        self.use_cuda = self.get_parameter("use_cuda").value
        self.image_encoding = self.get_parameter("image_encoding").value

        self.get_logger().info("FastBEV node initializing...")
        self.get_logger().info(f"config_path: {self.config_path}")
        self.get_logger().info(f"json_path: {self.json_path}")
        self.get_logger().info(f"fastbev_onnx_path: {self.fastbev_onnx_path}")
        self.get_logger().info(f"fastbev4d_onnx_path: {self.fastbev4d_onnx_path}")
        self.get_logger().info(f"use_cuda: {self.use_cuda}")

        # config/json読み込み
        self.data_config = load_config(self.config_path)
        json_file = load_json(self.json_path)
        self.data_infos = json_file["infos"]

        # sample generator
        self.sample_generator = NuScenesSampleGenerator(
            data_infos=self.data_infos,
            num_adj_frame=num_adj_frame
        )

        # image pipeline
        self.image_pipeline = PrepareImageInputs(
            self.data_config["data_config"],
            sequential=True,
            opencv_pp=False
        )

        # ONNX入力作成
        grid_config = self.data_config["geometry"]["grid_config"]
        image_size = self.data_config["data_config"]["input_size"]

        self.onnx_input_builder = OnnxInputBuilder(
            grid_config,
            image_size,
            stride=16,
            accelerate=True
        )

        # bbox decoder
        post_center_range = self.data_config["bbox_decoder"]["post_center_range"]
        max_num = self.data_config["bbox_decoder"]["max_num"]
        num_classes = self.data_config["bbox_decoder"]["num_classes"]

        self.bbox_decoder = BboxDecoder(
            post_center_range,
            max_num,
            num_classes,
            None
        )

        # visualizer
        save_path = self.data_config["visualizer"]["save_path"]
        save_format = self.data_config["visualizer"]["save_format"]
        save_prefix = self.data_config["visualizer"]["save_prefix"]
        fps = self.data_config["visualizer"]["fps"]

        self.visualizer = Visualizer(
            0.5,
            save_path,
            save_format,
            save_prefix,
            fps,
            scale_factor=3,
            color_map=(0, 255, 255)
        )

        # ONNX Runtime Provider
        if self.use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.get_logger().info(f"ONNX providers: {providers}")

        # ONNXモデル読み込み
        self.fastbev = ort.InferenceSession(
            self.fastbev_onnx_path,
            providers=providers
        )

        self.fastbev4d = ort.InferenceSession(
            self.fastbev4d_onnx_path,
            providers=providers
        )

        self.get_logger().info(f"fastbev providers: {self.fastbev.get_providers()}")
        self.get_logger().info(f"fastbev4d providers: {self.fastbev4d.get_providers()}")

        # ROS Image publisher
        self.bridge = CvBridge()

        self.image_publisher = self.create_publisher(
            Image,
            "/fastbev/visualized_image",
            10
        )

        # Timer
        self.timer = self.create_timer(
            period_sec,
            self.timer_callback
        )

        self.get_logger().info("FastBEV node started.")
        self.get_logger().info(f"num samples: {len(self.data_infos)}")
        self.get_logger().info(f"start_index: {self.index}")
        self.get_logger().info(f"end_index: {self.end_index}")

    def timer_callback(self):
        """
        一定周期で1サンプル分のFastBEV処理を実行する。
        """

        if self.index >= self.end_index:
            self.get_logger().info("All samples have been processed.")
            return

        sample_index = self.index

        try:
            self.get_logger().info(f"Start FastBEV processing: sample_index={sample_index}")

            drawed_img = self.run_fastbev(sample_index)

            # 画像メッセージの作成
            image_msg = self.bridge.cv2_to_imgmsg(
                drawed_img,
                encoding=self.image_encoding
            )

            image_msg.header.frame_id = "fastbev"
            image_msg.header.stamp = self.get_clock().now().to_msg()

            # 画像トピックの送信
            self.image_publisher.publish(image_msg)

            self.get_logger().info(f"Publish visualized image: sample_index={sample_index}")

            self.index += 1

        except Exception as e:
            self.get_logger().error(f"FastBEV processing failed: sample_index={sample_index}, error={e}")

    def run_fastbev(self, sample_index):
        """
        1サンプル分のFastBEV処理を実行する。

        Args:
            sample_index (int): data_infosのindex

        Returns:
            np.ndarray: bbox描画済み画像
        """

        with torch.no_grad():
            # 入力データ作成
            input_data = self.sample_generator.get_data_info(sample_index)

            # 前処理
            input_data = self.image_pipeline(input_data)
            img_curr, sensor2keyegos_curr, _, _, _, _, bda_curr = input_data["img_inputs_curr"]
            img_prev, sensor2keyegos_prev, _, _, _, _, _        = input_data["img_inputs_prev"]

            # 現在フレームのFastRay入力作成
            _, coors_img_curr, coors_depth_curr = self.onnx_input_builder.get_fastray_input(input_data["img_inputs_curr"])
            coors_img_curr   = coors_img_curr[0]
            coors_depth_curr = coors_depth_curr[0]

            # 過去フレームのFastRay入力作成
            # align_after_view_transformation対応
            input_data["img_inputs_prev"][1] = input_data["img_inputs_curr"][1]
            input_data["img_inputs_prev"][2] = input_data["img_inputs_curr"][2]
            _, coors_img_prev, coors_depth_prev = self.onnx_input_builder.get_fastray_input(input_data["img_inputs_prev"])
            coors_img_prev   = coors_img_prev[0]
            coors_depth_prev = coors_depth_prev[0]

            # 現在フレームのBEV特徴作成
            bev_feat_curr_onnx = self.fastbev.run(
                ["bev_feat"],
                {
                    "img": img_curr.squeeze(0).cpu().numpy().astype(np.float32),
                    "coors_img": coors_img_curr.cpu().numpy().astype(np.int64),
                    "coors_depth": coors_depth_curr.cpu().numpy().astype(np.int64),
                }
            )[0]

            # 過去フレームのBEV特徴作成
            bev_feat_prev_onnx = self.fastbev.run(
                ["bev_feat"],
                {
                    "img": img_prev.squeeze(0).cpu().numpy().astype(np.float32),
                    "coors_img": coors_img_prev.cpu().numpy().astype(np.int64),
                    "coors_depth": coors_depth_prev.cpu().numpy().astype(np.int64),
                }
            )[0]

            # 後段の処理のために、一旦torchに戻す
            bev_feat_curr = torch.from_numpy(bev_feat_curr_onnx).float()
            bev_feat_prev = torch.from_numpy(bev_feat_prev_onnx).float()

            # 過去BEV特徴の位置合わせ
            bev_feat_list = [bev_feat_curr, bev_feat_prev]
            bev_feat_list[1] = self.onnx_input_builder.shift_feature(bev_feat_list[1],[sensor2keyegos_curr, sensor2keyegos_prev], bda_curr)

            # BEV特徴を結合
            bev_feats = torch.cat(bev_feat_list, dim=1)

            # 物体認識
            detections = self.fastbev4d.run(
                None,
                {
                    "bev_feats": bev_feats.cpu().numpy().astype(np.float32)
                }
            )

            # 後処理
            detections = [torch.from_numpy(detection).float() for detection in detections]
            bboxes, scores, labels = self.bbox_decoder.get_bbox(detections)

            # nuScenes形式へ変換
            nusc_results, nusc_annos = self.visualizer.format_bbox(
                bboxes,
                scores,
                labels,
                self.data_infos[sample_index]
            )

            # 可視化
            drawed_img = self.visualizer.draw_bbox(
                nusc_results,
                self.data_infos[sample_index],
                sample_index
            )

            return drawed_img


def main(args=None):
    # 初期化
    rclpy.init(args=args)
    node = FastBEVNode()

    # ノードの起動
    rclpy.spin(node)

    # 終了
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()