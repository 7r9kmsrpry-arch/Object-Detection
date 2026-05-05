import json
import os

import numpy as np
import torch
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory

from .img_pipeline import PrepareImageInputs
from .onnx_input_builder import OnnxInputBuilder
from .bbox_decoder import BboxDecoder
from .utils import load_config


def to_jsonable(obj):
    """
    PythonオブジェクトをJSON化可能な形式に変換する。
    """

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, dict):
        return {
            key: to_jsonable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            to_jsonable(value)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return [
            to_jsonable(value)
            for value in obj
        ]

    return obj


class FastBEVInferenceNode(Node):
    """
    FastBEVの前処理、ONNX推論、後処理を行うノード。

    Subscribe:
        /fastbev/input_data : std_msgs/String

    Publish:
        /fastbev/detections : std_msgs/String
    """

    def __init__(self):
        super().__init__("fastbev_inference_node")

        package_share_dir = get_package_share_directory("fastbev_ros")

        default_config_path = os.path.join(
            package_share_dir,
            "config",
            "config.yaml"
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
        self.declare_parameter("fastbev_onnx_path", default_fastbev_onnx_path)
        self.declare_parameter("fastbev4d_onnx_path", default_fastbev4d_onnx_path)
        self.declare_parameter("use_cuda", False)

        config_path = self.get_parameter("config_path").value
        fastbev_onnx_path = self.get_parameter("fastbev_onnx_path").value
        fastbev4d_onnx_path = self.get_parameter("fastbev4d_onnx_path").value
        use_cuda = self.get_parameter("use_cuda").value

        self.get_logger().info(f"config_path: {config_path}")
        self.get_logger().info(f"fastbev_onnx_path: {fastbev_onnx_path}")
        self.get_logger().info(f"fastbev4d_onnx_path: {fastbev4d_onnx_path}")

        # config読み込み
        self.data_config = load_config(config_path)

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

        # ONNX Runtime Provider
        if use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.get_logger().info(f"ONNX providers: {providers}")

        # ONNXモデル読み込み
        self.fastbev = ort.InferenceSession(
            fastbev_onnx_path,
            providers=providers
        )

        self.fastbev4d = ort.InferenceSession(
            fastbev4d_onnx_path,
            providers=providers
        )

        self.get_logger().info(
            f"fastbev providers: {self.fastbev.get_providers()}"
        )
        self.get_logger().info(
            f"fastbev4d providers: {self.fastbev4d.get_providers()}"
        )

        # Subscriber
        self.subscriber = self.create_subscription(
            String,
            "/fastbev/input_data",
            self.input_data_callback,
            10
        )

        # Publisher
        self.publisher = self.create_publisher(
            String,
            "/fastbev/detections",
            10
        )

        self.get_logger().info("FastBEV Inference Node started.")

    def input_data_callback(self, msg):
        """
        input_dataを受信したら推論を実行する。
        """

        try:
            input_data = json.loads(msg.data)

            sample_index = input_data.get("sample_index", -1)

            self.get_logger().info(
                f"Receive input_data: sample_index={sample_index}"
            )

            result_msg = self.run_fastbev(input_data)

            self.publisher.publish(result_msg)

            self.get_logger().info(
                f"Publish detections: sample_index={sample_index}"
            )

        except Exception as e:
            self.get_logger().error(f"FastBEV inference failed: {e}")

    def run_fastbev(self, input_data):
        """
        1サンプル分のFastBEV推論を実行する。

        Args:
            input_data (dict): data_nodeから受信したinput_data

        Returns:
            std_msgs/String: bbox, score, labelを含むJSON文字列
        """

        sample_index = input_data.get("sample_index", -1)

        with torch.no_grad():
            # 前処理
            input_data = self.image_pipeline(input_data)

            (
                img_curr,
                sensor2keyegos_curr,
                ego2globals_curr,
                intrins_curr,
                post_rots_curr,
                post_trans_curr,
                bda_curr,
            ) = input_data["img_inputs_curr"]

            (
                img_prev,
                sensor2keyegos_prev,
                ego2globals_prev,
                intrins_prev,
                post_rots_prev,
                post_trans_prev,
                bda_prev,
            ) = input_data["img_inputs_prev"]

            # 現在フレームのFastRay入力
            _, coors_img_curr, coors_depth_curr = \
                self.onnx_input_builder.get_fastray_input(
                    input_data["img_inputs_curr"]
                )

            coors_img_curr = coors_img_curr[0]
            coors_depth_curr = coors_depth_curr[0]

            # 過去フレームのFastRay入力
            # align_after_view_transformation対応
            input_data["img_inputs_prev"][1] = input_data["img_inputs_curr"][1]
            input_data["img_inputs_prev"][2] = input_data["img_inputs_curr"][2]

            _, coors_img_prev, coors_depth_prev = \
                self.onnx_input_builder.get_fastray_input(
                    input_data["img_inputs_prev"]
                )

            coors_img_prev = coors_img_prev[0]
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

            # torchに変換
            bev_feat_curr = torch.from_numpy(bev_feat_curr_onnx).float()
            bev_feat_prev = torch.from_numpy(bev_feat_prev_onnx).float()

            # 過去BEV特徴の位置合わせ
            bev_feat_list = [bev_feat_curr, bev_feat_prev]

            bev_feat_list[1] = self.onnx_input_builder.shift_feature(
                bev_feat_list[1],
                [sensor2keyegos_curr, sensor2keyegos_prev],
                bda_curr
            )

            # BEV特徴を結合
            bev_feats = torch.cat(bev_feat_list, dim=1)

            # FastBEV4D推論
            detections = self.fastbev4d.run(
                None,
                {
                    "bev_feats": bev_feats.cpu().numpy().astype(np.float32)
                }
            )

            # 後処理
            detections = [
                torch.from_numpy(detection).float()
                for detection in detections
            ]

            bboxes, scores, labels = self.bbox_decoder.get_bbox(detections)

            result = {
                "sample_index": sample_index,
                "bboxes": to_jsonable(bboxes.tensor),
                "scores": to_jsonable(scores),
                "labels": to_jsonable(labels),
            }

            result_msg = String()
            result_msg.data = json.dumps(result)

            return result_msg


def main(args=None):
    rclpy.init(args=args)
    node = FastBEVInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()