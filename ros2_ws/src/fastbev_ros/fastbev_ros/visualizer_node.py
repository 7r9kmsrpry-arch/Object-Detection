import json
import os

import torch

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

from .visualizer import Visualizer
from .utils import load_config, load_json


class FastBEVVisualizerNode(Node):
    """
    FastBEVの検出結果を可視化するノード。

    Subscribe:
        /fastbev/detections : std_msgs/String

    Publish:
        /fastbev/visualized_image : sensor_msgs/Image
    """

    def __init__(self):
        super().__init__("fastbev_visualizer_node")

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

        # パラメータ定義
        self.declare_parameter("config_path", default_config_path)
        self.declare_parameter("json_path", default_json_path)
        self.declare_parameter("image_encoding", "rgb8")

        # パラメータ取得
        config_path = self.get_parameter("config_path").value
        json_path = self.get_parameter("json_path").value
        self.image_encoding = self.get_parameter("image_encoding").value

        self.get_logger().info(f"config_path: {config_path}")
        self.get_logger().info(f"json_path: {json_path}")
        self.get_logger().info(f"image_encoding: {self.image_encoding}")

        # config/json読み込み
        self.data_config = load_config(config_path)

        json_file = load_json(json_path)
        self.data_infos = json_file["infos"]

        # Visualizer作成
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

        self.bridge = CvBridge()

        # Subscriber
        self.subscriber = self.create_subscription(
            String,
            "/fastbev/detections",
            self.detection_callback,
            10
        )

        # Publisher
        self.image_publisher = self.create_publisher(
            Image,
            "/fastbev/visualized_image",
            10
        )

        self.get_logger().info("FastBEV Visualizer Node started.")

    def detection_callback(self, msg):
        """
        検出結果を受信して、bboxを画像に描画する。
        """

        try:
            result = json.loads(msg.data)

            sample_index = result["sample_index"]

            self.get_logger().info(
                f"Receive detections: sample_index={sample_index}"
            )

            if sample_index < 0 or sample_index >= len(self.data_infos):
                self.get_logger().warn(
                    f"Invalid sample_index: {sample_index}"
                )
                return

            # JSONからTensorへ戻す
            bboxes_tensor = torch.tensor(result["bboxes"], dtype=torch.float32)
            scores = torch.tensor(result["scores"], dtype=torch.float32)
            labels = torch.tensor(result["labels"], dtype=torch.long)

            # # bboxが0個の場合
            # if bboxes.numel() == 0:
            #     self.get_logger().warn(
            #         f"No bboxes: sample_index={sample_index}"
            #     )

            # TensorをLiDARInstance3DBoxesに戻す
            bboxes = LiDARInstance3DBoxes(
                bboxes_tensor,
                box_dim=bboxes_tensor.shape[-1],
                origin=(0.5, 0.5, 0.5)
            )

            # nuScenes形式へ変換
            nusc_results, nusc_annos = self.visualizer.format_bbox(
                bboxes,
                scores,
                labels,
                self.data_infos[sample_index]
            )

            # bbox描画
            drawed_img = self.visualizer.draw_bbox(
                nusc_results,
                self.data_infos[sample_index],
                sample_index
            )

            # OpenCV画像をROS Imageメッセージへ変換
            image_msg = self.bridge.cv2_to_imgmsg(
                drawed_img,
                encoding=self.image_encoding
            )

            image_msg.header.frame_id = "fastbev_visualizer"

            self.image_publisher.publish(image_msg)

            self.get_logger().info(
                f"Publish visualized image: sample_index={sample_index}"
            )

        except Exception as e:
            self.get_logger().error(f"Visualization failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FastBEVVisualizerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()