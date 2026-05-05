import json
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .nuscenes_sample_generator import NuScenesSampleGenerator
from .utils import load_json


def to_jsonable(obj):
    """
    input_data内のNumPy配列などをJSON保存可能な形式に変換する。

    Args:
        obj: 変換対象のオブジェクト

    Returns:
        JSON化可能なオブジェクト
    """

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


class FastBEVDataNode(Node):
    """
    NuScenesSampleGeneratorからinput_dataを作成してpublishするノード。

    出力:
        /fastbev/input_data : std_msgs/String

    内容:
        input_dataをJSON文字列に変換して送信する。
    """

    def __init__(self):
        super().__init__("fastbev_data_node")

        # パラメータ
        self.declare_parameter(
            "json_path",
            "/home/kenta/ros2_ws/src/fastbev_ros/data/nuscenes/bevdetv3-nuscenes_infos_val.json"
        )
        self.declare_parameter("start_index", 0)
        self.declare_parameter("end_index", 5)
        self.declare_parameter("period_sec", 1.0)

        json_path = self.get_parameter("json_path").value
        self.index = self.get_parameter("start_index").value
        self.end_index = self.get_parameter("end_index").value
        period_sec = self.get_parameter("period_sec").value

        # json読み込み
        json_file = load_json(json_path)
        self.data_infos = json_file["infos"]

        # sample generator
        self.sample_generator = NuScenesSampleGenerator(
            data_infos=self.data_infos,
            num_adj_frame=1
        )

        # Publisher
        self.publisher = self.create_publisher(
            String,
            "/fastbev/input_data",
            10
        )

        # Timer
        self.timer = self.create_timer(
            period_sec,
            self.timer_callback
        )

        self.get_logger().info("FastBEV Data Node started.")

    def timer_callback(self):
        """
        一定周期でinput_dataをpublishする。
        """

        if self.index >= self.end_index:
            self.get_logger().info("すべてのinput_dataをpublishしました。")
            return

        # sample_generatorからinput_dataを作成
        input_data = self.sample_generator.get_data_info(self.index)

        # 後段でindexも使えるように追加
        input_data["sample_index"] = self.index

        # JSON化可能な形式に変換
        input_data_jsonable = to_jsonable(input_data)

        # JSON文字列に変換
        msg = String()
        msg.data = json.dumps(input_data_jsonable)

        # publish
        self.publisher.publish(msg)

        self.get_logger().info(f"Publish input_data: sample_index={self.index}")

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = FastBEVDataNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()