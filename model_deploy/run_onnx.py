import numpy as np
import torch
import onnxruntime as ort

from nuscenes_sample_generator import NuScenesSampleGenerator
from img_pipeline import PrepareImageInputs
from onnx_input_builder import OnnxInputBuilder
from bbox_decoder import BboxDecoder
from util import load_config, load_json

# パスの設定
json_path   = "bevdetv3-nuscenes_infos_val.json"
config_path = "config.yaml"

# jsonファイルとconfigファイルの読み込み
json_file   = load_json(json_path)
data_config = load_config(config_path)

# sample Generator
data_infos = json_file['infos']
sample_generator = NuScenesSampleGenerator(data_infos=data_infos, num_adj_frame=1)

# image pipeline
image_pipline = PrepareImageInputs(data_config["data_config"], sequential=True, opencv_pp=False)

# 入力データの作成
grid_config = data_config["geometry"]["grid_config"]
image_size  = data_config["data_config"]["input_size"]
onnx_input_builder = OnnxInputBuilder(grid_config, image_size, stride=16, accelerate=True)

# # モデルの読み込み
fastbev = ort.InferenceSession(
    "./onnx/fastbev.onnx",
    #providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers=["CPUExecutionProvider"]
)

# fastbev4d = ort.InferenceSession(
#     "./onnx/fastbev_4d.onnx",
#     providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
# )

# print("===fastbev===")
# print("=== Inputs ===")
# for i, x in enumerate(fastbev.get_inputs()):
#     print(f"[{i}]")
#     print(" name :", x.name)
#     print(" shape:", x.shape)
#     print(" type :", x.type)

# print("\n=== Outputs ===")
# for i, x in enumerate(fastbev.get_outputs()):
#     print(f"[{i}]")
#     print(" name :", x.name)
#     print(" shape:", x.shape)
#     print(" type :", x.type)

# print("\n===fastbev4d===")
# print("=== Inputs ===")
# for i, x in enumerate(fastbev4d.get_inputs()):
#     print(f"[{i}]")
#     print(" name :", x.name)
#     print(" shape:", x.shape)
#     print(" type :", x.type)

# print("\n=== Outputs ===")
# for i, x in enumerate(fastbev4d.get_outputs()):
#     print(f"[{i}]")
#     print(" name :", x.name)
#     print(" shape:", x.shape)
#     print(" type :", x.type)