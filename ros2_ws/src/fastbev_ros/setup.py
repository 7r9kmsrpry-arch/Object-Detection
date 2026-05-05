from glob import glob
from setuptools import setup, find_packages

package_name = "fastbev_ros"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name,
            ["package.xml"]),

        ("share/" + package_name + "/config",
            glob("config/*.yaml")),

        ("share/" + package_name + "/models/onnx",
            glob("models/onnx/*.onnx")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="kenta",
    maintainer_email="m.kenta1105@ezweb.ne.jp",
    description="FastBEV inference package for ROS2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "data_node = fastbev_ros.data_node:main",
            "inference_node = fastbev_ros.inference_node:main",
            "visualizer_node = fastbev_ros.visualizer_node:main",
        ],
    },
)