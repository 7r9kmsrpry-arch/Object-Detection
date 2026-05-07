#!/bin/bash

set -e

# condaを使えるようにする
source /home/kenta/miniconda3/etc/profile.d/conda.sh
conda activate fastbev_ros_py310

# ROS2環境を読み込み
source /opt/ros/humble/setup.bash

# ワークスペースへ移動
cd /home/kenta/ros2_ws

# ビルド
colcon build --packages-select fastbev_ros --symlink-install

# install環境を読み込み
source /home/kenta/ros2_ws/install/setup.bash

# fastbevノードだけshebangをconda Pythonに修正
sed -i '1c #!/home/kenta/miniconda3/envs/fastbev_ros_py310/bin/python' \
  /home/kenta/ros2_ws/install/fastbev_ros/lib/fastbev_ros/fastbev

echo "======================================"
echo "Build finished."
echo "Run command:"
echo "source /home/kenta/ros2_ws/install/setup.bash"
echo "ros2 run fastbev_ros fastbev"
echo "======================================"