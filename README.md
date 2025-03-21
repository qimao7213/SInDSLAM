# SInDSLAM
Semantic-Independent Dynamic SLAM based on Geometric Re-clustering and Optical Flow Residuals, IEEE TCSVT, 2024.

[PDF](https://ieeexplore.ieee.org/document/10750834), [IEEE](https://ieeexplore.ieee.org/abstract/document/10750834), [BiliBili](https://www.bilibili.com/video/BV1V6mYYrEyr/?spm_id_from=333.1387.upload.video_card.click)

[加一个主图]

Please kindly star :star: this project if it helps you. We take great efforts to develope and maintain it :grin::grin:.

SInDSLAM is based on the the excellent work of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). We achieve SORT localization accuracy on the TUM and Boon datasets, without utilizing semantic segmentation or object detection.

The main modified files are::

- "*SInDSLAM/ORB_SLAM2/Examples/RGB-D/rgbd_tum_xx.cc*": Interface function
- "*SInDSLAM/ORB_SLAM2/src/DynaDetect.cc*": Dynamic Region Detection
- "*SInDSLAM/ORB_SLAM2/src/ORBextractor.cc*": Dynamic Features Points Erasion
- "*SInDSLAM/octomap_pub/src/pubPointCloud.cc*": Constrction of Dense PointCloud Map and OctoMap

## 1. Build and Start
**Dependencies:** Ubuntn20.04, ROS1, OpenCV (with CUDA, Optional), xterm, OpenMP and Pangolin-v0.5 (Included in this Git).

我们的程序有两种运行方式，一种不依赖ros的，你可以得到相机的位姿；另一种是依赖ros的，可以在rviz里面看到稠密重建的点云和八叉树地图。
ros的cv_bridge依赖opencv4.2.0，因此，如果你的opencv版本不是4.2.0，则不能使用ros版本。如果你的opencv是支持cuda的，则我们使用cuda加速的broxflow算法，否则使用deepflow算法。使用cuda的，平均帧率约为9hz，而不使用的，帧率为5hz。
所以，请在XX里正确配置cuda和opencv的版本，并确保在编译DwBag库时CMakeLists.txt时 opencv版本和根目录CMkaLists.txt一致。在vscode里面，你可以搜索"find_package(OpenCV "来确保正配置OpenCV版本。
- Step1:
```
  cd ${YOUR_WORKSPACE_PATH}/src
  git clone https://github.com/qimao7213/SInDSLAM
  cd SInDSLAM/ORB_SLAM2
  ./build_new.sh # 这将会编译一些第三方库
  cd ../../..
  catkin_make
  source /devel/setup.bash
```
如果顺利，你讲正确安装SInDSLAM
- Step2:
如果你使用非ros模式，进入到ORB_SLAM2/Example/RGB_D目录下，run
```
./rgbd_tum_noros VOC_FILE CAMERA_FILE DATASET_FILE ASSOCIATIONS_FILE
```
Change *VOC_FILE*, *CAMERA_FILE*, *DATASET_FILE*, *ASSOCIATIONS_FILE* to your files.
如果你使用ros模式，则修改*SInDSLAM/ORB_SLAM2/launch/sindslam_ros.launch*下的文件路径，然后run

```
roslaunch SInDSLAM sindslam_ros.launch
```
然后你将在rviz里面看到重建的三维地图。
octomap_pub节点从ORB_SLAM2里面订阅相机位姿、彩色图和深度图，然后生成稠密的点云地图和八叉树地图，这将会非常耗时。如果计算负载过大，你可以取消发布点云地图和八叉树地图到RVIZ。octomap_pub节点在xterm小窗口里面运行，如果你使用Ctrl+C来关闭它，生成的点云地图和八叉树地图将会保存在*sindslam_ros.launch*的*pt_output_file*路径下。

- Step3:
如果你想保存过程中产生的image，你可以在*SInDSLAM/ORB_SLAM2/src/DynaDetect.cc Line 38*设置你的路径。
如果你想在RVIZ里面显示真实的轨迹，你可以在*SInDSLAM/ORB_SLAM2/Examples/RGB-D/rgbd_tum_withros.cc Line 63*处设置对齐的矩阵，which是由EVO计算出来的。

## 2. TUM数据集的对齐问题
在多篇论文里面报道了TUM RGBD数据集rgb图像和depth图像无法对齐的问题，我认为这个问题是因为在生成XX文件时使用了```xx```指令。

但其实，rgb图像和depth图像之间有一个offset，所以如果使用指令：``` xx ```，那么图像可以正确对齐。
【给图片实例。】

## 3. EVO
请使用TUM数据集官方提供的python脚本（在本仓库的ORB_SLAM2/EVO/下），而不是github上的EVO库进行性能评估。

## 4. Our Datasets
我们自己使用D455i相机录制了一个数据集，格式和TUM RGBD一样。里面的动态物体是仿人机器人，通常的语义分割算法很难辨认它。


### Citation

If you find this work useful, please cite our paper:

```bibtex
@ARTICLE{10750834,
  author={Qi, Hengbo and Chen, Xuechao and Yu, Zhangguo and Li, Chao and Shi, Yongliang and Zhao, Qingrui and Huang, Qiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Semantic-Independent Dynamic SLAM Based on Geometric Re-Clustering and Optical Flow Residuals}, 
  year={2025},
  volume={35},
  number={3},
  pages={2244-2259},
  keywords={Dynamics;Simultaneous localization and mapping;Vehicle dynamics;Semantics;Cameras;Accuracy;Optical flow;Image reconstruction;Heuristic algorithms;Circuits and systems;Dynamic SLAM;dynamic region detection;moving object segmentation;dense construction;semantic-independent},
  doi={10.1109/TCSVT.2024.3496489}}




