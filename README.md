# SInDSLAM
Semantic-Independent Dynamic SLAM based on Geometric Re-clustering and Optical Flow Residuals, IEEE TCSVT, 2024.

[PDF](https://ieeexplore.ieee.org/document/10750834), [IEEE](https://ieeexplore.ieee.org/abstract/document/10750834), [BiliBili](https://www.bilibili.com/video/BV1V6mYYrEyr/?spm_id_from=333.1387.upload.video_card.click)

[加一个主图]

Please kindly star :star: this project if it helps you. We take great efforts to develope and maintain it :grin::grin:.

SInDSLAM is based on the the excellent work of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). We achieve SORT localization accuracy on the [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and [Boon](http://www.ipb.uni-bonn.de/data/rgbd-dynamicdataset) datasets, without utilizing semantic segmentation or object detection.

The main modified files are::

- "*SInDSLAM/ORB_SLAM2/Examples/RGB-D/rgbd_tum_xx.cc*": Interface function
- "*SInDSLAM/ORB_SLAM2/src/DynaDetect.cc*": Dynamic Region Detection
- "*SInDSLAM/ORB_SLAM2/src/ORBextractor.cc*": Dynamic Features Points Erasion
- "*SInDSLAM/octomap_pub/src/pubPointCloud.cc*": Constrction of Dense PointCloud Map and OctoMap

## 1. Build and Start
**Dependencies:** Ubuntn20.04, ROS1, OpenCV (with CUDA, Optional), xterm, OpenMP and Pangolin-v0.5 (Included in this Git).

Our code has two modes of operation. One mode does not rely on ROS, and you can obtain the camera pose. The other mode relies on ROS, allowing you to view the dense reconstructed point cloud and octree map in RViz.

**(1)** The ROS version depends on the *cv_bridge* package, which depends on OpenCV 4.2.0. If your OpenCV version is not 4.2.0, you cannot use the ROS version.

**(2)** If your OpenCV supports CUDA, the code uses the CUDA-accelerated BroxFlow algorithm; otherwise, the code uses the DeepFlow algorithm. The average frame rate with CUDA is approximately 9 Hz, while without it, the frame rate is around 5 Hz.

**Therefore, please correctly configure the CUDA and OpenCV versions** in "*SInDSLAM/ORB_SLAM2/CMakeLists.txt Line14-15*" and ensure that the **OpenCV version** specified in DBoW2 library matches the version in the main CMakeLists.txt. In VSCode, you can search for "find_package(OpenCV" to ensure that the OpenCV version is correctly configured.

## Step1:
```
  cd ${YOUR_WORKSPACE_PATH}/src
  git clone https://github.com/qimao7213/SInDSLAM
  cd SInDSLAM/ORB_SLAM2
  ./build_new.sh # Compile some third-party libraries.
  cd ../../..
  catkin_make
  source /devel/setup.bash
```
If everything goes smoothly, you will correctly install SInDSLAM.
## Step2:

If you are using the non-ROS mode, navigate to "*SInDSLAM/ORB_SLAM2/Examples/RGB-D/*" and run:

```
./rgbd_tum_noros VOC_FILE CAMERA_FILE DATASET_FILE ASSOCIATIONS_FILE
```
Change *VOC_FILE*, *CAMERA_FILE*, *DATASET_FILE*, *ASSOCIATIONS_FILE* to your files. Example Code:
```
./rgbd_tum_noros Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
```
---------------------------------------------
If you are using the ROS mode, modify the file paths in "*SInDSLAM/ORB_SLAM2/launch/sindslam_ros.launch*", and run:

```
roslaunch SInDSLAM sindslam_ros.launch
```
You will see the reconstructed 3D map in RViz.

The *octomap_pub* node subscribes to the camera pose, color image, and depth image from *ORB_SLAM2* node, and then generates a dense point cloud map and an octree map. This process can be very time-consuming. If the computational load is too high, you can stop publishing the point cloud map and octree map to RVIZ. 

The *octomap_pub* node runs in a small **xterm** window. If you use *Ctrl+C* to close it, the generated point cloud map and octree map will be saved to the path specified in "*sindslam_ros.launch pt_output_file*".

## Step3:

If you want to save the images generated during the process, you can set your recording path at "*SInDSLAM/ORB_SLAM2/src/DynaDetect.cc Line 38*".

If you want to display the GT trajectory in RVIZ, you can set the alignment matrix at "SInDSLAM/ORB_SLAM2/Examples/RGB-D/rgbd_tum_withros.cc Line 63", which is calculated by EVO.

## 2. Alignment of RGB-Depth Images in TUM RGBD dataset
The misalignment issue between RGB and depth images in the TUM RGBD dataset has been reported in many papers. 

I think this issue arises from the use the following code when generating the *associations.txt* file.
```
python associate.py rgb.txt depth.txt > associations.txt
```
But in fact, there is a 1-frame offset between the RGB and the Depth images. Therefore, using the following code can correctly align the images:：
```
python associate.py rgb.txt depth.txt > associations.txt --offset -0.033
```
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




