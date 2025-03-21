# SInDSLAM
Semantic-Independent Dynamic SLAM based on Geometric Re-clustering and Optical Flow Residuals, IEEE TCSVT, 2024.

[PDF](file/PDF.pdf), [IEEE](https://ieeexplore.ieee.org/abstract/document/10750834), [BiliBili](https://www.bilibili.com/video/BV1V6mYYrEyr/?spm_id_from=333.1387.upload.video_card.click)

<p align="center">
    <img src="file/overall.png" alt="Overall Results" width="80%">
</p>

<p align="center">
    <img src="file/重建.png" alt="Environment Mapping" width="80%">
</p>
Please kindly star :star: this project if it helps you. We take great efforts to develope and maintain it :grin::grin:.

SInDSLAM is based on the the excellent work of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). We achieve SORT localization accuracy on the [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) and [Boon](http://www.ipb.uni-bonn.de/data/rgbd-dynamicdataset) datasets, without utilizing semantic segmentation or object detection.

The main modified files are::

- "*SInDSLAM/ORB_SLAM2/Examples/RGB-D/rgbd_tum_xx.cc*": Interface function
- "*SInDSLAM/ORB_SLAM2/src/DynaDetect.cc*": Dynamic Region Detection
- "*SInDSLAM/ORB_SLAM2/src/ORBextractor.cc*": Dynamic Features Points Erasion
- "*SInDSLAM/octomap_pub/src/pubPointCloud.cc*": Constrction of Dense PointCloud Map and OctoMap

## 1. Build and Start
**Dependencies:** Ubuntn20.04, ROS1, OpenCV (with CUDA, Optional), xterm, OpenMP and [Pangolin-v0.5](file/Pangolin_v0.5.zip) (Included in this Git).

Our code has two modes of operation. One mode does not rely on ROS, and you can obtain the camera pose. The other mode relies on ROS, allowing you to view the dense reconstructed point cloud and octree map in RViz.

**(1)** The ROS version depends on the *cv_bridge* package, which depends on OpenCV 4.2.0. If your OpenCV version is not 4.2.0, you cannot use the ROS version.

**(2)** If your OpenCV supports CUDA, the code uses the CUDA-accelerated BroxFlow algorithm; otherwise, the code uses the DeepFlow algorithm. The average frame rate with CUDA is approximately 9 Hz, while without it, the frame rate is around 5 Hz.

**Therefore, please correctly configure the CUDA and OpenCV configs** in "*SInDSLAM/ORB_SLAM2/CMakeLists.txt Line14-15*" and ensure that the **OpenCV version** specified in DBoW2 library matches the version in the main CMakeLists.txt. In VSCode, you can search for "find_package(OpenCV" to ensure that the OpenCV version is correctly configured.

If you want to use **CUDA + ROS**, you can compile OpenCV 4.2.0 with CUDA acceleration; or you can compile `cv_bridge` from source, name it `cv_bridge_1`, and make it depend on the CUDA-enabled OpenCV version. In the `CMakeLists.txt` file, use `find_package(cv_bridge_1)`.


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
<p align="center">
    <img src="file/ImageAlignment.png" alt="ImageAlignment" width="80%">
</p>

## 3. EVO
Please use the Python script provided by the TUM dataset (located in the [EVO](/ORB_SLAM2/EVO/)) for performance evaluation, instead of the EVO library on GitHub.

## 4. Our Datasets
We recorded our own dataset using the D455i camera, with a format same to that of the TUM RGBD dataset. 

The dynamic object within it is a humanoid robot, which is typically challenging for semantic segmentation algorithms to recognize.
[Scene_1](https://drive.google.com/file/d/1WN0vHl33vBscGQAoQZ1gDdMmwCbvlylL/view?usp=sharing), [Scene_2](https://drive.google.com/file/d/1Z7jZLiR5aM18TnesJSy7eRUwxQDexsOV/view?usp=sharing)
## 5
If you encounter any issues during the usage, please create an issue or contact me via email.

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




