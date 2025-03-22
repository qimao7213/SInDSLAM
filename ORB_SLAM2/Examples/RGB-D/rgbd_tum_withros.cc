/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <string>
#include <opencv2/core/core.hpp>
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <tf/tf.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
// #include <cv_bridge/cv_bridge.h>
// #include <cv_bridge>
#include <image_transport/image_transport.h>
#include <Converter.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <System.h>
#include "DynaDetect.h"
#include <unistd.h>
using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    ros::init(argc, argv, "SInDSLAM");
    ros::NodeHandle node("~");
    std::string s1, s2, s3, s4, s5;
    std::string voc_file, camera_file, dataset_file, associ_file;
    node.param("SInDSLAM/voc_file", voc_file, s1);
    node.param("SInDSLAM/camera_file", camera_file, s2);
    node.param("SInDSLAM/dataset_file", dataset_file, s3);
    node.param("SInDSLAM/associ_file", associ_file, s4);
    std::cout << voc_file << "\n" << camera_file << "\n" << dataset_file << "\n" << associ_file << "\n" << std::endl;
    //-----------------Load the ground truth trajectory for visualization in rviz.-------
    string groundtruthFilePath = dataset_file + "/groundtruth.txt";
    nav_msgs::Path gt_path;
    geometry_msgs::PoseStamped gt_pose;
    geometry_msgs::PoseStamped gt_pose_in_camera;
    // The Alignment Matrix, choose your sequence
    // //removing_no_box
    // Eigen::Vector3d twc(0.30859026, -2.10763832, 1.3795153);
    // Eigen::Matrix3d Rwc;
    // Rwc << -0.94703147,  0.02472766, -0.32018736, 
    //         0.3188248,   0.1919202,  -0.92817961,
    //         0.03849871, -0.98109897, -0.18963826;
    // //person_tracking2
    // Eigen::Vector3d twc(1.01925436, -1.49807801,  2.00893291);
    // Eigen::Matrix3d Rwc;
    // Rwc << -0.95919239, -0.24431121,  0.14234462,
    //        -0.25653084,  0.5401908,  -0.80148976,
    //        0.11891968, -0.80529866, -0.58082026;
    // //placing_no_box
    // Eigen::Vector3d twc(0.54948554, -2.11794646,  1.4383995);
    // Eigen::Matrix3d Rwc;
    // Rwc << -0.96328264, -0.21634141,  0.15900611,
    //        -0.23508046,  0.39350183, -0.88875952,
    //        0.1297063,  -0.89350585, -0.42991112;

    // //walking_xyz
    Eigen::Vector3d twc(-0.69978522, -3.11996455,  1.42594347);
    Eigen::Matrix3d Rwc;
    Rwc << 0.99645181,  0.07953277, -0.02753762,
           0.04179624, -0.18360939,  0.98211031,
           0.07305379, -0.97977657, -0.18628208;
    // //walking_half
    // Eigen::Vector3d twc(-0.53551702, -2.88315517,  1.53144349);
    // Eigen::Matrix3d Rwc;
    // Rwc << 0.99479593,  0.00550217, -0.10173878,
    //        0.10164651, -0.12222659,  0.98728347,
    //        -0.00700298, -0.99248697, -0.12214979;
    //walking_static
    // Eigen::Vector3d twc(-0.81354462, -3.22311766,  1.3809757);
    // Eigen::Matrix3d Rwc;
    // Rwc << 0.99947568, -0.00464641,  0.0320435 ,
    //        -0.03200546, -0.29159073, 0.95600758,
    //        0.00490158, -0.95653189, -0.29158655;

    // //walking_rpy 
//     Eigen::Vector3d twc(-0.6240421,  -2.77269904,  1.47526572);
//     Eigen::Matrix3d Rwc;
//     Rwc << 0.99746151,  0.05037503, -0.05032784,
//    0.05664536, -0.13305714,  0.9894883,  
//    0.04314902, -0.98982734, -0.13557289;

    // //record_5
//     Eigen::Vector3d twc(1.19610486, 0.90456142, 1.28459812);
//     Eigen::Matrix3d Rwc;
//     Rwc << -0.98513723,  0.12242596, -0.12048455,
//    0.13212362,  0.09185543, -0.98696805,  
//    -0.10976335, -0.98821782, -0.10666556;

    std::ifstream fin;
    fin.open(groundtruthFilePath, std::ios::in);
    int countkk = 0;
    if (!fin.is_open())
    {
        std::cout << "can not open file" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "open a file at: " << groundtruthFilePath << std::endl;
        while (!fin.eof())
        {
            std::string s;
            getline(fin, s);
            // cout << s.c_str() << endl;
            if (!s.empty())
            {
                std::stringstream ss;
                ss << s;
                std::string sss;
                ss >> sss;
                if (sss == "#")
                {
                    continue;
                }
                ss >> sss;
                gt_pose.pose.position.x = (double)atof(sss.c_str());
                // std::cout << (double)atof(sss.c_str()) << " ";
                ss >> sss;
                gt_pose.pose.position.y = (double)atof(sss.c_str());
                // std::cout << (double)atof(sss.c_str()) << " ";
                ss >> sss;
                gt_pose.pose.position.z = (double)atof(sss.c_str());
                // std::cout << (double)atof(sss.c_str()) << " ";
                ss >> sss;
                gt_pose.pose.orientation.x = (double)atof(sss.c_str());
                // std::cout << (double)atof(sss.c_str()) << " ";
                ss >> sss;
                gt_pose.pose.orientation.y = (double)atof(sss.c_str());
                // std::cout << (double)atof(sss.c_str()) << " ";
                ss >> sss;
                gt_pose.pose.orientation.z = (double)atof(sss.c_str());
                // std::cout << (double)atof(sss.c_str()) << " ";
                ss >> sss;
                gt_pose.pose.orientation.w = (double)atof(sss.c_str());
                // std::cout << (double)atof(sss.c_str()) << std::endl;
                
                Eigen::Vector3d pt_w(gt_pose.pose.position.x, gt_pose.pose.position.y, gt_pose.pose.position.z);
                Eigen::Vector3d pt_c = Rwc.inverse() * pt_w - Rwc.inverse() * twc;
                gt_pose_in_camera.pose.position.x = pt_c.x();
                gt_pose_in_camera.pose.position.y = pt_c.y();
                gt_pose_in_camera.pose.position.z = pt_c.z();
                gt_path.poses.push_back(gt_pose_in_camera);
                countkk++;
            }
        }
    }
    std::cout << "The size of gt trajectory pose is " << gt_path.poses.size() << std::endl;

    ros::Publisher pose_pub;
    ros::Publisher poseKF_pub;
    ros::Publisher rgbd_path_pub;
    ros::Publisher gt_path_pub;
    image_transport::Publisher imgDyna_pub;
    image_transport::Publisher imgRGB_pub;
    image_transport::Publisher imgDepth_pub;
    image_transport::Publisher imgLabel_pub;

    ros::NodeHandle nh;
    image_transport::ImageTransport imgNh(nh);

    nav_msgs::Path rgbd_path;
    pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/orbslam2/poseCamera", 10);
    poseKF_pub = nh.advertise<geometry_msgs::PoseStamped>("/orbslam2/poseKF", 10);
    rgbd_path_pub = nh.advertise<nav_msgs::Path>("/orbslam2/path", 10);
    gt_path_pub = nh.advertise<nav_msgs::Path>("/orbslam2/gt_path", 10);
    imgDyna_pub = imgNh.advertise("/orbslam2/imgDynaMask", 10);
    imgRGB_pub = imgNh.advertise("/orbslam2/imgRGB", 10);
    imgDepth_pub = imgNh.advertise("/orbslam2/imgDepth", 10);
    imgLabel_pub = imgNh.advertise("/orbslam2/imgLabel", 10);


    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    // string strAssociationFilename = string(argv[4]);
    string strAssociationFilename = associ_file;
    cout << strAssociationFilename.c_str() << endl;
    LoadImages(strAssociationFilename.c_str(), vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        cerr << endl
             << "No images found in provided path." << endl;
        return 1;
    }
    else if (vstrImageFilenamesD.size() != vstrImageFilenamesRGB.size())
    {
        cerr << endl
             << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.

    // ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);
    ORB_SLAM2::System SLAM(voc_file,
                           camera_file, ORB_SLAM2::System::RGBD, true);
    cv::FileStorage fSettings(string(camera_file), cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << string(camera_file) << endl;
       exit(-1);
    }
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    float depthScale = fSettings["DepthMapFactor"];

    // Vector for tracking time statistics
    vector<float> vTimesTrack, vTimesDynamic;
    vTimesTrack.resize(nImages);
    vTimesDynamic.resize(nImages);
    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    // Main loop
    cv::Mat imRGB, imD;
    cv::Mat imDynaMask(cv::Size(640, 480), CV_8UC1, cv::Scalar(0));
    cv::Mat imLabel(cv::Size(640, 480), CV_8UC1, cv::Scalar(0));
    bool isDynaDetect = 0;
    cv::Mat imgLast = cv::imread(dataset_file + "/" + vstrImageFilenamesRGB[0], -1);
    cv::Mat imgLastLast;
    imgLast.copyTo(imgLastLast);
    std::shared_ptr<ORB_SLAM2::DynaDetect> detertor = std::make_shared<ORB_SLAM2::DynaDetect>
                                                      (imgLast, imgLastLast, fx, fy, cx, cy, depthScale);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));

    for (int ni = 0; ni < nImages; ni++)
    {
        cout << "----------now processing " << ni << " img-----------------" << endl;
        if (ni >= 1)
        {
            isDynaDetect = true;
        }
        // if(ni == 3)
        // {
        //     cv::waitKey(0);
        // }
        // isDynaDetect = 0;

        // Read image and depthmap from file
        imRGB = cv::imread(dataset_file + "/" + vstrImageFilenamesRGB[ni], -1);
        imD = cv::imread(dataset_file + "/" + vstrImageFilenamesD[ni], -1);

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t0 = std::chrono::monotonic_clock::now();
#endif
        if (isDynaDetect)
        {
            // std::assert(!img.empty() && !imgDepth.empty() && !imgLast.empty() && !imgLastLast.empty() );
            detertor->DetectDynaArea(imRGB, imD, imDynaMask, imLabel, ni);
            if(!imDynaMask.empty())
            {
                cv::morphologyEx(imDynaMask, imDynaMask, cv::MORPH_DILATE, element);
            } 
        }
        // cout << "动态区域检测完成，第" << ni << "张图像" << endl;
        double tframe = vTimestamps[ni];
        if (imRGB.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << dataset_file << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }
        if(imDynaMask.empty() || imLabel.empty())
        {
            cout << endl << "Failed to run dyna-detection " << endl;
            return 1;
        }

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        // SLAM.TrackRGBD(imRGB,imD,tframe);
        // SLAM.TrackRGBD(imRGB,imD,imDynaMask,imLabel,tframe);
        cv::Mat Tcw;
        bool isKeyFrame = false;
        bool isLost = false;
        Tcw = SLAM.TrackRGBD(imRGB, imD, imDynaMask, imLabel, tframe, isKeyFrame);
        if (!Tcw.empty())
        {
            cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);
            vector<float> q = ORB_SLAM2::Converter::toQuaternion(Rwc);
            if (ni % 5 == 0)
            {
                sensor_msgs::ImagePtr imgRGBMsg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imRGB).toImageMsg();
                sensor_msgs::ImagePtr imgDynaMaskMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imDynaMask).toImageMsg();
                sensor_msgs::ImagePtr imgDepthMsg = cv_bridge::CvImage(std_msgs::Header(), "mono16", imD).toImageMsg();
                sensor_msgs::ImagePtr imgLabelMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imLabel).toImageMsg();
                geometry_msgs::PoseStamped poseKF;
                poseKF.pose.position.x = twc.at<float>(0, 0);
                poseKF.pose.position.y = twc.at<float>(1, 0);
                poseKF.pose.position.z = twc.at<float>(2, 0);
                
                poseKF.pose.orientation.x = q[0];
                poseKF.pose.orientation.y = q[1];
                poseKF.pose.orientation.z = q[2];
                poseKF.pose.orientation.w = q[3];

                imgRGBMsg->header.stamp = ros::Time::now();
                imgRGB_pub.publish(imgRGBMsg);

                imgDynaMaskMsg->header.stamp = ros::Time::now();
                imgDyna_pub.publish(imgDynaMaskMsg);

                imgDepthMsg->header.stamp = ros::Time::now();
                imgDepth_pub.publish(imgDepthMsg);

                imgLabelMsg->header.stamp = ros::Time::now();
                imgLabel_pub.publish(imgLabelMsg);

                poseKF.header.stamp = ros::Time::now();
                poseKF.header.frame_id = "world";
                poseKF_pub.publish(poseKF);
                // std::cout << imgDepthMsg->header.stamp << std::endl;
                // std::cout << poseKF.header.stamp << std::endl;

                // std::cout << ORB_SLAM2::Converter::toSE3Quat(Tcw).inverse() << std::endl;
                // std::cout << q[0] << "," << q[1] << "," << q[2] << "," << q[3] << std::endl;
                // std::cout << poseKF.pose.orientation << ", " << poseKF.pose.position << std::endl;
            }
            geometry_msgs::PoseStamped pose;
            pose.header.stamp = ros::Time::now();
            pose.header.frame_id = "world";
            pose.pose.position.x = twc.at<float>(0, 0);
            pose.pose.position.y = twc.at<float>(1, 0);
            pose.pose.position.z = twc.at<float>(2, 0);
            
            pose.pose.orientation.x = q[0];
            pose.pose.orientation.y = q[1];
            pose.pose.orientation.z = q[2];
            pose.pose.orientation.w = q[3];
            rgbd_path.poses.push_back(pose);
            rgbd_path.header.stamp = ros::Time::now();
            rgbd_path.header.frame_id = "world";
            gt_path.header.stamp = ros::Time::now();
            gt_path.header.frame_id = "world";
            pose_pub.publish(pose);
            rgbd_path_pub.publish(rgbd_path);
            gt_path_pub.publish(gt_path);
        }

#ifdef COMPILEDWITHC14
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        double tdynamic = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        vTimesTrack[ni] = ttrack;
        vTimesDynamic[ni] = tdynamic;
        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);

        // cv::waitKey(200);
        // usleep(0.05 * 1e6);
    }
    // while(1);
    //  Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0, totaltime2 = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
        totaltime2 += vTimesDynamic[ni];
    }
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;
    cout << "mean dynamic detecting time: " << totaltime2 / nImages << endl;
    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while (!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);
        // cout << s.c_str() << endl;
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);
        }
    }
}
