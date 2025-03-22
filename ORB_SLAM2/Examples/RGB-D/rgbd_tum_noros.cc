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

    if (argc != 5)
    {
        cerr << endl
             << "Usage: ./rgbd_tum_noros path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    // string strAssociationFilename = string(argv[4]);
    string strAssociationFilename = argv[4];
    // cout << strAssociationFilename.c_str() << endl;
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
    ORB_SLAM2::System SLAM(argv[1],
                           argv[2], ORB_SLAM2::System::RGBD, true);
    cv::FileStorage fSettings(string(argv[2]), cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << string(argv[2]) << endl;
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
    cv::Mat imgLast = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[0], -1);
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
        imRGB = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesRGB[ni], -1);
        imD = cv::imread(string(argv[3]) + "/" + vstrImageFilenamesD[ni], -1);

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
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
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
