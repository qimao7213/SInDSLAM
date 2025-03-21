#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv2/core.hpp>
#include <opencv4/opencv2/optflow.hpp>
#ifdef USECUDA
    #include <opencv2/cudaimgproc.hpp>
    #include <opencv2/cudaoptflow.hpp>
#endif
#include <pcl/point_types.h>
// #include <pcl/point_cloud.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/features/normal_3d_omp.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/filters/statistical_outlier_removal.h>
// #include <pcl/features/principal_curvatures.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/segmentation/sac_segmentation.h>
// #include <pcl/sample_consensus/model_types.h>
// #include <pcl/filters/extract_indices.h>

#include <thread>
#include <omp.h>

#include "plane_fitter_pcl.hpp"
#include "DynaDetect.h"

using namespace cv::ml;
using cv::Mat;
using std::vector;
using cv::bitwise_and;
using cv::bitwise_or;

// #define IMGSHOW
// #define IMGSAVE
static const std::string save_path = "/home/bhrqhb/catkin_ws/vSLAM/SInDSLAM/src/ORB_SLAM2/output/"; 

namespace ORB_SLAM2
{

const int width = 640;
const int height = 480;
const cv::Size imgSize(width, height);
const int nRowCluster = 3, nColCluster = 4;
const int numCluster = nRowCluster * nColCluster;
const float depth_weight = 1.5f;


const Mat element1 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(1, 1));
const Mat element2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
const Mat element3 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
const Mat element4 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(4, 4));
const Mat element5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
const Mat element7 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
const Mat element9 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
const Mat element10 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(10, 10));
const Mat element11 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11));
typedef struct EndPoint{
    cv::Point2i point;
    float curvature;
    //0->x, 1->y
    cv::Vec2i direction = cv::Vec2i(0, 0);
}EndPoint;

struct pointWithWeight
{
    pointWithWeight(int x_, int y_, float weight_)
    {
        x = x_;
        y = y_;
        weight = weight_;
    }
    int x;
    int y;
    float weight;
};

/**
 * @brief 
*/
vector<cv::Scalar> get_color(int n);
vector<double> cal_hist(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &imgDepth, cv::Mat &hist_img, const cv::Mat &Mask);
inline float calDistanceTwoPx(const cv::Point2f &pt1, const cv::Point2f &pt2)
{
    float deltaX = pt1.x - pt2.x;
    float deltaY = pt1.y - pt2.y;
    return sqrt(deltaX * deltaX + deltaY * deltaY);
}
inline float calDistanceTwoPx(const cv::KeyPoint &kpt1, const cv::KeyPoint &kpt2)
{
    return calDistanceTwoPx(kpt1.pt, kpt2.pt);
}


inline bool inBorder(const cv::Point2f &pt,const int width,const int height,const int border)
{
    return (((uint)((int)pt.x - border) <= (width - 2 * border)) && ((uint)((int)pt.y - border) <= (height - 2 * border)));
    // return ((int)pt.x > border && (int)pt.y > border &&
    //         (int) pt.x < width - border && (int)pt.y < height - border);
}
inline bool inBorder(const int row, const int col, const int height, const int weight, const int border)
{
    return (((uint)(row - border) <= (height - 2 * border)) && ((uint)(col - border) <= (weight - 2 * border)));
}

/*
*/
void applyNMS(std::vector<EndPoint>& endpoints, float distanceThreshold) {
    std::vector<EndPoint> selectedEndpoints;
    
    // Sort the keypoints based on their scores in descending order
    std::sort(endpoints.begin(), endpoints.end(), [](const EndPoint& point1, const EndPoint& point2) {
        return point1.curvature > point2.curvature;
    });

    // Iterate over the keypoints
    for (const auto& endpoint : endpoints) {
        bool isOverlapping = false;

        // Check if the current keypoint overlaps with any of the selected keypoints
        for (const auto& selectedKeypoint : selectedEndpoints) {
            int deltaX = endpoint.point.x - selectedKeypoint.point.x;
            int deltaY = endpoint.point.y - selectedKeypoint.point.y;
            float distanceSquare = (float)deltaX * deltaX + deltaY * deltaY;

            // If the distance between keypoints is below the threshold, consider the current keypoint as a duplicate
            if (distanceSquare < distanceThreshold * distanceThreshold) {
                isOverlapping = true;
                break;
            }
        }

        // If the current keypoint is not overlapping with any selected keypoint, add it to the list of selected keypoints
        if (!isOverlapping) {
            selectedEndpoints.push_back(endpoint);
        }
    }

    // Update the original vector with the selected keypoints
    endpoints = selectedEndpoints;
}

float calCosAngle(cv::Vec2i vec1, cv::Vec2i vec2)
{
    //vec: 0->x, 1->y
    float normSqure1 = vec1[0] * vec1[0] + vec1[1] * vec1[1];
    float normSqure2 = vec2[0] * vec2[0] + vec2[1] * vec2[1];
    float cosDirection = (vec1[0] * vec2[0] + vec1[1] * vec2[1])/(float)(sqrt(normSqure1 * normSqure2));
    return cosDirection;
}

cv::Mat getHistogramImage(const cv::Mat& grayImage)
{
    // 计算直方图
    cv::Mat histogram;
    int histSize = 256; // 直方图的大小
    float range[] = { 0, 256 }; // 像素值范围
    const float* histRange = { range };
    bool uniform = true, accumulate = false;
    cv::calcHist(&grayImage, 1, nullptr, cv::Mat(), histogram, 1, &histSize, &histRange, uniform, accumulate);

    // 创建直方图图像
    int histImageWidth = 512, histImageHeight = 400;
    int binWidth = cvRound(static_cast<double>(histImageWidth) / histSize);
    cv::Mat histImage(histImageHeight, histImageWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    // 归一化直方图值
    cv::normalize(histogram, histogram, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

    // 绘制直方图
    for (int i = 1; i < histSize; ++i)
    {
        cv::line(histImage, cv::Point(binWidth * (i - 1), histImageHeight - cvRound(histogram.at<float>(i - 1))),
            cv::Point(binWidth * (i - 1), histImageHeight - 1),
            cv::Scalar(196, 129, 24), 2, cv::LINE_8);
    }

    return histImage;
}

void getLabelColored(const cv::Mat &imgLabel, cv::Mat &imgLabelColor, int numCluster)
{
    int index1 = 0;
    vector<cv::Scalar> colorTab1 = get_color(numCluster);
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            index1 = row *  width + col;	
            int label = static_cast<int>(imgLabel.ptr<uchar>(row)[col]);
            // if(label > colorTab.size())
            // 	int k =0;
            cv::Scalar c = colorTab1[label];
            imgLabelColor.ptr<cv::Vec3b>(row)[col][0] = c[0];
            imgLabelColor.ptr<cv::Vec3b>(row)[col][1] = c[1];
            imgLabelColor.ptr<cv::Vec3b>(row)[col][2] = c[2];
        }
    }
}

void OpticalFlowVisualization(const cv::Mat &ImgOpticalFlow, cv::Mat &imgRGB)
{
    Mat flow_parts[2];
    //split是通道分离函数
    split(ImgOpticalFlow, flow_parts);
    Mat magnitude, angle, magn_norm;
    //cartToPolar是算出光流的幅值和方向
    cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    normalize(magnitude, magn_norm, 0.0f, 1.0f, 32);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    // //build hsv image
    Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cvtColor(hsv8, bgr, cv::COLOR_HSV2BGR);
    imgRGB = bgr.clone();
}


// If you calculate optical flow elsewhere and then want to load the optical flow data.
cv::Mat readFlowFile(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        return cv::Mat();
    }

    // Read the magic number
    float magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(float));
    if (magic != 202021.25) {
        std::cerr << "Invalid .flo file: " << filepath << std::endl;
        return cv::Mat();
    }

    // Read the width and height
    int width, height;
    file.read(reinterpret_cast<char*>(&width), sizeof(int));
    file.read(reinterpret_cast<char*>(&height), sizeof(int));

    // Read the flow data
    cv::Mat flow(height, width, CV_32FC2);
    file.read(reinterpret_cast<char*>(flow.data), width * height * 2 * sizeof(float));

    // Close the file
    file.close();
    return flow;
}


//----------------------------------myCluster's member function-------------------------------------------
void ORB_SLAM2::myCluster::calCenterPoint(const Mat &pointsPyra, const Mat &imgDepth)
{

        float totalCount = (float)cv::countNonZero(img_);
		cv::Vec3f total3D;//3D的空间点
		cv::Point2f total2D(0.0f, 0.0f);//2D的像素点
        total3D << 0.0f, 0.0f, 0.0f;
        float float1 = 0.0f;
        float float2 = 0.0f;
        float float3 = 0.0f;
        float float4 = 0.0f;
        float float5 = 0.0f;
        omp_set_num_threads(8);
        #pragma omp parallel for reduction(+:float1, float2, float3, float4, float5)
		for(int row = 0; row < height; ++row)
		{
			for(int col = 0; col < width; ++col)
			{
				int index = row * width + col;
				if(img_.ptr<uchar>((row))[col] > 1e-5)
				{
    				float1 += pointsPyra.ptr<float>(int(index))[0];
					float2 += pointsPyra.ptr<float>(int(index))[1];
					float3 += pointsPyra.ptr<float>(int(index))[2];
					// float4 += col;
					// float5 += row;
				}
			}
		}
		float1 /= (float)totalCount;
        float2 /= (float)totalCount;
        float3 /= (float)totalCount;
		float4 /= (float)totalCount;
        float5 /= (float)totalCount;
		centerPoint_ << float1, float2, float3;
		centerPx_.x = float4;
		centerPx_.y = float5;
}
void ORB_SLAM2::myCluster::merge(const myCluster &toBeMerge){
	CV_Assert(img_.size() == toBeMerge.cluster().size());
	bitwise_or(img_, toBeMerge.cluster(), img_);
	if(!this->lianjie_.empty() && !toBeMerge.lianjie().empty())
	{
		bitwise_or(lianjie_, toBeMerge.lianjie(), lianjie_);
	}
	for(int i = 0; i < toBeMerge.Contours_.size(); i++)
	{
		this->setContours(toBeMerge.Contours_[i]);
	}
}


//--------------------------------DynaDetect's member function ------------------------------------
/**
 * @brief segmentation or clustering
 * @param imgLabelInOut imgLabelInOut
 * @param points cloud points of the orginal depth image
 * @param centers clusters' 3D center point 
*/
void ORB_SLAM2::DynaDetect::SegByKmeans(cv::Mat &imgLabelInOut, cv::Mat &points, cv::Mat &centers)
{
    double timecost = 0.0;
    int index = 0;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 4, 0.07);
    const int pyramid_num = 4;
    const float pyramid_scale = 0.5f;
    const float scales[] = {1.0f, 0.5f, 0.25f, 0.125f};
    vector<Mat> depth_pyramids, resultKmeans(pyramid_num), imgLabel(pyramid_num);
    for(int level = 0; level < pyramid_num; level++)
    {
        if(level == 0)
        {
            depth_pyramids.emplace_back(imgDepth);
        }
        else
        {
            Mat depth_pyd;
            resize(depth_pyramids[level - 1], depth_pyd, cv::Size(depth_pyramids[level - 1].cols * pyramid_scale, 
                    depth_pyramids[level - 1].rows * pyramid_scale));
            depth_pyramids.emplace_back(depth_pyd);
        }
    }
    for(int level = pyramid_num - 1; level >= 0; level--)
    {
        const int height_pyramid = height * scales[level];
        const int width_pyramid = width * scales[level];
        Mat depth_pyramid = depth_pyramids[level];
        Mat labels_pyramid(cv::Size(1, height_pyramid * width_pyramid), CV_32SC1), centers_pyramid;
        Mat points_pyramid(height_pyramid * width_pyramid, 3, CV_32FC1, cv::Scalar(0));
        omp_set_num_threads(8);
        #pragma omp parallel for 
        for (int row = 0; row < height_pyramid; ++row)
        {
            for (int col = 0; col < width_pyramid; ++col)
            {
                int index = row * width_pyramid + col;
                ushort depth = depth_pyramid.ptr<ushort>(row)[col] * scales[level];//Depth scale also needs to be reduced 
                if(depth/depthScale >= (ushort)6 || depth == (ushort)0)
                {
                    // depth = -10000;
                    points_pyramid.ptr<float>(index)[2] = 0;
                    points_pyramid.ptr<float>(index)[0] = 0;
                    points_pyramid.ptr<float>(index)[1] = 0;
                }
                else
                {
                    float depth2 = (float)(depth) * (1.0f/ depthScale);
                    points_pyramid.ptr<float>(index)[2] = (float)(depth2 * depth_weight);
                    points_pyramid.ptr<float>(index)[0] = (float)((col - cx* scales[level]) * depth2 * (1.0f / (fx * scales[level]))) ;//内参全部缩小，相当于没有变化
                    points_pyramid.ptr<float>(index)[1] = (float)((row - cy* scales[level]) * depth2 * (1.0f / (fy * scales[level]))) ;
                }
                
             }
        }
        //Initial from the above layer
        if(level == pyramid_num - 1)
        {
            //The 0-th frame
            if(cv::countNonZero(imgLabelLast) == 0)
            {
                float batch_rows = (float)height_pyramid/nRowCluster;
                float batch_cols = (float)width_pyramid/nColCluster;
                for(int i = 0; i < height_pyramid; i++)
                {
                    for(int j = 0; j < width_pyramid; j++)
                    {
                        int num = cvFloor( i/batch_rows) * nColCluster + cvFloor( j/batch_cols);
                        index = i * width_pyramid + j;	
                        labels_pyramid.ptr<int>(index)[0] = num;
                    }
                }
            }
            else
            {
                cv::Mat imgLabelLast32f, imgLabelLast32fCol;
                imgLabelLast.convertTo(imgLabelLast32f, CV_32FC1);
                cv::resize(imgLabelLast32f, labels_pyramid, cv::Size(width_pyramid , height_pyramid));
                labels_pyramid = labels_pyramid.reshape(1, width_pyramid * height_pyramid);
                labels_pyramid.convertTo(labels_pyramid, CV_32SC1);
            }
            // Mat label_Img = labels_pyramid.reshape(1, height_pyramid);
            cv::kmeans(points_pyramid, numCluster, labels_pyramid, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers_pyramid);
            // Mat label_Img = labels_pyramid.reshape(1, height_pyramid);
            // int hsdf = 0;
        }
        else{
            Mat pyramid_temp, pyramid_temp1;
            imgLabel[level + 1].convertTo(pyramid_temp1, CV_32FC1);
            resize(pyramid_temp1, pyramid_temp, cv::Size(width_pyramid , height_pyramid));//resize不能处理int类型
            labels_pyramid = pyramid_temp.reshape(1, width_pyramid * height_pyramid);
            labels_pyramid.convertTo(labels_pyramid, CV_32SC1);
            // Mat label_Img = labels_pyramid.reshape(1, height_pyramid);
            cv::kmeans(points_pyramid, numCluster, labels_pyramid, criteria, 1, cv::KMEANS_USE_INITIAL_LABELS, centers_pyramid);
            // label_Img = labels_pyramid.reshape(1, height_pyramid);
        }
        // resultKmeans[level] = Mat::zeros(depth_pyramid.size(), CV_32FC1);
        imgLabel[level] =  labels_pyramid.reshape(1, height_pyramid).clone();
        if(level == 0)
        {
            points = points_pyramid.clone();
            centers = centers_pyramid.clone();
        }
    }
    imgLabelInOut = imgLabel[0].clone();
}

/**
 * @brief caculate edge from depth image
 * @param imgLabelForSegEdge Regions of the first n clusters sorted by depth.
 * @param imgTotalArea Regions with depth in (0, 6)m 
 * @param imgOccluded1 Gradient Edge
 * @param imgOccluded2 Plane Edge
*/
void ORB_SLAM2::DynaDetect::CalOccluded(const cv::Mat &imgLabelForSegEdge, 
                                        cv::Mat &imgTotalArea, 
                                        cv::Mat &imgOccluded1, 
                                        cv::Mat &imgOccluded2)
{
    cv::Mat imgDepthFiltered, imgDepth1;
    imgDepth.convertTo(imgDepth1, CV_32FC1);
    cv::medianBlur(imgDepth1,imgDepthFiltered, 5);
    double depth_max;
    cv::minMaxLoc(imgDepthFiltered, NULL, &depth_max, NULL, NULL);
    Mat imgOccluded(imgSize, CV_8UC1, cv::Scalar(0));
    const int range = 3;
    omp_set_num_threads(8);
    #pragma omp parallel for 
    for(int row = range; row < height - range; ++row)
    {
        for(int col = range; col < width - range; ++col)
        {
            float val_max = 0.0f;
            //Depth value of the block centroids.
            float depth1 = imgDepthFiltered.ptr<float>(row)[col];
            if(depth1 > 0.0f && depth1/depthScale < 6.0f)
            {
                imgTotalArea.ptr<uchar>(row)[col] = 255;
            }
            // if(depth1 <= 0)
            // {
            //     continue;
            // }
            if(row == 300 && col == 300)
                int kkdk = 1;
            // Mat px_block = imgDepthFiltered(cv::Range(row - range + 1, row + range), cv::Range(col - range + 1, col + range));
            for(int i = 0; i < 2 * range - 1; i++)
            {
                for(int j = 0; j < 2 * range - 1; j++)
                {
                    // float depth_neibor = (px_block.ptr<float>(i)[j]);
                    float depth_neibor = (imgDepthFiltered.ptr<float>(row + i - range + 1)[col + j - range + 1]);
                    // std::cout << depth_neibor2 - depth_neibor << std::endl;
                    //这里是为了排除一些深度值为0的无效区域
                    if((depth1 - depth_neibor) > (float)depth_max * 0.5f)
                    {
                        continue;
                    }
                    val_max = (abs(val_max)  > abs(depth1 - depth_neibor)) ? abs(val_max) : abs(depth1 - depth_neibor);
                }
            }

            if((val_max) > depth1 * 0.03f && (val_max) > 400.0f)  //强烈觉得这里应该增加阈值TODO
            {
                imgOccluded.ptr<uchar>(row)[col] = 255;
            }
        }
    }

    // can be calculated by MORPH_GRADIENT
    // cv::Mat imgOccludedMorpGrad, imgOccludedMorpGradBin;
    // cv::morphologyEx(imgDepthFiltered, imgOccludedMorpGrad, cv::MORPH_GRADIENT, element4);
    // imgOccludedMorpGradBin = (imgOccludedMorpGrad > 0.1 * imgDepthFiltered);
    // imgOccludedMorpGradBin.setTo(0, imgDepthFiltered == 0.0f);
    // imgOccludedMorpGradBin.setTo(0, imgDepthFiltered >= 20.0f * depthScale);
    // imgOccludedMorpGradBin.setTo(0, imgOccludedMorpGrad < 400.0f);


    //--------------------calculate endpoint----------------
    cv::morphologyEx(imgOccluded, imgOccluded, cv::MORPH_OPEN, element4);
    std::vector<EndPoint> endPoints;
    std::vector<bool[12]> fastPointAround;
    // cv::bitwise_not;
    for(int row = 3; row < height - 3; ++row)
    {
        for(int col = 3; col < width - 3; ++col)
        {
            if(imgOccluded.ptr<uchar>(row)[col] != 255)
            {
                continue;
            }
            //这个方向是一个向量，记录周围255的点的x和y的和
            cv::Vec2i tmpDirection(0, 0);
            bool around[12] = {0};
            int aroundSum = 0;
            for(int i = 0; i < 12; ++i)
            {
                if(imgOccluded.ptr<uchar>(row + aroundPoint[i].y)[col + aroundPoint[i].x] == 255)
                {
                    around[i] = 1;
                    aroundSum ++;
                    
                    tmpDirection[0] += aroundPoint[i].x;
                    tmpDirection[1] += aroundPoint[i].y;
                }
            }
            if(aroundSum <= 4)
            {
                EndPoint tmpPoint;
                tmpPoint.point.x = col;
                tmpPoint.point.y = row;
                // tmpPoint.curvature = (float)imgCurvature.ptr<float>(row)[col];
                tmpPoint.direction[0] = tmpDirection[0];
                tmpPoint.direction[1] = tmpDirection[1];
                endPoints.emplace_back(tmpPoint);
            }
        }
    }

    cv::Mat imgOccludedForPlane = imgOccluded.clone();
    // cv::morphologyEx(imgOccluded, imgOccluded, cv::MORPH_OPEN, element3);
    applyNMS(endPoints, 6.0f);
    //--------------------for draw image------------------
    #ifdef IMGSAVE    
        cv::Mat imgOccludedWithEndpointNoMaxSup = imgOccluded.clone();
        cv::cvtColor(imgOccludedWithEndpointNoMaxSup, imgOccludedWithEndpointNoMaxSup, cv::COLOR_GRAY2BGR);    
        for(auto pt:endPoints)
        {   
            //画图的时候需要调整一下
            // if(imgOccluded.ptr<uchar>(pt.point.y)[pt.point.x] == 255)
            {
                cv::circle(imgOccludedWithEndpointNoMaxSup, pt.point, 3, cv::Scalar(0, 0, 255), cv::FILLED);
            }
            // cv::circle(imgOccludedForPlane, pt.point, 2, cv::Scalar(0), cv::FILLED);
        }
        cv::imwrite(save_path + "imgEdge1.png", imgOccludedWithEndpointNoMaxSup);
    #endif
    //--------------------for draw image------------------

    //imgHalf用来限定imgdiff22的区域，不想在整张图像上都提取
    // Mat imgHalf(imgSize, CV_8UC1, cv::Scalar(0));
    // cv::rectangle(imgHalf, cv::Point2i(0.0f * width , 0.6f * height), cv::Point2i(1.0f * width, 1.0f * height), cv::Scalar(255), cv::FILLED);

    cv::Mat imgEdgeByPlane(imgSize, CV_8UC1, cv::Scalar(0));
    pcl::PointCloud<pcl::PointXYZ> Cloud;
	Cloud.points.reserve(imgDepth1.rows * imgDepth1.cols);
	Cloud.is_dense = false;
    for (int v = 0; v < imgDepth.rows; v += 1)
    {
        for (int u = 0; u < imgDepth.cols; u += 1) 
		{
            float d = (float)imgDepth.ptr<ushort>(v)[u]; // 深度值
            if (d < 1e-3f) 
			{
				pcl::PointXYZ p;
				p.x = std::numeric_limits<float>::quiet_NaN ();
				p.y = std::numeric_limits<float>::quiet_NaN ();
				p.z = std::numeric_limits<float>::quiet_NaN ();
				Cloud.points.push_back(p);
				continue;
			} // 为0表示没有测量到
            Eigen::Vector3f point;
            point[2] = (d) * (1.0f / depthScale);
            point[0] = (u - cx) * point[2] / fx;
            point[1] = (v - cy) * point[2] / fy;
            Eigen::Vector3f pointWorld = point;
            pcl::PointXYZ p;
            p.x = pointWorld[0];
            p.y = pointWorld[1];
            p.z = pointWorld[2];
            Cloud.points.push_back(p);
        }
    }
	Cloud.width = imgDepth1.cols;
    Cloud.height = imgDepth1.rows;    
    
    // Plane segmentation
    PlanarContourExtraction pce(Cloud);
    pce.run(imgEdgeByPlane);

    #ifdef IMGSAVE
        cv::imwrite(save_path + "planeContours/" + std::to_string(nImg) + ".png", imgEdgeByPlane);
    #endif
    cv::Mat imgEdgeByPlaneCopy = imgEdgeByPlane.clone();
    imgEdgeByPlane = imgEdgeByPlane - imgOccludedForPlane;
    // bitwise_and(imgHalf, imgEdgeByPlane, imgEdgeByPlane);
    // cv::morphologyEx(imgEdgeByPlane, imgEdgeByPlane, cv::MORPH_DILATE, element7);
    {
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(imgEdgeByPlane, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );

        cv::Mat oneContour(imgSize, CV_8UC1, cv::Scalar(0));
        imgEdgeByPlane.setTo(0);
        for(int i = 0; i < contours.size(); i ++)
        {
            if((contours[i].size()) < 25)
            {
                continue;
            }
            int isEndPoint = 0;
            oneContour.setTo(0);
            cv::drawContours(oneContour, contours, i, cv::Scalar(255), 2);
            cv::morphologyEx(oneContour, oneContour, cv::MORPH_DILATE, element10);
            for(int j = 0; j < endPoints.size(); j++)
            {
                if(oneContour.ptr<uchar>(endPoints[j].point.y)[endPoints[j].point.x] == 255)
                {
                    isEndPoint = isEndPoint + 1;
                    if(isEndPoint >= 1)
                    {
                        break;
                    }
                }
            }
            if(isEndPoint >= 1)
            {
                cv::morphologyEx(oneContour, oneContour, cv::MORPH_ERODE, element7);
                imgEdgeByPlane += oneContour;
                // cv::drawContours(imgOccludedWithEndpointNoMaxSup, contours, i, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    // bitwise_and(imgLabelForSegEdge, imgEdgeByPlane, imgEdgeByPlane);
    imgOccluded2 = imgEdgeByPlane.clone();
    bitwise_or(imgOccluded, imgEdgeByPlane, imgOccluded1);
    cv::morphologyEx(imgOccluded1, imgOccluded1, cv::MORPH_CLOSE, element3);
}

/**
 * @brief SegAndMerge
 * @param imgOccluded Depth Edge = Gradient Edge + Plane Edge
 * @param imgOccluded2 Plane Edge
 * @param imgLabelForSegEdge 
 * @param points_pyramid Point Cloud
 * @param imgLabelNew 
 * @param count0 not use
*/
void ORB_SLAM2::DynaDetect::SegAndMergeV2(const std::vector<Mat> &allLabels, 
                                            const cv::Mat &imgOccluded, 
                                            const cv::Mat &imgOccluded2,
                                            const cv::Mat &imgLabelForSegEdge, 
                                            const cv::Mat &points_pyramid,
                                            cv::Mat &imgLabelNew,
                                            int count0)
{
    vector<myCluster> allClusters;
    //The last cluster is the region with too large or zero depth values, so it is not processed.
    double end1 = cv::getTickCount();
    for(int i = 0; i < allLabels.size() - 1; i++)
    {
        Mat eachLabel = allLabels[i].clone();
        Mat eachLabelOrign = eachLabel.clone();
        eachLabel = eachLabel - imgOccluded;
        // cv::morphologyEx(eachLabel, eachLabel, cv::MORPH_ERODE, element);
        Mat imgOccludedDilate;
        cv::morphologyEx(imgOccluded, imgOccludedDilate, cv::MORPH_DILATE, element10);
        cv::morphologyEx(eachLabel, eachLabel, cv::MORPH_OPEN, element4);
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(eachLabel, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );
        for(int c = 0; c < contours.size(); c++)
        {
            if(contours[c].size() > 50 && contourArea(contours[c]) > 80)//Only contour pixel size larger than n are considered.
            {
                Mat temp = Mat::zeros(imgSize, CV_8UC1);
                Mat drawContour = temp.clone();
                drawContours(temp, contours, c, 255, cv::FILLED);
                cv::morphologyEx(temp, temp, cv::MORPH_DILATE, element9);
                bitwise_and(temp, eachLabelOrign, temp);
                myCluster newCluster(temp);
                newCluster.setContours(contours[c]);
                newCluster.area = (float)countNonZero(temp);
                cv::morphologyEx(temp, temp, cv::MORPH_DILATE, element7);
                newCluster.setClusterDilated(temp);
                drawContours(drawContour, contours, c, 255, 2);
                Mat temp1 = drawContour - imgOccludedDilate;
                // cv::bitwise_and(imgOccludedDilate, drawContour, temp1);
                // temp1 = drawContour - temp1;

                bitwise_and(temp1, imgLabelForSegEdge, temp1);
                if(countNonZero(temp1) > 20)
                {
                    vector<vector<cv::Point>> contours2;
                    vector<cv::Vec4i> hierarchy2;
                    cv::findContours(temp1, contours2, hierarchy2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE );
                    for(auto iter = contours2.begin(); iter != contours2.end(); )
                    {
                        if(iter->size() < 30)
                        {
                            contours2.erase(iter);
                            continue;
                        }
                        iter++;
                    }
                    if(contours2.size() > 0)
                    {
                        temp1.setTo(0);
                        drawContours(temp1, contours2, -1, cv::Scalar(255), cv::FILLED);
                        newCluster.lianjie_ = temp1.clone();
                    }
                }
                // if(i < count0)
                // {
                //     newCluster.isClose = 1;
                // }
                // double t1 = cv::getTickCount();
                newCluster.calCenterPoint(points_pyramid, imgDepth);
                // double t2 = cv::getTickCount();
                // double time_elapsed0 = (t2-t1)/(double)cv::getTickFrequency();
                // std::cout<<"计算中心点耗时 : "<< time_elapsed0 << std::endl;
                allClusters.emplace_back(newCluster);
            }
        }						
    }
    double end2 = cv::getTickCount();
    std::cout << "Cluser divid timecost: " << (end2 - end1)/cv::getTickFrequency() << std::endl;

    int countCluster = allClusters.size();
    Mat imgTotalCluster(imgSize, CV_8UC1, cv::Scalar(countCluster + 1));

    std::vector<float> Score;
    for(int i = 0; i < allClusters.size(); i++)
    {   
        // Mat oneCluster = allClusters[i].cluster();
        // Mat lianjie = allClusters[i].lianjie();
        allClusters[i].socre = (float)(allClusters[i].area * 0.0003f - allClusters[i].centerPoint()[2]);
        Score.emplace_back((float)(allClusters[i].area * 0.0003f - allClusters[i].centerPoint()[2]));        
        // int kk = 0;
    }
    //Sort in descending order.
    std::sort(allClusters.begin(), allClusters.end(), 
            [&](const myCluster &cluster1, const myCluster &cluster2)->bool
            {return cluster1.socre > cluster2.socre;});
    for(int i = 0; i < allClusters.size(); i++)
    {   
        // Mat oneCluster = allClusters[i].cluster();
        imgTotalCluster.setTo(i, allClusters[i].cluster());
        // int kk = 0;
    }
    //--------------------for draw image------------------
    // Mat imgTotalClusterColor(imgSize, CV_8UC3, cv::Scalar(0, 0, 0));
    // getLabelColored(imgTotalCluster, imgTotalClusterColor, countCluster);
    // Mat imgTotalClusterSort = imgTotalCluster.clone() + 1;
    // Mat imgTotalClusterSortColor(imgSize, CV_8UC3, cv::Scalar(0, 0, 0));;
    // imgTotalClusterSort.setTo(0, imgTotalCluster == (countCluster + 1));
    // getLabelColored(imgTotalClusterSort, imgTotalClusterSortColor, countCluster);
    // cv::imwrite(save_path + "imgLabel2.png", imgTotalClusterSortColor);
    //--------------------for draw image------------------

    double depth_max;
    cv::minMaxLoc(imgDepth, NULL, &depth_max, NULL, NULL);
    cv::Mat imgDepthNormalized =  imgDepth/depth_max * 255;
    imgDepthNormalized.convertTo(imgDepthNormalized, CV_8UC1);
    //Build Region Adjacency Graph (RAG) Matrix to record correlation between each two clusters
    double end4 = cv::getTickCount();
    // std::cout << "中间处理：" << (end4 - end2)/cv::getTickFrequency() << std::endl;
    cv::Mat correlationMatrixTotal(cv::Size(countCluster + 1, countCluster + 1), CV_32FC1, cv::Scalar(0.0f));//Total
    cv::Mat correlationMatrix1(cv::Size(countCluster + 1, countCluster + 1), CV_32FC1, cv::Scalar(0.0f));//Adjacency
    cv::Mat correlationMatrix2(cv::Size(countCluster + 1, countCluster + 1), CV_32FC1, cv::Scalar(0.0f));//Common fakeEdge
    cv::Mat correlationMatrix3(cv::Size(countCluster + 1, countCluster + 1), CV_32FC1, cv::Scalar(0.0f));//Depth similarity
    cv::Mat correlationWeight(cv::Size(countCluster + 1, countCluster + 1), CV_32FC1, cv::Scalar(1.0f));//Weight
    cv::Mat rejectMarix(cv::Size(countCluster + 1, countCluster + 1), CV_32FC1, cv::Scalar(1.0f));//Rejection Matrix, if the Depth similarity is too low.
    float correlationValue = 0.0f;
    float correlationValue2 = 0.0f;
    float correlationValue3 = 0.0f;
    float thredshold = 0.9f;
    int smallLabel = (int)std::min(0.7f * countCluster, 15.0f);
    double timecostXX;
    for(int i = 0; i < countCluster; i ++)
    {
        myCluster cluster1 = allClusters[i];
        Mat imgCluster1 = cluster1.cluster();
        Mat imgClusterDilated1 = cluster1.clusterDilated();
        Mat lianjie1 = cluster1.lianjie(); 

        for(int j = i + 1; j < countCluster; j ++)
        {
            correlationValue = 0.0f;
            correlationValue2 = 0.0f;
            correlationValue3 = 0.0f;
            int lessOne;
            float lessArea;
            int lessLabel;
            
            myCluster cluster2 = allClusters[j];
            if(cluster1.area < cluster2.area)
            {
                lessOne = 1;
                lessArea = (float)cluster1.area;
                lessLabel = i;
            }
            else
            {
                lessOne = 2;
                lessArea = (float)cluster2.area;
                lessLabel = j;
            }
            if(lessLabel < 10)
            {
                correlationWeight.ptr<float>(i)[j] = 0.7f;
                correlationWeight.ptr<float>(j)[i] = 0.7f;
            }
            //It is a small area, and it needs to be encouraged for merging.
            else if(lessLabel > smallLabel)
            {
                correlationWeight.ptr<float>(i)[j] = 2.0f;
                correlationWeight.ptr<float>(j)[i] = 2.0f;
            }
            Mat imgCluster2 = cluster2.cluster();
            Mat imgClusterDilated2 = cluster2.clusterDilated();
            Mat lianjie2 = cluster2.lianjie(); 

            Mat imgOverlap;
            cv::bitwise_and(imgClusterDilated1, imgClusterDilated2, imgOverlap);

            bool isMust = false;
            if(cv::countNonZero(imgOverlap) > std::min(200.0f, lessArea * 0.4f) )
            {
                Mat imgOverlapWithPlaneEdge;
                cv::bitwise_and(imgOverlap, imgOccluded2, imgOverlapWithPlaneEdge);
                correlationValue = 1.0f;
                //----------------Depth Similarity-----------------
                Mat hist;
                int isMergeFlag;
                double end4 = cv::getTickCount();
                vector<double> isMerge = cal_hist(imgCluster1, imgCluster2, imgDepthNormalized, hist, Mat());
                double end5 = cv::getTickCount();
                timecostXX += end5 - end4;
                correlationValue3 = (float)(isMerge[0] + isMerge[1] + isMerge[2] * 0.0005);
                //If the depth similarity is too low or there are common planar edges, it should be directly rejected.
                //However, if one of the clusters is a small area, this criterion is not applied.
                if((cv::countNonZero(imgOverlapWithPlaneEdge) > 100 && lessLabel < smallLabel))
                {
                    rejectMarix.ptr<float>(i)[j] = 0.0f;
                    rejectMarix.ptr<float>(j)[i] = 0.0f;
                    continue;
                }
                else if(correlationValue3 < 0.19f  &&  lessLabel < smallLabel && !isMust)
                {
                    rejectMarix.ptr<float>(i)[j] = 0.0f;
                    rejectMarix.ptr<float>(j)[i] = 0.0f;
                    continue;
                }
                //-----------------calculate common fake edges-------
                if(!lianjie1.empty() && !lianjie2.empty())
                {
                    Mat l1Andl2;
                    bitwise_and(lianjie1, lianjie2, l1Andl2);
                    int overlapLianjie = cv::countNonZero(l1Andl2);
                    if(overlapLianjie > 0)
                    {
                        int lianjie1_area = cv::countNonZero(lianjie1);
                        int lianjie2_area = cv::countNonZero(lianjie2);
                        if(overlapLianjie > std::min(50, (int)(0.5 * std::min(lianjie1_area, lianjie2_area))))
                        {
                            correlationValue2 = (float)overlapLianjie;
                            // int k1 = cv::countNonZero(lianjie1);
                            // int k2 = cv::countNonZero(lianjie2);
                            // Mat lianjieAndAdj;
                            // bitwise_and(l1Andl2, imgOverlap, lianjieAndAdj);
                            if((overlapLianjie > 0.62 * lianjie1_area) || (overlapLianjie > 0.62 * lianjie2_area))
                            {
                                correlationValue2 = std::max(250, overlapLianjie);
                                isMust = true;
                            }
                        }
                    }
                    int kkk = 0;
                }
                correlationMatrix1.ptr<float>(i)[j] = correlationValue;
                correlationMatrix1.ptr<float>(j)[i] = correlationValue;
                correlationMatrix2.ptr<float>(i)[j] = correlationValue2;
                correlationMatrix2.ptr<float>(j)[i] = correlationValue2;
                correlationMatrix3.ptr<float>(i)[j] = correlationValue3;
                correlationMatrix3.ptr<float>(j)[i] = correlationValue3;  
            }
        }
    }
    correlationMatrixTotal = (correlationMatrix2 * 0.01f + correlationMatrix3).mul(rejectMarix).mul(correlationWeight);
    int countMerged = 0;
    Mat correlationMatrixTotalCopy = correlationMatrixTotal.clone();
    double end5 = cv::getTickCount();
    std::cout << "Build RAG timecost: " << (end5 - end4)/cv::getTickFrequency() << std::endl;
    // std::cout << "bitwise_and timecost :" << timecostXX/cv::getTickFrequency() << std::endl;
    //-----------Visualizing the RAG Matrix using a heatmap color------------------- 
    //--------------------for draw image------------------
    // Mat correlationMatrixTotalCopy2 = correlationMatrixTotal.clone();
    // Mat correlationMatrixTotalCopy3(correlationMatrixTotalCopy2.size(), CV_8UC3, cv::Scalar(0, 0, 0));;
    // double maxValue, minValue;
    // cv::minMaxLoc(correlationMatrixTotalCopy, &minValue, &maxValue);
    // correlationMatrixTotalCopy2 -= minValue;
    // maxValue -= minValue;
    // correlationMatrixTotalCopy2 *=  255.0/maxValue;
    // correlationMatrixTotalCopy2.convertTo(correlationMatrixTotalCopy2, CV_8UC1);
    // cv::Mat colormap_my(cv::Size(256, 1), CV_8UC1, cv::Scalar(0));
    // for(int i = 0; i < 256; ++i)
    // {
    //     colormap_my.at<uchar>(0, i) = i;
    // }
    // cv::Mat colormap_my2, colormap_my4(1, 256, CV_8UC1);
    // cv::applyColorMap(colormap_my, colormap_my2, cv::COLORMAP_JET);
    // cv::Mat colormap_my3 = colormap_my2.colRange(32,224);
    // cv::resize(colormap_my3, colormap_my3, cv::Size(256, 1));
    // for(int row = 0; row < correlationMatrixTotalCopy2.rows; row++)
    // {
    //     for(int col = 0; col < correlationMatrixTotalCopy3.cols; col++)
    //     {
    //         int value = (int)correlationMatrixTotalCopy2.at<uchar>(row, col);
    //         correlationMatrixTotalCopy3.at<cv::Vec3b>(row, col)[0] = colormap_my3.at<cv::Vec3b>(value)[0];
    //         correlationMatrixTotalCopy3.at<cv::Vec3b>(row, col)[1] = colormap_my3.at<cv::Vec3b>(value)[1];
    //         correlationMatrixTotalCopy3.at<cv::Vec3b>(row, col)[2] = colormap_my3.at<cv::Vec3b>(value)[2];
    //     }

    // }
    // // cv::applyColorMap(correlationMatrixTotalCopy2, correlationMatrixTotalCopy2, cv::COLORMAP_JET);
    // cv::imwrite(save_path + "imgMatrix.png", correlationMatrixTotalCopy3);
    //---------------------for draw image-------------------------

    //A better approach is to use hierarchical clustering.
    //Record the merging relationships between each two clusters.
    vector<vector<int>> merge(countCluster + 1);
    //Record whether a cluster has been merged. 0 indicates no, and 1 indicates yes.
    vector<int> mergeSitutation(countCluster + 1, 0); 
    //i is the column of the RAG matrix, representing the current cluster; 
    //j is the row of the RAG matrix, representing the one to be merged if found.
    for(int i = 0; i < std::min(numCluster - 1 + countMerged, countCluster); i++)
    {
        for(int j = i + 1; j < std::min(numCluster - 1 + countMerged, countCluster); j ++)
        {
            float sorce = correlationMatrixTotal.ptr<float>(j)[i];
            //If this condition is met, it indicates that j will be merged, 
            //but it is not yet determined with which cluster it will be merged.
            if(sorce > thredshold)
            {
                auto determineWhichToMerge = correlationMatrixTotal.colRange(j, j+1).rowRange(0, j).clone();
                int toMerge = i;
                float toMergeValue = correlationMatrixTotal.ptr<float>(j)[i];
                for(int k = 0; k < determineWhichToMerge.rows; k++)
                {
                    if(determineWhichToMerge.at<float>(k, 0) > toMergeValue)
                    {
                        toMerge = k; 
                    }
                }
                mergeSitutation[j] = 1;
                merge[toMerge].push_back(j);
                auto oneCol = correlationMatrixTotal.colRange(j, j+1).clone();
                correlationMatrixTotal.colRange(toMerge, toMerge+1) += oneCol;
                correlationMatrixTotal.rowRange(toMerge, toMerge+1) += oneCol.t();
                correlationMatrixTotal.colRange(j, j+1).setTo(0.0f);//clear the information about cluster j
                correlationMatrixTotal.rowRange(j, j+1).setTo(0.0f);
                countMerged ++;
            }
        }
    }
    //Process the remaining clusters with small scores
    for(int i = std::min(numCluster - 1 + countMerged, countCluster); i < countCluster; i ++)
    {
        int mergeCluster = countCluster;
        float maxScore = 0.2f;
        for(int j = 0; j < i; j ++)
        {
            //By default, these small areas are merged into the invalid region. 
            //Moreover, small areas can only be merged with the larger areas that precede them, not with smaller areas.
            float score = correlationMatrixTotal.ptr<float>(j)[i];
            if(score > maxScore)
            {
                maxScore = score;
                mergeCluster = j;
            }
        }
        mergeSitutation[i] = 1;
        merge[mergeCluster].push_back(i);
        auto oneCol = correlationMatrixTotal.colRange(i, i+1).clone();
        correlationMatrixTotal.colRange(mergeCluster, mergeCluster+1) += oneCol;
        correlationMatrixTotal.rowRange(mergeCluster, mergeCluster+1) += oneCol.t();
        correlationMatrixTotal.colRange(i, i+1).setTo(0.0f);
        correlationMatrixTotal.rowRange(i, i+1).setTo(0.0f);
    }
    //label 0 is depth-invalid region
    int labelindex = 1;
    for(int i = 0; i <  countCluster; i++)
    {
        if(!mergeSitutation[i])
        {
            Mat oneCluster = (imgTotalCluster == i);
            for(int j = 0; j < merge[i].size(); j++)
            {
                int megerLabel = merge[i][j];
                // Mat temp = (imgTotalCluster == merge[i][j]);
                cv::bitwise_or((imgTotalCluster == megerLabel), oneCluster, oneCluster);
                for(int k = 0; k < merge[megerLabel].size(); k++)
                {
                    cv::bitwise_or((imgTotalCluster == merge[megerLabel][k]), oneCluster, oneCluster);
                }
            }
            // cv::morphologyEx(oneCluster, oneCluster, cv::MORPH_DILATE, element3);
            imgLabelNew.setTo(labelindex, oneCluster);
            labelindex ++;
        }
    }
    int bbb = 0;
}

/**
 * @brief Parallel thread for Optical Flow Calculation
*/
void ORB_SLAM2::DynaDetect::DetectDynaByDenseOpticalFLow(std::promise<stImgMasks> &imgMasks)
{
    static double timecost5, timecost6;

    // If CUDA is available, use BroxFlow; otherwise, use DeepFlow
    #ifdef USECUDA
        cv::Ptr<cv::cuda::BroxOpticalFlow> denseFlow3 = cv::cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);
    #else
        cv::Ptr<cv::DenseOpticalFlow> denseFlow = cv::optflow::createOptFlow_DeepFlow();
    #endif
    const float scale_element = 0.6f;
    cv::Mat imgDenseFlow;
    //Scale down the image by scale_element.
    cv::Mat imgGrayMin, imgGrayLastMin, imgGrayLastLastMin;
    cv::resize(imgGray,imgGrayMin, cv::Size(scale_element * width, scale_element * height));
    cv::resize(imgGrayLast,imgGrayLastMin, cv::Size(scale_element * width, scale_element * height));
    cv::resize(imgGrayLastLast,imgGrayLastLastMin, cv::Size(scale_element * width, scale_element * height));   
    #ifdef USECUDA 
        cv::cuda::GpuMat imgGrayCuda, imgGrayCudaLast, imgGrayCudaLastLast, imgResultCuda1, imgResultCuda2;
        cv::cuda::GpuMat imgGrayCuda_32F, imgGrayCudaLast_32F, imgGrayCudaLastLast_32F;
        imgGrayCuda.upload(imgGrayMin);
        imgGrayCudaLast.upload(imgGrayLastMin);
        imgGrayCudaLastLast.upload(imgGrayLastLastMin);
        imgGrayCuda.convertTo(imgGrayCuda_32F, CV_32F, 1.0f / 255.0f);
        imgGrayCudaLast.convertTo(imgGrayCudaLast_32F, CV_32F, 1.0f / 255.0f);
        imgGrayCudaLastLast.convertTo(imgGrayCudaLastLast_32F, CV_32F, 1.0f / 255.0f);
    #endif
    //用来检验变换后的图像是否正确
    // cv::Mat kk1, kk2, kk3;
    // imgGrayCuda_32F.download(kk1);
    // imgGrayCudaLast_32F.download(kk2);
    // imgGrayCudaLastLast_32F.download(kk2);
    
    //将RGB图也缩放，是为了给RLOF光流使用
    // cv::Mat imgRGBMin, imgRGBLastMin, imgRGBLastLastMin;
    // cv::resize(imgRGB,imgRGBMin, cv::Size(0.5 * width, 0.5 * height));
    // cv::resize(imgRGBLast,imgRGBLastMin, cv::Size(0.5 * width, 0.5 * height));
    // cv::resize(imgRGBLastLast,imgRGBLastLastMin, cv::Size(0.5 * width, 0.5 * height));
    double beginTotal, endTotal, begin1, end1;
    beginTotal = cv::getTickCount();
    float baseFlow = 0.0f;

    //We first attempt to calculate the optical flow using frames n and n−2. 
    //If the average optical flow value is too large, then we use frames n and n−1.
    bool largeMotion = false;
    {
        begin1 = cv::getTickCount();
        //Note that imgGray is the first parameter!!
        #ifdef USECUDA
            denseFlow3->calc(imgGrayCuda_32F, imgGrayCudaLastLast_32F, imgResultCuda1);
            imgResultCuda1.download(imgDenseFlow); 
        #else       
            denseFlow->calc(imgGrayMin, imgGrayLastLastMin, imgDenseFlow);
        #endif
        // end1 = cv::getTickCount();
        // std::cout << "稠密光流Brox:" << ((end1 - begin1)/ cv::getTickFrequency()) << std::endl;
        // imgDenseFlow
        imgDenseFlow *= -1.0f;
        Mat imgFlowParts[2];
        split(imgDenseFlow, imgFlowParts);
        Mat imgMagnitude, imgAngle;
        cartToPolar(imgFlowParts[0], imgFlowParts[1], imgMagnitude, imgAngle, true);

        //Calculate average optical flow value.
        cv::Mat imgMagnitudeNomolized;
        double maxFlow;
        cv::minMaxLoc(imgMagnitude, 0, &maxFlow);
        imgMagnitudeNomolized = imgMagnitude * (255.0/maxFlow);
        imgMagnitudeNomolized.convertTo(imgMagnitudeNomolized, CV_8UC1);
        cv::Mat histogram;
        int histSize = 256; 
        float range[] = { 0, 256 }; 
        const float* histRange = { range };
        cv::calcHist(&imgMagnitudeNomolized, 1, nullptr, cv::Mat(), histogram, 1, &histSize, &histRange, true, false);
        float ratio = 0.0f;
        int endFlow = 10.0f * scale_element * 255.0f/maxFlow;//之前使用的是4，取得了不错的效果
        int endFlow2 = 0;
        float totalpixel =  width * height * scale_element * scale_element;
        for(int i = 0; i < 255; ++i)
        {
            ratio += histogram.ptr<float>(i)[0];
            if(ratio > 0.3f * totalpixel)
            {
                endFlow2 = i;
                break;
            }
            // if(ratio)
        }
        if(endFlow2 > endFlow)
        {
            largeMotion = true;
        }
        // auto time4 = cv::getTickCount();
        // std::cout << "方法2:" << ((time4 - time3)/ cv::getTickFrequency()) << std::endl;
    }
 
    // cout << "-------largeMotionFlag -----" << (largeMotion == largeMotion1 )<< "------------- "  << endl;
    // If largeMotion
    if(largeMotion)
    {
        #ifdef USECUDA
            denseFlow3->calc(imgGrayCuda_32F, imgGrayCudaLast_32F, imgResultCuda1);
            imgResultCuda1.download(imgDenseFlow); 
        #else       
            denseFlow->calc(imgGrayMin, imgGrayLastMin, imgDenseFlow);
        #endif
        imgDenseFlow *= -1.0f;
        std::cout << ">>>>>>>>>>>>LargeMotion Flag" << std::endl;
    }

    cv::Ptr<cv::DenseOpticalFlow> ptr_refine = cv::VariationalRefinement::create();

    // begin1 = cv::getTickCount();
    if(!largeMotion)
    {
        ptr_refine->calc(imgGrayMin, imgGrayLastLastMin, imgDenseFlow);
    }
    else
    {
        ptr_refine->calc(imgGrayMin, imgGrayLastMin, imgDenseFlow);
    }
    cv::resize(imgDenseFlow,imgDenseFlow, cv::Size(width, height));
    //Since the image has been scaled down, the optical flow will also be reduced. 
    //Therefore, we need to scale it back up here.
    imgDenseFlow *= 1.0f/ scale_element;

    // If you calculate optical flow elsewhere and then want to load the optical flow data.
    // const std::string denseFlow_file = "/home/bhrqhb/dataset/bonn/rgbd_bonn_person_tracking/dense_flow/";
    // cv::Mat imgDenseFlowDIP = readFlowFile(denseFlow_file + std::to_string(nImg) + "_flow.flo");
    // cv::Mat imgDenseFlowVis, imgDenseFlowDIPVis;
    // OpticalFlowVisualization(imgDenseFlow, imgDenseFlowVis);
    // OpticalFlowVisualization(imgDenseFlowDIP, imgDenseFlowDIPVis);
    // // cv::imshow("imgDenseFlowDIP", imgDenseFlowDIPVis);
    // // cv::imshow("imgDenseFlow", imgDenseFlowVis);
    // // cv::waitKey(0);
    // imgDenseFlow = imgDenseFlowDIP.clone();
    
    end1 = cv::getTickCount();
    std::cout << "DenseFlow + Refine:" << ((end1 - begin1)/ cv::getTickFrequency()) << std::endl;
    timecost5 += ((end1 - begin1)/ cv::getTickFrequency());
    cv::RNG rng(12345);
    cv::Mat imgWeight(imgSize, CV_8UC1, cv::Scalar(0));
    std::vector<pointWithWeight> vPtsWithWeight;
    vPtsWithWeight.reserve(width * height * 0.02f);

    //The dynamic ratio within each cluster in the previous frame is also taken into consideration.
    vector<float> clusterWeight(numCluster, 0.0f);
    Mat oneCluster, oneClusterDyna;
    Mat imgDynaLast0 = (imgDynaLast == 255);
    for(int i = 1; i < numCluster; i ++)
    {
        oneCluster = (imgLabelLast == i);
        cv::bitwise_and(oneCluster, imgDynaLast0, oneClusterDyna);
        clusterWeight[i] = (float)cv::countNonZero(oneClusterDyna)/(float)(cv::countNonZero(oneCluster) + 1.0f);
    }

    //Calculate the weight for each downsampled point.
    // omp_set_num_threads(8);
    // #pragma omp parallel for 
    for(int row = 10; row < height - 0; row += 10)
    {
        for(int col = 10; col < width - 0; col += 10)
        {   

            float randomd = (float)rng.gaussian(0.5);
            //Depth-Invalid Regions
            if(imgDynaLast.ptr<uchar>(row)[col] < 20)
            {
                vPtsWithWeight.emplace_back(pointWithWeight(col, row, randomd + 1.0f));
            }
            //Static Regions
            else if((uint)(imgDynaLast.ptr<uchar>(row)[col] - 20) <= 230 - 20)
            {
                int label = imgLabelLast.ptr<uchar>(row)[col];
                vPtsWithWeight.emplace_back(pointWithWeight(col, row, randomd + 1.2f * (1.0f - clusterWeight[label])));
            }
            else
            {
                vPtsWithWeight.emplace_back(pointWithWeight(col, row, randomd + 0.4f));
            }
        }
    }
    //--------------------for draw image------------------
    // Visualize Point Weight   
    #ifdef IMGSAVE 
        for(int i = 0; i < vPtsWithWeight.size(); i ++)
        {
            cv::circle(imgWeight, cv::Point2i(vPtsWithWeight[i].x, vPtsWithWeight[i].y), 
                    abs(vPtsWithWeight[i].weight) * 3, cv::Scalar(255), cv::FILLED);
        }
        cv::imwrite(save_path + "WeightOfSampledPoints.png", imgWeight);
    #endif
    // //--------------------for draw image------------------

    std::sort(vPtsWithWeight.begin(), vPtsWithWeight.end(), [&](const pointWithWeight &pt1, const pointWithWeight &pt2)->bool{
        return pt1.weight > pt2.weight;
    });
    std::vector<cv::Point2f> inputPoints, inputPointsLast;
    for(int i = 0; i < vPtsWithWeight.size(); ++i)
    {
        float ptCol = (float)vPtsWithWeight[i].x;
        float ptRow = (float)vPtsWithWeight[i].y;
        cv::Vec2f optFlowXY = imgDenseFlow.at<cv::Vec2f>(ptRow, ptCol);
        if(inBorder(ptRow - optFlowXY[1], ptCol - optFlowXY[0], height, width, 0))
        {
            inputPoints.emplace_back(cv::Point2f(ptCol, ptRow));
            inputPointsLast.emplace_back(cv::Point2f(ptCol - optFlowXY[0], ptRow - optFlowXY[1]));
        }
    }

    cv::Mat imgDenseFlowDiff = imgDenseFlow.clone();

    cv::Mat HMatrix = cv::findHomography(inputPoints, inputPointsLast, cv::noArray(), cv::RHO);//double类型的
    double h1 = HMatrix.ptr<double>(0)[0];
    double h2 = HMatrix.ptr<double>(0)[1];
    double h3 = HMatrix.ptr<double>(0)[2];
    double h4 = HMatrix.ptr<double>(1)[0];
    double h5 = HMatrix.ptr<double>(1)[1];
    double h6 = HMatrix.ptr<double>(1)[2];
    double h7 = HMatrix.ptr<double>(2)[0];
    double h8 = HMatrix.ptr<double>(2)[1];
    double h9 = HMatrix.ptr<double>(2)[2];

    // Calculate optical flow residuals
    // cv::Mat imgGrayWrapedByHMatrix;
    // warpPerspective(imgGrayT, imgGrayWrapedByHMatrix,HMatrix,imgGrayT.size());
    // begin1 = cv::getTickCount();
    // omp_set_num_threads(8);
    // #pragma omp parallel for 
    for(int row = 0; row < height; ++row)
    {
        for(int col = 0; col < width; ++col)
        {
            // cv::Mat ptNext = (cv::Mat_<double>(3,1) << col, row, 1);
            // cv::Mat ptPrvs = (HMatrix * ptNext);
            // float flowX = (float)(col - ptPrvs.ptr<double>(0)[0]/ptPrvs.ptr<double>(2)[0]);
            // float flowY = (float)(row - ptPrvs.ptr<double>(1)[0]/ptPrvs.ptr<double>(2)[0]);
            double flowX2 = (col - (h1*col + h2*row + h3)/(h7*col + h8*row + h9));
            double flowY2 = (row - (h4*col + h5*row + h6)/(h7*col + h8*row + h9));
            imgDenseFlowDiff.ptr<cv::Vec2f>(row)[col][0] -= (float)flowX2;
            imgDenseFlowDiff.ptr<cv::Vec2f>(row)[col][1] -= (float)flowY2;
            // std::cout <<  flowX - flowX2 << "," << 
            // flowY - flowY2 << std::endl;
        }
    }
    Mat imgFlowParts[2];
    split(imgDenseFlowDiff, imgFlowParts);
    Mat imgMagnitude, imgAngle;
    cartToPolar(imgFlowParts[0], imgFlowParts[1], imgMagnitude, imgAngle, true);
    // end1 = cv::getTickCount();
    // std::cout << "Calculate optical flow residuals:" << ((end1 - begin1)/ cv::getTickFrequency()) << std::endl;
    // auto time1 = cv::getTickCount();
    //Use Otsu and the triangle method to find the thresholds. Plot the histogram.
    cv::Mat imgThhd1, imgThhd2, imgThhd3, imgMagnitudeNomolized;
    double maxError;
    
    cv::minMaxLoc(imgMagnitude, 0, &maxError);
    float maxErrorf = (float)maxError;
    imgMagnitudeNomolized = imgMagnitude * (255.0/maxError);
    imgMagnitudeNomolized.convertTo(imgMagnitudeNomolized, CV_8UC1);
    
    float thred1 = (float)cv::threshold(imgMagnitudeNomolized, imgThhd1, 80, 255, cv::ThresholdTypes::THRESH_OTSU);
    float thred2 = (float)cv::threshold(imgMagnitudeNomolized, imgThhd2, 80, 255, cv::ThresholdTypes::THRESH_TRIANGLE);
    // cv::adaptiveThreshold(imgMagnitudeNomolized, imgThhd3, 255, cv::AdaptiveThresholdTypes::ADAPTIVE_THRESH_GAUSSIAN_C, cv::ThresholdTypes::THRESH_BINARY, 5, 0.01);
    std::cout << "thred1 : " << thred1<< ", thred2 : " << thred2 << std::endl;
    // cv::Mat imgHist = getHistogramImage(imgMagnitudeNomolized);
    //--------------------for draw image------------------
    #ifdef IMGSAVE 
        cv::Mat imgDenseFlowRGB;
        OpticalFlowVisualization(imgDenseFlow, imgDenseFlowRGB);
        cv::imwrite(save_path + "imgOpticalFlowError.png", imgMagnitudeNomolized);
        cv::imwrite(save_path + "denseflow/" + std::to_string(nImg) + ".png", imgDenseFlowRGB);
        cv::imwrite(save_path + "imgHist.png", imgHist);
    #endif
    //--------------------for draw image------------------
    // begin1 = cv::getTickCount();

    // float thred_loww = min(thred1, thred2);
    // float thred_high = max(thred1, thred2);
    // thred_loww = std::clamp(thred_loww, 1.7f * 255.0f / maxErrorf, 3.0f * 255.0f / maxErrorf);
    // thred_high = std::clamp(thred_high, std::max(3.0f * 255.0f / maxErrorf, thred1 * 1.2f), 10.0f * 255.0f / maxErrorf);
    // imgThhd1 = (imgMagnitudeNomolized > thred_loww);
    // imgThhd2 = (imgMagnitudeNomolized > thred_high);        
    // stImgMasks imgMaskOut(thred_loww * 0.5, thred_high);
    // imgMasks.set_value(imgMaskOut);

    if(thred1 < thred2)
    {
        if(thred1 < 1.7f * 255.0f / maxErrorf)
        {
            thred1 = 1.7f * 255.0f / maxErrorf;
        }
        else if(thred1 > 3.0f * 255.0f / maxErrorf)
        {
            thred1 = 3.0f * 255.0f / maxErrorf;
        }
        imgThhd1 = (imgMagnitudeNomolized > thred1);
        if(cv::countNonZero(imgThhd1) > 0.5 * width * height)
        {
            thred1 = thred1 + 0.2f * 255.0f / maxErrorf;
            imgThhd1 = (imgMagnitudeNomolized > thred1);
        }                                                        
        if(thred2 < std::max(3.0f * 255.0f / maxErrorf, thred1 * 1.2f) )
        {
            thred2 = std::max(3.0f * 255.0f / maxErrorf, thred1 * 1.2f);
        }
        else if(thred2 > 10.0f * 255.0f / maxErrorf)
        {
            thred2 = 10.0f * 255.0f / maxErrorf;
        }
        imgThhd2 = (imgMagnitudeNomolized > thred2);        
        stImgMasks imgMaskOut(imgThhd1 * 0.5, imgThhd2);
        imgMasks.set_value(imgMaskOut);
    }
    else
    {
        if(thred2 < 1.7f * 255.0f / maxErrorf)
        {
            thred2 = 1.7f * 255.0f / maxErrorf;
        }
        else if(thred2 > 3.0f * 255.0f / maxErrorf)
        {
            thred2 = 3.0f * 255.0f / maxErrorf;
        }
        imgThhd2 = (imgMagnitudeNomolized > thred2);
        if(cv::countNonZero(thred2) > 0.5 * width * height)
        {
            thred2 = thred2 + 0.2f * 255.0f / maxErrorf;
            imgThhd2 = (imgMagnitudeNomolized > thred2);
        }

        if(thred1 < std::max(3.0f * 255.0f / maxErrorf, thred2 * 1.2f))
        {
            thred1 = std::max(3.0f * 255.0f / maxErrorf, thred2 * 1.2f);
        }
        else if(thred1 > 10.0f * 255.0f / maxErrorf)
        {
            thred1 = 10.0f * 255.0f / maxErrorf;
        }
        imgThhd1 = (imgMagnitudeNomolized > thred1);        
        // cv::bitwise_and(imgThhd2, imgDepthValid, imgThhd2);
        // cv::bitwise_or(imgThhd2 * 0.5, imgThhd1, imgMaskTotal2);
        stImgMasks imgMaskOut(imgThhd2 * 0.5 , imgThhd1);
        imgMasks.set_value(imgMaskOut);
    } 
    // end1 = cv::getTickCount();
    // std::cout << "寻找动态区域:" << ((end1 - begin1)/ cv::getTickFrequency()) << std::endl;
    endTotal = cv::getTickCount();
    std::cout << ">>>>>>>>>>>>DenseFlow + Refinement + DynamicDection:" << ((endTotal - beginTotal)/ cv::getTickFrequency()) << std::endl;
    timecost6 += ((endTotal - beginTotal)/ cv::getTickFrequency());
    std::cout << timecost5/(double)nImg << ", " << timecost6/(double)nImg << std::endl;
}


void ORB_SLAM2::DynaDetect::DetectDynaArea
                       ( const cv::InputArray &img_,
						 const cv::InputArray &imgDepth_,
						 cv::OutputArray &imgDynaOut,
						 cv::OutputArray &imgLabelOut,
				         int nImg_)
{
    static double timecost1, timecost2, timecost3, timecost4;
    //这里是读取数据，然后把数据都准备好
    img_.getMat().copyTo(imgRGB);
    imgDepth_.getMat().copyTo(imgDepth);
    nImg = nImg_;

    cv::cvtColor(imgRGB, imgGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRGBLast, imgGrayLast, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imgRGBLastLast, imgGrayLastLast, cv::COLOR_BGR2GRAY);
    // imgDyna.copyTo(imgDynaLast);
    imgDyna.setTo(0);
    // --------------Using Multithreading---------------------
    std::promise<stImgMasks> pImgMasks;
    std::future<stImgMasks> fImgMasks = pImgMasks.get_future();
    std::thread threadDenseOptFlow = std::thread([&] { this->DetectDynaByDenseOpticalFLow(std::ref(pImgMasks)); });

    double begin_time, end_time;
    double beginTotal, endTotal;

    //--------------------------------K-Means----------------------------------------//
    //centers_pyramid records the centroids of all clusters, 
    //points_pyramid records the 3D coordinates of all points.
    Mat centers_pyramid, points_pyramid;
    begin_time = cv::getTickCount();
    beginTotal = cv::getTickCount();
    // Mat imgLabel;
    SegByKmeans(imgLabel, points_pyramid, centers_pyramid);
    if(imgLabel.type() != CV_8UC1)
    {
        imgLabel.convertTo(imgLabel, CV_8UC1);
    }
    //make imgLabel to be colorful
    //--------------------for draw image------------------
    // Mat imgLabelColor1(imgSize, CV_8UC3, cv::Scalar(0, 0, 0));
    // getLabelColored(imgLabel, imgLabelColor1, numCluster);
    //--------------------for draw image------------------
    end_time = cv::getTickCount();
    std::cout << "K-means timecost:" << ((end_time - begin_time)/ cv::getTickFrequency()) << std::endl;
    timecost1 += ((end_time - begin_time)/ cv::getTickFrequency());
    /* "Occluded" means "edge" */
    //------------------------------------提取Occluded-------------------------------------------//
    Mat sortDepth;
    //Sort the clusters based on the z-value of their centroids. 
    //We prioritize considering clusters with smaller z-values (closer).
    Mat depth_vals = centers_pyramid.colRange(2, 3);
    for(int i = 0; i < depth_vals.rows; i++)
    {
        //Set the clusters with z=0 directly to the last cluster, 
        //which is generally not considered or processed.
        if(depth_vals.at<float>(i, 0) < 0.2)
        {
            depth_vals.at<float>(i, 0) += 20.0f;
        }
    }
    sortIdx(depth_vals, sortDepth, cv::SORT_EVERY_COLUMN + cv::SORT_ASCENDING);
    //-------------这一部分是给imgLabel排个序，0深度的为黑色，为了论文里面展示------------
    //--------------------for draw image------------------
    // Mat imgLabelSort(imgSize, CV_8UC1, cv::Scalar(0));
    // Mat imgLabelColorSort1(imgSize, CV_8UC3, cv::Scalar(0, 0, 0));
    // int countKKK = 0;
    // for(int i = numCluster - 1; i >= 0; i--)
    // {
    //     int label = sortDepth.ptr<int>(i)[0];
    //     if(depth_vals.ptr<float>(label)[0] == 20.0f)
    //     {
    //         countKKK = 0;
    //     }
    //     else
    //     {
    //         countKKK ++;
    //     }
    //     imgLabelSort.setTo(countKKK, (imgLabel == label));
    // }
    // getLabelColored(imgLabelSort, imgLabelColorSort1, numCluster);
    // cv::imwrite(save_path + "imgLabelInitial.png", imgLabelColorSort1);
    // //-------------结束排序------------------------
    // //--------------------for draw image------------------

    //The regions occupied by the first n clusters.
    Mat imgLabelForSegEdge(imgSize, CV_8UC1, cv::Scalar(0));
    //The region where the depth is in (0, 6)m.
    Mat imgTotalArea(imgSize, CV_8UC1, cv::Scalar(0));
    //Save each cluster
    std::vector<Mat> allLabels;
    float ratioArea = 0.0f;
    const float TotalArea = height * width;
    int count0 = 0;//
    int NoValidCount = 0;
    for(int i = 0; i < numCluster; i++)
    {
        int index_depth = sortDepth.at<int>(i, 0);
        Mat eachLabel = (imgLabel == index_depth);
        if(countNonZero(eachLabel) < 60)
        {
            ++NoValidCount;
            continue;
        }
        allLabels.emplace_back(eachLabel);
        float ratio = (float)cv::countNonZero(eachLabel) * (1.0f/TotalArea);
        ratioArea += ratio;
        //Consider at most the first 5 clusters, and the total area of the regions is less than 60%.
        if(count0 <= 5 && ratioArea < 0.6f)
        {
            bitwise_or(eachLabel, imgLabelForSegEdge, imgLabelForSegEdge);
            ++count0;
        }
    }
    cv::morphologyEx(imgLabelForSegEdge, imgLabelForSegEdge, cv::MORPH_DILATE, element7); 

    begin_time = cv::getTickCount();
    //imgOccluded2 is "plane edge", imgOccluded1 is "plane edge + gradient Edge".
    Mat imgOccluded2(imgSize, CV_8UC1, cv::Scalar(0));
    Mat imgOccluded1(imgSize, CV_8UC1, cv::Scalar(0));
    CalOccluded(imgLabelForSegEdge, imgTotalArea, imgOccluded1, imgOccluded2);
    end_time = cv::getTickCount();
    std::cout << "Calculate DepthEdge:" << ((end_time - begin_time)/ cv::getTickFrequency()) << std::endl;
    timecost2 += ((end_time - begin_time)/ cv::getTickFrequency());
    #ifdef IMGSHOW 
        // cv::imshow("imgOccluded1", imgOccluded1);
    #endif	
    #ifdef IMGSAVE 	
        cv::imwrite(save_path + "edge/" + std::to_string(nImg) + ".png" , imgOccluded1);	
    #endif
    
    //--------------------------SegAndMerge--------------------------
    Mat imgLabel3(imgSize, CV_8UC1, cv::Scalar(0)); 

    begin_time = cv::getTickCount();
    // bitwise_and(imgLabelForSegEdge, imgOccluded1, imgOccluded1);
    // bitwise_and(imgLabelForSegEdge, imgOccluded2, imgOccluded2);  
    SegAndMergeV2(allLabels, imgOccluded1, imgOccluded2, imgLabelForSegEdge, points_pyramid, imgLabel3, count0);
    //imgLabel is output!!!
    imgLabel3.copyTo(imgLabel);
    end_time = cv::getTickCount();
    std::cout << "SegAndMergeV2 timecost:" << ((end_time - begin_time)/ cv::getTickFrequency()) << std::endl;
    timecost3 += ((end_time - begin_time)/ cv::getTickFrequency());
    double maxNumCluster;
    cv::minMaxLoc(imgLabel3, 0, &maxNumCluster);
    int maxNumClusteri = (int)maxNumCluster;    
    vector<Mat> EveryLabel;
    for(int i = 0; i <= maxNumClusteri; i++)
    {
        Mat eachLabel = (imgLabel == i);
        EveryLabel.push_back(eachLabel);
    }
    //--------------------for draw image------------------
    //make imgLabel to be colorful
    Mat imgLabelColor3(imgSize, CV_8UC3, cv::Scalar(0, 0, 0));
    getLabelColored(imgLabel, imgLabelColor3, numCluster);
    // // cv::putText(imgLabelColor3, std::to_string(nImg), cv::Point(50, 50), 2, 2.0, cv::Scalar(0, 0, 0));	
    #ifdef IMGSAVE 		
        cv::imwrite(save_path + "Seg/" + std::to_string(nImg) + ".png" , imgLabelColor3);	
    #endif
    #ifdef IMGSHOW
        // imshow("imgLabelColor3", imgLabelColor3);
    #endif
    //--------------------for draw image------------------

    //--------------------------------动态区域检测---------------------------------------//
    begin_time = cv::getTickCount();
    // DetectDynaByOpticalFLow(imgTotalArea, imgLabel2, EveryLabel);

    // --------------NO Using Multithreading---------------------
    // cv::Mat imgMaskLowError = cv::Mat(imgGray.size(), CV_8UC1, cv::Scalar(0));
    // cv::Mat imgMaskHighError = cv::Mat(imgGray.size(), CV_8UC1, cv::Scalar(0));
    // cv::Mat imgMaskTotal = cv::Mat(imgGray.size(), CV_8UC1, cv::Scalar(0));
    // DetectDynaByDenseOpticalFLow(imgMaskLowError, imgMaskHighError, imgMaskTotal);

    // --------------Using Multithreading---------------------
    auto imgMasks = fImgMasks.get();
    threadDenseOptFlow.join();
    cv::Mat imgMaskLowError = imgMasks.imgMaskLowError.clone();
    cv::Mat imgMaskHighError = imgMasks.imgMaskHighError.clone();
    cv::Mat imgMaskTotal;
    // --------------Using Multithreading---------------------

    cv::bitwise_or(imgMaskHighErrorLast, imgMaskLowError, imgMaskLowError);
    imgMaskLowError.setTo(128, imgMaskLowError);
    cv::bitwise_or(imgMaskLowError, imgMaskHighError, imgMaskTotal);
    cv::bitwise_and(imgMaskLowError, imgTotalArea, imgMaskLowError);
    //merge the images from dynamic detection and segmentation
    cv::morphologyEx(imgMaskLowError, imgMaskLowError, cv::MORPH_DILATE, element5); 
    for(int n = 1; n <= maxNumClusteri; n++)
    {
        cv::Mat oneCluster = EveryLabel[n];
        cv::Mat oneClusterWithBorder;
        cv::copyMakeBorder(oneCluster, oneClusterWithBorder,1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
        cv::bitwise_not(oneClusterWithBorder, oneClusterWithBorder);
        cv::Mat oneClusterWithHighError;
        cv::bitwise_and(oneCluster, imgMaskHighError, oneClusterWithHighError);
        int totalArea =  cv::countNonZero(oneClusterWithHighError);
        if(totalArea > 100)
        {
            std::vector<std::vector<cv::Point>> contours2;
            std::vector<cv::Vec4i> hierarchy2;
            cv::findContours(oneClusterWithHighError, contours2, hierarchy2, cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE );
            cv::Mat imgContours(imgSize, CV_8UC1, cv::Scalar(0));
            for(int i = 0; i < contours2.size(); i ++)
            {
                // floodfill from low dyanmic region
                double area = cv::contourArea(contours2[i]);
                double len = cv::arcLength(contours2[i], true);
                double roundness = (4 * CV_PI * area) / (len * len);
                float centerX = 0.0f, centerY = 0.0f;
                cv::Point2f seedPoint;
                for(int j = 0; j < contours2[i].size(); j++)
                {
                    if(imgMaskLowError.ptr<uchar>(contours2[i][j].y)[contours2[i][j].x] == 128)
                    {
                        seedPoint.x = contours2[i][j].x;
                        seedPoint.y = contours2[i][j].y;
                        break;
                    }
                    // centerX += (float)contours2[i][j].x;
                    // centerY += (float)contours2[i][j].y;
                }
                // centerX /= contours2[i].size();
                // centerY /= contours2[i].size();
                cv::drawContours(imgContours, contours2, i, cv::Scalar(255), cv::FILLED);
                if(((area > 100.0 && roundness > 0.2) || area > 2000.0) )
                {
                    cv::floodFill(imgMaskLowError, oneClusterWithBorder, seedPoint, 50, 0, 5, 5, 8|cv::FLOODFILL_MASK_ONLY| ( 50 << 8 ) );
                    int bb = 0;
                }
            }
        }
        cv::Mat oneClusterWithNoBorder = oneClusterWithBorder(cv::Rect(1,1, width, height)).clone();
        cv::compare(oneClusterWithNoBorder, (cv::Mat::ones(imgLabel .size(), imgLabel.type()) * 50), oneClusterWithNoBorder, cv::CMP_EQ);
        if(cv::countNonZero(oneClusterWithNoBorder) > 0.5 * (cv::countNonZero(oneCluster)))
        {
            cv::bitwise_or(imgDyna, oneCluster, imgDyna);
        }
        else
        {
            cv::bitwise_or(imgDyna, oneClusterWithNoBorder, imgDyna);
        }
        int vv = 0;
    }
    cv::morphologyEx(imgDyna, imgDyna, cv::MORPH_DILATE, element9);
    // cv::bitwise_or(imgDyna, imgMaskHighError, imgDyna2);
    // imgDyna.setTo(255, imgMaskHighError);
    //--------------------for draw image------------------
    // Mat imgRGBWithDyna, imgRGBWithDyna2;
    // imgRGB.convertTo(imgRGBWithDyna, cv::COLOR_GRAY2RGB);
    // imgRGB.convertTo(imgRGBWithDyna2, cv::COLOR_GRAY2RGB);
    // imgRGBWithDyna2.setTo(cv::Scalar(0, 0, 255), imgDyna);
    // cv::addWeighted(imgRGBWithDyna, 0.5, imgRGBWithDyna2, 0.5, 0, imgRGBWithDyna);
    //--------------------for draw image------------------

    Mat imgStatic2 = (imgTotalArea - imgDyna) * 125/255;
    imgDyna += imgStatic2;
    imgDyna.copyTo(imgDynaOut);	
    imgLabel.copyTo(imgLabelOut);
    #ifdef IMGSHOW
        // imshow("imgMaskTotal", imgMaskTotal);    
    #endif
    imshow("Gray", imgGray);
    imshow("imgDyna", imgDyna);     
    end_time = cv::getTickCount();
    std::cout << "Dynamic detection timecost:" << ((end_time - begin_time)/ cv::getTickFrequency()) << std::endl;
    timecost4 += ((end_time - begin_time)/ cv::getTickFrequency());
    endTotal = cv::getTickCount();
    std::cout << "Total timecost:" << ((endTotal - beginTotal)/ cv::getTickFrequency()) << std::endl;
    std::cout << timecost1/(double)nImg << ", " << timecost2/(double)nImg << ", " << timecost3/(double)nImg << ", "
              << timecost4/(double)nImg << ", " << std::endl;
    
    #ifdef IMGSAVE 
        cv::imwrite(save_path + "dynaMask/" + std::to_string(nImg) + ".png" , imgDyna);	
        cv::imwrite(save_path + "errorMask/" + std::to_string(nImg) + ".png" , imgMaskTotal);	
        // cv::imwrite(save_path + "dynaMaskWithRGB/" + std::to_string(nImg) + ".png" , imgRGBWithDyna);
        // cv::imwrite(save_path + "rgb/" + to_string(nImg) + ".png", imgRGB);
        // cv::imwrite(save_path + "depth/" + to_string(nImg) + ".png", imgDepth);
    #endif

    //save imgs to next frame
    imgDyna.copyTo(imgDynaLast);
    imgRGBLast.copyTo(imgRGBLastLast);
    imgRGB.copyTo(imgRGBLast);	
    imgMaskHighError.copyTo(imgMaskHighErrorLast);
    imgLabel.copyTo(imgLabelLast);
    cv::waitKey(1);
}

//--------------------------utility function---------------------------------------------
vector<cv::Scalar> get_color(int n){
	vector<cv::Scalar> colors;
	int k = n/7 + 1;
    colors.push_back(cv::Scalar(0,0,0));
	for(int i = 0; i < k; i++)
	{
		colors.push_back(cv::Scalar(0,0,255)/(i+1.0));
		colors.push_back(cv::Scalar(0,255,0)/(i+1.0));
		colors.push_back(cv::Scalar(255,0,0)/(i+1.0));
		colors.push_back(cv::Scalar(0,255,255)/(i+1.0));
		colors.push_back(cv::Scalar(255,0,255)/(i+1.0));
		colors.push_back(cv::Scalar(255,255,0)/(i+1.0));
		colors.push_back(cv::Scalar(255,255,255)/(i+1.0));
	}
	return colors;
}
vector<double> cal_hist(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &imgDepth, cv::Mat &hist_img, const cv::Mat &Mask){
    //目前这个hist是用来画深度图的。
    // double  maxVal1, maxVal2, maxVal;
    // cv::minMaxLoc(imgDepth, NULL, &maxVal1, NULL, NULL, img1);
    // cv::minMaxLoc(imgDepth, NULL, &maxVal2, NULL, NULL, img2);
	// maxVal = max(maxVal1, maxVal2);
    const int bins[1] = { 256 };
    float hranges[2] = { 0,255 };
    const float* ranges[1] = { hranges };
    Mat hist1, hist2;
    calcHist(&imgDepth, 1, 0, img1, hist1, 1, bins, ranges);
	calcHist(&imgDepth, 1, 0, img2, hist2, 1, bins, ranges);
    // double total = sum(hist)[0];
    int hist_w = 256 * 2;
    int hist_h = 400;
    int bin_w = cvRound((float)hist_w / bins[0]);  //两个灰度级之间的距离
    hist_img = Mat::zeros(hist_h, hist_w, CV_8UC3); //直方图画布
    // 归一化直方图数据
	// double count1 = countNonZero(img1);
	// double count2 = countNonZero(img2);
	double maxCount, maxCount1, maxCount2;
	cv::minMaxLoc(hist1, NULL, &maxCount1, NULL, NULL);
    cv::minMaxLoc(hist2, NULL, &maxCount2, NULL, NULL);
	maxCount = std::max(maxCount1, maxCount2);
	if(maxCount1 > maxCount2)
	{
		normalize(hist1, hist1, 0, hist_h, cv::NORM_MINMAX, -1, Mat()); //将数据规一化到0和hist_h之间
		hist2 /= (maxCount1/hist_img.rows);
	}
	else
	{
		normalize(hist2, hist2, 0, hist_h, cv::NORM_MINMAX, -1, Mat()); //将数据规一化到0和hist_h之间
		hist1 /= (maxCount2/hist_img.rows);
	}
		// normalize(hist2, hist2, 0, hist_img.rows, NORM_MINMAX, -1, Mat()); //将数据规一化到0和hist_h之间
		// normalize(hist1, hist1, 0, hist_img.rows, NORM_MINMAX, -1, Mat()); //将数据规一化到0和hist_h之间
    // 绘制直方图曲线
    // for (int i = 1; i < bins[0]; i++) {
    //     line(hist_img, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist1.at<float>(i - 1))),
    //         cv::Point(bin_w*(i), hist_h - cvRound(hist1.at<float>(i))), cv::Scalar(0, 255, 0), 2, 8, 0);
	// 	line(hist_img, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist2.at<float>(i - 1))),
	// 		cv::Point(bin_w*(i), hist_h - cvRound(hist2.at<float>(i))), cv::Scalar(255, 0, 0), 2, 8, 0);
    // }
	double hist_hist1 = compareHist(hist1, hist2, cv::HISTCMP_CORREL);
	double hist_hist2 = 1 - compareHist(hist1, hist2, cv::HISTCMP_BHATTACHARYYA);
	double hist_hist3 = compareHist(hist1, hist2, cv::HISTCMP_INTERSECT);
	double hist_hist4 = compareHist(hist1, hist2, cv::HISTCMP_CHISQR);

	// cout << hist_hist1 << "," << hist_hist2  << "," << hist_hist3 << "," <<  hist_hist4 << endl;
	vector<double> result;
	result.push_back(hist_hist1);
	result.push_back(hist_hist2);
    result.push_back(hist_hist3);
	return result;
}



}//namespace ORB_SLAM

