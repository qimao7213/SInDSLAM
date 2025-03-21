
#include <string>
#include <vector>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <octomap/octomap.h>    // for octomap 
#include <octomap/ColorOcTree.h>
#include <octomap/math/Pose6D.h>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace sensor_msgs;
using namespace message_filters;
typedef sync_policies::ApproximateTime<Image, Image, Image, Image, geometry_msgs::PoseStamped> ApproximateSyncPolicy;
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> myPointCloud;
// typedef pcl::PCLPointCloud2 myPointCloud;

int indexKF = 0;

Eigen::Vector3d translate(const Eigen::Vector2d &px_uv, 
                          const Eigen::Matrix4d &T_i_j, 
                          const Eigen::Matrix3d &K)
{//uv按照一个Point构造，使用齐次坐标
    Eigen::Matrix3d RotMatirx;
    Eigen::Vector3d translation;
    RotMatirx << T_i_j(0, 0) , T_i_j(0, 1) , T_i_j(0, 2),
                T_i_j(1, 0) , T_i_j(1, 1) , T_i_j(1, 2),
                T_i_j(2, 0) , T_i_j(2, 1) , T_i_j(2, 2);
    translation << T_i_j(0, 3) , T_i_j(1, 3) , T_i_j(2, 3);
    Eigen::Vector3d pt = K.inverse() * Eigen::Vector3d(px_uv[0], px_uv[1], 1);
    return  K *(RotMatirx* pt + translation); 
}

void useStatisticalOutlierFilter(myPointCloud::Ptr &cloud)
{
    // myPointCloud::Ptr tmp =  pcl::make_shared<myPointCloud>();// point cloud is null ptr
    pcl::StatisticalOutlierRemoval<PointT> sor;
    sor.setInputCloud (cloud);  //设置输入
    sor.setMeanK (50);  //设置用于平均距离估计的 KD-tree最近邻搜索点的个数.
    sor.setStddevMulThresh (1.0); //高斯分布标准差的倍数, 也就是 u+1*sigma,u+2*sigma,u+3*sigma 中的 倍数1、2、3 
    sor.filter (*cloud);
    // tmp.swap(cloud);

}

class SubscribeAndPublish
{
private:
    ros::NodeHandle nh; 
    image_transport::ImageTransport it;

    image_transport::SubscriberFilter *imgSubRGB, *imgSubDepth, *imgSubDynaMask, *imgSubLabel;
    message_filters::Subscriber<geometry_msgs::PoseStamped> *poseSub;
    message_filters::Synchronizer<ApproximateSyncPolicy> *syncApproximate;

    ros::Publisher pcl_pub;
    ros::Publisher pcl_pub2;
    ros::Publisher octomap_pub;

    myPointCloud::Ptr globalPointCloudMap = pcl::make_shared<myPointCloud>();
    octomap::ColorOcTree* octotree = new octomap::ColorOcTree(0.020);
    pcl::PCDWriter pclWriter;
    myPointCloud::Ptr tempCloudOneFrame = pcl::make_shared<myPointCloud>();
    myPointCloud::Ptr temp1 = pcl::make_shared<myPointCloud>();
    pcl::StatisticalOutlierRemoval<PointT> sor;


public:
    SubscribeAndPublish(): nh("~"), it(nh)
    {

        std::string s1;
        std::string camera_file;
        nh.param("SInDSLAM/camera_file", camera_file, s1);
        nh.param("SInDSLAM/pt_output_file", pt_output_file, s1);
                cv::FileStorage fSettings(std::string(camera_file), cv::FileStorage::READ);
        if(!fSettings.isOpened())
        {
           cerr << "Failed to open settings file at: " << std::string(camera_file) << endl;
           exit(-1);
        }
        fx = (double)fSettings["Camera.fx"];
        fy = (double)fSettings["Camera.fy"];
        cx = (double)fSettings["Camera.cx"];
        cy = (double)fSettings["Camera.cy"];
        depthScale = (double)fSettings["DepthMapFactor"];

        K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
        octotree->setOccupancyThres (0.7);
        image_transport::TransportHints hints(0 ? "compressed" : "raw");
        imgSubRGB = new image_transport::SubscriberFilter(it, "/orbslam2/imgRGB", 500, hints);
        imgSubDepth = new image_transport::SubscriberFilter(it, "/orbslam2/imgDepth", 500, hints);
        imgSubDynaMask = new image_transport::SubscriberFilter(it, "/orbslam2/imgDynaMask", 500, hints);
        imgSubLabel = new image_transport::SubscriberFilter(it, "/orbslam2/imgLabel", 500, hints); 
        poseSub = new Subscriber<geometry_msgs::PoseStamped>(nh, "/orbslam2/poseKF", 500);

        pcl_pub = nh.advertise<sensor_msgs::PointCloud2> ("/orb_slam2/pointcloud", 10); 
        pcl_pub2 = nh.advertise<sensor_msgs::PointCloud2> ("/orb_slam2/pointcloud_by_octomap", 10); 
        octomap_pub = nh.advertise<octomap_msgs::Octomap>("/orb_slam2/octomap_full", 10, true);

        ROS_INFO("prepare to subscribe the topics ....");
        // TimeSynchronizer<Image, Image, Image, geometry_msgs::PoseStamped> sync(imgSubRGB, imgSubDepth, imgSubDynaMask, poseSub, 20);
        syncApproximate = new Synchronizer<ApproximateSyncPolicy>(ApproximateSyncPolicy(10), *imgSubRGB, *imgSubDepth, *imgSubDynaMask, *imgSubLabel, *poseSub);
        syncApproximate->registerCallback(boost::bind(&SubscribeAndPublish::Callback, this, _1, _2, _3, _4, _5));
    }

    ~SubscribeAndPublish(){
        // sensor_msgs::PointCloud2 outputPointCloudbyOctomap;
        // myPointCloud::Ptr octomapPointCloud = pcl::make_shared<myPointCloud>();
        // int kkk = octotree->size();
        // // int kk = octomapPointCloud->points.size();
        // for(auto iter = octotree->begin_leafs(); iter != octotree->end_leafs(); ++iter)
        // {
        //     auto key = iter.getIndexKey();
        //     auto node = octotree->search(key);
        //     if (octotree->isNodeOccupied(node))            
        //     {
        //         PointT p;
        //         p.x = iter.getX();
        //         p.y = iter.getY();
        //         p.z = iter.getZ();
        //         auto color = node->getColor();
        //         p.b = color.b;
        //         p.g = color.g;
        //         p.r = color.r;
        //         octomapPointCloud->points.push_back(p);
        //     }
        // }
        // if(octotree->size() > 0)
        // {
        //     int kk = octomapPointCloud->points.size();
        //     octomapPointCloud->height = octomapPointCloud->points.size();
        //     octomapPointCloud->width = 1;
        //     pcl::toROSMsg(*octomapPointCloud, outputPointCloudbyOctomap);
        //     outputPointCloudbyOctomap.header.stamp = ros::Time::now();
        //     outputPointCloudbyOctomap.header.frame_id = "world";
        //     for(int i = 0; i < 100; i++)
        //     {
        //         pcl_pub2.publish(outputPointCloudbyOctomap);  
        //         ROS_INFO("Octomap transfer to pointcloud done!");
        //     }

        //     // t2 = std::chrono::steady_clock::now();
        //     // cout << "transfer octomap cost time: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count()*1000 << " ms ." <<endl;
        // }
        delete syncApproximate;
        delete imgSubDepth;
        delete imgSubDynaMask;
        delete imgSubLabel;
        delete imgSubRGB;
        delete poseSub;
        std::cout << "octotree size is: " << octotree-> size() << std::endl;        
        std::cout << "pointcloud size is: " << globalPointCloudMap-> size() << std::endl;   
        std::cout << "octo_map saving done? " << octotree->write( pt_output_file + "octo1.ot" ) << std::endl;
        std::cout << "pointcloud saving done? " << pclWriter.write( pt_output_file + "pointcloud.pcd", *globalPointCloudMap ) << std::endl;
        usleep(5 * 1e6);
        delete octotree;
    }

private:
    void Callback(const ImageConstPtr& imgRGBMsg,
                const ImageConstPtr& imgDepthMsg, 
                const ImageConstPtr& imgDynaMaskMsg, 
                const ImageConstPtr& imgLabelMsg, 
                const geometry_msgs::PoseStamped::ConstPtr& poseCameraMsg
                )           
    {
        indexKF ++;
        ROS_INFO("---------------- KF index is %i -----------------", indexKF);
        cv::Mat imgRGB, imgDepth, imgDynaMask, imgLabel;

        cv_bridge::CvImageConstPtr pCvImage;

        pCvImage = cv_bridge::toCvShare(imgRGBMsg, "bgr8");
        pCvImage->image.copyTo(imgRGB);
        pCvImage = cv_bridge::toCvShare(imgDepthMsg, imgDepthMsg->encoding);
        pCvImage->image.copyTo(imgDepth);
        pCvImage = cv_bridge::toCvShare(imgDynaMaskMsg, "mono8");
        pCvImage->image.copyTo(imgDynaMask);
        pCvImage = cv_bridge::toCvShare(imgLabelMsg, "mono8");
        pCvImage->image.copyTo(imgLabel); 

        // ROS_INFO("The type of imgRGB is %i", imgRGB.type());
        // ROS_INFO("The type of imgDepth is %i", imgDepth.type());
        // ROS_INFO("The type of imgDynaMask is %i", imgDynaMask.type());
        // ROS_INFO("The type of imgLabel is %i", imgLabel.type());

        //从camera到world
        //这里赋值的问题，太坑了
        Eigen::Quaterniond q(poseCameraMsg->pose.orientation.w, 
                            poseCameraMsg->pose.orientation.x, 
                            poseCameraMsg->pose.orientation.y, 
                            poseCameraMsg->pose.orientation.z);

        Eigen::Vector3d t(poseCameraMsg->pose.position.x, 
                        poseCameraMsg->pose.position.y, 
                        poseCameraMsg->pose.position.z);
        Eigen::Isometry3d Twc(q);
        Twc.pretranslate(t); 
        // std::cout << imgDepthMsg->header.stamp << std::endl;
        // std::cout << poseCameraMsg->header.stamp << std::endl;
        // std::cout << Twc.matrix() << std::endl;
        // std::cout << poseCameraMsg->pose.orientation << ", " << poseCameraMsg->pose.position << std::endl;
        if(imgDynaMask.empty())
        {
            imgDynaMask = cv::Mat::zeros(imgDepth.size(), CV_8UC1);
        }
        ROS_INFO("now start inserting image...");
        vecImgdepth.push_back(imgDepth);
        vecImgDynaMask.push_back(imgDynaMask);
        vecPose.push_back(Twc);

        ROS_INFO("Subscribe done! now start processing...");

        //---------------数据已经订阅好了，下面开始处理点云------------------
         
        
        myPointCloud::Ptr tem_cloud2(new myPointCloud());
        myPointCloud::Ptr tem_cloud3(new myPointCloud());//用于离群值滤波
        ROS_INFO("Now start generating PointCloud...");
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        if(vecImgdepth.size() != vecPose.size())
        {
            ROS_ERROR("-------------vecImgdepth.size() != vecPose.size()------------------");
        }
        tempCloudOneFrame->clear();

        int sizeKF = vecImgdepth.size();
        Eigen::Matrix4d poseRelative;//Last到Curr的
        Eigen::Isometry3d poseCurr = vecPose[sizeKF - 1];
        // generatePointCloud(imgRGB, imgDepth, imgDynaMask, Twc); 
        if(sizeKF == 1 && !imgDynaMask.empty())
        {
            // generatePointCloud(imgRGB, imgDepth, imgDynaMask, Twc); 
        }
        else if(sizeKF == 2 && !imgDynaMask.empty() && !imgLabel.empty())
        {
            Eigen::Isometry3d poseLast = vecPose[sizeKF - 2];
            poseRelative = poseCurr.matrix() * poseLast.inverse().matrix();
            generatePointCloud(imgRGB, vecImgdepth[sizeKF - 1], vecImgdepth[sizeKF - 2], 
                              vecImgDynaMask[sizeKF - 1], vecImgDynaMask[sizeKF - 2], imgLabel, poseRelative, Twc);
            // vecImgdepth.erase(vecImgdepth.begin());
            // vecPose.erase(vecPose.begin());
        }
        else if(sizeKF >= 3 && !imgDynaMask.empty() && !imgLabel.empty())
        {
            Eigen::Isometry3d poseLast = vecPose[sizeKF - 3];
            poseRelative = poseCurr.matrix() * poseLast.inverse().matrix();
            generatePointCloud(imgRGB, vecImgdepth[sizeKF - 1], vecImgdepth[sizeKF - 3], 
                               vecImgDynaMask[sizeKF - 1], vecImgDynaMask[sizeKF - 3], imgLabel, poseRelative, Twc);
            vecImgdepth.erase(vecImgdepth.begin());
            vecImgDynaMask.erase(vecImgDynaMask.begin());
            vecPose.erase(vecPose.begin());
        }
        else
        {
            ROS_ERROR("the number of KF is wrong!");
        }
        //因为放到函数里面，在释放指针什么的时候会报错，所以把这个滤波函数放到callback函数里面，就不会存在释放的问题吧
        //但是依然有错
        sor.setInputCloud (tempCloudOneFrame);  //设置输入
        sor.setMeanK (100);  //设置用于平均距离估计的 KD-tree最近邻搜索点的个数.
        sor.setStddevMulThresh (1.0); //高斯分布标准差的倍数, 也就是 u+1*sigma,u+2*sigma,u+3*sigma 中的 倍数1、2、3 
        sor.filter (*temp1);
        // temp1.swap(tempCloudOneFrame);
        *globalPointCloudMap += *temp1;
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        cout << "generate pointcloud cost time: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count()*1000 << " ms ." <<endl;
        //--------------------------处理 八叉树 -----------------------------------
        t1 = std::chrono::steady_clock::now();
        octomap::Pointcloud cloud_octo;
        auto Tcw = Twc.inverse();
        octomap::point3d center_point(Twc(0,3), Twc(1,3), Twc(2,3));
        for (auto p:tempCloudOneFrame->points)
        {
            if(p.z >= 0)
            {
                // cloud_octo.push_back( p.x, p.y, p.z );
                octotree->insertRay(center_point, octomap::point3d(p.x, p.y, p.z));   
            }
        }
        // ROS_INFO("Octomap generating done!");
        // octotree->insertPointCloud( cloud_octo, center_point);
        t2 = std::chrono::steady_clock::now();
        cout << "insert octomap cost time: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count()*1000 << " ms ." <<endl;
        ROS_INFO("Octomap inserting done!");
        for (auto p:tempCloudOneFrame->points)
        {
            octotree->integrateNodeColor( p.x, p.y, p.z, p.r, p.g, p.b );
        }
        // octotree->updateInnerOccupancy();
        ROS_INFO("Octomap updating done!");
        t2 = std::chrono::steady_clock::now();
        cout << "update octomap cost time: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count()*1000 << " ms ." <<endl;

        //----------------发布octomap--------------------
        octomap_msgs::Octomap octoMapMsg;
        octoMapMsg.header.frame_id = "world";
        octoMapMsg.header.stamp = ros::Time::now();
        //fullMapToMsg负责转换成message
        // octoMapMsg
        if (octomap_msgs::fullMapToMsg(*octotree, octoMapMsg))
        {
            //转换成功，可以发布了
            octomap_pub.publish(octoMapMsg);
        } 
        else
        {
            ROS_ERROR("Error serializing OctoMap");
        }
        ROS_INFO("Octomap publishing done!");
        t2 = std::chrono::steady_clock::now();
        cout << "pub octomap cost time: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count()*1000 << " ms ." <<endl;
        //-----------------将octomap转换成pointcloud，再发布-------
        sensor_msgs::PointCloud2 outputPointCloudbyOctomap;
        myPointCloud::Ptr octomapPointCloud = pcl::make_shared<myPointCloud>();
        int kkk = octotree->size();
        // int kk = octomapPointCloud->points.size();
        for(auto iter = octotree->begin_leafs(); iter != octotree->end_leafs(); ++iter)
        {
            auto key = iter.getIndexKey();
            auto node = octotree->search(key);
            if (octotree->isNodeOccupied(node))            
            {
                PointT p;
                p.x = iter.getX();
                p.y = iter.getY();
                p.z = iter.getZ();
                auto color = node->getColor();
                p.b = color.b;
                p.g = color.g;
                p.r = color.r;
                octomapPointCloud->points.push_back(p);
            }
        }
        if(octotree->size() > 0)
        {
            int kk = octomapPointCloud->points.size();
            octomapPointCloud->height = octomapPointCloud->points.size();
            octomapPointCloud->width = 1;
            pcl::toROSMsg(*octomapPointCloud, outputPointCloudbyOctomap);
            outputPointCloudbyOctomap.header.stamp = ros::Time::now();
            outputPointCloudbyOctomap.header.frame_id = "world";
            pcl_pub2.publish(outputPointCloudbyOctomap);  
            ROS_INFO("Octomap transfer to pointcloud done!");
            t2 = std::chrono::steady_clock::now();
            cout << "transfer octomap cost time: "<< std::chrono::duration_cast<std::chrono::duration<double>>( t2 - t1 ).count()*1000 << " ms ." <<endl;
        }


        // //-----------------发布PointCloud---------------------
        temp1->height = temp1->points.size();
        temp1->width = 1;
        sensor_msgs::PointCloud2 outputPointCloud;  
        pcl::toROSMsg(*temp1, outputPointCloud);
        outputPointCloud.header.stamp = ros::Time::now();
        outputPointCloud.header.frame_id = "world";
        pcl_pub.publish(outputPointCloud);  
        ROS_INFO("PointCloud publishing done!");
        std::cout << outputPointCloud.data.size() << " , " << temp1->size() << std:: endl;
    }

    void generatePointCloud(const cv::Mat &imgRGB, 
                        const cv::Mat &imgDepth, 
                        const cv::Mat &imgDynaMask, 
                        const Eigen::Isometry3d &Twc
                        // myPointCloud::Ptr cloud1
                        )
    {
        myPointCloud::Ptr tmp =  pcl::make_shared<myPointCloud>();// point cloud is null ptr
        myPointCloud::Ptr tmp2 =  pcl::make_shared<myPointCloud>();// point cloud is null ptr
        myPointCloud::Ptr tmp3 =  pcl::make_shared<myPointCloud>();// point cloud is null ptr
        // myPointCloud::Ptr tmp2( new myPointCloud() );// point cloud is null ptr
        
        
        int width = imgDepth.cols;
        int height = imgDepth.rows;
        tmp->points.reserve(width * height);
        for ( int m = 0; m < height; m += 3 )
        {
            for ( int n = 0; n < width; n += 3 )
            {
                float d = (float)(imgDepth.ptr<ushort>(m)[n] * (1.0/ depthScale));
                PointT p;
                //可以选择只重建静态区域
                // if((int)dynaMask.ptr<uchar>(m)[n] != 125)
                if((int)imgDynaMask.ptr<uchar>(m)[n] >= 240)
                {
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN ();
                }
                else if (d < 0.01 || d > 10)
                {
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN ();
                }
                else
                {
                    p.z = d;
                    p.x = ( n - (float)cx) * p.z * (float)(1.0/ fx);
                    p.y = ( m - (float)cy) * p.z * (float)(1.0/ fy);
                }
                p.b = imgRGB.ptr<uchar>(m)[n*3];
                p.g = imgRGB.ptr<uchar>(m)[n*3+1];
                p.r = imgRGB.ptr<uchar>(m)[n*3+2];
   
                tmp->points.push_back(p);
            }
        }
        tmp->is_dense = false;
        // cout << tmp->points.size() << endl; 
        tmp->width = tmp->points.size();
        // tmp->height = height;

        pcl::transformPointCloud( *tmp, *tempCloudOneFrame, Twc.matrix());
        // useStatisticalOutlierFilter(tempCloudOneFrame);

        pcl::VoxelGrid<PointT>  voxel; //体素滤波
        // voxel.setInputCloud( tmp );
        // voxel.setLeafSize (0.008f, 0.008f, 0.008f);	
        // voxel.filter( *tmp2 );

        //这个不会报错，但是一直卡住了
        
        pcl::StatisticalOutlierRemoval<PointT> sor;
        // sor.setInputCloud (tmp);  //设置输入
        // sor.setMeanK (50);  //设置用于平均距离估计的 KD-tree最近邻搜索点的个数.
        // sor.setStddevMulThresh (1.0); //高斯分布标准差的倍数, 也就是 u+1*sigma,u+2*sigma,u+3*sigma 中的 倍数1、2、3 
        // sor.filter (*tempCloudOneFrame);

        //这两个函数都是安全的

        // cloud1->swap( *tmp );

        cout << "generate point cloud from  kf-ID:" << indexKF << ", size = " << tempCloudOneFrame->points.size() << endl;
        //都不是空指针
        // cout << "Null pointer? " << (tmp == nullptr) << ", " << (tmp == nullptr) << ", " << (cloud1 == nullptr) << endl;
        // free(tmp);
        // free voxel;
        
        // cout << "点云的宽和高：" << cloud1->width << "," << cloud1->height << endl;
    }
    void generatePointCloud(const cv::Mat& imgRGB, 
                            const cv::Mat& imgDepth, 
                            const cv::Mat& imgDepthLast,  
                            const cv::Mat &imgDynaMask, 
                            const cv::Mat &imgDynaMaskLast, 
                            const cv::Mat &imgLabel, 
                            const Eigen::Matrix4d &poseRelative,
                            const Eigen::Isometry3d &Twc
                            // myPointCloud::Ptr cloud1
                            )
    {
        if(indexKF == 15)
        {
            int fsdf = 0;
        }
        cv::Mat imgDynaMaskNew = imgDynaMask.clone();
        //把上一帧的深度图投到这一帧来比较
        cv::Mat imgDepthNew(imgDynaMaskNew.size(), imgDepth.type(), cv::Scalar(0));
        int width = imgDepth.cols;
        int height = imgDepth.rows;
        // for(int row = 0; row < imgDepth.rows; ++row)
        // {
        //     for(int col = 0; col < imgDepth.cols; ++col)
        //     {
        //         float d = (float)(imgDepthLast.ptr<ushort>(row)[col] * (1.0 / depthScale));
        //         // std::cout << iLabel << std::endl;
        //             // Eigen::Vector2d px_uv(n ,m);//此时row和col是在dstImg上遍历
        //         Eigen::Vector3d p_xyz(( col - (float)cx) * (float)d / (float)fx ,( row - (float)cy) * d / (float)fy, d);
        //         auto T_i_j = poseRelative.inverse();
        //         Eigen::Matrix3d RotMatirx;
        //         Eigen::Vector3d translation;
        //         RotMatirx << T_i_j(0, 0) , T_i_j(0, 1) , T_i_j(0, 2),
        //                     T_i_j(1, 0) , T_i_j(1, 1) , T_i_j(1, 2),
        //                     T_i_j(2, 0) , T_i_j(2, 1) , T_i_j(2, 2);
        //         translation << T_i_j(0, 3) , T_i_j(1, 3) , T_i_j(2, 3);
        //         Eigen::Vector3d px_translate = K *(RotMatirx* p_xyz + translation); 

        //         // Eigen::Vector3d px_translate = translate(px_uv, poseRelative.inverse(), K);//找到dstImg上每个像素点在srcImg中对应的位置
        //         float dLast0 = (float)px_translate(2);
        //         px_translate /= px_translate(2);
        //         float x_translate = (float)(px_translate(0));
        //         float y_translate = (float)(px_translate(1));
        //         float dLast = dLast0;
        //         // if(y_translate >= 200 && y_translate < 202 && x_translate >= 200 && x_translate < 202)
        //         // {
        //         //     int jjj = 0;
        //         // }
        //         if(y_translate >= 0.0f && y_translate < height && x_translate >= 0.0f && x_translate < width)
        //         {
        //             // dLast = (float)(imgDepthLast.ptr<ushort>((int)y_translate)[(int)x_translate] * (1.0 / depthScale));
        //             imgDepthNew.ptr<ushort>((int)y_translate)[(int)x_translate] = (ushort)(dLast0 * depthScale);
        //         }
        //     }
        // }

        myPointCloud::Ptr tmp = pcl::make_shared<myPointCloud>();// point cloud is null ptr, 所有的点云
        tmp->points.reserve(width * height);
        
        std::vector<myPointCloud::Ptr> tmpCluster;// 每个簇都单独生成点云
        std::vector<double> vecOcclusion(12);//每个簇Occlusion点的数量
        // std::vector<cv::Mat> vecImgOcclusion(12);
        for(int i = 0; i < 12; i++)
        {
            myPointCloud::Ptr tmp1( new myPointCloud() );
            tmp1->points.reserve(width * height);
            tmp1->is_dense = false;
            tmpCluster.push_back(tmp1);
        }
        cv::Mat imgKK(imgDynaMask.size(), CV_32FC1, cv::Scalar(0));
        imgKK.setTo(0);
        auto T_i_j = poseRelative;
        Eigen::Matrix3d RotMatirx;
        Eigen::Vector3d translation;
        RotMatirx << T_i_j(0, 0) , T_i_j(0, 1) , T_i_j(0, 2),
                    T_i_j(1, 0) , T_i_j(1, 1) , T_i_j(1, 2),
                    T_i_j(2, 0) , T_i_j(2, 1) , T_i_j(2, 2);
        translation << T_i_j(0, 3) , T_i_j(1, 3) , T_i_j(2, 3);
        for ( int m = 0; m < height; m += 2 )
        {
            for ( int n = 0; n < width; n += 2 )
            {
                //可以选择只重建静态区域
                //不可以，因为这样远方的点不会被重建，很多噪声不能被更新
                // if((int)dynaMask.ptr<uchar>(m)[n] != 125)

                float dCurrent = (float)(imgDepth.ptr<ushort>(m)[n] * (1.0 / depthScale));
                int iLabel = (int)imgLabel.ptr<uchar>(m)[n];
                // std::cout << iLabel << std::endl;
                if(iLabel >= 12)
                {
                    continue;
                    // cout << "错误！！！！！！！！！！！！！！！！！！！！！！！！！！！"<< endl;
                }
                // if(iLabel != 0)
                {
                    // Eigen::Vector2d px_uv(n ,m);//此时row和col是在dstImg上遍历
                    Eigen::Vector3d p_xyz(( n - (float)cx) * (float)dCurrent / (float)fx ,( m - (float)cy) * dCurrent / (float)fy, dCurrent);
                    Eigen::Vector3d px_translate = K *(RotMatirx* p_xyz + translation); 

                    // Eigen::Vector3d px_translate = translate(px_uv, poseRelative.inverse(), K);//找到dstImg上每个像素点在srcImg中对应的位置
                    float dLast0 = (float)px_translate(2);
                    px_translate /= px_translate(2);
                    float x_translate = (float)(px_translate(0));
                    float y_translate = (float)(px_translate(1));
                    // float dLast = dLast0;
                    float dLast = 0.0f;
                    bool isDynaLast = 0;
                    // if(y_translate >= 200 && y_translate < 202 && x_translate >= 200 && x_translate < 202)
                    // {
                    //     int jjj = 0;
                    // }
                    if(y_translate >= 0.0f && y_translate < height && x_translate >= 0.0f && x_translate < width)
                    {
                        dLast = (float)(imgDepthLast.ptr<ushort>((int)y_translate)[(int)x_translate] * (1.0 / depthScale));
                        imgDepthNew.ptr<ushort>((int)m)[(int)n] = (ushort)(dLast * depthScale);
                        isDynaLast = (imgDynaMaskLast.ptr<uchar>((int)y_translate)[(int)x_translate] > 240);
                    }
                    if(dCurrent >= 0 && dCurrent < 10 && dLast >= 0 && dLast < 10)
                    {
                        float diff = dCurrent - dLast;
                        // if(diff == 0)
                        // {
                        //     imgKK.at<float>(m, n) = -1;
                        // }
                        // else
                        // {
                            // imgKK.at<float>(m, n) = diff;
                        // }
                        
                        if((diff * diff) > (0.13 * dCurrent) * (0.13 * dCurrent) || isDynaLast)
                        // if(diff < -0.13f * dCurrent)
                        {
                            vecOcclusion[iLabel]++;
                            imgKK.at<float>(m, n) = diff;
                            // continue;
                        }
                    }
                } 
                PointT p;
                if((int)imgDynaMask.ptr<uchar>(m)[n] >= 240)
                {
                    // continue;
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN ();
                   
                }
                else if (dCurrent < 0.01 || dCurrent > 10)
                {
                    // continue;
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN ();
                }
                else
                {
                    p.z = dCurrent;
                    p.x = ( n - cx) * p.z / fx;
                    p.y = ( m -cy) * p.z / fy;  
                } 
                p.b = imgRGB.ptr<uchar>(m)[n*3];
                p.g = imgRGB.ptr<uchar>(m)[n*3+1];
                p.r = imgRGB.ptr<uchar>(m)[n*3+2];
                    
                tmpCluster[iLabel]->points.push_back(p);
                // tmp->points.push_back(p);
            }
        }
        cv::Mat imgDepthError(imgDynaMask.size(), CV_32FC1, cv::Scalar(0));
        cv::Mat imgDepthFloat, imgDepthNewFloat;
        imgDepth.convertTo(imgDepthFloat, CV_32FC1);
        imgDepthNew.convertTo(imgDepthNewFloat, CV_32FC1);
        cv::subtract(imgDepthFloat, imgDepthNewFloat, imgDepthError);

        for(int i = 0; i < 12; i++)
        {
            // *tmp += *tmpCluster[i];
            if(i == 0)
            {
                *tmp += *tmpCluster[i];
                continue;
            }
            cv::Mat oneCluster;
            cv::compare(imgLabel, (cv::Mat::ones(imgLabel .size(), imgLabel.type()) * i), oneCluster, CV_CMP_EQ);
            // cv::Mat AndArea;
            // cv::bitwise_and(oneCluster, imgOcclusion, AndArea);
            // *tmp += *tmpCluster[i];
            //------------------------------//
            if( vecOcclusion[i] * 9 <= 0.4 * cv::countNonZero(oneCluster))
            {
                *tmp += *tmpCluster[i];
            }
            //TODO: 这个口是用来更新dynaMask的
            else
            {
                imgDynaMaskNew.setTo(255, oneCluster);
            }
            // cv::imwrite("/home/bhrqhb/catkin_ws/src/octomap_pub/output/dynaMask/" + std::to_string(indexKF) + ".png", imgDynaMask);
            // cv::imwrite("/home/bhrqhb/catkin_ws/src/octomap_pub/output/dynaMaskNew/" + std::to_string(indexKF) + ".png", imgDynaMaskNew);
            // cv::imwrite("/home/bhrqhb/catkin_ws/src/octomap_pub/output/imgKK/" + std::to_string(indexKF) + ".png", imgDepthError);
        }
        
        pcl::transformPointCloud( *tmp, *tempCloudOneFrame, Twc.matrix());
        // pcl::transformPointCloud( *tmp, *cloud2, T.inverse().matrix()); //专用于点云匹配
        // cloud1->is_dense = false;
        cout << "generate point cloud from  kf-ID:" << indexKF << ", size = " << tempCloudOneFrame->points.size() << endl;
        // cv::imshow("1", imgDynaMask);
        // cv::imshow("2", imgDynaMaskNew);
        // cv::waitKey(1);

        // cout << "点云的宽和高：" << cloud1->width << "," << cloud1->height << endl;
    }

    double cx;
    double cy;
    double fx;
    double fy;
    double depthScale;

    Eigen::Matrix3d K;
    std::vector<cv::Mat> vecImgdepth;
    std::vector<cv::Mat> vecImgDynaMask;
    std::vector<Eigen::Isometry3d ,Eigen::aligned_allocator<Eigen::Isometry3d>> vecPose;

    std::string pt_output_file;
};



void imgCallback(const sensor_msgs::ImageConstPtr &imgMsg)
{
    indexKF ++;
    ROS_INFO("KF index is %i", indexKF);
    cv::Mat imgRGB, imgDepth, imgDynaMask;
    cv_bridge::CvImageConstPtr pCvImage;

    pCvImage = cv_bridge::toCvShare(imgMsg, "bgr8");
    pCvImage->image.copyTo(imgRGB);
    try
    {
        ROS_INFO("The type of imgRGB is %i", imgRGB.type());
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", imgMsg->encoding.c_str());
    }

}


int main(int argc, char** argv)
{
    
    ros::init (argc, argv, "publish_pointcloud");  
    std::cout << "Creat OctoMap and Pub" << std::endl;
    SubscribeAndPublish SAPObject;
    ros::spin();
    

    return 0;
}