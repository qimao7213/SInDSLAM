#ifndef DYNADETECT_H
#define DYNADETECT_H


#include <opencv4/opencv2/opencv.hpp>
#include <omp.h>
#include <future>
using cv::Mat;
using std::vector;
namespace ORB_SLAM2
{
//-------------------------------------------------------//
/**
 * @brief 这是一组imgMask
*/
struct stImgMasks
{
    stImgMasks(const cv::Mat &imgMaskLowError_, const cv::Mat &imgMaskHighError_):
    imgMaskLowError(imgMaskLowError_), imgMaskHighError(imgMaskHighError_)
    {
        
    }
    stImgMasks(const stImgMasks &instance):
    imgMaskLowError(instance.imgMaskLowError), imgMaskHighError(instance.imgMaskHighError)
    {

    }
    cv::Mat imgMaskLowError;
    cv::Mat imgMaskHighError;
};

class myCluster{
public:
	myCluster(const cv::Mat &img)
	{
		img_.create(img.size(), img.type());
		img.copyTo(img_);
		imgDilated_.create(img.size(), img.type());
		lianjie_.create(img.size(), img.type());
		// img_ = img.clone();
	};
	myCluster(){};
	~myCluster(){};
	// Mat cluster() const{
	// 	return img_.clone();
	// }
	const cv::Mat& cluster() const 
	{
    	return img_;
	}
	const cv::Mat& clusterDilated() const 
	{
    	return imgDilated_;
	}
	void setClusterDilated(const cv::Mat &img)
	{
		
		img.copyTo(imgDilated_);
		// imgDilated_ = img.clone();
	}
	void setContours(const vector<cv::Point> &oneContour)
	{
		Contours_.push_back(oneContour);
	}
	cv::Vec3f centerPoint() const{
		return centerPoint_;
	}
	cv::Point2f centerPx() const{
		return centerPx_;
	}	
	Mat setlianjie(const cv::Mat &lianjie){
		
		lianjie.copyTo(lianjie_);
		// lianjie_ = lianjie.clone();
	}
	const cv::Mat& lianjie() const 
	{
    	return lianjie_;
	}
	void calCenterPoint(const Mat &pointsPyra, const Mat &imgDepth);
	void merge(const myCluster &toBeMerge);
	Mat lianjie_; //"lianjie" means connection, adjacency
	bool isClose = 0;
	float area = 0;
	float socre = -10;
private:
	Mat img_; //CV_8U，filled
	Mat imgDilated_; // Dilated based on img_, used to determine whether it is adjacent to other clusters.
	cv::Vec3f centerPoint_;
	cv::Point2f centerPx_;
	vector<vector<cv::Point>> Contours_;
};

//------------------------------------------------//
class DynaDetect
{
public:
    DynaDetect( const cv::InputArray &imgLast_,
				const cv::InputArray &imgLastLast_,
				float fx_,
				float fy_,
				float cx_,
				float cy_,
				float depthScale_):
    imgRGBLast(imgLast_.getMat()), imgRGBLastLast(imgLastLast_.getMat()), fx(fx_), fy(fy_), cx(cx_), cy(cy_), depthScale(depthScale_)
	{
		// cv::cvtColor(imgRGBLast, imgGrayLast, cv::COLOR_BGR2GRAY);
		// cv::cvtColor(imgRGBLastLast, imgGrayLastLast, cv::COLOR_BGR2GRAY);
		imgDyna = cv::Mat(imgRGBLast.size(), CV_8UC1, cv::Scalar(0)).clone();
		imgDynaLast = imgDyna.clone();
		imgMaskHighErrorLast = imgDyna.clone();
		imgLabelLast = imgDyna.clone();
		aroundPoint.resize(12);
		aroundPoint[0].x = 0; aroundPoint[0].y = -2;
		aroundPoint[1].x = 1; aroundPoint[1].y = -2;
		aroundPoint[2].x = 2; aroundPoint[2].y = -1;
		aroundPoint[3].x = 2; aroundPoint[3].y = 0;
		aroundPoint[4].x = 2; aroundPoint[4].y = 1;
		aroundPoint[5].x = 1; aroundPoint[5].y = 2;
		aroundPoint[6].x = 0; aroundPoint[6].y = 2;
		aroundPoint[7].x = -1; aroundPoint[7].y = 2;
		aroundPoint[8].x = -2; aroundPoint[8].y = 1;
		aroundPoint[9].x = -2; aroundPoint[9].y = 0;
		aroundPoint[10].x = -2; aroundPoint[10].y = -1;
		aroundPoint[11].x = -1; aroundPoint[11].y = -2;
	};
	void DetectDynaArea( const cv::InputArray &img_,
						 const cv::InputArray &imgDepth_,
						 cv::OutputArray &imgDyna_,
						 cv::OutputArray &imgLabel_,
				         int nImg_);
	// imgRGB(img_.getMat()), imgDepth(imgDepth_.getMat()), nImg(nImg_);	 	

private:
    void SegByKmeans(cv::Mat &imgLabelInOut, cv::Mat &points, cv::Mat &centers);
    void CalOccluded(const cv::Mat &imgLabelForSegEdge, 
					 cv::Mat &imgTotalArea, 
					 cv::Mat &imgOccluded1, 
					 cv::Mat &imgOccluded2);
	void SegAndMerge(const std::vector<Mat> &allLabels, 
					const cv::Mat &imgOccluded, 
					const cv::Mat &imgOccluded2,
					const cv::Mat &imgLabelForSegEdge, 
					const cv::Mat &points_pyramid,
					cv::Mat &imgLabelNew,
					int count0);
	void SegAndMergeV2(const std::vector<Mat> &allLabels, 
					const cv::Mat &imgOccluded, 
					const cv::Mat &imgOccluded2,
					const cv::Mat &imgLabelForSegEdge, 
					const cv::Mat &points_pyramid,
					cv::Mat &imgLabelNew,
					int count0);
					
    void DetectDynaByOpticalFLow(const cv::Mat &imgTotalArea, const cv::Mat &imgLabel, const std::vector<Mat> &EveryLabel);
	void DetectDynaByDenseOpticalFLow(const cv::Mat &imgTotalArea, const cv::Mat &imgLabel, const std::vector<Mat> &EveryLabel);
	void DetectDynaByDenseOpticalFLow(
									cv::Mat &imgMaskLowError,
									cv::Mat &imgMaskHighError,
									cv::Mat &imgMaskTotal);
	void DetectDynaByDenseOpticalFLow(std::promise<stImgMasks> &imgMasks);
	

	cv::Mat imgRGB;
    cv::Mat imgRGBLast;
    cv::Mat imgRGBLastLast;
    cv::Mat imgDepth;

	cv::Mat imgGray;
    cv::Mat imgGrayLast;
    cv::Mat imgGrayLastLast;
    cv::Mat imgDyna;
	cv::Mat imgDynaLast;
	cv::Mat imgLabel;
	cv::Mat imgLabelLast;
	// cv::Mat imgMaskTotalSave;
	// cv::Mat imgMaskLowError;
	cv::Mat imgMaskHighErrorLast;
	// cv::Mat imgMaskTotal;
	int nImg;

	const float cx;
	const float cy;
	const float fx;
	const float fy;
	const float depthScale;
	//用来选取周围的点
    std::vector<cv::Point2i> aroundPoint;
};



}//namespace ORB_SLAM
#endif 