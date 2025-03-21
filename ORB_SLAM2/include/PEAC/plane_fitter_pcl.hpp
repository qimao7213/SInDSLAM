#pragma once
#pragma warning(disable: 4996)
#pragma warning(disable: 4819)
#define _CRT_SECURE_NO_WARNINGS

#include <map>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <pcl/common/transforms.h>
#include <pcl/io/openni_grabber.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>

#include "AHCPlaneFitter.hpp"
using ahc::utils::Timer;

// pcl::PointCloud interface for our ahc::PlaneFitter
template<class PointT>
struct OrganizedImage3D {
	const pcl::PointCloud<PointT>& cloud;
	//note: ahc::PlaneFitter assumes mm as unit!!!
	const double unitScaleFactor;

	OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1) {}
	OrganizedImage3D(const OrganizedImage3D& other) : cloud(other.cloud), unitScaleFactor(other.unitScaleFactor) {}

	inline int width() const { return cloud.width; }
	inline int height() const { return cloud.height; }
	inline bool get(const int row, const int col, double& x, double& y, double& z) const {
		const PointT& pt=cloud.at(col,row);
		x=pt.x*unitScaleFactor; y=pt.y*unitScaleFactor; z=pt.z*unitScaleFactor; //TODO: will this slowdown the speed?
		return pcl_isnan(z)==0; //return false if current depth is NaN
	}
};
typedef OrganizedImage3D<pcl::PointXYZ> ImageXYZ;
typedef ahc::PlaneFitter< ImageXYZ > PlaneFitter;
typedef pcl::PointCloud<pcl::PointXYZRGB> CloudXYZRGB;

namespace global {
std::map<std::string, std::string> ini;
PlaneFitter pf;
bool showWindow = true;

#ifdef _WIN32
const char filesep = '\\';
#else
const char filesep = '/';
#endif

// similar to matlab's fileparts
// if in=parent/child/file.txt
// then path=parent/child
// name=file, ext=txt
void fileparts(const std::string& str, std::string* pPath=0,
	std::string* pName=0, std::string* pExt=0)
{
	std::string::size_type last_sep = str.find_last_of(filesep);
	std::string::size_type last_dot = str.find_last_of('.');
	if (last_dot<last_sep) // "D:\parent\child.folderA\file", "D:\parent\child.folderA\"
		last_dot = std::string::npos;

	std::string path, name, ext;

	if (last_sep==std::string::npos) {
		path = "";
		if(last_dot==std::string::npos) { // "test"
			name = str;
			ext = "";
		} else { // "test.txt"
			name = str.substr(0, last_dot);
			ext = str.substr(last_dot+1);
		}
	} else {
		path = str.substr(0, last_sep);
		if(last_dot==std::string::npos) { // "d:/parent/test", "d:/parent/child/"
			name = str.substr(last_sep+1);
			ext = "";
		} else { // "d:/parent/test.txt"
			name = str.substr(last_sep+1, last_dot-last_sep-1);
			ext = str.substr(last_dot+1);
		}
	}

	if(pPath!=0) {
		*pPath = path;
	}
	if(pName!=0) {
		*pName = name;
	}
	if(pExt!=0) {
		*pExt = ext;
	}
}

//"D:/test/test.txt" -> "D:/test/"
std::string getFileDir(const std::string &fileName)
{
	std::string path;
	fileparts(fileName, &path);
	return path;
}

//"D:/parent/test.txt" -> "test"
//"D:/parent/test" -> "test"
std::string getNameNoExtension(const std::string &fileName)
{
	std::string name;
	fileparts(fileName, 0, &name);
	return name;
}

void iniLoad(std::string iniFileName) {
	std::ifstream in(iniFileName);
	if(!in.is_open()) {
		std::cout<<"[iniLoad] "<<iniFileName<<" not found, use default parameters!"<<std::endl;
		return;
	}
	while(in) {
		std::string line;
		std::getline(in, line);
		if(line.empty() || line[0]=='#') continue;
		std::string key, value;
		size_t eqPos = line.find_first_of("=");
		if(eqPos == std::string::npos || eqPos == 0) {
			std::cout<<"[iniLoad] ignore line:"<<line<<std::endl;
			continue;
		}
		key = line.substr(0,eqPos);
		value = line.substr(eqPos+1);
		std::cout<<"[iniLoad] "<<key<<"=>"<<value<<std::endl;
		ini[key]=value;
	}
}

template<class T>
T iniGet(std::string key, T default_value) {
	std::map<std::string, std::string>::const_iterator itr=ini.find(key);
	if(itr!=ini.end()) {
		std::stringstream ss;
		ss<<itr->second;
		T ret;
		ss>>ret;
		return ret;
	}
	return default_value;
}

template<> std::string iniGet(std::string key, std::string default_value) {
	std::map<std::string, std::string>::const_iterator itr=ini.find(key);
	if(itr!=ini.end()) {
		return itr->second;
	}
	return default_value;
}
}//global
int index_image = 1;
void processOneFrame(pcl::PointCloud<pcl::PointXYZ>& cloud, const std::string& outputFilePrefix, cv::Mat &imgEdgeByPlane)
{
	using global::pf;
	cv::Mat seg(cloud.height, cloud.width, CV_8UC3);
	// cv::Mat planeCountous(cloud.height, cloud.width, CV_8UC1, cv::Scalar(0));
	//run PlaneFitter on the current frame of point cloud
	ImageXYZ Ixyz(cloud);
	Timer timer(1000);
	timer.tic();
	pf.run(&Ixyz,imgEdgeByPlane, 0, &seg);
	double process_ms=timer.toc();
	// std::cout<<process_ms<<" ms"<<std::endl;

	//save seg image
	cv::cvtColor(seg,seg,cv::COLOR_RGB2BGR);
	seg.setTo(cv::Scalar(0, 255, 0), imgEdgeByPlane);
	// cv::imwrite("/home/bhrqhb/myCode/DepthBaseSeg/output/imgPlaneOutput.png", seg);
	// cv::imwrite("/home/bhrqhb/myCode/DepthBaseSeg/output/planeContours/" + std::to_string(index_image) + ".png", seg);
	
	// std::cout<<"output: "<<outputFilePrefix<<".seg.png"<<std::endl;
	// result_polytopics.clear();
	// for (auto & poly:pf.extractedPlanes)
	// {
	// 	Eigen::Vector3f normal(poly->normal[0], poly->normal[1], poly->normal[2]);
	// 	Eigen::Vector3f center(poly->center[0], poly->center[1], poly->center[2]);
	// 	polytopic tmppolytopic(normal, poly->contours, poly->N, center, poly->mse);
	// 	result_polytopics.emplace_back(tmppolytopic);
	// }
	
	// //save seg cloud
	// CloudXYZRGB xyzrgb(cloud.width, cloud.height);
	// for(int r=0; r<(int)xyzrgb.height; ++r) {
	// 	for(int c=0; c<(int)xyzrgb.width; ++c) {
	// 		pcl::PointXYZRGB& pix = xyzrgb.at(c, r);
	// 		const pcl::PointXYZ& pxyz = cloud.at(c, r);
	// 		const cv::Vec3b& prgb = seg.at<cv::Vec3b>(r,c);
	// 		pix.x=pxyz.x;
	// 		pix.y=pxyz.y;
	// 		pix.z=pxyz.z;
	// 		pix.r=prgb(2);
	// 		pix.g=prgb(1);
	// 		pix.b=prgb(0);
	// 	}
	// }
	// pcl::io::savePCDFileBinary(outputFilePrefix + std::to_string(index_image) + ".seg.pcd", xyzrgb);
	index_image++;

	// if(global::showWindow) {
	// 	//show frame rate
	// 	std::stringstream stext;
	// 	stext<<"Frame Rate: "<<(1000.0/process_ms)<<"Hz";
	// 	cv::putText(seg, stext.str(), cv::Point(15,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255,1));

	// 	cv::imshow("seg", seg);
	// 	cv::waitKey(1);
	// }
}

int process(pcl::PointCloud<pcl::PointXYZ> & orderd_cloud, ahc::FitterAllParams & parameters, cv::Mat &imgEdgeByPlane) {
	// global::iniLoad("../plane_fitter_pcd.ini");
	const double unitScaleFactor = global::iniGet<double>("unitScaleFactor", 1.0f);
    // std::cout<<"unitScaleFactor: "<<unitScaleFactor<<std::endl;
	const std::string outputDir = global::iniGet<std::string>("outputDir", ".");
	{//create outputDir
#ifdef _WIN32
		std::string cmd="mkdir "+outputDir;
#else
		std::string cmd="mkdir -p "+outputDir;
#endif
		system(cmd.c_str());
	}

	using global::pf;
	//setup fitter
	pf.minSupport = parameters.minSupport;
	pf.windowWidth = parameters.windowWidth;
	pf.windowHeight = parameters.windowHeight;
	pf.doRefine = parameters.doRefine;

	pf.params.initType = parameters.para.initType;

	//T_mse
	pf.params.stdTol_merge = parameters.para.stdTol_merge;
	pf.params.stdTol_init = parameters.para.stdTol_init;
	pf.params.depthSigma = parameters.para.depthSigma;

	//T_dz
	pf.params.depthAlpha = parameters.para.depthAlpha;
	pf.params.depthChangeTol = parameters.para.depthChangeTol;

	//T_ang
	pf.params.z_near = parameters.para.z_near;
	pf.params.z_far = parameters.para.z_far;

	pf.params.angle_near = parameters.para.angle_near;
	pf.params.angle_far = parameters.para.angle_far;

	pf.params.similarityTh_merge = parameters.para.similarityTh_merge;
	pf.params.similarityTh_refine = parameters.para.similarityTh_refine;

	using global::showWindow;
	showWindow = global::iniGet("showWindow", true);
	// if(showWindow)
	// 	cv::namedWindow("seg");
    std::string outputFilePrefix = outputDir+global::filesep;
    pcl::transformPointCloud<pcl::PointXYZ>(
				orderd_cloud, orderd_cloud,
				Eigen::Affine3f(Eigen::UniformScaling<float>(
				(float)unitScaleFactor)));
    processOneFrame(orderd_cloud, outputFilePrefix, imgEdgeByPlane);
	
	return 0;
}


class PlanarContourExtraction
{
private:
	/**
	 * input
	*/
	pcl::PointCloud<pcl::PointXYZ> cloud;
	ahc::FitterAllParams parameters;

	/**
	 * output
	*/
	// vector<polytopic> polytopics;
public:
	PlanarContourExtraction()
	{

	}
	PlanarContourExtraction(pcl::PointCloud<pcl::PointXYZ> & input_cloud)
	{
		cloud = input_cloud;
	}
	PlanarContourExtraction(pcl::PointCloud<pcl::PointXYZ> & input_cloud, ahc::FitterAllParams & input_parameters)
	{
		cloud = input_cloud;
		parameters = input_parameters;
	}
	// void run()
	// {
	// 	process(cloud, parameters);
	// 	polytopics = result_polytopics;
	// }
	void run(cv::Mat &imgEdgeByPlane)
	{
		process(cloud, parameters, imgEdgeByPlane);
		// polytopics = result_polytopics;
	}
	~PlanarContourExtraction()
	{
		cloud.clear();
		// polytopics.clear();
	}
};

