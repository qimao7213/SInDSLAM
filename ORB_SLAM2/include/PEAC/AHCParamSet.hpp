// Copyright (C) 2014,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#include "math.h"

namespace ahc {

#define MACRO_DEG2RAD(d) ((d)*M_PI/180.0)
#define MACRO_RAD2DEG(r) ((r)*180.0/M_PI)

enum InitType {
	INIT_STRICT=0,	//no nan point is allowed in any valid init blocks
	INIT_LOOSE=1	//at most half of a init block can be nan point
};

/**
*  \brief ParamSet is a struct representing a set of parameters used in ahc::PlaneFitter
*/
struct ParamSet {
	// related to T_mse
	double depthSigma;		//\sigma in the paper, unit: u^-1 mm^-1
	double stdTol_init;		//\epsilon in the paper, used when init graph, unit: u mm
	double stdTol_merge;	//\epsilon in the paper, used when merging nodes, unit: u mm

	// related to T_ang
	double z_near, z_far;			//unit: u mm, closest/farthest z to be considered
	double angle_near, angle_far;	//unit: rad, corresponding normal deviation angle
	double similarityTh_merge;		//unit: none, 1 means the same, 0 means perpendicular
	double similarityTh_refine;		//unit: none

	// related to T_dz
	double depthAlpha;	//unit: none, corresponds to the 2*\alpha in the paper
	double depthChangeTol;		//unit: u mm

	InitType initType;

	enum Phase {
		P_INIT=0,
		P_MERGING=1,
		P_REFINE=2
	};
	
	// 注意参数：
	// depthSigma和相机的型号有关，传感器的噪声参数，如果是数据集里面或在普通的相机，那么这个depthSigma应该设置地相对大一点
	// indoor
	ParamSet() : depthSigma(3e-6),
		stdTol_init(10), stdTol_merge(17),//算MSE时的参数
		z_near(500), z_far(6000),
		angle_near(MACRO_DEG2RAD(10.0)), angle_far(MACRO_DEG2RAD(20.0)),//这个near和far是为了线形插值
		similarityTh_merge(std::cos(MACRO_DEG2RAD(15.0))),//两个平面合并时角度差
		similarityTh_refine(std::cos(MACRO_DEG2RAD(20.0))),
		depthAlpha(0.04), depthChangeTol(0.02 * 1000),
		initType(INIT_STRICT)//初始化类型，是不是只要有深度无效的点就直接判断这个grid为novalide
	{}

	//outdoor
	// ParamSet() : depthSigma(5.0e-7),
	// 	stdTol_init(3), stdTol_merge(5),//算MSE时的参数
	// 	z_near(3000), z_far(40000),
	// 	angle_near(MACRO_DEG2RAD(10.0)), angle_far(MACRO_DEG2RAD(20.0)),//这个near和far是为了线形插值
	// 	similarityTh_merge(std::cos(MACRO_DEG2RAD(10.0))),//两个平面合并时角度差
	// 	similarityTh_refine(std::cos(MACRO_DEG2RAD(15.0))),
	// 	depthAlpha(0.02), depthChangeTol(0.02 * 1000), //计算深度不连续性时使用
	// 	initType(INIT_STRICT)//初始化类型，是不是只要有深度无效的点就直接判断这个grid为novalide
	// {}

	/**
	 *  \brief Dynamic MSE threshold, depending on depth z
	 *
	 *  \param [in] phase specify whether invoked in initGraph or when merging
	 *  \param [in] z current depth, unit: u mm
	 *  \return the MSE threshold at depth z, unit: u^2 mm^2
	 *
	 *  \details Reference: 2012.Sensors.Khoshelham.Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications
	 */
	inline double T_mse(const Phase phase, const double z=0) const {
		//theoretical point-plane distance std = sigma * z * z
		//sigma corresponds to \sigma * (m/f/b) in the 2012.Khoshelham paper
		//we add a stdTol to move the theoretical curve up as tolerances
		switch(phase) {
		case P_INIT:
			return std::pow(depthSigma*z*z+stdTol_init,2);
		case P_MERGING:
		case P_REFINE:
		default:
			return std::pow(depthSigma*z*z+stdTol_merge,2);
		}
	}

	/**
	 *  \brief Dynamic normal deviation threshold, depending on depth z
	 *
	 *  \param [in] phase specify whether invoked in initGraph or when merging
	 *  \param [in] z current depth (z>=0)
	 *  \return cos of the normal deviation threshold at depth z
	 *
	 *  \details This is simply a linear mapping from depth to thresholding angle
	 *  and the threshold will be used to reject edge when initialize the graph;
	 *  this function corresponds to T_{ANG} in our paper
	 */
	inline double T_ang(const Phase phase, const double z=0) const {
		switch(phase) {
		case P_INIT:
			{//linear maping z->thresholding angle, clipping z also
				double clipped_z = z;
				clipped_z=std::max(clipped_z,z_near);
				clipped_z=std::min(clipped_z,z_far);
				const double factor = (angle_far-angle_near)/(z_far-z_near);
				return std::cos(factor*clipped_z+angle_near-factor*z_near);
			}
		case P_MERGING:
			{
				return similarityTh_merge;
			}
		case P_REFINE:
		default:
			{
				return similarityTh_refine;
			}
		}
	}

	/**
	 *  \brief Dynamic threshold to test whether the two adjacent pixels are discontinuous in depth
	 *
	 *  \param [in] z depth of the current pixel
	 *  \return the max depth change allowed at depth z to make the points connected in a single block
	 *
	 *  \details This is modified from pcl's segmentation code as well as suggested in 2013.iros.Holzer
	 *  essentially returns factor*z+tolerance
	 *  (TODO: maybe change this to 3D-point distance threshold)
	 */
	inline double T_dz(const double z) const {
		return depthAlpha * fabs(z) + depthChangeTol;
	}
};//ParamSet

//indoor
struct FitterAllParams
{
	ParamSet para;
	int minSupport = 2000;
	int windowWidth = 16;
	int windowHeight = 16;
	bool doRefine = true;
};

// //outdoor
// struct FitterAllParams
// {
// 	ParamSet para;
// 	int minSupport = 2000;
// 	int windowWidth = 10;
// 	int windowHeight = 10;
// 	bool doRefine = true;
// };

}//ahc
