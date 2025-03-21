cmake_minimum_required(VERSION 3.1)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)
project(ORB_SLAM2)

# rosbuild_init()

set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native -msse")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native -msse")


# Set true or false, Not comment out!
set(CV_CUDA false)
set(CV_420 true)

if(CV_CUDA)
   add_definitions(-DUSECUDA)
   find_package(CUDA REQUIRED)
   # Ensure the "arch" and "code" fit to you GPU and CUDA version
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_89,code=sm_89;-std=c++14;)
endif(CV_CUDA)

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX14)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC14)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.2.0 EXACT REQUIRED)

find_package(Eigen3 3.1.0 REQUIRED NO_MODULE)
find_package(Pangolin REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
# find_package(CNPY REQUIRED)

include_directories(
   ${PROJECT_SOURCE_DIR}
   ${PROJECT_SOURCE_DIR}/include
   ${PROJECT_SOURCE_DIR}/include/PEAC
   ${EIGEN3_INCLUDE_DIR}
   ${Pangolin_INCLUDE_DIRS}
   ${OpenCV_INCLUDE_DIRS}
)


MESSAGE ( STATUS "libraries ${OpenCV_LIBRARIES}" )
MESSAGE ( STATUS "${OpenCV_DIR}" )
MESSAGE ( STATUS "${OpenCV_CONFIG_PATH}" )
MESSAGE ( STATUS "${CMAKE_SOURCE_DIR}")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc

src/DynaDetect.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PCL_LIBRARIES}
)

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

if(CV_420)
   find_package(catkin REQUIRED COMPONENTS 
            roscpp 
            std_msgs 
            geometry_msgs 
            sensor_msgs 
            image_transport 
            tf
            cv_bridge
            )
   include_directories(include ${catkin_INCLUDE_DIRS} )
   add_executable(rgbd_tum_withros
   Examples/RGB-D/rgbd_tum_withros.cc)
   target_link_libraries(rgbd_tum_withros ${PROJECT_NAME} ${catkin_LIBRARIES} OpenMP::OpenMP_CXX)   
else(CV_420)
   add_executable(rgbd_tum_noros
   Examples/RGB-D/rgbd_tum_noros.cc)
   target_link_libraries(rgbd_tum_noros ${PROJECT_NAME} OpenMP::OpenMP_CXX)   
endif(CV_420)

