cmake_minimum_required(VERSION 3.0.2)
project(lidarOdom)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  roscpp
  sensor_msgs
  std_msgs
)

find_package(PCL 1.8 REQUIRED)

include_directories(${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

catkin_package()

add_executable(lidarOdom src/lidarOdom.cpp)
target_link_libraries(lidarOdom ${catkin_LIBRARIES} ${PCL_LIBRARIES})
