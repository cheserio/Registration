cmake_minimum_required(VERSION 2.8)

project(pcl_Registration)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(PCL REQUIRED COMPONENT common io registration visualization features filters)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(interactive_icp interactive_icp.cpp)
target_link_libraries(interactive_icp ${PCL_LIBRARIES})

add_executable(iterative_cloesest_point iterative_closest_point.cpp)
target_link_libraries(iterative_cloesest_point ${PCL_LIBRARIES})

add_executable(ndt normal_distributions_transform.cpp)
target_link_libraries(ndt ${PCL_LIBRARIES})

add_executable(alignment alignment_prerejective.cpp)
target_link_libraries(alignment ${PCL_LIBRARIES})