#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

int main(int argc, char **argv){
    // 加载房间的第一次扫描点云作为target
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if(pcl::io::loadPCDFile<pcl::PointXYZ>("../room_scan1.pcd", *target_cloud) == -1){
        PCL_ERROR("Couldn't read file room_scan1.pcd\n");
        exit(1);
    }
    std::cout << "Loaded " << target_cloud->size() << " data points from room_scan1.pcd" << std::endl;
    // 第二次扫描作为source
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if(pcl::io::loadPCDFile<pcl::PointXYZ>("../room_scan2.pcd", *input_cloud) == -1)
    {
        PCL_ERROR("Couldn't read file room_scan2.pcd \n");
        exit(1);
    }
    std::cout << "Loaded " << input_cloud->size() << " data points from room_scan2.pcd" << std::endl;


    // 对第二次扫描的点云降采样
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
    approximate_voxel_filter.setLeafSize(0.2, 0.2, 0.2);
    approximate_voxel_filter.setInputCloud(input_cloud);
    approximate_voxel_filter.filter(*filtered_cloud);
    std::cout << "Filtered cloud contains "<< filtered_cloud->size()
              << " data points from room_scan2.pcd" << std::endl;
    
    // NDT
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);
    ndt.setMaximumIterations(35);
    ndt.setInputSource(filtered_cloud);
    ndt.setInputTarget(target_cloud);
    Eigen::AngleAxisf init_rotation(0.6931, Eigen::Vector3f::UnitZ());
    Eigen::Translation3f init_translation(1.79387, 0.720047, 0);
    Eigen::Matrix4f init_guess = (init_translation * init_rotation).matrix();

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*output_cloud, init_guess);
    std::cout << "Normal Distribution transform has converged:" << ndt.hasConverged()
              << " Score:" << ndt.getFitnessScore() << std::endl;
    
    // 将两帧点云对齐
    pcl::transformPointCloud(*input_cloud, *output_cloud, ndt.getFinalTransformation());
    pcl::io::savePCDFileASCII("room_scan2_transformed.pcd", *output_cloud);

    // 可视化
    boost::shared_ptr<pcl::visualization::PCLVisualizer>
    viewer_final(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_final->setBackgroundColor(0, 0, 0);
    
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> 
    target_color(target_cloud, 255, 0, 0); // target点云1被设置成红色
    viewer_final->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 
                                                    1, "target cloud");
    
    // green 变换后的点云设置成绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
    output_color (output_cloud, 0, 255, 0);
    viewer_final->addPointCloud<pcl::PointXYZ> (output_cloud, output_color, "output cloud");
    viewer_final->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
                                                  1, "output cloud");

    viewer_final->addCoordinateSystem(1.0);
    viewer_final->initCameraParameters();

    while(!viewer_final->wasStopped())
    {
        viewer_final->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}