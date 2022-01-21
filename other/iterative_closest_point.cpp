#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <sstream>

int main(int argc, char **argv){
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZ>);

    float x_trans = 0.7; // x轴方向偏移0.7m
    if(argc >= 2){
        std::istringstream xss(argv[1]); // 可以自己通过主函数参数输入偏移量
        xss >> x_trans;
    }

    // 自己做个点云
    cloud_in->width    = 5;
    cloud_in->height   = 1;
    cloud_in->is_dense = false;
    cloud_in->points.resize(cloud_in->width * cloud_in->height);
    for(size_t i = 0; i < cloud_in->points.size(); ++i){
        // 对于其中每个点填充值
        cloud_in->points[i].x = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud_in->points[i].y = 1024 * rand() / (RAND_MAX + 1.0f);
        cloud_in->points[i].z = 1024 * rand() / (RAND_MAX + 1.0f);
    }
    std::cout << "Saved "<< cloud_in->points.size() << " data points to input:" << std::endl;
    
    for(size_t i = 0; i < cloud_in->points.size(); ++i){
        std::cout << "    " << cloud_in->points[i].x << " " << 
            cloud_in->points[i].y << " " << cloud_in->points[i].z << std::endl;
    }

    *cloud_out = *cloud_in;

    std::cout << "size:" << cloud_out->points.size() << std::endl;

    // 将点云中的每个点都沿x轴移动0.7m
    for(size_t i = 0; i < cloud_in->points.size(); ++i){
        cloud_out->points[i].x = cloud_in->points[i].x + x_trans;
    }
    std::cout << "Transformed " << cloud_in->points.size() << " data points:" << std::endl;
    
    // 输出构造出的点云
    for(size_t i = 0; i < cloud_out->points.size(); ++i){
        std::cout << "    " << cloud_out->points[i].x << " " << 
            cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;
    }


    // icp
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputCloud(cloud_in);
    icp.setInputTarget(cloud_out);
    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);


    // 打印变换后的点云
    std::cout << "Assiend " << Final.points.size() << " data points:" << std::endl;
    for(size_t i = 0; i < Final.points.size(); ++i){
        std::cout << "    " << cloud_out->points[i].x << " " << 
            cloud_out->points[i].y << " " << cloud_out->points[i].z << std::endl;
    }

    // 打印配准相关输入信息
    std::cout << "has converged:" << icp.hasConverged() << " score:" <<
    icp.getFitnessScore() << std::endl;
    std::cout << "Transformation: " << "\n" << icp.getFinalTransformation() << std::endl;

    return 0;
}