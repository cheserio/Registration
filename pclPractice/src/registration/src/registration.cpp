#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <thread>
#include <chrono>

enum method{ICP_METHOD = 1, VICP_METHOD, GICP_METHOD, NDT_METHOD};
typedef pcl::PointXYZ PointType;
typedef pcl::PointNormal PointNormalT;
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

// pcd文件读取与报错路径
std::string savepath = "/home/chenshiyang/pclPractice/src/registration/output/";
std::string readpath = "/home/chenshiyang/pclPractice/src/registration/data/";

// icp point-to-point
bool icpAlign(const pcl::PointCloud<PointType>::Ptr cloud_src, pcl::PointCloud<PointType>::Ptr cloud_tgt, pcl::PointCloud<PointType>::Ptr output, Eigen::Matrix4d &final_transform, bool downsample = false){

    // 降采样
    pcl::PointCloud<PointType>::Ptr src(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgt(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> grid;
    if(downsample){
        grid.setLeafSize(0.05, 0.05, 0.05);
        grid.setInputCloud(cloud_src);
        grid.filter(*src);

        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
    }else{
        src = cloud_src;
        tgt = cloud_tgt;
    }

    // icp
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations(30);
    icp.setEuclideanFitnessEpsilon(1e-5); // 收敛条件
    icp.setMaxCorrespondenceDistance(0.10); // 距离超过10cm的点对不考虑
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    icp.align(*src);

    if(icp.hasConverged()){
        std::cout << "\nICP has converged, score is " << icp.getFitnessScore() << std::endl;
        Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
        Ti = icp.getFinalTransformation().cast<double>();
        Eigen::Matrix4d targetToSource = Ti.inverse();
        pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);
        final_transform = targetToSource;
        return true;
    }
    final_transform = Eigen::Matrix4d::Identity();
    *output = *cloud_tgt;
    return false;
}

// icp point-to-plane
bool vicpAlign(const pcl::PointCloud<PointType>::Ptr cloud_src, pcl::PointCloud<PointType>::Ptr cloud_tgt, pcl::PointCloud<PointType>::Ptr output, Eigen::Matrix4d &final_transform, bool downsample = false){
    
    // 下采样
    pcl::PointCloud<PointType>::Ptr src(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgt(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> grid;
    if(downsample){
        grid.setLeafSize(0.05, 0.05, 0.05);
        grid.setInputCloud(cloud_src);
        grid.filter(*src);

        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
    }else{
        src = cloud_src;
        tgt = cloud_tgt;
    }

    // 计算表面法向量和曲率
    // 先声明两个带法向量和曲率的点云用于接收
    PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
    PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

    pcl::NormalEstimation<PointType, PointNormalT> norm_est;
    // 为法线获取设置搜索方法
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(30);

    norm_est.setInputCloud(src);
    norm_est.compute(*points_with_normals_src);
    pcl::copyPointCloud(*src, *points_with_normals_src);

    norm_est.setInputCloud(tgt);
    norm_est.compute(*points_with_normals_tgt);
    pcl::copyPointCloud(*tgt, *points_with_normals_tgt);

    // icp
    pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
    reg.setMaximumIterations(30);
    reg.setTransformationEpsilon(1e-6);
    reg.setMaxCorrespondenceDistance(0.1);
    reg.setInputSource(points_with_normals_src);
    reg.setInputTarget(points_with_normals_tgt);
    reg.align(*points_with_normals_src);

    if(reg.hasConverged()){
        std::cout << "\nVICP has converged, score is " << reg.getFitnessScore() << std::endl;
        Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
        Ti = reg.getFinalTransformation().cast<double>();
        Eigen::Matrix4d targetToSource = Ti.inverse();
        pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);

        final_transform = targetToSource;
        return true;
    }

    return false;
}

// icp plane-to-plane
bool gicpAlign(const pcl::PointCloud<PointType>::Ptr cloud_src, pcl::PointCloud<PointType>::Ptr cloud_tgt, pcl::PointCloud<PointType>::Ptr output, Eigen::Matrix4d &final_transform, bool downsample = false){
    // 下采样
    pcl::PointCloud<PointType>::Ptr src(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgt(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> grid;
    if(downsample){
        grid.setLeafSize(0.05, 0.05, 0.05);
        grid.setInputCloud(cloud_src);
        grid.filter(*src);

        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
    }else{
        src = cloud_src;
        tgt = cloud_tgt;
    }

    // gicp
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations(30);
    icp.setTransformationEpsilon(1e-6);
    icp.setMaxCorrespondenceDistance(0.10);
    icp.setInputSource(src);
    icp.setInputTarget(tgt);
    icp.align(*src);

    if(icp.hasConverged()){
        std::cout << "\nGICP has converged, score is " << icp.getFitnessScore() << std::endl;
        Eigen::Matrix4d Ti = icp.getFinalTransformation().cast<double>();
        Eigen::Matrix4d targetToSource;
        targetToSource = Ti.inverse();
        pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);

        final_transform = targetToSource;

        return true;
    }
    return false;
}

// ndt
bool ndtAlign(const pcl::PointCloud<PointType>::Ptr cloud_src, pcl::PointCloud<PointType>::Ptr cloud_tgt, pcl::PointCloud<PointType>::Ptr output, Eigen::Matrix4d &final_transform, bool downsample = false){
    // 下采样
    pcl::PointCloud<PointType>::Ptr src(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgt(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> grid;
    if(downsample){
        grid.setLeafSize(0.05, 0.05, 0.05);
        grid.setInputCloud(cloud_src);
        grid.filter(*src);

        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
    }else{
        src = cloud_src;
        tgt = cloud_tgt;
    }
    *tgt = *cloud_tgt; // ????

    // ndt
    pcl::NormalDistributionsTransform<PointType, PointType> ndt;
    ndt.setTransformationEpsilon(1e-2);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);
    ndt.setMaximumIterations(30);
    ndt.setInputSource(src);
    ndt.setInputTarget(tgt);
    ndt.align(*src);

    if(ndt.hasConverged()){
        std::cout << "\nndt has converged, score is " << ndt.getFitnessScore() << std::endl;
        Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
        Ti = ndt.getFinalTransformation().cast<double>();
        Eigen::Matrix4d targetToSource;
        targetToSource = Ti.inverse();
        pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);
        final_transform = targetToSource;
        return true;
    }
    return false;
}

void writePCD(const pcl::PointCloud<PointType>::Ptr cloud, const std::string str){
    std::stringstream ss;
    ss << savepath << str << ".pcd";
    pcl::io::savePCDFile(ss.str(), *cloud, true);
    return ;
}

void readPCD(const pcl::PointCloud<PointType>::Ptr cloud, const size_t i){
    std::vector<int> mapping;
    std::stringstream ss;
    ss << readpath << "Rf" << i+2 << ".pcd"; // 给的数据是从Rf3开始
    pcl::io::loadPCDFile<PointType>(ss.str(), *cloud);
    pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    return ;
}


int main(int argc, char **argv){
    ros::init(argc, argv, "ccf_reg");
    ros::NodeHandle n;
    int select_method = ICP_METHOD;
    if(argc >= 2){
        select_method = atoi(argv[1]);
        std::cout << "select_method:" << select_method << std::endl;
    }
    // 上一时刻的点云和当前时刻的点云
    pcl::PointCloud<PointType>::Ptr cloud_l(new pcl::PointCloud<PointType>), cloud_c(new pcl::PointCloud<PointType>);
    // 拼接点云
    pcl::PointCloud<PointType>::Ptr cloud_total(new pcl::PointCloud<PointType>);
    // 临时存放点云
    pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>);
    // 坐标转换后的点云
    pcl::PointCloud<PointType>::Ptr result(new pcl::PointCloud<PointType>);

    // 定义变换矩阵 
    Eigen::Matrix4d GlobalTransform = Eigen::Matrix4d::Identity(); // 全局
    Eigen::Matrix4d icpTransform = Eigen::Matrix4d::Identity(); // 局部

    readPCD(cloud_l, 1);
    *cloud_total += *cloud_l;
    for(size_t cnt = 1; cnt < 16; cnt++){
        readPCD(cloud_c, cnt+1);
        std::cout << "cnt:" << cnt << std::endl;
        switch(select_method){
            case ICP_METHOD:
                icpAlign(cloud_l, cloud_c, temp, icpTransform, true); // point-to-point
                break;
            case VICP_METHOD:
                vicpAlign(cloud_l, cloud_c, temp, icpTransform, true); // point-to-plane
                break;
            case GICP_METHOD:
                gicpAlign(cloud_l, cloud_c, temp, icpTransform, true); // plane-to-plane
                break;
            case NDT_METHOD:
                ndtAlign(cloud_l, cloud_c, temp, icpTransform, true); // ndt
                break; 
            default:
                icpAlign(cloud_l, cloud_c, temp, icpTransform, true); // point-to-point
        }

        // 把当前两两配准后的点云temp转换到全局坐标系下返回result
        pcl::transformPointCloud(*temp, *result, GlobalTransform);

        // 用当前的两组点云之间的变换更新全局变换
        // 这里必须要使用全局变换，如果不用的化只是两帧点云之间的变换可能只是在原点处有旋转
        // 位移量旋转都没有累加上来 这里非常重要
        GlobalTransform = GlobalTransform * icpTransform;

        writePCD(result, std::to_string(cnt));

        // 点云拼接
        *cloud_total += *result;
        // 上一帧点云变成当前帧 与下一帧继续配准
        *cloud_l = *cloud_c;
    }

    switch(select_method)
    {
        case ICP_METHOD:
            writePCD(cloud_total, "reg_icp");
            break;
        case VICP_METHOD:
            writePCD(cloud_total, "reg_vicp");
            break;
        case GICP_METHOD:
            writePCD(cloud_total, "reg_gicp");
            break;
        case NDT_METHOD:
            writePCD(cloud_total, "reg_ndt");
            break;
        default:
            writePCD(cloud_total, "reg_icp");
    }

    pcl::visualization::CloudViewer viewer("pcd viewer");
    viewer.showCloud(cloud_total);
    ros::Rate loop_rate(10);
    while(ros::ok()){
        ros::spinOnce();
        loop_rate.sleep();
    }
    ros::shutdown();
    
    return 0;
}



