#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

#include <pcl/range_image/range_image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/cloud_viewer.h>

#include <iostream>
#include <thread>
#include <chrono>

using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

ros::Publisher ground_points_pub_;
ros::Publisher groundless_points_pub_;
ros::Publisher map_pub_;

MatrixXf normal_;

// 参数
// 高度
double sensor_height_ = 2.0;
// 区分阈值
double th_dist_ = 0.3;
float th_dist_d_ = 0.2;

// 取z轴最低的数量
int num_lpr_ = 20;
// 种子阈值
float th_seeds_ = 1.2;
int num_iter_ = 3;

enum method { ICP_METHOD = 1, VICP_METHOD, GICP_METHOD, NDT_METHOD };
int select_method = VICP_METHOD;

typedef pcl::PointXYZI PointType; 
typedef pcl::PointNormal PointNormalT; // xyz + 法向量 + 曲率
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;

pcl::PointCloud<PointType>::Ptr g_seeds_pc(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr g_ground_pc(new pcl::PointCloud<PointType>);
pcl::PointCloud<PointType>::Ptr g_not_ground_pc(new pcl::PointCloud<PointType>);

// 点云数据
pcl::PointCloud<PointType>::Ptr cloud_l(new pcl::PointCloud<PointType>); // 上一帧点云last
pcl::PointCloud<PointType>::Ptr cloud_c(new pcl::PointCloud<PointType>); // 当前帧点云current
pcl::PointCloud<PointType>::Ptr cloud_total(new pcl::PointCloud<PointType>); // 拼接点云
pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>); // 临时存放的点云
pcl::PointCloud<PointType>::Ptr result(new pcl::PointCloud<PointType>); // 坐标转换后的点云


// 点云中点的高度对比
bool point_cmp(PointType a, PointType b){
    return a.z < b.z;
}

// 地面预测
void estimate_plane_(){
    /*
    进行平面拟合 得到法向量normal_和 th_dist_d_.
    通过全局变量传参 g_ground_pc(最大平面数据)
    */
    // 创建协方差矩阵
    // 1.计算地面点云的xyz的平均值（mean）
    float x_mean = 0, y_mean = 0, z_mean = 0;
    for(int i = 0; i < g_ground_pc->points.size(); i++){
        x_mean += g_ground_pc->points[i].x;
        y_mean += g_ground_pc->points[i].y;
        z_mean += g_ground_pc->points[i].z;
    }
    // 避免一开始没有点除0的情况
    int size = g_ground_pc->points.size() != 0?g_ground_pc->points.size():1;
    x_mean /= size;
    y_mean /= size;
    z_mean /= size;
    // 2.计算点云每个点的协方差
    // cov(x,x), cov(y,y), cov(z,z)
    // cov(x,y), cov(x,z), cov(y,z)
    float xx = 0, yy = 0, zz = 0;
    float xy = 0, xz = 0, yz = 0;
    for(int i = 0; i < g_ground_pc->points.size(); i++){
        xx += (g_ground_pc->points[i].x - x_mean) * (g_ground_pc->points[i].x - x_mean);
        xy += (g_ground_pc->points[i].x - x_mean) * (g_ground_pc->points[i].y - y_mean);
        xz += (g_ground_pc->points[i].x - x_mean) * (g_ground_pc->points[i].z - z_mean);
        yy += (g_ground_pc->points[i].y - y_mean) * (g_ground_pc->points[i].y - y_mean);
        yz += (g_ground_pc->points[i].y - y_mean) * (g_ground_pc->points[i].z - z_mean);
        zz += (g_ground_pc->points[i].z - z_mean) * (g_ground_pc->points[i].z - z_mean);
    }
    // 3.将协方差填入协方差矩阵
    MatrixXf cov(3, 3);
    cov << xx, xy, xz,
           xy, yy, yz,
           xz, yz, zz;
    cov /= size;

    // 奇异值分解SVD 这里协方差矩阵是个方阵其实就是特征分解
    // 求取PCA的主元，就是对协方差矩阵进行特征值分解，投影方向就是协方差矩阵的特征向量
    // 而投影方向上的方差，就是对应的特征值。（注意：前提是数据为0均值）
    JacobiSVD<MatrixXf> svd(cov, Eigen::DecompositionOptions::ComputeFullU);
    // 取最小的特征值对应的特征向量作为法向量
    normal_ = (svd.matrixU().col(2));
    // 平均地面种子值
    MatrixXf seeds_mean(3, 1);
    seeds_mean << x_mean, y_mean, z_mean;
    // 计算在法向量方向上的差 normal.T*[x,y,z] = -d
    // 将均值投影到法向量方向上取负号
    float d_ = -(normal_.transpose()*seeds_mean)(0, 0);
    // 这里只考虑在拟合的平面上方的点 小于这个范围的点当做地面
    th_dist_d_ = th_dist_d_ - d_;
    
    return ;
}

// 设置ROI
// 传入排序好的点云,获取地面点？
void extract_initial_seeds_(const pcl::PointCloud<PointType>& p_sorted){

    // 靠近地面的低点的平均高度值
    double sum = 0;
    int cnt = 0;
    // 取最下面20个点？
    for(int i = 0; i < p_sorted.points.size() && cnt < num_lpr_; i++){
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt != 0?sum/cnt:0; // 去除cnt=0作为除数的情况
    
    g_seeds_pc->clear();
    for(int i = 0; i < p_sorted.points.size(); i++){
        if(p_sorted.points[i].z < lpr_height + th_seeds_){
            // 如果点的z值在阈值中添加到
            g_seeds_pc->points.push_back(p_sorted.points[i]);
        }
    }
    return ;
}

// icp point-to-plane
bool vicpAlign(const pcl::PointCloud<PointType>::Ptr cloud_src, pcl::PointCloud<PointType>::Ptr cloud_tgt, pcl::PointCloud<PointType>::Ptr output, Eigen::Matrix4d &final_transform, bool downsample = false){
    // 下采样
    pcl::PointCloud<PointType>::Ptr src(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr tgt(new pcl::PointCloud<PointType>);
    pcl::VoxelGrid<PointType> grid;
    if(downsample){
        grid.setLeafSize(2.0, 2.0, 2.0);
        grid.setInputCloud(cloud_src);
        grid.filter(*src);
        grid.setInputCloud(cloud_tgt);
        grid.filter(*tgt);
    }else{
        src = cloud_src;
        tgt = cloud_tgt;
    }

    PointCloudWithNormals::Ptr points_with_normals_src(new PointCloudWithNormals);
    PointCloudWithNormals::Ptr points_with_normals_tgt(new PointCloudWithNormals);

    // 估计出带法向量的点云
    pcl::NormalEstimation<PointType, PointNormalT> norm_est;
    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>());
    norm_est.setSearchMethod(tree);
    norm_est.setKSearch(30);
    norm_est.setInputCloud(src);
    norm_est.compute(*points_with_normals_src);
    pcl::copyPointCloud(*src, *points_with_normals_src);

    norm_est.setInputCloud(tgt);
    norm_est.compute(*points_with_normals_tgt); // 计算出法向量
    pcl::copyPointCloud(*tgt, *points_with_normals_tgt); // 把点云复制过去

    pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;
    reg.setMaximumIterations(10);
    reg.setRANSACOutlierRejectionThreshold(1.0);
    reg.setTransformationEpsilon(1e-2);
    reg.setMaxCorrespondenceDistance(0.1);
    reg.setInputSource(points_with_normals_src);
    reg.setInputTarget(points_with_normals_tgt);
    reg.align(*points_with_normals_src);

    if(reg.hasConverged()){
        std::cout << "\nVICP has converged, score is " << reg.getFitnessScore() << std::endl;
        Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
        Ti = reg.getFinalTransformation().cast<double>();
        Eigen::Matrix4d targetToSource;
        targetToSource = Ti.inverse();
        pcl::transformPointCloud(*cloud_tgt, *output, targetToSource);
        final_transform = targetToSource;
        return true;
    }
    return false;
}

// 定义变换矩阵4*4
Eigen::Matrix4d GlobalTransform = Eigen::Matrix4d::Identity();
Eigen::Matrix4d icpTransform = Eigen::Matrix4d::Identity();

// 点云信息回调函数
void receive_pointsCallback(const sensor_msgs::PointCloud2ConstPtr &in_cloud_msg){
    // 1.ros pointcloud to pcl pointcloud
    pcl::PointCloud<PointType> laserCloudIn;
    pcl::fromROSMsg(*in_cloud_msg, laserCloudIn);
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);

    // 2.按照z轴大小排序
    std::sort(laserCloudIn.points.begin(), laserCloudIn.end(), point_cmp);

    // 3.移除错误点
    pcl::PointCloud<PointType>::iterator it = laserCloudIn.points.begin();
    for(int i = 0; i < laserCloudIn.points.size(); i++){
        if(laserCloudIn.points[i].z < -1.5 * sensor_height_){
            it++; 
        }else{
            break; // 直到点大于阈值后停止
        }
    }
    // 把起始位置到标记处都移除
    laserCloudIn.points.erase(laserCloudIn.points.begin(), it);
    // 4.提取初始地面种子
    extract_initial_seeds_(laserCloudIn);
    g_ground_pc = g_seeds_pc;

    // 5.提取地面
    for(int i = 0; i < num_iter_; i++){
        estimate_plane_(); // 会求出地面法向量
        g_ground_pc->clear();
        g_not_ground_pc->clear();

        // 把点云中所有点写入矩阵 N * 3
        MatrixXf points(laserCloudIn.points.size(), 3);
        int j = 0;
        for(auto p:laserCloudIn.points){
            points.row(j++) << p.x, p.y, p.z;
        }

        // 地面模型
        // 将点云中所有点都投影到法向量方向
        VectorXf result = points*normal_;

        // 按照阈值过滤出其他点 和地面点
        for(int r = 0; r < result.rows(); r++){
            if(result[r] < th_dist_d_){
                g_ground_pc->points.push_back(laserCloudIn[r]);
            }else{
                g_not_ground_pc->points.push_back(laserCloudIn[r]);
            }
        }
    }

    static int cnt = 0;
    pcl::VoxelGrid<PointType> grid;
    if(!cnt){
        // 使用非地面点来进行配准 过滤四次再配准
        *cloud_l = *g_not_ground_pc;
        grid.setLeafSize(0.4, 0.4, 0.4);
        grid.setInputCloud(cloud_l);
        grid.filter(*cloud_total);
    }else if(cnt % 5 == 0){
        *cloud_c = *g_not_ground_pc;
        std::cout << "cnt:" << cnt << std::endl;
        vicpAlign(cloud_l, cloud_c, temp, icpTransform, true);
        pcl::transformPointCloud(*temp, *result, GlobalTransform);
        // 把当前的两组点云之间的变换 变成全局变换
        GlobalTransform = GlobalTransform * icpTransform;
        // 降采样
        grid.setLeafSize(0.5, 0.5, 0.5);
        grid.setInputCloud(result);
        grid.filter(*result);
        std::cout << "GlobalTransform:\n" << GlobalTransform << std::endl;

        // 拼接点云
        *cloud_total += *result;

        *cloud_l = *cloud_c;
    }
    cnt++;
    // 地面点云转换成ros消息类型
    sensor_msgs::PointCloud2 g_ground_cloud;
    pcl::toROSMsg(*g_ground_pc, g_ground_cloud);
    g_ground_cloud.header = in_cloud_msg->header;
    ground_points_pub_.publish(g_ground_cloud);

    // 没有地面的点云
    sensor_msgs::PointCloud2 g_groundless_cloud;
    pcl::toROSMsg(*g_not_ground_pc, g_groundless_cloud);
    g_groundless_cloud.header = in_cloud_msg->header;
    groundless_points_pub_.publish(g_groundless_cloud);

    // 拼接点云地图并发布
    grid.setLeafSize(0.2, 0.2, 0.2);
    grid.setInputCloud(cloud_total);
    grid.filter(*cloud_total);
    sensor_msgs::PointCloud2 total_cloud;
    pcl::toROSMsg(*cloud_total, total_cloud);
    total_cloud.header = in_cloud_msg->header;
    map_pub_.publish(total_cloud);

}

int main(int argc, char **argv){
    ros::init(argc, argv, "lidarOdom");
    ros::NodeHandle n;
    if(argc >= 2){
        select_method = atoi(argv[1]);
        std::cout << "select_method:" << select_method << std::endl;
    } 

    // 发布话题
    ground_points_pub_ = n.advertise<sensor_msgs::PointCloud2>("/g_ground_pc", 100);
    groundless_points_pub_ = n.advertise<sensor_msgs::PointCloud2>("/g_not_ground_pc", 100);
    map_pub_ = n.advertise<sensor_msgs::PointCloud2>("/map_pc", 100);
    // 订阅话题
    ros::Subscriber receive_points_sub = n.subscribe("/rslidar_points", 100, receive_pointsCallback);

    ros::spin();
    
    return 0;
}

