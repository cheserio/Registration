#include <ros/ros.h>

//sensor_msgs
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>

//PCL
#include <pcl/range_image/range_image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <pcl/filters/voxel_grid.h>           //用于体素网格化的滤波类头文件 
#include <pcl/filters/filter.h>               //滤波相关头文件

#include <pcl/features/normal_3d.h>           //法线特征头文件

#include <pcl/registration/icp.h>             //ICP类相关头文件
#include <pcl/registration/icp_nl.h>          //非线性ICP 相关头文件
#include <pcl/registration/transforms.h>      //变换矩阵类头文件
#include <pcl/registration/gicp.h>  //gicp
#include <pcl/registration/ndt.h>      //NDT(正态分布)配准类头文件

#include <pcl/visualization/cloud_viewer.h>//点云可视化
//
#include <iostream>
#include<thread>
#include<chrono>

using Eigen::MatrixXf;
using Eigen::JacobiSVD;
using Eigen::VectorXf;

ros::Publisher ground_points_pub_;
ros::Publisher groundless_points_pub_;
ros::Publisher map_pub_;

MatrixXf normal_;
//参数：
//传感器高度
double sensor_height_= 2.0;

//区分阈值
double th_dist_= 0.3;
float th_dist_d_= 0.2;

//取z轴最低的数量
int num_lpr_= 20;
//种子阈值
float th_seeds_= 1.2;
//迭代次数
int num_iter_= 3;

enum method { ICP_METHOD = 1, VICP_METHOD, GICP_METHOD, NDT_METHOD };
int select_method = VICP_METHOD;

typedef pcl::PointXYZI PointType;
typedef pcl::PointNormal PointNormalT; 

typedef pcl::PointNormal PointNormalT;     //　x,y,z＋法向量＋曲率　点
typedef pcl::PointCloud<PointNormalT> PointCloudWithNormals;//带有法向量的点云

pcl::PointCloud<PointType>::Ptr g_seeds_pc(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_ground_pc(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr g_not_ground_pc(new pcl::PointCloud<PointType>());

// 点云数据
pcl::PointCloud<PointType>::Ptr cloud_l(new pcl::PointCloud<PointType>), cloud_c(new pcl::PointCloud<PointType>);//上一时刻的点云，下一时刻的点云
pcl::PointCloud<PointType>::Ptr cloud_total(new pcl::PointCloud<PointType>);//点云拼接
pcl::PointCloud<PointType>::Ptr temp (new pcl::PointCloud<PointType>);//临时存放点云
pcl::PointCloud<PointType>::Ptr result (new pcl::PointCloud<PointType>);//坐标转换后的点云

bool point_cmp(PointType a, PointType b)
{
    return a.z<b.z;
}


void  estimate_plane_()
{
    // Create covarian matrix.
    // 1. calculate (x,y,z) mean
    float x_mean = 0, y_mean = 0, z_mean = 0;
    for(int i=0;i<g_ground_pc->points.size();i++){
        x_mean += g_ground_pc->points[i].x;
        y_mean += g_ground_pc->points[i].y;
        z_mean += g_ground_pc->points[i].z;
    }
    // incase of divide zero
    int size = g_ground_pc->points.size()!=0?g_ground_pc->points.size():1;
    x_mean /= size;
    y_mean /= size;
    z_mean /= size;
    // 2. calculate covariance
    // cov(x,x), cov(y,y), cov(z,z)
    // cov(x,y), cov(x,z), cov(y,z)
    float xx = 0, yy = 0, zz = 0;
    float xy = 0, xz = 0, yz = 0;
    for(int i=0;i<g_ground_pc->points.size();i++){
        xx += (g_ground_pc->points[i].x-x_mean)*(g_ground_pc->points[i].x-x_mean);
        xy += (g_ground_pc->points[i].x-x_mean)*(g_ground_pc->points[i].y-y_mean);
        xz += (g_ground_pc->points[i].x-x_mean)*(g_ground_pc->points[i].z-z_mean);
        yy += (g_ground_pc->points[i].y-y_mean)*(g_ground_pc->points[i].y-y_mean);
        yz += (g_ground_pc->points[i].y-y_mean)*(g_ground_pc->points[i].z-z_mean);
        zz += (g_ground_pc->points[i].z-z_mean)*(g_ground_pc->points[i].z-z_mean);
    }
    // 3. setup covarian matrix cov
    MatrixXf cov(3,3);
    cov << xx,xy,xz,
           xy, yy, yz,
           xz, yz, zz;
    cov /= size;
    // Singular Value Decomposition: SVD
    JacobiSVD<MatrixXf> svd(cov,Eigen::DecompositionOptions::ComputeFullU);
    // use the least singular vector as normal
    normal_ = (svd.matrixU().col(2));
    // mean ground seeds value
    MatrixXf seeds_mean(3,1);
    seeds_mean<<x_mean,y_mean,z_mean;
    // according to normal.T*[x,y,z] = -d
    float d_ = -(normal_.transpose()*seeds_mean)(0,0);
    // set distance threhold to `th_dist - d`
    th_dist_d_ = th_dist_ - d_;
	
 
    // return the equation parameters
}
void extract_initial_seeds_(const pcl::PointCloud<PointType>& p_sorted)
{
    //the mean of low point representative
    double sum = 0;
    int cnt = 0;
    // Calculate the mean height value.
    for(int i=0;i<p_sorted.points.size() && cnt<num_lpr_;i++){
        sum += p_sorted.points[i].z;
        cnt++;
    }
    double lpr_height = cnt!=0?sum/cnt:0;// in case divide by 0
    g_seeds_pc->clear();
    // iterate pointcloud, filter those height is less than lpr.height+th_seeds_
    for(int i=0;i<p_sorted.points.size();i++){
        if(p_sorted.points[i].z < lpr_height + th_seeds_){
            g_seeds_pc->points.push_back(p_sorted.points[i]);
        }
    }
    // return seeds points
}


//icp变体 point-to-plane
bool vicpAlign (const pcl::PointCloud<PointType>::Ptr cloud_src, pcl::PointCloud<PointType>::Ptr cloud_tgt, pcl::PointCloud<PointType>::Ptr output, Eigen::Matrix4d &final_transform, bool downsample = false)
{
	
	// 	Downsample for consistency and speed
	pcl::PointCloud<PointType>::Ptr src (new pcl::PointCloud<PointType>);//存储滤波后的源点云
	pcl::PointCloud<PointType>::Ptr tgt (new pcl::PointCloud<PointType>);//存储滤波后的目标点云
	pcl::VoxelGrid<PointType> grid;         //体素格滤波器 滤波处理对象
	if (downsample)
	{
		grid.setLeafSize (2.0, 2.0, 2.0);//设置滤波时采用的体素大小
		grid.setInputCloud (cloud_src);
		grid.filter (*src);

		grid.setInputCloud (cloud_tgt);
		grid.filter (*tgt);
	}
	else
	{
		src = cloud_src;
		tgt = cloud_tgt;
	}
	
	// 计算表面的法向量和曲率
	PointCloudWithNormals::Ptr points_with_normals_src (new PointCloudWithNormals);
	PointCloudWithNormals::Ptr points_with_normals_tgt (new PointCloudWithNormals);
	
	pcl::NormalEstimation<PointType, PointNormalT> norm_est;//点云法线估计对象
	// 添加搜索算法 kdtree search  最近的几个点 估计平面 协方差矩阵PCA分解 求解法线
	pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType> ());
	norm_est.setSearchMethod (tree);
	norm_est.setKSearch (30);// 指定临近点数量

	norm_est.setInputCloud (src);//全部的点云
	norm_est.compute (*points_with_normals_src);// 计算表面法线特征
	pcl::copyPointCloud (*src, *points_with_normals_src);
	// 不同类型的点云之间进行类型转换 pcl::copyPointClou

	norm_est.setInputCloud (tgt);
	norm_est.compute (*points_with_normals_tgt);
	pcl::copyPointCloud (*tgt, *points_with_normals_tgt);
	
	
	pcl::IterativeClosestPointNonLinear<PointNormalT, PointNormalT> reg;   // 配准对象
	reg.setMaximumIterations (10);//设置最大的迭代次数
	reg.setRANSACOutlierRejectionThreshold(1.0);
	reg.setTransformationEpsilon (1e-2);   ///设置收敛判断条件，越小精度越大，收敛也越慢 
	reg.setMaxCorrespondenceDistance (0.1);// 10cm大于此值的点对不考虑
	reg.setInputSource (points_with_normals_src);   // 设置源点云
	reg.setInputTarget (points_with_normals_tgt);   // 设置目标点云
	reg.align (*points_with_normals_src);//匹配后源点云
	
	
	if (reg.hasConverged ())
	{
		std::cout << "\nVICP has converged, score is " << reg.getFitnessScore () << std::endl;
		Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity (), targetToSource;
		Ti = reg.getFinalTransformation ().cast<double>();
		targetToSource = Ti.inverse();
		pcl::transformPointCloud (*cloud_tgt, *output, targetToSource);
		
		final_transform = targetToSource;

		return true;
	}
	return false;
}


//  定义旋转矩阵和平移向量Matrix4d是为4*4的矩阵
Eigen::Matrix4d GlobalTransform = Eigen::Matrix4d::Identity ();
Eigen::Matrix4d icpTransform = Eigen::Matrix4d::Identity ();

//回调函数
void receive_pointsCallback(const sensor_msgs::PointCloud2ConstPtr& in_cloud_msg)
{
	
	// 1.Msg to pointcloud
	pcl::PointCloud<PointType> laserCloudIn;
	pcl::fromROSMsg(*in_cloud_msg, laserCloudIn);
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn,indices);
	
	// 2.Sort on Z-axis value.
	std::sort(laserCloudIn.points.begin(),laserCloudIn.end(),point_cmp);
	
	// 3.Error point removal
	pcl::PointCloud<PointType>::iterator it = laserCloudIn.points.begin();
	for(int i=0;i<laserCloudIn.points.size();i++)
	{
		if(laserCloudIn.points[i].z < -1.5*sensor_height_)
		{
			it++;
		}
		else
		{
			break;
		}
	}
	laserCloudIn.points.erase(laserCloudIn.points.begin(),it);
	// 4. Extract init ground seeds.
	extract_initial_seeds_(laserCloudIn);
	g_ground_pc = g_seeds_pc;

	// 5. Ground plane fitter mainloop
	for(int i=0;i<num_iter_;i++)
	{
		estimate_plane_();
		g_ground_pc->clear();
		g_not_ground_pc->clear();

		//pointcloud to matrix
		MatrixXf points(laserCloudIn.points.size(),3);
		int j =0;
		for(auto p:laserCloudIn.points)
		{
			points.row(j++)<<p.x,p.y,p.z;
		}
		
		// ground plane model
		VectorXf result = points*normal_;
		
		// threshold filter
		for(int r=0;r<result.rows();r++)
		{
			if(result[r]<th_dist_d_)
			{
				g_ground_pc->points.push_back(laserCloudIn[r]);
			}
			else
			{
				g_not_ground_pc->points.push_back(laserCloudIn[r]);
			}
		}
	}
	
	static int cnt = 0;
	pcl::VoxelGrid<PointType> grid;         //体素
	
	if(!cnt)
	{
		*cloud_l = *g_not_ground_pc;
		
		grid.setLeafSize (0.4, 0.4, 0.4);//设置滤波时采用的体素大小 
		grid.setInputCloud (cloud_l);
		grid.filter (*cloud_total);

	}
	else if(cnt%5==0)
	{
		*cloud_c = *g_not_ground_pc;
		std::cout << "cnt:" << cnt << std::endl;
		// icpTransform返回从目标点云cloud_c到cloud_l的变换矩阵
		vicpAlign(cloud_l, cloud_c, temp, icpTransform,true);//point-to-plane
		
		//把当前两两配准后的点云temp转化到全局坐标系下返回result
		pcl::transformPointCloud (*temp, *result, GlobalTransform);

		//用当前的两组点云之间的变换更新全局变换
		GlobalTransform = GlobalTransform * icpTransform;

		//降采样
		grid.setLeafSize (0.5, 0.5, 0.5);//设置滤波时采用的体素大小 
		grid.setInputCloud (result);
		grid.filter (*result);
		std::cout<< "GlobalTransform:\n"<<GlobalTransform<<std::endl;
		//点云拼接
		*cloud_total += *result;
		
		*cloud_l = *cloud_c;
	}
	cnt++;
	//地面的点云
	sensor_msgs::PointCloud2 g_ground_cloud;
	pcl::toROSMsg(*g_ground_pc,g_ground_cloud);
	g_ground_cloud.header = in_cloud_msg->header;//时间戳，id
	ground_points_pub_.publish(g_ground_cloud);
	
	//没有地面的点云
	sensor_msgs::PointCloud2 g_groundless_cloud;
	pcl::toROSMsg(*g_not_ground_pc,g_groundless_cloud);
	g_groundless_cloud.header = in_cloud_msg->header;//时间戳，id
	groundless_points_pub_.publish(g_groundless_cloud);
	
	//拼接点云地图并发布
	grid.setLeafSize (0.2, 0.2, 0.2);//设置滤波时采用的体素大小 
	grid.setInputCloud (cloud_total);
	grid.filter (*cloud_total);
	sensor_msgs::PointCloud2 total_cloud;
	pcl::toROSMsg(*cloud_total,total_cloud);
	total_cloud.header = in_cloud_msg->header;//时间戳，id
	map_pub_.publish(total_cloud);

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "lidarodom");
    ros::NodeHandle n;
	
	if(argc>=2)
	{
		select_method = atoi(argv[1]);
		std::cout <<"select_method:"<< select_method <<std::endl;
	}

	//发布话题
	ground_points_pub_ = n.advertise<sensor_msgs::PointCloud2>("/g_ground_pc",100);
	groundless_points_pub_ = n.advertise<sensor_msgs::PointCloud2>("/g_not_ground_pc",100);
	map_pub_ = n.advertise<sensor_msgs::PointCloud2>("/map_pc",100);
	
	//订阅话题
	ros::Subscriber receive_points_sub = n.subscribe("/rslidar_points",100,receive_pointsCallback);

	ros::spin();

    return 0;
}