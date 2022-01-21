/*
这是一个可视化icp程序，使用一个ply点云手动添加一个旋转和位移使其到其他位置，
使用icp将其变换回来。主函数参数 是ply文件 如果想的化还可以传入迭代次数。
通过按空格增加迭代次数
*/

#include <iostream>
#include <string.h>
#include <stdlib.h>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

bool next_iteration = false; // 声明一个全局变量在按下空格的回调函数中使用

// 将当前变换矩阵输出（在自己给定时输出一次，在完成icp配准后输出一次）
void print4x4Matrix(const Eigen::Matrix4d &matrix){
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), (1, 3), (2, 3));
}

// 通过按下空格增加迭代次数
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event,
                            void *nothing)
{
    if(event.getKeySym() == "space" && event.keyDown())
        next_iteration = true;
}

int main(int argc, char **argv){
    PointCloudT::Ptr cloud_in(new PointCloudT);
    PointCloudT::Ptr cloud_tr(new PointCloudT);
    PointCloudT::Ptr cloud_icp(new PointCloudT);

    if(argc < 2){
        printf("Usage:\n");
        printf("\t\t%s file.ply number_of_ICP_iterations\n", argv[0]);
        PCL_ERROR("Provide one ply file.\n");
        exit(1);
    }

    int iterations = 1;
    if(argc > 2){
        // 传入了迭代次数
        iterations = atoi(argv[2]);
        // 如果传入的迭代次数小于1报错
        if(iterations < 1){
            PCL_ERROR("Number of initial itertions must be >= 1\n");
            exit(1);
        }
    }

    pcl::console::TicToc time;
    time.tic();
    if(pcl::io::loadPLYFile(argv[1], *cloud_in) < 0){
        PCL_ERROR("Error loading cloud %s.\n", argv[1]);
        exit(1);
    }
    std::cout << std::endl << "Loaded file" << argv[1] << "(" << cloud_in->size() << "points)" << time.toc() << " ms\n" << std::endl;

    // 手动修改传入的点云位姿，之后将其作为源点云与输入的文件作icp配准
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    double theta = M_PI / 8;
    // 绕z轴旋转12.5度
    transformation_matrix(0, 0) = cos(theta);
    transformation_matrix(0, 1) = -sin(theta);
    transformation_matrix(1, 0) = sin(theta);
    transformation_matrix(1, 1) = cos(theta);
    // 沿z轴移动0.4
    transformation_matrix(2, 3) = 0.4;

    // 输出自己设置的变换矩阵
    std::cout << "Applying this rigid transformation to: cloud_in -> cloud_icp" << std::endl;
    print4x4Matrix(transformation_matrix);

    pcl::transformPointCloud(*cloud_in, *cloud_icp, transformation_matrix);
    *cloud_tr = *cloud_icp;

    // icp
    time.tic();
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setMaximumIterations(iterations);
    icp.setInputSource(cloud_icp);
    icp.setInputTarget(cloud_in);
    icp.align(*cloud_icp);
    icp.setMaximumIterations(1);
    std::cout << "Applied "<< iterations << " ICP iterations" << time.toc() << " ms" << std::endl;

    if(icp.hasConverged()){
        std::cout << "\nICP has converged, score is "<< icp.getFitnessScore() << std::endl;
        std::cout << "\nICP transformation "<< iterations << " :cloud_icp -> cloud_in" << std::endl;
        transformation_matrix = icp.getFinalTransformation().cast<double>();
        print4x4Matrix(transformation_matrix); 
    }else{
        PCL_ERROR("\nICP has not converged.\n");
        exit(-1);
    }

    // 可视化
    pcl::visualization::PCLVisualizer viewer("ICP demo");
    // 创建两个观察点
    int v1(0);
    int v2(1);
    viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

    // 定义显示的颜色信息
    float bckgr_gray_level = 0.0; // 黑色
    float txt_gray_level = 1.0 - bckgr_gray_level; 

    // 设置颜色参数 白色
    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_in_color_h(cloud_in,
                    (int)255 * txt_gray_level,
                    (int)255 * txt_gray_level,
                    (int)255 * txt_gray_level);

    // 将颜色参数填入
    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v1", v1);
    viewer.addPointCloud(cloud_in, cloud_in_color_h, "cloud_in_v2", v2);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_tr_color_h(cloud_tr, 20, 180, 20);
    viewer.addPointCloud(cloud_tr, cloud_tr_color_h, "cloud_tr_v1", v1);

    pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_icp_color_h(cloud_icp, 180, 20, 20);
    viewer.addPointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2", v2);

    viewer.addText("White:Original point cloud\nGreen: Matrix transformed point cloud", 10, 15, 16, txt_gray_level, txt_gray_level, txt_gray_level, "icp_info_1", v1);
    viewer.addText("White:Original point cloud\nRed: ICP aligned point cloud", 10, 15, 16, txt_gray_level, txt_gray_level, txt_gray_level, "icp_info_2", v2);

    std::stringstream ss;
    ss << iterations;
    std::string iterations_cnt = "ICP iterations = " + ss.str();
    viewer.addText(iterations_cnt, 10, 60, 16, txt_gray_level, txt_gray_level, txt_gray_level, "iterations_cnt", v2);

    // 设置背景颜色
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v1);
    viewer.setBackgroundColor(bckgr_gray_level, bckgr_gray_level, bckgr_gray_level, v2);

    // 设置相机坐标和方向
    viewer.setCameraPosition (-3.68332, 2.94092, 5.71266, 0.289847, 0.921947, -0.256907, 0);
    viewer.setSize(1280, 1024);

    // 注册按键回调函数
    viewer.registerKeyboardCallback(&keyboardEventOccurred, (void*)NULL);

    // 显示
    while(!viewer.wasStopped())
    {
        viewer.spinOnce();
        if(next_iteration){
            time.tic();
            icp.align(*cloud_icp);
            std::cout << "Applied 1 ICP iteration in " << time.toc() << " ms" << std::endl;

            if(icp.hasConverged()){
                printf("\033[11A");
                printf("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
                std::cout << "\nICP transformation " << ++iterations << " : cloud_icp -> cloud_in" << std::endl;
                transformation_matrix *= icp.getFinalTransformation().cast<double>();
                print4x4Matrix(transformation_matrix);
                ss.str("");
                ss << iterations;
                std::string iterations_cnt = "ICP iterations = " + ss.str();
                viewer.updateText(iterations_cnt, 10, 60, 16, txt_gray_level, txt_gray_level, txt_gray_level, "iterations_cnt");
                viewer.updatePointCloud(cloud_icp, cloud_icp_color_h, "cloud_icp_v2");
            }else{
                PCL_ERROR("\nICP has not converged.\n");
                exit(1);
            }
        }
        next_iteration = false;
    }
    return 0;
}

