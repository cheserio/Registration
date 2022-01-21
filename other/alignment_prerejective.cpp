#include <Eigen/Core>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/sample_consensus_prerejective.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/pcl_visualizer.h>

typedef pcl::PointNormal PointNT; // xyz + 法线 + 曲率
typedef pcl::PointCloud<PointNT> PointCloudT;
typedef pcl::FPFHSignature33 FeatureT; // fphf特征
typedef pcl::FPFHEstimationOMP<PointNT, PointNT, FeatureT> FeatureEstimationT;
typedef pcl::PointCloud<FeatureT> FeatureCloudT; // 特征点云
typedef pcl::visualization::PointCloudColorHandlerCustom<PointNT> ColorHandlerT;

int main(int argc, char **argv){
    PointCloudT::Ptr object(new PointCloudT);
    PointCloudT::Ptr object_aligned(new PointCloudT);
    PointCloudT::Ptr scene(new PointCloudT);
    FeatureCloudT::Ptr object_features(new FeatureCloudT);
    FeatureCloudT::Ptr scene_features(new FeatureCloudT);

    if(argc != 3){
        pcl::console::print_error("Syntax is: %s object.pcd scene.pcd\n", argv[0]);
        exit(1);
    }

    // 传入物体点云和场景点云
    pcl::console::print_highlight("Loading point clouds...\n");
    if(pcl::io::loadPCDFile<PointNT>(argv[1], *object) < 0 || 
       pcl::io::loadPCDFile<PointNT>(argv[2], *scene) < 0){
        pcl::console::print_error("Error loading object/scene file!\n");
        exit(1);
       }
    
    // 对读取到的点云下采样
    pcl::console::print_highlight("Downsampling...\n");
    pcl::VoxelGrid<PointNT> grid;
    const float leaf = 0.005f;
    grid.setLeafSize(leaf, leaf, leaf);
    grid.setInputCloud(object);
    grid.filter(*object);
    grid.setInputCloud(scene);
    grid.filter(*scene);


    // 估计场景scene 点云object的法线
    pcl::console::print_highlight("Estimating scene normals...\n");
    pcl::NormalEstimationOMP<PointNT, PointNT> nest; // 多线程 法线特征估计scan to map?
    nest.setRadiusSearch(0.01);

    nest.setInputCloud(scene);
    nest.compute(*scene);

    nest.setInputCloud(object);
    nest.compute(*object); 

    // 估计场景scene和物体object的fphf特征
    pcl::console::print_highlight("Estimating features...\n");
    FeatureEstimationT fest;
    fest.setRadiusSearch(0.025);
    
    fest.setInputCloud(object);
    fest.setInputNormals(object); // 输入刚刚计算的法向量点云
    fest.compute(*object_features); // 最开头定义了

    fest.setInputCloud(scene);
    fest.setInputNormals(scene);
    fest.compute(*scene_features);

    // 随机采样一致性 配准
    // RANSAC我的理解是从点云中自己拟合出特征
    pcl::console::print_highlight("Starting alignment..\n");
    pcl::SampleConsensusPrerejective<PointNT, PointNT, FeatureT> align;
    align.setInputSource(object);
    align.setSourceFeatures(object_features);
    align.setInputTarget(scene);
    align.setTargetFeatures(scene_features);
    align.setMaximumIterations(50000);
    align.setNumberOfSamples(3); // 采样点数
    align.setCorrespondenceRandomness(5); // 使用的特征数量
    align.setSimilarityThreshold(0.9f); // 相似性
    align.setMaxCorrespondenceDistance(2.5f * leaf); // 内点阈值
    align.setInlierFraction(0.25f); // 也是一个阈值
    {
        pcl::ScopeTime t("Alignment");
        align.align(*object_aligned);
    }

    if(align.hasConverged()){
        printf("\n");
        Eigen::Matrix4f transformation = align.getFinalTransformation();
        pcl::console::print_info("      | %6.3f %6.3f %6.3f | \n", transformation(0, 0), transformation(0, 1), transformation(0, 2));
        pcl::console::print_info("R =   | %6.3f %6.3f %6.3f | \n", transformation(1, 0), transformation(1, 1), transformation(1, 2));
        pcl::console::print_info("      | %6.3f %6.3f %6.3f | \n", transformation(2, 0), transformation(2, 1), transformation(2, 2));
        pcl::console::print_info("\n");
        pcl::console::print_info("t = < %0.3f, %0.3f, %0.3f >\n", transformation(0, 3), transformation(1, 3), transformation(2, 3));
        pcl::console::print_info("\n");
        pcl::console::print_info("Inliers: %i/%i\n", align.getInliers().size(), object->size());

        // 显示配准
        pcl::visualization::PCLVisualizer visu("Aligment");
        visu.addPointCloud(scene, ColorHandlerT(scene, 0.0, 255.0, 0.0), "scene");
        visu.addPointCloud(object_aligned, ColorHandlerT(object_aligned, 0.0, 0.0, 255.0), "object_aligned");
        visu.spin();
    }else{
        pcl::console::print_error("Aligment failed!\n");
        exit(1);
    }


    return 0;
}