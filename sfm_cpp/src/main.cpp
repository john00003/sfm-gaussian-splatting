#define GLEW_STATIC
#include <GL/glew.h>
#include <pangolin/pangolin.h>
#include "gui.h"
#include "viewer.h"
#include "sfm_system.h"

#include <iostream>
#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <exiv2/exiv2.hpp>
#include <filesystem>
#include <Eigen/Dense>

namespace fs = std::filesystem;

bool GetIntrinsicsFromExif(const std::string& image_path, int width, int height, cv::Mat& K);

void runSfMOnly(const std::string& folder, std::vector<Eigen::Matrix4d>& poses, std::vector<Eigen::Vector3d>& points3D, std::vector<Eigen::Vector3f>& colors) {
    std::vector<std::string> image_paths;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
            image_paths.push_back(entry.path().string());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());
    if (image_paths.size() < 2) {
        std::cerr << "[ERROR] Need at least two images." << std::endl;
        return;
    }

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> Ks;
    std::vector<std::vector<cv::KeyPoint>> all_kps;
    std::vector<cv::Mat> all_des;

    auto sift = cv::SIFT::create();
    cv::BFMatcher matcher(cv::NORM_L2);

    for (const auto& path : image_paths) {
        cv::Mat img = cv::imread(path);
        if (img.empty()) continue;
        images.push_back(img);
        cv::Mat K;
        if (!GetIntrinsicsFromExif(path, img.cols, img.rows, K)) {
            std::cerr << "[WARN] Missing EXIF in: " << path << std::endl;
            continue;
        }
        Ks.push_back(K);
        std::vector<cv::KeyPoint> kps;
        cv::Mat des;
        sift->detectAndCompute(img, cv::noArray(), kps, des);
        all_kps.push_back(kps);
        all_des.push_back(des);
    }

    if (images.size() < 2) return;

    std::cout << "[INFO] Matching first image pair..." << std::endl;
    std::vector<cv::DMatch> good_matches;
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(all_des[0], all_des[1], knn, 2);
    for (auto& m : knn) {
        if (m[0].distance < 0.75f * m[1].distance)
            good_matches.push_back(m[0]);
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (auto& m : good_matches) {
        pts1.push_back(all_kps[0][m.queryIdx].pt);
        pts2.push_back(all_kps[1][m.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, Ks[0], cv::RANSAC, 0.999, 1.0, mask);
    cv::Mat R, t;
    cv::recoverPose(E, pts1, pts2, Ks[0], R, t, mask);

    std::cout << "[INFO] Triangulating initial 3D points..." << std::endl;

    cv::Mat P1 = Ks[0] * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P2 = Ks[1] * Rt;
    std::vector<cv::Point2f> inliers1, inliers2;
    for (int i = 0; i < mask.rows; ++i) {
        if (mask.at<uchar>(i)) {
            inliers1.push_back(pts1[i]);
            inliers2.push_back(pts2[i]);
        }
    }
    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, inliers1, inliers2, points4D);

    for (int i = 0; i < points4D.cols; ++i) {
        cv::Mat x = points4D.col(i);
        x /= x.at<float>(3);
        points3D.emplace_back(x.at<float>(0), x.at<float>(1), x.at<float>(2));

        auto pt = inliers1[i];
        const cv::Vec3b& bgr = images[0].at<cv::Vec3b>(cvRound(pt.y), cvRound(pt.x));
        Eigen::Vector3f color(bgr[2] / 255.0f, bgr[1] / 255.0f, bgr[0] / 255.0f);
        colors.push_back(color);
    }

    poses.emplace_back(Eigen::Matrix4d::Identity());
    Eigen::Matrix4d pose2 = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d R_e;
    Eigen::Vector3d t_e;
    cv::cv2eigen(R, R_e);
    cv::cv2eigen(t, t_e);
    pose2.block<3,3>(0,0) = R_e;
    pose2.block<3,1>(0,3) = t_e;
    poses.push_back(pose2);
}

void runSfMOnly(char *folder) {
    SfMMap map;
    std::vector<Eigen::Vector3d> current_points;
    std::vector<Eigen::Matrix4d> current_poses;
    std::vector<Eigen::Vector3f> current_colors;

    int registered_since_last_ba = 0;

    std::cout << "[INFO] SfM started on folder: " << folder << std::endl;

    {
        map.views.clear();
        map.tracks.clear();
        current_points.clear();
        current_poses.clear();
        current_colors.clear();
    }

    int id = 0;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
            map.AddView(id, entry.path().string(), entry.path().filename().string());
            auto& view = map.views[id];
            if (!GetIntrinsicsFromExif(entry.path().string(), view.image.cols, view.image.rows, view.K)) {
                std::cerr << "[WARN] Missing EXIF intrinsics for " << entry.path() << std::endl;
            }
            else {
                std::cout << "[INFO] Intrinsics for view " << id << ":\n" << view.K << std::endl;
            }
            id++;
        }
    }

    if (map.views.size() < 2) {
        std::cerr << "[ERROR] Need at least two images to run SfM." << std::endl;
        return;
    }

    IncrementalSfM sfm(map);
    std::cout << "[INFO] Initializing SfM..." << std::endl;
    sfm.Initialize();

    for (int i = 2; i < (int)map.views.size(); ++i) {

        if (map.views[i].registered) continue;

        std::cout << "[INFO] Registering view " << i << std::endl;
        bool reg_ok = sfm.RegisterNextView(i);
        if (!reg_ok) {
            std::cerr << "[WARN] Failed to register view " << i << std::endl;
            continue;
        }
        std::cout << "[INFO] Triangulating new points for view " << i << std::endl;
        sfm.TriangulateNewPoints(i);

        registered_since_last_ba++;
        if (registered_since_last_ba >= 3) {
            std::cout << "[INFO] Running local bundle adjustment..." << std::endl;
            sfm.LocalBundleAdjust(i);
            registered_since_last_ba = 0;
        }
    }

    std::cout << "[INFO] Running global bundle adjustment..." << std::endl;
    sfm.BundleAdjust();

    {
        current_points.clear();
        current_poses.clear();
        current_colors.clear();

        for (const auto& t : map.tracks) {
            current_points.push_back(t.point);
            if (!t.observations.empty()) {
                const auto& obs = t.observations.front();
                const auto& view = map.views[obs.first];
                const auto& kp = view.keypoints[obs.second];
                const cv::Vec3b& bgr = view.image.at<cv::Vec3b>(cvRound(kp.pt.y), cvRound(kp.pt.x));
                current_colors.emplace_back(bgr[2] / 255.f, bgr[1] / 255.f, bgr[0] / 255.f);
            }
            else {
                current_colors.emplace_back(1.0f, 1.0f, 0.0f);
            }
        }

        for (auto& kv : map.views) {
            const auto& v = kv.second;
            if (v.registered) {
                current_poses.push_back(v.pose);
            }
        }

        std::cout << "[INFO] Visualization ready with "
            << current_points.size() << " points and "
            << current_poses.size() << " camera poses." << std::endl;
    }

    sfm.GenerateCOLMAPOutput();


}

int main(int argc, char** argv)
{

    if (argc > 1) {
        if (strcmp(argv[1], "--no_gui") == 0) {
            if (argc == 3) {
                runSfMOnly(argv[2]);
                return 0;
            }
            else {
                std::cerr << "Must provide folder with --no_gui flag! Exiting.\n";
                return -1;
            }
            
        }
        else {
            std::cerr << "First argument must be --no_gui! Exiting.\n";
            return -1;
        }
    }

    GUIManager gui;
    std::cout << "[DEBUG] Initializing GUI..." << std::endl;
    gui.Init();
    std::cout << "[DEBUG] GUI initialized." << std::endl;

    std::atomic<bool> running(false);
    std::atomic<bool> viewer_pending(false);
    std::mutex result_mutex;

    SfMMap map;
    std::vector<Eigen::Vector3d> current_points;
    std::vector<Eigen::Matrix4d> current_poses;
    std::vector<Eigen::Vector3f> current_colors;

    Viewer viewer;
    int registered_since_last_ba = 0;

    while (!gui.ShouldQuit()) {
        gui.Render([&](const std::string& folder) {
            if (!running) {
                running = true;
                std::thread([&, folder]() {
                    std::cout << "[INFO] SfM thread started on folder: " << folder << std::endl;

                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        map.views.clear();
                        map.tracks.clear();
                        current_points.clear();
                        current_poses.clear();
                        current_colors.clear();
                    }

                    int id = 0;
                    for (const auto& entry : fs::directory_iterator(folder)) {
                        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
                            map.AddView(id, entry.path().string(), entry.path().filename().string());
                            auto& view = map.views[id];
                            if (!GetIntrinsicsFromExif(entry.path().string(), view.image.cols, view.image.rows, view.K)) {
                                std::cerr << "[WARN] Missing EXIF intrinsics for " << entry.path() << std::endl;
                            } else {
                                std::cout << "[INFO] Intrinsics for view " << id << ":\n" << view.K << std::endl;
                            }
                            id++;
                        }
                    }

                    if (map.views.size() < 2) {
                        std::cerr << "[ERROR] Need at least two images to run SfM." << std::endl;
                        running = false;
                        return;
                    }

                    IncrementalSfM sfm(map);
                    std::cout << "[INFO] Initializing SfM..." << std::endl;
                    sfm.Initialize();

                    for (int i = 2; i < (int)map.views.size(); ++i) {

                        if (map.views[i].registered) continue;

                        std::cout << "[INFO] Registering view " << i << std::endl;
                        bool reg_ok = sfm.RegisterNextView(i);
                        if (!reg_ok) {
                            std::cerr << "[WARN] Failed to register view " << i << std::endl;
                            continue;
                        }
                        std::cout << "[INFO] Triangulating new points for view " << i << std::endl;
                        sfm.TriangulateNewPoints(i);

                        registered_since_last_ba++;
                        if (registered_since_last_ba >= 3) {
                            std::cout << "[INFO] Running local bundle adjustment..." << std::endl;
                            sfm.LocalBundleAdjust(i);
                            registered_since_last_ba = 0;
                        }
                    }

                    std::cout << "[INFO] Running global bundle adjustment..." << std::endl;
                    sfm.BundleAdjust();

                    {
                        std::lock_guard<std::mutex> lock(result_mutex);
                        current_points.clear();
                        current_poses.clear();
                        current_colors.clear();

                        for (const auto& t : map.tracks) {
                            current_points.push_back(t.point);
                            if (!t.observations.empty()) {
                                const auto& obs = t.observations.front();
                                const auto& view = map.views[obs.first];
                                const auto& kp = view.keypoints[obs.second];
                                const cv::Vec3b& bgr = view.image.at<cv::Vec3b>(cvRound(kp.pt.y), cvRound(kp.pt.x));
                                current_colors.emplace_back(bgr[2] / 255.f, bgr[1] / 255.f, bgr[0] / 255.f);
                            } else {
                                current_colors.emplace_back(1.0f, 1.0f, 0.0f);
                            }
                        }

                        for (auto& kv : map.views) {
                            const auto& v = kv.second;
                            if (v.registered) {
                                current_poses.push_back(v.pose);
                            }
                        }

                        std::cout << "[INFO] Visualization ready with "
                                  << current_points.size() << " points and "
                                  << current_poses.size() << " camera poses." << std::endl;
                    }

                    sfm.GenerateCOLMAPOutput();

                    viewer_pending = true;
                    running = false;
                }).detach();
            }
        });

        if (viewer_pending) {
            std::lock_guard<std::mutex> lock(result_mutex);
            viewer.SetScene(current_points, current_poses, current_colors);
            viewer.Run();
            viewer_pending = false;
        }
    }

    gui.Shutdown();
    return 0;
}
