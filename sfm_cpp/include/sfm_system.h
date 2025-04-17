#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <string>
#include <array>

struct View {
    int id = -1;
    bool registered = false;
    std::string image_path;
    cv::Mat image;
    cv::Mat K;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Vec3b> colors; // a parallel array with keypoints
    cv::Mat descriptors;
    std::map<int, std::vector<std::vector<cv::DMatch>>> matches_map;
    std::vector<std::pair<int, int>> points_3d;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
};

struct Track {
    Eigen::Vector3d point;
    cv::Vec3b color;
    std::vector<std::pair<int, int>> observations;
};

struct SfMMap {
    std::map<int, View> views;
    std::vector<Track> tracks;

    void AddView(int id, const std::string& path, const std::string& path_to_write);
    void AddObservation(int view_id, int kp_idx, int track_id);
};

bool GetIntrinsicsFromExif(const std::string& image_path, int width, int height, cv::Mat& K, bool johns_phone);

class IncrementalSfM {
public:
    explicit IncrementalSfM(SfMMap& map);
    void FilterBadPointsAfterBA(float reproj_thresh);
    int Initialize(bool sequential);
    void PerformInitialPair(View& v1, View& v2);
    bool RegisterNextView(int view_id);
    void TriangulateNewPoints(int view_id);
    void BundleAdjust();
    void LocalBundleAdjust(int current_view_id);
    void GenerateCOLMAPOutput();
    void WriteToBinary();
    void Write3DPoints();
    void GetPointColor(const Track& track, std::vector<cv::Mat> images, int* R_p, int* G_p, int* B_p);
    size_t MatchPair(View& v1, View& v2);
    size_t MatchViewsBF(int* best_i, int* best_j);
    size_t MatchViewsSequential(int* best_i, int* best_j);
    size_t MatchViewsWindow(int* best_i, int* best_j, int window_size, int anchor_interval, int anchor_window);

private:
    SfMMap& map_;
    cv::Ptr<cv::SIFT> sift_;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;
    std::vector<cv::DMatch> MatchAndFilterKNN(const cv::Mat& desc1, const cv::Mat& desc2, View v1, View v2) const;
};
