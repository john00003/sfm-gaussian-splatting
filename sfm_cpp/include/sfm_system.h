#pragma once
#include <opencv2/opencv.hpp>
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
    cv::Mat descriptors;
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
};

struct Track {
    Eigen::Vector3d point;
    std::vector<std::pair<int, int>> observations;
};

struct SfMMap {
    std::map<int, View> views;
    std::vector<Track> tracks;

    void AddView(int id, const std::string& path);
    void AddObservation(int view_id, int kp_idx, int track_id);
};

bool GetIntrinsicsFromExif(const std::string& image_path, int width, int height, cv::Mat& K);

class IncrementalSfM {
public:
    explicit IncrementalSfM(SfMMap& map);
    void FilterBadPointsAfterBA(float reproj_thresh);
    void Initialize();
    void PerformInitialPair(View& v1, View& v2);
    bool RegisterNextView(int view_id);
    void TriangulateNewPoints(int view_id);
    void BundleAdjust();
    void LocalBundleAdjust(int current_view_id);

private:
    SfMMap& map_;
    cv::Ptr<cv::SIFT> sift_;
    cv::BFMatcher matcher_;
    std::vector<cv::DMatch> MatchAndFilterKNN(const cv::Mat& desc1, const cv::Mat& desc2) const;
};
