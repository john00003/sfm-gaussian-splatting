#pragma once
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <mutex>
#include <vector>

class Viewer {
public:
    Viewer();
    ~Viewer();

    void SetScene(const std::vector<Eigen::Vector3d>& points,
                  const std::vector<Eigen::Matrix4d>& poses,
                  const std::vector<Eigen::Vector3f>& colors);

    void Run();
    void Stop() { quit_ = true; }

private:
    std::mutex mtx_;
    std::vector<Eigen::Vector3d> points_;
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<Eigen::Vector3f> colors_;
    bool quit_;
};
