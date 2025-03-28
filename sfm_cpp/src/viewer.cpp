#include "viewer.h"
#include <iostream>

Viewer::Viewer() : quit_(false) {}
Viewer::~Viewer() {}

void Viewer::SetScene(const std::vector<Eigen::Vector3d>& points,
                      const std::vector<Eigen::Matrix4d>& poses,
                      const std::vector<Eigen::Vector3f>& colors) {
    std::lock_guard<std::mutex> lock(mtx_);
    points_ = points;
    poses_ = poses;
    colors_ = colors;
}

void Viewer::Run() {
    pangolin::CreateWindowAndBind("Incremental SfM Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
    );

    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetHandler(&handler);

    while(!pangolin::ShouldQuit() && !quit_) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            glPointSize(3.0f);
            glBegin(GL_POINTS);
            for(size_t i=0; i<points_.size(); ++i) {
                const auto &pt = points_[i];
                Eigen::Vector3f c(1.0f, 1.0f, 0.0f);
                if(i<colors_.size()) {
                    c = colors_[i];
                }
                glColor3f(c.x(), c.y(), c.z());
                glVertex3d(pt.x(), pt.y(), pt.z());
            }
            glEnd();

            for(auto &pose : poses_) {
                Eigen::Vector3d Ow = pose.block<3,1>(0,3);
                double size = 0.2;
                Eigen::Vector3d X = Ow + size*pose.block<3,1>(0,0);
                Eigen::Vector3d Y = Ow + size*pose.block<3,1>(0,1);
                Eigen::Vector3d Z = Ow + size*pose.block<3,1>(0,2);

                glBegin(GL_LINES);
                glColor3f(1,0,0);
                glVertex3d(Ow[0], Ow[1], Ow[2]);
                glVertex3d(X[0], X[1], X[2]);

                glColor3f(0,1,0);
                glVertex3d(Ow[0], Ow[1], Ow[2]);
                glVertex3d(Y[0], Y[1], Y[2]);

                glColor3f(0,0,1);
                glVertex3d(Ow[0], Ow[1], Ow[2]);
                glVertex3d(Z[0], Z[1], Z[2]);
                glEnd();
            }
        }

        pangolin::FinishFrame();
    }
    pangolin::DestroyWindow("Incremental SfM Viewer");
}
