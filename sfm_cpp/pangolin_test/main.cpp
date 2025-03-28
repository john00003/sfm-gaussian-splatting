#include <pangolin/pangolin.h>
#include <Eigen/Core>

int main() {
    pangolin::CreateWindowAndBind("Minimal Pangolin Demo", 1024, 768);
    glEnable(GL_DEPTH_TEST);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 384, 0.1, 1000),
        pangolin::ModelViewLookAt(0, 0, -5, 0, 0, 0, pangolin::AxisY)
    );

    pangolin::View& d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, 0.0, 1.0)
        .SetHandler(new pangolin::Handler3D(s_cam));

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        glLineWidth(3);
        glBegin(GL_LINES);
        glColor3f(1,0,0); glVertex3f(0,0,0); glVertex3f(1,0,0);
        glColor3f(0,1,0); glVertex3f(0,0,0); glVertex3f(0,1,0);
        glColor3f(0,0,1); glVertex3f(0,0,0); glVertex3f(0,0,1);
        glEnd();

        glPointSize(5);
        glBegin(GL_POINTS);
        glColor3f(1,1,0);
        for (float x = -1.0; x <= 1.0; x += 0.1f) {
            glVertex3f(x, 0.5 * x, 0.5 * x);
        }
        glEnd();

        pangolin::FinishFrame();
    }

    return 0;
}
