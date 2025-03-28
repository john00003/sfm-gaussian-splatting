#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>

struct ReprojectionError {
    ReprojectionError(double x, double y, double fx, double fy, double cx, double cy)
        : x_(x), y_(y), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const camera, const T* const point, T* residuals) const {
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T u = T(fx_) * xp + T(cx_);
        T v = T(fy_) * yp + T(cy_);

        residuals[0] = u - T(x_);
        residuals[1] = v - T(y_);
        return true;
    }

    static ceres::CostFunction* Create(double x, double y, double fx, double fy, double cx, double cy) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(x, y, fx, fy, cx, cy));
    }

    double x_, y_;
    double fx_, fy_, cx_, cy_;
};