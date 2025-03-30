#include "sfm_system.h"

#include <exiv2/exiv2.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <set>
#include <iostream>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <fstream>

bool GetIntrinsicsFromExif(const std::string& image_path, int width, int height, cv::Mat& K) {
    try {
        auto image = Exiv2::ImageFactory::open(image_path);
        image->readMetadata();
        Exiv2::ExifData &exifData = image->exifData();
        if (exifData.empty()) {
            return false;
        }

        float focal_mm  = 0.f;
        float focal_35  = 0.f;
        if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.FocalLength")) != exifData.end()) {
            focal_mm = exifData["Exif.Photo.FocalLength"].toFloat();
        }
        if (exifData.findKey(Exiv2::ExifKey("Exif.Photo.FocalLengthIn35mmFilm")) != exifData.end()) {
            focal_35 = exifData["Exif.Photo.FocalLengthIn35mmFilm"].toFloat();
        }

        if (focal_mm <= 1e-5 || focal_35 <= 1e-5) {
            return false;
        }

        float sensor_w = (focal_mm / focal_35) * 36.0f;
        float sensor_h = sensor_w * 0.75f; 

        float fx = (focal_mm / sensor_w) * width;
        float fy = (focal_mm / sensor_h) * height;
        float cx = width / 2.0f;
        float cy = height / 2.0f;

        K = (cv::Mat_<double>(3,3) << fx, 0, cx,
                                      0, fy, cy,
                                      0, 0, 1);
        return true;
    } catch (...) {
        return false;
    }
}

void SfMMap::AddView(int id, const std::string& path) {
    View view;
    view.id = id;
    view.image_path = path;
    view.image = cv::imread(path);
    if(view.image.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << path << std::endl;
    }
    views[id] = view;
}

void SfMMap::AddObservation(int view_id, int kp_idx, int track_id) {
    tracks[track_id].observations.emplace_back(view_id, kp_idx);
}

struct ReprojError {
    ReprojError(double u, double v, const Eigen::Matrix3d& K)
        : u_(u), v_(v), K_(K) {}

    template<typename T>
    bool operator()(const T* const cam, const T* const point, T* residuals) const 
    {
        const T* rot = cam;
        const T* tran = cam + 3;

        T p[3];
        ceres::AngleAxisRotatePoint(rot, point, p);
        p[0] += tran[0];
        p[1] += tran[1];
        p[2] += tran[2];

        if (p[2] < T(1e-6)) {
            residuals[0] = T(1000.0);
            residuals[1] = T(1000.0);
            return true;
        }

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        T fx = T(K_(0,0)), fy = T(K_(1,1));
        T cx = T(K_(0,2)), cy = T(K_(1,2));

        T u_proj = fx * xp + cx;
        T v_proj = fy * yp + cy;

        residuals[0] = u_proj - T(u_);
        residuals[1] = v_proj - T(v_);

        return true;
    }

    double u_, v_;
    Eigen::Matrix3d K_;
};

IncrementalSfM::IncrementalSfM(SfMMap& map)
    : map_(map)
{
    sift_ = cv::SIFT::create();
    matcher_ = cv::BFMatcher(cv::NORM_L2);
}

bool IsGoodTriangulatedPoint(const cv::Point2f& pt1, const cv::Point2f& pt2,
                              const cv::Mat& P1, const cv::Mat& P2,
                              const cv::Point3d& point3d, float reproj_thresh=3.0f)
{
    cv::Mat pt_h = (cv::Mat_<double>(4,1) << point3d.x, point3d.y, point3d.z, 1.0);
    cv::Mat proj1 = P1 * pt_h;
    cv::Mat proj2 = P2 * pt_h;
    cv::Point2f re1(proj1.at<double>(0)/proj1.at<double>(2),
                    proj1.at<double>(1)/proj1.at<double>(2));
    cv::Point2f re2(proj2.at<double>(0)/proj2.at<double>(2),
                    proj2.at<double>(1)/proj2.at<double>(2));
    float err1 = cv::norm(re1 - pt1);
    float err2 = cv::norm(re2 - pt2);
    return err1 < reproj_thresh && err2 < reproj_thresh;
}

void IncrementalSfM::FilterBadPointsAfterBA(float reproj_thresh) {
    std::vector<Track> filtered;
    for (const auto& track : map_.tracks) {
        bool good = true;
        for (const auto& obs : track.observations) {
            const auto& view = map_.views.at(obs.first);
            if (!view.registered) continue;

            Eigen::Vector3d pt_cam = view.pose.block<3,3>(0,0) * track.point + view.pose.block<3,1>(0,3);
            if (pt_cam[2] <= 0.0) {
                good = false;
                break;
            }
            double u_proj = view.K.at<double>(0,0) * pt_cam[0] / pt_cam[2] + view.K.at<double>(0,2);
            double v_proj = view.K.at<double>(1,1) * pt_cam[1] / pt_cam[2] + view.K.at<double>(1,2);
            cv::Point2f proj(u_proj, v_proj);
            cv::Point2f obs_pt = view.keypoints[obs.second].pt;
            double err = cv::norm(proj - obs_pt);
            if (err > reproj_thresh) {
                good = false;
                break;
            }
        }
        if (good) filtered.push_back(track);
    }
    std::cout << "[INFO] Removed " << (map_.tracks.size() - filtered.size()) << " bad 3D points after BA." << std::endl;
    map_.tracks = std::move(filtered);
}

void IncrementalSfM::Initialize() {
    if (map_.views.size() < 2) return;

    int best_i = -1, best_j = -1;
    size_t best_matches = 0;

    for (auto& [id, view] : map_.views) {
        if (view.keypoints.empty()) {
            sift_->detectAndCompute(view.image, cv::noArray(), view.keypoints, view.descriptors);
        }
    }

    for (auto it1 = map_.views.begin(); it1 != map_.views.end(); ++it1) {
        for (auto it2 = std::next(it1); it2 != map_.views.end(); ++it2) {
            const auto& v1 = it1->second;
            const auto& v2 = it2->second;

            std::vector<std::vector<cv::DMatch>> knn;
            matcher_.knnMatch(v1.descriptors, v2.descriptors, knn, 2);

            size_t good = 0;
            for (const auto& m : knn) {
                if (m.size() < 2) continue;
                if (m[0].distance < 0.75f * m[1].distance) ++good;
            }

            if (good > best_matches) {
                best_matches = good;
                best_i = v1.id;
                best_j = v2.id;
            }
        }
    }

    if (best_i == -1 || best_j == -1) {
        std::cerr << "[ERROR] Failed to find suitable initial image pair." << std::endl;
        return;
    }

    std::cout << "[INFO] Initialize using best pair: view " << best_i << " and view " << best_j
              << " with " << best_matches << " matches." << std::endl;

    PerformInitialPair(map_.views[best_i], map_.views[best_j]);
}

void IncrementalSfM::PerformInitialPair(View& v1, View& v2) {
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_.knnMatch(v1.descriptors, v2.descriptors, knn, 2);

    std::vector<cv::DMatch> good;
    for (auto& m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.75f * m[1].distance)
            good.push_back(m[0]);
    }

    if (good.size() < 30) {
        std::cerr << "[WARN] Too few matches for initialization." << std::endl;
        return;
    }

    std::vector<cv::Point2f> pts1, pts2;
    for (auto& dm : good) {
        pts1.push_back(v1.keypoints[dm.queryIdx].pt);
        pts2.push_back(v2.keypoints[dm.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, v1.K, cv::RANSAC, 0.999, 1.0, mask);
    if (E.empty()) {
        std::cerr << "[ERROR] Essential matrix estimation failed." << std::endl;
        return;
    }

    cv::Mat R, t;
    if (cv::recoverPose(E, pts1, pts2, v1.K, R, t, mask) < 10) {
        std::cerr << "[WARN] Too few inliers in recoverPose." << std::endl;
        return;
    }

    v1.pose = Eigen::Matrix4d::Identity();
    v1.registered = true;

    Eigen::Matrix3d Re;
    Eigen::Vector3d te;
    cv::cv2eigen(R, Re);
    cv::cv2eigen(t, te);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = Re;
    T.block<3,1>(0,3) = te;
    v2.pose = T;
    v2.registered = true;

    cv::Mat P1 = v1.K * cv::Mat::eye(3,4,CV_64F);
    cv::Mat Rt2;
    cv::Mat R2_cv, t2_cv;
    cv::eigen2cv(Re, R2_cv);
    cv::eigen2cv(te, t2_cv);
    cv::hconcat(R2_cv, t2_cv, Rt2);
    cv::Mat P2 = v2.K * Rt2;

    std::vector<cv::Point2f> inliers1, inliers2;
    for (int i = 0; i < (int)pts1.size(); ++i) {
        if (mask.at<uchar>(i)) {
            inliers1.push_back(pts1[i]);
            inliers2.push_back(pts2[i]);
        }
    }

    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, inliers1, inliers2, pts4D);

    int valid = 0;
    for (int i = 0; i < pts4D.cols; ++i) {
        cv::Mat x = pts4D.col(i);
        x /= x.at<float>(3);
        float X = x.at<float>(0);
        float Y = x.at<float>(1);
        float Z = x.at<float>(2);
        cv::Point3d pt3d(X, Y, Z);
        
        if (Z < 0.001 || Z > 1e6 || 
            !IsGoodTriangulatedPoint(inliers1[i], inliers2[i], P1, P2, pt3d, 3.0f)) {
            continue;
        }        

        Track track;
        track.point = Eigen::Vector3d(x.at<float>(0), x.at<float>(1), Z);

        for (auto& dm : good) {
            if (v1.keypoints[dm.queryIdx].pt == inliers1[i] &&
                v2.keypoints[dm.trainIdx].pt == inliers2[i]) {
                track.observations.push_back({v1.id, dm.queryIdx});
                track.observations.push_back({v2.id, dm.trainIdx});
                map_.tracks.push_back(track);
                ++valid;
                break;
            }
        }
    }
    std::cout << "[DEBUG] Initial translation norm: " << te.norm() << std::endl;

    std::cout << "[INFO] Triangulated " << valid
              << " points between view " << v1.id << " and " << v2.id << std::endl;
}

std::vector<cv::DMatch> IncrementalSfM::MatchAndFilterKNN(
    const cv::Mat& desc1, const cv::Mat& desc2) const
{
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_.knnMatch(desc1, desc2, knn, 2);

    std::vector<cv::DMatch> good;
    good.reserve(knn.size());
    for (const auto& m : knn) {
        if (m.size() < 2) continue;
        if (m[0].distance < 0.75f * m[1].distance)
            good.push_back(m[0]);
    }
    return good;
}

bool IncrementalSfM::RegisterNextView(int view_id) {
    auto& view = map_.views[view_id];
    if(view.registered) {
        return false;
    }

    sift_->detectAndCompute(view.image, cv::noArray(), view.keypoints, view.descriptors);

    std::vector<cv::Point3f> pts3D;
    std::vector<cv::Point2f> pts2D;
    pts3D.reserve(5000);
    pts2D.reserve(5000);

    for (auto& kv : map_.views) {
        auto& reg_view = kv.second;
        if(!reg_view.registered) continue;

        auto good_matches = MatchAndFilterKNN(view.descriptors, reg_view.descriptors);

        for (auto &m : good_matches) {
            int matched_kp_idx = m.trainIdx;
            for (size_t tid=0; tid<map_.tracks.size(); ++tid) {
                bool found = false;
                for(auto &obs : map_.tracks[tid].observations) {
                    if(obs.first == reg_view.id && obs.second == matched_kp_idx) {
                        Eigen::Vector3d ept = map_.tracks[tid].point;
                        pts3D.push_back(cv::Point3f(ept[0], ept[1], ept[2]));
                        pts2D.push_back(view.keypoints[m.queryIdx].pt);
                        map_.AddObservation(view_id, m.queryIdx, (int)tid);                        
                        found = true;
                        break;
                    }
                }
                if(found) break;
            }
        }
    }

    if(pts3D.size() < 10) {
        std::cerr << "[WARN] View " << view_id << " doesn't have enough 2D-3D matches to solvePnP." << std::endl;
        return false;
    }

    cv::Mat rvec, tvec, inliers;
    bool ok = cv::solvePnPRansac(pts3D, pts2D, view.K, cv::noArray(),
                                 rvec, tvec, false,
                                 100, 8.0F, 0.99, inliers);
    if(!ok) {
        std::cerr << "[WARN] solvePnP failed for view " << view_id << std::endl;
        return false;
    }
    if(inliers.rows < 10) {
        std::cerr << "[WARN] PnP inliers too few for view " << view_id << std::endl;
        return false;
    }

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d Re;
    Eigen::Vector3d te;
    cv::cv2eigen(R, Re);
    cv::cv2eigen(tvec, te);

    view.pose.setIdentity();
    view.pose.block<3,3>(0,0) = Re;
    view.pose.block<3,1>(0,3) = te;
    view.registered = true;

    std::cout << "[INFO] View " << view_id << " is registered with " 
              << inliers.rows << " PnP inliers." << std::endl;

    return true;
}

void IncrementalSfM::TriangulateNewPoints(int view_id) {
    auto& view = map_.views[view_id];
    for(auto& kv : map_.views) {
        auto& v2 = kv.second;
        if(!v2.registered || v2.id == view_id) continue;

        auto good_matches = MatchAndFilterKNN(view.descriptors, v2.descriptors);

        for (auto &m : good_matches) {
            int qidx = m.queryIdx;
            int tidx = m.trainIdx;

            bool already_tracked = false;
            for(auto &track : map_.tracks) {
                bool found_view_obs = false;
                bool found_v2_obs = false;
                for(auto &obs : track.observations) {
                    if(obs.first == view_id && obs.second == qidx) {
                        found_view_obs = true;
                    }
                    if(obs.first == v2.id && obs.second == tidx) {
                        found_v2_obs = true;
                    }
                    if(found_view_obs && found_v2_obs) {
                        already_tracked = true;
                        break;
                    }
                }
                if(already_tracked) break;
            }
            if(already_tracked) continue;

            cv::Point2f pt1 = view.keypoints[qidx].pt;
            cv::Point2f pt2 = v2.keypoints[tidx].pt;

            Eigen::Matrix3d R1 = view.pose.block<3,3>(0,0);
            Eigen::Vector3d t1 = view.pose.block<3,1>(0,3);
            Eigen::Matrix3d R2 = v2.pose.block<3,3>(0,0);
            Eigen::Vector3d t2 = v2.pose.block<3,1>(0,3);

            cv::Mat R1_cv, t1_cv, R2_cv, t2_cv;
            cv::eigen2cv(R1, R1_cv);
            cv::eigen2cv(t1, t1_cv);
            cv::eigen2cv(R2, R2_cv);
            cv::eigen2cv(t2, t2_cv);

            cv::Mat Rt1, Rt2;
            cv::hconcat(R1_cv, t1_cv, Rt1);
            cv::hconcat(R2_cv, t2_cv, Rt2);

            cv::Mat P1 = view.K * Rt1;
            cv::Mat P2 = v2.K * Rt2;

            std::vector<cv::Point2f> vpt1 = {pt1}, vpt2 = {pt2};
            cv::Mat pt4D;
            cv::triangulatePoints(P1, P2, vpt1, vpt2, pt4D);

            cv::Mat x = pt4D.col(0);
            x /= x.at<float>(3);
            float X = x.at<float>(0);
            float Y = x.at<float>(1);
            float Z = x.at<float>(2);

            cv::Point3d pt3d(X, Y, Z);
            if (Z < 0.001 || Z > 1e8 || 
                !IsGoodTriangulatedPoint(pt1, pt2, P1, P2, pt3d, 3.0f)) {
                continue;
            }            

            Track track;
            track.point = Eigen::Vector3d(X, Y, Z);
            track.observations.push_back({view_id, qidx});
            track.observations.push_back({v2.id, tidx});
            map_.tracks.push_back(track);
        }
    }
}

void IncrementalSfM::BundleAdjust() {
    ceres::Problem problem;

    std::map<int, std::array<double,6>> cameras;
    for(auto &kv : map_.views) {
        auto &v = kv.second;
        if(!v.registered) continue;

        Eigen::Matrix3d R = v.pose.block<3,3>(0,0);
        Eigen::Vector3d t = v.pose.block<3,1>(0,3);
        Eigen::AngleAxisd aa(R);

        auto &cam = cameras[v.id];
        double angle = aa.angle();
        Eigen::Vector3d axis = aa.axis();

        if(angle < 1e-10) {
            cam[0] = 0; 
            cam[1] = 0; 
            cam[2] = 0;
        } else {
            cam[0] = angle * axis[0];
            cam[1] = angle * axis[1];
            cam[2] = angle * axis[2];
        }
        cam[3] = t[0];
        cam[4] = t[1];
        cam[5] = t[2];
    }

    std::vector<std::array<double,3>> points(map_.tracks.size());
    for(size_t i=0; i<map_.tracks.size(); ++i) {
        points[i][0] = map_.tracks[i].point[0];
        points[i][1] = map_.tracks[i].point[1];
        points[i][2] = map_.tracks[i].point[2];
    }

    for(size_t i=0; i<map_.tracks.size(); ++i) {
        auto &track = map_.tracks[i];
        for(auto &obs : track.observations) {
            int vid = obs.first;
            int kpid = obs.second;

            auto &v = map_.views[vid];
            if(!v.registered) continue;

            cv::Point2f pt = v.keypoints[kpid].pt;
            Eigen::Matrix3d K;
            cv::cv2eigen(v.K, K);

            ceres::CostFunction* cost =
                new ceres::AutoDiffCostFunction<ReprojError,2,6,3>(
                    new ReprojError(pt.x, pt.y, K)
                );
            problem.AddResidualBlock(cost,
                                     new ceres::HuberLoss(1.0),
                                     cameras[vid].data(),
                                     points[i].data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = 12;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-5;
    options.gradient_tolerance = 1e-6;
    options.parameter_tolerance = 1e-8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << std::endl;

    for(size_t i=0; i<map_.tracks.size(); ++i) {
        map_.tracks[i].point = Eigen::Vector3d(points[i][0], points[i][1], points[i][2]);
    }
    for(auto &kv : map_.views) {
        auto &v = kv.second;
        if(!v.registered) continue;
        auto &cam = cameras[v.id];

        Eigen::Vector3d aa_vec(cam[0], cam[1], cam[2]);
        double angle = aa_vec.norm();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if(angle > 1e-10) {
            Eigen::Vector3d axis = aa_vec / angle;
            Eigen::AngleAxisd aa(angle, axis);
            R = aa.toRotationMatrix();
        }
        Eigen::Vector3d t(cam[3], cam[4], cam[5]);

        Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
        M.block<3,3>(0,0) = R;
        M.block<3,1>(0,3) = t;
        v.pose = M;
    }

    double minZ = 1e9, maxZ = -1e9;
    double minX = 1e9, maxX = -1e9;
    double minY = 1e9, maxY = -1e9;
    for (auto& track : map_.tracks) {
        const auto& pt = track.point;
        minX = std::min(minX, pt[0]);
        maxX = std::max(maxX, pt[0]);
        minY = std::min(minY, pt[1]);
        maxY = std::max(maxY, pt[1]);
        minZ = std::min(minZ, pt[2]);
        maxZ = std::max(maxZ, pt[2]);
    }

    std::cout << "[DEBUG] 3D point range:" << std::endl;
    std::cout << "X: " << minX << " ~ " << maxX << std::endl;
    std::cout << "Y: " << minY << " ~ " << maxY << std::endl;
    std::cout << "Z: " << minZ << " ~ " << maxZ << std::endl;

    FilterBadPointsAfterBA(3.0f);
}

void IncrementalSfM::LocalBundleAdjust(int current_view_id) {
    ceres::Problem problem;
    std::set<int> local_view_ids = {current_view_id};

    for (int offset = 1; offset <= 2; ++offset) {
        if (map_.views.count(current_view_id - offset) && map_.views.at(current_view_id - offset).registered) {
            local_view_ids.insert(current_view_id - offset);
        }
    }

    std::map<int, std::array<double,6>> cameras;
    for (int vid : local_view_ids) {
        auto& v = map_.views[vid];
        Eigen::Matrix3d R = v.pose.block<3,3>(0,0);
        Eigen::Vector3d t = v.pose.block<3,1>(0,3);
        Eigen::AngleAxisd aa(R);
        auto& cam = cameras[vid];
        cam[0] = aa.angle() * aa.axis()[0];
        cam[1] = aa.angle() * aa.axis()[1];
        cam[2] = aa.angle() * aa.axis()[2];
        cam[3] = t[0]; cam[4] = t[1]; cam[5] = t[2];
    }

    std::vector<std::array<double,3>> points(map_.tracks.size());

    for (size_t i = 0; i < map_.tracks.size(); ++i) {
        points[i][0] = map_.tracks[i].point[0];
        points[i][1] = map_.tracks[i].point[1];
        points[i][2] = map_.tracks[i].point[2];

        bool used = false;

        for (auto& obs : map_.tracks[i].observations) {
            int vid = obs.first;
            if (!map_.views[vid].registered) continue;

            if (local_view_ids.count(vid)) {
                used = true;
                const auto& view = map_.views[vid];
                Eigen::Matrix3d K;
                cv::cv2eigen(view.K, K);
                cv::Point2f pt = view.keypoints[obs.second].pt;

                ceres::CostFunction* cost = new ceres::AutoDiffCostFunction<ReprojError, 2, 6, 3>(
                    new ReprojError(pt.x, pt.y, K)
                );
                problem.AddResidualBlock(cost,
                                         new ceres::HuberLoss(1.0),
                                         cameras[vid].data(),
                                         points[i].data());
            }
        }

        if (!used) continue;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.num_threads = 12;
    options.max_num_iterations = 30;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;

    for (auto& kv : cameras) {
        int vid = kv.first;
        auto& cam = kv.second;
        Eigen::Vector3d aa_vec(cam[0], cam[1], cam[2]);
        double angle = aa_vec.norm();
        Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
        if (angle > 1e-10) {
            Eigen::Vector3d axis = aa_vec / angle;
            R = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }
        Eigen::Vector3d t(cam[3], cam[4], cam[5]);
        map_.views[vid].pose.setIdentity();
        map_.views[vid].pose.block<3,3>(0,0) = R;
        map_.views[vid].pose.block<3,1>(0,3) = t;
    }

    for (size_t i = 0; i < map_.tracks.size(); ++i) {
        map_.tracks[i].point = Eigen::Vector3d(points[i][0], points[i][1], points[i][2]);
    }
}

void IncrementalSfM::GenerateCOLMAPOutput(){
    // Assumptions:
    //    - all photos were taken from the same camera
    //		- this means that any View in the SfMMap will have the same intrinsics
    //		- we can just get map_.views.first as our representative view
    //		[f 0 px]
    //		[0 f py]
    //		[0 0 1 ]

    std::ofstream cameraFile;
    // open cameras.txt
    cameraFile.open("cameras.txt");
    // TODO: check if we should use pinhole or opencv cam. gaussian splat wants pinhole, but do we need to do further conversion?
    // TODO: is GetIntrinsicsFromExif used anywhere?
    View representative_cam = map_.views.begin()->second;
    cv::Mat intrinsics = representative_cam.K;
    cameraFile << 1 << " SIMPLE_PINHOLE " << representative_cam.image.cols << " " << representative_cam.image.rows << " " << intrinsics.at<float>(0, 0) << " " << intrinsics.at<float>(0, 2) << " " << intrinsics.at<float>(1, 2) << "\n";
    cameraFile.close();
    
    std::ofstream imageFile;
    imageFile.open("images.txt");
    // open images.txt
    // iterate through map_.views
    // for (auto it = num_map.begin(); it != num_map.end(); ++it)
    for (auto it = map_.views.begin(); it != map_.views.end(); ++it) {
        View view = it->second;
        Eigen::Quaterniond q(view.pose.block<3,3>(0,0));
        imageFile << view.id << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " " << view.pose(0, 3) << " " << view.pose(1, 3) << " " << view.pose(2, 3) << " " << 1 << " " << view.image_path << "\n";
    }
    imageFile.close();

    // open points3D.txt
    std::ofstream points3DFile;
    points3DFile.open("points3D.txt");
    int id = 0;
    // initialize counter to assign 3D point IDs
        // TODO: see if 3D points already have IDs
    
    for (const Track& track : map_.tracks) {
        // TODO: get RGB value or initialize to some random color
        // TODO: get error
        int R = 255, G = R, B = R;
        int error = 0;
        points3DFile << id << " " << track.point[0] << " " << track.point[1] << " " << track.point[2] << " " << R << " " << G << " " << B << " " << error;
        for (const std::pair<int, int>& observation : track.observations) {
            points3DFile << " " << observation.first << " " << observation.second;
        }
        points3DFile << "\n";
        ++id;
    }
    points3DFile.close();

    std::string script = "python3 convert_script.py"; // TODO: modify with args for file name specification if needed
    int result = std::system(script.c_str());

    if (result != 0) {
        std::cerr << "Error while converting camera, images, and 3D point files, and converting them to binary.\n" << std::endl;
    }
    else {
        std::cout << "Successfully read camera, images, and 3D point files, and converted them to binary.\n" << std::endl;
    }
}
