#ifndef __SKEL__
#define __SKEL__

#include <array>
#include <iostream>
#include <fstream>
#include <iterator>
#include <map>
#include <vector>
#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.hpp>

#include <thread>
#include <queue>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

//#define DEBUG

const uint8_t COMBINED_BOD_ID = 15;
const uint8_t LOAD_BOD_ID = 16;

std::vector<Eigen::MatrixXf> load_skeleton_hist(std::string _filename);
void store_skeleton_hist(std::vector<Eigen::MatrixXf> _skel_hist, std::string _filename);

Eigen::MatrixXf k4a_body_to_eigen(k4abt_body_t _main_body);
k4abt_body_t eigen_to_k4a_body(Eigen::MatrixXf _in_mat);
k4abt_body_t transfer_bods(k4abt_body_t _bod_in, Eigen::Matrix4f _mat_trans);
void draw_skeleton_on_img(std::vector<k4abt_body_t> bodies, cv::Mat color, k4a_calibration_t calibration);

class SkeletonCapture {
private:
    std::vector<k4a::device> devices;
    std::vector<k4abt::tracker> trackers;

    k4a_device_configuration_t get_config(bool b_master, bool do_color = true);

public:
    bool s_isRunning = true;
    int depthWidth, depthHeight;
    std::vector<k4a_calibration_t> sensorCalibrations;
    std::vector<std::queue<std::pair<long, k4abt::frame>>> bodyFrameQueues;
    uint8_t device_count;

    SkeletonCapture();
    void run(void);
};

#endif //__SKEL__
