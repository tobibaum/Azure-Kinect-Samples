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

#define DEBUG

const uint8_t COMBINED_BOD_ID = 15;
const uint8_t LOAD_BOD_ID = 16;

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
