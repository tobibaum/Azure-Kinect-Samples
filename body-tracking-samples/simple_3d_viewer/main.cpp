// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <iostream>
#include <map>
#include <vector>
#include <k4a/k4a.h>
#include <k4abt.hpp>

#include <numeric>
#include <thread>
#include <queue>
#include <signal.h>

#include "util.h"
#include "skel_cap.hpp"

typedef enum
{
    DEFAULT = 0,
    RECORD_CLICKED,
    RECORDING,
    SAVE_RECORDING,
    FIND_START_POSE,
    COPYING_POSE
} multi_kinect_state_t;

SkeletonCapture sc;

void callback_handler(int arg){
    sc.s_isRunning = false;
}

bool is_calibrated = false;
std::list<Eigen::Matrix4f> trans_memory;
Eigen::Matrix4f mat_trans = Eigen::Matrix4f::Zero();
Eigen::Matrix4f mat_trans_back = Eigen::Matrix4f::Zero();
std::vector<std::pair<long, Eigen::MatrixXf>> skel_hist_mas;

void update_calibration(Eigen::MatrixXf body_mat0, Eigen::MatrixXf body_mat1){
    // ====================
    // ===== UMEYAMA ======
    // ====================
    Eigen::Matrix4f umey_trans;
    umey_trans = Eigen::umeyama(body_mat0.transpose(), body_mat1.transpose(), false);

    trans_memory.push_back(umey_trans);
    while (trans_memory.size() > 10) {
        trans_memory.pop_front();
    }

    // ====================
    // rotation/translation/scale between cams
    // ====================
    if (trans_memory.size() == 10) {
        // sum up
        for (auto _k = trans_memory.begin(); _k != trans_memory.end(); _k++) {
            mat_trans += *_k;
        }
        // normalize to average
        mat_trans /= mat_trans(3, 3);

        Eigen::MatrixXf mat_trans_temp(3, 4);
        Eigen::Matrix3f transp = mat_trans.block(0, 0, 3, 3).transpose();
        mat_trans_temp << transp, -transp * mat_trans.block(0, 3, 3, 1);

        mat_trans_back << mat_trans_temp, 0, 0, 0, 1;
        is_calibrated = true;

        //std::cout << "calibration done!" << std::endl;
    }
}

Eigen::MatrixXf combine_bodies(k4abt_body_t bod0, k4abt_body_t bod1){
// compute average position in master camera
    Eigen::MatrixXf mat0 = k4a_body_to_eigen(bod0);
    Eigen::MatrixXf mat1 = k4a_body_to_eigen(transfer_bods(bod1, mat_trans_back));

    Eigen::VectorXd conf0((int)K4ABT_JOINT_COUNT);
    Eigen::VectorXd conf1((int)K4ABT_JOINT_COUNT);
    int i = 0;
    for (const k4abt_joint_t& joint : bod0.skeleton.joints) {
        conf0(i) = joint.confidence_level;
        i++;
    }
    i = 0;
    for (const k4abt_joint_t& joint : bod1.skeleton.joints) {
        conf1(i) = joint.confidence_level;
        i++;
    }
    Eigen::VectorXd res = conf0 - conf1;
    Eigen::MatrixXf combined((int)K4ABT_JOINT_COUNT, 3);

    //for (int d : res) {
    for (int j = 0; j < res.size(); j++) {
        int d = res[j];
        if (d > 0) {
            combined.block(j, 0, 1, 3) = mat0.block(j, 0, 1, 3);
        }
        else if (d < 0)
        {
            combined.block(j, 0, 1, 3) = mat1.block(j, 0, 1, 3);
        }
        else {
            combined.block(j, 0, 1, 3) = (mat0.block(j, 0, 1, 3) + mat1.block(j, 0, 1, 3)) / 2.f;
        }
    }

    return combined;
}

std::vector<std::pair<long, Eigen::MatrixXf>> example;
Eigen::Matrix4f umey_trans_tpose;
uint32_t state = 0;

k4abt_body_t fit_example(Eigen::MatrixXf bod_mat, uint32_t it_frame){
    // ========================
    // ==== FIT the T-POSE ====
    // ========================
    k4abt_body_t loaded_body, loaded_body_raw;
    if (state == FIND_START_POSE) {
        loaded_body_raw = eigen_to_k4a_body(example.at(0).second);
        umey_trans_tpose = Eigen::umeyama(example.at(0).second.transpose(), bod_mat.transpose(), true);
        Eigen::MatrixXf fitted_mat = k4a_body_to_eigen(transfer_bods(loaded_body_raw, umey_trans_tpose));
        float diff = (fitted_mat - bod_mat).cwiseAbs().sum();
        if (diff < 4000) {
            state = COPYING_POSE;
        }
    }
    else {
        // TODO! include timing here!
        loaded_body_raw = eigen_to_k4a_body(example.at(it_frame % example.size()).second);
    }
    loaded_body = transfer_bods(loaded_body_raw, umey_trans_tpose);
    loaded_body.id = LOAD_BOD_ID;

    return loaded_body;
}


int main(int argc, char** argv)
{
#ifdef DEBUG
    std::cout << "==[WARNING] DEBUG MODE==" << std::endl << std::endl;
#endif

    int dev_id = -1;
    if(argc == 2){
        dev_id = atoi(argv[1]);
    }

    // runner for pumping stuff into the bodyframe queue
    std::thread body_frame_thread(&SkeletonCapture::run, &sc);

    cv::Mat color, color1, color_flip, color1_flip;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto start_time_state = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    uint32_t frame_count = 0;
    uint32_t it_frame = 0;


    signal(SIGINT, callback_handler);
    std::string msg;


    std::vector<k4abt::frame> most_recent_frames(sc.device_count);
    std::vector<long> most_recent_times(sc.device_count);

    while(sc.s_isRunning){
        bool plot_it = false;
        for(int dev_ind=0; dev_ind < sc.device_count; dev_ind++){
            while(!sc.bodyFrameQueues[dev_ind].empty()){
                // found a new frame for this camera
                k4abt::frame bf = sc.bodyFrameQueues[dev_ind].front().second;
                most_recent_frames[dev_ind] = bf;
                most_recent_times[dev_ind] = sc.bodyFrameQueues[dev_ind].front().first;
                
                // =======================
                // do something with the bodyFrame!
                // =======================
                if(sc.device_count == 2){
                    long time_diff = std::abs(most_recent_times[1] - most_recent_times[0]);
                    // less than 5ms difference between frames. same!!!
                    if(time_diff < 5){
                        if(most_recent_frames[0] != nullptr &&
                            most_recent_frames[1] != nullptr)
                        {
                            plot_it = true;
                        }
                    }
                }
                else if(sc.device_count == 4){
                    // TODO implement 4 cams!
                } 

                if(plot_it){
                    // from here on out we assume 2 cameras with synched tracked
                    // bodies for now
                    k4abt::frame frame0 = most_recent_frames[0];
                    k4abt::frame frame1 = most_recent_frames[1];

                    //=====================
                    //========MAIN=========
                    //=====================
                    k4a::capture cap = frame0.get_capture();
                    k4a::image colImage = cap.get_color_image();
                    color = k4a::get_mat(colImage);

                    k4a::capture cap1 = frame1.get_capture();
                    k4a::image colImage1 = cap1.get_color_image();
                    color1 = k4a::get_mat(colImage1);

                    std::vector<k4abt_body_t> bodies, bodies1;

                    // combine the two synched frames!
                    if (frame0.get_num_bodies() >= 1 && frame1.get_num_bodies() >= 1) {
                        // assume that the first body in get bodies is the biggest
                        // one?
                        k4abt_body_t bod0 = frame0.get_body(0);
                        k4abt_body_t bod1 = frame1.get_body(0);

                        Eigen::MatrixXf body_mat0 = k4a_body_to_eigen(bod0);
                        Eigen::MatrixXf body_mat1 = k4a_body_to_eigen(bod1);

                        update_calibration(body_mat0, body_mat1);

                        if(is_calibrated)
                        {
                            k4abt_body_t bod0_trans = transfer_bods(bod0, mat_trans);
                            k4abt_body_t bod1_trans = transfer_bods(bod1, mat_trans_back);


                            Eigen::MatrixXf combined_bod_mat = combine_bodies(bod0, bod1_trans);
                            k4abt_body_t combined_bod = eigen_to_k4a_body(combined_bod_mat, COMBINED_BOD_ID);

                            
                            skel_hist_mas.push_back({most_recent_times[0], combined_bod_mat});

                            bodies.push_back(combined_bod);
                            bodies1.push_back(transfer_bods(combined_bod, mat_trans));

                            if(example.size() > 0){
                                k4abt_body_t loaded = fit_example(combined_bod_mat, it_frame);
                                bodies.push_back(loaded);
                                bodies1.push_back(transfer_bods(loaded, mat_trans));
                            }
                        }
                    }

                    // plot it all
                    draw_skeleton_on_img(bodies, color, sc.sensorCalibrations[0]);
                    cv::flip(color, color_flip, 1);
                    cv::putText(color_flip, msg, {50,100},
                            cv::FONT_HERSHEY_SIMPLEX, 3, {0,0,0}, 3,
                            cv::LINE_AA);
                    cv::imshow("color window", color_flip);

                    draw_skeleton_on_img(bodies1, color1, sc.sensorCalibrations[1]);
                    cv::flip(color1, color1_flip, 1);
                    cv::imshow("color window 1", color1_flip);
                    //=====================
                    //========MAIN=========
                    //=====================

                    // vvvvvv FPS vvvvvv
                    now = std::chrono::high_resolution_clock::now();
                    float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    if (durr > 1000) {
                        start_time = now;
                        std::cout << "FPS: " << frame_count << std::endl;
                        frame_count = 0;
                    }
                    frame_count++;
                    it_frame++;
                    plot_it = false;
                }

                // check and process some key inputs
                const int32_t key = cv::waitKey(1); //1ms
                if(key != -1){
                    std::cout << "key press: " << (char)key << std::endl;
                }
                if (key == 'q') {
                    sc.s_isRunning = false;
                    break;
                }
                else if (key == 'r') {
                    // start countdown
                    start_time_state = std::chrono::high_resolution_clock::now();
                    state = RECORD_CLICKED;
                }
                else if (key == 's') {
                    // save recording
                    start_time_state = std::chrono::high_resolution_clock::now();
                    state = SAVE_RECORDING;
                }
                else if (key == 'l') {
                    example = load_skeleton_hist("master_record.txt");
                    state = FIND_START_POSE;
                    std::cout << "load clicked!!" << std::endl;
                }
                else if (key == 'x') {
                    example.clear();
                    state = DEFAULT;
                    std::cout << "reset loaded" << std::endl;
                    msg = "";
                }
                else if (key == 'd') {
                    state = DEFAULT;
                }

                switch (state) {
                    case RECORD_CLICKED:
                    {
                        now = std::chrono::high_resolution_clock::now();
                        auto countdown = 3000 - std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_state).count();
                        //std::cout << "record in " << countdown << std::endl;
                        std::ostringstream ss;
                        ss << "record in " << (float)countdown/1000.f;
                        msg = ss.str();
                        if (countdown < 0) {
                            // 3 seconds have past. record now!!
                            skel_hist_mas.clear();
                            state = RECORDING;
                            start_time_state = now;
                            msg = "";
                        }
                        break;
                    }
                    case RECORDING:
                    {
                        now = std::chrono::high_resolution_clock::now();
                        auto countdown = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_state).count();
                        //std::cout << "record for " << countdown << std::endl;
                        std::ostringstream ss;
                        ss << "record for " << (float)countdown/1000.f;
                        msg = ss.str();
                        break;
                    }
                    case SAVE_RECORDING:
                    {
                        // 10 seconds have past. record now!!
                        std::cout << "store recording" << std::endl;
                        store_skeleton_hist(skel_hist_mas, "master_record.txt");
                        state = DEFAULT;
                        start_time_state = now;
                        msg = "";
                        break;
                    }
                    default:
                    {
                        break;
                    }
                }

                sc.bodyFrameQueues[dev_ind].pop();
            }
        }
    }
    std::cout << "end run!" << std::endl;
    body_frame_thread.join();

    return 0;
}
