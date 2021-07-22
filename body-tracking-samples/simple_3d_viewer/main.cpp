// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <iostream>
#include <map>
#include <vector>
#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.hpp>

#include <BodyTrackingHelpers.h>
#include <Utilities.h>
#include <Window3dWrapper.h>

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
    FIND_START_POSE,
    COPYING_POSE
} multi_kinect_state_t;

SkeletonCapture sc;

// Global State and Key Process Function
int64_t ProcessKey(void* /*context*/, int key)
{
    // https://www.glfw.org/docs/latest/group__keys.html
    switch (key)
    {
        // Quit
    case GLFW_KEY_ESCAPE:
        sc.s_isRunning = false;
        break;
    case GLFW_KEY_K:
        break;
    case GLFW_KEY_B:
        break;
    }
    return 1;
}

int64_t CloseCallback(void* /*context*/)
{
    sc.s_isRunning = false;
    return 1;
}

void callback_handler(int arg){
    sc.s_isRunning = false;
}

void VisualizeResult(k4abt::frame bodyFrame, Window3dWrapper& window3d, int depthWidth, int depthHeight) {

    // Obtain original capture that generates the body tracking result
    k4a::capture originalCapture = bodyFrame.get_capture();
    k4a::image depthImage = originalCapture.get_depth_image();
    k4a::image colImage = originalCapture.get_color_image();

    std::vector<Color> pointCloudColors(depthWidth * depthHeight, { 1.f, 1.f, 1.f, 1.f });

    // Read body index map and assign colors
    k4a::image bodyIndexMap = bodyFrame.get_body_index_map();
    const uint8_t* bodyIndexMapBuffer = bodyIndexMap.get_buffer();
    for (int i = 0; i < depthWidth * depthHeight; i++)
    {
        uint8_t bodyIndex = bodyIndexMapBuffer[i];
        if (bodyIndex != K4ABT_BODY_INDEX_MAP_BACKGROUND)
        {
            uint32_t bodyId = bodyFrame.get_body_id(bodyIndex);
            pointCloudColors[i] = g_bodyColors[bodyId % g_bodyColors.size()];
        }
    }

    // Visualize point cloud
    window3d.UpdatePointClouds(depthImage.handle(), pointCloudColors);

    // Visualize the skeleton data
    window3d.CleanJointsAndBones();
    uint32_t numBodies = bodyFrame.get_num_bodies();
    for (uint32_t i = 0; i < numBodies; i++)
    {
        k4abt_body_t body;
        bodyFrame.get_body_skeleton(i, body.skeleton);
        body.id = bodyFrame.get_body_id(i);

        // Assign the correct color based on the body id
        Color color = g_bodyColors[body.id % g_bodyColors.size()];
        color.a = 0.4f;
        Color lowConfidenceColor = color;
        lowConfidenceColor.a = 0.1f;

        // Visualize joints
        for (int joint = 0; joint < static_cast<int>(K4ABT_JOINT_COUNT); joint++)
        {
            if (body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                const k4a_float3_t& jointPosition = body.skeleton.joints[joint].position;
                const k4a_quaternion_t& jointOrientation = body.skeleton.joints[joint].orientation;

                window3d.AddJoint(
                    jointPosition,
                    jointOrientation,
                    body.skeleton.joints[joint].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM ? color : lowConfidenceColor);
            }
        }

        // Visualize bones
        for (size_t boneIdx = 0; boneIdx < g_boneList.size(); boneIdx++)
        {
            k4abt_joint_id_t joint1 = g_boneList[boneIdx].first;
            k4abt_joint_id_t joint2 = g_boneList[boneIdx].second;

            if (body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW &&
                body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_LOW)
            {
                bool confidentBone = body.skeleton.joints[joint1].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM &&
                    body.skeleton.joints[joint2].confidence_level >= K4ABT_JOINT_CONFIDENCE_MEDIUM;
                const k4a_float3_t& joint1Position = body.skeleton.joints[joint1].position;
                const k4a_float3_t& joint2Position = body.skeleton.joints[joint2].position;

                window3d.AddBone(joint1Position, joint2Position, confidentBone ? color : lowConfidenceColor);
            }
        }
    }

}

bool is_calibrated = false;
std::list<Eigen::Matrix4f> trans_memory;
Eigen::Matrix4f mat_trans = Eigen::Matrix4f::Zero();
Eigen::Matrix4f mat_trans_back = Eigen::Matrix4f::Zero();
std::vector<Eigen::MatrixXf> skel_hist_mas;

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

k4abt_body_t combine_bodies(k4abt_body_t bod0, k4abt_body_t bod1){
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
            combined.block(j, 0, 1, 3) = (mat0.block(j, 0, 1, 3));// + mat1.block(j, 0, 1, 3)) / 2;
        }
    }

    k4abt_body_t combined_bod0 = eigen_to_k4a_body(combined);
    combined_bod0.id = COMBINED_BOD_ID;

    skel_hist_mas.push_back(combined);

    return combined_bod0;
}

std::vector<Eigen::MatrixXf> example;
Eigen::Matrix4f umey_trans_tpose;
uint32_t state = 0;

k4abt_body_t fit_example(Eigen::MatrixXf bod_mat, uint32_t it_frame){
    // ========================
    // ==== FIT the T-POSE ====
    // ========================
    k4abt_body_t loaded_body, loaded_body_raw;
    if (state == FIND_START_POSE) {
        loaded_body_raw = eigen_to_k4a_body(example.at(0));
        umey_trans_tpose = Eigen::umeyama(example.at(0).transpose(), bod_mat.transpose(), true);
        Eigen::MatrixXf fitted_mat = k4a_body_to_eigen(transfer_bods(loaded_body_raw, umey_trans_tpose));
        float diff = (fitted_mat - bod_mat).cwiseAbs().sum();
        if (diff < 4000) {
            state = COPYING_POSE;
        }
    }
    else {
        loaded_body_raw = eigen_to_k4a_body(example.at(it_frame % example.size()));
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

    bool render3d = false;

    // Initialize the 3d window controller
    Window3dWrapper window3d;
    if(render3d){
        window3d.Create("3D Visualization", sc.sensorCalibrations[0]);
        window3d.SetCloseCallback(CloseCallback);
        window3d.SetKeyCallback(ProcessKey);
        window3d.SetJointFrameVisualization(false);
    }
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
                // TODO implement 4 cams!

                if(plot_it){
                    // from here on out we assume 2 cameras with synched tracked
                    // bodies for now
                    k4abt::frame frame0 = most_recent_frames[0];
                    k4abt::frame frame1 = most_recent_frames[1];

                    if(render3d){
                        VisualizeResult(frame0, window3d, sc.depthWidth, sc.depthHeight);
                        window3d.Render();
                    }

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

                            k4abt_body_t combined_bod = combine_bodies(bod0, bod1_trans);
                            Eigen::MatrixXf combined_bod_mat = k4a_body_to_eigen(combined_bod);


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
                        std::cout << "record in " << countdown << std::endl;
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
                        auto countdown = 15000 - std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_state).count();
                        std::cout << "record for " << countdown << std::endl;
                        std::ostringstream ss;
                        ss << "record for " << (float)countdown/1000.f;
                        msg = ss.str();
                        if (countdown < 0) {
                            // 10 seconds have past. record now!!
                            store_skeleton_hist(skel_hist_mas, "master_record.txt");
                            state = DEFAULT;
                            start_time_state = now;
                            msg = "";
                        }
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

    if(render3d){
        window3d.Delete();
        std::cout << "deleted window" << std::endl;
    }
    return 0;
}
