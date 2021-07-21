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

    // Initialize the 3d window controller
    Window3dWrapper window3d;
    window3d.Create("3D Visualization", sc.sensorCalibrations[0]);
    window3d.SetCloseCallback(CloseCallback);
    window3d.SetKeyCallback(ProcessKey);
    window3d.SetJointFrameVisualization(false);
    cv::Mat color, color_flip;

    auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    signal(SIGINT, callback_handler);


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
                        plot_it = true;
                    }
                }
                // TODO implement 4 cams!

                if(plot_it){
                    // frames are guaranteed synchronized here!!! :D
                    k4abt::frame bf = most_recent_frames[0];
                    VisualizeResult(bf, window3d, sc.depthWidth, sc.depthHeight);
                    window3d.Render();

                    //=====================
                    //========MAIN=========
                    //=====================
                    k4a::capture cap = bf.get_capture();
                    k4a::image colImage = cap.get_color_image();
                    color = k4a::get_mat(colImage);
                    //draw_skeleton_on_img(bodies, color, calibrations[0]);
                    cv::flip(color, color_flip, 1);
                    cv::imshow("color window", color_flip);

                    // vvvvvv FPS vvvvvv
                    now = std::chrono::high_resolution_clock::now();
                    float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    if (durr > 1000) {
                        start_time = now;
                        std::cout << "FPS: " << frame_count << std::endl;
                        frame_count = 0;
                    }
                    frame_count++;
                    plot_it = false;
                }else{
                    const int32_t key = cv::waitKey(1); //1ms
                }

                sc.bodyFrameQueues[dev_ind].pop();
            }
        }
    }
    std::cout << "end run!" << std::endl;
    body_frame_thread.join();
    window3d.Delete();
    std::cout << "deleted window" << std::endl;
    return 0;
}
