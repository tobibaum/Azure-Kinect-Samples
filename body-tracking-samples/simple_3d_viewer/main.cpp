// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <iostream>
#include <map>
#include <vector>
#include <k4arecord/playback.h>
#include <k4a/k4a.h>
#include <k4abt.h>
#include <k4abt.hpp>

#include <BodyTrackingHelpers.h>
#include <Utilities.h>
#include <Window3dWrapper.h>

#include <numeric>
#include <thread>
#include <queue>
#include <signal.h>

// Global State and Key Process Function
bool s_isRunning = true;
Visualization::Layout3d s_layoutMode = Visualization::Layout3d::OnlyMainView;
bool s_visualizeJointFrame = false;

int64_t ProcessKey(void* /*context*/, int key)
{
    // https://www.glfw.org/docs/latest/group__keys.html
    switch (key)
    {
        // Quit
    case GLFW_KEY_ESCAPE:
        s_isRunning = false;
        break;
    case GLFW_KEY_K:
        s_layoutMode = (Visualization::Layout3d)(((int)s_layoutMode + 1) % (int)Visualization::Layout3d::Count);
        break;
    case GLFW_KEY_B:
        s_visualizeJointFrame = !s_visualizeJointFrame;
        break;
    }
    return 1;
}

int64_t CloseCallback(void* /*context*/)
{
    s_isRunning = false;
    return 1;
}

void VisualizeResult(k4abt::frame bodyFrame, Window3dWrapper& window3d, int depthWidth, int depthHeight) {

    // Obtain original capture that generates the body tracking result
    k4a::capture originalCapture = bodyFrame.get_capture();
    k4a::image depthImage = originalCapture.get_depth_image();

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

std::vector<std::queue<k4abt::frame>> bodyFrameQueues;

std::vector<k4a::device> devices;
std::vector<k4abt::tracker> trackers;
uint8_t device_count;

std::vector<k4a_calibration_t> sensorCalibrations;
int depthWidth, depthHeight;

void bodyReaderInit(int arg){
    device_count = k4a_device_get_installed_count();
    printf("Found %d connected devices:\n", device_count);

    if(arg!=-1){
        device_count = 1;
    }

    devices.reserve(device_count);
    trackers.reserve(device_count);
    int depthWidth, depthHeight;
    sensorCalibrations.reserve(device_count);

    for(uint8_t dev_ind = 0; dev_ind < device_count; dev_ind++){
        int device_id = dev_ind;

        if(arg!=-1){
            device_id = arg;
        }

        k4a::device dev = k4a::device::open(device_id);

        // Start camera. Make sure depth camera is enabled.
        k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
        deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
        deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;

        dev.start_cameras(&deviceConfig);

        k4a::calibration sensorCalib = dev.get_calibration(deviceConfig.depth_mode, deviceConfig.color_resolution);

        depthWidth = sensorCalib.depth_camera_calibration.resolution_width;
        depthHeight = sensorCalib.depth_camera_calibration.resolution_height;

        // Create Body Tracker
        k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
        tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU;
        //tracker_config.model_path =  "/usr/bin/dnn_model_2_0_lite_op11.onnx";
        trackers[dev_ind] = k4abt::tracker::create(sensorCalib, tracker_config);

        devices[dev_ind] = std::move(dev);
        sensorCalibrations[dev_ind] = std::move(sensorCalib);

        std::queue<k4abt::frame> empty;
        bodyFrameQueues.push_back(empty);
    }

}

void run(void){
    auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    std::vector<std::map<std::string, int>> counters(device_count);

    std::vector<k4abt::frame> bodyFrames(device_count);

    auto t_delay = std::chrono::milliseconds(0);
    //auto t_delay = std::chrono::milliseconds(K4A_WAIT_INFINITE);

    while (s_isRunning)
    {
        for(int dev_ind = 0; dev_ind < device_count; dev_ind++){
            k4a::capture sensor_capture;
            bool getCaptureResult = devices[dev_ind].get_capture(&sensor_capture, t_delay);
            if (getCaptureResult)
            {
                counters[dev_ind]["cap"]++;

                bool queueCaptureResult = trackers[dev_ind].enqueue_capture(sensor_capture, t_delay);
                if(queueCaptureResult){
                    counters[dev_ind]["enq"]++;
                }
            }

            // Pop Result from Body Tracker
            k4abt::frame bf;
            bool popFrameResult = trackers[dev_ind].pop_result(&bf, t_delay);
            if (popFrameResult)
            {
                counters[dev_ind]["pop"]++;
                bodyFrameQueues[dev_ind].push(bf);
            }
        }

        // print timing
        now = std::chrono::high_resolution_clock::now();
        float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
        if (durr > 1000) {
            start_time = now;
            for(int di=0;di<device_count;di++){
                std::cout << "==" << di << "==" << std::endl;
                for(auto m : counters[di]){
                    std::cout << m.first << " " << m.second << std::endl;
                    counters[di][m.first]=0;
                }
            }
        }
    }
    std::cout << "Finished body tracking processing!" << std::endl;
}

void callback_handler(int arg){
    std::cout << "yep" << std::endl;
    s_isRunning = false;
}

int main(int argc, char** argv)
{
    int dev_id = -1;
    if(argc == 2){
        dev_id = atoi(argv[1]);
    }

    bodyReaderInit(dev_id);

    std::thread body_frame_thread(run);

    // Initialize the 3d window controller
    Window3dWrapper window3d;
    window3d.Create("3D Visualization", sensorCalibrations[0]);
    window3d.SetCloseCallback(CloseCallback);
    window3d.SetKeyCallback(ProcessKey);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    signal(SIGINT, callback_handler);
    while(s_isRunning){
        //std::this_thread::sleep_for(std::chrono::milliseconds(30));
        for(int i=0; i < bodyFrameQueues.size();i++){

            int count = 0;

            while(!bodyFrameQueues[i].empty()){
                //std::cout << "plot it?" << std::endl;
                k4abt::frame bf = bodyFrameQueues[i].front();
                
                // do something with the bodyFrame!
                if(i==0){
                    //std::cout << "got some" << std::endl;
                    VisualizeResult(bf, window3d, depthWidth, depthHeight);
                    window3d.SetLayout3d(s_layoutMode);
                    window3d.SetJointFrameVisualization(s_visualizeJointFrame);
                    window3d.Render();

                    // Count frames for plotting
                    now = std::chrono::high_resolution_clock::now();
                    float durr = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    if (durr > 1000) {
                        start_time = now;
                        std::cout << "FPS: " << frame_count << std::endl;
                        frame_count = 0;
                    }
                    frame_count++;
                }

                bodyFrameQueues[i].pop();
                count++;
            }
        }
    }
    std::cout << "end run!" << std::endl;
    body_frame_thread.join();
    window3d.Delete();
    std::cout << "deleted window" << std::endl;
    return 0;
}
