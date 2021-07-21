#include "skel_cap.hpp"

k4a_device_configuration_t SkeletonCapture::get_config(bool b_master, bool do_color) {
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig.camera_fps = K4A_FRAMES_PER_SECOND_30;

    if (do_color) {
        deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
        deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_720P;
        deviceConfig.synchronized_images_only = true;
        deviceConfig.depth_delay_off_color_usec = 10;
        if (b_master) {
            deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_MASTER;
        }
        else {
            deviceConfig.wired_sync_mode = K4A_WIRED_SYNC_MODE_SUBORDINATE;
            // TODO: insert delay for off-cycle cams! (once we have 4 of them)
            //deviceConfig.subordinate_delay_off_master_usec = 16500;
            //deviceConfig.subordinate_delay_off_master_usec = 24500;
        }
    }
    else {
        deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_OFF;
    }

    return deviceConfig;
}

SkeletonCapture::SkeletonCapture(){
    device_count = k4a_device_get_installed_count();
    printf("Found %d connected devices:\n", device_count);

    //trackers.reserve(device_count);
    sensorCalibrations.reserve(device_count);

    bool master_found = false;
    uint8_t subord_cnt = 0;
    std::vector<k4a::device> sub_devices;

    for (int dev_ind = 0; dev_ind < device_count; dev_ind++) {
        std::cout << dev_ind << std::endl;
        k4a::device device = k4a::device::open(dev_ind);

        if (device.is_sync_out_connected() && !master_found)
        {
            master_found = true;
            devices.push_back(std::move(device));
        }
        else
        {
            sub_devices.push_back(std::move(device));
            subord_cnt++;
        }
    }

    for(int i=0; i<subord_cnt; i++){
        devices.push_back(std::move(sub_devices[i]));
    }

    for (int dev_ind = device_count - 1; dev_ind >= 0; dev_ind--)
    {
        // Start camera. Make sure depth camera is enabled.
        k4a_device_configuration_t deviceConfig = get_config(dev_ind == 0);

        devices[dev_ind].start_cameras(&deviceConfig);

        k4a::calibration sensorCalib = devices[dev_ind].get_calibration(deviceConfig.depth_mode,
            deviceConfig.color_resolution);

        sensorCalibrations[dev_ind] = std::move(sensorCalib);

        // Get calibration information
        depthWidth = sensorCalib.depth_camera_calibration.resolution_width;
        depthHeight = sensorCalib.depth_camera_calibration.resolution_height;

        // Create Body Tracker
        k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
#ifdef DEBUG
        tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU;
        tracker_config.model_path =  "/usr/bin/dnn_model_2_0_lite_op11.onnx";
#else
        tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_TENSORRT;
#endif //DEBUG
        trackers.insert(trackers.begin(), k4abt::tracker::create(sensorCalib, tracker_config));

        std::queue<std::pair<long, k4abt::frame>> empty;
        bodyFrameQueues.push_back(empty);
    }
}

void SkeletonCapture::run(void){
    auto start_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();
    std::vector<std::map<std::string, int>> counters(device_count);

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

                k4a::capture originalCapture = bf.get_capture();
                k4a::image colImage = originalCapture.get_color_image();
                long ts = colImage.get_device_timestamp().count() / 1000;

                // insert this bodyframe together with its exact capture time
                bodyFrameQueues[dev_ind].push({ts, bf});
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

Eigen::MatrixXf k4a_body_to_eigen(k4abt_body_t _main_body) {
	int i = 0;
	Eigen::MatrixXf _body_mat((int)K4ABT_JOINT_COUNT, 3);
	for (const k4abt_joint_t& joint : _main_body.skeleton.joints) {
		_body_mat.row(i) << joint.position.v[0], joint.position.v[1], joint.position.v[2];
		i++;
	}
	return _body_mat;
}

k4abt_body_t eigen_to_k4a_body(Eigen::MatrixXf _in_mat) {
	int i = 0;
	k4abt_body_t bod = k4abt_body_t();
	for (k4abt_joint_t& joint : bod.skeleton.joints) {
		joint.position.xyz.x = _in_mat(i, 0);
		joint.position.xyz.y = _in_mat(i, 1);
		joint.position.xyz.z = _in_mat(i, 2);
		i++;
	}
	return bod;
}

std::vector<Eigen::MatrixXf> load_skeleton_hist(std::string _filename) {
	std::vector<Eigen::MatrixXf> output;

	std::ifstream infile(_filename);
	std::string line;
	if (infile.is_open())
	{
		while (std::getline(infile, line)) {
			std::vector<float> v;

			// Build an istream that holds the input string
			std::istringstream iss(line);

			// Iterate over the istream, using >> to grab floats
			// and push_back to store them in the vector
			std::copy(std::istream_iterator<float>(iss),
				std::istream_iterator<float>(),
				std::back_inserter(v));

			Eigen::Map<Eigen::MatrixXf> line_mat(v.data(), K4ABT_JOINT_COUNT, 3);
			output.push_back(line_mat);
		}
	}
	return output;
}

void store_skeleton_hist(std::vector<Eigen::MatrixXf> _skel_hist, std::string _filename) {
	// store the skel histories.
	std::ofstream outfile(_filename);
	if (outfile.is_open())
	{
		// space separated line-wise elements
		for (const Eigen::MatrixXf& skel : _skel_hist) {
			for (int i = 0; i < skel.size(); i++) {
				outfile << skel.data()[i] << " ";
			}
			outfile << std::endl;
		}
	}
	outfile.close();
}

std::vector<std::pair<int, int>> draw_pairs =
{
	{ K4ABT_JOINT_PELVIS, K4ABT_JOINT_SPINE_NAVEL },
	{ K4ABT_JOINT_SPINE_NAVEL, K4ABT_JOINT_SPINE_CHEST },
	{ K4ABT_JOINT_SPINE_CHEST, K4ABT_JOINT_NECK },
	{ K4ABT_JOINT_NECK, K4ABT_JOINT_CLAVICLE_LEFT},
	{ K4ABT_JOINT_CLAVICLE_LEFT, K4ABT_JOINT_SHOULDER_LEFT},
	{ K4ABT_JOINT_SHOULDER_LEFT, K4ABT_JOINT_ELBOW_LEFT},
	{ K4ABT_JOINT_ELBOW_LEFT, K4ABT_JOINT_WRIST_LEFT},
	{ K4ABT_JOINT_WRIST_LEFT, K4ABT_JOINT_HAND_LEFT},
	{ K4ABT_JOINT_NECK, K4ABT_JOINT_CLAVICLE_RIGHT},
	{ K4ABT_JOINT_CLAVICLE_RIGHT, K4ABT_JOINT_SHOULDER_RIGHT},
	{ K4ABT_JOINT_SHOULDER_RIGHT, K4ABT_JOINT_ELBOW_RIGHT},
	{ K4ABT_JOINT_ELBOW_RIGHT, K4ABT_JOINT_WRIST_RIGHT},
	{ K4ABT_JOINT_WRIST_RIGHT, K4ABT_JOINT_HAND_RIGHT},
	{ K4ABT_JOINT_PELVIS, K4ABT_JOINT_HIP_LEFT },
	{ K4ABT_JOINT_HIP_LEFT,K4ABT_JOINT_KNEE_LEFT },
	{ K4ABT_JOINT_KNEE_LEFT,K4ABT_JOINT_ANKLE_LEFT },
	{ K4ABT_JOINT_ANKLE_LEFT,K4ABT_JOINT_FOOT_LEFT },
	{ K4ABT_JOINT_PELVIS, K4ABT_JOINT_HIP_RIGHT },
	{ K4ABT_JOINT_HIP_RIGHT,K4ABT_JOINT_KNEE_RIGHT },
	{ K4ABT_JOINT_KNEE_RIGHT,K4ABT_JOINT_ANKLE_RIGHT },
	{ K4ABT_JOINT_ANKLE_RIGHT,K4ABT_JOINT_FOOT_RIGHT },
	{ K4ABT_JOINT_NECK, K4ABT_JOINT_HEAD},
	{ K4ABT_JOINT_HEAD, K4ABT_JOINT_NOSE}
};

void draw_skeleton_on_img(std::vector<k4abt_body_t> bodies, cv::Mat color, k4a_calibration_t calibration)
{
	// Visualize Skeleton
	for (const k4abt_body_t& body : bodies) {
		std::map<int, cv::Point> joint_to_point;
		cv::Vec3b col, col2;

		if (body.id == COMBINED_BOD_ID) {
			col = cv::Vec3b(255, 0, 0);
			col2 = cv::Vec3b(0, 255, 0);
		}
		else {
			col = cv::Vec3b(255, 255, 0);
			col2 = cv::Vec3b(0, 255, 255);
		}
		int thicc = ((body.id == COMBINED_BOD_ID) || (body.id == LOAD_BOD_ID)) ? 10 : 2;

		cv::Point last_point;
		int k = 0;
		for (const k4abt_joint_t& joint : body.skeleton.joints) {
			k4a_float2_t position;
			int valid = 0;


			k4a_result_t result = k4a_calibration_3d_to_2d(&calibration, &joint.position, k4a_calibration_type_t::K4A_CALIBRATION_TYPE_DEPTH, k4a_calibration_type_t::K4A_CALIBRATION_TYPE_COLOR, &position, &valid);
			if (K4A_RESULT_SUCCEEDED != result) {
				k++;
				continue;
			}

			const int32_t thickness = (joint.confidence_level >= k4abt_joint_confidence_level_t::K4ABT_JOINT_CONFIDENCE_MEDIUM) ? -1 : 1;
			const cv::Point point(static_cast<int32_t>(position.xy.x), static_cast<int32_t>(position.xy.y));
			cv::circle(color, point, thicc, col2, thickness);
			joint_to_point.insert({ k, point });
			k++;
		}

		for (const std::pair<int, int>& dp : draw_pairs) {
			cv::Point p1 = joint_to_point[dp.first];
			cv::Point p2 = joint_to_point[dp.second];
			if ((p1.x == 0 && p1.y == 0) || (p2.x == 0 && p2.y == 0)) {
				// this joint isnt visible
				continue;
			}
			cv::line(color, p1, p2, col, thicc);
		}
	}
}
