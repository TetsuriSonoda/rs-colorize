// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.


#include <iostream>
#include <fstream>
#include <string>
#include <atomic>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API

// Save as a pointcloud
void SavePLY(std::string file_name, std::vector<float>& point_cloud, std::vector<unsigned char>& point_color)
{
	std::ofstream new_file;
	new_file.open(file_name, std::ios::out);

	new_file << "ply" << std::endl;
	new_file << "format ascii 1.0" << std::endl;
	new_file << "comment Realsense D400 series generated" << std::endl;
	new_file << "element vertex " + std::to_string(point_cloud.size() / 3) << std::endl;
	new_file << "property float x" << std::endl;
	new_file << "property float y" << std::endl;
	new_file << "property float z" << std::endl;
	new_file << "property uchar red" << std::endl;
	new_file << "property uchar green" << std::endl;
	new_file << "property uchar blue" << std::endl;
	new_file << "end_header" << std::endl;
	new_file << std::endl;

	for (int i = 0; i < point_cloud.size(); i += 3)
	{
		new_file << point_cloud[i] << " " << point_cloud[i + 1] << " " << point_cloud[i + 2] << " " << (int)point_color[i + 2] << " " << (int)point_color[i + 1] << " " << (int)point_color[i + 0] << std::endl;
	}
}

// convert rgb value to quantization 0-1535 value
unsigned short RGBtoD(unsigned char r, unsigned char g, unsigned char b)
{
	if (r + g + b > 128)
	{
		// conversion from RGB color to quantized depth value
		if (b + g + r < 128)
		{
			return 0;
		}
		else if (r >= g && r >= b)
		{
			if (g >= b)
			{
				return g - b;
			}
			else
			{
				return (g - b) + 1535;
			}
		}
		else if (g >= r && g >= b)
		{
			return b - r + 512;
		}
		else if (b >= g && b >= r)
		{
			return r - g + 1024;
		}
	}

	return 0;
}

void ColorizedDisparityToDepth(rs2_intrinsics intrinsic, float stereo_baseline_meter, float min_depth, float max_depth, float depth_units, cv::Mat& color_mat, cv::Mat& depth_mat)
{
	auto _width = color_mat.size().width;
	auto _height = color_mat.size().height;

	// same code to librealsense
	auto _focal_lenght_mm = intrinsic.fx;
	const uint8_t fractional_bits = 5;
	const uint8_t fractions = 1 << fractional_bits;
	auto _d2d_convert_factor = (stereo_baseline_meter * _focal_lenght_mm * fractions) / depth_units;

	auto in = reinterpret_cast<const unsigned char*>(color_mat.data);
	auto out = reinterpret_cast<unsigned short*>(depth_mat.data);

	auto min_disparity = _d2d_convert_factor / max_depth;
	auto max_disparity = _d2d_convert_factor / min_depth;

	float input{};
	//TODO SSE optimize
	for (auto i = 0; i < _height; i++)
		for (auto j = 0; j < _width; j++)
		{
			auto R = *in++;
			auto G = *in++;
			auto B = *in++;

			auto out_value = RGBtoD(R, G, B);

			if (out_value > 0)
			{
				input = min_disparity + (max_disparity - min_disparity) * out_value / 1535.0f;
				*out++ = static_cast<unsigned short>((_d2d_convert_factor / input) / depth_units + 0.5f);
			}
			else
			{
				*out++ = 0;
			}
		}
}

void ColorizedDepthToDepth(float min_depth, float max_depth, float depth_units, cv::Mat& color_mat, cv::Mat& depth_mat)
{
	auto _width = color_mat.size().width;
	auto _height = color_mat.size().height;

	unsigned short out_value = 0; // from 0 to 256 * 6 = 1536 by Hue colorization
	auto in = reinterpret_cast<const unsigned char*>(color_mat.data);
	auto out = reinterpret_cast<unsigned short*>(depth_mat.data);

	float input{};
	//TODO SSE optimize
	for (auto i = 0; i < _height; i++)
	{
		for (auto j = 0; j < _width; j++)
		{
			auto R = *in++;
			auto G = *in++;
			auto B = *in++;

			auto out_value = RGBtoD(R, G, B);

			if(out_value > 0)
			{
				auto z_value = static_cast<unsigned short>((min_depth + (max_depth - min_depth) * out_value / 1535.0f) / depth_units + 0.5f);
				*out++ = z_value;
			}
			else
			{
				*out++ = 0;
			}
		}
	}
}

void DisparityToDepth(rs2_intrinsics intrinsic, float stereo_baseline_meter, float depth_units, cv::Mat& disparity_mat, cv::Mat& depth_mat)
{
	auto _width = disparity_mat.size().width;
	auto _height = disparity_mat.size().height;

	// same code to librealsense
	auto _focal_lenght_mm = intrinsic.fx;
	const uint8_t fractional_bits = 5;
	const uint8_t fractions = 1 << fractional_bits;
	auto _d2d_convert_factor = (stereo_baseline_meter * _focal_lenght_mm * fractions) / depth_units;

	auto in = reinterpret_cast<const float*>(disparity_mat.data);
	auto out = reinterpret_cast<unsigned short*>(depth_mat.data);

	float input{};
	//TODO SSE optimize
	for (auto i = 0; i < _height; i++)
		for (auto j = 0; j < _width; j++)
		{
			input = *in;
			if (std::isnormal(input))
				*out++ = static_cast<unsigned short>((_d2d_convert_factor / input) + 0.5f);
			else
				*out++ = 0;
			in++;
		}
}

// Compute point cloud: Output with right handed coordinate system as x: right y: down z: forward.
void ComputePointCloud(rs2_intrinsics intrinsic, float depth_units, cv::Mat& depth_mat, cv::Mat& color_mat, std::vector<float>& point_cloud, std::vector<unsigned char>& point_color)
{
	point_cloud.clear();
	point_color.clear();

	for (int y = 0; y < intrinsic.height; y++)
	{
		for (int x = 0; x < intrinsic.width; x++)
		{
			auto depth_value = depth_mat.at<unsigned short>(y, x);
			auto color_r = color_mat.at<cv::Vec3b>(y, x)[0];
			auto color_g = color_mat.at<cv::Vec3b>(y, x)[1];
			auto color_b = color_mat.at<cv::Vec3b>(y, x)[2];

			if (depth_value > 0)
			{
				point_cloud.push_back(depth_value * (x - intrinsic.ppx) / intrinsic.fx * 3.5f * depth_units * 1000.0f);
				point_cloud.push_back(depth_value * (y - intrinsic.ppy) / intrinsic.fy * 3.5f * depth_units * 1000.0f);
				point_cloud.push_back((float)depth_value * depth_units * 1000.0f);

				point_color.push_back(color_r);
				point_color.push_back(color_g);
				point_color.push_back(color_b);
			}
		}
	}
}

int main(int argc, char * argv[]) try
{
	auto image_width = 1280;
	auto image_height = 720;

	auto min_depth = 1.0f;
	auto max_depth = 3.0f;

	// Declare RealSense pipeline, encapsulating the actual device and sensors
	rs2::pipeline pipe;
	rs2::config cfg;
	// Use a configuration object to request only depth from the pipeline
	cfg.enable_stream(RS2_STREAM_DEPTH, image_width, image_height, RS2_FORMAT_Z16, 30);
	cfg.enable_stream(RS2_STREAM_COLOR, image_width, image_height, RS2_FORMAT_BGR8, 30);
	// Start streaming with the above configuration
	pipe.start(cfg);

	rs2::sensor ir_sensor;
	auto active_sensors = pipe.get_active_profile().get_device().query_sensors();
	for (auto target_sensor : active_sensors)
	{
		if (strcmp(target_sensor.get_info(rs2_camera_info::RS2_CAMERA_INFO_NAME), "Stereo Module") == 0) { ir_sensor = target_sensor; }
	}

	// stereo camera settings
	ir_sensor.set_option(rs2_option::RS2_OPTION_DEPTH_UNITS, 0.001f);
	ir_sensor.set_option(rs2_option::RS2_OPTION_LASER_POWER, 60);
	ir_sensor.set_option(rs2_option::RS2_OPTION_ENABLE_AUTO_EXPOSURE, 1);
//	ir_sensor.set_option(rs2_option::RS2_OPTION_EXPOSURE, 6000);

	// Declare filters
	rs2::decimation_filter dec_filter;  // Decimation - reduces depth frame density
	rs2::threshold_filter thr_filter;   // Threshold  - removes values outside recommended range
	rs2::align	aligned_to_color(RS2_STREAM_COLOR);
	rs2::disparity_transform depth_to_disparity(true);	// Converting depth to disparity 
	rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
	rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise
	rs2::disparity_transform disparity_to_depth(false);	// Converting disparity to depth
	rs2::colorizer color_filter;						// Colorize - convert from depth to RGB color
	rs2::rates_printer printer;							// 

	// filter settings
	dec_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 1.0f);
	thr_filter.set_option(RS2_OPTION_MIN_DISTANCE, min_depth);
	thr_filter.set_option(RS2_OPTION_MAX_DISTANCE, max_depth);
	spat_filter.set_option(RS2_OPTION_FILTER_MAGNITUDE, 2.0f);
	spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.5f);
	spat_filter.set_option(RS2_OPTION_FILTER_SMOOTH_DELTA, 20.0f);
	temp_filter.set_option(RS2_OPTION_FILTER_SMOOTH_ALPHA, 0.4f);

	// color filter setting
	color_filter.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 0);
	color_filter.set_option(RS2_OPTION_COLOR_SCHEME, 9.0f);		// Hue colorization
	color_filter.set_option(RS2_OPTION_MAX_DISTANCE, max_depth);
	color_filter.set_option(RS2_OPTION_MIN_DISTANCE, min_depth);

	// Declaring two concurrent queues that will be used to push and pop frames from different threads
	rs2::frame_queue original_data;
	rs2::frame_queue filtered_data;

	// Atomic boolean to allow thread safe way to stop the thread
	std::atomic<bool> stopped(false);
	std::atomic<bool> is_enabled(true);

	bool is_colorized(true);
	bool is_disparity(true);

	// Declare objects that will hold the calculated pointclouds and colored frames
	// We save the last set of data to minimize flickering of the view
	rs2::frame colored_depth;
	rs2::frame colored_filtered;

	// OpenCV matrices
	// for color
	auto rgb_color_mat = cv::Mat(cv::Size(image_width, image_height), CV_8UC3).setTo(0);

	// for depth
	auto depth_mat = cv::Mat(cv::Size(image_width, image_height), CV_16U).setTo(0);
	auto disparity_mat = cv::Mat(cv::Size(image_width, image_height), CV_32F).setTo(0);
	auto rgb_color_depth_mat = cv::Mat(cv::Size(image_width, image_height), CV_8UC3).setTo(0);
	auto bgr_color_depth_mat = cv::Mat(cv::Size(image_width, image_height), CV_8UC3).setTo(0);

	// aligned to color intrinsics
	auto depth_intrinsics = pipe.get_active_profile().get_stream(rs2_stream::RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
	auto depth_units = ir_sensor.get_option(rs2_option::RS2_OPTION_DEPTH_UNITS);
	auto stereo_baseline = ir_sensor.get_option(rs2_option::RS2_OPTION_STEREO_BASELINE);
	auto is_loaded = false;

	cv::TickMeter tick_meter;

	tick_meter.start();

	while (true)
	{
		rs2::frameset frames = pipe.wait_for_frames(); // Wait for next set of frames from the camera
		if (!frames) { break; } // Should not happen but if the pipeline is configured differently
									//  it might not provide depth and we don't want to crash

		rs2::frame filtered;// = frames.get_depth_frame(); //Take the depth frame from the frameset

		// apply post processing filters
		if (is_enabled)
		{
			//			filtered = dec_filter.process(filtered);	// not used
			frames = aligned_to_color.process(frames);
			filtered = frames.get_depth_frame();
			filtered = thr_filter.process(filtered);

			filtered = depth_to_disparity.process(filtered);
			filtered = spat_filter.process(filtered);
			filtered = temp_filter.process(filtered);
			if (!is_disparity) { filtered = disparity_to_depth.process(filtered); }
			if (is_colorized) { filtered = color_filter.process(filtered); }
			filtered = printer.process(filtered);
		}
		else
		{
			filtered = frames.get_depth_frame();
		}


		rs2::frame color_frame = frames.get_color_frame();

		if (is_colorized)
		{
			if (!is_loaded)
			{
				// copy colorized RGB frame to RGB mat
				memcpy(rgb_color_depth_mat.data, filtered.get_data(), sizeof(unsigned char) * rgb_color_depth_mat.size().width * rgb_color_depth_mat.size().height * 3);
				// copy RGB frame to RGB mat
				memcpy(rgb_color_mat.data, color_frame.get_data(), sizeof(unsigned char) * rgb_color_mat.size().width * rgb_color_mat.size().height * 3);
			}

			// depth conversion here
			if (is_disparity)
			{
				ColorizedDisparityToDepth(depth_intrinsics, stereo_baseline, min_depth, max_depth, depth_units, rgb_color_depth_mat, depth_mat);
				cv::imshow("Depth", depth_mat * 10);
			}
			else
			{
				ColorizedDepthToDepth(min_depth, max_depth, depth_units, rgb_color_depth_mat, depth_mat);
				cv::imshow("Depth", depth_mat * 10);
			}

			// for display, color channel is realigned from RGB to BGR
			cv::cvtColor(rgb_color_depth_mat, bgr_color_depth_mat, cv::COLOR_BGR2RGB);
			cv::imshow("Colorized depth", bgr_color_depth_mat);
		}
		else
		{
			if (is_disparity)
			{
				memcpy(disparity_mat.data, filtered.get_data(), sizeof(float) * disparity_mat.size().width * disparity_mat.size().height);
				DisparityToDepth(depth_intrinsics, stereo_baseline * 0.001f, depth_units, disparity_mat, depth_mat);
				cv::imshow("Depth", depth_mat * 10);
			}
			else
			{
				memcpy(depth_mat.data, filtered.get_data(), sizeof(unsigned short) * depth_mat.size().width * depth_mat.size().height);
				cv::imshow("Depth", depth_mat * 10);
			}
		}

		cv::imshow("Color", rgb_color_mat);

		auto in_key = cv::waitKey(1);

		if (in_key == 'q' || in_key == 27) { break; }
		if (in_key == 'c') { is_colorized = !is_colorized; }
		if (in_key == 'd') { is_disparity = !is_disparity; }
		if (in_key == 's')	// save depth as point cloud and RGB image
		{
			std::vector<float> point_cloud;
			std::vector<unsigned char> point_color;
			ComputePointCloud(depth_intrinsics, depth_units, depth_mat, rgb_color_mat, point_cloud, point_color);
			SavePLY("pointcloud.ply", point_cloud, point_color);
		}
		if (in_key == 'w' && is_colorized)	// save depth as image
		{
			std::vector<int> compression_params;
			compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
			compression_params.push_back(80);
			cv::imwrite("Color.JPG", rgb_color_mat, compression_params);
			cv::imwrite("Depth.JPG", rgb_color_depth_mat, compression_params);
		}
		if (in_key == 'l' && is_colorized)	// load depth as image
		{
			if (!is_loaded)
			{
				rgb_color_depth_mat = cv::imread("Depth.JPG");
				rgb_color_mat = cv::imread("Color.JPG");
				is_loaded = true;
			}
			else
			{
				is_loaded = false;
			}

		}

	}

	// Signal the processing thread to stop, and join
	// (Not the safest way to join a thread, please wrap your threads in some RAII manner)
	stopped = true;

	return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
	std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
	return EXIT_FAILURE;
}
catch (const std::exception& e)
{
	std::cerr << e.what() << std::endl;
	return EXIT_FAILURE;
}
