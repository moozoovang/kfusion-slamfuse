/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */


#include <stdint.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <csignal>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>
#include <kernels.h>
#include <Parameters.h>

#include <SLAMBenchAPI.h>
#include <io/SLAMFrame.h>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/DepthSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <metrics/MetricManager.h>
#include <metrics/Phase.h>
#include <timings.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>


#include <vector>







/***
 *  Extra parameters types from KFusion
 */

inline std::ostream & operator<<(std::ostream & out, const int2 & m) {
    return out << "" << m.x << "," << m.y << "" ;
}

inline std::ostream & operator<<(std::ostream & out, const uint3 & m) {
    return out << ""  << m.x << "," << m.y << ","<< m.z << "" ;
}

inline std::ostream & operator<<(std::ostream & out, const float3 & m) {
    return out << ""  << m.x << "," << m.y << ","<< m.z << "" ;
}

inline std::ostream & operator<<(std::ostream & out, const float4 & m) {
    return out << ""  << m.x << "," << m.y << ","<< m.z << ","<< m.w << "" ;
}


template<> inline void  TypedParameter<float3>::copyValue(float3* to,const float3* from) {*(float3*)to = *(float3*)from;;};
template<> inline void  TypedParameter<float3>::setValue(const char* otarg)                  {(*(float3*)_ptr)= atof3<float3>(otarg);};
template<> inline const std::string  TypedParameter<float3>::getValue(const void * ptr)  {
    float3 v = *((float3*)ptr);
    std::ostringstream ss;
    ss << "" << (float) v.x << "," << v.y << "," << v.z<< "" ;
    return ss.str();
}


template<> inline void  TypedParameter<float4>::copyValue(float4* to,const float4* from) {*(float4*)to = *(float4*)from;;};
template<> inline void  TypedParameter<float4>::setValue(const char* otarg)                  {(*(float4*)_ptr)= atof4<float4>(otarg);};
template<> inline const std::string  TypedParameter<float4>::getValue(const void * ptr)  {
    float4 v = *((float4*)ptr);
    std::ostringstream ss;
    ss << "" <<  (float) v.x << "," << v.y << "," << v.z<< "," << v.w<< "" ;
    return ss.str();
}


template<> inline void  TypedParameter<uint3>::copyValue(uint3* to,const uint3* from) {*(uint3*)to = *(uint3*)from;;};
template<> inline void  TypedParameter<uint3>::setValue(const char* otarg)                  {(*(uint3*)_ptr)= atoi3<uint3>(otarg);};
template<> inline const std::string  TypedParameter<uint3>::getValue(const void * ptr)  {
    uint3 v = *((uint3*)ptr);
    std::ostringstream ss;
    ss << "" <<  (float) v.x << "," << v.y << "," << v.z<< "" ;
    return ss.str();
}






/***
 *  Default parameters
 */

const std::vector<int> default_pyramid = {10,5,4};

const float  default_mu = 0.1f;
const float  default_icp_threshold = 1e-5;
const int    default_compute_size_ratio = 1;
const int    default_integration_rate = 2;
const int    default_rendering_rate = 4;
const int    default_tracking_rate = 1;
const uint3  default_volume_resolution = make_uint3(256, 256, 256);
const float3 default_volume_size = make_float3(8.f, 8.f, 8.f);
const float3 default_volume_direction = make_float3(4.f, 4.f, 4.f);
const bool   default_render_volume_fullsize = false;



/***
 *  Parameters
 */




	int compute_size_ratio;
	int integration_rate;
	int rendering_rate;
	int tracking_rate;
	uint3 volume_resolution;
	float3 volume_direction;
	float3 volume_size;
    std::vector<int> pyramid = {0,0,0};
    float mu;
    float icp_threshold;



    /***
     *  KFusion buffers
     */



	uchar3*   inputRGB;
	uint16_t* inputDepth;
	uchar4*   depthRender;
	uchar4*   trackRender ;
	uchar4*   volumeRender ;
	Kfusion*  kfusion;

	bool tracked ;
	bool integrated;

	uint2 inputSize;
	uint2 computationSize;

	float4 camera ;


    /***
     *  Sensors
     */


	slambench::io::DepthSensor *depth_sensor;
	slambench::io::CameraSensor *rgb_sensor;


    /***
     *  Outputs
     */



	slambench::outputs::Output *pose_output;
	slambench::outputs::Output *pointcloud_output;
	slambench::outputs::Output *raycast_output;

	slambench::outputs::Output *rgb_frame_output;
	slambench::outputs::Output *depth_frame_output;
	slambench::outputs::Output *track_frame_output;
	slambench::outputs::Output *render_frame_output;


bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings) {

	/**
	 * Declare parameters
	 */

    slam_settings->addParameter(DiscretParameter<int>({1, 2, 4, 8}, "c", "compute-size-ratio", "Compute ratio", &compute_size_ratio, &default_compute_size_ratio));
    slam_settings->addParameter(DiscretParameter<int>({1, 5, 10, 20, 30} , "r", "integration-rate",   "integration-rate",    &integration_rate,   &default_integration_rate));
    slam_settings->addParameter(DiscretParameter<int>({1, 3, 5, 7, 9} , "t", "tracking-rate",      "tracking-rate",    &tracking_rate, &default_tracking_rate));
    slam_settings->addParameter(TypedParameter<int>("z", "rendering-rate",     "rendering rate",    &rendering_rate, &default_rendering_rate));

    slam_settings->addParameter(DiscretParameter<float>( {0,0.0001,0.00001,0.000001,1}, "l", "icp-threshold",      "icp threshold",    &icp_threshold, &default_icp_threshold));
    slam_settings->addParameter(DiscretParameter<float>({0.025, 0.075, 0.1, 0.2}, "m", "mu",                 "mu",    &mu, &default_mu));
    slam_settings->addParameter(TypedParameter<float3>("s", "volume-size",        "volume-size",    &volume_size, &default_volume_size));
    slam_settings->addParameter(TypedParameter<float3>("d", "volume-direction",        "volume-direction",    &volume_direction, &default_volume_direction));
    slam_settings->addParameter(DiscretParameter<uint3>( {{64,64,64}, {128,128,128}, {256,256,256}, {512,512,512}  } , "v", "volume-resolution",  "volume-resolution",    &volume_resolution, &default_volume_resolution));
    slam_settings->addParameter(DiscretParameter<int>({3, 5, 7, 9, 11} , "y1", "pyramid-level1",     "pyramid-level1",    &(pyramid[0]), &(default_pyramid[0])));
    slam_settings->addParameter(DiscretParameter<int>({3, 5, 7, 9, 11} , "y2", "pyramid-level2",     "pyramid-level2",    &(pyramid[1]), &(default_pyramid[1])));
    slam_settings->addParameter(DiscretParameter<int>({3, 5, 7, 9, 11} , "y3", "pyramid-level3",     "pyramid-level3",    &(pyramid[2]), &(default_pyramid[2])));


    return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings)  {


	/**
	 * Declare Phases
	 */

    slam_settings->GetMetricManager().AddPhase("Preprocessing");
    slam_settings->GetMetricManager().AddPhase("Tracking");
    slam_settings->GetMetricManager().AddPhase("Integration");
    slam_settings->GetMetricManager().AddPhase("Raycasting");
    slam_settings->GetMetricManager().AddPhase("Render");



	/**
	 * Declare Outputs
	 */

    pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);

    pointcloud_output = new slambench::outputs::Output("PointCloud", slambench::values::VT_POINTCLOUD, true);
    pointcloud_output->SetKeepOnlyMostRecent(true);

    raycast_output = new slambench::outputs::Output("Raycast", slambench::values::VT_POINTCLOUD);
    raycast_output->SetKeepOnlyMostRecent(true);

    slam_settings->GetOutputManager().RegisterOutput(pose_output);
    slam_settings->GetOutputManager().RegisterOutput(pointcloud_output);
    slam_settings->GetOutputManager().RegisterOutput(raycast_output);

    rgb_frame_output = new slambench::outputs::Output("RGB Frame", slambench::values::VT_FRAME);
    rgb_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(rgb_frame_output);

    depth_frame_output = new slambench::outputs::Output("Depth Frame", slambench::values::VT_FRAME);
    depth_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(depth_frame_output);

    track_frame_output = new slambench::outputs::Output("Tracking Frame", slambench::values::VT_FRAME);
    track_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(track_frame_output);

    render_frame_output = new slambench::outputs::Output("Rendered frame", slambench::values::VT_FRAME);
    render_frame_output->SetKeepOnlyMostRecent(true);
    slam_settings->GetOutputManager().RegisterOutput(render_frame_output);


	/**
	 * Inspect sensors
	 */

	slambench::io::CameraSensorFinder sensor_finder;
	rgb_sensor = sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "rgb"}});
	depth_sensor = (slambench::io::DepthSensor*)sensor_finder.FindOne(slam_settings->get_sensors(), {{"camera_type", "depth"}});

    if ((rgb_sensor == nullptr) || (depth_sensor == nullptr)) {
        std::cerr << "Invalid sensors found, RGB or Depth not found." << std::endl;
        return false;
    }

	if(rgb_sensor->FrameFormat != slambench::io::frameformat::Raster) {
		std::cerr << "RGB data is in wrong format" << std::endl;
		return false;
	}
	if(depth_sensor->FrameFormat != slambench::io::frameformat::Raster) {
		std::cerr << "Depth data is in wrong format" << std::endl;
		return false;
	}
	if(rgb_sensor->PixelFormat != slambench::io::pixelformat::RGB_III_888) {
		std::cerr << "RGB data is in wrong format pixel" << std::endl;
		return false;
	}
	if(depth_sensor->PixelFormat != slambench::io::pixelformat::D_I_16) {
		std::cerr << "Depth data is in wrong pixel format" << std::endl;
		return false;
	}

	if(rgb_sensor->Width != depth_sensor->Width || rgb_sensor->Height != depth_sensor->Height) {
		std::cerr << "Sensor size mismatch" << std::endl;
		return false;
	}

	/**
	 * Inspect Parameters
	 */


    assert(compute_size_ratio > 0);
    assert(integration_rate > 0);
    assert(volume_size.x > 0);
    assert(volume_resolution.x > 0);

    inputSize = make_uint2(depth_sensor->Width,depth_sensor->Height);
    std::cerr << "input Size is = " << inputSize.x << "," << inputSize.y
            << std::endl;
    computationSize = make_uint2(
                inputSize.x / compute_size_ratio,
                inputSize.y / compute_size_ratio);


    camera =  make_float4(
			depth_sensor->Intrinsics[0],
			depth_sensor->Intrinsics[1],
			depth_sensor->Intrinsics[2],
			depth_sensor->Intrinsics[3]);

	camera.x = camera.x *  computationSize.x;
	camera.y = camera.y *  computationSize.y;
	camera.z = camera.z *  computationSize.x;
	camera.w = camera.w *  computationSize.y;

	std::cerr << "camera is = " << camera.x  << "," << camera.y  << "," << camera.z  << "," << camera.w
			<< std::endl;


	/**
	 * Allocate buffers
	 */


	 inputRGB     = new uchar3[inputSize.x * inputSize.y];
	 inputDepth   = new uint16_t[inputSize.x * inputSize.y];
	 depthRender  = new uchar4[computationSize.x * computationSize.y];
	 trackRender  = new uchar4[computationSize.x * computationSize.y];
	 volumeRender = new uchar4[computationSize.x * computationSize.y];

	 /**
	  * Start KFusion
	  */


		Matrix4 poseMatrix;
		poseMatrix.data[0].x = depth_sensor->Pose(0,0) ;
		poseMatrix.data[0].y = depth_sensor->Pose(0,1) ;
		poseMatrix.data[0].z = depth_sensor->Pose(0,2) ;
		poseMatrix.data[0].w = depth_sensor->Pose(0,3) + volume_direction.x;

		poseMatrix.data[1].x = depth_sensor->Pose(1,0);
		poseMatrix.data[1].y = depth_sensor->Pose(1,1);
		poseMatrix.data[1].z = depth_sensor->Pose(1,2);
		poseMatrix.data[1].w = depth_sensor->Pose(1,3) + volume_direction.y;

		poseMatrix.data[2].x = depth_sensor->Pose(2,0);
		poseMatrix.data[2].y = depth_sensor->Pose(2,1);
		poseMatrix.data[2].z = depth_sensor->Pose(2,2);
		poseMatrix.data[2].w = depth_sensor->Pose(2,3)+ volume_direction.z;

		poseMatrix.data[3].x = depth_sensor->Pose(3,0);
		poseMatrix.data[3].y = depth_sensor->Pose(3,1);
		poseMatrix.data[3].z = depth_sensor->Pose(3,2);
		poseMatrix.data[3].w = depth_sensor->Pose(3,3);



	kfusion = new Kfusion(computationSize, volume_resolution, volume_size, poseMatrix, pyramid);

	return true;

}


bool sb_update_frame (SLAMBenchLibraryHelper * , slambench::io::SLAMFrame* s) {

	// TODO: this is ugly
	static bool depth_ready = false;
	static bool rgb_ready = false;



	assert(s != nullptr);

	char *target = nullptr;

	if(s->FrameSensor == depth_sensor) {
		target = (char*)inputDepth;
		depth_ready = true;
	} else if(s->FrameSensor == rgb_sensor) {
		target = (char*)inputRGB;
		rgb_ready = true;
	} else {
	  //std::cerr << "Unexpected sensor " << s->FrameSensor << ":" << s->FrameSensor->Description << std::endl;
	}

	if(target != nullptr) {
		memcpy(target, s->GetData(), s->GetSize());
		s->FreeData();
	}

	if (depth_ready && rgb_ready) {
		depth_ready = false;
		rgb_ready = false;
		return true;
	} else {
		return false;
	}
}

bool sb_process_once (SLAMBenchLibraryHelper * slam_settings)  {

	static int frame = 0;

	auto &metrics = slam_settings->GetMetricManager();
	slambench::metrics::Phase *preprocessing = metrics.GetPhase("Preprocessing");
	slambench::metrics::Phase *tracking = metrics.GetPhase("Tracking");
	slambench::metrics::Phase *integration = metrics.GetPhase("Integration");
	slambench::metrics::Phase *raycasting = metrics.GetPhase("Raycasting");
	slambench::metrics::Phase *rendering = metrics.GetPhase("Render");

	preprocessing->Begin();
	kfusion->preprocessing(inputDepth, inputSize);
	preprocessing->End();

	tracking->Begin();
	tracked = kfusion->tracking(camera, icp_threshold, tracking_rate, frame);
	tracking->End();

	integration->Begin();
	integrated = kfusion->integration(camera, integration_rate, mu, frame);
	integration->End();

	raycasting->Begin();
    kfusion->raycasting(camera, mu, frame);
	raycasting->End();

    rendering->Begin();
    kfusion->renderDepth(depthRender, computationSize);
    kfusion->renderTrack(trackRender, computationSize);
    kfusion->renderVolume(volumeRender, computationSize, frame, rendering_rate, camera, 0.75 * mu);
	rendering->End();


    frame++;

    return true;

}

bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *ts_p) {
	(void)lib;

	slambench::TimeStamp ts = *ts_p;

	if(pose_output->IsActive()) {
		// Get the current pose as an eigen matrix
		Matrix4 pose = kfusion->getPose();

		Eigen::Matrix4f mat;
		for (int i = 0 ; i < 4 ; i++) {
			mat(i,0) = pose.data[i].x;
			mat(i,1) = pose.data[i].y;
			mat(i,2) = pose.data[i].z;
			mat(i,3) = pose.data[i].w;
		}

		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		pose_output->AddPoint(ts, new slambench::values::PoseValue(mat));
	}

	if(pointcloud_output->IsActive()) {
		auto map = kfusion->getMap();

		// Take lock only after generating the map
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		if(pointcloud_output->IsActive()and map) {
			pointcloud_output->AddPoint(ts, map);
		}
	}

	if(raycast_output->IsActive()) {
		auto map = kfusion->getRaycast();
		// Take lock only after generating the map
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		if(raycast_output->IsActive() and map) {
			raycast_output->AddPoint(ts, map);
		}
	}

	if(rgb_frame_output->IsActive()) {
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		rgb_frame_output->AddPoint(ts, new slambench::values::FrameValue(inputSize.x, inputSize.y, slambench::io::pixelformat::RGB_III_888, inputRGB));
	}
	if(depth_frame_output->IsActive()) {
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		depth_frame_output->AddPoint(ts, new slambench::values::FrameValue(computationSize.x, computationSize.y, slambench::io::pixelformat::RGBA_IIII_8888, depthRender));
	}
	if(track_frame_output->IsActive()) {
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		track_frame_output->AddPoint(ts, new slambench::values::FrameValue(computationSize.x, computationSize.y, slambench::io::pixelformat::RGBA_IIII_8888, trackRender));
	}
	if(render_frame_output->IsActive()) {
		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		render_frame_output->AddPoint(ts, new slambench::values::FrameValue(computationSize.x, computationSize.y, slambench::io::pixelformat::RGBA_IIII_8888, volumeRender));
	}

	return true;
}

bool sb_clean_slam_system() {
    delete kfusion;
    delete[] inputRGB;
    delete[] inputDepth;
    delete[] depthRender;
    delete[] trackRender;
    delete[] volumeRender;
    return true;
}





bool sb_get_tracked  (bool* tracking)  {
    *tracking = tracked;
    return true;
}






















