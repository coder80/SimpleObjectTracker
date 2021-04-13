#include "stdafx.h"
#include <fstream>
#include <sstream>

#include <iostream>


#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "CTracker.h"

using namespace cv;
using namespace dnn;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

void postprocess(Mat& frame, const std::vector<Mat>& out, Net& net);

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

void drawCar(int left, int top, int right, int bottom, Mat& frame);

void callback(int pos, void* userdata);

std::vector<String> getOutputsNames(const Net& net);

CTracker trackers;

int main(int argc, char** argv)
{

	// cv::setNumThreads(0);
	std::string fileName = "C:\\Users\\andy\\Downloads\\Videos\\iDS2CD9396AIS 20180815AIC42956209_20190226123500_20190226125000.mp4";

	if (argc > 1) {
		fileName = std::string(argv[1]);
	}

	//std::string modelConfiguration = "/home/andrei_morozov/diskb/workspace/MobilNet_SSD_opencv/MobileNetSSD_deploy.prototxt";
	//std::string modelBinary = "/home/andrei_morozov/diskb/workspace/MobilNet_SSD_opencv/MobileNetSSD_deploy.caffemodel";

	std::string modelConfiguration = "MobileNetSSD_deploy.prototxt";
	std::string modelBinary = "MobileNetSSD_deploy.caffemodel";

	

	confThreshold = 0.9f;
	nmsThreshold = 0.4f;
	float scale = 1.0f / 255.0f;
	Scalar mean = Scalar(127.5, 127.5, 127.5);
	bool swapRB = true;
	int inpWidth = 300;
	int inpHeight = 300;

	int frameWidth = 380;
	int frameHeight = 320;

	// Open file with classes names.

	//VOC classes

	/*classes = std::vector<std::string>({"background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
			"motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"});*/

	classes = std::vector<std::string>({ "background", "person", "bicycle", "car", "motorcycle",
			"airplane", "bus", "train", "truck", "boat", "light", "fire hydrant", "street sign", " stop sign",
			"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep" });



	// Load a model.

	Net net = readNet(modelBinary, modelConfiguration, "");
	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Create a window
	static const std::string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	int initialConf = (int)(confThreshold * 100);
	createTrackbar("Confidence threshold, %", kWinName, &initialConf, 99, callback);
	
	// Open a video file or an image file or a camera stream.
	VideoCapture cap;

	cap.open(fileName);

	// Process frames.
	Mat frame, blob;

	while (waitKey(1) < 0)
	{
		cap >> frame;
		if (frame.empty())
		{
			waitKey();
			break;
		}
		Rect roi(200, 200, frame.size().width * 0.8f - 200, frame.size().height - 200);
		frame = frame(roi);
		cv::resize(frame, frame, cv::Size(800, 600));
		// Create a 4D blob from a frame.
		Size inpSize(inpWidth > 0 ? inpWidth : frame.cols,
			inpHeight > 0 ? inpHeight : frame.rows);
		blobFromImage(frame, blob, scale, inpSize, mean, swapRB, false);

		// Run a model.
		net.setInput(blob);
		if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
		{
			resize(frame, frame, inpSize);
			Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
			net.setInput(imInfo, "im_info");
		}
		std::vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));
		postprocess(frame, outs, net);

		// Put efficiency information.
		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		std::string label = format("Inference time: %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
		imshow(kWinName, frame);
	}
	return 0;
}

void postprocess(Mat& frame, const std::vector<Mat>& outs, Net& net)
{
	static std::vector<int> outLayers = net.getUnconnectedOutLayers();
	static std::string outLayerType = net.getLayer(outLayers[0])->type;

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;
	if (net.getLayer(0)->outputNameToIndex("im_info") != -1)  // Faster-RCNN or R-FCN
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float* data = (float*)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > confThreshold)
			{
				int left = (int)data[i + 3];
				int top = (int)data[i + 4];
				int right = (int)data[i + 5];
				int bottom = (int)data[i + 6];
				int width = right - left + 1;
				int height = bottom - top + 1;
				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "DetectionOutput")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float* data = (float*)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > confThreshold)
			{
				int left = (int)(data[i + 3] * frame.cols);
				int top = (int)(data[i + 4] * frame.rows);
				int right = (int)(data[i + 5] * frame.cols);
				int bottom = (int)(data[i + 6] * frame.rows);
				int width = right - left + 1;
				int height = bottom - top + 1;

				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "Convolution")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() == 1);
		float* data = (float*)outs[0].data;
		for (size_t i = 0; i < outs[0].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > confThreshold)
			{
				int left = (int)(data[i + 3] * frame.cols);
				int top = (int)(data[i + 4] * frame.rows);
				int right = (int)(data[i + 5] * frame.cols);
				int bottom = (int)(data[i + 6] * frame.rows);
				int width = right - left + 1;
				int height = bottom - top + 1;

				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}
	else if (outLayerType == "Region")
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				Point classIdPoint;
				double confidence;
				minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
				if (confidence > confThreshold)
				{
					int centerX = (int)(data[0] * frame.cols);
					int centerY = (int)(data[1] * frame.rows);
					int width = (int)(data[2] * frame.cols);
					int height = (int)(data[3] * frame.rows);
					int left = centerX - width / 2;
					int top = centerY - height / 2;

					classIds.push_back(classIdPoint.x);
					confidences.push_back((float)confidence);
					boxes.push_back(Rect(left, top, width, height));
				}
			}
		}
	}
	else
		CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);

	std::vector<int> indices;
	auto objects = trackers.getObjects(frame);
	for (auto& obj : objects)
	{
		int idx = 0;
		Rect box = obj;
		drawCar(box.x, box.y,
			box.x + box.width, box.y + box.height, frame);

	}

	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		/*if (classes[classIds[idx]] == "car" |
			classes[classIds[idx]] == "bus" |
			classes[classIds[idx]] == "motorbike") {
			std::cerr <<  classes[classIds[idx]] << std::endl;
		}*/
		trackers.addRect(frame, box);
		//drawPred(classIds[idx], confidences[idx], box.x, box.y,
			//box.x + box.width, box.y + box.height, frame);

	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0));

	std::string label = format("%.2f", conf);

	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ": " + label;
	}

	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - labelSize.height),
		Point(left + labelSize.width, top + baseLine), Scalar::all(255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar());
}

void drawCar(int left, int top, int right, int bottom, Mat& frame)
{
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 255, 0), 2);
}

void callback(int pos, void*)
{
	confThreshold = pos * 0.01f;
}

std::vector<String> getOutputsNames(const Net& net)
{
	static std::vector<String> names;
	if (names.empty())
	{
		std::vector<int> outLayers = net.getUnconnectedOutLayers();
		std::vector<String> layersNames = net.getLayerNames();
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}
