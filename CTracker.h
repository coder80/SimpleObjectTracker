#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <unordered_map>
#include <vector>
using namespace cv;

constexpr uint32_t maxTrackObjects = 16;
constexpr float iouTreshold = 0.1f;

class CTracker
{
public:
	CTracker();
	~CTracker();
	void addRect(const Mat& frame, const Rect2d& rec);
	bool trackerUpdate(const Mat& frame, Rect2d& rec);
	std::vector<Rect2d> getObjects(const Mat& frame);
	bool iou(const Rect2d& rec1, const Rect2d& rec2);

private:

	void updateTrackerId();
	std::unordered_map<int64_t, Ptr<Tracker> > mTrackers;
	std::unordered_map<int64_t, Rect2d> mObjects;
	int64_t mTrackerId = 0;
};

