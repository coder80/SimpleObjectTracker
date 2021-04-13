#include "CTracker.h"

CTracker::CTracker()
{
}

CTracker::~CTracker()
{
}

void CTracker::addRect(const Mat& frame, const Rect2d& rec) {
    bool b = false;
    for (auto obj : mObjects) {
        if (iou(obj.second, rec)) {
            b = true;
            break;
        }
    }

    if (b) {
        return;
    }

    TrackerCSRT::Params param;
    param.number_of_scales = 75;
    param.scale_step = 1.020f;
    param.admm_iterations = 40;
    param.scale_sigma_factor = 0.99;//0.250f;
    param.scale_model_max_area = 512; //512.0f;
    param.scale_lr = 0.025f;//0.025f;
    mTrackers[mTrackerId] = TrackerCSRT::create(param);
    auto tracker = mTrackers[mTrackerId];
    tracker->init(frame, rec);
    mObjects[mTrackerId] = std::move(rec);
    updateTrackerId();
}

bool  CTracker::trackerUpdate(const Mat& frame, Rect2d& rec) {
	auto tracker = mTrackers[mTrackerId];
	bool ok = tracker->update(frame, rec);
	return ok;
}

std::vector<Rect2d>  CTracker::getObjects(const Mat& frame) {
	std::vector<Rect2d> result;
	std::vector<int64_t> objForDelete;
	result.reserve(maxTrackObjects);
	for (auto& obj : mTrackers) {
        Rect2d rec;
        auto& tracker = obj.second;
        bool ok = tracker->update(frame, rec);
        ok = ok && (rec.x + rec.width < frame.size().width)
            && (rec.y + rec.height < frame.size().height)
            && (rec.y > 0)
            && (rec.x > 0)
            && (rec.width <= frame.size().width / 2)
            && (rec.height <= frame.size().height / 2);

        if (ok) {
            mObjects[obj.first] = rec;
            result.push_back(std::move(rec));
        }
        else
        {
            objForDelete.push_back(obj.first);
        }
	}

    for (auto i : objForDelete) {
        mTrackers[i].release();
        mTrackers.erase(i);
        mObjects.erase(i);
    }

	return result;
}

bool CTracker::iou(const Rect2d& rec1, const Rect2d& rec2) {
    float intersaction = (rec1 & rec2).area();
    float unionrect = (rec1 | rec2).area();
    float iouf = intersaction / (unionrect + std::numeric_limits<float>::min());
    return (iouf > iouTreshold);
}

void CTracker::updateTrackerId() {
    if (mTrackerId > std::numeric_limits<int64_t>::max() - 1) {
        mTrackerId = 0;
    }

    ++mTrackerId;
}
