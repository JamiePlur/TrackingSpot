import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import copy
from FrameHelper import Frame
import FrameHelper

# kernel for erode and dilate operation
UpDownkernel = np.array(
    [[0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 1, 0],
     [0, 1, 1, 1, 0]],
    dtype='uint8')

Rectkernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype='uint8')


class Counter():
    def __init__(self, fh):
        self.objs = []
        self.track_roi = []
        self.detect_roi = []
        self.fh = fh

    def detect(self, frame):
        # delete edge
        bboxes = self.detect_bbox(frame)
        # bboxes = self._detect_bboxes_by_dp(frame)
        bboxes = self.bbox_filter(bboxes, frame)
        for bbox in bboxes:
            obj = TrackedObj(bbox, self.track_roi)
            self.objs.append(obj)
        # print("all objects：", [obj.bbox for obj in self.objs])

    def detect_bbox(self, frame):
        bboxes = []
        f = copy.deepcopy(frame)
        f.reset()
        for point in frame.points:
            r, j, _ = point
            x, y = FrameHelper.convert_coord(r, j, self.fh.rmax)
            dist = cv2.pointPolygonTest(self.fh.bg.data, (x, y), True)
            if dist < 20 and x > 20:  # outside roi
                continue
            else:  # inside roi
                cv2.circle(f.data, (x, y), 4, [255, 255, 255], -1)

        self.track_roi = copy.deepcopy(f)  # save track roi

        f.data = cv2.erode(f.data, Rectkernel, iterations=3)
        f.data = cv2.dilate(f.data, UpDownkernel, iterations=5)

        def grab_contours(data):
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, 0)
            cnts = cv2.findContours(thresh, cv2.RETR_TREE,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            return cnts

        def has_large_bbox(cnts):
            for c in cnts:
                bbox = cv2.boundingRect(c)
                x, y, w, h = bbox
                if w > 50 or h > 50:
                    return True
            return False

        cnts = grab_contours(f.data)
        while (has_large_bbox(cnts)):
            f.data = cv2.erode(f.data, Rectkernel, iterations=1)
            cnts = grab_contours(f.data)

        for c in cnts:
            bbox = cv2.boundingRect(c)
            x, y, w, h = bbox
            if w < 5 or h < 5 or y > 150:  # leaving zone
                print("not detect small obj {}".format(bbox))
                f.data[y:y + h, x:x + w, :] = 0
                pass
            else:
                f.append_bbox(bbox)
                bboxes.append(bbox)

        for b in f.bboxes:
            self.fh.draw_bbox(f, b)

        # cv2.drawContours(f.data, self.fh.bg.data, -1, (0, 0, 255), 10)
        cv2.imshow("detect_roi", f.data)
        cv2.imshow("track_roi", self.track_roi.data)
        return bboxes

    def bbox_filter(self, bboxes, frame):
        b = []
        for bbox in bboxes:
            ok = True
            # filter by shape
            # data = frame.clip_bbox(bbox)
            # f = Frame(data=data)
            # c = self._filter_by_shape(f)
            # if len(c) is not 2:
            #     ok = False
            #     continue
            # if repeat with exsiting obj:
            # remain the detecting obj
            # remove the tracked obj
            for obj in self.objs:
                qbox = obj.bbox
                if self.bbox_overlap(bbox, qbox) > 0.5:
                    self.objs.remove(obj)
            if ok:
                b.append(bbox)
        return b

    def _filter_by_shape(self, f):
        bboxes = []
        f.data = imutils.resize(f.data, width=500)
        # first erode then dilate
        # to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        d = cv2.erode(f.data, kernel, iterations=7)
        # d = cv2.dilate(d, kernel, iterations= 1)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        # convert to binary img
        # and find contours
        ret, thresh = cv2.threshold(d, 127, 255, 0)
        fcnts = cv2.findContours(d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fcnts = imutils.grab_contours(fcnts)

        for c in fcnts:
            if cv2.contourArea(c) < 200:
                continue
            bbox = cv2.boundingRect(c)
            # print(bbox)
            if self._bbox_out_of_bound(bbox, 500):
                continue
            bboxes.append(bbox)
        return bboxes

    def _bbox_out_of_bound(self, bbox, bound):
        if bbox[0] <= 0 or bbox[1] <= 0:
            return True
        if bbox[2] + bbox[0] >= bound \
                or bbox[3] + bbox[1] >= bound:
            return True
        return False

    def track(self, frame):
        leaving_objs = []
        for i, obj in enumerate(self.objs):
            ok = obj.update(self.track_roi)
            if ok:
                if self._is_bbox_leaving(obj):
                    print("obj {} is acutually leaving".format(obj.bbox))
                    ok = False

            if not ok:
                if self._is_bbox_not_leave(obj):
                    print("obj {} is deleted for not leaving".format(obj.bbox))
                    self.objs.remove(obj)
                    continue

                leaving_objs.append(obj)
                self.objs.remove(obj)

            if ok:
                # if self._is_bbox_not_move(obj):
                #     print("obj {} is deleted for not moving".format(obj.bbox))
                #     self.objs.remove(obj)
                #     # points = self.fh.frame.points
                #     # self.fh.bg.revised_by_tracking(obj, points)
                #     continue

                if self._is_bbox_repeated(obj):
                    print("obj {} is deleted for repeatness".format(obj.bbox))
                    self.objs.remove(obj)
                    continue
                # print("obj {}, travel{}, life {}".format(obj.bbox, obj.travel_distance, obj.life))

        leaving_obj_num = len(leaving_objs)
        for obj in self.objs:
            frame.append_bbox(obj.bbox)
        for obj in leaving_objs:
            frame.append_bbox(obj.bbox, color="red")
        return leaving_obj_num

    def _is_bbox_leaving(self, obj):
        x, y, w, h = obj.bbox
        if y > 200:  # leaving zone
            return True
        else:
            return False

    def _is_bbox_repeated(self, obj):
        x, y, w, h = obj.bbox
        max_overlap = 0
        if y < 150:  # if y not in leaving zone
            for qobj in [o for o in self.objs if o is not obj]:
                overlap = self.bbox_overlap(obj.bbox, qobj.bbox)
                max_overlap = max(max_overlap, overlap)
                if overlap > 0.7:
                    return True
        # test overlap
        # print('max_overlap:', max_overlap)
        return False

    def _is_bbox_not_leave(self, obj):
        x, y, w, h = obj.bbox
        # # travel distance should larger than d
        # if obj.travel_distance < 10:
        #     print("too short!")
        #     return True

        # leaving area should in accord with
        # travel direction
        if y < 150:  # if obj not in leaving zone
            print("position wrong!")
            return True
        return False

    def _is_bbox_not_move(self, obj):
        if obj.life < 10:
            return False
        else:
            # print("obj speed:", obj.travel_distance / obj.life)
            if obj.travel_distance / obj.life < 0.5:
                return True
            else:
                return False

    def bbox_overlap(self, box, query_box):
        overlap = 0
        box_area = box[2] * box[3]
        query_box_area = query_box[2] * query_box[3]
        iw = (min(box[0] + box[2], query_box[0] + query_box[2]) - max(
            box[0], query_box[0]))
        if iw > 0:
            ih = (min(box[1] + box[3], query_box[1] + query_box[3]) - max(
                box[1], query_box[1]))
            if ih > 0:
                ua = float(min(query_box_area, box_area))
                overlap = iw * ih / ua
        return overlap


class TrackedObj:
    create_tracker = cv2.TrackerCSRT_create

    def __init__(self, bbox, frame):
        self.bbox = bbox

        self.travel_distance = 0
        x, y, w, h = bbox
        self.start_position = (x + w / 2, y + h / 2)
        self.last_position = self.start_position

        self.life = 0
        self.like_bg_counter = 0
        tracker = self.create_tracker()
        tracker.init(frame.data, bbox)
        self.tracker = tracker

    def update(self, frame):
        ok, bbox = self.tracker.update(frame.data)
        if ok:
            self.bbox = bbox
            x, y, w, h = bbox
            present_position = (x + w / 2, y + h / 2)

            self.travel_distance = abs(self.start_position[0] - present_position[0])\
                + abs(self.start_position[1] - present_position[1])
            self.life += 1
        return ok
