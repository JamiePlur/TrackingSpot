import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import copy
from FrameHelper import Frame


class Counter():

    def __init__(self, fh):
        self.objs = []
        self.frame_helper = fh

    def detect(self, frame):
        # delete edge:
        bboxes = self._detect_bboxes_by_dp(frame)
        bboxes = self._filter(bboxes, frame)
        for bbox in bboxes:
            obj = TrackedObj(bbox, frame)
            self.objs.append(obj)
        # print("all objects：", [obj.bbox for obj in self.objs])

    def _filter_by_shape(self, f):
        bboxes = []

        f.data = imutils.resize(f.data, width=500)
        # first erode then dilate
        # to remoce small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        d = cv2.erode(f.data, kernel, iterations=7)
        # d = cv2.dilate(d, kernel, iterations= 1)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        # convert to binary img
        # and find contours
        ret, thresh = cv2.threshold(d, 127, 255, 0)
        fcnts = cv2.findContours(d, cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
        fcnts = imutils.grab_contours(fcnts)

        for c in fcnts:
            if cv2.contourArea(c) < 200:
                continue
            bbox = cv2.boundingRect(c)
            # print(bbox)
            if self._bbox_out_of_bound(bbox, 500):
                continue
            bboxes.append(bbox)

        # print("number of cnts:",len(bboxes))
        # f.show()

        return bboxes

    def _bbox_out_of_bound(self, bbox, bound):
        if bbox[0] <= 0 or bbox[1] <= 0:
            return True
        if bbox[2] + bbox[0] >= bound \
                or bbox[3] + bbox[1] >= bound:
            return True
        return False

    def _detect_bboxes_by_dp(self, frame):
        bboxes = []

        f = copy.deepcopy(frame)
        f.reset()
        for dp in frame.dpoints:
            r, j, _ = dp
            f.append_point(r, j)
        for point in f.points:
            self.frame_helper.draw_point(f, point)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        f.data = cv2.dilate(f.data, kernel, iterations=4)
        f.data = cv2.erode(f.data, kernel, iterations=2)

        gray = cv2.cvtColor(f.data, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        cnts = cv2.findContours(thresh, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            bbox = cv2.boundingRect(c)
            f.append_bbox(bbox)
            bboxes.append(bbox)

        for b in f.bboxes:
            self.frame_helper.draw_bbox(f, b)

        cv2.imshow("roi", f.data)
        return bboxes

    def _filter(self, bboxes, frame):
        b = []
        for bbox in bboxes:
            ok = True
            for obj in self.objs:
                qbox = obj.bbox
                if self.bbox_overlap(bbox, qbox) > 0.5:
                    ok = False
                    break
            # filter by size
            x, y, w, h = bbox
            if w < 10 or h < 10:
                ok = False
            if w > 80 and h > 80:
                ok = False
            if h/w < 1:
                ok = False
            # filter by shape
            data = frame.clip_bbox(bbox)
            f = Frame(data=data)
            c = self._filter_by_shape(f)
            if len(c) is not 2:
                ok = False
            # filter by positon
            x1, y1, x2, y2 = x, y, x+w, y+h
            if y1 < 50 or y2 > 370:
                ok = False
            if ok:
                b.append(bbox)
        return b

    def track(self, frame):
        leaving_objs = []
        for i, obj in enumerate(self.objs):
            ok = obj.update(frame)
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
                if self._is_bbox_like_bg(obj):
                    print("obj {} is deleted for liking bg".format(obj.bbox))
                    self.objs.remove(obj)
                    points = self.frame_helper.frame.points
                    self.frame_helper.bg.revised_by_tracking(obj, points)
                    continue

                if self._is_bbox_not_move(obj):
                    print("obj {} is deleted for not moving".format(obj.bbox))
                    self.objs.remove(obj)
                    points = self.frame_helper.frame.points
                    self.frame_helper.bg.revised_by_tracking(obj, points)
                    continue

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
        # start_position = (x+w/2, y+h/2)
        if y < 0 or y > 450:
            return True
        else:
            return False

    def _is_bbox_repeated(self, obj):
        for qobj in [o for o in self.objs if o is not obj]:
            overlap = self.bbox_overlap(obj.bbox, qobj.bbox)
            if overlap > 0.7:
                return True
        return False

    def _is_bbox_not_leave(self, obj):
        x, y, w, h = obj.bbox
        if obj.travel_distance < 10:
            print("too short!")
            return True
        y1, y2 = y, y+h
        if y1 > 50 and y2 < 200:
            print("position wrong!")
            return True
        return False

    def _is_bbox_not_move(self, obj):
        if obj.life < 10:
            return False
        else:
            print("obj speed:", obj.travel_distance / obj.life)
            if obj.travel_distance / obj.life < 0.5:
                return True
            else:
                return False

    def _is_bbox_like_bg(self, obj):
        x, y, w, h = obj.bbox
        if w > 40 or h > 40:
            return True

        # j_start = int(np.arctan(y/(x+w))*250*2/np.pi)
        # j_end = int(np.arctan((y+h)/x)*250*2/np.pi)
        # dpoints = self.frame_helper.frame.dpoints
        # dp_ind = [dpoints[x][1] for x in range(len(dpoints))]

        # ok = False
        # for j in range(j_start, j_end+1):
        #     if j in dp_ind:
        #         ok = True
        #         break
        # if not ok:
        #     print("not include dp!")
        #     return True

        return False

    def bbox_overlap(self, box, query_box):
        overlap = 0
        box_area = box[2] * box[3]
        query_box_area = query_box[2] * query_box[3]
        iw = (
            min(box[0] + box[2], query_box[0] + query_box[2]) -
            max(box[0], query_box[0]) 
        )
        if iw > 0:
            ih = (
                min(box[1] + box[3], query_box[1] + query_box[3]) -
                max(box[1], query_box[1]) 
            )
            if ih > 0:
                ua = float(
                    min(query_box_area, box_area)
                )
                overlap = iw * ih / ua
        return overlap


class TrackedObj:
    create_tracker = cv2.TrackerCSRT_create

    def __init__(self, bbox, frame):
        self.bbox = bbox

        self.travel_distance = 0
        x, y, w, h = bbox
        self.start_position = (x+w/2, y+h/2)
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
            present_position = (x+w/2, y+h/2)
            # one_travel_distance = abs(self.last_position[0] - present_position[0])\
            #     + abs(self.last_position[1] - present_position[1])
            # self.last_position = present_position
            # # print("the td of {} is {}:".format(bbox, one_travel_distance))
            # if one_travel_distance < 1:
            #     self.like_bg_counter += 1
            # else:
            #     self.like_bg_counter = 0

            self.travel_distance = abs(self.start_position[0] - present_position[0])\
                + abs(self.start_position[1] - present_position[1])
            self.life += 1
        return ok
