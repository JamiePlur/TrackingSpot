import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import copy
from FrameHelper import Frame

class Counter():
    
    def __init__(self):
        self.objs = []
        pass
    
    def detect(self, frame):
        
        bboxes = self._detect_bboxes_by_dp(frame)
        bboxes = self._filter(bboxes, frame)
        print("在这一帧检测到的新目标有：", bboxes)
        for bbox in bboxes:
            obj = TrackedObj(bbox, frame)
            self.objs.append(obj)
        print("共有目标：", [obj.bbox for obj in self.objs])
    
    def _filter_by_shape(self, frame):
        
        bboxes = []

        f = copy.deepcopy(frame)
        f.data = imutils.resize(f.data, width = 500)
        # first erode then dilate
        # to remoce small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        d = cv2.erode(f.data, kernel, iterations = 7)
        # d = cv2.dilate(d, kernel, iterations= 1)
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        # convert to binary img
        # and find contours
        ret, thresh = cv2.threshold(d, 127, 255, 0)
        fcnts = cv2.findContours(d, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        fcnts= imutils.grab_contours(fcnts)

        for c in fcnts:

            if cv2.contourArea(c) < 200:
                continue
            bbox = cv2.boundingRect(c)
            # print(bbox)           
            if self._bbox_out_of_bound(bbox, 500):
                continue
            bboxes.append(bbox)
            f.draw_bbox(bbox)

        # print("轮廓数为",len(bboxes))
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
            r, j = dp
            f.draw_point(r, j, frame.rmax)
        
        f.data = cv2.dilate(f.data, None, iterations= 3)
        gray = cv2.cvtColor(f.data, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, 0)
        cnts = cv2.findContours(thresh, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            bbox = cv2.boundingRect(c)
            f.draw_bbox(bbox)
            bboxes.append(bbox) 
        
        f.show(win = 'img2')
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
                
            x, y, w, h = bbox
            if w < 10 or h < 10: #不能太小
                ok = False
            if w > 30 or h > 30:  #不能太大
                ok = False
            
            data = frame.clip_bbox(bbox)
            f = Frame(data = data, rmax = frame.rmax)
            c = self._filter_by_shape(f)

            if len(c) is not 2:
                ok = False

            if ok:
                print("检测成功，联通域为", len(c))
                b.append(bbox)
        
        
        return b
                
    def track(self, frame):
        
        leaving_objs = []

        for i, obj in enumerate(self.objs):
            ok = obj.update(frame)
            if not ok:
                leaving_objs.append(obj)
                self.objs.remove(obj)
            if ok:
                if self._is_repeated(obj):
                    print("目标疑似重复，删除")
                    self.objs.remove(obj)
                    
                print("目标{}跟踪成功".format(obj.bbox))

                
        leaving_obj_num = len(leaving_objs)
        
        for obj in self.objs:
            frame.draw_bbox(obj.bbox)
        
        for obj in leaving_objs:
            frame.draw_bbox(obj.bbox, color = "red")
        
        return leaving_obj_num
    
    def _is_repeated(self, obj):
        
        for qobj in [o for o in self.objs if o is not obj]:
            overlap = self.bbox_overlap(obj.bbox, qobj.bbox)
            if overlap > 0.7:
                return True
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
        tracker = self.create_tracker()
        tracker.init(frame.data, bbox)
        self.tracker = tracker
        
    def update(self, frame):
        
        ok, bbox = self.tracker.update(frame.data)
        if ok:
            self.bbox = bbox
        return ok
        
    
    
    
    