import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
import copy

class Counter():
    
    def __init__(self):
        self.objs = []
        pass
    
    def detect(self, frame):
        
        bboxes = self.detect_bboxes(frame)
        bboxes = self.nms(bboxes)
        print(bboxes)
        for bbox in bboxes:
            obj = TrackedObj(bbox)
            self.objs.append(obj)
    
    def detect_bboxes(self, frame):
        
        bboxes = []
        
        f = copy.deepcopy(frame)
        f.reset()
        for dp in frame.dpoints:
            r, j = dp
            f.draw_point(r, j, frame.rmax)
        
        
        f = cv2.dilate(f.frame, None, iterations= 1)
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,127,255,0)
        
        cnts = cv2.findContours(thresh, cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        
        for c in cnts:
            bbox = cv2.boundingRect(c)
            frame.show_bbox(bbox)
            bboxes.append(bbox) 
        
        return bboxes
    
    def nms(self, bboxes):
        
        b = []
        for bbox in bboxes:
            ok = True
            for obj in self.objs:
                qbox = obj.bbox
                if self.bbox_overlap(bbox, qbox) > 0.6:
                    ok = False
                    break
            
            if ok:
                b.append(bbox)
        
        return b
                
    
    def track(self, frame):
        
        leaving_obj_num = 0
        for obj in self.objs:
            leaving_obj_num += obj.update()
        
        return leaving_obj_num

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
    
    def __init__(self, bbox):
        self.bbox = bbox
    
    
    def update(self):
        pass
    
    
    
    