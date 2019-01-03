import numpy as np
import cv2
import imutils
from cv2 import VideoWriter,VideoWriter_fourcc
"""
    运动检测算法采用已高斯背景建模为基础算法的mog算法
"""
#mog2 = cv2.createBackgroundSubtractorMOG2()
mog = cv2.bgsegm.createBackgroundSubtractorMOG()

"""
    跟踪算法采用的是CSRT
"""
create_tracker = cv2.TrackerCSRT_create


video_dir = 'test_5.avi'

trackers = []
objs = []
move_states = []

def show_coord(event,x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("坐标是",x,y)
        
def bbox_overlap(box, query_box):
    """
        这里overlap于传统detection中不同
        由于更加关注是否一个框是否包括另一个框的情况
        所以在判断时，计算的是重叠面积/较小的面积
    """
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

def detect_new_objects(frame):
    
    def is_tracked(new_obj):          
        for obj in objs:
            overlap = bbox_overlap(new_obj, obj)
            if overlap > 0.1:
                return True
        return False
    
    def is_suspicious(new_obj, frame):
        (x, y, w, h) = new_obj
        if w < 15 and h < 15: #不能太小
            return False
        if w > 30 or h > 30:  #不能太大
            return False
        area = frame[y:y + h, x:x + w]
        if area.sum() < 30:
            return False
        return True
    
    new_objs = []
    
    #前景检测   
    roi = frame[0:260,0:30]
    fgmask = mog.apply(roi)
    
    #根据前景找到感兴趣的轮廓    
    fgmask= cv2.dilate(fgmask, None, iterations= 1)
    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        new_obj = cv2.boundingRect(c)       
        #这里对检测框的形状作初步的筛除
        if not is_suspicious(new_obj, frame):
            continue
        
        if not is_tracked(new_obj):
            new_objs.append(new_obj)
        
    return new_objs



def track_all_objs(frame):
    
    def update_move_state(bbox, i):
        move_states[i]['exit'] = bbox
        (x1, y1, w1, h1) = bbox
        (x2, y2, w2, h2) = objs[i]
        dx = abs(x1 - x2) + abs(w1 - w2)
        dy = abs(y1 - y2) + abs(y1 - y2)
        d = dx + dy
        move_states[i]['distance'] += d
        move_states[i]['time'] += 1
    
    def is_repeated(bbox):
        """
            根据检测框和其他检测框的overlap,判断是否重复
        """
        for obj in [obj for obj in objs if obj != bbox]:
            overlap = bbox_overlap(obj, bbox)
            if overlap > 0.7:
                return True
        return False
    
    def is_losing_track(obj):
        if obj[0] > 5.5 or obj[1] < 30:
            print("被认为跟丢了",obj)
            return True
        return False
        
    leaving_objs = []
    for i, tracker in enumerate(trackers):
        ok, bbox = tracker.update(frame)
        if ok:
            update_move_state(bbox, i)
            objs[i] = bbox
            if is_repeated(bbox):
                del trackers[i]
                del objs[i]
                del move_states[i]
                
        else:
            if not is_losing_track(objs[i]):
                leaving_objs.append(objs[i])
            del trackers[i]
            del objs[i]
            del move_states[i]
    return leaving_objs


               

if __name__ == "__main__":
    
    
    
    cap = cv2.VideoCapture(video_dir)
    fourcc=VideoWriter_fourcc(*"MJPG")
    videoWriter=cv2.VideoWriter(video_dir[:-4] + "_count.avi",fourcc,25,(500, 500))
    count = 0
    
    ind = 0
    last_frame = None
    
    while(1):
        
        timer = cv2.getTickCount()
        #从视频流取帧
        ok, frame = cap.read()
        if not ok:
            break
        ind += 1
        frame = imutils.resize(frame, width = 500)
        
        #异常帧检测
        def is_abnormal_frame(frame, last_frame):
            diff = cv2.absdiff(frame, last_frame).sum() // 1e5
            if diff > 20:
                return True
            return False
        
        if last_frame is None:
            last_frame = frame
        if is_abnormal_frame(frame, last_frame):
            continue
        last_frame = frame
        
        #追踪所有的目标，并统计将要离开的目标
        leaving_objs = track_all_objs(frame)
        
        #当目标离开时，对其计数
        count += len(leaving_objs)
                 
        #检测新的目标            
        new_objs = detect_new_objects(frame)
        
        #初始化新的目标
        def init_new_obj(frame, new_obj):
            tracker = create_tracker()
            tracker.init(frame, new_obj)
            trackers.append(tracker)
            objs.append(new_obj)
            move_state = {
                    'entry':new_obj,
                    'exit':None,
                    'distance':0,
                    'time':0
                    }
            move_states.append(move_state)
            
        for new_obj in new_objs:
            init_new_obj(frame, new_obj)

        #根据运动状态，过滤一些目标
        for i, move_state in enumerate(move_states):
            if ind > 10 and ind % 10 == 1:           
                if move_state['distance'] < 5:
                    del trackers[i]
                    del objs[i]
                    del move_states[i]
                else:
                    move_states[i]['distance'] = 0
        
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)            
            
        #显示将要离开的对象
        for leaving_obj in leaving_objs:
            p1 = (int(leaving_obj[0]), int(leaving_obj[1]))
            p2 = (int(leaving_obj[0] + leaving_obj[2]), int(leaving_obj[1] + leaving_obj[3]))
            cv2.rectangle(frame, p1, p2, (0,0,255), 2, 1) 
            
        #显示所有已检测的对象
        for bbox in objs:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)        
        
        cv2.putText(frame, "counter : " + str(count), (200,250), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (200,300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        videoWriter.write(frame)
        
        
        
        cv2.namedWindow('frame')
        cv2.imshow('frame',frame)
#        cv2.setMouseCallback("frame", show_coord)
#        cv2.waitKey(0)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    print("最终行人的个数为：",count)
    
    

    videoWriter.release()
    cap.release()
    cv2.destroyAllWindows()
