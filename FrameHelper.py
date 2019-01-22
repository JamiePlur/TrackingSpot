import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize



class FrameHelper:
    
    frame_ind = 0
    
    def __init__(self, dir, w = 224, h = 224, fps = 25):
        data = np.load(dir)
        self.rmax = max(data[:, 0])
        self.frame_num = data.shape[0] // 250
        self.data = data[:, 0]
        self.frame = Frame(w, h, fps, self.rmax)
        self.bg = BackGround()
        

    def read(self):
        
        if self.frame_ind >= self.frame_num:
            return False, self.frame

        self.frame.reset()

        for j in range(250):
            
            offset = self.frame_ind * 250 + j               

            r = self.data[offset]
            #静点          
            r_last = self.data[offset - 250] if offset - 250 >= 0 else r       
            dr_last = abs(r - r_last)
            if dr_last < 100:
                self.frame.spoints.append((r, j))
            #动点
            r_bg = np.array(self.bg.data[j]) if len(self.bg.data[j])> 0 else r
            dr_bg = abs(r_bg - r).min()
            if dr_bg > 200:
                self.frame.dynamic_point_num += 1
#                self.frame.draw_point(r, theta, self.rmax, p = 10 , color = 'red')
                self.frame.dpoints.append((r, j)) 
                  
            self.frame.draw_point(r, j, self.rmax)
            self.frame.points.append((r, j))
            
        self.frame_ind += 1
        return True, self.frame
    
    def update_bg(self, use_all_points = False):
        
        if use_all_points:
            for r, i in self.frame.points:
                self.bg.enqueue(r, i)
                        
        for r, i in self.frame.spoints:
            self.bg.enqueue(r, i)
        
    
    def close(self):
        cv2.destroyAllWindows()
        


class Frame:
    """
        Frame 存储且仅存储一帧数据并提供处理这些数据的方法
            frame : 图像，三维numpy数组
            points: 所有的描点
            dpoints：动点，和背景差大于200的点
            spoints: 静点，和前一帧差小于100的点
    """
    
   
    def __init__(self, w, h, fps, rmax):
        self.w = w
        self.h = h
        self.fps = fps
        self.rmax = rmax # to do : 以后要去掉这个
        self.frame = np.zeros((self.w, self.h, 3),np.uint8)
        self.points = []
        self.dpoints = []
        self.spoints = []
        self.dynamic_point_num = 0
     
    def reset(self):
        self.dynamic_point_num = 0
        self.frame = np.zeros((self.w, self.h, 3),np.uint8)
        self.points = []
        self.dpoints = []
        self.spoints = []
        
    def draw_point(self, r, j, rmax, p = 1, color = 'white'):
        
        
        # convert to x-y coordiantes
        x, y = self.__convert_coord__(r, j, rmax)
        yt, yd, xl, xr = self.__gen_rect_by_point__(x, y, p)

        if color == 'white':
            self.frame[yt:yd,xl:xr] = 255
        elif color == 'red':
            self.frame[yt:yd,xl:xr, 2] = 224
        else:
            print("not supported!")
            pass
    
    
    def show_bbox(self, bbox):
        
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(self.frame, p1, p2, (0,0,255), 2, 1)
    
    def __convert_coord__(self, r, j, rmax = 1):
        
        theta = j * np.pi / 2 / 250
        x = int(self.w * r * np.cos(theta) // rmax)
        y = int(self.h * r * np.sin(theta) // rmax)
        
        return x, y
    
    def __gen_rect_by_point__(self, x, y, p = 1):
        
        xl = x - p if x - p >= 0 else 0
        xr = x + p if x + p < self.w else self.w - 1
        yd = y + p if y + p < self.h else self.h - 1
        yt = y - p if y - p >= 0 else 0
        
        return yt, yd, xl, xr
    
    def show(self):
        self.frame = imutils.resize(self.frame, width = 500)
        cv2.imshow('img', self.frame)
        cv2.waitKey(0)  

    
class BackGround:
    
    
    def __init__(self, len = 10):
        self.data = [[] for _ in range(250)] 
        self.len = len
        
    def reset(self):
        
        self.data = [[] for _ in range(250)] 
    
    def enqueue(self, r, i):
        
        if r not in self.data[i]: 
            self.data[i].append(r)
        if len(self.data[i]) > self.len:
            self.data[i].pop(0)
        
        
        

        
    
    
