import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
from cv2 import VideoWriter,VideoWriter_fourcc,imread,resize
import os

class Frame:
    
    def __init__(self, w, h, fps):
        self.w = w
        self.h = h
        self.fps = fps
        self.frame = np.zeros((self.w, self.h, 3),np.uint8)
        self.points = []
        self.dpoints = []
     
    def init(self):
        self.frame = np.zeros((self.w, self.h, 3),np.uint8)
        self.points = []
        self.dpoints = []
        
    def draw_point(self, r, theta, rmax, p = 1, color = 'white'):
        # convert to x-y coordiantes
        x = int(self.w * r * np.cos(theta) // rmax)
        y = int(self.h * r * np.sin(theta) // rmax)
        # draw_point
        xl = x - p if x - p >= 0 else 0
        xr = x + p if x + p < self.w else self.w - 1
        yd = y + p if y + p < self.h else self.h - 1
        yt = y - p if y - p >= 0 else 0
        if color == 'white':
            self.frame[yt:yd,xl:xr] = 255
        elif color == 'red':
            self.frame[yt:yd,xl:xr, 2] = 224
        else:
            print("not supported!")
            pass
        # save point
        self.points.append(r)
        
    def save_dpoints(self, ind, r):
        p = (ind, r)
        self.dpoints.append(p)
        
        
            
    def imshow(self):
        self.frame = imutils.resize(self.frame, width = 500)
        cv2.imshow('img',self.frame)
        cv2.waitKey(0)  
    
    def close():
        cv2.destroyAllWindows()
    
        
    
    
class CountingStateMachine:
    
    def __init__(self, dir):


        data = np.load(dir)
        self.rmax = max(data[:, 0])
        self.frame_num = data.shape[0] // 250
        self.data = data[:, 0]
        
        self.Frame = Frame(224, 224, 25)
        
        self.state = 0                              #1是动态，0是静态
        self.state_handler = [self.handle_static_state,
                              self.handle_dynamic_state]
        self.static_frame_in_row = 0
        
        self.frame_handler = [self.handle_static_frame,
                              self.handle_dynamic_frame]
        self.bg = []
        


    def run(self):
        
        for i in range(self.frame_num):
            #before load data
            dynamic_point_num = 0
            self.Frame.init()
            #loading data for a frame
            for j in range(250):
                offset = i * 250 + j               
                theta = j * np.pi / 2 / 250
                r = self.data[offset]                      
#                r_last = self.data[offset - 250] if offset - 250 >= 0 else r       
#                r_diff = abs(r - r_last)
                bg = np.array(self.bg[j]) if len(self.bg)> 0 else r
                r_diff = abs(bg - r).min()
                if r_diff > 200:
                    dynamic_point_num += 1
                    self.Frame.draw_point(r, theta, self.rmax, p = 10 , color = 'red')
#                    self.Frame.save_dpoints(j, r)                    
                self.Frame.draw_point(r, theta, self.rmax)                
            #after load data
            self.state_handler[self.state](dynamic_point_num)
            self.frame_handler[self.state]()
            self.Frame.imshow()        
            print("第{}帧的diff总和为{}".format(i,dynamic_point_num))
        self.Frame.close()

    def handle_dynamic_state(self, dynamic_point_num):
        
        if dynamic_point_num < 4:
            self.static_frame_in_row += 1
        else:
            self.static_frame_in_row = 0
            
        if self.static_frame_in_row >= 5: #连续5帧的动点个数小于一定数量则转化为静态
            self.state = 0
            print("convert to static state")
        print("dynamic!")

      
    def handle_static_state(self, dynamic_point_num):
        
        if dynamic_point_num > 10:        #当动点个数大于一定数量则转化为动态
            self.state = 1 
            self.static_frame_in_row = 0
            print("convert to dynamic state")
        print("static")
    
    def handle_static_frame(self):

        points = self.Frame.points
        
        if len(self.bg) == 0:
            # bg has shape of (250,x)
            self.bg = [[p] for p in points]
            return
        
        for i in range(250):
            p = points[i]
            if p not in self.bg[i]:
                self.bg[i].append(p)
        
        #to do: 队列？
#        print(self.bg)
        
    
    def handle_dynamic_frame(self):
#        print(self.Frame.dpoints)
        
        
        
        pass
          
      

              
if __name__ == '__main__':
    ind = 17
    dirs = os.listdir("data")
    dir = os.path.join("data", dirs[ind])
    sm = CountingStateMachine(dir)
    sm.run()

