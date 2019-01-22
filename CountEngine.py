from FrameHelper import FrameHelper    
from Counter import Counter

class State:
    
    static_frame_in_row = 0
    

class StaticState(State):
    
    def get_state(self):
        return "static"
    
    def update(self, sm):
        if sm.dynamic_point_num > 10:        #当动点个数大于一定数量则转化为动态
            sm.change_state(DynamicState())
            State.static_frame_in_row = 0
            print("convert to dynamic state")
            return self.get_state()
        print("now is static state")
        return self.get_state()
            

class DynamicState(State):
      
    def get_state(self):
        return "dynamic"
    
    def update(self, sm):
        if sm.dynamic_point_num < 4:
            State.static_frame_in_row += 1
        else:
            State.static_frame_in_row = 0
            
        if State.static_frame_in_row >= 5: #连续5帧的动点个数小于一定数量则转化为静态
            sm.change_state(StaticState())
            print("convert to static state")
            return self.get_state()
        print("now is dynamic state")
        return self.get_state()
    
    
class CountingStateMachine:
    
    state = StaticState()    
    dynamic_point_num = 0
    
    def change_state(self, state):
        self.state = state
        
    def update(self, frame):
        self.dynamic_point_num = frame.dynamic_point_num 
        s = self.state.update(self)
        return s
    
    def get_state(self):
        return self.state.get_state()
    
    
    
class CountingEngine:
    

    
    def __init__(self, dir):

        self.frame_helper = FrameHelper(dir)
        self.counter = Counter()
        self.cnt = 0


    def run(self):
        
        sm = CountingStateMachine()
        fh = self.frame_helper

        
        while(1):
            ok, frame = fh.read()
            if not ok:
                break

            state = sm.update(frame)

            count_handler = getattr(self, "count_" + state + "_frame")
            self.cnt += count_handler(frame)
            
            frame.show()        
            print("第{}帧的diff总和为{}".format(fh.frame_ind, frame.dynamic_point_num))
            
        fh.close()

    
    def count_static_frame(self, frame):

        fh = self.frame_helper
        fh.update_bg(use_all_points = True)
              
#        self.counter.track(frame)
#        print(fh.bg.data)0
        
        return 0
        
    
    def count_dynamic_frame(self, frame):
        
        self.counter.detect(frame)
        
#        cnt = self.counter.track(frame)        
        
        return 0
          
      

              
if __name__ == '__main__':
    import os
    ind = 17
    dirs = os.listdir("data")
    dir = os.path.join("data", dirs[ind])
    sm = CountingEngine(dir)
    sm.run()

