import copy

import cv2
import numpy as np

from Counter import Counter
from FrameHelper import Frame, FrameHelper


class State:

    def __init__(self, count_machine):
        self.count_machine = count_machine
        self.frame_helper = count_machine.fh


class InitialState(State):

    def __init__(self, count_machine, l=5):
        super().__init__(count_machine)
        self.reset(l)
        self.bg_init_data = [[] for _ in range(250)]

    def reset(self, l=5):
        self.bg_init_data = [[] for _ in range(250)]
        self.l = l

    def update(self):
        s = "init"
        self.l -= 1
        if self.l < 0:
            # change to normal state
            cm = self.count_machine
            cm.change_state(cm.normal_state)
            print("convert to normal state")
        return s

    def handle(self):
        fh = self.frame_helper
        points = fh.frame.points

        for r, j, _ in points:
            self.bg_init_data[j].append(r)
        if self.l is 0:
            data = self.init_bg(np.array(self.bg_init_data))
            fh.init_bg(data)

        return 0

    def test_bg(self, data, dp_ind=None):
        fh = self.frame_helper
        frame = Frame(500, 500)
        for i, r in enumerate(data):
            if dp_ind is not None and i in dp_ind:
                frame.append_point(r, i, 'red')
            else:
                frame.append_point(r, i)
        for point in frame.points:
            fh.draw_point(frame, point)
        # cv2.imshow('test bg', frame.data)

    def init_bg(self, data):
        std = np.std(data, axis=1)
        # detect dp
        dp_ind = np.where(std > 50)[0].tolist()
        data = np.median(data, axis=1)

        if len(dp_ind) is 250:
            dp_ind.pop()
        while(len(dp_ind) > 0):
            for ind in dp_ind:
                sign = 1 if np.random.randint(0, 2) > 0 else -1
                if ind + 1*sign < 250 and ind + 1*sign not in dp_ind:
                    # find the nearest sp for each dp
                    # change the value of dp with sp + grad
                    if ind + 2*sign < 250 and ind + 2*sign not in dp_ind:
                        grad = data[ind + 1*sign] - data[ind + 2*sign]
                        data[ind] = data[ind + 1*sign] + grad \
                            if abs(grad) < 100 else data[ind + 1*sign]
                    else:
                        data[ind] = data[ind + 1*sign]
                    dp_ind.remove(ind)
                    break
        self.test_bg(data)
        return data


class NormalState(State):

    def __init__(self, count_machine):
        super().__init__(count_machine)
        self.abnormal_frame_in_row = 0
        self.counter = Counter(self.frame_helper)

    def update(self):
        s = "normal"
        self._append_dpoint()
        # check abnormal frame
        normal = self._check_abnormal_frame()
        if not normal:
            self.abnormal_frame_in_row += 1
            s = "abnormal"
        else:
            self.abnormal_frame_in_row = 0
        if self.abnormal_frame_in_row >= 10:
            # change to initial state
            cm = self.count_machine
            print("convert to initial state")
            cm.change_state(cm.initial_state)
            cm.state.reset()
        return s

    def handle(self):
        self._updata_bg()
        # detect new objs
        self._detect()
        # track existing objs, if lose, count
        cnt = self._track()
        return cnt

    def _detect(self):
        frame = self.frame_helper.frame
        self.counter.detect(frame)

    def _track(self):
        frame = self.frame_helper.frame
        cnt = self.counter.track(frame)
        return cnt

    def _append_dpoint(self):
        points = self.frame_helper.frame.points
        bg_data = self.frame_helper.bg.data
        N = self.frame_helper.bg.N
        R = self.frame_helper.bg.R
        min = self.frame_helper.bg.min
        for i in range(len(points)):
            count, index = 0, 0
            while count < min and index < N:
                d = np.abs(points[i][0] - bg_data[i, index])
                if d < R:
                    count += 1
                index += 1
            if count < min:
                self.frame_helper.frame.append_dpoint(points[i][0], i)

    def _check_abnormal_frame(self):
        if self.frame_helper.last_normal_frame is None:
            self.frame_helper.last_normal_frame = self.frame_helper.frame
        dp_num = self.frame_helper.frame.dynamic_point_num()
        last_dp_num = self.frame_helper.last_normal_frame.dynamic_point_num()

        dp = self.frame_helper.frame.dpoints
        dp = [dp[x][0] for x in range(len(dp))]
        dp_std = np.array(dp).std()

        if dp_num - last_dp_num > 50 or dp_std > 5000:
            return False
        else:
            self.frame_helper.last_normal_frame = self.frame_helper.frame
            return True

    def _updata_bg(self):
        fh = self.frame_helper
        points = fh.frame.points
        dpoints = fh.frame.dpoints
        dp_ind = [dpoints[x][1] for x in range(len(dpoints))]
        bg_data = fh.bg.update(points, dp_ind)
        self.test_bg(np.median(bg_data, axis=1), dp_ind)

    def test_bg(self, data, dp_ind=None):
        fh = self.frame_helper
        frame = Frame(500, 500)
        for i, r in enumerate(data):
            if dp_ind is not None and i in dp_ind:
                frame.append_point(r, i, 'red')
            else:
                frame.append_point(r, i)
        for point in frame.points:
            fh.draw_point(frame, point)
        # cv2.imshow('test bg', frame.data)


class CountingMachine:

    def __init__(self, dirs):
        self.fh = FrameHelper(dirs)
        self.initial_state = InitialState(self)
        self.normal_state = NormalState(self)
        self.state = self.initial_state
        self.cnt = 0

    def run(self, display=True, save=False):
        while(1):
            ok, frame = self.fh.read()
            if not ok:
                break

            s = self.state.update()
            if s is "abnormal":
                print("skip this frame!")
                continue

            self.cnt += self.state.handle()

            if display:
                self.display(self.cnt)

            if save:
                self.fh.write()

        self.fh.close()

    def display(self, cnt=None, win='img'):
        frame = self.fh.frame
        # for dpoint in frame.dpoints:
        #     self.fh.draw_point(frame, dpoint)
        for bbox in frame.bboxes:
            self.fh.draw_bbox(frame, bbox)
        if cnt is not None:
            cv2.putText(frame.data, "counter : " + str(cnt), (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # cv2.imshow(win, frame.data)
        # cv2.waitKey(0)
        print("th dp num of {}th frame is {}".format(
            self.fh.frame_ind, self.fh.frame.dynamic_point_num()))

    def change_state(self, state):
        self.state = state


if __name__ == '__main__':
    import os
#    inds = [22]
#    list_dirs = os.listdir("data")
#    dirs = []
#    for i in inds:
#        dir = os.path.join("data", list_dirs[i])
#        dirs.append(dir)

    dirs = []
    dir = os.path.join("data", 'exit.npy')
    dirs.append(dir)
    ce = CountingMachine(dirs)
    ce.run(save=True)
