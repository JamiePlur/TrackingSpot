import copy

import cv2
import numpy as np

from Counter import Counter
from FrameHelper import Frame, FrameHelper
import FrameHelper as framehelper


class State:

    def __init__(self, count_machine):
        self.count_machine = count_machine
        self.frame_helper = count_machine.fh


class InitialState(State):

    def __init__(self, count_machine, l=5):
        super().__init__(count_machine)
        self.reset(l)
        self.bg_init_data = [[] for _ in range(250)]

    def reset(self, l=10):
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

    def test_bg(self, data, convex, dp_ind=None):
        frame = Frame(500, 500)
        for d in data:
            x, y = d
            framehelper.draw_point(frame, x, y)
        convex = np.reshape(convex, [1, -1, 2])
        cv2.drawContours(frame.data, convex, -1, (0, 0, 255), 10)
        # cv2.imshow('test bg', frame.data)

    def init_bg(self, data):
        sps = []
        # detect sp by std
        std = np.std(data, axis=1)
        sp_inds = np.where(std < 50)[0]
        # collect sp
        rmax = max(data[:, 0])
        data = np.median(data, axis=1)
        for sp_ind in sp_inds:
            x, y = framehelper.convert_coord(data[sp_ind], sp_ind, rmax)
            sps.append((max(x - 30, 0), max(y - 30, 0)))
        # gen convex
        c = np.array([sps])
        epsilon = 0.01 * cv2.arcLength(c, True)
        convex = cv2.convexHull(c, epsilon, True)
        self.test_bg(sps, convex)
        return convex


class NormalState(State):

    def __init__(self, count_machine):
        super().__init__(count_machine)
        self.abnormal_frame_in_row = 0
        self.counter = Counter(self.frame_helper)

    def update(self):
        s = "normal"
        return s

    def handle(self):
        self._updata_bg()
        # detect new objs
        self.detect()
        # track existing objs, if lose, count
        cnt = self.track()
        return cnt

    def detect(self):
        frame = self.frame_helper.frame
        roi_frame = self.counter.detect(frame)
        return roi_frame

    def track(self):
        frame = self.frame_helper.frame
        cnt = self.counter.track(frame)
        return cnt

    def _updata_bg(self):
        pass


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
            print("count num:", self.cnt)
            if display:
                if s is not 'init':
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
        # cv2.drawContours(frame.data, self.fh.bg.data, -1, (0, 0, 255), 10)
        cv2.imshow(win, frame.data)
        cv2.waitKey(0)
        # print("th dp num of {}th frame is {}".format(
        #     self.fh.frame_ind, self.fh.frame.dynamic_point_num()))

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
