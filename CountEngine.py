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
        cv2.imshow('test bg', frame.data)

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

    def update(self):
        s = "normal"
        # append dpoints
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
        # check abnormal frame
        dp_num = self.frame_helper.frame.dynamic_point_num()
        last_dp_num = self.frame_helper.last_frame.dynamic_point_num()
        print("dp diff:", dp_num - last_dp_num)
        if dp_num - last_dp_num > 35:
            self.abnormal_frame_in_row += 1
            s = "abnormal"
        else:
            self.abnormal_frame_in_row = 0
        if self.abnormal_frame_in_row >= 5:
            # change to initial state
            cm = self.count_machine
            cm.change_state(cm.initial_state)
            print("convert to initial state")
        return s

    def handle(self):
        self.updata_bg()
        return 0

    def updata_bg(self):
        fh = self.frame_helper
        bg_data = fh.bg.data
        N = fh.bg.N
        points = fh.frame.points
        dpoints = fh.frame.dpoints
        dp_ind = [dpoints[x][1] for x in range(len(dpoints))]

        for i in range(len(points)):
            # if not dpoint
            if i not in dp_ind:
                # 1/2 odds to update this point
                p = np.random.randint(0, 2)
                if p == 0:
                    r = np.random.randint(0, N)
                    bg_data[i, r] = points[i][0]
                p = np.random.randint(0, 5)
                # 1/3 odds t0 update near point
                if p == 0:
                    x = np.random.randint(-1, 2)
                    r = np.random.randint(0, N)
                    try:
                        bg_data[i + x, r] = points[i][0]
                    except:
                        pass
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
        cv2.imshow('test bg', frame.data)


class CountingMachine:

    def __init__(self, dir):
        self.fh = FrameHelper(dir)
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
        frame = copy.deepcopy(self.fh.frame)
        for point in frame.points:
            self.fh.draw_point(frame, point)
        for dpoint in frame.dpoints:
            self.fh.draw_point(frame, dpoint)
        for bbox in frame.bboxes:
            self.fh.draw_bbox(bbox)
        if cnt is not None:
            cv2.putText(frame.data, "counter : " + str(cnt), (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow(win, frame.data)
        cv2.waitKey(0)
        print("th dp num of {}th frame is {}".format(
            self.fh.frame_ind, self.fh.frame.dynamic_point_num()))

    def change_state(self, state):
        self.state = state


if __name__ == '__main__':
    import os
    ind = 22
    dirs = os.listdir("data")
    dir = os.path.join("data", dirs[ind])
    ce = CountingMachine(dir)
    ce.run()
    