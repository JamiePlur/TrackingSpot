import cv2

from Counter import Counter
from FrameHelper import FrameHelper, Frame
import numpy as np


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
        dp_num = self.frame_helper.frame.dynamic_point_num()
        last_dp_num = self.frame_helper.last_frame.dynamic_point_num()
        if dp_num - last_dp_num > 50:
            s = "abnormal"
            return s
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
            data = self.init_bg3(np.array(self.bg_init_data))
            fh.init_bg(data)
            # test bg
            frame = Frame(500, 500)
            for i, r in enumerate(data):
                frame.append_point(r, i)
            for point in frame.points:
                fh._draw_point(frame, point)
            cv2.imshow('test bg', frame.data)
        return 0

    def init_bg(self, data):
        median = np.median(data, axis=1)
        grad = np.zeros([250])
        for i in range(250):
            grad_l = abs(median[i-1]-median[i]) if i > 0 else 0
            grad_r = abs(median[i+1]-median[i]) if i < 249 else 0
            grad[i] = grad_l + grad_r

        data = median
        dp_ind = np.where(grad > 100)[0].tolist()
        if len(dp_ind) is 250:
            dp_ind.pop()
        while(len(dp_ind) > 0):
            for ind in dp_ind:
                if ind + 1 < 250 and ind + 1 not in dp_ind:
                    data[ind] = data[ind + 1]
                    dp_ind.remove(ind)
                    break
                if ind - 1 >= 0 and ind - 1 not in dp_ind:
                    data[ind] = data[ind - 1]
                    dp_ind.remove(ind)
                    break
        return data

    def init_bg3(self, data):
        data_bg = data
        std = np.std(data, axis=1)
        dp_ind = np.where(std > 50)[0].tolist()
        if len(dp_ind) is 250:
            dp_ind.pop()
        while(len(dp_ind) > 0):
            for ind in dp_ind:
                if ind + 1 < 250 and ind + 1 not in dp_ind:
                    data_bg[ind] = data_bg[ind + 1]
                    dp_ind.remove(ind)
                    break
                if ind - 1 >= 0 and ind - 1 not in dp_ind:
                    data_bg[ind] = data_bg[ind - 1]
                    dp_ind.remove(ind)
                    break
        return data_bg.mean(axis=1, dtype=int)

    def init_bg2(self, data):
        data_bg = None
        for r in range(data.shape[0]):
            # remove max value and min value
            d = np.sort(data[r, :])
            d = np.delete(d, [0, data.shape[1] - 1])
            data_bg = np.vstack([data_bg, d]) if r > 0 else d
        # detect dp
        std = np.std(data_bg, axis=1)
        dp_ind = np.where(std > 200)[0].tolist()
        # find the nearest sp for each dp
        if len(dp_ind) is 250:
            dp_ind.pop()
        while(len(dp_ind) > 0):
            for ind in dp_ind:
                if ind + 1 < 250 and ind + 1 not in dp_ind:
                    data_bg[ind] = data_bg[ind + 1]
                    dp_ind.remove(ind)
                    break
                if ind - 1 >= 0 and ind - 1 not in dp_ind:
                    data_bg[ind] = data_bg[ind - 1]
                    dp_ind.remove(ind)
                    break
        return data_bg.mean(axis=1, dtype=int)


class NormalState(State):

    def __init__(self, count_machine):
        super().__init__(count_machine)
        self.abnormal_frame_in_row = 0

    def update(self):
        s = "normal"
        dp_num = self.frame_helper.frame.dynamic_point_num()
        last_dp_num = self.frame_helper.last_frame.dynamic_point_num()
        if dp_num - last_dp_num > 50:
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
        return 0


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
        self.fh.display(cnt, win)
        print("第{}帧的动点数为{}".format(self.fh.frame_ind,
                                         self.fh.frame.dynamic_point_num()))
        print("计数：", cnt)

    def change_state(self, state):
        self.state = state


if __name__ == '__main__':
    import os
    ind = 22
    dirs = os.listdir("data")
    dir = os.path.join("data", dirs[ind])
    ce = CountingMachine(dir)
    ce.run()
