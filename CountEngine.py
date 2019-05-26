import cv2
import numpy as np

from Counter import Counter
import FrameHelper as fh
from BackGround import PolyBackGround


class State:
    count_machine = None
    bg = PolyBackGround()


class InitialState(State):
    def __init__(self, count_machine=None, l=5):
        State.count_machine = count_machine
        self.reset(l)
        self.last_gray = None

    def reset(self, l=10):
        self.bg_init_data = [[] for _ in range(250)]
        self.l = l

    def update(self, frame):
        self.l -= 1
        if self.l < 0:
            # change to normal state
            cm = self.count_machine
            cm.change_state(cm.normal_state)
            print("convert to normal state")

    def handle(self, frame):
        # optical flow
        for point in frame.points:
            r, ind = point
            x, y = fh.convert_coord(r, ind, frame.rmax)
            fh.draw_point(frame, x, y)
            self.bg_init_data[ind].append(r)
        gray = cv2.cvtColor(frame.data, cv2.COLOR_BGR2GRAY)

        if self.last_gray is None:
            corners = cv2.goodFeaturesToTrack(
                gray, 25, 0.01, 10, mask=None, blockSize=7, gradientSize=3)
            self.last_gray = gray
            self.p0 = corners
        else:
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray, self.last_gray,
                                                   self.p0, None)
        if self.l is 0:

            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            sdx = 0
            sdy = 0
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                x1, y1 = new.ravel()
                x0, y0 = old.ravel()
                dy = y1 - y0
                dx = x1 - x0
                sdx -= dx
                sdy -= dy

            if sdy > 0:
                print("down!")
                print("sdx {}, sdy{}".format(sdx, sdy))
            else:
                print("up!")
                print("sdx {}, sdy{}".format(sdx, sdy))

                # mask = np.zeros_like(gray)
                # color = np.random.randint(0, 255, (100, 3))
                # cv2.line(mask, (x1, y1), (x0, y0), color[i].tolist(), 2)
                # cv2.imshow('lk', lines)
            self.init_bg(np.array(self.bg_init_data), frame.rmax)
        return 0

    def test_bg(self, data, convex, rmax, dp_ind=None):
        frame = fh.Frame(500, 500)
        for d in data:
            x, y = d
            fh.draw_point(frame, x, y)
        convex = np.reshape(convex, [1, -1, 2])
        for ind, r in enumerate(State.bg.points):
            x, y = fh.convert_coord(r, ind, rmax)
            cv2.circle(frame.data, (x, y), 4, [0, 255, 0], -1)
        # cv2.drawContours(frame.data, convex, -1, (0, 0, 255), 10)
        cv2.imshow('test bg', frame.data)

    def init_bg(self, data, rmax):
        sps = []
        # detect sp by std
        std = np.std(data, axis=1)
        sp_inds = np.where(std < 50)[0]
        # collect sp
        data = np.median(data, axis=1)
        for sp_ind in sp_inds:
            x, y = fh.convert_coord(data[sp_ind], sp_ind, rmax)
            sps.append((max(x - 30, 0), max(y - 30, 0)))
        # get roi
        c = np.array([sps])
        epsilon = 10 * cv2.arcLength(c, True)
        roi = cv2.convexHull(c, epsilon, True)
        State.bg.set_roi(roi, rmax)

        self.test_bg(sps, roi, rmax)


class NormalState(State):
    def __init__(self):
        self.abnormal_frame_in_row = 0
        self.last_normal_frame = None
        self.counter = Counter(State.bg)

    def update(self, frame):
        # append dp
        def update_dpoints(frame):
            frame_points = frame.points
            bg_points = State.bg.points
            for ind, frame_point in enumerate(frame_points):
                br = bg_points[ind]
                fr = frame_point[0]
                if abs(fr - br) > 1000:
                    frame.append_dpoint(fr, ind)

        update_dpoints(frame)
        # check if this frame normal
        normal = self.check_normal(frame)
        if not normal:
            self.abnormal_frame_in_row += 1
        else:
            self.abnormal_frame_in_row = 0
        # chenge state when abnormal
        if self.abnormal_frame_in_row >= 10:
            # change to initial state
            cm = State.count_machine
            print("convert to initial state")
            cm.change_state(cm.initial_state)
            cm.state.reset()

    def handle(self, frame):
        self.update_bg()
        # detect new objs
        self.detect(frame)
        # track existing objs, if lose, count
        count = self.track(frame)
        return count

    def check_normal(self, frame):
        if self.last_normal_frame is None:
            self.last_normal_frame = frame
            return True

        dp_num = frame.dynamic_point_num()
        last_dp_num = self.last_normal_frame.dynamic_point_num()

        dp = frame.dpoints
        dp = [dp[x][0] for x in range(len(dp))]
        dp_std = np.array(dp).std()

        if dp_num - last_dp_num > 50 or dp_std > 5000:
            return False
        else:
            self.last_normal_frame = frame
            return True

    def detect(self, frame):
        self.counter.detect(frame)

    def track(self, frame):
        count = self.counter.track(frame)
        return count

    def update_bg(self):
        pass


class CountingMachine:
    def __init__(self, dirs):
        self.frame_reader = fh.FrameReader(dirs)
        self.initial_state = InitialState(self)
        self.normal_state = NormalState()
        self.state = self.initial_state
        self.count = 0

    def run(self, display=True, save=False):
        while (1):
            ok, frame = self.frame_reader.read()
            if not ok:
                break
            self.state.update(frame)
            self.count += self.state.handle(frame)
            print("count num:", self.count)

            if display:
                self.display(frame, self.count)

            if save:
                self.frame_reader.write()

        self.fh.close()

    def display(self, frame, cnt=None, win='img'):
        # for dpoint in frame.dpoints:
        #     self.fh.draw_point(frame, dpoint)
        for point in frame.points:
            r, ind = point
            x, y = fh.convert_coord(r, ind, frame.rmax)
            fh.draw_point(frame, x, y)
        for bbox in frame.bboxes:
            fh.draw_bbox(frame, bbox)
        if cnt is not None:
            cv2.putText(frame.data, "counter : " + str(cnt), (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # cv2.drawContours(frame.data, self.fh.bg.data, -1, (0, 0, 255), 10)
        cv2.imshow(win, frame.data)
        cv2.waitKey(0)
        print("th dp num of {}th frame is {}".format(
            self.frame_reader.frame_ind, frame.dynamic_point_num()))

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
    ce.run(save=False)