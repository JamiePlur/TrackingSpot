import numpy as np
import cv2
import imutils
from cv2 import VideoWriter, VideoWriter_fourcc
import copy


class FrameHelper:

    def __init__(self, dir, w=500, h=500, fps=25):
        data = np.load(dir)
        self.rmax = max(data[:, 0])
        self.frame_ind = 0
        self.frame_num = data.shape[0] // 250
        self.data = data[:, 0]
        self.w = w
        self.h = h

        fourcc = VideoWriter_fourcc(*"MJPG")
        self.Writer = cv2.VideoWriter(dir + "_count.avi", fourcc,
                                      fps, (500, 500))

    def read(self):
        if self.frame_ind >= self.frame_num:
            return False, self.frame

        frame = Frame(self.w, self.h)
        for j in range(250):
            offset = self.frame_ind * 250 + j
            r = self.data[offset]
            frame.append_point(r, j)

        self.last_frame = frame if self.frame_ind is 0 else self.frame
        self.frame = frame
        self.frame_ind += 1
        return True, self.frame

    def init_bg(self, data):
        self.bg = BackGround(data)

    def draw_point(self, frame, point, p=2):
        r, j, c = point
        x, y = self._convert_coord(r, j, self.rmax)
        yt, yd, xl, xr = self._gen_rect_by_point(x, y, p)

        if c is 'white':
            frame.data[yt:yd, xl:xr] = 255
        else:
            frame.data[yt:yd, xl:xr, 1] = 0
            frame.data[yt:yd, xl:xr, 2] = 224
            frame.data[yt:yd, xl:xr, 0] = 0

    def _convert_coord(self, r, j, rmax=1):
        theta = j * np.pi / 2 / 250
        x = int(self.w * r * np.cos(theta) // rmax)
        y = int(self.h * r * np.sin(theta) // rmax)
        return x, y

    def _gen_rect_by_point(self, x, y, p=2):
        xl = x - p if x - p >= 0 else 0
        xr = x + p if x + p < self.w else self.w - 1
        yd = y + p if y + p < self.h else self.h - 1
        yt = y - p if y - p >= 0 else 0
        return yt, yd, xl, xr

    def draw_bbox(self, frame, bbox):
        p1, p2, c = bbox
        cv2.rectangle(frame.data, p1, p2, c, 2, 1)

    def write(self):
        self.Writer.write(self.frame.data)

    def update_bg(self, use_all_points=False):
        if use_all_points:
            for r, i in self.frame.points:
                self.bg.enqueue(r, i)

    def close(self):
        cv2.destroyAllWindows()


class Frame:
    """
        data and corresponding methods in a single frame
    """

    def __init__(self, w=500, h=500, data=None):
        if data is None:
            self.w = w
            self.h = h
            self.data = np.zeros((self.w, self.h, 3), np.uint8)
        else:
            self.data = data
            self.h = data.shape[0]
            self.w = data.shape[1]

        self.points = []
        self.dpoints = []
        self.bboxes = []

    def dynamic_point_num(self):
        return len(self.dpoints)

    def reset(self):
        self.data = np.zeros((self.w, self.h, 3), np.uint8)
        self.points = []
        self.dpoints = []
        self.spoints = []

    def append_dpoint(self, r, j, c='red'):
        point = (r, j, c)
        self.dpoints.append(point)

    def append_point(self, r, j, c='white'):
        point = (r, j, c)
        self.points.append(point)

    def append_bbox(self, bbox, color="blue"):
        if color == "red":
            c = (0, 0, 255)  # red
        else:
            c = (0, 255, 0)  # blue

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        self.bboxes.append((p1, p2, c))

    def clip_bbox(self, bbox):
        xl = int(bbox[0]) if int(bbox[0]) > 0 else 0
        yt = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xr = int(bbox[0] +
                 bbox[2]) if int(bbox[0] + bbox[2]) <= self.w else self.w
        yd = int(bbox[1] +
                 bbox[3]) if int(bbox[1] + bbox[3]) <= self.h else self.h
        c = self.data[yt:yd, xl:xr]
        return c


class BackGround:
    def __init__(self, data, N=10, R=1000, min=5):
        data = np.pad(data, 1, "symmetric")
        len = data.shape[0]
        bg = np.zeros([len, N])
        for i in range(1, len - 1):
            for n in range(N):
                x = np.random.randint(-1, 2)
                r = i + x
                bg[i, n] = data[r]
        bg = bg[1: len - 1]
        self.N = N
        self.R = R
        self.min = min
        self.data = bg

    def enqueue(self, r, i):
        # allow repeatness
        # if r not in self.data[i]:
        self.data[i].append(r)
        if len(self.data[i]) > self.len:
            self.data[i].pop(0)
