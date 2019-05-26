import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc


def convert_coord_back(x, y, rmax=1):
    if x is 0:
        ind = 0
    else:
        ind = np.arctan(y/x) * 2 / np.pi * 250
    scale = rmax / 500
    r = np.floor(np.sqrt(pow(x, 2) + pow(y, 2)) * scale)
    return int(r), int(ind)


def convert_coord(r, ind, rmax=1):
    theta = ind * np.pi / 2 / 250
    x = int(500 * r * np.cos(theta) // rmax)
    y = int(500 * r * np.sin(theta) // rmax)
    return x, y


def gen_rect_by_point(x, y, w=500, h=500, p=2):
    xl = x - p if x - p >= 0 else 0
    xr = x + p if x + p < w else w - 1
    yd = y + p if y + p < h else h - 1
    yt = y - p if y - p >= 0 else 0
    return yt, yd, xl, xr


def draw_point(frame, x, y, p=2):
    yt, yd, xl, xr = gen_rect_by_point(x, y, frame.w, frame.h, p)
    frame.data[yt:yd, xl:xr] = 255


def draw_bbox(frame, bbox):
    p1, p2, c = bbox
    cv2.rectangle(frame.data, p1, p2, c, 2, 1)


def close():
    cv2.destroyAllWindows()


class FrameReader:

    def __init__(self, dirs, w=500, h=500, fps=25):
        # concat all the data
        data = np.load(dirs[0])
        for i in range(1, len(dirs)):
            append_data = np.load(dirs[i])
            data = np.vstack([data, append_data])
        self.data = data[:, 0]

        self.rmax = max(self.data)
        self.frame_ind = 0
        self.frame_num = data.shape[0] // 250
        self.w = w
        self.h = h
        # init video writer
        fourcc = VideoWriter_fourcc(*"MJPG")
        self.Writer = VideoWriter(dirs[0][:-4] + "_count.avi", fourcc,
                                  fps, (self.w, self.h))

    def read(self):
        if self.frame_ind >= self.frame_num:
            return False, self.frame

        frame = Frame(self.w, self.h, rmax=self.rmax)
        for j in range(250):
            offset = self.frame_ind * 250 + j
            r = self.data[offset]
            frame.append_point(r, j)

        self.frame = frame
        self.frame_ind += 1

        return True, self.frame

    def write(self):
        self.Writer.write(self.frame.data)


class Frame:

    def __init__(self, w=500, h=500, rmax=None, data=None):
        if data is None:
            self.w = w
            self.h = h
            self.data = np.zeros((self.w, self.h, 3), np.uint8)
        else:
            self.data = data
            self.h = data.shape[0]
            self.w = data.shape[1]

        self.rmax = rmax
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

    def append_dpoint(self, r, ind):
        point = (r, ind)
        self.dpoints.append(point)

    def append_point(self, r, ind):
        point = (r, ind)
        self.points.append(point)

    def append_bbox(self, bbox, c=(0, 255, 0)):
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        self.bboxes.append((p1, p2, c))

    def clip_bbox(self, bbox, p=0):
        xl = int(bbox[0]) if int(bbox[0]) > 0 else 0
        yt = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xr = int(bbox[0] +
                 bbox[2]) if int(bbox[0] + bbox[2]) <= self.w-p else self.w-p
        yd = int(bbox[1] +
                 bbox[3]) if int(bbox[1] + bbox[3]) <= self.h-p else self.h-p
        c = self.data[yt:yd, xl:xr]
        return c
