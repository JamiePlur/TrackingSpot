import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import imutils


def convert_coord(r, j, rmax=1):
    theta = j * np.pi / 2 / 250
    x = int(500 * r * np.cos(theta) // rmax)
    y = int(500 * r * np.sin(theta) // rmax)
    return x, y


def gen_rect_by_point(x, y, w=500, h=500, p=2):
    xl = x - p if x - p >= 0 else 0
    xr = x + p if x + p < w else w - 1
    yd = y + p if y + p < h else h - 1
    yt = y - p if y - p >= 0 else 0
    return yt, yd, xl, xr


def draw_point(frame, x, y, c='white', p=2):
    yt, yd, xl, xr = gen_rect_by_point(x, y, frame.w, frame.h, p)

    if c is 'white':
        frame.data[yt:yd, xl:xr] = 255
    else:
        frame.data[yt:yd, xl:xr, 1] = 0
        frame.data[yt:yd, xl:xr, 2] = 224
        frame.data[yt:yd, xl:xr, 0] = 0


class FrameHelper:

    def __init__(self, dirs, w=500, h=500, fps=25):
        data = np.load(dirs[0])
        for i in range(1, len(dirs)):
            append_data = np.load(dirs[i])
            data = np.vstack([data, append_data])
        self.rmax = max(data[:, 0])
        self.frame_ind = 0
        self.frame_num = data.shape[0] // 250
        self.last_normal_frame = None
        self.frame = None
        self.data = data[:, 0]
        self.w = w
        self.h = h

        fourcc = VideoWriter_fourcc(*"MJPG")
        self.Writer = VideoWriter(dirs[0] + "_count.avi", fourcc,
                                  fps, (500, 500))

    def read(self):
        if self.frame_ind >= self.frame_num:
            return False, self.frame

        frame = Frame(self.w, self.h)
        for j in range(250):
            offset = self.frame_ind * 250 + j
            r = self.data[offset]
            frame.append_point(r, j)

#        self.last_frame = frame if self.frame_ind is 0 else self.frame
        self.frame = frame
        self.frame_ind += 1

        for point in self.frame.points:
            self.draw_point(self.frame, point)
        return True, self.frame

    def init_bg(self, data):
        self.bg = PolyBackGround(data)

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

    def clip_bbox(self, bbox, p=0):
        xl = int(bbox[0]) if int(bbox[0]) > 0 else 0
        yt = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xr = int(bbox[0] +
                 bbox[2]) if int(bbox[0] + bbox[2]) <= self.w-p else self.w-p
        yd = int(bbox[1] +
                 bbox[3]) if int(bbox[1] + bbox[3]) <= self.h-p else self.h-p
        c = self.data[yt:yd, xl:xr]
        return c


class PolyBackGround:
    def __init__(self, data):
        self.data = np.reshape(data, [1, -1, 2])


class VibeBackGround:
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

    def update(self, points, dp_ind):
        for i in range(len(points)):
            # if not dpoint
            if i not in dp_ind:
                # 1/N odds to update this point
                p = np.random.randint(0, self.N)
                if p == 0:
                    r = np.random.randint(0, self.N)
                    self.data[i, r] = points[i][0]
                p = np.random.randint(0, self.N)
                # 1/N odds to update near point
                if p == 0:
                    x = np.random.randint(-1, 2)
                    r = np.random.randint(0, self.N)
                    try:
                        self.data[i + x, r] = points[i][0]
                    except:
                        pass
        return self.data

    def revised_by_tracking(self, obj, points):
        x, y, w, h = obj.bbox
        j_start = int(np.arctan(y/(x+w))*250*2/np.pi)
        j_end = int(np.arctan((y+h)/x)*250*2/np.pi)
        print("j range:{}-{}".format(j_start, j_end))
        for j in range(j_start, j_end+1):
            self.data[j, 5:] = points[j][0]
