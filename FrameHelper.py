import numpy as np
import cv2
import imutils
from cv2 import VideoWriter, VideoWriter_fourcc
import copy


class FrameHelper:

    frame_ind = 0

    def __init__(self, dir, w=224, h=224, fps=25):
        data = np.load(dir)
        self.rmax = max(data[:, 0])
        self.frame_num = data.shape[0] // 250
        self.data = data[:, 0]
        self.frame = Frame(rmax=self.rmax)
        self.bg = BackGround()

        fourcc = VideoWriter_fourcc(*"MJPG")
        self.Writer = cv2.VideoWriter(dir + "_count.avi", fourcc, fps,
                                      (500, 500))

    def read(self):

        if self.frame_ind >= self.frame_num:
            return False, self.frame

        frame = Frame(rmax=self.rmax)
        # self.frame.reset()

        for j in range(250):

            offset = self.frame_ind * 250 + j

            r = self.data[offset]
            # static point (sp)
            r_last = self.data[offset - 250] if offset - 250 >= 0 else r
            dr_last = abs(r - r_last)
            if dr_last < 2:
                frame.spoints.append((r, j))
            # dynamic point (dp)
            r_bg = np.array(self.bg.data[j]) if len(self.bg.data[j]) > 0 else r
            dr_bg = abs(r_bg - r).min()
            if dr_bg > 200:
                frame.dynamic_point_num += 1
                #                self.frame.draw_point(r, theta, self.rmax, p = 10 , color = 'red')
                frame.dpoints.append((r, j))

            frame.draw_point(r, j, self.rmax)
            frame.points.append((r, j))

        self.frame_ind += 1

        # detect obnormal frame
        if frame.dynamic_point_num - self.frame.dynamic_point_num > 50:
            print("检测到异常帧!")
            return 1, self.frame

        self.frame = frame
        return True, self.frame

    def write(self):

        self.Writer.write(self.frame.data)

    def update_bg(self, use_all_points=False):

        if use_all_points:
            for r, i in self.frame.points:
                self.bg.enqueue(r, i)

        # for r, i in self.frame.spoints:
        #     self.bg.enqueue(r, i)

    def close(self):
        cv2.destroyAllWindows()


class Frame:
    """
        一帧内部的数据和方法
            data : 图像，三维numpy数组
            points: 所有的描点
            dpoints：动点，和背景差大于200的点
            spoints: 静点，和前一帧差小于100的点
    """

    def __init__(self, rmax, w=224, h=224, fps=25, data=None):
        if data is None:
            self.w = w
            self.h = h
            self.data = np.zeros((self.w, self.h, 3), np.uint8)
        else:
            self.data = data
            self.h = data.shape[0]
            self.w = data.shape[1]

        self.fps = fps
        self.rmax = rmax
        self.points = []
        self.dpoints = []
        self.spoints = []
        self.bboxes = []
        self.dynamic_point_num = 0

    def reset(self):
        self.dynamic_point_num = 0
        self.data = np.zeros((self.w, self.h, 3), np.uint8)
        self.points = []
        self.dpoints = []
        self.spoints = []

    def draw_point(self, r, j, rmax, p=1, color='white'):

        # convert to x-y coordiantes
        x, y = self.__convert_coord__(r, j, rmax)
        yt, yd, xl, xr = self.__gen_rect_by_point__(x, y, p)

        if color == 'white':
            self.data[yt:yd, xl:xr] = 255
        elif color == 'red':
            self.data[yt:yd, xl:xr, 2] = 224
        else:
            print("not supported!")
            pass

    def draw_bbox(self, bbox, color="blue"):

        if color == "red":
            c = (0, 0, 255)  # red
        else:
            c = (0, 255, 0)  # blue

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        self.bboxes.append((p1, p2, c))
        # cv2.rectangle(self.data, p1, p2, c , 2, 1)

    def clip_bbox(self, bbox):

        xl = int(bbox[0]) if int(bbox[0]) > 0 else 0
        yt = int(bbox[1]) if int(bbox[1]) > 0 else 0
        xr = int(bbox[0] +
                 bbox[2]) if int(bbox[0] + bbox[2]) <= self.w else self.w
        yd = int(bbox[1] +
                 bbox[3]) if int(bbox[1] + bbox[3]) <= self.h else self.h
        c = self.data[yt:yd, xl:xr]
        return c

    def __convert_coord__(self, r, j, rmax=1):

        theta = j * np.pi / 2 / 250
        x = int(self.w * r * np.cos(theta) // rmax)
        y = int(self.h * r * np.sin(theta) // rmax)

        return x, y

    def __gen_rect_by_point__(self, x, y, p=1):

        xl = x - p if x - p >= 0 else 0
        xr = x + p if x + p < self.w else self.w - 1
        yd = y + p if y + p < self.h else self.h - 1
        yt = y - p if y - p >= 0 else 0

        return yt, yd, xl, xr

    def show(self, cnt = None, win = 'img'):

        data = copy.deepcopy(self.data)
        for bbox in self.bboxes:
            p1, p2, c = bbox
            cv2.rectangle(data, p1, p2, c , 2, 1)
        data = imutils.resize(data, width=500)
        if cnt is not None:
            cv2.putText(data, "counter : " + str(cnt), (200, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow(win, data)
        cv2.waitKey(0)


class BackGround:
    def __init__(self, len=10):
        self.data = [[] for _ in range(250)]
        self.len = len

    def reset(self):

        self.data = [[] for _ in range(250)]

    def enqueue(self, r, i):

        if r not in self.data[i]:
            self.data[i].append(r)
        if len(self.data[i]) > self.len:
            self.data[i].pop(0)
