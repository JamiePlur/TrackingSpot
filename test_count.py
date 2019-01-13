# -*- coding: utf-8 -*-
"""
统计若干视频的行人数量
"""

import os 
from count import Count

label = 0
cnt = 0
root = "tmp"

for dir in os.listdir(root):
    _, l = dir[:-4].split('_')
    label += int(l)
    path = os.path.join(root, dir)
    cnt += Count(path)
    
    
print("目测人数为:", label)
print("计算人数：", cnt)


