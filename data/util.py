
import numpy as np


def order_points_clockwise(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]
    return rect


def order_points_clockwise_8int(pts):
    pts_arr = np.array(pts).reshape((4, 2)).astype(np.float32)
    rect = order_points_clockwise(pts_arr)
    rect = rect.reshape(8).astype(np.int32)
    return rect


if __name__ == '__main__':
    with open('/home/xin.zhang6/data_br/val.txt', 'r') as fi:
        for line in fi:
            line = line.strip('\n').split(',')
            # arr = line.strip().split(',')
            lable = list(map(int, line[1:]))
            pts = order_points_clockwise_8int(lable[6:14])
            print(line)
            print(pts)
