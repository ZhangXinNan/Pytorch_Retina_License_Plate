
import os
import math
import matplotlib.pyplot as plt


def draw(x, ys, labels, output='show.png'):
    # x = [5, 7, 11, 17, 19, 25]  # 点的横坐标
    # k1 = [0.8222, 0.918, 0.9344, 0.9262, 0.9371, 0.9353]  # 线1的纵坐标
    # k2 = [0.8988, 0.9334, 0.9435, 0.9407, 0.9453, 0.9453]  # 线2的纵坐标
    plt.plot(x, ys[0], 's-', color='r', label=labels[0])  # s-:方形
    plt.plot(x, ys[1], 'o-', color='g', label=labels[1])  # o-:圆形
    plt.plot(x, ys[2], '--', color='b', label=labels[2])  # o-:圆形
    plt.xlabel("epoch")  # 横坐标名字
    plt.ylabel("loss")  # 纵坐标名字
    plt.legend(loc="best")  # 图例
    # plt.show()
    plt.savefig(output)
    plt.clf()


# draw(None, None, None)
'''
x = [i for i in range(100)]
ys = [[math.log(i+0.1) for i in range(100)],
      [math.sin(i * math.pi / 10) for i in range(100)],
      [math.cos(i * math.pi / 10) for i in range(100)]]
titles = ['log', 'sin', 'cos']
draw(x, ys, labels=titles)
'''