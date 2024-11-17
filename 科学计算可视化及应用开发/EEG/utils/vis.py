import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mne.viz import plot_topomap as vis_topomap

from math import sqrt

from .Channel import channelList, exceptChs, get_coor
from .DataLoader import Normalizer

xy = get_coor()

# 可视化电极位置
def vis_Channels():
    Xs, Ys = xy[:,0], xy[:,1]

    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(1,1,1)

    # circle    
    cz = xy[channelList.index('Cz')]
    iz = xy[channelList.index('Iz')]
    c = Circle(xy=cz, radius=abs(iz[1]-cz[1]), edgecolor='black', fill=False)
    ax.add_patch(c)

    # scatter plot
    ax.scatter(Xs, Ys, s=3)
    for idx, ch in enumerate(channelList):
        if ch in exceptChs: continue
        ax.annotate(ch, (Xs[idx], Ys[idx]), xytext=(-6, 5), textcoords='offset points')

    plt.title('Location of 62 Channels')
    plt.show()
    # Cz-Iz 是半径
    return

# ref: https://blog.csdn.net/qq_37566138/article/details/119646578
def vis_EEG_mne(W):
    fig = plt.figure(figsize=(5,5))
    ax  = fig.add_subplot(1,1,1)
    W = Normalizer.zscored(W)
    # 可以通过 vmin, vmax 设置 bar 的最大最小值
    im, cn = vis_topomap(W, xy/5.3, show=False, axes=ax)
    plt.colorbar(im)
    plt.show()
    return

def IDW(known_x, known_y, known_z, x, y):
    Zs = []
    for i in range(len(x)):
        Ds = []
        z = None
        for j in range(len(known_x)): # 已知点
            d = calc_dis(known_x[j], known_y[j], x[i], y[i])
            if d == 0.0: #就是已知点
                z = known_z[j]
                break
            Ds.append(d)
        if z is None:
            squared_dis = list(1/np.power(Ds, 2))
            sum_dis = np.sum(squared_dis)
            z = np.sum(
                np.array(known_z)*np.array(squared_dis)/sum_dis
                )
        Zs.append(z)
    return np.array(Zs)

# 欧式距离
def calc_dis(x1, y1, x2, y2):
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def interpolation(known_z, gird_size):
    known_x, known_y = xy[:, 0], xy[:, 1]
    known_z = Normalizer.zscored(known_z)

    # 生成等间隔 400 的 x 坐标，返回 ndarray = (400, )
    # un_known_x = np.linspace(min(known_x), max(known_x), gird_size)
    # un_known_y = np.linspace(min(known_y), max(known_y), gird_size)
    r = max(abs(min(known_x)), abs(max(known_x)), abs(min(known_y)), abs(max(known_y)))
    un_known_x = np.linspace(-r, r, gird_size)
    un_known_y = np.linspace(-r, r, gird_size)

    # x_grid y_grid = (400, 400)
    # 分别存储了 grid[i][j] 的横纵坐标
    # pos_grid[0][0] = (x_grid[0][0], y_grid[0][0]) 
    x_grid, y_grid = np.meshgrid(un_known_x, un_known_y)
    x, y = x_grid.flatten(), y_grid.flatten()

    return IDW(
        known_x, known_y, known_z, 
        x, y
    ).reshape(gird_size, gird_size)

def vis_idw(W, gird_size=400, twoD=False):
    # 这边 x,y 坐标改成 [0, grid_size] 了
    x_grid = np.tile(np.array([[ i for i in range(gird_size)]]).T, (1,gird_size))
    y_grid = np.tile(np.arange(gird_size), (gird_size,1))

    Zs = interpolation(W, gird_size)
    if twoD:
        sns.heatmap(
            Zs,
            cmap='seismic',
            linewidth=0,
            xticklabels=[], yticklabels=[],
            square=True
        )
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(x_grid, y_grid, Zs, cmap=plt.cm.seismic)
    plt.show()
