import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mne.viz import plot_topomap as vis_topomap

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