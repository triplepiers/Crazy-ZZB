from utils.DataLoader import MatLoader
from utils.vis import vis_Channels, vis_EEG_mne, vis_idw, interpolation, xy, channelList
from byVTK import vis

# 可能需要安装的包：mne, vtk, pandas, numpy, sciy
# 可以直接 python app.py，然后缺什么装什么

if __name__ == '__main__':
    idx = 1           # 病人 ID
    eyeOpen = True    # 睁眼 / 闭眼
    t = 50            # 取数据的哪一帧 in [0, 38401)
    N = 200           # IDW 的网格数量 => N*N
    known_x, known_y = xy[:, 0], xy[:, 1]
    r = max(abs(min(known_x)), abs(max(known_x)), abs(min(known_y)), abs(max(known_y))) # = Y_Cz
    known_x = [(x+r)/(2*r)*N for x in known_x]
    known_y = [(y+r)/(2*r)*N for y in known_y]

    data_t = MatLoader(idx, eyeOpen).get_data()[:, t]
    title = f"Patient{idx + 1:0>2d} @t={idx + 1:0>4d} with{'EO' if eyeOpen else 'EC'}."

    """
    把需要看的图解除注释就行
    => 请互斥的执行，否则可能会造成谜之覆盖
    """
    # vis_Channels()              # 单纯画一下电极的位置
    # vis_EEG_mne(data_t)         # 拿 MNE 包画的，因为用的插值算法不一样、差异较大
    # vis_idw(data_t, twoD=False) # 基于 IDW 插值，拿 matplot 画的三维
    # vis_idw(data_t, twoD=True)  # 基于 IDW 插值，拿 matplot 画的二维（heatmap）

    # 基于 IDW 插值，拿 VTK 画的二维
    # TODO: 基于 Marching Square 的等值线提取 + 拖动形成动图？（算法效率低，会卡顿orz）
    vis(interpolation(data_t, gird_size=N), N - 1, title, known_x, known_y)
