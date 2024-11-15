import math
import numpy as np

channelList = [x.capitalize() for x in [
    'Fp1','AF7','AF3','F1','F3','F5','F7','FT7','FC5','FC3','FC1','C1','C3','C5','T7','TP7','CP5','CP3','CP1','P1','P3','P5','P7',
    'P9','PO7','PO3','O1','Iz','Oz','POz','Pz','CPz','Fpz',
    'Fp2','AF8','AF4','Afz','Fz','F2','F4','F6','F8','FT8','FC6','FC4','FC2','FCz','Cz',
    'C2','C4','C6','T8','TP8','CP6','CP4','CP2','P2','P4','P6','P8','P10','PO8','PO4','O2'
]]
exceptChs = ['P9', 'P10']

# ctmd besa 既不是直角坐标，又不是极坐标
# 以 Cz 为原点(0,0)
# coor = {
#     'Fp1': (-92, -72), 'AF7': (-92, -54), 'AF3': (-74, -65), 'F1': (-50, -68), 'F3': (-60, -51), 'F5': (-75, -41), 'F7': (-92, -36), 'FT7': (-92, -18), 'FC5': (-72, -21), 'FC3': (-50, -28), 'FC1': (-32, -45), 'C1': (-23, 0), 'C3': (-46, 0), 'C5': (-69, 0), 'T7': (-92, 0), 'TP7': (-92, 18), 'CP5': (-72, 21), 'CP3': (-50, 28), 'CP1': (-32, 45), 'P1': (-50, 68), 'P3': (-60, 51), 'P5': (-75, 41), 'P7': (-92, 36), 'P9': (-115, 36), 'PO7': (-92, 54), 'PO3': (-74, 65), 'O1': (-92, 72), 'Iz': (115, -90), 'Oz': (92, -90), 'POz': (69, -90), 'Pz': (46, -90), 'CPz': (23, -90), 'Fpz': (92, 90), 'Fp2': (92, 72), 'AF8': (92, 54), 'AF4': (74, 65), 'Afz': (69, 90), 'Fz': (46, 90), 'F2': (50, 68), 'F4': (60, 51), 'F6': (75, 41), 'F8': (92, 36), 'FT8': (92, 18), 'FC6': (72, 21), 'FC4': (50, 28), 'FC2': (32, 45), 'FCz': (23, 90), 'Cz': (0, 0), 'C2': (23, 0), 'C4': (46, 0), 'C6': (69, 0), 'T8': (92, 0), 'TP8': (92, -18), 'CP6': (72, -21), 'CP4': (50, -28), 'CP2': (32, -45), 'P2': (50, -68), 'P4': (60, -51), 'P6': (75, -41), 'P8': (92, -36), 'P10': (115, -36), 'PO8': (92, -54), 'PO4': (74, -65), 'O2': (92, -72)
# }

def get_coor(file_url: str = './dataset/biosemi_64_besa_sph.besa'):
    # coor = {}
    # with open(file_url, 'r') as f:
    #     for line in f.readlines():
    #         cols = line.split()
    #         coor[cols[1]] = (int(cols[2]), int(cols[3]))
    # return coor  
    return polar_to_cartesian(get_polar_coor())

def get_polar_coor(file_url: str = './dataset/eloc64.txt'):
    coor = {}
    with open(file_url, 'r') as f:
        for line in f.readlines():
            cols = line.split('\t')
            if len(cols) < 4: break
            ch   = cols[3][:cols[3].index('.')]
            coor[ch] = (int(cols[1]), float(cols[2]))
    return coor

def polar_to_cartesian(polar_coor):
    xy = np.zeros([len(polar_coor), 2])
    for ch, coor in polar_coor.items():
        try:    idx = channelList.index(ch)
        except: continue
        theta, rou = coor
        xy[idx, 1] = math.cos(math.radians(theta))*rou
        xy[idx, 0] = math.sin(math.radians(theta))*rou
    return xy

def ch_to_idx(ch: str):
    return channelList.index(ch)

def idx_to_ch(idx: int):
    assert idx>=0 and idx<64
    return channelList[idx]

if __name__ == '__main__':
    print(polar_to_cartesian(get_polar_coor()))