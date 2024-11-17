from utils.DataLoader import MatLoader
from utils.vis import vis_Channels, vis_EEG_mne, vis_idw, interpolation, xy, channelList
from byVTK import vis


# import numpy as np


if __name__ == '__main__':
    idx = 1
    eyeOpen = True
    t = 50
    N = 200
    known_x, known_y = xy[:, 0], xy[:, 1]
    r = max(abs(min(known_x)), abs(max(known_x)), abs(min(known_y)), abs(max(known_y))) # = Y_Cz
    known_x = [(x+r)/(2*r)*N for x in known_x]
    known_y = [(y+r)/(2*r)*N for y in known_y]

    data_t = MatLoader(idx, eyeOpen).get_data()[:, t]
    title = f"Patient{idx + 1:0>2d} @t={idx + 1:0>4d} with{'EO' if eyeOpen else 'EC'}."

   #  # vis_Channels()
   #  # vis_EEG_mne(t_slice)
   #  # vis_idw(t_slice)
    z = interpolation(data_t, gird_size=N)
    vis(z, N - 1, title, known_x, known_y)
