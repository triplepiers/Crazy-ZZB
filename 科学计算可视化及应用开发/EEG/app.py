from utils.DataLoader import MatLoader
from utils.vis import vis_Channels, vis_EEG_mne, vis_idw, interpolation
from byVTK import vis

# import numpy as np



if __name__ == '__main__':
    idx = 1
    eyeOpen = True
    t = 50
    N = 400

    data_t = MatLoader(idx, eyeOpen).get_data()[:, t]
    title = f"Patient{idx + 1:0>2d} @t={idx + 1:0>4d} with{'EO' if eyeOpen else 'EC'}."

    # vis_Channels()
    # vis_EEG_mne(t_slice)
    # vis_idw(t_slice)
    data = interpolation(data_t, gird_size=N).flatten()
    # print(f'{data.shape = }, {np.unique(data).shape = }')
    vis(data, N - 1, title)
   