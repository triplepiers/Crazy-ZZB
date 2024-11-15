import pandas as pd 
import numpy as np

from utils.DataLoader import MatLoader
from utils.vis import vis_Channels, vis_EEG_mne

loader = MatLoader()
# load & normed
data = loader.get_data()
#  Normalizer.minmaxed(
#     loader.get_data()
# )

if __name__ == '__main__':
    # vis_Channels()
    t = 50
    t_slice = data[:, t]
    vis_EEG_mne(t_slice)
   