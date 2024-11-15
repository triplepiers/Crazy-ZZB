import scipy.io as sio

class MatLoader(object):
    def __init__(self, idx: int=1, eyeOpen: bool=True):
        assert idx>=1 and idx<=10

        self.url = f"./dataset/S{idx+1:0>2d}_restingPre_{'EO' if eyeOpen else 'EC'}.mat"
        print(f'data loaded from: {self.url}')

        return
    
    # 只返回前 64 纬电极数据，shape = (64, 38401)
    # rtpye: np.ndarray
    def get_data(self):
        return sio.loadmat(self.url)['dataRest'][:64]

class Normalizer(object):

    @staticmethod
    def zscored(data):
        return (data - data.mean())/data.std()
    
    @staticmethod
    def minmaxed(data):
        return (data - data.min())/(data.max()-data.min())

if __name__ == '__main__':
    idx     = int(input('Pleas input patient index in [1,10]: '))
    eyeOpen = int(input('Choose a status (eyeOpen-0, eyeClosed-1): ')) == 0
    # load data
    loader = MatLoader(idx, eyeOpen)
    data = loader.get_data()
    print(f'{data.shape = }')
