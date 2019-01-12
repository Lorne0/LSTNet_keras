import numpy as np
import pandas as pd
import pickle as pk

def raw_to_npz(fn):
    df = pd.read_csv("./raw/"+fn, header=None)
    A = df.values.astype(np.float32)
    fn = fn.split(".")[0]
    np.savez_compressed('./data/'+fn, a=A)

class Data(object):
    def __init__(self, fn, tn=0.6, vd=0.2, horizon=3, window=12):
        self.h, self.w = horizon, window
        self.raw = np.load(fn)['a']
        self.n, self.m = self.raw.shape
        self.col_max = np.max(self.raw, axis=0)+1
        self.raw /= self.col_max
        self._slice(tn, vd)

    def _slice(self, tn, vd):
        X = np.zeros((self.n-self.w-self.h, self.w, self.m))
        Y = np.zeros((self.n-self.w-self.h, self.m))
        for i in range(self.w+self.h, self.n):
            X[i-self.w-self.h] = self.raw[i-self.w-self.h:i-self.h].copy()
            Y[i-self.w-self.h] = self.raw[i].copy()

        l = len(X)
        _tn = int(l*tn)
        _vd = int(l*(tn+vd))
        self.train = (X[:_tn].copy(), Y[:_tn].copy())
        self.valid = (X[_tn:_vd].copy(), Y[_tn:_vd].copy())
        self.test = (X[_vd:].copy(), Y[_vd:].copy())
        

if __name__ == '__main__':
    #raw_to_npz("ele.txt")
    #raw_to_npz("er.txt")
    #raw_to_npz("solar.txt")
    #raw_to_npz("traffic.txt")
    pass
