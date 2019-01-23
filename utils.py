import numpy as np
import pandas as pd
import pickle as pk

def raw_to_npz(fn):
    df = pd.read_csv("./raw/"+fn, header=None)
    A = df.values.astype(np.float32)
    fn = fn.split(".")[0]
    np.savez_compressed('./data/'+fn, a=A)

class Data(object):
    def __init__(self, args, tn=0.6, vd=0.2):
        self.h, self.w, self.skip, self.ps, self.Ck = args.horizon, args.window, args.skip, args.ps, args.CNN_kernel
        self.raw = np.load(args.data)['a']
        self.n, self.m = self.raw.shape
        self.tn, self.vd = tn, vd

        if args.normalize==1:
            self.mx = np.max(np.abs(self.raw))
            self.raw /= self.mx
        elif args.normalize==2:
            self.col_max = np.max(np.abs(self.raw), axis=0)+1
            self.raw /= self.col_max

        if args.multi:
            self._split(self._slice_multi())
        else:
            self._split(self._slice())

    def _slice(self):
        s = self.w+self.h-1
        X = np.zeros((self.n-s, self.w, self.m))
        Y = np.zeros((self.n-s, self.m))
        for i in range(s, self.n):
            #X[i-s] = self.raw[i-s:i-s+self.w].copy()
            X[i-s] = self.raw[i-self.h+1-self.w:i-self.h+1].copy()
            Y[i-s] = self.raw[i].copy()
        return X, Y

    def _slice_multi(self):
        s = self.ps*self.skip+self.Ck-1 + self.h-1
        X1 = np.zeros((self.n-s, self.w, self.m))
        X2 = np.zeros((self.n-s, self.ps*self.Ck, self.m))
        Y  = np.zeros((self.n-s, self.m))
        for i in range(s, self.n):
            t = i-self.h+1
            X1[i-s] = self.raw[t-self.w:t].copy()
            idx = []
            for k in range(self.ps):
                idx = list(range(t-self.Ck-k*self.skip, t-k*self.skip)) + idx
            idx = np.array(idx, dtype=int)
            X2[i-s] = self.raw[idx].copy()
            Y[i-s]  = self.raw[i].copy()
        return X1, X2, Y

    def _split(self, *args):
        tn = int(self.n*self.tn)
        vd = int(self.n*(self.tn+self.vd))
        self.train, self.valid, self.test = [], [], []
        arg = args[0]
        for A in arg:
            self.train.append(A[:tn].copy())
            self.valid.append(A[tn:vd].copy())
            self.test.append(A[vd:].copy())
        
