import numpy as np
import pandas as pd
import pickle as pk

def raw_to_npz(fn):
    df = pd.read_csv("./raw/"+fn, header=None)
    A = df.values.astype(np.float32)
    fn = fn.split(".")[0]
    np.savez_compressed('./data/'+fn, a=A)

if __name__ == '__main__':
    raw_to_npz("ele.txt")
    raw_to_npz("er.txt")
    raw_to_npz("solar.txt")
    raw_to_npz("traffic.txt")
