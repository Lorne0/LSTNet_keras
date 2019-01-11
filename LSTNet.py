import numpy as np
import keras
from keras.layers import Input, Dense, TimeDistributed, Conv1D, GRU, concatenate, add
from keras.models import Model, Sequential
import keras.backend as K

class Model():
    def __init__(self, args, dims):
        super(Model, self).__init__()
        self.P = args.window
        self.m = dims
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        self.pt = int((self.P-self.Ck)/self.skip)
        self.hw = args.highway_window
        self.dropout = args.dropout
        self.output = args.output_fun

    def make_model(self, batch_size):
        
        x = Input(shape=(self.P, self.m))

        # CNN
        c = Conv1D(self.hidC, self.Ck, activation='relu')(x)
        c = Dropout(self.dropout)(c)

        # RNN
        r = GRU(self.hidR)(TimeDistributed(c))
        r = Dropout(self.dropout)(r)

        # skip-RNN
        if self.skip > 0:
            # c: batch_size*steps*filters, steps=P-Ck
            s = c[:, int(-self.pt*self.skip):, :] # need to make the steps divisible
            s = K.reshape(s, (batch_size, self.pt, self.skip, self.hidC)) 
            s = K.permute_dimensions(s, (0,2,1,3))
            s = K.reshape(s, (batch_size*self.skip, self.pt, self.hidC))
            s = GRU(self.hidS)(TimeDistributed(s))
            s = Dropout(self.dropout)(s)
            r = concatenate([r,s], axis=1)
        
        res = Dense(self.m)(r)

        # highway
        if self.hw > 0:
            z = x[:, -self.hw:, :]
            z = K.permute_dimensions(z, (0,2,1))
            z = K.reshape(z, (batch_size*self.m, self.hw))
            z = Dense(1, activation=self.output)(z)
            z = K.reshape(z, (batch_size, self.m))
            res = add([res, z])
        
        return res



