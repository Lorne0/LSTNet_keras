import numpy as np
import keras
from keras.layers import Input, Dense, Conv1D, GRU, Dropout, Flatten, Activation
from keras.layers import concatenate, add, Lambda
from keras.models import Model, Sequential
from keras.optimizers import Adam
import keras.backend as K

class LSTNet(object):
    def __init__(self, args, dims):
        super(LSTNet, self).__init__()
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
        self.lr = args.lr
        self.loss = args.loss
        self.clip = args.clip

    def make_model(self):
        
        x = Input(shape=(self.P, self.m))

        # CNN
        c = Conv1D(self.hidC, self.Ck, activation='relu')(x)
        c = Dropout(self.dropout)(c)

        # RNN

        r = GRU(self.hidR)(c)
        r = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r)
        r = Dropout(self.dropout)(r)

        # skip-RNN
        if self.skip > 0:
            # c: batch_size*steps*filters, steps=P-Ck
            s = Lambda(lambda k: k[:, int(-self.pt*self.skip):, :])(c)
            s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.skip, self.hidC)))(s)
            s = Lambda(lambda k: K.permute_dimensions(k, (0,2,1,3)))(s)
            s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.hidC)))(s)

            s = GRU(self.hidS)(s)
            s = Lambda(lambda k: K.reshape(k, (-1, self.skip*self.hidS)))(s)
            s = Dropout(self.dropout)(s)
            r = concatenate([r,s])
        
        res = Dense(self.m)(r)

        # highway
        if self.hw > 0:
            z = Lambda(lambda k: k[:, -self.hw:, :])(x)
            z = Lambda(lambda k: K.permute_dimensions(k, (0,2,1)))(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.hw)))(z)
            z = Dense(1)(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.m)))(z)
            res = add([res, z])
       
        if self.output != 'no':
            res = Activation(self.output)(res)

        model = Model(inputs=x, outputs=res)
        model.compile(optimizer=Adam(lr=self.lr, clipnorm=self.clip), loss=self.loss)
        return model

class LSTNet_multi_inputs(object):
    def __init__(self, args, dims):
        super(LSTNet_multi_inputs, self).__init__()
        self.P = args.window
        self.m = dims
        self.hidR = args.hidRNN
        self.hidC = args.hidCNN
        self.hidS = args.hidSkip
        self.Ck = args.CNN_kernel
        self.skip = args.skip
        #self.pt = int((self.P-self.Ck)/self.skip)
        self.pt = args.ps
        self.hw = args.highway_window
        self.dropout = args.dropout
        self.output = args.output_fun
        self.lr = args.lr
        self.loss = args.loss
        self.clip = args.clip

    def make_model(self):
        
        # Input1: short-term time series
        input1 = Input(shape=(self.P, self.m))
        # CNN
        conv1 = Conv1D(self.hidC, self.Ck, strides=1, activation='relu') # for input1
        # It's a probelm that I can't find any way to use the same Conv1D layer to train the two inputs, 
        # since input2's strides should be Ck, not 1 as input1
        conv2 = Conv1D(self.hidC, self.Ck, strides=self.Ck, activation='relu') # for input2
        conv2.set_weights(conv1.get_weights()) # at least use same weight

        c1 = conv1(input1)
        c1 = Dropout(self.dropout)(c1)
        # RNN
        r1 = GRU(self.hidR)(c1)
        #r1 = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r1)
        r1 = Dropout(self.dropout)(r1)

        # Input2: long-term time series with period
        input2 = Input(shape=(self.pt*self.Ck, self.m))
        # CNN
        c2 = conv2(input2)
        c2 = Dropout(self.dropout)(c2)
        # RNN
        r2 = GRU(self.hidS)(c2)
        #r2 = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r2)
        r2 = Dropout(self.dropout)(r2)

        r = concatenate([r1,r2])
        res = Dense(self.m)(r)

        # highway
        if self.hw > 0:
            z = Lambda(lambda k: k[:, -self.hw:, :])(input1)
            z = Lambda(lambda k: K.permute_dimensions(k, (0,2,1)))(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.hw)))(z)
            z = Dense(1)(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.m)))(z)
            res = add([res, z])
       
        if self.output != 'no':
            res = Activation(self.output)(res)

        model = Model(inputs=[input1, input2], outputs=res)
        model.compile(optimizer=Adam(lr=self.lr, clipnorm=self.clip), loss=self.loss)
        return model
