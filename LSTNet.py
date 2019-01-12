import numpy as np
import keras
from keras.layers import Input, Dense, TimeDistributed, Conv1D, GRU, concatenate, add, Dropout, Flatten, Activation, Lambda
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

    def make_model(self, batch_size):
        
        x = Input(shape=(self.P, self.m))

        # CNN
        c = Conv1D(self.hidC, self.Ck, activation='relu')(x)
        c = Dropout(self.dropout)(c)

        # RNN
        #r = GRU(self.hidR)(TimeDistributed(c))
        _, r = GRU(self.hidR, return_state=True)(c)
        #r = K.reshape(r, (-1, self.hidR))
        #r = Lambda(lambda k: K.reshape(k, (batch_size, self.hidR)))(r)
        r = Lambda(lambda k: K.reshape(k, (-1, self.hidR)))(r)
        r = Dropout(self.dropout)(r)

        # skip-RNN
        if self.skip > 0:
            # c: batch_size*steps*filters, steps=P-Ck
            #s = c[:, int(-self.pt*self.skip):, :] # need to make the steps divisible
            s = Lambda(lambda k: k[:, int(-self.pt*self.skip):, :])(c)
            #s = K.reshape(s, (batch_size, self.pt, self.skip, self.hidC)) 
            #s = Lambda(lambda k: K.reshape(k, (batch_size, self.pt, self.skip, self.hidC)))(s)
            s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.skip, self.hidC)))(s)
            #s = K.permute_dimensions(s, (0,2,1,3))
            s = Lambda(lambda k: K.permute_dimensions(k, (0,2,1,3)))(s)
            #s = K.reshape(s, (batch_size*self.skip, self.pt, self.hidC))
            #s = Lambda(lambda k: K.reshape(k, (batch_size*self.skip, self.pt, self.hidC)))(s)
            s = Lambda(lambda k: K.reshape(k, (-1, self.pt, self.hidC)))(s)
            #s = GRU(self.hidS)(TimeDistributed(s))
            _, s = GRU(self.hidS, return_state=True)(s)
            #s = K.reshape(s, (-1, self.skip*self.hidS)) 
            #s = K.reshape(s, (batch_size, self.skip*self.hidS)) 
            #s = Lambda(lambda k: K.reshape(k, (batch_size, self.skip*self.hidS)))(s)
            s = Lambda(lambda k: K.reshape(k, (-1, self.skip*self.hidS)))(s)
            s = Dropout(self.dropout)(s)
            r = concatenate([r,s])
        
        #res = Flatten()(r)
        res = Dense(self.m)(r)
        #res = TimeDistributed(Dense(self.m))(r)

        # highway
        if self.hw > 0:
            #z = x[:, -self.hw:, :]
            z = Lambda(lambda k: k[:, -self.hw:, :])(x)
            #z = K.permute_dimensions(z, (0,2,1))
            z = Lambda(lambda k: K.permute_dimensions(k, (0,2,1)))(z)
            #z = K.reshape(z, (batch_size*self.m, self.hw))
            #z = K.reshape(z, (-1, self.hw))
            z = Lambda(lambda k: K.reshape(k, (-1, self.hw)))(z)
            z = Dense(1)(z)
            #z = K.reshape(z, (batch_size, self.m))
            #z = Lambda(lambda k: K.reshape(k, (batch_size, self.m)))(z)
            z = Lambda(lambda k: K.reshape(k, (-1, self.m)))(z)
            res = add([res, z])
       
        if self.output != 'no':
            res = Activation(self.output)(res)

        model = Model(inputs=x, outputs=res)
        model.compile(optimizer=Adam(lr=self.lr), loss=self.loss)
        return model



