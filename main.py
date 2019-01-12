import argparse
from utils import *
from LSTNet import *
import numpy as np
import keras.backend as K
import tensorflow as tf

def get_session(gpu_fraction=0.4):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(get_session())


def evaluate(y, yp, mx):
    # rrse
    rrse = np.sqrt(np.sum(np.square(y-yp)) / np.sum(np.square(np.mean(y)-y)))
    # corr
    m, mp = np.mean(y, axis=0), np.mean(yp, axis=0)
    corr = np.mean(np.sum((y-m)*(yp-mp), axis=0) / np.sqrt(np.sum(np.square(y-m), axis=0)*np.sum(np.square(yp-mp), axis=0)))
    #m, mp, sig, sigp = y.mean(axis=0), yp.mean(axis=0), y.std(axis=0), yp.std(axis=0)
    #corr = ((((y-m)*(yp-mp)).mean(axis=0) / (sig*sigp))[sig!=0]).mean()
    #corr = ((((y-m)*(yp-mp)).mean(axis=0) / (sig*sigp))).mean()
    
    # rmse
    rmse = mx*np.sqrt(np.mean(np.square(y[:,0]-yp[:,0])))
    
    return rrse, corr, rmse

def main(args):
    data = Data(args.data, horizon=args.horizon, window=args.window)
    print(data.train[0].shape, data.train[1].shape)
    print(data.valid[0].shape, data.valid[1].shape)
    print(data.test[0].shape, data.test[1].shape)
    M = LSTNet(args, data.m)
    model = M.make_model(args.batch_size)

    ### Train ###
    l = len(data.train[0])
    order = np.arange(l)
    train_batch_num = int(l/args.batch_size)
    for e in range(1,1000+1):
        np.random.shuffle(order)
        for b in range(train_batch_num):
            b_x = data.train[0][order][b*args.batch_size:(b+1)*args.batch_size]
            b_y = data.train[1][order][b*args.batch_size:(b+1)*args.batch_size]
            model.train_on_batch(b_x, b_y)
        y = model.predict(data.valid[0])
        rrse, corr, rmse = evaluate(data.valid[1], model.predict(data.valid[0]), data.col_max[0])
        print("Valid | rrse: %.4f | corr: %.4f | rmse: %.4f" %(rrse, corr, rmse))
        if e%5==0:
            rrse, corr, rmse = evaluate(data.test[1], model.predict(data.test[0]), data.col_max[0])
            print("\tTest | rrse: %.4f | corr: %.4f | rmse: %.4f" %(rrse, corr, rmse))
            model.save(args.save)
            




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Keras Time series forecasting')
    parser.add_argument('--data', type=str, required=True, help='location of the data file')
    parser.add_argument('--hidCNN', type=int, default=100, help='number of CNN hidden units')
    parser.add_argument('--hidRNN', type=int, default=100, help='number of RNN hidden units')
    parser.add_argument('--hidSkip', type=int, default=5)
    parser.add_argument('--window', type=int, default=24*7, help='window size')
    parser.add_argument('--horizon', type=int, default=3)
    parser.add_argument('--skip', type=float, default=24)
    parser.add_argument('--CNN_kernel', type=int, default=6, help='the kernel size of the CNN layers')
    parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')
    #parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=54321, help='random seed')
    #parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default='model/model.pt', help='path to save the final model')
    #parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss', type=str, default='mae')
    #parser.add_argument('--normalize', type=int, default=2)
    parser.add_argument('--output_fun', type=str, default='sigmoid')
    args = parser.parse_args()

    main(args)
