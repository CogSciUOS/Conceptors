from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from hyperopt import hp, fmin, tpe

import numpy as np
from keras.utils import np_utils

from DeepRNN import DeepRNN

def data():

    syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej','fa','ff','ha','hk']

    songs = [['aa','bm','ck'],
         ['ao','da','ao','ej'],
         ['ba','ck','ck','dl','ao'],
         ['da','ff','ff'],
         ['ba','ba','fa','fa'],
         ['dl','ha','dl','ha','bm'],
         ['hk','aa','da','hk']]

    nTestSongs = 100
    nTrainRep = 2400
    nvalRep = 3

    max_song_len  = max([len(s) for s in songs])

    X_train = []
    Y_train = []
    X_val = []
    Y_val = []

    testData = []
    testLabels = []

    SNR = np.array([4,2,1,0.5,0.25,0.125])

    scoresPerNum = []
    for l,song in enumerate(songs):
        X_train.extend(int(nTrainRep / len(song)) * [[syllables.index(s) for s in song]])
        Y_train.extend(int(nTrainRep / len(song)) * [l])
        
        X_val.extend(nvalRep * [[syllables.index(s) for s in song]])
        Y_val.extend(nvalRep *[l])

        testData.extend(nTestSongs * [[syllables.index(s) for s in song]])
        testLabels.extend(nTestSongs *[l])

    Y_val *= len(SNR)

    X_train = pad_sequences(X_train, maxlen=max_song_len).astype(np.float)    
    Y_train = to_categorical(Y_train)      
    X_val = pad_sequences(X_val, maxlen=max_song_len).astype(np.float)    
    Y_val = to_categorical(Y_val) 

    data = np.zeros((len(SNR) * X_val.shape[0], X_val.shape[1]))
    for i,snr in enumerate(SNR):
        d = np.copy(X_val)
        noiseLVL = np.sqrt(np.var(d) / snr)
        d += noiseLVL*np.random.randn(X_val.shape[0],X_val.shape[1])
        
        ind = i*X_val.shape[0]
        data[ind:ind+X_val.shape[0],:] = d

    X_val = data 


    X_test = pad_sequences(testData, maxlen=max_song_len).astype(np.float)  
    Y_test = to_categorical(testLabels) 

    return X_train, Y_train, X_val, Y_val, SNR, X_test, Y_test


def model(params):

    max_len = 5
    n_songs = 7
    classifier = DeepRNN(*params, max_len, n_songs)

    acc = classifier.train(X_train, Y_train, X_val, Y_val, SNR)

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}



if __name__ == '__main__':

    X_train, Y_train, X_val, Y_val, SNR, X_test, Y_test = data()

    space = DeepRNN.space

    best_run = fmin(model, space, algo = tpe.suggest, max_evals = 100)
    print()
    print(best_run)
