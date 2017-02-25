# -*- coding: utf-8 -*-
"""
@author: Kai Standvoss
"""

import numpy as np
from DeepRNN import DeepRNN
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import random

import pickle
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

#%% Parameters

# set random seeds for both numpy and random
SEED = 100
np.random.seed(SEED)
random.seed(SEED)

# network parameters
dropout = 0.416
neurons = 100
layers = 3
activation = 'relu'
epochs = 10
batch_size = 64

# list of syllables to initialize songClassifier with
syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej','fa','ff','ha','hk']

# list of songs to train RFC on
songs = [['aa','bm','aa','ck'],
         ['da','ck','aa','ck'],
         ['dl','ea','ej','ca'],
         ['hk','bm','bm','ca'],
         ['da','da','ao','ao'],
         ['dl','ea','ej','ea'],
         ['ba','ba','fa','fa'],
         ['ca','ff','hk','dl'],
         ['aa','ao','ba','bm'],
         ['ca','ck','da','dl'],
         ['ea','ej','fa','ff','ej'],
         ['bm','hk','hk','bm','ca'],
         ['dl','ba','dl','ba','bm'],
         ['ba','ck','ck','dl','ao'],
         ['ea','ao','ea','ao','aa'],
         ['aa','bm','ck'],
         ['da','ff','ff'],
         ['ff','ck','ck'],
         ['hk','ao','hk'],
         ['ej','dl','ba']]

max_len = max([len(s) for s in songs])

# size of test data set and length of pauses added 
nTestSongs = 100

# number of runs per snr-nSongs combination
cvalN = 10

# independent variables
SongNumbers = np.arange(2,8)
SNR = np.array([4,2,1,0.5,0.25,0.125])

#%% Run songClassifier with above specified parameters and measure classification performance

meanPerformance = np.zeros((len(SongNumbers), len(SNR)))
allPerformances = np.zeros((len(SongNumbers), len(SNR),cvalN))
l = 1

nTestSongs = 100
nTrainRep = 2400

# loop over cval runs
for k in range(cvalN):
    
    # loop over different number of songs
    for i,nSongs in enumerate(SongNumbers):
        
        meanSongLength = nTestSongs/nSongs
    
        rnnClassifier = DeepRNN(dropout,neurons,activation,layers,epochs,batch_size,max_len,nSongs)
        
        idx = np.arange(0,len(songs))
        np.random.shuffle(idx)
    
        # create random songs of random length from syllable list
        data = []
        for n in range(nSongs):
            data.append(songs[idx[n]])
        
        trainData = []
        trainLabels = []

        valData = []
        valLabels = []

        testData = []
        testLabels = []

        for l,song in enumerate(data):
            trainData.extend(int(nTrainRep / len(song)) * [[syllables.index(s) for s in song]])
            testData.extend(nTestSongs * [[syllables.index(s) for s in song]])
            trainLabels.extend(int(nTrainRep / len(song)) *[l])
            testLabels.extend(nTestSongs *[l])


        trainX = pad_sequences(trainData, maxlen=max_len).astype(np.float)    
        trainY = to_categorical(trainLabels)    
        testX = pad_sequences(testData, maxlen=max_len).astype(np.float)  
        testY = to_categorical(testLabels)    
        

        # train classifier
        acc = rnnClassifier.train(trainX,trainY,testX,testY,SNR)
        print()
        print('Test accuracy without noise: {}'.format(acc))
        # loop over different noise scalings
        for j,snr in enumerate(SNR):
            
            # set noise lvl
            data = np.copy(testX)
            noiseLVL = np.sqrt(np.var(data) / snr)
            data += noiseLVL*np.random.randn(data.shape[0],data.shape[1])
            
            performance = rnnClassifier.evaluate(data, testY)
            
            # measure classification error
            allPerformances[i,j,k] = performance
                
            meanPerformance[i,j] = np.mean(allPerformances[i,j,:])
            
            #SC.H.plot_input()
            print(l,'. of ',len(SongNumbers)*len(SNR)*cvalN,' runs finished.')
            print('Accuracy: {}'.format(performance))
            l += 1
        
pickle.dump(meanPerformance, open('rnnResults.pkl', 'wb'))
# plot mean performance over independent variables
plt.matshow(meanPerformance, cmap = 'jet', vmin = 0, vmax = 1, interpolation = 'nearest')
plt.colorbar()
plt.title('Mean song classification performance over all patterns')
plt.xlabel('Signal-to-Noise Ratio')
plt.xticks(np.arange(0,len(SNR)), SNR)
plt.ylabel('Number of Songs')
plt.yticks(np.arange(0,len(SongNumbers)), SongNumbers)
plt.show()
