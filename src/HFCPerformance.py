# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 11:54:35 2016

@author: Richard Gast
"""

import numpy as np
from songClassifier import *
from matplotlib.pyplot import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#%% Parameters

# network parameters
RFCParams = {'N': 400,
             'K': 2000,
             'NetSR': 1.5,
             'bias_scale': 1.2,
             'inp_scale': 1.5}
             
loadingParams = {'gradient_c': True}

dataPrepParams = {}

cLearningParams = {'neurons': 10,
                   'spectral_radius': 1.2,
                   'bias_scale': 0.2,
                   'inp_scale': 1.,
                   'conn': 1.,
                   'gammaPos': 25,
                   'gammaNeg': 27}
                   
HFCParams = {'sigma': 0.82,
             'drift': 0.,
             'gammaRate': 0.005,
             'dcsv': 4,
             'SigToNoise': float('inf')}

# list of syllables to initialize songClassifier with
syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej','fa','ff','ha','hk']

# list of songs to train RFC on
songs = [['aa','bm','ck'],
         ['ao','da','ao','ej'],
         ['ba','ck','ck','dl','ao'],
         ['da','ff','ff'],
         ['ba','ba','fa','fa'],
         ['dl','ha','dl','ha','bm'],
         ['hk','aa','da','hk']]
         
# size of test data set and length of pauses added 
nTestSongs = 100
maxPauseLength = 1

# independent variables
SongNumbers = np.arange(2,8)
SNR = np.array([4,2,1,0.5,0.25,0.125])

#%% Run songClassifier with above specified parameters and measure classification performance

meanPerformance = np.zeros((len(SongNumbers), len(SNR)))
k = 1
# loop over different number of songs
for i,nSongs in enumerate(SongNumbers):
    
    meanSongLength = nTestSongs/nSongs

    SC = SongClassifier(syllables, verbose = True)
    
    idx = np.arange(0,nSongs)
    np.random.shuffle(idx)

    # create random songs of random length from syllable list
    for n in range(nSongs):
        SC.addSong(len(songs[idx[n]]), song = songs[idx[n]])

    # load songs into RFC
    SC.loadSongs(RFCParams = RFCParams, loadingParams = loadingParams)    
    
    # loop over different noise scalings
    for j,snr in enumerate(SNR):
        
        # set noise lvl
        HFCParams['SigToNoise'] = snr
        
        # run HFC with patterns        
        SC.run(patterns = SC.patterns, nLayers = 1, pattRepRange = [meanSongLength, meanSongLength+1], maxPauseLength = maxPauseLength, HFCParams = HFCParams)
        
        # measure classification error
        performance = SC.H.checkPerformance()
        meanPerformance[i,j] = np.mean(performance[-1,:])
        #SC.H.plot_input()
        print(k,'. of ',len(SongNumbers)*len(SNR),' runs finished.')
        k += 1
        
#%% plot mean performance over independent variables

matshow(meanPerformance, cmap = 'jet', vmin = 0, vmax = 1, interpolation = 'nearest')
colorbar()
title('Mean song classification performance over all patterns')
xlabel('Signal-to-Noise Ratio')
xticks(np.arange(0,len(SNR)), SNR)
ylabel('Number of Songs')
yticks(np.arange(0,len(SongNumbers)), SongNumbers)

