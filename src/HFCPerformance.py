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
import random

#%% Parameters

# set random seeds for both numpy and random
SEED = 100
np.random.seed(SEED)
random.seed(SEED)

# network parameters
RFCParams = {'N': 600,
             'K': 3000,
             'NetSR': 1.1,
             'bias_scale': 1.5,
             'inp_scale': 1.7}
             
loadingParams = {'gradient_c': True}

dataPrepParams = {}
                   
HFCParams = {'sigma': 0.82,
             'drift': 0.,
             'gammaRate': 0.005,
             'dcsv': 4,
             'SigToNoise': float('inf')}

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
         
# size of test data set and length of pauses added 
nTestSongs = 100
maxPauseLength = 1

# number of runs per snr-nSongs combination
cvalN = 1

# independent variables
SongNumbers = np.arange(2,8)
SNR = np.array([4,2,1,0.5,0.25,0.125])

#%% Run songClassifier with above specified parameters and measure classification performance

meanPerformance = np.zeros((len(SongNumbers), len(SNR)))
allPerformances = np.zeros((len(SongNumbers), len(SNR),cvalN))
l = 1

# loop over cval runs
for k in range(cvalN):
    
    # loop over different number of songs
    for i,nSongs in enumerate(SongNumbers):
        
        meanSongLength = nTestSongs/nSongs
    
        SC = SongClassifier(syllables, verbose = True)
        
        idx = np.arange(0,len(songs))
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
            allPerformances[i,j,k] = np.mean(performance[-1,:])
                
            meanPerformance[i,j] = np.mean(allPerformances[i,j,:])
            
            #SC.H.plot_input()
            print(l,'. of ',len(SongNumbers)*len(SNR)*cvalN,' runs finished.')
            l += 1
        
#%% plot mean performance over independent variables

matshow(meanPerformance, cmap = 'jet', vmin = 0, vmax = 1, interpolation = 'nearest')
colorbar()
title('Mean song classification performance over all patterns')
xlabel('Signal-to-Noise Ratio')
xticks(np.arange(0,len(SNR)), SNR)
ylabel('Number of Songs')
yticks(np.arange(0,len(SongNumbers)), SongNumbers)

