# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:28:35 2016

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

dataPrepParams = {
        'sample_rate': 20000,
        'ds_type': 'mean',
        'mel_channels': 20,
        'inv_coefforder': True,
        'winsize': 20,
        'frames': 64,
        'smooth_length': 5,
        'poly_order': 3,
        'inc_der': [True, True]
        }

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
         
# size of test data set, length of pauses added, number of songs and number of test runs
nTestSongs = 100
maxPauseLength = 1
nSongs = 3
nRuns = 10

#%% run Model

meanSongLength = nTestSongs/nSongs
performance = np.zeros(nRuns)
idx = np.arange(0,nSongs)
for i in range(nRuns):
    
    SC = SongClassifier(syllables, verbose = True)
    
    np.random.shuffle(idx)
    for n in range(nSongs):
        SC.addSong(len(songs[idx[n]]), song = songs[idx[n]])
    
    SC.loadSongs(useSyllRecog = True, SyllPath ='D:/Data/Projects/StudyProject/syll',RFCParams = RFCParams, loadingParams = loadingParams, cLearningParams = cLearningParams, dataPrepParams = dataPrepParams)
    SC.run(patterns = SC.patterns, nLayers = 1, pattRepRange = [meanSongLength, meanSongLength+1], maxPauseLength = maxPauseLength, HFCParams = HFCParams, cLearningParams = cLearningParams, dataPrepParams = dataPrepParams)
    
    p = SC.H.checkPerformance()
    performance[i] = np.mean(p[-1,:])
    print(i+1,'. run of ',nRuns,' runs is finished.')


    
    
    