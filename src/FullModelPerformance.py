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
import random

#%% Parameters

# set random seeds for both numpy and random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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
        'inc_der': [True, True],
        'snr': 0
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
cvalN = 10

# independent variables
SongNumbers = np.arange(2,8)

#%% run Model

meanSongLength = nTestSongs/len(songs)
performance = np.zeros((len(SongNumbers),cvalN))

idx = np.arange(0,len(songs))
l = 0

for i in range(cvalN):
    
    for j,nSongs in enumerate(SongNumbers):
        
        SC = SongClassifier(syllables, verbose = True)
        
        np.random.shuffle(idx)
        for n in range(nSongs):
            SC.addSong(len(songs[idx[n]]), song = songs[idx[n]])
        
        SC.loadSongs(useSyllRecog = True, SyllPath ='../data/birddb/syll',RFCParams = RFCParams, loadingParams = loadingParams, cLearningParams = cLearningParams, dataPrepParams = dataPrepParams)
        SC.run(patterns = SC.patterns, nLayers = 1, pattRepRange = [meanSongLength, meanSongLength+1], maxPauseLength = maxPauseLength, HFCParams = HFCParams, cLearningParams = cLearningParams, dataPrepParams = dataPrepParams)
        
        p = SC.H.checkPerformance()
        performance[j,i] = np.mean(p[-1,:])
        print(l+1,'. run of ',cvalN*len(SongNumbers),' runs is finished.')

    np.save('/home/rgast/Documents/GitRepo/BirdsongRecog/src/evaluation/fullModel_Results', performance)