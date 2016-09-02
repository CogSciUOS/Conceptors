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
             'drift': 0.01,
             'gammaRate': 0.005,
             'dcsv': 4,
             'SigToNoise': float('inf')}

# list of syllables to initialize songClassifier with
syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej','fa','ff','ha','hk']

# song length and repetition and pauses
minSongLength = 3
maxSongLength = 6
pattRepRange = (5,15)
maxPauseLength = 1

# independent variables
SongNumbers = np.arange(2,3)
NoiseScaling = np.arange(0.5,0.6,0.1)

#%% Run songClassifier with above specified parameters and measure classification performance

meanPerformance = np.zeros((len(SongNumbers), len(NoiseScaling)))
# loop over different number of songs
for i,nSongs in enumerate(SongNumbers):
    
    # loop over different noise scalings
    for j,noise in enumerate(NoiseScaling):
        
        
        SC = SongClassifier(syllables, verbose = True)

        # create random songs of random length from syllable list
        songLengths = np.random.randint(minSongLength, high = maxSongLength, size = nSongs)
        for n in range(nSongs):
            SC.addSong(songLengths[n])

        # load songs into RFC
        SC.loadSongs(RFCParams = RFCParams, loadingParams = loadingParams)

        # add white noise to song patterns
        patterns = SC.patterns
        patterns = [p + np.random.randn(p.shape[0], p.shape[1]) * noise for p in patterns]
        
        # run HFC with patterns        
        SC.run(patterns = patterns, nLayers = 1, pattRepRange = pattRepRange, maxPauseLength = maxPauseLength, HFCParams = HFCParams)
        
        # measure classification error
        performance = SC.H.checkPerformance()
        meanPerformance[i,j] = np.mean(performance[-1,:])
        #SC.H.plot_gamma(pltMeanGamma = True, songLengths = songLengths)
        
#%% plot mean performance over independent variables

matshow(meanPerformance, cmap = 'jet', vmin = 0, vmax = 1, interpolation = 'nearest')
colorbar()
title('Mean song classification performance over all patterns')
xlabel('Noise Scaling')
xticks(np.arange(0,len(NoiseScaling)), NoiseScaling)
ylabel('Number of Songs')
yticks(np.arange(0,len(SongNumbers)), SongNumbers)
