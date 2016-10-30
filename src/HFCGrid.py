# -*- coding: utf-8 -*-


import numpy as np
import sys
from matplotlib.pyplot import *
from songClassifier import *
import warnings
import random
import json
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def main(args):
    
    seedMeanPerformances = []
    seedMeanVariances = []
    for seed in [23, 42, 1337]:

        np.random.seed(seed)
        random.seed(seed)
        #%% Parameters

        # network parameters
        networkSize = int(args[1])
        gamma = float(args[2])
        targetDir = args[3]

        RFCParams = {'N': networkSize,
                     'K': networkSize * 5,
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
                     'gammaRate': gamma,
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
        SongNumber = 4
        SNR = np.array([4,2,1,0.5,0.25,0.125])

        meanSongLength = nTestSongs/SongNumber

        #%% Run songClassifier with above specified parameters and measure classification performance

        SC = SongClassifier(syllables, verbose = True)

        idx = np.arange(0,SongNumber)
        np.random.shuffle(idx)

        # create random songs of random length from syllable list
        for n in range(SongNumber):
            SC.addSong(len(songs[idx[n]]), song = songs[idx[n]])

        # load songs into RFC
        SC.loadSongs(RFCParams = RFCParams, loadingParams = loadingParams)    


        # set noise lvl
        HFCParams['SigToNoise'] = 0.25
            
        # run HFC with patterns        
        SC.run(patterns = SC.patterns, nLayers = 1, pattRepRange = [meanSongLength, meanSongLength+1], maxPauseLength = maxPauseLength, HFCParams = HFCParams)

        # measure classification error
        performance = SC.H.checkPerformance()
        meanPerformance = np.mean(performance[-1,:])
        variances = np.std(SC.H.gammaColl[-1], axis=0)
        meanVariance = np.mean(variances)
        
        seedMeanVariances.append(meanVariance)
        seedMeanPerformances.append(meanPerformance)        
        print(meanPerformance)
        print(meanVariance)
    
    json.dump({'gamma': gamma, 'n': networkSize, 'performance': np.mean(seedMeanPerformances), 'variance': np.mean(seedMeanVariances)}, open(targetDir + '/results.json','w'))

if __name__ == "__main__":
    main(sys.argv)
