# -*- coding: utf-8 -*-
import numpy as np
from songClassifier import *
from matplotlib.pyplot import *
import warnings
import random
import json
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#%% Parameters


def main(args):
    # network parameters

    netsr = float(args[1])
    bias_scale = float(args[2])
    inp_scale = float(args[3])
    targetDir = args[4]

    RFCParams = {'N': 600,
                 'K': 2400,
                 'NetSR': netsr,
                 'bias_scale': bias_scale,
                 'inp_scale': inp_scale}
                 
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
    nRuns = 3

    seeds = [23, 42, 1337]

    #%% run Model

    meanSongLength = nTestSongs/nSongs
    performance = np.zeros(nRuns)
    variance = np.zeros(nRuns)
    idx = np.arange(0,nSongs)
    failed = 0
    while i < nRuns:
        
        seed = seeds[i]

        np.random.seed(seed)
        random.seed(seed)

        SC = SongClassifier(syllables, verbose = True)
        
        np.random.shuffle(idx)
        for n in range(nSongs):
            SC.addSong(len(songs[idx[n]]), song = songs[idx[n]])
        
        try:
            SC.loadSongs(useSyllRecog = True, SyllPath ='../../Data/',RFCParams = RFCParams, loadingParams = loadingParams, cLearningParams = cLearningParams, dataPrepParams = dataPrepParams)
        except:
            seed[i] += 1
            failed += 1
            continue

        SC.run(patterns = SC.patterns, nLayers = 1, pattRepRange = [meanSongLength, meanSongLength+1], maxPauseLength = maxPauseLength, HFCParams = HFCParams, cLearningParams = cLearningParams, dataPrepParams = dataPrepParams)
        
        p = SC.H.checkPerformance()
        performance[i] = np.mean(p[-1,:])
        variances = np.std(SC.H.gammaColl[-1], axis=0)
        variance[i] = np.mean(variances)
        print(i+1,'. run of ',nRuns,' runs is finished.')
        i += 1

    json.dump({'netSR': netsr, 'bias_scale': bias_scale, 'inp_scale': inp_scale, 'performance': np.mean(performance), 'variance': np.mean(variance), 'loadFailed': failed}, open(targetDir + '/results.json','w'))

if __name__ == "__main__":
    main(sys.argv)
        
        
        
