# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 14:32:30 2016

@author: Richard Gast
"""

import numpy as np
from scipy.spatial.distance import cdist
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
             'inp_scale': 1.4}
             
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
        
nSongs = 4
nRuns = 50
nComb = np.sum(np.arange(1,nSongs))

#%% learn song conceptors and compare their similarity to the similarity of the actual songs
xCorr = np.zeros(nRuns*nComb)
xEucl = np.zeros_like(xCorr)
yCorr = np.zeros_like(xCorr)
yEucl = np.zeros_like(xCorr)
idx = np.arange(0,len(songs))

for r in range(nRuns):
    
    #%% learn song conceptors
    
    while True:
        
        SC = SongClassifier(syllables, verbose = True)
        
        np.random.shuffle(idx)
        for n in range(nSongs):
            SC.addSong(len(songs[idx[n]]), song = songs[idx[n]])
        
        try:
            SC.loadSongs(useSyllRecog = True, SyllPath ='D:/Data/Projects/StudyProject/syll',RFCParams = RFCParams, loadingParams = loadingParams, cLearningParams = cLearningParams, dataPrepParams = dataPrepParams)
            break
        except:
            print('loading songs failed, next try...')
            
    #%% generate sequences of syllables according to songs 
    
    nFeatures = dataPrepParams['mel_channels'] * (1 + np.sum(dataPrepParams['inc_der']))
    nTimesteps = dataPrepParams['smooth_length']
    songVecs = []
    for song in SC.Songs:
        song_tmp = np.array([np.array(SC.Sylls) == s for s in song]) * 1.
        fullsong = np.zeros((nFeatures, len(song)*nTimesteps))
        for j,syll in enumerate(song_tmp):
            samples = np.array(SC.SyllClassData['train_data'][np.argmax(syll)])
            fullsong[:,j*nTimesteps:(j+1)*nTimesteps] = np.mean(samples,axis=0).T
        songVecs.append(fullsong)
        
    #%% compare similarity between original songs with similarity between song conceptors
    songCorr = np.zeros((nSongs,nSongs))
    songEucl = np.zeros_like(songCorr)
    conceptorCorr = np.zeros_like(songCorr)
    conceptorEucl = np.zeros_like(songCorr)
    for i in range(nSongs):
        for j in range(nSongs):
            
            song1 = songVecs[i]
            song2 = songVecs[j]
            lenDiff = song1.shape[1] - song2.shape[1]
            if lenDiff < 0:
                song1 = np.concatenate((song1,np.zeros((song1.shape[0],-lenDiff))), axis = 1)
            elif lenDiff > 0:
                song2 = np.concatenate((song2,np.zeros((song2.shape[0],lenDiff))), axis = 1)
            songCorr[i,j] = np.mean(np.diag(cdist(song1.T,song2.T,metric='correlation')))
            songEucl[i,j] = np.mean(np.diag(cdist(song1.T,song2.T,metric='euclidean')))
            
            conceptor1 = np.reshape(SC.R.C[i],(1,len(SC.R.C[i])))
            conceptor2 = np.reshape(SC.R.C[j],(1,len(SC.R.C[j])))
            conceptorCorr[i,j] = np.mean(np.diag(cdist(conceptor1,conceptor2,metric='correlation')))
            conceptorEucl[i,j] = np.mean(np.diag(cdist(conceptor1,conceptor2,metric='euclidean')))
            
    xCorr_tmp = np.triu(songCorr,k=1)
    xEucl_tmp = np.triu(songEucl,k=1)
    yCorr_tmp = np.triu(conceptorCorr,k=1)
    yEucl_tmp = np.triu(conceptorEucl,k=1)
    
    xCorr[r*nComb:(r+1)*nComb] = xCorr_tmp[xCorr_tmp != 0]
    yCorr[r*nComb:(r+1)*nComb] = yCorr_tmp[yCorr_tmp != 0]
    xEucl[r*nComb:(r+1)*nComb] = xEucl_tmp[xEucl_tmp != 0]
    yEucl[r*nComb:(r+1)*nComb] = yEucl_tmp[yEucl_tmp != 0]
    
    print('run ',r,' finished.')
    
print('Correlation between correlational distance of songs and conceptors:', np.min(np.corrcoef(xCorr,y=yCorr)))
print('Correlation between euclidean distance of songs and conceptors:', np.min(np.corrcoef(xEucl,y=yEucl)))

#%% Plotting

figure()
scatter(xCorr,yCorr)
xlabel('Similarity of Songs')
ylabel('Similarity of Conceptors')
title('Correlation between SongSimilarity and ConceptorSimilarity')

figure()
scatter(xEucl,yEucl)
xlabel('Similarity of Songs')
ylabel('Similarity of Conceptors')
title('Correlation between Euclidean Distance of Songs and Conceptors')
