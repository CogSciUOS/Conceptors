# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:15:55 2016

@author: asus
"""

import numpy as np
import numpy.polynomial.polynomial as poly
import scipy as sp
import scipy.interpolate
import scipy.signal as ss
import math
import sklearn.decomposition as sd
import os
import scipy.io.wavfile as wav
#import sys
#sys.path.append('C:/Program Files/Python/python_speech_features-master')
#sys.path.append('/Users/apple1/Dropbox/Conceptors/Task1_Recognition/python_speech_features-master')
from features import mfcc

#%%

def load_data(syllable, N, used_samples, sample_order = None):
    """Function that goes through all N samples of syllable and loads its wave data.
    
    :param syllable: complete path name of syllable (string)
    :param N: number of samples to load
    :param used_samples: number of samples to skip in the beginning
    :param sample_order: if not None should be vector of indices of samples to be loaded (default = None)
    
    :returns syllable_waves: list of N sample waves of syllable
    """
        
    samples = [files for files in os.listdir(syllable)]
    syllable_waves = []
    if sample_order is None:
        for i in range(int(N)):
            rate, wave = wav.read(syllable + '/' + samples[i + used_samples])
            syllable_waves.append([wave,rate])
    else:
        for i in sample_order:
            rate, wave = wav.read(syllable + '/' + samples[i])
            if wave.size == 0: print(i, samples[i])
            syllable_waves.append([wave,rate])
    return syllable_waves

#%%

""" Zero Pad Data """

def zeroPad(data):
    """ Function that searches for array in data with maximum length
        and adds zeros to the end of each other array to make them the same length.
    
    :param data: list of arrays of different lengths
    
    :returns syllables: list with same number of entries as data, but with zero
                        padded arrays
    """
    
    max_length = 0
    syllables = []
    
    for syll in data:
        for samp in syll:
            max_length = len(samp[0]) if len(samp[0]) > max_length else max_length
            
    for syllable in data:
        samples = []
        for sample in syllable:
            sample_tmp = np.zeros(max_length)
            sample_tmp[0:len(sample[0])] = sample[0]
            samples.append([sample_tmp,sample[1]])
        syllables.append(samples)
        
    return syllables

#%%

def downSample(data, sampleRate = 20000, dsType = 'mean'):
    """ Function that downsamples data.
    
    :param data: list including syllables with sample data
    :param sampleRate: desired samplerate
    :param dsType: Type of interpolating used for downsampling.
                   Can be mean or IIR, which uses an order 8 Chebyshev type 1 filter (default = mean)
    
    :returns syllables: downsampled data, in same format as input data
    """
    
    syllables = []
    for syllable in data:
        samples = []
        for sample in syllable:
            SR = np.round(sample[1]/float(sampleRate))
            if dsType == 'mean':
                pad_size = math.ceil(float(sample[0].size)/SR)*SR - sample[0].size
                s_padded = np.append(sample[0], np.zeros(pad_size)*np.NaN)
                s_new = sp.nanmean(s_padded.reshape(-1,SR), axis=1)
            elif dsType == 'FIR':
                s_new = ss.decimate(sample[0],SR)
            samples.append([s_new, sampleRate])
        syllables.append(samples)
    return syllables
                
#%%

""" get MEL Frequencies """

def getMEL(data, n_mfcc = 12, invCoeffOrder = False, winsize = 20, frames = 64): 
    """ Function that goes through all samples of each syllable and extracts the
        mfccs for the 12 mel frequency channels.
        
    :param data: list of syllables with sample data
    :param n_mfcc: number of mel frequency cepstral coefficients to return (default = 12)
    :param invCoeffOrder: if True, extract last n mfcc instead of first n (default = False)
    :param wisize: size of the time window used for mfcc extraction
    :param frames: desired number of time frames in final mfcc data
    
    :returns syllables: list with mfccs for n_mfcc mel channels for each sample of each syllable
    """
    
    syllables = []
    i = 0
    for syllable in data:
        samples = []
        for sample in syllable:
            W = winsize/1000. * sample[1]
            winstep = (np.round(1 + (len(sample[0]) - W) / (frames - 1))) / float(sample[1])
            i += 1
            if invCoeffOrder:
                samples.append(mfcc(sample[0], samplerate = sample[1], winlen = winsize/1000., winstep = winstep, numcep = n_mfcc)[:,-n_mfcc::])
            else:
                samples.append(mfcc(sample[0], samplerate = sample[1], winlen = winsize/1000., winstep = winstep, numcep = n_mfcc + 1)[:,1::])
        syllables.append(samples)
    return syllables

#%%

def getShiftsAndScales(data):
    """ Function that extracts shift and scale from data.
    
    :params data: list of syllables with sample data
    
    :returns shifts: negative minimum mfcc for each of the 12 channels (vector of length 12)
    :returns scales: 1 / (maximum mfcc - minimum mfcc) for each of the 12 channels (vector of length 12)
    """
    
    allData = []
    for syllable in data:
        for sample in syllable:
            allData.extend(sample)
    allData = np.array(allData)
    maxVals = allData.max(axis = 0)
    minVals = allData.min(axis = 0)
    shifts = -minVals
    scales = 1.0/(maxVals - minVals)
    return shifts, scales

def normalizeData(data, shifts, scales):
    """ Function that normalizes data with shifts and scales.
    
    :param data: list of syllables with sample data
    :param shifts: negative minimum mfcc for each of the 12 mel channels (vector of length 12)
    :param scales: 1 / (maximum mfcc - minimum mfcc) for each of the 12 channels (vector of length 12)
    
    :returns newData: list of normalized data
    """
    
    newData = []
    for syllable_i, syllable in enumerate(data):
        newData.append([])
        for sample_i, sample in enumerate(syllable):
            sample += np.tile(shifts, (sample.shape[0], 1))
            sample = np.dot(sample, np.diag(scales))
            newData[syllable_i].append(sample)
    return newData

#%%

def smoothenData(data, smoothLength, polyOrder, channelsN):
    """ Function that smooths the data with polynomial and downsamples it.
    
    :param data: list of syllables with samples
    :param smoothLength: sampling points to downsample data to
    :param polyOrder: Order of the polynomial to smooth data with
    :param channelsN: Number of mel frequency channels in data    
    
    :returns newData: smoothend and downsampled data
    """

    newData = []
    for syllable in data:
        newSyllable = []
        for sample in syllable:
            newSample = np.zeros((smoothLength, channelsN))
            size = sample.shape[0]
            xVals = np.arange(1, size + 1)
            interpolCoords = np.linspace(1, size, smoothLength) 
            polycoeff = poly.polyfit(list(range(size)), sample, polyOrder)
            sampleSmooth = poly.polyval(list(range(size)), polycoeff)
            for channel_i in range(channelsN):
                f = sp.interpolate.interp1d(xVals, sampleSmooth[channel_i])
                newSample[:, channel_i] = f(interpolCoords)
            newSyllable.append(newSample)
        newData.append(newSyllable)
    return newData
    
#%%
    
def runPCA(data, n):
    """ Function that runs PCA on data and reduces data to first n components.
    
    :param data: list of syllables with mfcc sample data
    :param n: Number of principal components to use / dimensions to reduce data to (default = 10)
    
    :returns dataRed: Dimensionality reduced list
    """

    if n > data[0][0].shape[1]:
        raise ValueError('Number of PCs to use exceeds dimensionality of the data')
        
    dataRed = []
    
    for i, syll in enumerate(data):     
        train = np.array([])
        for samp in syll:
            if len(train) == 0:
                train = samp
            else:
                train = np.concatenate((train,samp), axis = 0)
                
        #R = np.dot(train.T, train)
        #eigvals, eigvecs = np.linalg.eig(R)
        #pcs = eigvecs * eigvals
        #ind = np.squeeze(np.fliplr(np.array(np.argsort(eigvals), ndmin = 2)))
        #print(ind)
        #print(eigvals)
        #comps = np.squeeze(pcs[:,ind])
        model = sd.PCA(n_components = n)
        results = model.fit(train)
        comps = results.components_.T
        dataRed_tmp = np.dot(train, comps)
        nT = syll[0].shape[0]
        dataRed_tmp2 = np.zeros((len(syll), nT, n))
        for j in range(nT):
            dataRed_tmp2[:,j,:] = dataRed_tmp[j*len(syll):(j+1)*len(syll),:]
        dataRed.append(dataRed_tmp2)
    return dataRed
            
#%%
            
def mfccDerivates(data, Der1 = True, Der2 = True):
    """ Function that adds subsequent pairwise differences in mfcc data over time and their change over time.
    
    :param data: List of syllables including samples with mfcc arrays
    :param Dev1: If true, add first derivate of mfcc (default = True)
    :param Dev2: If true, add second derivate of mfcc (default = True)
    
    :return devdata: MFCC data including derivatives
    """
    
    if not Der1:
        return data
    if Der1:
        devdata = []
        for syll in data:
            samples = []
            for samp in syll:
                if not Der2:
                    newData = np.zeros((samp.shape[0], samp.shape[1] * 2))
                    for i in range(samp.shape[0]):
                        newData[i,0:samp.shape[1]] = samp[i,:]
                        if (i+1) < samp.shape[0]:
                            newData[i+1,samp.shape[1]:2*samp.shape[1]] = samp[i+1,:] - samp[i,:]
                else:
                    newData = np.zeros((samp.shape[0], samp.shape[1] * 3))
                    for i in range(samp.shape[0]):
                        newData[i,0:samp.shape[1]] = samp[i,:]
                        if (i+1) < samp.shape[0]:
                            newData[i+1,samp.shape[1]:2*samp.shape[1]] = samp[i+1,:] - samp[i,:]
                        if i > 0:
                            newData[i,2*samp.shape[1]:3*samp.shape[1]] = newData[i,samp.shape[1]:2*samp.shape[1]] - newData[i-1,samp.shape[1]:2*samp.shape[1]]
                samples.append(newData)
            devdata.append(samples)
    return devdata
                