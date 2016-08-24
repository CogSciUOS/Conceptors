# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:28:51 2016

@author: Richard Gast
"""

import os.path
from rfReservoirConceptor import *
from hierarchicalConceptor import *
from syllableClassifier import *
import itertools
from functions import checkRecall

#%%

class SongClassifier:

    def __init__(self, syllables):
        """
        Initializes SongClassifier with a list of syllables

        :param syllables: List including all possible syllables to draw from later on
        """

        self.Sylls = syllables
        self.nSylls = len(self.Sylls)
        self.Songs = []

    def addSong(self, nSongSylls = 5, sequenceReps = 1, song = None):
        """
        Function that generates a deterministic song from the syllable list

        :param nSongSylls: Number of syllables the song should consist of
        :param sequenceReps: How often we randomly draw nSongSylls syllables from the nSongSylls
                             target syllables in order to create song (default = 1)
        :param song: List of strings representing syllables can be passed here, if None
                     a random sequence is created (default = None)

        :returns song: list of syllables that is appended to self.Songs
        """

        # check whether complete song was passed to method
        if song is not None:
            if type(song) != list: raise ValueError('Song has to be a list of strings representing syllables')
            self.Songs.append(song)
        else:
            # generate random sequence of syllables
            songSylls = [self.Sylls[i] for i in np.random.choice(range(len(self.Sylls)), nSongSylls, replace=False)]

            # append random sequences of nSongSylls syllables drawn from songSylls and append them to song list
            self.Songs.append(list(itertools.chain.from_iterable([[songSylls[i] for i in np.random.choice(range(len(songSylls)), nSongSylls, replace = True)] for j in range(sequenceReps)])))

    def loadSongs(self, t_learn = 400, t_cadapt = 2000, t_wash = 200, t_recall = 200, RFCParams = {}, loadingParams = {}):
                 #N = 400, K = 2000, alpha = 8, NetSR = 1.5, bias_scale = 0.2, inp_scale = 1.5,t_learn = 400, t_cadapt = 2000, t_wash = 200, TyA_wout = 1., TyA_wload = 0.1,
                 #gradient_c = True, gradient_window = 1, c_adapt_rate = 0.5, gradient_cut = 2.0, t_recall = 200):
        """
        Function that loads all songs stored in the SongClassifier instance in a RFC

        """

        # create clean training pattern for each song
        self.patterns = []
        reps = t_learn + t_cadapt + t_wash
        usedSylls = np.zeros(self.nSylls)

        for song in self.Songs:

            song_tmp = np.array([np.array(self.Sylls) == s for s in song]) * 1.
            self.patterns.append(np.tile(song_tmp, [round(reps/len(song_tmp)) + 1, 1]))
            usedSylls += (np.sum(self.patterns[-1], axis = 0) > 0) * 1

        # delete colums of unused syllables
        usedSylls = usedSylls != 0
        new_patts = []
        for p in self.patterns:
            new_patts.append(p[:,usedSylls])
        self.patterns = new_patts
        self.Sylls = list(np.array(self.Sylls)[usedSylls])
        self.nSylls = len(self.Sylls)
        
        # display syllables and songs in use        
        print('Final set of syllables used: ', self.Sylls)
        for i, s in enumerate(self.Songs):
            print('Song ',i,': ', s)
        
        # try loading patterns into RFC until each pattern can be recalled correctly
        success = False
        print('Loading songs into RFC...')
        while not success:
            self.R = RF_Reservoir(**RFCParams)
            self.R.load(self.patterns, t_learn = t_learn, t_cadapt = t_cadapt, t_wash = t_wash, **loadingParams)
            self.R.recall(t_recall = t_recall)
            recallError = checkRecall(self.patterns, self.R.Y_recalls)
            print('Mean recall error of each pattern (in range [0, 1]): ', recallError)
            if np.sum(recallError) == 0:
                print('Songs succesfully loaded into RFC.')
                success = True
            else:
                print('Loading failed for at least one song. Next try...')

    def run(self, patterns = None, nLayers = 3, pattRepRange = (2,20), maxPauseLength = 10, useSyllRecog = False, SyllPath = None,
            nTrain = 30, cType = 2, dataPrepParams = {}, cLearningParams = {}, HFCParams = {}):
            #sigma = 0.99, drift = 0.01, gammaRate = 0.002, dcsv = 8, SigToNoise = 0.5,
            #gammaPos = 25, gammaNeg = 20, cType = 2):
        """
        :Description: Function that uses an HFC to recognize which of the songs loaded in self.R is currently
                     used as input to the HFC.

        :param patterns:        list with entries for each song, consisting of an m by n array,
                                with m = number of syllables the pattern is played and n = number of
                                syllables of the Classifier (if None, stored patterns are used)
        :param nLayers:         Number of layers the HFC should consist of (default = 3)
        :param pattRepRange:    tuple including the lower and upper bound of the uniform distribution
                                from which the number of repetitions of each song are drawn
        :param maxPauseLength:  Maximum number of 'zero' syllables to be added after a song ended (default = 10)
        :param useSyllRecog:    If True, train a syllableClassifier on all syllables stored in the songClassifier
                                and run classification on the stored patterns afterwards. The resulting evidences
                                will then be used to run the HFC
        :param SyllPath:        If useSyllRecog = True, this needs to be the full path to the folder including
                                the syllable data
        """

        if patterns is not None: self.patterns = patterns

        # generate repetition times for each song from a uniform distribution of range pattRepRange
        pattTimesteps = [np.random.randint(low = pattRepRange[0], high = pattRepRange[1]) * len(self.Songs[i]) for i in range(len(self.patterns))]

        # putt all patterns into syllable recognizer, if syllable recognition is to be done
        if useSyllRecog:
            print('Running syllable recognition...')
            path = os.path.dirname(os.path.abspath(__file__)) if SyllPath is None else SyllPath

            # generate sequence of syllables from patterns to use syllableClassifier on
            self.syllClassPatts = np.zeros((1,self.nSylls))
            for i,t in enumerate(pattTimesteps):
                patt = self.patterns[i][0:len(self.Songs[i]),:]
                self.syllClassPatts = np.append(self.syllClassPatts, np.tile(patt, [t,1]), axis = 0)
            self.syllClassPatts = self.syllClassPatts[1:,:]

            # initialize syllableClassifier and train it on the stored syllables contained in songs
            self.SyllClass = syllableClassifier(path)
            songs = []
            for s in self.Songs:
                songs += s
            songs = set(songs)
            self.SyllClass.prep_data(self.nSylls, nTrain, np.ones(len(songs)), syll_names = songs, **dataPrepParams)
            self.SyllClass.cLearning(**cLearningParams)

            # run classification on syllClassPatts and store the evidences for each syllable in appropriate format for HFC
            self.SyllClass.cTest(pattern = self.syllClassPatts)
            evidences = self.SyllClass.evidences[cType]
            t_all = 0
            self.patterns = []
            for i,t in enumerate(pattTimesteps):

                patt = np.zeros((1,self.nSylls))
                for j in range(round(t/len(self.Songs[i]))):
                    pause_length = np.random.randint(maxPauseLength)
                    patt_tmp = np.concatenate((evidences[t_all + j*len(self.Songs[i]) : t_all + (j+1)*len(self.Songs[i]),:], np.zeros((pause_length,self.nSylls))), axis = 0)
                    patt = np.vstack((patt, patt_tmp))
                    pattTimesteps[i] += pause_length
                patt = patt[1:,:]
                self.patterns.append(patt)
                t_all += t*len(self.Songs[i])

        # initialize and run HFC with patterns
        print('Driving HFC with syllable sequences...')
        self.H = Hierarchical(self.R, nLayers)
        self.H.run(self.patterns, pattTimesteps = pattTimesteps, plotRange = pattTimesteps, **HFCParams)
        print('Done!')
