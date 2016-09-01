# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:28:51 2016

@author: Richard Gast
"""

import os.path
from rfReservoirConceptor import *
from hierarchicalConceptor import *
from syllableClassifier import *
from preprocessing import preprocess
import itertools
from functions import checkRecall

#%%

class SongClassifier:

    def __init__(self, syllables, verbose = True):
        """
        :Description: Initializes SongClassifier with a list of syllables

        :Parameters:
            1. syllables:    List including all possible syllables to draw from later on
            2. verbose:      if False, all print outputs will be supressed
        """

        self.Sylls = syllables
        self.nSylls = len(self.Sylls)
        self.Songs = []
        self.verbose = verbose

    def addSong(self, nSongSylls, sequenceReps = 1, song = None):
        """
        :Description: Function that generates a deterministic song from the syllable list
        
        :Parameters:
            1. nSongSylls:     Number of syllables the song should consist of
            2. sequenceReps:   How often we randomly draw nSongSylls syllables from the nSongSylls
                               target syllables in order to create song (default = 1)
            3. song:           List of strings representing syllables can be passed here, if None
                               a random sequence is created (default = None)

        :Returns:
            1. song:           list of syllables that is appended to self.Songs
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

    def loadSongs(self, useSyllRecog = False, SyllPath = None, t_learn = 400, t_cadapt = 2000, t_wash = 200, t_recall = 200, 
                  RFCParams = {}, loadingParams = {}, dataPrepParams = {}, cLearningParams = {}):
        """
        :Description: Function that loads all songs stored in the SongClassifier instance in a RFC
        
        :Parameters:
            1.  useSyllRecog:    If True, run syllable recognition on stored patterns and use raw output for RFC training (default = False)
            2.  SyllPath:        Needs to be directory to syllable data if useSyllRecog is True (default = None)
            3.  t_learn:         number of timesteps used for feeding each pattern into the reservoir (default = 400)
            4.  t_cadapt:        number of timesteps used for letting the autoconceptor adapt for each pattern (default = 2000)
            5.  t_wash:          number of timesteps used for running the reservoir with input before learning period starts (default = 200)
            6.  t_recall:        number of timesteps used for recalling each pattern by applying the conceptor on the reservoir (default = 200)
            7.  RFCParams:       dictionary of keyword arguments to initilaze the RFC class with (defaults of RFC class will be used if not specified)
            8.  loadingParams:   dictionary of keyword arguments to load the patterns into the RFC with (defaults of RFC class will be used if not specified)
            9.  dataPrepParams:  dictionary of keyword arguments for data preprocessing if syllable recognition is to be used (defaults of preprocessing function will be used if not specified)
            10. cLearningParams: dictionary of keyword arguments to learn a conceptor for each syllable if syllable recognition is to be used (defaults of syllable classifier will be used if not specified)

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
        if self.verbose: 
            print('Final set of syllables used: ', self.Sylls)
            for i, s in enumerate(self.Songs):
                print('Song ',i,': ', s)
        
        # if syllable recognition is to be used, train syllable recognizer on syllables in songs and drive it with self.patterns
        self.syllableConceptorsLearned = False
        if useSyllRecog:
            patternsRFC = self.runSyllableClassification(SyllPath, useStoredPatts = True, maxPauseLength = 1, dataPrepParams = dataPrepParams, cLearningParams = cLearningParams)
        else:
            patternsRFC = self.patterns

        # try loading patterns into RFC until each pattern can be recalled correctly
        success = False
        if self.verbose: print('Loading songs into RFC...')
        while not success:
            self.R = RF_Reservoir(**RFCParams)
            self.R.load(patternsRFC, t_learn = t_learn, t_cadapt = t_cadapt, t_wash = t_wash, **loadingParams)
            self.R.recall(t_recall = t_recall)
            recallError = checkRecall(self.patterns, self.R.Y_recalls)
            if self.verbose: print('Mean recall error of each pattern (in range [0, 1]): ', recallError)
            if np.sum(recallError) == 0:
                if self.verbose: print('Songs succesfully loaded into RFC.')
                success = True
            elif self.verbose:
                print('Loading failed for at least one song. Next try...')
    
    def runSyllableClassification(self, SyllPath = None, nTrain = 50, nTest = 10, cType = 2, useStoredPatts = True, 
                                  useRawOutput = True, pattTimesteps = None, maxPauseLength = 3, dataPrepParams = {}, cLearningParams = {}):
        """
        :Description: Function that learns conceptors for each syllable in self.Songs and
                      tries to classify the sequence of syllable generated from repeating the songs several times
        
        :Parameters:
            1. SyllPath:            If useSyllRecog = True, this needs to be the full directory to the folder including the syllable data (default = None)
            2. nTrain:              number of training samples to be used for each syllable (default = 50)
            3. nTest:               number of test samples to be used for each syllable (default = 10)
            4. cType:               index that indicates from which conceptor to use the recognition results {0 = pos, 1 = neg, 2 = combined} (default = 2)
            5. useStoredPatts:      if True, run syllable classification on self.patterns, else create new sequence according to repetition times in pattTimesteps (default = True)              
            6. useRawOutput:        if True, store evidences from chosen conceptore as patterns. If False, apply winner-takes-it-all classification on evidences (default = True)       
            7. pattTimesteps:       list of scalars representing the lengths each song in self.Songs should be presented at test time (only necessary if useStoredPatts is False)
            8. maxPauseLength:      Maximal length of pauses to be added randomly after each song (default = 3)            
            8. dataPrepParams:      dictionary of keyword arguments for data preprocessing if syllable recognition is to be used (defaults of preprocessing function will be used if not specified)
            9. cLearningParams:     dictionary of keyword arguments to learn a conceptor for each syllable if syllable recognition is to be used (defaults of syllable classifier will be used if not specified)
        
        :Returns: 
            1. newPatts:            List of patterns with recognition evidences for each syllable played
        """
        
        if self.verbose: print('Running syllable recognition...')
        path = os.path.dirname(os.path.abspath(__file__)) if SyllPath is None else os.path.abspath(SyllPath)
        self.path = path
        
        # generate sequence of syllables from patterns to use syllableClassifier on
        syllClassPatts = np.zeros((1,self.nSylls))
        # either use stored patterns
        if useStoredPatts:
            pattTimesteps = []
            for p in self.patterns:
                syllClassPatts = np.append(syllClassPatts, p, axis = 0)
                pattTimesteps.append(len(p))
        # or create sequences of lengths according to entries in pattTimesteps
        else:
            for i,t in enumerate(pattTimesteps):
                patt = self.patterns[i][0:len(self.Songs[i]),:]
                syllClassPatts = np.append(syllClassPatts, np.tile(patt, [round(t/len(self.Songs[i])),1]), axis = 0)
        syllClassPatts = syllClassPatts[1:,:]
        
        # if conceptors for syllables have not been learned already, learn them 
        if not self.syllableConceptorsLearned:
            # get list with unique syllables and create preprocessed  training and test data
            songs = []
            for s in self.Songs:
                songs += s
            songs = set(songs)
            self.SyllClassData = preprocess(path, self.nSylls, nTrain, np.ones(len(songs)) * nTest, syll_names = songs, **dataPrepParams)
            # initialize syllableClassifier and train it on training data
            self.SyllClass = syllableClassifier(
                cLearningParams['neurons'],
                cLearningParams['spectral_radius'],
                cLearningParams['bias_scale'],
                cLearningParams['inp_scale'],
                cLearningParams['conn'])
            self.SyllClass.cLearning(nTrain, self.SyllClassData['train_data'], cLearningParams['gammaPos'], cLearningParams['gammaNeg'])
            self.syllableConceptorsLearned = True            

        # run classification on syllClassPatts and store the evidences for each presented syllable
        results = self.SyllClass.cTest(self.SyllClassData['test_data'], pattern = syllClassPatts)
        evidences = results['evidences'][cType]
        if not useRawOutput:
            evidences_tmp = np.zeros_like(evidences)
            for syll in range(evidences.shape[0]):
                evidences_tmp[syll,np.argmax(evidences[syll,:])] = 1
            evidences = evidences_tmp
        
        # create list with entries for each pattern and store the respective evidences in those entries
        t_all = 0
        newPatts = []
        for i,t in enumerate(pattTimesteps):
            patt = np.zeros((1,self.nSylls))
            for j in range(round(t/len(self.Songs[i]))):
                pause_length = np.random.randint(maxPauseLength)
                patt_tmp = np.concatenate((evidences[t_all + j*len(self.Songs[i]) : t_all + (j+1)*len(self.Songs[i]),:], np.zeros((pause_length,self.nSylls))), axis = 0)
                patt = np.vstack((patt, patt_tmp))
                pattTimesteps[i] += pause_length
            patt = patt[1:,:]
            newPatts.append(patt)
            t_all += (j+1)*len(self.Songs[i])
        
        return newPatts
        
        
    def run(self, patterns = None, nLayers = 3, pattRepRange = (2,20), maxPauseLength = 3, useSyllRecog = False, SyllPath = None,
            dataPrepParams = {}, cLearningParams = {}, HFCParams = {}):
        """
        :Description: Function that uses an HFC to recognize which of the songs loaded in self.R is currently
                      used as input to the HFC.
        
        :Parameters: 
            1. patterns:        list with entries for each song, consisting of an m by n array,
                                with m = number of syllables the pattern is played and n = number of
                                syllables of the Classifier. If None, stored patterns are used (default = None)
            2. nLayers:         Number of layers the HFC should consist of (default = 3)
            3. pattRepRange:    tuple including the lower and upper bound of the uniform distribution
                                from which the number of repetitions of each song are drawn (default = (2,20))
            4. maxPauseLength:  Maximum number of 'zero' syllables to be added after a song ended (default = 3)
            5. useSyllRecog:    If True, train a syllableClassifier on all syllables stored in the songClassifier
                                and run classification on the stored patterns afterwards. The resulting evidences
                                will then be used to run the HFC (default = False)
            6. SyllPath:        If useSyllRecog = True, this needs to be the full path to the folder including
                                the syllable data (default = None)
        """

        if patterns is not None: self.patterns = patterns

        # generate repetition times for each song from a uniform distribution of range pattRepRange
        pattTimesteps = [np.random.randint(low = pattRepRange[0], high = pattRepRange[1]) * len(self.Songs[i]) for i in range(len(self.patterns))]

        # putt all patterns into syllable recognizer, if syllable recognition is to be done
        if useSyllRecog: 
            patternsHFC = self.runSyllableClassification(SyllPath, useStoredPatts = False, useRawOutput = self.syllableConceptorsLearned, maxPauseLength = maxPauseLength, pattTimesteps = pattTimesteps, dataPrepParams = dataPrepParams, cLearningParams = cLearningParams )
        else:      
            patternsHFC = [p[0:pattTimesteps[i]] for i,p in enumerate(self.patterns)]
            
        # initialize and run HFC with patterns
        if self.verbose: print('Driving HFC with syllable sequences...')
        self.H = Hierarchical(self.R, nLayers)
        self.H.run(patternsHFC, pattTimesteps = pattTimesteps, plotRange = pattTimesteps, **HFCParams)
        if self.verbose: print('Done!')