"""
This class is used to run the SongClassifier with randomly generated
syllable sequences.
"""

from songClassifier import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# create list of syllables and initialize SongClassifier with it
syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej']
SC = SongClassifier(syllables)

# define parameters of songClassifier
RFCParams = {'N': 400,
             'K': 2000,
             'NetSR': 1.5,
             'bias_scale': 1.2,
             'inp_scale': 1.5}
loadingParams = {'gradient_c': True}
dataPrepParams = {}
cLearningParams = {}
HFCParams = {'sigma': 0.02,
             'drift': 0.05,
             'gammaRate': 0.004,
             'dcsv': 2,
             'SigToNoise': float('inf')}

# include plots
plotLoadedSongs = True
plotBeliefs = True

#%%

# create random songs and load them into a RFC

s1_length = 3
s2_length = 5
s3_length = 4

SC.addSong(s1_length)
SC.addSong(s2_length)
#SC.addSong(s3_length)
SC.loadSongs(RFCParams = RFCParams, loadingParams = loadingParams)

# plot RFC recall
if plotLoadedSongs:
    plotrange = 50
    figure()
    for s in range(len(SC.Songs)):
        recall = np.argmax(SC.R.Y_recalls[s][0:plotrange,:], axis = 1)
        target = np.argmax(SC.patterns[s][0:plotrange,:], axis = 1)
        subplot(len(SC.Songs),1,s+1)
        plot(recall, 'r')
        plot(target, 'b')
        ylim([0,SC.nSylls - 1])
        ylabel(' Syllable #')
        xlabel('t')
    show()

# run song classification and plot gammas
SC.run(pattRepRange = (10,20), maxPauseLength = 10, nLayers = 2, useSyllRecog = True, SyllPath = '../data/birddb/syll/',
       dataPrepParams = dataPrepParams, cLearningParams = cLearningParams, HFCParams = HFCParams)
if plotBeliefs:
    SC.H.plot_gamma()