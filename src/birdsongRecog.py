"""
This class is used to run the SongClassifier with randomly generated
syllable sequences.
"""

from songClassifier import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#%%

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
HFCParams = {'sigma': 0.82,
             'drift': 0.01,
             'gammaRate': 0.005,
             'dcsv': 4,
             'SigToNoise': float('inf')}

# include plots
plotLoadedSongs = True
plotBeliefs = True

#%%

# create random songs and load them into a RFC
s1 = ['aa', 'ao', 'ba', 'bm', 'ca']
s2 = ['aa', 'ao', 'ck', 'da', 'dl', 'ea']
SC.addSong(song = s1)
SC.addSong(song = s2)

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
SC.run(pattRepRange = (10,20), maxPauseLength = 3, nLayers = 2, useSyllRecog = False, SyllPath = '../data/birddb/syll/',
       dataPrepParams = dataPrepParams, cLearningParams = cLearningParams, HFCParams = HFCParams)
if plotBeliefs:
    SC.H.plot_gamma()