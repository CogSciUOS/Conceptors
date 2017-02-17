"""
This class is used to run the SongClassifier with randomly generated
syllable sequences.
"""

from songClassifier import *
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#%%

# create list of syllables and initialize SongClassifier with it
syllables = ['aa','ao','ba','bm','ca','ck','da','dl','ea','ej','fa','ff','ha','hk']
SC = SongClassifier(syllables)

# define parameters of songClassifier
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
             'drift': 0.01,
             'gammaRate': 0.005,
             'dcsv': 4,
             'SigToNoise': float('inf')}

plotBeliefs = True

#%%

# create random songs and load them into a RFC
s1_length = 3
s2_length = 5
s3_length = 4
SC.addSong(s1_length)
SC.addSong(s2_length)
SC.addSong(s3_length)
SC.loadSongs(
        useSyllRecog = False, SyllPath = '../data/birddb/syll/', RFCParams = RFCParams,
        loadingParams = loadingParams, cLearningParams = cLearningParams
        )

# run song classification and plot gammas
SC.run(pattRepRange = (5,15), maxPauseLength = 3, nLayers = 1, useSyllRecog = False, SyllPath = '../data/birddb/syll/',
       dataPrepParams = dataPrepParams, cLearningParams = cLearningParams, HFCParams = HFCParams)
if plotBeliefs:
    SC.H.plot_gamma(songLenghts = [len(s) for s in SC.Songs])