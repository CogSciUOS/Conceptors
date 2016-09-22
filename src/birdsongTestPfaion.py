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

# define parameters of songClassifier
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

plotBeliefs = True

#%%

# create random songs and load them into a RFC
SC = SongClassifier(syllables)
s1_length = 3
s2_length = 5
s3_length = 4
SC.addSong(s1_length)
SC.addSong(s2_length)
SC.addSong(s3_length)
SC.loadSongs(useSyllRecog = False, SyllPath = '../data/birddb/syll/', RFCParams = RFCParams, loadingParams = loadingParams, cLearningParams = cLearningParams)

# run song classification and plot gammas
SC.run(pattRepRange = (5,15), maxPauseLength = 3, nLayers = 1, useSyllRecog = False, SyllPath = '../data/birddb/syll/',
       dataPrepParams = dataPrepParams, cLearningParams = cLearningParams, HFCParams = HFCParams)
#if plotBeliefs:
#    SC.H.plot_gamma(songLenghts = [len(s) for s in SC.Songs])


import matplotlib.gridspec as gridspec



def f(self, songLenghts = None):

    t_all = np.sum(self.pattTimesteps)
    xspace = np.arange(t_all)

    # make a figure for every HFC level
    for l in range(self.M):
        figure()
        gs = gridspec.GridSpec(2, 1, hspace = 0, height_ratios = [1,6])

        subplot(gs[1])
        # plot gamma and play-area for all patterns
        for p_idx, p in enumerate(self.patterns):

            # calculate start and stop idxs for this song in whole classification
            start_idx = np.sum(self.pattTimesteps[0:p_idx], dtype = np.int)
            end_idx = np.sum(self.pattTimesteps[0:p_idx + 1], dtype = np.int)

            # plot gamma values for this song
            gamma_plot = plot(xspace, self.gammaColl[l, p_idx, :].T,
                label = 'Gamma of pattern {}'.format(p_idx))

            # show areas where the song was played
            pattern_not_empty = p.any(axis = 1)
            fill_between(np.arange(start_idx, end_idx), 0, 1,
                where = pattern_not_empty,
                facecolor = gamma_plot[0].get_color(),
                alpha = 0.2,
                label = 'Pattern {}'.format(p_idx))

            # plot lines after every single song iteration
            if songLenghts:
                for i in range(start_idx, end_idx):
                    if i % songLenghts[p_idx] == 0:
                        axvline(i, color = 'black', alpha = 0.2)

        # plot desriptions
        xlabel('timesteps')
        ylabel('Gamma')
        suptitle('Gamma lvl {}'.format(l))

        # place legend to bottom with dynamic offset depending on number of patterns
        legend_offset = 0.15 + self.n_patts * 0.05
        gcf().subplots_adjust(bottom=legend_offset)
        legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0.0, ncol=2)


    subplot(gs[0])
    gca().matshow(self.class_predictions[l].reshape(1, t_all), aspect='auto')
    gca().get_xaxis().set_ticks([])
    gca().get_yaxis().set_ticks([])
    ylabel('Predicted\nSong', rotation = 'horizontal', ha = 'right', va = 'center')


    show()

f(SC.H, songLenghts = [len(s) for s in SC.Songs])