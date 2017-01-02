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




def plot_gamma(self, songLenghts = None):
    t_all = np.sum(self.pattTimesteps)
    xspace = np.arange(t_all)

    # make a figure for every HFC level
    for l in range(self.M):

        # first: calculate and plot wrong predictions in the background
        predict = SC.H.class_predictions[l]
        original = np.array([idx for idx, steps in enumerate(SC.H.pattTimesteps) for _ in range(steps)])
        mismatch = predict != original

        for i, m in enumerate(mismatch):
            if m:
                fill_between([i, i+1], 0, 1,
                    facecolor = 'gray',
                    linewidth = 0,
                )

        fill_between([0], 0, 0, facecolor = 'gray', linewidth = 0, label = 'Mismatch')

        # plot gamma and play-area for all patterns
        for p_idx, p in enumerate(self.patterns):

            # calculate start and stop idxs for this song in whole classification
            start_idx = np.sum(self.pattTimesteps[0:p_idx], dtype = np.int)
            end_idx = np.sum(self.pattTimesteps[0:p_idx + 1], dtype = np.int)

            # plot gamma values for this song
            gamma_plot = plot(xspace, self.gammaColl[l, p_idx, :].T,
                label = 'Gamma for song {}'.format(p_idx))

            # show areas where the song was played
            pattern_not_empty = p.any(axis = 1)
            # adjust range of song to one more step to have connected areas
            if p_idx < len(self.patterns) - 1:
                end_idx += 1
                pattern_not_empty = np.append(pattern_not_empty, True)
            fill_between(np.arange(start_idx, end_idx), -0.05, 0,
                where = pattern_not_empty,
                facecolor = gamma_plot[0].get_color(),
                label = 'Song {}'.format(p_idx),
                linewidth = 0,
                )

        # set y axis for gamma levels
        ylim(-0.05, 1)
        xlim(0, xspace[-1])
        yticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        ylabel('Gamma')

        # make legend (must be here, because the data belongs to standard y-axis)
        legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', borderaxespad=0.0, ncol=2)

        # tight layout removes layouting issues with the twinx y-axis
        tight_layout()

        # create dynamic offset for the legend depending on number of patterns
        legend_offset = 0.2 + self.n_patts * 0.05
        gcf().subplots_adjust(bottom=legend_offset)

    # show all figures for all hfc levels
    xticks([])
    xticks([])
    gca().axis('off')
    savefig('song_gamma.png')
    show()

figure(figsize=(10,6))
plot_gamma(SC.H, songLenghts = [len(s) for s in SC.Songs])
