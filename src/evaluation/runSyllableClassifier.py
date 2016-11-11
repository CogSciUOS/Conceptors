""" Libraries """

from matplotlib.pyplot import *
import os
import argparse
import sys

"""
this weird section of code allows modules in the parent directory to be imported here
it's the only way to do it in a way that allows you to run the file from other directories
and still have it work properly
"""
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import syllableClassifier as sC
import preprocessing

import numpy as np
import random

import evaluation.mongolog as logger
logger.configure('new')

# set random seeds for both numpy and random
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

logger.write_val("starting with seed", SEED)

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=np.ComplexWarning)

""" Function """

def runSyllClass(path, syllN, trainN, cvalRuns, sampRate, interpolType, mfccN, invCoeffOrder, winsize, melFramesN,
        smoothL, polyOrder, incDer, resN, specRad, biasScale, inpScale, conn, gammaPos, gammaNeg, plotExample,
        snr):
    """
    Function that runs syllable classification in a supervised manner using positive, negative and combined
    conceptors.
    """
    logger.write_frame_info(inspect.currentframe())
    path = os.path.abspath(path)

    """ assign parameters """

    prepParams = {
        'sample_rate': sampRate,
        'ds_type': interpolType,
        'mel_channels': mfccN,
        'inv_coefforder': invCoeffOrder,
        'winsize': winsize,
        'frames': melFramesN,
        'smooth_length': smoothL,
        'inc_der': incDer,
        'poly_order': polyOrder,
        'snr': snr
    }

    clearnParams = {
        'neurons': resN,
        'spectral_radius': specRad,
        'bias_scale': biasScale,
        'inp_scale': inpScale,
        'conn': conn}

    logger.write_val('prep params', prepParams)
    logger.write_val('conceptor params', clearnParams)

    performances = []
    evidences = []

    for i in range(cvalRuns):

        logger.write_val('start cross validation run', i)
        syllClass = sC.syllableClassifier(**clearnParams)

        samples = []
        n_test = np.random.random_integers(10, 50, syllN)
        # n_test = [15, 15, 26, 46, 41, 43, 48, 19, 16, 28, 14, 18, 10, 35, 13, 10, 10, 29, 22, 30, 15, 13, 50, 15, 11, 29, 30, 23, 14, 45]

        logger.write_arr("n_test", n_test)

        """ Get and preprocess data """
        logger.write('start preprocessing')
        data = preprocessing.preprocess(path, syllN, trainN, n_test, **prepParams)
        logger.write('end preprocessing')

        logger.write('start learning')
        # reinitialize syllable classifier
        syllClass.cLearning(trainN, data['train_data'], gammaPos, gammaNeg)
        logger.write('end learning')

        logger.write('start testing')
        results = syllClass.cTest(data['test_data'])
        logger.write('end testing')

        logger.write_arr("results", results)

        evidences.append(results['evidences'])
        performances.append(results['class_perf'])

    cval_results = np.array(performances)

    """ Plotting """
    if plotExample:
        plot_results(data, cval_results, evidences, cvalRuns)

    return cval_results * 100. #returns the results of the conceptors in percentage


def plot_results(data, cval_results, evidences, cvalRuns):

    sylls = figure(figsize=(15, 18))
    syllables = [0, 1]
    for syllable_i, syllable in enumerate(syllables):
        subplot(3, len(syllables), syllable_i + 1)
        # utteranceDataRaw = syllClass.trainDataDS[syllable][0][0]
        utteranceDataRaw = data['train_data_downsample'][syllable][0][0]
        plot(utteranceDataRaw)
        xlim(0, 9000)
        ylim(-18000, 18000)
        xlabel('t in ms/10')
        ylabel('amplitude')
        subplot(3, len(syllables), syllable_i + 1 + len(syllables))
        # utteranceDataMel = syllClass.trainDataMel[syllable - 1][0]
        utteranceDataMel = data['train_data_mel'][syllable - 1][0]
        for channel in range(utteranceDataMel.shape[1]):
            plot(utteranceDataMel[:, channel])
        xlim(0, 60)
        ylim(0, 120)
        xlabel('timeframes')
        ylabel('mfcc value')
        subplot(3, len(syllables), syllable_i + 1 + 2 * len(syllables))
        utteranceData = data['train_data'][syllable - 1][0]
        for channel in range(utteranceData.shape[1]):
            plot(utteranceData[:, channel])
        xlim(0, 3)
        ylim(0, 1)
        xlabel('timeframes')
        ylabel('mfcc value')
    tight_layout()
    sylls.savefig('exampleData.png')

    # syllable evidences

    h_pos = evidences[cvalRuns-1][0]
    h_neg = evidences[cvalRuns-1][1]
    h_comb = evidences[cvalRuns-1][2]

    evs = figure(figsize=(15, 15))
    suptitle('A', fontsize=20, fontweight='bold', horizontalalignment='left')
    subplot(1, 3, 1)
    imshow(h_pos, origin='lower', extent=[0, h_pos.shape[1], 0, h_pos.shape[0]], aspect='auto',
           interpolation='none', cmap='Greys')
    xlabel('Syllable #')
    ylabel('Test Trial #')
    title('Positive')
    subplot(1, 3, 2)
    imshow(h_neg, origin='lower', extent=[0, h_pos.shape[1], 0, h_pos.shape[0]], aspect='auto',
           interpolation='none', cmap='Greys')
    xlabel('Syllable #')
    ylabel('Test trial #')
    title('Negative')
    subplot(1, 3, 3)
    imshow(h_comb, origin='lower', extent=[0, h_pos.shape[1], 0, h_pos.shape[0]], aspect='auto',
           interpolation='none', cmap='Greys')
    xlabel('Syllable #')
    ylabel('Test Trial #')
    title('Combined')
    colorbar()
    tight_layout(rect=(0.1, 0.1, 0.9, 0.9))
    evs.savefig('Evidences.png')

    # classification performances

    perfData = cval_results * 100.
    width = 0.35
    fig, ax = subplots(figsize=(10, 8))

    rects1 = ax.bar(width, np.mean(perfData[:, 0]),
                    width,
                    color='MediumSlateBlue',
                    yerr=np.std(perfData[:, 0]),
                    error_kw={'ecolor': 'Black',
                              'linewidth': 3})

    rects2 = ax.bar(3 * width, np.mean(perfData[:, 1]),
                    width,
                    color='Tomato',
                    yerr=np.std(perfData[:, 1]),
                    error_kw={'ecolor': 'Black',
                              'linewidth': 3})

    rects3 = ax.bar(5 * width, np.mean(perfData[:, 2]),
                    width,
                    color='MediumSlateBlue',
                    yerr=np.std(perfData[:, 2]),
                    error_kw={'ecolor': 'Black',
                              'linewidth': 3})

    axes = gca()
    axes.set_ylim([0, 100])
    fig.suptitle('B', fontsize=20, fontweight='bold', horizontalalignment='left')
    ax.set_ylabel('Classification Performance')
    ax.set_xticks(np.arange(perfData.shape[1]) * 2 * width + 1.5 * width)
    ax.set_xticklabels(('Positive', 'Negative', 'Combined'))

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 0.9 * height,
                    '%d' % int(height),
                    ha='center',  # vertical alignment
                    va='bottom'  # horizontal alignment
                    )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    tight_layout(rect=(0.1, 0.1, 0.9, 0.9))
    fig.savefig('classPerfs.png')

    show()


""" argument parser """

parser = argparse.ArgumentParser(
    description='Passes arguments on to syllable Classifier function'
)
parser.add_argument(
    '--path',
    default='../../data/birddb/syll',
    type=str,
    help='directory to the folder that includes syllable folders with wave data'
)
parser.add_argument(
    '--syllN',
    type=int,
    default=2,
    help='number of syllables to include in train/test data'
)
parser.add_argument(
    '--trainN',
    default=30,
    type=int,
    help='number of training samples to use for each syllable (default = 30)'
)
parser.add_argument(
    '--cvalRuns',
    # default=2,
    default=1,
    type=int,
    help='Number of cross validation runs with different training/test data splits (default = 1)'
)
parser.add_argument(
    '--sampRate',
    default=20000,
    type=int,
    help='Sampling Rate that raw data will be downsampled to (default = 20000)'
)
parser.add_argument(
    '--interpolType',
    default='mean',
    type=str,
    help='type of interpolation to be used for downsampling.'
)
parser.add_argument(
    '--mfccN',
    # default=20,
    default=25,
    type=int,
    help='Number of mel frequency cepstral coefficients to extract for each mel frame (default = 25, which is the maximum possible)'
)
parser.add_argument(
    '--invCoeffOrder',
    default=False,
    help='Boolean, if true: Extract last n mfcc instead of first n (default = False)'
)
parser.add_argument(
    '--winsize',
    default=20,
    type=int,
    help='Size of the time-window to be used for mfcc extraction in ms (default = 20)'
)
parser.add_argument(
    '--melFramesN',
    default=64,
    type=int,
    help='Desired number of time bins for mfcc data (default = 64)'
)
parser.add_argument(
    '--smoothL',
    # default=5,
    default=5,
    type=int,
    help='Desired length of the smoothed mfcc data (default = 4)'
)
parser.add_argument(
    '--polyOrder',
    default=3,
    type=int,
    help='Order of the polynomial used for mfcc data smoothing (default = 3)'
)
parser.add_argument(
    '--incDer',
    default=[True, True],
    type=list,
    help='List of 2 booleans indicating whether to include 1./2. derivative of mfcc data or not (default = [True,True])'
)
parser.add_argument(
    '--resN',
    # default=20,
    default=10,
    type=int,
    help='Size of the reservoir to be used for conceptor learning (default = 10)'
)
parser.add_argument(
    '--specRad',
    default=1.1,
    # default=1.2,
    type=float,
    help='Spectral radius of the connectivity matrix of the reservoir (default = 1.2)'
)
parser.add_argument(
    '--biasScale',
    default=0.5,
    type=float,
    help='Scaling of the bias term to be introduced to each reservoir element (default = 0.2)'
)
parser.add_argument(
    '--inpScale',
    default=1.0,
    type=float,
    help='Scaling of the input of the reservoir (default = 1.0)'
)
parser.add_argument(
    '--conn',
    default=1.0,
    type=float,
    help='Downscaling of the reservoir connections (default = 1.0)'
)
parser.add_argument(
    '--gammaPos',
    default=25,
    type=int,
    help='Aperture to be used for computation of the positive conceptors'
)
parser.add_argument(
    '--gammaNeg',
    default=20,
    type=int,
    help='Aperture to be used for computation of the negative conceptors'
)
parser.add_argument(
    '--plotExample',
    default=False,
    help='If true, plot raw & preprocessed mfcc data as well as conceptor evidences (default = False)'
)
parser.add_argument(
    '--targetDir',
    default=None,
    type=str,
    help='Subdirectory in which results are to be stored'
)
parser.add_argument(
    '-syllNames',
    default=['as','bl','ck','dm','el'],
    type=list,
    help='List of names of syllables to be used'
)


""" Run script via command window """
# can be also run using an IDE, but uses the default parameters then
try:
    args = parser.parse_args()
except:
    sys.exit(0)

perf = []

#noiseRange = [4, 2, 1, 0.5, 0.25, 0.125, 0.0]
snrRange = [0.0]
#numSyllRange = np.arange(20, 31, 5).tolist()
numSyllRange = [2]

perf_points = np.empty([3, len(snrRange) * len(numSyllRange)])

for snr in snrRange:
    for numSyll in numSyllRange:
        cval_perc = 0
        perf_val = 0
        # try:
        cval_perc = runSyllClass(path=args.path, syllN=numSyll, trainN=args.trainN, cvalRuns=args.cvalRuns,
            sampRate=args.sampRate, interpolType=args.interpolType, mfccN=args.mfccN,
            invCoeffOrder=args.invCoeffOrder, winsize=args.winsize, melFramesN=args.melFramesN,
            smoothL=args.smoothL, polyOrder=args.polyOrder, incDer=args.incDer, resN=args.resN,
            specRad=args.specRad, biasScale=args.biasScale, inpScale=args.inpScale, conn=args.conn,
            gammaPos=args.gammaPos, gammaNeg=args.gammaNeg, plotExample=args.plotExample, snr=snr)
        print(cval_perc)
        perf_val = np.mean(cval_perc, axis=0)[2]
        # except Exception as err:
        #     print(str(snr) + ' and ' + str(numSyll) + ' have not been working...')
        #     print(err)

        perf.append(perf_val)
        print(perf_val)

print(perf)
