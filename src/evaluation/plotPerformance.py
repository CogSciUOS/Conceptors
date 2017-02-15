import argparse
import random

import numpy.random
from matplotlib.pyplot import *

from evaluation.runSyllableClassifier import runSyllClass, log_results

# set random seeds for both numpy and random
SEED = 100
np.random.seed(SEED)
random.seed(SEED)

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
    default=10,
    help='number of syllables to include in train/test data'
)
parser.add_argument(
    '--trainN',
    default=50,
    type=int,
    help='number of training samples to use for each syllable (default = 30)'
)
parser.add_argument(
    '--cvalRuns',
    default=2,
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
    default=20,
    type=int,
    help='Number of mel frequency cepstral coefficients to extract for each mel frame (default = 25, which is the maximum possible)'
)
parser.add_argument(
    '--invCoeffOrder',
    default=True,
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
    default=4,
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
    default=10,
    type=int,
    help='Size of the reservoir to be used for conceptor learning (default = 10)'
)
parser.add_argument(
    '--specRad',
    default=0.9,
    type=float,
    help='Spectral radius of the connectivity matrix of the reservoir (default = 1.2)'
)
parser.add_argument(
    '--biasScale',
    default=0.4,
    type=float,
    help='Scaling of the bias term to be introduced to each reservoir element (default = 0.2)'
)
parser.add_argument(
    '--inpScale',
    default=0.3,
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
    default=27,
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
    default=None,
    type=list,
    help='List of names of syllables to be used'
)
parser.add_argument(
    '--snr',
    type=float,
    default=0.0,
    help='signal to noise ratio in the syllable data'
)

parser.add_argument(
    '--trial',
    type=int,
    default=0,
    help='the number of the trial that is used in documenting the results'
)

parser.add_argument(
    '--logPath',
    type=str,
    default='',
    help='the directory the logfile should be written to'
)


""" Run script via command window """
# can be also run using an IDE, but uses the default parameters then
try:
    args = parser.parse_args()
except:
    sys.exit(0)

syll_numbers = np.array([5, 10, 20, 25, 30, 35, 40, 45, 50, 55])
#syll_numbers = np.array([5, 10, 15])
snrs = np.array([0.0])

meanPerformance = np.zeros((len(syll_numbers), len(snrs)))
k = 1

for i,syll_num in enumerate(syll_numbers):
    for j,snr in enumerate(snrs):
        try:
            args.snr = snr
            args.syllN = syll_num
            perf_val = 0.0
            cval_perc = runSyllClass(path=args.path, syllN=args.syllN, trainN=args.trainN, cvalRuns=args.cvalRuns,
                                     sampRate=args.sampRate, interpolType=args.interpolType, mfccN=args.mfccN,
                                     invCoeffOrder=args.invCoeffOrder, winsize=args.winsize, melFramesN=args.melFramesN,
                                     smoothL=args.smoothL, polyOrder=args.polyOrder, incDer=args.incDer, resN=args.resN,
                                     specRad=args.specRad, biasScale=args.biasScale, inpScale=args.inpScale,
                                     conn=args.conn,gammaPos=args.gammaPos, gammaNeg=args.gammaNeg,
                                     plotExample=args.plotExample, snr=args.snr, syllNames=args.syllNames)
            perf_val = np.mean(cval_perc, axis=0)[2]
            meanPerformance[i,j] = perf_val
            print('Run',k,'of',len(syll_numbers)*len(snrs),' (SyllNum: ',syll_num,')finished with mean performance: ', perf_val)
            log_results(args.logPath, args, perf_val, k) #logging values for later use
            k += 1
        except Exception:
            log_results(args.logPath, args, perf_val, k, error = sys.exc_info()[0])
            print('Run', k, 'of', len(syll_numbers) * len(snrs), 'failed to finish with: ', perf_val)
            k += 1

# Plotting the results
matshow(meanPerformance, cmap = 'viridis', vmin = 0, vmax = 1, interpolation = 'nearest')
colorbar()
title('Mean syllable classification performance')
xlabel('Signal-to-Noise Ratio')
xticks(np.arange(0,len(snrs)), snrs)
ylabel('Number of Syllables')
yticks(np.arange(0,len(syll_numbers)), syll_numbers)

show()
