"""
Class for the syllable classification
"""

import os.path
import numpy as np
import reservoirConceptor as c
import functions as fct
import random


class syllableClassifier:
    __slots__ = (
        'res',      # reservoir
        'size',     # reservoir size
        'c_pos',    # positive conceptors
        'c_neg'     # negative conceptors
    )

    def __init__(self, neurons, spectral_radius, bias_scale, inp_scale, conn):
        self.size = neurons
        self.res = c.Reservoir(N=neurons, NetSR=spectral_radius, bias_scale=bias_scale, inp_scale=inp_scale, conn=conn)
        self.c_pos = []
        self.c_neg = []

    def cLearning(self, n_train, train_data, gamma_pos=25, gamma_neg=27):
        """ Function that learns positive and negative conceptors on data with the following steps:
        1. create Reservoir
        2. Feed each sample of each syllable in reservoir and collect its states
        3. Use states to compute positive conceptor
        4. Use Conceptor logic to compute negative conceptor

        :param gamma_pos: aperture of the positive conceptors
        :param gamma_neg: aperture of the negative conceptors
        :param data: list of syllables with sample data
        """

        for syllable in np.array(train_data):
            R_syll = np.zeros((syllable.shape[1] * (self.size + syllable.shape[2]), syllable.shape[0]))

            for i, sample in enumerate(syllable):
                self.res.run([sample], t_learn=len(sample), t_wash=0, load=False)
                states = np.concatenate((np.squeeze(self.res.TrainArgs.T), sample), axis=1)
                R_syll[:, i] = np.reshape(states, states.shape[0] * states.shape[1])

            R = np.dot(R_syll, R_syll.T) / n_train
            C_tmp = np.dot(R, np.linalg.inv(R + np.eye(len(R))))
            self.c_pos.append(C_tmp)

        for i in range(len(self.c_pos)):
            C = np.zeros_like(self.c_pos[0])
            for j in list(range(0, i)) + list(range(i + 1, len(self.c_pos))):
                C = fct.OR(C, self.c_pos[j])
            self.c_neg.append(C)

        for i in range(len(self.c_pos)):
            self.c_pos[i] = fct.phi(self.c_pos[i], gamma_pos)
            self.c_neg[i] = fct.phi(self.c_neg[i], gamma_neg)

    def cTest(self, test_data, pattern=None):
        """ Function that uses trained conceptors to recognize syllables in data by going through the following steps:
        1. Feed each sample of each syllable into reservoir and collect its states
        2. Analyize similarity of collected states and trained conceptors
        3. Choose syllable, for which similarity is highest

        :param pattern: the list of syllables that make up the pattern of the song

        :returns evidences: list of arrays of evidences with rows = trials and columns = syllables
                            for positive, negative and combined conceptors
        :returns class_perf: Mean classification performance on test data set for
                             positive, negative and combined conceptors
        """

        h_pos = []
        h_neg = []
        h_comb = []
        class_pos = []
        class_neg = []
        class_comb = []

        if pattern is not None:
            test_data = np.array([test_data[np.argmax(syll)] for syll in pattern])

        for syll_i, syllable in enumerate(test_data):

            for sample in syllable:

                self.res.run([sample], t_learn=sample.shape[0], t_wash=0, load=False)
                states = np.concatenate((np.squeeze(self.res.TrainArgs).T, sample), axis=1)
                z = np.reshape(states, states.shape[0] * states.shape[1])

                h_pos_tmp = np.zeros(len(self.c_pos))
                h_neg_tmp = np.zeros(len(self.c_pos))
                h_comb_tmp = np.zeros(len(self.c_pos))

                for k in range(len(self.c_pos)):
                    h_pos_tmp[k] = np.dot(np.dot(z.T, self.c_pos[k]), z)
                    h_neg_tmp[k] = np.dot(np.dot(z.T, self.c_neg[k]), z)

                h_pos_tmp = h_pos_tmp - np.min(h_pos_tmp)
                h_pos_tmp = h_pos_tmp / np.max(h_pos_tmp)
                h_neg_tmp = h_neg_tmp - np.min(h_neg_tmp)
                h_neg_tmp = 1 - h_neg_tmp / np.max(h_neg_tmp)
                h_comb_tmp = (h_pos_tmp + h_neg_tmp) / 2.0
                h_pos.append(h_pos_tmp/np.sum(h_pos_tmp))
                h_neg.append(h_neg_tmp/np.sum(h_neg_tmp))
                h_comb.append(h_comb_tmp/np.sum(h_comb_tmp))

                dec_pos = np.where(h_pos_tmp == np.max(h_pos_tmp))[0][0]
                dec_neg = np.where(h_neg_tmp == np.max(h_neg_tmp))[0][0]
                dec_comb = np.where(h_comb_tmp == np.max(h_comb_tmp))[0][0]

                classification_pos_tmp = 1 if dec_pos == syll_i else 0
                classification_neg_tmp = 1 if dec_neg == syll_i else 0
                classification_comb_tmp = 1 if dec_comb == syll_i else 0

                class_pos.append(classification_pos_tmp)
                class_neg.append(classification_neg_tmp)
                class_comb.append(classification_comb_tmp)

        h_pos = np.array(h_pos)
        h_neg = np.array(h_neg)
        h_comb = np.array(h_comb)
        class_pos = np.array(class_pos)
        class_neg = np.array(class_neg)
        class_comb = np.array(class_comb)

        return {
            'evidences': [h_pos, h_neg, h_comb],
            'class_perf': [np.mean(class_pos), np.mean(class_neg), np.mean(class_comb)]
        }
