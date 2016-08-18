"""
Class for the syllable classification
"""

import os.path
import numpy as np
import reservoirConceptor as c
import functions as fct
# import preprocessing as prep


# %%

class syllableClassifier:
    def __init__(self, fname):
        """ Class that performs supervised learning on syllable data in order to perform classification.

        :param fname: Complete path to folder which includes folders for each syllable which include folders for each sample which include wave data
        """

        self.folder = fname


    def cLearning(self, gamma_pos=25, gamma_neg=27, N=10, SR=1.2, bias_scale=1.0, inp_scale=0.2, conn=1):
        """ Function that learns positive and negative conceptors on data with the following steps:
        1. create Reservoir
        2. Feed each sample of each syllable in reservoir and collect its states
        3. Use states to compute positive conceptor
        4. Use Conceptor logic to compute negative conceptor

        :param gamma_pos: aperture of the positive conceptors
        :param gamma_neg: aperture of the negative conceptors
        :param data: list of syllables with sample data
        :param N: size of the reservoir
        :param SR: spectral radius of the reservoir
        :param bias_scale: scaling of the bias while running reservoir
        :param inp_scale: scaling of the input when fed into the reservoir
        :param conn: scaling of the amount of connectivity within the reservoir

        :returns C_pos: List of positive conceptors
        :returns C_neg: List of negative conceptors
        """

        self.res = c.Reservoir(N=N, NetSR=SR, bias_scale=bias_scale, inp_scale=0.2, conn=conn)
        self.C_pos = []

        for syllable in np.array(self.trainDataFinal):

            R_syll = np.zeros((syllable.shape[1] * (N + syllable.shape[2]), syllable.shape[0]))

            for i, sample in enumerate(syllable):
                self.res.run([sample], t_learn=len(sample), t_wash=0, load=False)
                states = np.concatenate((np.squeeze(self.res.TrainArgs.T), sample), axis=1)
                R_syll[:, i] = np.reshape(states, states.shape[0] * states.shape[1])

            R = np.dot(R_syll, R_syll.T) / self.n_train
            C_tmp = np.dot(R, np.linalg.inv(R + np.eye(len(R))))
            self.C_pos.append(C_tmp)

        self.C_neg = []

        for i in range(len(self.C_pos)):
            C = np.zeros_like(self.C_pos[0])
            for j in list(range(0, i)) + list(range(i + 1, len(self.C_pos))):
                C = fct.OR(C, self.C_pos[j])
            self.C_neg.append(C)

        for i in range(len(self.C_pos)):
            self.C_pos[i] = fct.phi(self.C_pos[i], gamma_pos)
            self.C_neg[i] = fct.phi(self.C_neg[i], gamma_neg)

    def cTest(self, pattern=None):
        """ Function that uses trained conceptors to recognize syllables in data by going through the following steps:
        1. Feed each sample of each syllable into reservoir and collect its states
        2. Analyize similarity of collected states and trained conceptors
        3. Choose syllable, for which similarity is highest

        :param data: list of syllables with sample data (different from training data)
        :param C_pos: list of trained positive Conceptors
        :param C_neg: list of trained negative Conceptors

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
            self.testData = np.array([self.testDataFinal[np.argmax(syll)] for syll in pattern])
        else:
            self.testData = self.testDataFinal

        for syll_i, syllable in enumerate(self.testData):

            for sample in syllable:

                self.res.run([sample], t_learn=sample.shape[0], t_wash=0, load=False)
                states = np.concatenate((np.squeeze(self.res.TrainArgs).T, sample), axis=1)
                z = np.reshape(states, states.shape[0] * states.shape[1])

                h_pos_tmp = np.zeros(len(self.C_pos))
                h_neg_tmp = np.zeros(len(self.C_pos))
                h_comb_tmp = np.zeros(len(self.C_pos))

                for k in range(len(self.C_pos)):
                    h_pos_tmp[k] = np.dot(np.dot(z.T, self.C_pos[k]), z)
                    h_neg_tmp[k] = np.dot(np.dot(z.T, self.C_neg[k]), z)

                h_pos_tmp = h_pos_tmp - np.min(h_pos_tmp)
                h_pos_tmp = h_pos_tmp / np.max(h_pos_tmp)
                h_neg_tmp = h_neg_tmp - np.min(h_neg_tmp)
                h_neg_tmp = 1 - h_neg_tmp / np.max(h_neg_tmp)
                h_comb_tmp = (h_pos_tmp + h_neg_tmp) / 2.0
                h_pos.append(h_pos_tmp)
                h_neg.append(h_neg_tmp)
                h_comb.append(h_comb_tmp)

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

        self.evidences = [h_pos, h_neg, h_comb]
        self.class_perf = [np.mean(class_pos), np.mean(class_neg), np.mean(class_comb)]
