import numpy as np
import random as r


def crossValAperture(cval_runs, n_train, n_syllables, model, gamma_pos, gamma_neg, **kwargs):
    """ Function that performs cross validation on model for different positive and negative apertures.

    :param cval_runs: Number of cross validation runs with different training/test sample splits for each set of apertures
    :param n_train: Number of training samples
    :param n_syllables: Number of syllables from which to load samples
    :param model: Model to perform cross validation on (type = syllableClassifier)
    :param gamma_pos: Aperture for positive conceptors (vector)
    :param gamma_neg: Aperture for negative conceptors (vector)
    :param kwargs: keyworded arguments for data preprocessing and cLearning (will be passed to crossVal)

    :returns performances: List with 3 arrays (pos, neg, comb) of mean classification performance for each set of apertures.
                           Rows = entries in gamma_pos, Columns = entries in gamma_neg
    """

    performances_pos = np.zeros((len(gamma_pos), len(gamma_neg)))
    performances_neg = np.zeros_like(performances_pos)
    performances_comb = np.zeros_like(performances_pos)

    for i, g_pos in enumerate(gamma_pos):

        for j, g_neg in enumerate(gamma_neg):
            performances = crossVal(cval_runs, n_train, n_syllables, model, g_pos, g_neg, **kwargs)
            performances_pos[i, j] = np.mean(performances[:, 0])
            performances_neg[i, j] = np.mean(performances[:, 1])
            performances_comb[i, j] = np.mean(performances[:, 2])

    return [performances_pos, performances_neg, performances_comb]


def crossVal(cval_runs, n_train, n_syllables, model, gamma_pos, gamma_neg, prepParams={}, clearnParams={}):
    """ Function that performs cross validation on model.

    :param cval_runs: Number of cross validation runs with different training/test sample splits
    :param n_train: Number of training samples
    :param n_syllables: Number of syllables from which to load samples
    :param model: Model to perform cross validation on (type = syllableClassifier)
    :param gamma_pos: Aperture for positive conceptors (scalar)
    :param gamma_neg: Aperture for negative conceptors (scalar)
    :param prepParams: Parameters to pass on to the preprocessing (default = {})
    :param clearnParams: Parameters to pass on to the conceptor learning (defualt = {})

    :returns performances: Array with mean classification performance of model on test data set
    """

    performances = []

    for i in range(cval_runs):

        samples = []
        n_test = np.random.random_integers(10, 50, n_syllables)

        for j in range(n_syllables):
            indices = np.arange(0, n_train + n_test[j], 1)
            ind_tmp = indices.copy().tolist()
            r.shuffle(ind_tmp)
            ind_tmp = np.array(ind_tmp)

            samples.append(ind_tmp)

        if cval_runs > 1:
            model.prepData(n_syllables, n_train, n_test, samples=samples, **prepParams)
        else:
            model.prepData(n_syllables, n_train, n_test, **prepParams)
        model.cLearning(gamma_pos, gamma_neg, **clearnParams)
        model.cTest()
        performances.append(model.class_perf)

    return np.array(performances)