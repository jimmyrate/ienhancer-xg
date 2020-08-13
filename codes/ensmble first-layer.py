


import numpy as np
from numpy import array
from pandas import DataFrame
import itertools
import time

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef

# input sequences.........................................................................
def getSequences(f):
    seqslst = []
    while True:
        s = f.readline()
        if not s:
            break
        else:

            seq = s.split('\n')[0]
            seq = seq.upper()
            seqslst.append(seq)
    return seqslst


# global cross validation:10-fold.........................................................
def globalCrossValidation(all_X, global_y, all_features, all_ensemble_types, classifier, folds):
    global_predicted_proba = -np.ones(len(global_y))
    global_predicted_label = -np.ones(len(global_y))
    all_optimal_feature_ensembles = []
    cv_round = 1
    for global_train_index, global_test_index in folds.split(X):
        print('..........................................................................')
        print('global cross validation, round %d, beginning' % cv_round)
        start = time.clock()

        predicted_proba_matrix, predicted_label_matrix, weights, global_y_train = getIndividualFeatureResult(all_X,
                                                                                                             global_y,
                                                                                                             global_train_index)
        optimal_ensemble, optimal_weight = getOptimalEnsemble(predicted_proba_matrix, predicted_label_matrix, weights,
                                                              global_y_train, all_features, all_ensemble_types)

        optimal_proba_matrix, optimal_label_matrix = [], []
        for number in optimal_ensemble:
            global_X = all_X[number]
            global_X_train, global_X_test, global_y_train = ConstructTrainAndTestDataset(global_X, global_y, number,
                                                                                         global_train_index,
                                                                                         global_test_index)
            temp_proba = (classifier.fit(global_X_train, global_y_train)).predict_proba(global_X_test)
            optimal_proba_matrix.append(temp_proba[:, 1])
            temp_label = (classifier.fit(global_X_train, global_y_train)).predict(global_X_test)
            optimal_label_matrix.append(temp_label)
        optimal_proba_matrix = array(optimal_proba_matrix).T
        optimal_label_matrix = array(optimal_label_matrix).T

        global_predicted_proba[global_test_index] = np.sum(optimal_proba_matrix * optimal_weight, 1)
        global_predicted_label[global_test_index] = np.sum(optimal_label_matrix * optimal_weight, 1) > 0.5

        for i in range(len(optimal_ensemble)):
            if i == 0:
                optimal_feature_ensemble_name = all_features[optimal_ensemble[i]]
            else:
                optimal_feature_ensemble_name += '+' + all_features[optimal_ensemble[i]]
        all_optimal_feature_ensembles.append(optimal_feature_ensemble_name)

        end = time.clock()
        print('round ' + str(cv_round) + ', optimal ensemble model: ' + optimal_feature_ensemble_name)
        print('round %d, running time: %.3f hour' % (cv_round, (end - start) / 3600))
        print('..........................................................................\n')
        cv_round += 1

    unique_optimal_feature_ensembles = {element: all_optimal_feature_ensembles.count(element) for element in
                                        all_optimal_feature_ensembles}

    fpr, tpr, thresholds = roc_curve(global_y, global_predicted_proba, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy = accuracy_score(global_y, global_predicted_label)
    sensitivity = recall_score(global_y, global_predicted_label)
    specificity = (accuracy * len(global_y) - sensitivity * sum(global_y)) / (len(global_y) - sum(global_y))
    MCC = matthews_corrcoef(global_y, global_predicted_label)
    return auc_score, accuracy, sensitivity, specificity, MCC, unique_optimal_feature_ensembles


def getIndividualFeatureResult(all_X, global_y, global_train_index):
    predicted_proba_matrix, predicted_label_matrix, weights = [], [], []
    global_y_train = global_y[global_train_index]

    folds = KFold(5, True, 1)
    clf = XGBClassifier(learning_rate=0.08, n_estimators=20, max_depth=7, objective='binary:logistic', metric='auc')
    for number in range(len(all_X)):
        global_X_train = all_X[number][global_train_index]
        auc_score, predicted_proba, predicted_label = InnerCrossValidation(global_X_train, global_y_train, number, clf,
                                                                           folds)
        predicted_proba_matrix.append(predicted_proba)
        predicted_label_matrix.append(predicted_label)
        weights.append(auc_score)
    predicted_proba_matrix = array(predicted_proba_matrix).T
    predicted_label_matrix = array(predicted_label_matrix).T
    weights = array(weights)

    return predicted_proba_matrix, predicted_label_matrix, weights, global_y_train


def InnerCrossValidation(X, y, number, clf, folds):
    predicted_proba = -np.ones(len(y))
    predicted_label = -np.ones(len(y))
    for train_index, test_index in folds.split(X):
        X_train, X_test, y_train = ConstructTrainAndTestDataset(X, y, number, train_index, test_index)
        temp_proba = (clf.fit(X_train, y_train)).predict_proba(X_test)
        predicted_proba[test_index] = temp_proba[:, 1]
        predicted_label[test_index] = (clf.fit(X_train, y_train)).predict(X_test)

    fpr, tpr, thresholds = roc_curve(y, predicted_proba, pos_label=1)
    auc_score = auc(fpr, tpr)

    return auc_score, predicted_proba, predicted_label


def ConstructTrainAndTestDataset(X, y, number, train_index, test_index):
    y_train = y[train_index]

    if number == 3:
        vdim = 35
        train_seqs = X[train_index]
        test_seqs = X[test_index]
        pssm = getPssmMatrix(train_seqs, y_train, vdim)
        X_train = getFeatureFromPssm(train_seqs, pssm, vdim)
        X_test = getFeatureFromPssm(test_seqs, pssm, vdim)
    else:
        X_train = X[train_index]
        X_test = X[test_index]

    return X_train, X_test, y_train


def getOptimalEnsemble(predicted_proba_matrix, predicted_label_matrix, weights, global_y_train, all_features,
                       all_ensemble_types):
    max_auc, max_acc, max_sen, max_spe, max_mcc = 0, 0, 0, 0, 0
    for feature_ensemble in all_ensemble_types:
        used_weights = weights[feature_ensemble]
        used_weithts = used_weights * 1.0 / sum(used_weights)
        used_predicted_proba = predicted_proba_matrix[:, feature_ensemble]
        used_predicted_label = predicted_label_matrix[:, feature_ensemble]
        wmean_predicted_proba = np.sum(used_predicted_proba * used_weithts, 1)
        wmean_predicted_label = np.sum(used_predicted_label * used_weithts, 1) > 0.5

        fpr, tpr, thresholds = roc_curve(global_y_train, wmean_predicted_proba, pos_label=1)
        auc_score = auc(fpr, tpr)
        acc = accuracy_score(global_y_train, wmean_predicted_label)
        sen = recall_score(global_y_train, wmean_predicted_label)
        spe = (acc * len(global_y_train) - sen * sum(global_y_train)) / (len(global_y_train) - sum(global_y_train))
        mcc = matthews_corrcoef(global_y_train, wmean_predicted_label)
        print('Inner cross validation on training dataset, results for feature subset:')
        print(list(array(all_features)[feature_ensemble]))
        print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f, mcc:%.3f****\n' % (
            auc_score, acc, sen, spe, mcc))

        if acc > max_acc:
            max_acc = acc
            max_auc = auc_score
            max_sen = sen
            max_spe = spe
            max_mcc = mcc
            optimal_ensemble = feature_ensemble
            optimal_weight = used_weithts
        elif acc == max_acc:
            if auc_score > max_auc:
                max_auc = auc_score
                max_sen = sen
                max_spe = spe
                max_mcc = mcc
                optimal_ensemble = feature_ensemble
                optimal_weight = used_weithts
            elif auc_score == max_auc:
                if sen + spe + mcc > max_sen + max_spe + max_mcc:
                    max_sen = sen
                    max_spe = spe
                    max_mcc = mcc
                    optimal_ensemble = feature_ensemble
                    optimal_weight = used_weithts


    return optimal_ensemble, optimal_weight


# getting pssm feature..............................................................................
def getPssmMatrix(train_seqs, y_train, vdim):
    alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    posi_train_seqs = train_seqs[list(y_train)]
    posi_train_seqs_num = len(posi_train_seqs)
    alphabet_num = len(alphabet)
    pssm = np.ones((alphabet_num, vdim)) * 10 ** (-10)
    for i in range(posi_train_seqs_num):
        seqlen = len(posi_train_seqs[i])
        for j in range(vdim):
            if j <= seqlen - 1:
                row_index = alphabet.get(posi_train_seqs[i][j])
                pssm[row_index, j] += 1
    pssm = np.log(pssm * alphabet_num / posi_train_seqs_num)
    return pssm


def getFeatureFromPssm(seqs, pssm, vdim):
    alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_num = len(seqs)
    features = np.zeros((seqs_num, vdim))
    for i in range(seqs_num):
        seqlen = len(seqs[i])
        for j in range(vdim):
            if j <= seqlen - 1:
                row_index = alphabet.get(seqs[i][j])
                features[i, j] = pssm[row_index, j]
    return features


########################################################################################

if __name__ == '__main__':

    # input sequences for pssm feature
    fz = open("enhancers3.fasta", 'r')
    enhancers = getSequences(fz)
    fx = open("non-enhancers3.fasta", 'r')
    nonenhancers = getSequences(fx)
    instances = array(enhancers + nonenhancers)
    global_y = array([1] * len(enhancers) + [0] * len(nonenhancers))
    print('The number of positive and negative samples: %d, %d' % (len(enhancers), len(nonenhancers)))

    # input all coded feature vectors besides pssm feature
    all_features = ['SpectrumProfile', 'MismatchProfile', 'SubsequenceProfile', 'Pssm', 'Psednc']
    all_X = []
    for feature in all_features:
        if feature != 'Pssm':
            X = np.loadtxt(feature + 'Feature2.txt')
            all_X.append(X)
        else:
            all_X.append(instances)

    # all types of ensemble models
    all_ensemble_types = []
    for i in range(1, len(all_features) + 1):
        all_ensemble_types += list(itertools.combinations(range(len(all_features)), i))
    all_ensemble_types = [list(all_ensemble_types[i]) for i in range(len(all_ensemble_types))]

    # prediction based on ensemble learning model
    tic = time.clock()

    print('The prediction based on ensemble learning model, beginning')
    print('This process may spend about one hour, please do not close the program')

    classifier = XGBClassifier(learning_rate=0.08, n_estimators=20, max_depth=7, objective='binary:logistic',
                               metric='auc')
    folds = KFold(10, True, 1)
    auc_score, accuracy, sensitivity, specificity, MCC, unique_optimal_feature_ensembles = \
        globalCrossValidation(all_X, global_y, all_features, all_ensemble_types, classifier, folds)

    print('############################################################################')
    print('results for global cross validation, optimal feature combinations: ', unique_optimal_feature_ensembles)
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f, MCC:%.3f****' % (
        auc_score, accuracy, sensitivity, specificity, MCC))

    toc = time.clock()
    print('total running time:%.3f hour' % ((toc - tic) / 3600))
    print('############################################################################\n')

    # output results
    results = DataFrame({'Optimal feature subset': [unique_optimal_feature_ensembles], \
                         'AUC': [auc_score], \
                         'ACC': [accuracy], \
                         'SN': [sensitivity], \
                         'SP': [specificity], \
                         'MCC': [MCC]
                         })
    results = results[['Optimal feature subset', 'AUC', 'ACC', 'SN', 'SP', 'MCC']]
    results.to_csv('EnsembleLearningResults2.csv', index=False)
