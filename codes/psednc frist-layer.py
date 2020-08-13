from repDNA.psenac import PseDNC
import numpy as np
from numpy import array
from pandas import DataFrame
import time

from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# Pseknc prediction............................................................................
def getCrossValidation(X, y, clf, folds):
    predicted_probability = -np.ones(len(y))
    predicted_label = -np.ones(len(y))
    X = np.array(X)
    y = np.array(y)
    for train_index, test_index in folds.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        probability_test = (clf.fit(X_train, y_train)).predict_proba(X_test)
        predicted_probability[test_index] = probability_test[:, 1]
        predicted_label[test_index] = (clf.fit(X_train, y_train)).predict(X_test)

    fpr, tpr, thresholds = roc_curve(y, predicted_probability, pos_label=1)
    auc_score = auc(fpr, tpr)
    accuracy = accuracy_score(y, predicted_label)
    sensitivity = recall_score(y, predicted_label)
    specificity = (accuracy * len(y) - sensitivity * sum(y)) / (len(y) - sum(y))
    MCC = matthews_corrcoef(y, predicted_label)
    return auc_score, accuracy, sensitivity, specificity, MCC
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

###########################################################################################


if __name__ == '__main__':
    featurename = 'Psednc'

    # getting psednc feature
    print('...............................................................................')
    print('Coding for ' + featurename + ' feature, beginning')
    tic = time.clock()

    psednc = PseDNC(lamada=1, w=0.05)
    pos_vec = psednc.make_psednc_vec(open('enhancers4.fasta'))
    neg_vec = psednc.make_psednc_vec(open('non-enhancers4.fasta'))
    Z = array(pos_vec + neg_vec)
    X = noramlization(Z)
    y = array([1] * len(pos_vec) + [0] * len(neg_vec))

    print('The number of positive and negative samples: %d,%d' % (len(pos_vec), len(neg_vec)))
    print('Dimension of ' + featurename + ' feature vectors: %d' % len(X[0]))

    toc = time.clock()
    print("Coding time: %.3f minutes" % ((toc - tic) / 60.0))
    print('...............................................................................')

    # output the psednc feature
    np.savetxt(featurename + 'Feature2.txt', X)

    # prediction based on psednc feature
    print('###############################################################################')
    print('The prediction based on ' + featurename + ', beginning')
    tic = time.clock()

    clf = XGBClassifier(learning_rate=0.1, n_estimators=20, max_depth=4, objective='binary:logistic')
    folds = KFold(10, True, 1)
    auc_score, accuracy, sensitivity, specificity, MCC = getCrossValidation(X, y, clf, folds)

    print('results for feature:' + featurename)
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f, MCC:%.3f****' % (
        auc_score, accuracy, sensitivity, specificity, MCC))

    toc = time.clock()
    print('The prediction time: %.3f minutes' % ((toc - tic) / 60.0))
    print('###############################################################################\n')

    # output result
    results = DataFrame({'Feature': [featurename], \
                         'AUC': [auc_score], \
                         'ACC': [accuracy], \
                         'SN': [sensitivity], \
                         'SP': [specificity], \
                         'MCC': [MCC]})
    results = results[['Feature', 'AUC', 'ACC', 'SN', 'SP', 'MCC']]
    results.to_csv(featurename + 'Results2.csv', index=False)
