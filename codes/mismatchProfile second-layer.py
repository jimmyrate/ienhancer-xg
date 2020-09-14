import numpy as np
from numpy import array
from pandas import DataFrame
from itertools import combinations_with_replacement, permutations
import time
import shap
import matplotlib
#import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display, Image
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import matthews_corrcoef

# input sequences................................................................................

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


# getting (k,m)-mismatch profile.................................................................
def getMismatchProfileMatrix(instances, enhancer, k, m):
    p = len(enhancer)
    kmerdict = getKmerDict(enhancer, k)
    features = []
    if k == 1 or k == 2:
        for sequence in instances:
            vector = getSpectrumProfileVector(sequence, kmerdict, p, k)
            features.append(vector)
    else:
        if m == 1:
            for sequence in instances:
                vector = getMismatchProfileVector(sequence, enhancer, kmerdict, p, k)
                features.append(vector)
        else:
            assert ('you should reset m<=1')
    return array(features)


def getKmerDict(enhancer, k):
    kmerlst = []
    partkmers = list(combinations_with_replacement(enhancer, k))
    for element in partkmers:
        elelst = set(permutations(element, k))
        strlst = [''.join(ele) for ele in elelst]
        kmerlst += strlst
    kmerlst = np.sort(kmerlst)
    kmerdict = {kmerlst[i]: i for i in range(len(kmerlst))}
    return kmerdict


def getSpectrumProfileVector(sequence, kmerdict, p, k):
    vector = np.zeros((1, p ** k))
    n = len(sequence)
    for i in range(n - k + 1):
        subsequence = sequence[i:i + k]
        position = kmerdict.get(subsequence)
        vector[0, position] += 1
    return list(vector[0])


def getMismatchProfileVector(sequence, enhancer, kmerdict, p, k):
    n = len(sequence)
    vector = np.zeros((1, p ** k))
    for i in range(n - k + 1):
        subsequence = sequence[i:i + k]
        position = kmerdict.get(subsequence)
        vector[0, position] += 1
        for j in range(k):
            substitution = subsequence
            for letter in list(set(piRNAletter) ^ set(subsequence[j])):
                substitution = list(substitution)
                substitution[j] = letter
                substitution = ''.join(substitution)
                position = kmerdict.get(substitution)
                vector[0, position] += 1
    return list(vector[0])


# prediction based on mismatch profile..........................................................
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
def getshapvalue(X, y,clf):
    shap.initjs()
    model = clf.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    image_output1 = shap.summary_plot(shap_values, X, plot_type="bar")
    image_output1.save("shap1.tiff")
    image_output2 = shap.summary_plot(shap_values, X)
    image_output2.save("shap2.tiff")
    image_output3 = shap.force_plot(explainer.expected_value, shap_values, X)
    image_output3.save("shap3.tiff")
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

#############################################################################################


if __name__ == '__main__':

    featurename = 'MismatchProfile'
    enhancer = ['A', 'C', 'G', 'T']

    # input sequences for pssm feature
    fp = open("strong enhancers3.fasta", 'r')
    strong_enchancer = getSequences(fp)
    fn = open("weak enhancers3.fasta", 'r')
    weak_enchancer = getSequences(fn)
    instances = array(strong_enchancer + weak_enchancer)
    y = array([1] * len(strong_enchancer) + [0] * len(weak_enchancer))
    global_y = array([1] * len(strong_enchancer) + [0] * len(weak_enchancer))
    print('The number of positive and negative samples: %d, %d' % (len(strong_enchancer), len(weak_enchancer)))
    # getting (k,m)-mismatch profiles for (k,m)=(1,0),(2,0),(3,1),(4,1),(5,1)
    for args in [[1, 0], [2, 0], [3, 1], [4, 1], [5, 1]]:

        k, m = args[0], args[1]

        print('...............................................................................')
        print('Coding for (' + str(k) + ',' + str(m) + ')-' + featurename + ', beginning')
        tic = time.clock()

        Z = getMismatchProfileMatrix(instances1, enhancer, k, m)
        X = noramlization(Z)
        print('Dimension of (' + str(k) + ',' + str(m) + ')-' + featurename + ': %d' % len(X[0]))
        toc = time.clock()
        print('Coding time: %.3f minutes' % ((toc - tic) / 60.0))
        print('...............................................................................')

        if k == 1:
            all_X = X
        else:
            all_X = np.hstack((all_X, X))
    np.savetxt(featurename + 'Feature1.txt', all_X)
    print('###############################################################################')
    print('The prediction based on ' + featurename + ', beginning')
    tic = time.clock()

    clf = XGBClassifier(learning_rate=0.17, n_estimators=10, max_depth=4, objective='binary:logistic')
    folds = KFold(10, True, 1)
    getshapvalue(all_X, y1, clf)
    auc_score, accuracy, sensitivity, specificity, MCC = getCrossValidation(all_X, y1, clf, folds)

    print('results for feature:' + featurename + 'strong or not')
    print('****AUC score:%.3f, accuracy:%.3f, sensitivity:%.3f, specificity:%.3f, MCC:%.3f****' % (
        auc_score, accuracy, sensitivity, specificity, MCC))

    toc = time.clock()
    print('The prediction time: %.3f minutes' % ((toc - tic) / 60.0))
    print('###############################################################################\n')
    results = DataFrame({'Feature': [featurename], \
                         'AUC': [auc_score], \
                         'ACC': [accuracy], \
                         'SN': [sensitivity], \
                         'SP': [specificity], \
                         'MCC': [MCC]}
                        )
    results = results[['Feature', 'AUC', 'ACC', 'SN', 'SP', 'MCC']]
    results.to_csv(featurename + 'Results1.csv', index=False)

