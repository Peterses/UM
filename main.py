# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

clfs = {
    'GNB': GaussianNB(),
    'SVM': SVC(),
    'kNN': KNeighborsClassifier()
}

datasets = [
    'cleveland',
    'appendicitis',
    'contraceptive',
    'monk-2',
    'bupa',
    'phoneme',
    'newthyroid',
    'hayes-roth',
    'winequality-white',
    'winequality-red',
    'ring',
    'banana',
    'titanic',
    'led7digit',
    'glass',
    'wine',
    'segment',
    'twonorm',
    'texture',
    'spambase'
]

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=1234
)
scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))


for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.dat" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

ensembleMethods = {
    'Bagging': BaggingClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'RandomForest': RandomForestClassifier(),
    'Voting': VotingClassifier()
}


for score in scores:
    BaggingClassifier(score)