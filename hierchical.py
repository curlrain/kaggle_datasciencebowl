from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA, TruncatedSVD, RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
import os
from collections import defaultdict


def multiclass_log_loss(y_true, y_pred, eps=1e-15, smoothing_alpha=0.025):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    if smoothing_alpha:
        predictions = smooth_probs(predictions, smoothing_alpha)

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), np.asarray(y_true).astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss


def smooth_probs(P, alpha=0.5):
    m, n = P.shape
    M = np.repeat(1/n, m)[:, np.newaxis]
    D = P - M
    return P - (alpha * D)


if __name__ == '__main__':

    classes1 = []
    classes0 = []
    code0_to_code1s = defaultdict(list)
    for i, line in enumerate(open('classes.txt')):
        class1, class0 = map(str.strip, line.split())
        if class0 not in classes0:
            classes0.append(class0)
        code0_to_code1s[classes0.index(class0)].append(i)
        classes1.append(class1)

    # for code0, code1s in code0_to_code1s.items():
    #     print(code0, code1s)
        
    im_size = 50
    n_c0_trees = 500
    n_c1_trees = 100
    n_pca_comps = 10

    df = pd.read_csv('data/train_c0_im_size=%d.csv.gz' % im_size,
                      compression='gzip')

    print(df.shape)

    X = df.iloc[:, :(im_size ** 2)]
    
    pca = PCA(n_components=n_pca_comps)
    X = pca.fit_transform(X)
    
    df_pca = pd.DataFrame(X)    
    df_pca['code1'] = pd.Categorical(df['class1'], categories=classes1).codes
    df_pca['code0'] = pd.Categorical(df['class0'], categories=classes0).codes

    print(df_pca.shape)

    df_train, df_test = train_test_split(df_pca, train_size=0.8)
    cols = list(range(n_pca_comps)) + ['code1', 'code0']
    df_train = pd.DataFrame(df_train, columns=cols)
    df_test = pd.DataFrame(df_test, columns=cols)

    X_train = df_train.iloc[:, :n_pca_comps]
    
    print(X_train.shape)

    ## TRAIN

    clf_c0 = RandomForestClassifier(n_estimators=n_c0_trees, n_jobs=-1)

    clf_c0.fit(X_train, df_train['code0'])

    code0_to_clf = {}
    for code0, df_train_c0 in dict(list(df_train.groupby('code0'))).items():
         #print(c0, df_train_c0.shape)
         code0_to_clf[code0] = RandomForestClassifier(n_estimators=n_c1_trees,
                                                      n_jobs=-1)
         X_train_c0 = df_train_c0.iloc[:, :n_pca_comps]
         code0_to_clf[code0].fit(X_train_c0, df_train_c0['code1'])

    ## TEST

    X_test = df_test.iloc[:, :n_pca_comps]

    print(X_test.shape)

    # N x 30
    c0_test_pred = clf_c0.predict_proba(X_test) 
    
    print('c0_test_pred:', c0_test_pred.shape)

    # N x 130
    c1_test_pred = np.zeros((c0_test_pred.shape[0],
                             len(classes1)))

    print('c1_test_pred:', c1_test_pred.shape)

    for code0, clf in code0_to_clf.items():
        c1_test_pred_c0 = clf.predict_proba(X_test)
        p = c1_test_pred_c0 * c0_test_pred[:, [code0]]
        c1_test_pred[:, code0_to_code1s[code0]] = p

    #print(np.sum(c1_test_pred, axis=1)) # should be all 1s

    #alpha = 0.025
    for alpha in np.arange(0, 0.5, 0.05):
        print(alpha, multiclass_log_loss(df_test['code1'], c1_test_pred,
                                         smoothing_alpha=alpha))
