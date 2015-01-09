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


classes = [c.split()[0].strip() for c in open('classes.txt')]
im_size = 75

df = pd.read_csv('data/train_im_size=%d.csv.gz' % im_size, compression='gzip')
X = df.ix[:, :(im_size ** 2)]
y = pd.Categorical(df['class'], categories=classes).codes

print(X.shape)

#clf = DummyClassifier('most_frequent')
#clf = KNeighborsClassifier(n_neighbors=5)
clf = RandomForestClassifier(n_estimators=150, n_jobs=-1)
#clf = SVC(probability=True)

pca = PCA(n_components=20)
X = pca.fit_transform(X)

print(X.shape)

###############################################################################

# mcll = make_scorer(multiclass_log_loss, needs_proba=True,
#                    greater_is_better=True)
# print(cross_val_score(clf, X, y, scoring=mcll, n_jobs=-1))

###############################################################################

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# print(X_train.shape)
# print(X_test.shape)

# clf.fit(X_train, y_train)
# y_pred = clf.predict_proba(X_test)

# # alpha = 0.025
# # print(multiclass_log_loss(y_test, y_pred, smoothing_alpha=alpha))
# for alpha in np.arange(0, 0.2, 0.01):
#     print(alpha, multiclass_log_loss(y_test, y_pred, smoothing_alpha=alpha))

###############################################################################

clf.fit(X, y)

del X, y

df_test = pd.read_csv('data/test_im_size=%d.csv.gz' % im_size,
                      compression='gzip')
X_test = df_test.ix[:, 1:]

print(X_test.shape)

X_test = pca.transform(X_test)

print(X_test.shape)

y_pred = clf.predict_proba(X_test)

alpha = 0.025
yp = pd.DataFrame(smooth_probs(y_pred, alpha), columns=classes,
                  index=df_test.image)
yp.index.name = 'image'

print(yp.shape)

yp.to_csv('submissions/im=%d_pca=20_rf=150_alpha=025.csv' % im_size)
os.system('gzip submissions/im=%d_pca=20_rf=150_alpha=025.csv' % im_size)
