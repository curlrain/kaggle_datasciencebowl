from __future__ import print_function, division
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
import pandas as pd
import os


srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    l1a = rectify(conv2d(X, w, border_mode='full'))
    l1 = max_pool_2d(l1a, (2, 2))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    l2 = max_pool_2d(l2a, (2, 2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    l3b = max_pool_2d(l3a, (2, 2))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

do_leaderboard = False

im_size = 28

df_train = pd.read_csv('../data/train_im_size=%d.csv.gz' % im_size,
                       compression='gzip')
if do_leaderboard:
    df_test = pd.read_csv('../data/test_im_size=%d.csv.gz' % im_size,
                          compression='gzip', index_col='image')
    
n_classes = 121
assert n_classes == len(df_train['class'].unique())

# # shuffle whole training set
df_train = df_train.reindex(np.random.permutation(df_train.index))
df_train.reset_index(inplace=True, drop=True)

#df_train = df_train.iloc[:10000, :]
#df_test = df_test.iloc[:10000, :]

# will be in alphabetical order
class_categorical = pd.Categorical(df_train['class'])
class_codes = class_categorical.codes
assert len(class_codes) == df_train.shape[0]
class_categories = class_categorical.categories
assert len(class_categories) == n_classes

if do_leaderboard:

    n_train = df_train.shape[0]
    trX = df_train.iloc[:, :(im_size ** 2)]
    trY = np.zeros((n_train, n_classes))
    trY[np.arange(n_train), class_codes] = 1 # one-hot

    n_test = df_test.shape[0]
    teX = df_test
    
else:

    train_prop = 0.8
    n_train = int(df_train.shape[0] * train_prop)          
    trX = df_train.iloc[:n_train, :(im_size ** 2)]
    trY = np.zeros((n_train, n_classes))
    trY[np.arange(n_train), class_codes[:n_train]] = 1 # one-hot

    n_test = df_train.shape[0] - n_train
    teX = df_train.iloc[n_train:, :(im_size ** 2)]
    assert teX.shape[0] == n_test
    teY = np.zeros((n_test, n_classes))
    teY[np.arange(n_test), class_codes[n_train:]] = 1 # one-hot

    # make sure test will see the 121 classes
    assert len(set(class_codes[:n_train])) == n_classes
    
trX = trX.values.reshape(-1, 1, im_size, im_size)
teX = teX.values.reshape(-1, 1, im_size, im_size)

print('train:', trX.shape, trY.shape)
print('test:', teX.shape, teY.shape if not do_leaderboard else '')

X = T.ftensor4()
Y = T.fmatrix()

w = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w_o = init_weights((625, n_classes))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3,
                                                           w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
#y_x = T.argmax(py_x, axis=1)

cost_train = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
cost_test = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost_train, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost_train, updates=updates,
                        allow_input_downcast=True)
if do_leaderboard:
    predict = theano.function(inputs=[X], outputs=py_x,
                              allow_input_downcast=True)
else:
    predict = theano.function(inputs=[X, Y], outputs=cost_test,
                              allow_input_downcast=True)

n_epochs = 50
minibatch_size = 128

for i in range(n_epochs):
    train_costs = []
    for start, end in zip(range(0, n_train, minibatch_size),
                          range(minibatch_size, n_train, minibatch_size)):
        train_costs.append(train(trX[start:end], trY[start:end]))
    if do_leaderboard:
        print(i, np.average(train_costs))
    else:
        print(i, np.average(train_costs), predict(teX, teY))

if do_leaderboard:
    sub = pd.DataFrame(predict(teX), columns=class_categories)
    sub.index = df_test.index
    sub.index.name = 'image'
    fn = '../submissions/bla.csv'
    sub.to_csv(fn, index=True)
    os.system('gzip %s' % fn)
