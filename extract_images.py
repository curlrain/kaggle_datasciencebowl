import numpy as np
import pandas as pd
import glob
from skimage.io import imread
import skimage.transform as tf
import sklearn.utils
import cPickle
import re
import os


def extract_train(im_size=20):

    n_train = 30336 # len(glob.glob('data/train/*/*.jpg'))
    im_area = im_size ** 2

    X = np.zeros((n_train, im_area))
    y = []

    for i, fn in enumerate(glob.glob('data/train/*/*.jpg')):
        c = re.search('data/train/(.*)/.*.jpg', fn).group(1)
        y.append(c)
        im = imread(fn, as_grey=True)
        im = resize(im, (im_size, im_size))
        X[i, 0:im_area] = np.reshape(im, (1, im_area))

    # i = 0
    # for fn in glob.glob('data/train/*/*.jpg'):
    #     c = re.search('data/train/(.*)/.*.jpg', fn).group(1)
    #     #y.append(c)
    #     im = imread(fn, as_grey=True)
    #     for j in [0, 1]:
    #         if j == 1:
    #             im = np.fliplr(im)
    #         for angle in [0, 90, 180, 270]:
    #             im2 = rotate(im, angle, resize=True)
    #             im2 = resize(im2, (im_size, im_size))
    #             X[i, 0:im_area] = np.reshape(im2, (1, im_area))
    #             i += 1
    #             y.append(c)

    D = pd.DataFrame(X)
    D['class'] = y
    D.to_csv('data/train_im_size=%d.csv' % im_size, index=False)
    os.system('gzip data/train_im_size=%d.csv' % im_size)
    print(D.shape)

def extract_train_trans2(im_size=20):

    n_train = 30336 # len(glob.glob('data/train/*/*.jpg'))
    im_area = im_size ** 2

    X = np.zeros((n_train * 5, im_area))
    y = []

    i = 0
    for fn in glob.glob('data/train/*/*.jpg'):
        c = re.search('data/train/(.*)/.*.jpg', fn).group(1)
        im = imread(fn, as_grey=True)
        for tr in [(0,0), (0,2), (2,0), (0,-2), (-2,0)]:
            im2 = tf.warp(im, tf.AffineTransform(translation=tr))
            if tr[0]:
                if tr[0] > 0:
                    im2[:, :tr[0]] = 1
                else:
                    im2[:, -tr[0]:] = 1
            elif tr[1]:
                if tr[1] > 0:
                    im2[:tr[1], :] = 1
                else:
                    im2[-tr[1]:, :] = 1
            im2 = tf.resize(im2, (im_size, im_size))
            X[i, 0:im_area] = np.reshape(im2, (1, im_area))
            i += 1
            y.append(c)

    D = pd.DataFrame(X)
    D['class'] = y
    D.to_csv('data/train_t2_im_size=%d.csv' % im_size, index=False)
    os.system('gzip data/train_t2_im_size=%d.csv' % im_size)
    print(D.shape)
    

def extract_train_c0(im_size=20):

    c1_to_c0 = {}
    for line in open('classes.txt'):
        c1, c0 = map(str.strip, line.split())
        c1_to_c0[c1] = c0
    
    n_train = 30336 # len(glob.glob('data/train/*/*.jpg'))
    im_area = im_size ** 2

    X = np.zeros((n_train, im_area))
    y = []

    for i, fn in enumerate(glob.glob('data/train/*/*.jpg')):
        c = re.search('data/train/(.*)/.*.jpg', fn).group(1)
        y.append(c)
        im = imread(fn, as_grey=True)
        im = resize(im, (im_size, im_size))
        X[i, 0:im_area] = np.reshape(im, (1, im_area))

    D = pd.DataFrame(X)
    D['class1'] = y
    D['class0'] = D['class1'].map(lambda c1: c1_to_c0[c1])
    D.to_csv('data/train_c0_im_size=%d.csv' % im_size, index=False)
    os.system('gzip data/train_c0_im_size=%d.csv' % im_size)
    print(D.shape)
    return D


def extract_test(im_size=20):
    
    n_test = 130400
    im_area = im_size ** 2

    X = np.zeros((n_test, im_area))
    fns = []

    for i, fn in enumerate(glob.glob('data/test/*.jpg')):
        fns.append(re.search('data/test/(.*.jpg)', fn).group(1))
        im = imread(fn, as_grey=True)
        im = tf.resize(im, (im_size, im_size))
        X[i, 0:im_area] = np.reshape(im, (1, im_area))

    return X, fns
    # D = pd.DataFrame(X, index=fns)
    # D.index.name = 'image'
    # D.to_csv('data/test_im_size=%d.csv' % im_size, index=True)
    # os.system('gzip data/test_im_size=%d.csv' % im_size)
    # print(D.shape)


def extract_for_theano(im_size=28):

    n_train = 30336 # len(glob.glob('data/train/*/*.jpg'))
    im_area = im_size ** 2

    X = np.zeros((n_train, im_area))
    y = []

    for i, fn in enumerate(glob.glob('data/train/*/*.jpg')):
        c = re.search('data/train/(.*)/.*.jpg', fn).group(1)
        y.append(c)
        im = imread(fn, as_grey=True)
        im = tf.resize(im, (im_size, im_size))
        X[i, 0:im_area] = np.reshape(im, (1, im_area))
        #if i > 10: break
        
    y = pd.Categorical(y).codes
    X, y = sklearn.utils.shuffle(X, y)
    datasets = []
    datasets.append((X[:24300], y[:24300]))
    datasets.append((X[24300:], y[24300:]))
    test, fns = extract_test(im_size)
    datasets.append((test, np.array([-1] * test.shape[0])))
    for i in range(3):
        print datasets[i][0].shape, datasets[i][1].shape
    with open('data/test_image_order.csv', 'w') as f:
        for fn in fns:
            f.write('%s\n' % fn)
    # with open('bowl.pkl', 'wb') as f:
    #     cPickle.dump(datasets, f)
    
    
if __name__ == '__main__':

    #extract_train(im_size=75)
    #extract_test(im_size=75)
    #extract_train_c0(im_size=50)
    #extract_train_trans2(im_size=20)
    extract_for_theano(im_size=28)
