import numpy as np
import pandas as pd
import glob
from skimage.io import imread
from skimage.transform import resize, rotate
import re
import os


def extract_train(im_size=20):

    n_train = 30336 # len(glob.glob('data/train/*/*.jpg'))
    im_area = im_size ** 2

    X = np.zeros((n_train, im_y))
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
        im = resize(im, (im_size, im_size))
        X[i, 0:im_area] = np.reshape(im, (1, im_area))

    D = pd.DataFrame(X, index=fns)
    D.index.name = 'image'
    D.to_csv('data/test_im_size=%d.csv' % im_size, index=True)
    os.system('gzip data/test_im_size=%d.csv' % im_size)
    print(D.shape)

    
if __name__ == '__main__':

    #extract_train(im_size=75)
    #extract_test(im_size=75)
    extract_train_c0(im_size=50)

