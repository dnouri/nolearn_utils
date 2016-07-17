"""
ipython -i --pdb examples/cifar10/train.py
"""
import argparse
import numpy as np

from sklearn.metrics import log_loss, accuracy_score
from keras.datasets.cifar10 import load_data as load_cifar10_data

from nolearn_utils.utils import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print
    print '-----------------------------'
    print 'Load data'
    print '-----------------------------'
    (X_train, y_train), (X_test, y_test) = load_cifar10_data()
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    print
    print '-----------------------------'
    print 'Load model'
    print '-----------------------------'
    model = get_model('cifar10_model')

    layers = model.get_layers()
    train_iterator, test_iterator = model.get_iterators()
    net = model.get_net(layers, train_iterator, test_iterator)

    X_train, X_valid, y_train, y_valid = net.train_split(X_train, y_train, net)
    print X_train.shape, X_train.dtype, y_train.shape, y_train.dtype
    print X_valid.shape, X_valid.dtype, y_valid.shape, y_valid.dtype
    print X_test.shape, X_test.dtype, y_test.shape, y_test.dtype

    print 'Compiling model'
    net.initialize()

    print
    print '-----------------------------'
    print 'Start training'
    print '-----------------------------'
    net.fit(X_train, y_train)
    print 'Loading weights from', model.model_weights_fname
    net.load_params_from(model.model_weights_fname)

    y_train_pred_proba = net.predict_proba(X_train)
    y_valid_pred_proba = net.predict_proba(X_valid)
    y_test_pred_proba = net.predict_proba(X_test)

    y_train_pred = y_train_pred_proba.argmax(axis=1)
    y_valid_pred = y_valid_pred_proba.argmax(axis=1)
    y_test_pred = y_test_pred_proba.argmax(axis=1)

    print
    print '-----------------------------'
    print 'Evaluation'
    print '-----------------------------'
    print 'Log loss'
    print 'train', log_loss(y_train, y_train_pred_proba)
    print 'valid', log_loss(y_valid, y_valid_pred_proba)
    print 'test', log_loss(y_test, y_test_pred_proba)
    print
    print 'Accuracy'
    print 'train', accuracy_score(y_train, y_train_pred)
    print 'valid', accuracy_score(y_valid, y_valid_pred)
    print 'test', accuracy_score(y_test, y_test_pred)
