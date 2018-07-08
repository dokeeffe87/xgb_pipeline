"""
              ___.                          __           .__              .__  .__
___  ___  ____\_ |__   ____   ____  _______/  |_  ______ |__|_____   ____ |  | |__| ____   ____
\  \/  / / ___\| __ \ /  _ \ /  _ \/  ___/\   __\ \____ \|  \____ \_/ __ \|  | |  |/    \_/ __ \
 >    < / /_/  > \_\ (  <_> |  <_> )___ \  |  |   |  |_> >  |  |_> >  ___/|  |_|  |   |  \  ___/
/__/\_ \\___  /|___  /\____/ \____/____  > |__|   |   __/|__|   __/ \___  >____/__|___|  /\___  >
      \/_____/     \/                  \/         |__|      |__|        \/             \/     \/


A custom pipeline for xgboost models.  It builds in automated outlier detection and removal, as well as feature
selection, all to be run as a replacement of a standard sklearn pipeline object during cross validation.

Version 1.0.0
"""

# Import modules

from __future__ import division
import sys
import os
import pandas as pd
import fuzzymatch as fz
from sklearn.metrics import auc


import xgboost as xgb
from pandas import rolling_median
import numpy as np
import datetime
from datetime import timedelta
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, median_absolute_error
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import IsolationForest, ExtraTreesRegressor
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin, clone
from itertools import compress
from math import exp, log
import pickle
import sys




# ultimately replace these with IsolationForest outlier detection.
def rolling_median_filter(df, col_use, threshold, window):
    median_col_name = col_use + '_med'
    df[median_col_name] = rolling_median(df[col_use], window = window, center=True).fillna(method='bfill').fillna(method='ffill')
    difference = np.abs(df[col_use] - df[median_col_name])
    outlier_idx = difference > threshold
    return outlier_idx


def detect_outlier_position_by_fft(signal, threshold_freq=0.1, frequency_amplitude=0.001):
    signal = signal.copy()
    fft_of_signal = np.fft.fft(signal)
    outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
    if np.any(np.abs(fft_of_signal[threshold_freq:]) > frequency_amplitude):
        index_of_outlier = np.where(signal == outlier)
        return index_of_outlier[0]
    else:
        return None


def apply_fft_detection(df, col, win):
    outlier_idx = []
    y = df[col].values
    # opt = dict(threshold_freq=0.01, frequency_amplitude=0.001)
    # Better...
    # opt = dict(threshold_freq=0.001, frequency_amplitude=0.01)
    opt = dict(threshold_freq=0.001, frequency_amplitude=0.001)

    for k in range(win * 2, y.size, win):
        idx = detect_outlier_position_by_fft(y[k - win:k + win], **opt)
        if idx is not None:
            outlier_idx.append(k + idx[0] - win)
    outlier_idx = list(set(outlier_idx))

    # Get some summary stats
    outlier_values = df[col][outlier_idx].values

    print('Mean value of data: {0}'.format(np.mean(y)))
    print('Maximum value in data: {0}'.format(max(y)))
    print('Minimum value in data: {0}'.format(min(y)))
    print('Mean value of outlier data: {0}'.format(np.mean(outlier_values)))
    print('Maximum value of the outlier data: {0}'.format(max(outlier_values)))
    print('Minimum value of the outlier data: {0}'.format(min(outlier_values)))
    return outlier_idx


def em(t, t_max, volume_support, s_unif, s_X, n_generated):
    """
    Excess mass algorithm, used to validate the unsupervised IsolationForest algorithm
    :param t:
    :param t_max:
    :param volume_support:
    :param s_unif:
    :param s_X:
    :param n_generated:
    :return:
    """
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.
    for u in s_X_unique:
        # if (s_unif >= u).sum() > n_generated / 1000:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                          t * (s_unif > u).sum() / n_generated
                          * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        print '\n failed to achieve t_max \n'
        amax = -1
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


def mv(axis_alpha, volume_support, s_unif, s_X, n_generated):
    """
    Mass-Volume algorithm, used to validate the unsupervised IsolationForest algorithm
    :param axis_alpha:
    :param volume_support:
    :param s_unif:
    :param s_X:
    :param n_generated:
    :return:
    """
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        # pdb.set_trace()
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return auc(axis_alpha, mv), mv





class OutlierRemovalFeatureSelectionRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, outlier_detector, regressor, feature_selector, use_tree_selection=False, select_alg=None):
        self.outlier_detector = outlier_detector
        self.regressor = regressor
        self.feature_selector = feature_selector
        self.use_tree_selection = use_tree_selection
        self.select_alg = select_alg

    def fit(self, X, y):
        X_resample, y_resample = self.resample(X, y)
        X_resample_features = self.feature_select(X_resample, y_resample)
        # Generate the validation set for early stopping
        X_train, X_valid, y_train, y_valid = self.make_validation_set(X_resample_features, y_resample)
        # set the early_stopping rounds to be about 10% the number of epochs. Otherwise risk under fitting...
        self.regressor_ = clone(self.regressor).fit(X_train, y_train, eval_metric='mae',
                                                    eval_set=[[X_valid, y_valid]], early_stopping_rounds=50)
        return self

    def predict(self, X):
        X_red = self.feature_selector.transform(X)
        return self.regressor_.predict(X_red)

    def resample(self, X, y):
        self.outlier_detector_ = clone(self.outlier_detector)
        y_shape = y.reshape((y.shape[0], 1))
        X_mat = np.hstack((X, y_shape))
        self.outlier_detector = self.outlier_detector_.fit(X_mat)
        mask = self.outlier_detector.predict(X_mat) == 1
        return X[mask], y[mask]

    def feature_select(self, X, y):
        # This will run the feature selection component.
        if self.use_tree_selection:
            # THIS DOESN'T WORK! DON'T USE IT YET!
            self.feature_selector_ = clone(self.feature_selector)
            # self.select_alg_ = clone(self.select_alg)
            self.feature_selector = SelectFromModel(self.feature_selector_.fit(X, y))
            # self.feature_selector = self.select_alg(self.feature_selector, prefit=True)
            # self.feature_selector = SelectFromModel(self.feature_selector, prefit=True)
            feature_selected_X = self.feature_selector.transform(X)
        else:
            self.feature_selector_ = clone(self.feature_selector)
            self.feature_selector = self.feature_selector_.fit(X, y)
            feature_selected_X = self.feature_selector.transform(X)
        return feature_selected_X

    def make_validation_set(self, X, y):
        # This will generate the validation set for the xgboost regressor.
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
        return X_train, X_valid, y_train, y_valid