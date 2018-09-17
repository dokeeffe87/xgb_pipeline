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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from matplotlib import markers
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelBinarizer


def make_score_distribution(df, proba_cols, true_col, true_label, title=None, figsize=(10, 10)):
    """
    Makes a score distribution for examining the performance of a classifier.
    :param df: Dataframe containing the true labels, and predicted probabilities for each class
    :param proba_cols: A list of the column names in df which have the predicted probabilities
    :param true_col: A string which is the column name in df of the true labels
    :param true_label: The actual value of the true label (i.e. 1, 2, etc..)
    :param title: String, the title of plot.  Default is None.
    :param figsize: Optional: a tuple of integers for the figure size.
    :return: Nothing
    """
    # TODO: Test this for multi-class problems.
    fig, ax = plt.subplots(figsize=figsize)
    sns.distplot(df[proba_cols[0]].loc[(df[true_col] == true_label)], kde=False, ax=ax)
    if title:
        sns.distplot(df[proba_cols[1]].loc[(df[true_col] == true_label)], kde=False, ax=ax).set_title(title)
    else:
        sns.distplot(df[proba_cols[1]].loc[(df[true_col] == true_label)], kde=False, ax=ax)
    ax.set(xlabel='score', ylabel='count')
    plt.show()


def make_roc_curves_array(true_vals, scores_vals, figsize=(10, 10)):
    """
    Function for making nice-ish looking roc curves.
    :param true_vals: Array of true values
    :param scores_vals: Array of classifier scores
    :param figsize: Optional: a tuple of integers for the figure size
    :return: false positive rate, true positive rate, the corresponding threshold, and the auc score
    """
    fpr, tpr, threshold = roc_curve(y_true=true_vals, y_score=scores_vals)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return fpr, tpr, threshold, roc_auc


def make_roc_curves_df(df, true_col, scores_col, figsize=(10, 10)):
    """
    Function for making nice-ish looking roc curves. Take a dataframe as in input instead of arrays.
    :param df: Dataframe with true labels and predicted scores
    :param true_col: String. Column name in the dataframe of the true labels
    :param scores_col: String. Column name in the dataframe of the predicted scores
    :param figsize: Optional: a tuple of intergers for the figure size
    :return: false positive rate, true positive rate, the corresponding threshold, and the auc score
    """
    fpr, tpr, threshold = roc_curve(y_true=df[true_col].values, y_score=df[scores_col].values)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=figsize)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()
    
    return fpr, tpr, threshold, roc_auc


def calibration_func_array(true_vals, probability_vals_list, legend_labels, title, n_bins=20, figsize=(10,10)):
    """Function for making reliability plots"""
    y_vals = []
    x_vals = []
    for vals in probability_vals_list:
        y_array, x_array = calibration_curve(true_vals, vals, n_bins=n_bins)
        y_vals.append(y_array)
        x_vals.append(x_array)
    
    fig, ax = plt.subplots(figsize=figsize)
    for i in range(0, len(legend_labels)):
        plt.plot(x_vals[i], y_vals[i], linewidth=1, label=legend_labels[i], marker=markers.MarkerStyle.filled_markers[i])
    
    # Reference line, legends, and axis labels
    line = mlines.line2D([0, 1], [0, 1], color='white')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle(title)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.legend()
    plt.show()


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


class EarlyStoppingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier, evaluation_metric, num_rounds, feature_selector=None, validation_size=None):
        self.classifier = classifier
        self.evaluation_metric = evaluation_metric
        self.num_rounds = num_rounds
        self.feature_selector = feature_selector
        self.validation_size = validation_size
        self.classes_ = None
        self.classifier_ = None
        self.feature_selector_ = None
        
    def fit(self, X, y):
        """Function for fitting the base estimator.  Implements feature selection and also auto generates a validation
        set for early stopping."""
        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        
        # Do feature selection if desired
        if self.feature_selector:
            X_use = self.feature_select(X, y)
        else:
            X_use = X
        
        # Make the validation set:
        X_train, X_valid, y_train, y_valid = self.make_validation_set(X_use, y)

        # Fit the base estimator
        self.classifier_ = clone(self.classifier)
        self.classifier = self.classifier_.fit(X_train,
                                               y_train,
                                               eval_metric=self.evaluation_metric,
                                               eval_set=[[X_valid, y_valid]],
                                               early_stopping_rounds=self.num_rounds)
        
        return self
    
    def predict(self, X):
        """Makes predictions on input data from trained classifier.  
        If feature selection is used, it is applied automatically to the input data."""
        # The predict function changes in XGBoost when early stopping is performed.
        if self.feature_selector:
            X_select = self.feature_selector.transform(X)
        else:
            X_select = X
            
        # This method should only be called once the model is fit anyway, and since we are always using early stopping, 
        # we can set ntree_limit by default.
        return self.classifier.predict(X_select, ntree_limit=self.classifier.best_ntree_limit)
    
    def predict_proba(self, X):
        """Makes probability predictions on input data from trained classifier.  If feature selection is used,
        it is applied automatically to the input data.  Since this method should only be called after the 
        classifier is fit, and since we are always using early stopping, the best_ntree_limit should be defined by default."""
        if self.feature_selector:
            X_select = self.feature_selector.transform(X)
        else:
            X_select = X
        
        return self.classifier.predict_proba(X_select, ntree_limit=self.classifier.best_ntree_limit)

    def feature_select(self, X, y):
        """Implements feature selection if so desired."""
        # TODO: Get tree based feature selection working.
        self.feature_selector_ = clone(self.feature_selector)
        self.feature_selector = self.feature_selector_.fit(X, y)
        
        return self.feature_selector.transform(X)
        
    def make_validation_set(self, X, y):
        """Makes the validation set."""
        if self.validation_size:
            val_size = self.validation_size
        else:
            val_size = 0.1
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=val_size)
        return X_train, X_valid, y_train, y_valid


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
        """This assumes an unsupervised method, so there is no target variable (hence stacking X and y).
        Should I really split off a subset for training here, and then predict outliers on the remainder,
        returning only those that made it through?"""
        # TODO: Add mv and em support to better diagnose performance of unsupervised outlier detection techniques.
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