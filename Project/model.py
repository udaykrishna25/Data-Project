import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import scipy
from sklearn.metrics import cohen_kappa_score
from functools import partial

import pandas as pd
import numpy as np

def train_and_test(params, X_train, y_train, X_test, verbose=False):
    """Trains an XGBoost classifier on training data, reports accuracy
    and classifies test points.

    Parameters:
    - :param params: the hyperparameters of the XGBoost classifier
    - :param X_train: a DataFrame of the training data
    - :param y_train: a Series of the output labels of the training data
    - :param X_test: a DataFrame of the test data
    - :param verbose: set to True if you would like log of the training process

    """
    if verbose:
        print('Training the classifier')
    _, oof_train, oof_test = run_xgb(params, X_train, y_train, X_test, verbose=False)
    optR = OptimizedRounder()
    optR.fit(oof_train, y_train)
    coefficients = optR.coefficients()

    if verbose:
        print('Computing accuracy')
    valid_pred = optR.predict(oof_train, coefficients)

    if verbose:
        print('Predicting on test set')
    preds = optR.predict(oof_test.mean(axis=1), coefficients.copy()).astype(np.int8)
    return preds, optR, cohen_kappa_score(y_train, valid_pred, weights='quadratic')

class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return -cohen_kappa_score(y, preds, weights='quadratic')

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = scipy.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [0, 1, 2, 3, 4])
        return preds

    def coefficients(self):
        return self.coef_['x']

def run_xgb(params, X_train, y_train, X_test, verbose=False):

    xgb_params = {
        'eval_metric': 'rmse',
        'seed': 1337,
        'eta': 0.0123,
        'subsample': 0.8,
        'colsample_bytree': 0.85,
        'silent': 1 if verbose else 0,
    }

    kf = StratifiedKFold(n_splits=params['n_splits'], shuffle=True, random_state=1337)

    oof_train = np.zeros((X_train.shape[0]))
    oof_test = np.zeros((X_test.shape[0], params['n_splits']))

    i = 0

    for train_idx, valid_idx in kf.split(X_train, y_train):

        X_tr = X_train.iloc[train_idx, :]
        X_val = X_train.iloc[valid_idx, :]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[valid_idx]

        d_train = xgb.DMatrix(data=X_tr, label=y_tr, feature_names=X_tr.columns)
        d_valid = xgb.DMatrix(data=X_val, label=y_val, feature_names=X_val.columns)

        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        model = xgb.train(dtrain=d_train, num_boost_round=params['num_rounds'], evals=watchlist,
                         early_stopping_rounds=params['early_stop'], verbose_eval=params['verbose_eval'], params=xgb_params)

        valid_pred = model.predict(xgb.DMatrix(X_val, feature_names=X_val.columns), ntree_limit=model.best_ntree_limit)
        test_pred = model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit)

        oof_train[valid_idx] = valid_pred
        oof_test[:, i] = test_pred

        i += 1
    return model, oof_train, oof_test
