import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit

problem_title = 'Wine review'
_target_column_names = ['price','points']
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=_target_column_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()

score_types = [
               rw.score_types.RMSE(),
               rw.score_types.RelativeRMSE(name='rel_rmse'),]


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_names].values
    X_df = data.drop(_target_column_names, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=42)
    return cv.split(X, y)
