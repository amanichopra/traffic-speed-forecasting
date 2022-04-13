import pickle
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from time import time
import numpy as np
from prophet import Prophet
from tensorflow.keras import Sequential
from subprocess import call

# loads preprocessed data
def load_processed_data(path_to_data, method='pickle'):
    if method == 'pickle':
        with open(f'{path_to_data}/adj_mat.dat', 'rb')  as f:
            adj_mat = pickle.load(f)

        with open(f'{path_to_data}/adj_mat_ind_station_mapper.dat', 'rb') as f:
            ind_station_mapper = pickle.load(f)

        with open(f'{path_to_data}/speeds.dat', 'rb') as f:
            speed_df = pickle.load(f)
    elif method == 'csv':
        adj_mat = pd.read_csv(f'{path_to_data}/csvs/adj_mat.csv', index_col=0).values
        ind_station_mapper = pd.read_csv(f'{path_to_data}/csvs/adj_mat_ind_station_mapper.csv', index_col=0).set_index('adj_mat_ind').to_dict()['station_ind']
        speed_df = pd.read_csv(f'{path_to_data}/csvs/speeds.csv', index_col=0)
        speed_df.index = pd.to_datetime(speed_df.index)
    else:
        raise Exception('Method must be pickle or csv.')
    return adj_mat, ind_station_mapper, speed_df


# performs k-fold cv
def cv(model, data, folds=5, metrics=['mse', 'mae', 'rmse', 'r2'], epochs=2, verbose=False):
    cv_metrics = {(fold+1): {} for fold in range(folds)}
    if metrics is None: return cv_metrics
    if type(data) == list:
        X = data[0]
        y = data[1]
        splits = [np.array_split(X, folds), np.array_split(y, folds)] # splits X, splits y
    else:
        splits = np.array_split(data, folds)
        
    for fold in range(folds):
        if type(model) == Sequential:
            mod = model
        else:
            mod = deepcopy(model)
        
        if type(data) == list: # indicates X, y are provided as data
            splits_X = splits[0]
            splits_y = splits[1]
            train = [np.concatenate(splits_X[:(fold+1)]), np.concatenate(splits_y[:(fold+1)])] # X, y
            valid = [splits_X[fold], splits_y[fold]] # X, y
        else:
            train = pd.concat(splits[:(fold+1)])
            valid = splits[fold]

        
        if type(mod) == Prophet:
            train = pd.DataFrame({'ds': train.index, 'y': train.values})
            start = time()
            mod.fit(train)
            end = time()
            preds = mod.predict(pd.DataFrame({'ds': valid.index}))
            preds = preds[preds['ds'].isin(train['ds'])]['yhat'].values
        elif type(train) == list and type(valid) == list:
            start = time()
            mod.fit(train[0], train[1], epochs=epochs, verbose=verbose)
            end = time()
            preds = mod.predict(valid[0]).flatten()
            valid = valid[1].flatten()
        else:
            start = time()
            mod.fit(train)
            end = time()
            preds = mod.predict(valid)
        cv_metrics[(fold+1)]['train_time'] = (end - start)
        for metric in metrics:
            if metric == 'mse':
                cv_metrics[(fold+1)][metric] = mean_squared_error(valid, preds)
            elif metric == 'mae':
                cv_metrics[(fold+1)][metric] = mean_absolute_error(valid, preds)
            elif metric == 'rmse':
                cv_metrics[(fold+1)][metric] = mean_squared_error(valid, preds, squared=False)
            elif metric == 'r2':
                cv_metrics[(fold+1)][metric] = r2_score(valid, preds)
            else:
                raise Exception(f'Invalid metric: {metric}! Only metrics are mse, mae, rmse, and r2.')
    return cv_metrics

def get_test_metrics(true, preds, metrics=['mse', 'mae', 'rmse', 'r2']):
    test_metrics = {}
    for metric in metrics:
        if metric == 'mse':
            test_metrics[metric] = mean_squared_error(true, preds)
        elif metric == 'mae':
            test_metrics[metric] = mean_absolute_error(true, preds)
        elif metric == 'rmse':
            test_metrics[metric] = mean_squared_error(true, preds, squared=False)
        elif metric == 'r2':
            test_metrics[metric] = r2_score(true, preds)
        else:
            raise Exception(f'Invalid metric: {metric}! Only metrics are mse, mae, rmse, and r2.') 
    return test_metrics

def STGCN_grid_search(metric, num_folds, param_grid, script):
    call(["python", script, '--metric', metric, '--folds', num_folds, 'grid_dict', str(param_grid)])

def STGCN_cv(num_folds, script):
    call(["python", script, '--folds', num_folds, 'cv', True])