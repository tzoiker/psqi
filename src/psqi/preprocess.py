import os

import dill as pickle
import numpy as np
import pandas as pd
import utm
from psqi import RESULTS_DIR, DATA_DIR
from psqi.scale import NormalCDFScaler
from sklearn.preprocessing import MinMaxScaler


def parse_raw_data():
    ignore = ['Hg', 'Cd', 'Co', 'Pb']
    path = os.path.join(DATA_DIR, 'raw_data.csv')
    df_all = pd.read_csv(path, parse_dates=['Date']).dropna()
    df = df_all[list(set(df_all.columns).difference(ignore))]
    df = df.reset_index(drop=True)
    df['Cu'] = df['Cu'].abs()

    x_columns = ['X1', 'X2']
    y_columns = sorted(list(set(df.columns).difference(['X1', 'X2', 'Date', 'Class'])))

    df_Y = df[y_columns]
    df_X = df[x_columns].apply(
        lambda x: utm.from_latlon(*x)[:2],
        axis=1,
        result_type='expand'
    ).rename({0: 'x', 1: 'y'}, axis=1)

    # Filter outliers
    x_min = df_X.min()
    x_max = df_X.max()
    df_X = df_X[(df_X['x'] > 362000) & (df_X['x'] < x_max[0])]
    df_X = df_X[(df_X['y'] > x_min[1]) & (df_X['x'] < x_max[1])]
    df_X = df_X[(df_X['x'] <= 409000)]
    df_Y = df_Y.loc[df_X.index]

    path = os.path.join(DATA_DIR, 'norms.csv')
    norms = pd.read_csv(path, sep=' ', index_col='el').loc[df_Y.columns]

    return df_all, df_X, df_Y, norms

def _get_y_limits(df_Y, norms):
    limits = np.array([
        np.inf if col != 'pH' else 14
        for col in df_Y.columns
    ])
    mx = np.vstack((norms['to'].fillna(-np.inf), df_Y.max(axis=0))).max(axis=0)
    inds = np.where(limits == np.inf)[0]
    limits[inds] = mx[inds]*10
    return limits

def scale_data(df_X, df_Y, norms):
    X = np.array(df_X)
    Y = np.array(df_Y)

    X_scaler = MinMaxScaler()
    X_scaler.fit(X)

    Y_limits = _get_y_limits(df_Y=df_Y, norms=norms)
    Y_scaler = NormalCDFScaler(Y_limits)
    Y_scaler.fit(Y)

    X = X_scaler.transform(X)
    Y = Y_scaler.transform(Y)

    scaled_norms = Y_scaler.transform(np.array(norms).T).T
    scaled_norms[np.isnan(scaled_norms)] = np.inf

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'data.pickle')
    with open(path, 'wb') as fout:
        pickle.dump(file=fout, obj={
            'X_scaler': X_scaler, 'Y_scaler': Y_scaler,
            'X': X, 'Y': Y,
            'scaled_norms': scaled_norms,
        })
    print('Saved scaled data to %s' % path)

    return X, Y, X_scaler, Y_scaler, scaled_norms
