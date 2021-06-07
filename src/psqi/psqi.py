import os
from time import time

import numpy as np
import pandas as pd
import utm
from ipypb import track
from psqi import RESULTS_DIR
from scipy.special import softmax
from scipy.stats.mvn import mvnun


def _calc_psqi(mean, cov, norms, R2, train_mean, train_var):
    mean = mean.copy()
    cov = cov.copy()

    # Set covariances for poor-predicted elements to zeros and dataset values
    inds = np.where(R2 < 0)[0]
    mean[inds] = train_mean[inds]
    cov[inds, :] = cov[:, inds] = 0
    assert (cov[:, inds][inds, :] == 0).all()
    cov[inds, inds] = train_var[inds]

    ps = []
    cs = []
    for i in range(len(norms)):
        lower = -np.inf * np.ones_like(norms[:, 0])
        upper = np.inf * np.ones_like(norms[:, 0])
        lower[i] = norms[i, 0]
        upper[i] = norms[i, 1]
        p = mvnun(lower, upper, mean, cov)[0]
        c = mvnun(lower, upper, (upper + lower) / 2, cov)[0]
        ps.append(p)
        cs.append(c)
    return np.array(ps), np.array(cs)


def calc_psqi(*, u, s, norms, R2, df_Y):
    # Consider only elements with bounded norms
    inds = [i for i, (lower, upper) in enumerate(norms) if np.isinf(upper)]
    cols = [col for i, col in enumerate(df_Y.columns) if i in inds]

    print('PSQI: consider only %s' % cols)

    # Average R^2 across splits
    R2 = R2.mean(axis=0).round(3)

    u = u[:, inds]
    s = s[:, :, inds][:, inds, :]
    R2 = R2[inds]
    df_Y = df_Y.iloc[:, inds]
    norms = norms[inds]

    assert len(u.shape) == 2
    assert len(s.shape) == 3
    ret_ps = []
    ret_cs = []

    train_mean = df_Y.mean(axis=0)
    train_var = df_Y.var(axis=0)

    ws = softmax(np.clip(R2, 0, None))

    t = -time()
    for i, (mean, cov) in track(enumerate(zip(u, s)), total=len(u)):
        if i % 1000 == 0:
            print('PSQI: processed %d / %d' % (i, len(u)))
        ps, cs = _calc_psqi(
            mean, cov, norms,
            R2=R2, train_mean=train_mean, train_var=train_var,
        )
        ret_ps.append(ps)
        ret_cs.append(cs)
    t += time()
    print('PSQI: computations took %.1f seconds' % t)
    return np.array(ret_ps), np.array(ret_cs), ws


def save_psqi(*, psqi, conf, ws, test_X, X_scaler):
    path = os.path.join(RESULTS_DIR, 'psqi.csv')
    x_scaled = test_X.detach().cpu().numpy()
    x = X_scaler.inverse_transform(x_scaled)
    x = np.apply_along_axis(lambda xy: utm.to_latlon(*xy, 37, 'U'), 1, x)
    ws = ws.reshape((-1, 1))
    psqi_no_weights = psqi.mean(axis=1, keepdims=True)
    psqi = psqi.dot(ws)
    conf_no_weights = conf.mean(axis=1, keepdims=True)
    conf = conf.dot(ws)
    df = pd.DataFrame(
        np.hstack((x, psqi, psqi_no_weights, conf, conf_no_weights)),
        columns=['lat', 'lon', 'PSQI', 'PSQI_no_weights', 'conf', 'conf_no_weights'],
    )
    df.to_csv(path)
    print('Saved PSQI data to %s' % path)
    return df
