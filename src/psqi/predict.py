import json
import os
from time import time

import numpy as np
import pandas as pd
import torch
import utm
from ipypb import track
from psqi import RESULTS_DIR, DEVICE
from scipy.stats.distributions import norm
from shapely.geometry import shape, Point


def parse_geojson(path, *, X_scaler):
    with open(path, 'r') as fin:
        geojson = json.load(fin)
        for polygon in geojson['geometries'][0]['coordinates']:
            for pair in polygon[0]:
                x, y = utm.from_latlon(*pair[::-1])[:2]
                x, y = X_scaler.transform([[x, y]])[0]
                pair[0] = x
                pair[1] = y
        return shape(geojson)


def compute_test_points(cell_size_m, *, shp, X_scaler):
    w, h = (
        X_scaler.inverse_transform([[1, 1]]) -
        X_scaler.inverse_transform([[0, 0]])
    ).flatten()
    print('Region %.2f km X %.2f km' % (w / 1000, h / 1000))

    Nx = int(w / cell_size_m)
    Ny = int(h / cell_size_m)

    x1 = np.linspace(0, 1, Nx + 1)
    x2 = np.linspace(0, 1, Ny + 1)
    x1, x2 = np.meshgrid(x1, x2)
    test_X_shape = (Ny + 1, Nx + 1)
    test_X = np.vstack((x1.flatten(), x2.flatten())).T

    print('Number of points before shape filtering %d' % len(test_X))

    t = -time()
    shp.contains(Point(*test_X[0]))
    t += time()

    t *= len(test_X)
    print('Applying shape will take %.1f seconds' % t)

    test_X_mask = np.array(
        list(
            map(
                lambda x: shp.contains(Point(*x)),
                test_X,
            )
        )
    )
    print('Number of points after shape filtering %d' % np.sum(test_X_mask))
    print('Kept %.1f %% of points' % (np.sum(test_X_mask) / len(test_X) * 100))

    test_X_orig = np.apply_along_axis(
        lambda xy: utm.to_latlon(*xy, 37, 'U'), 1,
        X_scaler.inverse_transform(test_X),
    )

    path = os.path.join(RESULTS_DIR, 'test_coords_%dm.npz' % cell_size_m)
    print('Saving test points to %s (use "read_test_points" to avoid recomputation)' % path)
    np.savez(
        path,
        scaled=test_X.reshape(test_X_shape + (2,)),
        orig=test_X_orig.reshape(test_X_shape + (2,)),
        mask=test_X_mask.reshape(test_X_shape),
    )
    return torch.tensor(test_X[test_X_mask], device=DEVICE, dtype=torch.float32)


def read_test_points(cell_size_m):
    try:
        fin = np.load(
            os.path.join(RESULTS_DIR, 'test_coords_%dm.npz' % cell_size_m)
        )
        test_X, test_X_mask = (
            fin['scaled'].reshape((-1, 2)), fin['mask'].flatten(),
        )
        return torch.tensor(
            test_X[test_X_mask], device=DEVICE, dtype=torch.float32,
        )
    finally:
        fin.close()



def compute_predictions(*, model, likelihood, test_X):
    t = -time()
    with torch.no_grad():
        batch = test_X[0:1]
        pred = likelihood(model(batch))
        del pred
    t += time()

    print('Predictions will take %.1f minutes' % (t * len(test_X) / 60))

    t = -time()
    u = []
    s = []
    with torch.no_grad():
        n = len(test_X)
        for i in track(range(n), total=n):
            batch = test_X[i:i + 1]
            pred = likelihood(model(batch))
            u.append(pred.mean.detach().cpu().numpy())
            s.append(pred.covariance_matrix.detach().cpu().numpy())
            del pred
    u = np.concatenate(u, axis=0)
    s = np.stack(s, axis=0)
    t += time()

    return u, s


def _percentile(mean, std):
    mean, std = mean[:, 0], std[:, 0]
    ret = []
    for (u, s) in zip(mean, std):
        ppf = norm(u, s).ppf
        p1 = ppf(0.01)
        p99 = ppf(0.99)
        ret.append((p1, p99))
    return np.array(ret)


def save_predictions(*, u, s, test_X, X_scaler, Y_scaler, y_columns):
    N = len(u)
    test_X = test_X.detach().cpu().numpy()
    x = X_scaler.inverse_transform(test_X)
    x = np.apply_along_axis(lambda xy: utm.to_latlon(*xy, 37, 'U'), 1, x)

    u_scaled = u.copy()
    s_scaled = np.array([np.sqrt(np.diag(cov)) for cov in s])
    u_original = Y_scaler.inverse_transform(u)

    p1 = np.zeros((N, 0))
    p99 = np.zeros((N, 0))

    dfs = []
    for i, col in enumerate(y_columns):
        col = {'Ions': 'Hardness'}.get(col, col)
        path = os.path.join(RESULTS_DIR, 'predict_%s.csv' % col)
        mean = u_original[:, i:i + 1]
        mean_scaled = u_scaled[:, i:i + 1]
        std_scaled = s_scaled[:, i:i + 1]
        perc = _percentile(mean_scaled, std_scaled)
        p1 = np.hstack((p1, perc[:, 0:1]))
        p99 = np.hstack((p99, perc[:, 1:]))

        df = pd.DataFrame(
            np.hstack((x, mean)),
            columns=['lat', 'lon', 'mean']
        )
        dfs.append((df, path))

    p1 = Y_scaler.inverse_transform(p1)
    p99 = Y_scaler.inverse_transform(p99)

    print('Saving predictions (lat, lon, mean, p1, p99)')
    for i, col in enumerate(y_columns):
        df, path = dfs[i]
        df['p1'] = p1[:, i]
        df['p99'] = p99[:, i]
        df.to_csv(path)
        print('Saved predictions for %s to %s' % (col, path))

    df = pd.DataFrame(
        np.hstack((x, u_original)),
        columns=['lat', 'lon'] + list(y_columns)
    )

    return df