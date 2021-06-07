import numpy as np
from scipy.stats.distributions import norm
from sklearn.preprocessing import MinMaxScaler


class NormalCDFScaler:
    def __init__(self, limits):
        self._fit_inner = False
        self.limits = np.array(limits)
        self.cdf = norm(0, 1).cdf
        self.icdf = norm(0, 1).ppf
        self.scaler = MinMaxScaler()

    def fit(self, X):
        mx = X.max(axis=0)
        inds = np.where(self.limits == np.inf)[0]
        self.limits[inds] = mx[inds]
        self.limits = self.limits.reshape((1, -1))

    def transform(self, X):
        X = X / self.limits
        np.place(X, np.isclose(X, 0), 1e-10)
        X = self.icdf(X)
        if self._fit_inner:
            X = self.scaler.transform(X)
        else:
            X = self.scaler.fit_transform(X)
            self._fit_inner = True
        return X

    def inverse_transform(self, X):
        X = self.scaler.inverse_transform(X)
        X = self.cdf(X)
        return X * self.limits
