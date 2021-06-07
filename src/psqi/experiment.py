import os
from collections import defaultdict
from time import time

import dill as pickle
import gpytorch
import numpy as np
import pandas as pd
import torch
from psqi import DEVICE
from psqi.model import get_model
from psqi.train import SimpleTrainer
from sklearn.model_selection import KFold
from torch import tensor


def get_key(rank, num_mixtures, n_splits, max_iter, split):
    ret = f'rank_{rank}_mixtures_{num_mixtures}_iters_{max_iter}_splits_{n_splits}'
    if split != None:
        ret += f'_i{split}'
    return ret

def get_dir(**kwargs):
    return os.path.join('results', get_key(**kwargs))


def _save_results(
        *,
        train_ix, test_ix,
        split, n_splits, max_iter,
        rank, num_mixtures,
        losses,
        mse, mse_orig, var, var_orig,
        mse_train, mse_train_orig, var_train, var_train_orig,
        time_train, time_eval,
        model, likelihood,
):
    key = get_key(
        rank=rank, num_mixtures=num_mixtures,
        max_iter=max_iter, n_splits=n_splits, split=split,
    )
    dr = os.path.join('results', key)
    os.makedirs(dr, exist_ok=True)

    ix_path = os.path.join(dr, 'ix.pickle')
    with open(ix_path, 'wb') as fout:
        pickle.dump(
            obj={
                'train_ix': train_ix,
                'test_ix': test_ix,
            }, file=fout,
        )

    results_path = os.path.join(dr, 'results.pickle')
    with open(results_path, 'wb') as fout:
        pickle.dump(obj={
            'loss': losses,
            'mse': mse,
            'mse_orig': mse_orig,
            'var': var,
            'var_orig': var_orig,
            'mse_train': mse_train,
            'mse_train_orig': mse_train_orig,
            'var_train': var_train,
            'var_train_orig': var_train_orig,
            'R2': 1.0 - mse / var,
            'R2_orig': 1.0 - mse_orig / var_orig,
            'R2_train': 1.0 - mse_train / var_train,
            'R2_train_orig': 1.0 - mse_train_orig / var_train_orig,
            'time_train': time_train,
            'time_eval': time_eval,
        }, file=fout)

    model_path = os.path.join(dr, 'model.pickle')
    torch.save(
        {'model': model, 'likelihood': likelihood},
        model_path,
        pickle_module=pickle
    )
    print('%d split: saving results to directory %s' % (split, dr))

def _evaluate_batch(*, model, inputs, outputs, Y_scaler):
    pred = model(inputs)
    mean = pred.mean.cpu().numpy()
    outputs = outputs.cpu().numpy()

    mean_orig = Y_scaler.inverse_transform(mean)
    outputs_orig = Y_scaler.inverse_transform(outputs)

    error = ((mean - outputs) ** 2).sum(axis=0)
    error_orig = ((mean_orig - outputs_orig) ** 2).sum(axis=0)

    del pred
    return error, error_orig


def _evaluate(*, model, test_x, test_y, Y_scaler, batch=32):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        n = len(test_x)
        iters = int(n // batch + (0 if n % batch == 0 else 1))
        se = np.zeros(test_y.shape[1])
        se_orig = np.zeros(test_y.shape[1])
        for i in range(iters):
            i1, i2 = i * batch, (i + 1) * batch
            e1, e2 = _evaluate_batch(
                model=model,
                inputs=test_x[i1:i2],
                outputs=test_y[i1:i2],
                Y_scaler=Y_scaler,
            )
            se += e1
            se_orig += e2

        test_y = test_y.cpu().numpy()
        test_y_orig = Y_scaler.inverse_transform(test_y)
        var = test_y.var(axis=0)
        var_orig = test_y_orig.var(axis=0)

        return se / n, se_orig / n, var, var_orig


def run_experiment(
        *, X, Y, rank, num_mixtures, n_splits, max_iter, random_state,
        X_scaler, Y_scaler,
):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (train_ix, test_ix) in enumerate(kfold.split(X)):
        train_x = tensor(X[train_ix], device=DEVICE, dtype=torch.float32)
        train_y = tensor(Y[train_ix], device=DEVICE, dtype=torch.float32)
        test_x = tensor(X[test_ix], device=DEVICE, dtype=torch.float32)
        test_y = tensor(Y[test_ix], device=DEVICE, dtype=torch.float32)

        model, likelihood = get_model(
            train_x=train_x, train_y=train_y,
            rank=rank, num_mixtures=num_mixtures, X_scaler=X_scaler,
        )
        trainer = SimpleTrainer(n_iter=max_iter)

        time_train = -time()
        losses = trainer.train(model, likelihood, train_x, train_y)
        time_train += time()

        model.eval()
        likelihood.eval()

        time_eval = -time()
        mse, mse_orig, var, var_orig = _evaluate(
            model=model, test_x=test_x, test_y=test_y, Y_scaler=Y_scaler,
        )
        time_eval += time()

        mse_train, mse_train_orig, var_train, var_train_orig = _evaluate(
            model=model, test_x=train_x, test_y=train_y, Y_scaler=Y_scaler,
        )

        _save_results(
            train_ix=train_ix, test_ix=train_ix,
            split=i, n_splits=n_splits, max_iter=max_iter,
            rank=rank, num_mixtures=num_mixtures,
            losses=losses,
            mse=mse, mse_orig=mse_orig, var=var, var_orig=var_orig,
            mse_train=mse_train, mse_train_orig=mse_train_orig,
            var_train=var_train, var_train_orig=var_train_orig,
            time_train=time_train, time_eval=time_eval,
            model=model, likelihood=likelihood,
        )

        # Free memory
        del train_x
        del train_y
        del test_x
        del test_y
        del model
        del likelihood


def read_results_and_models(params):
    results = defaultdict(list)
    models = defaultdict(list)
    for kwargs in params:
        for split in range(kwargs['n_splits']):
            dr = get_dir(**kwargs, split=split)
            key = get_key(**kwargs, split=None)
            with open(os.path.join(dr, 'results.pickle'), 'rb') as fin:
                results[key].append(pickle.load(file=fin))
            model = torch.load(os.path.join(dr, 'model.pickle'), pickle_module=pickle)
            models[key].append(model)
    return results, models


def get_results_R2_per_params(*, R2, params, y_columns):
    R2_per_params = np.array([v.mean(axis=0) for v in R2.values()])
    results_R2_per_params = pd.DataFrame()
    results_R2_per_params['El'] = y_columns
    for i, kwargs in enumerate(params):
        rank = kwargs['rank']
        mix = kwargs['num_mixtures']
        col = f'r={rank}, Q={mix}'
        results_R2_per_params[col] = R2_per_params[i]
    return results_R2_per_params
