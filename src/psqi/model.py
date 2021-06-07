import numpy as np
import torch
import torch.nn
from gpytorch.constraints import Interval
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.kernels import SpectralMixtureKernel, MultitaskKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import Mean, MultitaskMean
from gpytorch.models import ExactGP
from psqi import DEVICE, IS_CUDA
from scipy.sparse.linalg import eigs
from torch import tensor


class Square2DPolynomialMean(Mean):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_parameter(
            name="coeffs", parameter=torch.nn.Parameter(torch.zeros(6))
        )

    def forward(self, input):
        assert len(input.shape) == 2
        assert input.shape[1] == 2
        n = input.shape[0]
        coeffs = self.coeffs.expand((n, 6))
        return (
            coeffs[:, 0] +
            coeffs[:, 1] * input[:, 0] + coeffs[:, 2] * input[:, 1] +
            coeffs[:, 3] * input[:, 0] * input[:, 1] +
            coeffs[:, 4] * input[:, 0] * input[:, 0] +
            coeffs[:, 5] * input[:, 1] * input[:, 1]
        )


class PatchedSpectralMixtureKernel(SpectralMixtureKernel):
    def initialize_from_data(self, train_x, train_y, **kwargs):
        if not torch.is_tensor(train_x) or not torch.is_tensor(train_y):
            raise RuntimeError("train_x and train_y should be tensors")
        if train_x.ndimension() == 1:
            train_x = train_x.unsqueeze(-1)
        if train_x.ndimension() == 2:
            train_x = train_x.unsqueeze(0)

        train_x_sort = train_x.sort(1)[0]
        max_dist = train_x_sort[:, -1, :] - train_x_sort[:, 0, :]
        min_dist_sort = (
        train_x_sort[:, 1:, :] - train_x_sort[:, :-1, :]).squeeze(0)
        min_dist = torch.zeros(1, self.ard_num_dims, dtype=train_x.dtype, device=train_x.device)
        for ind in range(self.ard_num_dims):
            min_dist[:, ind] = min_dist_sort[
                ((min_dist_sort[:, ind]).nonzero())[0], ind]

        # Inverse of lengthscales should be drawn from truncated Gaussian | N(0, max_dist^2) |
        self.raw_mixture_scales.data.normal_().mul_(max_dist).abs_()

        ub = self.raw_mixture_scales_constraint.lower_bound.reshape((1, 1, -1)) ** (-1)
        lb = self.raw_mixture_scales_constraint.upper_bound.reshape((1, 1, -1)) ** (-1)
        self.raw_mixture_scales.data = torch.min(
            torch.max(self.raw_mixture_scales.data, lb),
            ub
        )
        self.raw_mixture_scales.data = self.raw_mixture_scales_constraint.inverse_transform(
            self.raw_mixture_scales.data.pow_(-1)
        )
        # Draw means from Unif(0, 0.5 / minimum distance between two points)
        self.raw_mixture_means.data.uniform_().mul_(0.5).div_(min_dist)

        lb = self.raw_mixture_means_constraint.lower_bound.reshape((1, 1, -1))
        ub = self.raw_mixture_means_constraint.upper_bound.reshape((1, 1, -1))
        self.raw_mixture_means.data = torch.min(
            torch.max(self.raw_mixture_means.data, lb),
            ub,
        )
        self.raw_mixture_means.data = self.raw_mixture_means_constraint.inverse_transform(
            self.raw_mixture_means.data,
        )
        # Mixture weights should be roughly the std of the y values divided by the number of mixtures
        self.raw_mixture_weights.data.fill_(train_y.std() / self.num_mixtures)
        self.raw_mixture_weights.data = self.raw_mixture_weights_constraint.inverse_transform(
            self.raw_mixture_weights.data
        )


from collections import OrderedDict


class MultitaskGPModel(ExactGP):
    def __init__(
            self,
            train_x, train_y, likelihood,
            rank, num_mixtures, X_scaler,
    ):
        super().__init__(train_x, train_y, likelihood)
        num_dims = train_x.shape[1]
        num_tasks = train_y.shape[1]

        self.mean_module = MultitaskMean(
            Square2DPolynomialMean(),
            num_tasks=num_tasks
        )

        xmax = X_scaler.inverse_transform(np.ones((1, 2)))[0]
        xmin = X_scaler.inverse_transform(np.zeros((1, 2)))[0]
        llcoeff = (xmax - xmin)[0] / (xmax - xmin)[1]

        lower_ll_mean = 0.1
        upper_ll_mean = 1e2
        lower_ll_scale = 0.1
        upper_ll_scale = 1e2
        lower_mean_constraint = np.array([
            1. / upper_ll_mean, 1. / (upper_ll_mean * llcoeff)
        ])
        upper_mean_constraint = np.array([
            1. / lower_ll_mean, 1. / (lower_ll_mean * llcoeff)
        ])
        lower_scale_constraint = np.array([
            1. / upper_ll_scale, 1. / (upper_ll_scale * llcoeff)
        ])
        upper_scale_constraint = np.array([
            1. / lower_ll_scale, 1. / (lower_ll_scale * llcoeff)
        ])

        lower_mean_constraint = tensor(
            lower_mean_constraint, dtype=torch.float32, device=DEVICE,
        )
        upper_mean_constraint = tensor(
            upper_mean_constraint, dtype=torch.float32, device=DEVICE,
        )
        lower_scale_constraint = tensor(
            lower_scale_constraint, dtype=torch.float32, device=DEVICE,
        )
        upper_scale_constraint = tensor(
            upper_scale_constraint, dtype=torch.float32, device=DEVICE,
        )

        covar_module = PatchedSpectralMixtureKernel(
            num_mixtures,
            ard_num_dims=num_dims,
            mixture_scales_constraint=Interval(
                lower_scale_constraint, upper_scale_constraint,
            ),
            mixture_means_constraint=Interval(
                lower_mean_constraint, upper_mean_constraint,
            ),
        )
        if IS_CUDA:
            covar_module = covar_module.cuda(device=DEVICE)

        covar_module.initialize_from_data(train_x, train_y)
        self.covar_module = MultitaskKernel(
            covar_module,
            num_tasks=num_tasks,
            rank=rank,
        )
        self.init_multitask_from_data(Y=train_y, rank=rank)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)

    def init_multitask_from_data(self, *, Y, rank):
        Y = Y.detach().cpu().numpy()
        Cov = np.cov(Y.T)

        w = float(self.covar_module.data_covar_module.mixture_weights.sum().detach().cpu().numpy())

        l, Q = eigs(Cov, k=rank)
        B = Q.dot(np.diag(np.sqrt(l / w))).real

        B = tensor(B, dtype=torch.float32, requires_grad=True)

        state_dict = OrderedDict({'covar_factor': B})
        self.covar_module.task_covar_module.load_state_dict(state_dict, strict=False)
        # Add small diagonal noise
        self.covar_module.task_covar_module._set_var(
            self.covar_module.task_covar_module.var * 1e-5,
        )


def get_model(*, train_x, train_y, rank, num_mixtures, X_scaler):
    likelihood = MultitaskGaussianLikelihood(
        num_tasks=train_y.shape[1],
        noise_constraint=Interval(1e-10, 1.0)
    )
    model = MultitaskGPModel(
        train_x, train_y, likelihood,
        rank=rank, num_mixtures=num_mixtures,
        X_scaler=X_scaler,
    )
    if IS_CUDA:
        model = model.cuda(device=DEVICE)
        likelihood = likelihood.cuda(device=DEVICE)

    return model, likelihood