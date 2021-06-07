from abc import ABC, abstractmethod

import torch
from gpytorch import ExactMarginalLogLikelihood
from psqi import DEVICE
from torch import tensor


class Trainer(ABC):

    def __init__(self):
        self.losses = []

    @abstractmethod
    def _train(self, model, likelihood, train_x, train_y):
        pass

    def train(self, model, likelihood, train_x, train_y):
        self.losses.clear()
        return self._train(model, likelihood, train_x, train_y)


class SimpleTrainer(Trainer):

    def __init__(self, lr=0.1, n_iter=50, step_callback=None):
        super().__init__()
        self.step_callback = step_callback or self.default_step_callback
        self.lr = lr
        self.n_iter = n_iter

    def default_step_callback(self, i, loss, optimizer):
        self.losses.append(loss.item())

    def _train(self, model, likelihood, train_x, train_y):
        # Find optimal model hyper-parameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(
            # Includes GaussianLikelihood parameters
            [{'params': model.parameters()}],
            lr=self.lr
        )

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood, model)

        losses = []

        for i in range(self.n_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if self.step_callback is not None:
                self.step_callback(i, loss, optimizer)
            if len(losses) > 1 and losses[-1] > losses[-2]:
                self.lr *= 0.1
            if self.lr < 1e-6:
                break

        return losses


def post_train_model(X, Y, model, likelihood, max_iter):
    train_x = tensor(X, device=DEVICE, dtype=torch.float32)
    train_y = tensor(Y, device=DEVICE, dtype=torch.float32)

    for param in model.parameters():
        param.requires_grad = False

    model.set_train_data(inputs=train_x, targets=train_y, strict=False)
    model.eval()
    likelihood.eval()
