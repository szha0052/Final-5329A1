import numpy as np
from MLP.layers.Linear import Linear

class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        self.eps = eps
        self.momentum = momentum

        # Parameters gamma and beta, used for scaling and shifting
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

        # Store mean and variance during training
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

        # Cache variables
        self.x_norm = None
        self.mean = None
        self.var = None
        self.x_centered = None

    def forward(self, x, training=True):
        if training:
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0)

            # Normalization
            self.x_centered = x - self.mean
            self.x_norm = self.x_centered / np.sqrt(self.var + self.eps)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            # Use running mean and variance during testing
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scaling and shifting
        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, grad):
        batch_size = grad.shape[0]
        # Gradients of parameters
        dgamma = np.sum(grad * self.x_norm, axis=0)
        dbeta = np.sum(grad, axis=0)

        # Gradient of input
        dx_norm = grad * self.gamma
        dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.var + self.eps)**(-1.5), axis=0)
        # dmean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=0) + dvar * np.sum(-2 * self.x_centered, axis=0) / batch_size

        dx = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * self.x_centered / batch_size + dmean / batch_size

        # Update parameters
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

