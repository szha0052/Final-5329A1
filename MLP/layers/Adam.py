import numpy as np
from MLP.layers.Linear import Linear
from MLP.layers.BatchNorm import BatchNorm

class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer
        :param beta1: Exponential decay rate for the first moment estimates
        :param beta2: Exponential decay rate for the second moment estimates
        :param epsilon: A small value to prevent division by zero
        """
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step

    def increment_time_step(self):
        """
        Call this at the beginning of each epoch to update the time step
        """
        self.t += 1

    def update(self, layers, lr, weight_decay):
        """
        Update the parameters of the given layers
        :param layers: List of layers to update
        :param lr: Learning rate
        :param weight_decay: Weight decay (L2 regularization)
        """
        for i, layer in enumerate(layers):
            if isinstance(layer, Linear):
                # Initialize first and second moments
                if i not in self.m:
                    self.m[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}
                    self.v[i] = {'W': np.zeros_like(layer.W), 'b': np.zeros_like(layer.b)}

                # Compute gradients
                grad_W = layer.dW + weight_decay * layer.W
                grad_b = layer.db

                # Update first and second moments
                self.m[i]['W'] = self.beta1 * self.m[i]['W'] + (1 - self.beta1) * grad_W
                self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * grad_b
                self.v[i]['W'] = self.beta2 * self.v[i]['W'] + (1 - self.beta2) * (grad_W ** 2)
                self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * (grad_b ** 2)

                # Compute bias-corrected moments
                m_hat_W = self.m[i]['W'] / (1 - self.beta1 ** self.t)
                m_hat_b = self.m[i]['b'] / (1 - self.beta1 ** self.t)
                v_hat_W = self.v[i]['W'] / (1 - self.beta2 ** self.t)
                v_hat_b = self.v[i]['b'] / (1 - self.beta2 ** self.t)

                # Update parameters
                layer.W -= lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
                layer.b -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

            elif isinstance(layer, BatchNorm):
                # Update gamma and beta parameters of BatchNorm
                layer.gamma -= lr * layer.dgamma
                layer.beta -= lr * layer.dbeta