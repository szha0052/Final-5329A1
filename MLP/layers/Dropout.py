import numpy as np

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio  # Dropout ratio
        self.mask = None  # Mask is an array with the same shape as the input x, containing True/False values indicating whether the corresponding element is dropped

    def forward(self, x, train_flg=True):
        if train_flg:  # TRUE indicates training mode, randomly dropping some neurons to prevent overfitting
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio  # Generate an array with the same shape as x, containing random numbers between 0 and 1; True if greater than dropout_ratio
            return x * self.mask  # Retain elements where mask is True
        else:
            # return x * (1.0 - self.dropout_ratio)  # During testing, scale the output neurons without randomly dropping them
            return x if train_flg else x * (1.0 / (1.0 - self.dropout_ratio))

    def backward(self, dout):
        a = dout * self.mask
        return dout * self.mask  # Retain gradients only for neurons where mask is True, set others to 0
