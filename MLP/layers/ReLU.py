class ReLU:
    def __init__(self):
        self.activated = None    # Records the output after ReLU activation
        self.mask = None         # Mask is TRUE where x <= 0

    def forward(self, x):
        self.mask = (x <= 0)  # Records positions where x is less than or equal to 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, grad):
        # dout[self.mask] = 0    # Mask is TRUE where x <= 0, set the gradients at these positions to 0
        # dx = dout # Only inputs greater than 0 in the forward pass will propagate gradients
        dx = grad.copy()
        dx[self.mask] = 0
        return dx