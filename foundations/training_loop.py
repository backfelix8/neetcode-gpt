import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        
        in_shape = X.shape
        n = in_shape[0]
        w = np.zeros(in_shape[1])
        b = 0.0

        for i in range(epochs):
            # 1. forward
            y_hat = np.dot(X, w) + b
            error = y_hat - y

            # 2. backward
            d_w = 2/n * np.dot(np.transpose(X), y_hat - y)
            d_b = 2/n * np.sum(y_hat - y)

            # 3. update
            w = w - lr * d_w
            b = b - lr * d_b
        
        return (np.round(w, 5), np.round(b, 5))
