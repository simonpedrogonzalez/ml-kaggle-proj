import sys, os; sys.path.insert(0, os.path.abspath('.')) if os.path.abspath('.') not in sys.path else None
from linear_regression.linear_regressor import LinearRegressor
from utils.stats import endless_batch_generator, mse, cost
import numpy as np

class BatchGradientDescent(LinearRegressor):

    def __init__(self, lr=0.01, atol=1e-6, batch_size=None, max_iter=1000):
        self.lr = lr
        self.atol = atol
        self.max_iter = max_iter
        self.n_iter = 0
        self.batch_size = batch_size
        self.batch_generator = None
        self.last_w = None
        super().__init__()

    def _weight_difference(self):
        return np.linalg.norm(self.w - self.last_w)
    
    def _cost_difference(self):
        return abs(mse(self.X @ self.last_w, self.y) - mse(self.X @ self.w, self.y))

    def converged(self):
        return self._weight_difference() < self.atol
    
    def _reached_max_iter(self):
        return self.n_iter >= self.max_iter if self.max_iter is not None else False

    def should_stop(self):
        # Stop if max_iter reached or converged
        # can disable max_iter by setting it to None
        return self.n_iter > 0 and (self._reached_max_iter() or self.converged())
    
    def _fit(self):
        self.n_iter = 0
        if self.batch_size is None: # Bootstrap
            self.batch_size = self.X.shape[0]

        self.batch_generator = endless_batch_generator(self.X, self.y, self.batch_size, random=False)

        while True:
            self.step()
            if self.should_stop():
                break
        
        return self
    
    def step(self):
        self.last_w = self.w
        X_batch, y_batch = next(self.batch_generator)
        w, lr = self.w, self.lr
        # print(f"x={X_batch}")
        y_pred = X_batch @ w
        error = y_pred - y_batch
        self.w = w - lr * ((X_batch.T @ error) / len(y_batch))
        if np.isnan(self.w).any():
             # Obviously diverged
             # TODO: create an earlier stopping condition
            raise ValueError("Learning rate too high")
        # print(f"w={self.w}")
        self.n_iter += 1 # keep count of the iters
        return self

class StochasticGradientDescent(BatchGradientDescent):

    def __init__(self, lr=0.01, atol=1e-6, max_iter=1000):
        # Do a BGD with batch_size=1 is equivalent to SGD
        super().__init__(lr=lr, atol=atol, max_iter=max_iter, batch_size=1)

    