import numpy as np
import pandas as pd


class LinearRegression:
    """
    Linear Regression using Gradient Descent.
    """

    def __init__(self, lr=0.01, n_iters=1000, fit_intercept=True, verbose=False, optimization_metric='mse'):
        self.lr = lr
        self.n_iters = n_iters
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.optimization_metric = optimization_metric

        self.w = None
        self.b = 0.0
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []
        self.metrics_history = []

        # validate hyperparams
        if lr <= 0:
            raise ValueError("Learning rate must be a positive number.")
        
        if n_iters <= 0 or not isinstance(n_iters, int):
            raise ValueError("Number of iterations must be a positive integer.")
        
        if fit_intercept not in [True, False]:
            raise ValueError("fit_intercept must be a boolean value.")
        
        if verbose not in [True, False]:
            raise ValueError("verbose must be a boolean value.")
        
        if self.optimization_metric not in ['mse', 'mae', 'r2']:
            raise ValueError("optimization_metric must be one of 'mse', 'mae', or 'r2'.")

    def fit(self, X, y):
        """
        Fit linear regression with gradient descent.
        """
        # convert into np array
        X, y = np.array(X), np.array(y)
        m, n = X.shape

        # validates
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        if X.shape[0] == 0:
            raise ValueError("Input X must have at least one sample.")
        if X.shape[1] == 0:
            raise ValueError("Input X must have at least one feature.")
        
        y = y.reshape(-1)
        if y.shape[0] != m or y.shape[0] == 0:
            raise ValueError("Input y must have the same number of samples as X. ")

        # initialize weights and bias
        self.w, self.b = np.zeros(n), 0.0
        
        # gradient descent loop
        for i in range(self.n_iters):
            # compute yhat
            y_hat = np.dot(X, self.w) + self.b

            # compute gradients dw and db based on defined cost function
            dw, db = self._compute_gradients(X, y, y_hat, metric=self.optimization_metric)

            # update weights and bias
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # metrics branching
            if self.optimization_metric == 'mse':
                loss = self._compute_mse(y_hat, y)
            elif self.optimization_metric == 'mae':
                loss = self._compute_mae(y_hat, y)
            elif self.optimization_metric == 'r2':
                loss = self._compute_r2(y_hat, y)
            else:
                raise ValueError(f"Unsupported optimization metric: {self.optimization_metric}")
            
            self.loss_history.append(loss)

            # compute metrics and store in history
            mse = self._compute_mse(y_hat, y)
            mae = self._compute_mae(y_hat, y)
            r2 = self._compute_r2(y_hat, y)
            self.metrics_history.append({'mse': mse, 'mae': mae, 'r2': r2})

            # verbose
            if self.verbose:
                if i == 0:
                    print("=" * 80)
                
                if (i + 1) % 100 == 0 or i == 0 or i == self.n_iters - 1:
                    print(f"Iteration {i+1}/{self.n_iters} | Loss: {loss:.6f} | MSE: {mse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
                
                if i == self.n_iters - 1:
                    print("=" * 80)
                    
        # set coef_ and intercept_
        self.coef_ = self.w
        if self.fit_intercept:
            self.intercept_ = self.b
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, X):
        """
        Predict target values for X.
        """
        # convert to np array
        X = np.array(X)
        m, n = X.shape

        # raise errors
        if self.w is None:
            raise ValueError("Model is not fitted yet. Please call 'fit' before 'predict'.")
        if n != self.w.shape[0]:
            raise ValueError(f"Input has {n} features, but model expects {self.w.shape[0]} features.")
        
        # return yhat
        y_hat = np.dot(X, self.w) + self.b
        return y_hat

    def score(self, X, y, metric='mse'):
        """
        Return the specified metric score (MSE, MAE, or R2).
        """
        # convert to np array
        metric_score = None
        X, y = np.array(X), np.array(y)

        # compute yhat
        y_hat = self.predict(X)

        # compute specified metric
        if metric == 'mse':
            metric_score = self._compute_mse(y_hat, y)
        elif metric == 'mae':
            metric_score = self._compute_mae(y_hat, y)      
        elif metric == 'r2':
            metric_score = self._compute_r2(y_hat, y)
        else:
            raise ValueError("Invalid metric specified. Choose from 'mse', 'mae', or 'r2'.")
        
        return metric_score

    def _compute_mse(self, y_hat, y):
        """
        Compute Mean Squared Error.
        """
        mse = np.mean((y_hat - y) ** 2)
        return mse

    def _compute_mae(self, y_hat, y):
        """
        Compute Mean Absolute Error.
        """
        mae = np.mean(np.abs(y_hat - y))
        return mae

    def _compute_r2(self, y_hat, y):
        """
        Compute R-squared score.
        """
        r2 = 1 - (np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2))
        return r2

    def _compute_gradients(self, X, y, y_hat, metric='mse'):
        """
        Compute gradients based on the optimization metric.
        """
        # compute error; y_hat - y
        error = y_hat - y

        # branch based on optimization_metric to set the gradient signal
        if metric == 'mse':
            gradient_signal = error
        elif metric == 'mae':
            gradient_signal = np.sign(error)
        elif metric == 'r2':
            gradient_signal = error
        else:
            raise ValueError(f"Unsupported metric for gradient: {metric}")

        # compute dw = gradient signal * features
        dw = (1 / X.shape[0]) * np.dot(X.T, gradient_signal)

        # compute db = gradient signal sum
        db = (1 / X.shape[0]) * np.sum(gradient_signal)

        return dw, db   

    def get_params(self):
        """
        Return model hyperparameters as a dict.
        """
        # GET: return a dict of hyperparameters
        return {
            'lr': self.lr,
            'n_iters': self.n_iters,
            'fit_intercept': self.fit_intercept,
            'verbose': self.verbose,
            'optimization_metric': self.optimization_metric
        }

    def set_params(self, **kwargs):
        """
        Set model hyperparameters from kwargs.
        """
        # SET: update hyperparameters and validate
        if 'lr' in kwargs:
            if not isinstance(kwargs['lr'], (float, int)) or kwargs['lr'] <= 0:
                raise ValueError("Learning rate must be a numeric positive number.")
            self.lr = kwargs['lr']

        if 'n_iters' in kwargs:
            if not isinstance(kwargs['n_iters'], int) or kwargs['n_iters'] <= 0:
                raise ValueError("Number of iterations must be a positive integer.")
            self.n_iters = kwargs['n_iters']

        if 'fit_intercept' in kwargs:
            if not isinstance(kwargs['fit_intercept'], bool):
                raise ValueError("fit_intercept must be a boolean value.")
            self.fit_intercept = kwargs['fit_intercept']

        if 'verbose' in kwargs:
            if not isinstance(kwargs['verbose'], bool):
                raise ValueError("verbose must be a boolean value.")
            self.verbose = kwargs['verbose']

        if 'optimization_metric' in kwargs:
            if not isinstance(kwargs['optimization_metric'], str):
                raise ValueError("optimization_metric must be a string.")
        
            if kwargs['optimization_metric'] not in ['mse', 'mae', 'r2']:
                raise ValueError("optimization_metric must be one of 'mse', 'mae', or 'r2'.")
            self.optimization_metric = kwargs['optimization_metric']

        return self