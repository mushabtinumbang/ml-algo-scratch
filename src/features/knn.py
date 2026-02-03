import numpy as np


class KNNClassifier:
    """
    K-Nearest Neighbors (KNN) classifier from scratch.
    """

    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

        # validate hyperparameters
        if not isinstance(self.k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")
        
        if self.distance_metric != "euclidean":
            raise ValueError("distance_metric must be 'euclidean'.")

    def fit(self, X, y):
        """
        Store training data for KNN.
        """
        # convert into np array
        X, y = np.array(X), np.array(y)

        # raise errors
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")
        
        # x and y must have same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input y must have the same number of samples as X.")
        
        # set self x train and y train
        self.X_train = X
        self.y_train = y

        return self

    def predict(self, X):
        """
        Predict class labels given X.
        """
        # convert to array
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")

        # compute distances between X and whole training set
        distances = np.array([self._compute_distances(x) for x in X])

        # get the nearest k points
        k_indices = np.argpartition(distances, kth=self.k, axis=1)[:, :self.k]

        # majority vote
        y_pred = []
        for indices in k_indices:
            neighbor_labels = self.y_train[indices]

            # find the most common class 
            values, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = values[np.argmax(counts)]
            y_pred.append(majority_label)   
        
        return np.array(y_pred) 

    def _compute_distances(self, x):
        """
        Compute distances between a single sample and all training samples.
        """
        return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))

    def score(self, X, y):
        """
        Compute accuracy on the given dataset.
        """
        X, y = np.array(X), np.array(y)
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)

        return accuracy
