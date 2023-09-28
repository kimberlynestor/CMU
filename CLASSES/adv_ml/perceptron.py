import numpy as np


class Perceptron(object):
    def __init__(self, d):
        """
        Perceptron Classifier
        The perceptron algorithm classifies data points of dimensionality `d`
        into {-1, +1} classes.
        """
        self.d = d
        self.w = np.zeros(d)  # Don't change this
        self.b = 0.0  # Don't change this

    def predict(self, x:  np.ndarray) -> np.ndarray:
        """Input: x eq R^ n x d, all subjs, d=784"""
        # predict method, step func
        y = np.array(list(map(lambda x_i: np.dot(self.w, x_i) + self.b, x)))
        y_hat = np.array([-1 if i < 0 else 1 for i in y])
        return y_hat

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Input: x eq R^ d, y=scalar, one subj"""
        # update method
        self.w = self.w + (x * y)
        self.b = self.b + y
        assert self.w.shape == (self.d,),\
            f'Check your weight dimensions. Expected: {(self.d,)}. Actual: {self.w.shape}.'


    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray, iterations: int) -> None:

        t = 0
        n = X_train.shape[0]
        # perceptron algorithm and store the trajectories
        self.trajectories = {'train': [], 'test': []}
        self.trajectories['train'].append(0)

        #training loop
        while (self.trajectories['train'][-1] < 1) & (t < iterations):
            curr_pred = self.predict(X_train)
            for i in range(n):
                if curr_pred[i] != y_train[i]:
                    # update based on x_i
                    self.update(X_train[i], y_train[i])
                    break
            # check accuracy
            train_acc = (np.array(y_train) == np.array(curr_pred)).sum() / n
            self.trajectories['train'].append(train_acc)

            # test alg each epoch
            test = self.predict(X_test)
            test_acc = (np.array(y_test) == np.array(test)).sum() / n
            self.trajectories['test'].append(test_acc)

            t+=1

        self.trajectories['train'] = self.trajectories['train'][1:]
        return self.trajectories

