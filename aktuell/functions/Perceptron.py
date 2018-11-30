
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    errors_ : list
      Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y, w):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        #self.w_ = [0.0, 0.0, 0.0]
        self.w_ = w
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                #print('xi, target: ', xi, target)
                update = self.eta * (target - self.predict(xi))
                #print('update: ', update)
                self.w_[1:] += update * xi
                self.w_[0] += update
                #print(self.w_)
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0:
                break
        return self

    def net_input(self, X):
        """Calculate net input"""
        ni = np.dot(X, self.w_[1:]) + self.w_[0]
        #print('net_input: ', ni)
        return ni

    def predict(self, X):
        """Return class label after unit step"""
        pred = np.where(self.net_input(X) >= 0.0, 1, -1)
        #print('cl: ', pred)
        return pred
    
    def show_tf(self, X, y, tf_iter = 1):
        plt.clf()
        w = [0.0, 0.0, 0.0]
        plt.axis([-10, 10, -10, 10])
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], 
                        y=X[y == cl, 1],
                        c=colors[idx],
                        marker = '*')

        for i in np.arange(tf_iter):
            w = self.fit(X, y, w).w_
            print(i, w)
            ymin = max(-10.0*(-self.w_[1]/self.w_[2])-self.w_[0]/self.w_[2], -10.0)
            ymax = min(10.0*(-self.w_[1]/self.w_[2])-self.w_[0]/self.w_[2], 10.0)
            plt.plot([-10, 10], [ymin, ymax])
        plt.grid()
        plt.show()
        
#mp = Perceptron(eta=1, n_iter=1)
#X = np.array([-1, -1, 2, 1, 3, 1, 1, 3, 2, 3, 3, 3]).reshape(-1,2)
#y = np.array([1, 1, 1, -1, -1, -1])

#mp.show_tf(X, y, tf_iter = 4)