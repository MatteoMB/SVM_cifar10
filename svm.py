import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False

class Kernel:
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f
    def rbf(gamma):
        def f(x, y):
            return np.exp( -gamma * np.linalg.norm(x-y,ord=2)**2)
        return f
    def polykernel(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f
    
class SVMPredictor(object):
    def __init__(self,kernel,b,multipliers,
                 support_vectors,support_vectors_y):
        self._kernel = kernel
        self._b = b
        self._multipliers = multipliers
        self._support_vectors= support_vectors
        self._support_vectors_y = support_vectors_y

    def predict_one(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._b
        for z_i, x_i, y_i in zip(self._multipliers,
                                 self._support_vectors,
                                 self._support_vectors_y):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()
    def predict(self,Xtest):
        res=np.zeros(Xtest.shape[0])
        for i,xi in enumerate(Xtest):
            res[i]= self.predict_one(xi)
        return res
    
class SVMTrainer():
    def __init__(self, kernel, C=1):
        if kernel=="rbf":
            self._kernel = Kernel.rbf(gamma)
        else:
            self._kernel = Kernel.linear()
        self._c = C
    def compute_b(self,multipliers, X, y):
         return np.mean([y_j - np.sum([alpha*y_i*self._kernel(x_i,x_j)
                         for (alpha,x_i,y_i) in zip(multipliers, X, y)])
                for (x_j,y_j) in zip(X,y)])
    def gram(self, X,n):
        # Gram matrix - The matrix of all possible inner products of X.
        return np.array([self._kernel(X[i], X[j])
                      for j in range(n)
                      for i in range(n)]).reshape((n,n))
    def fit(self, X, y):
        n_samples = X.shape[0]
        K=self.gram(X,n_samples)
        P = matrix(np.outer(y, y) * K)
        q = matrix(-1 * np.ones(n_samples))
        # Equality constraints
        A = matrix(y, (1, n_samples))
        b = matrix(0.0)
        # Inequality constraints
        G_std = matrix(np.diag(-1 * np.ones(n_samples)))
        h_std = matrix(np.zeros(n_samples))

        G_soft = matrix(np.diag(np.ones(n_samples)))
        h_soft = matrix(np.ones(n_samples) * self._c)

        G = matrix(np.vstack((G_std, G_soft)))
        h = matrix(np.vstack((h_std, h_soft)))


        # Solve the problem
        solution = solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        multipliers = np.ravel(solution['x'])
        # Support vectors have positive multipliers.
        has_positive_multiplier = multipliers > 1e-5
        sv_multipliers = multipliers[has_positive_multiplier]
        support_vectors = X[has_positive_multiplier]
        support_vectors_y = y[has_positive_multiplier]
        b=self.compute_b(sv_multipliers,support_vectors,support_vectors_y)
        return SVMPredictor(
            kernel=self._kernel,
            b=b,
            multipliers=sv_multipliers,
            support_vectors=support_vectors,
            support_vectors_y=support_vectors_y)