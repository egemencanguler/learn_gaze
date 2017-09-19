import numpy
# https://simplyml.com/the-simplest-machine-learning-algorithm/
# https://www.youtube.com/watch?v=5asL5Eq2x0A


class MyRidge:
    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda
        self.w = None

    def fit(self, X, y):
        print(X.T.dot(X).shape)
        C = X.T.dot(X) + self.lmbda * numpy.eye(X.shape[1])
        self.w = numpy.linalg.inv(C).dot(X.T.dot(y))

    def predict(self, X):
        return X.dot(self.w)

    def get_params(self, deep=True):
        return {"lmbda": self.lmbda}

    def set_params(self, lmbda=0.1):
        self.lmbda = lmbda
        return self




