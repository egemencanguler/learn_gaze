import numpy
class MyRidge:
    def __init__(self, lmbda=0.1):
        self.lmbda = lmbda

    def fit(self, X, y):
        print(X.T.dot(X).shape)
        C = X.T.dot(X) + self.lmbda * numpy.eye(X.shape[1])
        self.w = numpy.linalg.inv(C).dot(X.T.dot(y))
        print("W",self.w.shape)

    def predict(self, X):
        return X.dot(self.w)

    def get_params(self, deep=True):
        return {"lmbda": self.lmbda}

    def set_params(self, lmbda=0.1):
        self.lmbda = lmbda
        return self




