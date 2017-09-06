import numpy as np
from sklearn import linear_model
from gaze_data import GazeData

test_file = "results/bahadır.json"
gaze_data = GazeData(test_file)

features = np.array(gaze_data.cal_features)
pointsX = np.array([x[0] for x in gaze_data.cal_points])
pointsY = np.array([x[1] for x in gaze_data.cal_points])


n_test = int(len(features) / 4)
n_train = len(features) - n_test

idx = np.array(list(range(len(features))))
np.random.shuffle(idx)
train_features = features[idx[:n_train]]
train_x = pointsX[idx[:n_train]]
train_y = pointsY[idx[:n_train]]

test_features = features[idx[n_train:]]
test_x = pointsX[idx[n_train:]]
test_y = pointsY[idx[n_train:]]

modelX = linear_model.Ridge(alpha=1.0)
modelY = linear_model.Ridge(alpha=1.0)

modelX.fit(train_features,train_x)
modelY.fit(train_features,train_y)

predX = modelX.predict(test_features)
predY = modelY.predict(test_features)

print("Error(X,Y)", np.sum(np.abs(predX - test_x)) / n_test, np.sum(np.abs(predY - test_y)) / n_test)







