import numpy as np
from sklearn import linear_model
from gaze_data import GazeData
from my_ridge import MyRidge
from sklearn.svm import SVR

# Prepare the data
test_file = "results/bahadır.json"
gaze_data = GazeData(test_file)


data = {}
for pIdx in range(len(gaze_data.cal_features)):
    point = (gaze_data.cal_points[pIdx][0], gaze_data.cal_points[pIdx][1])
    feat = gaze_data.cal_features[pIdx]
    if point not in data:
        data[point] = []
    data[point].append(feat)


idx = np.array(list(range(len(data))))
np.random.shuffle(idx)

n_train = 15
n_test = len(data) - n_train

train_features = []
train_x = []
train_y = []

test_features = []
test_x = []
test_y = []


items = list(data.items())
for i in range(len(items)):
    item = items[idx[i]]
    features = item[1]
    point = item[0]
    f_list = train_features
    x_list = train_x
    y_list = train_y
    if i > n_train:
        f_list = test_features
        x_list = test_x
        y_list = test_y
    for f in features:
        f_list.append(f)
        x_list.append(point[0])
        y_list.append(point[1])

train_x = np.array(train_x)
train_y = np.array(train_y)
train_features = np.array(train_features)

test_x = np.array(test_x)
test_y = np.array(test_y)
test_features = np.array(test_features)

# print("TrainX", train_x)
# print("TrainY",train_y)
# print("TestX",test_x)
# print("TestY",test_y)


alpha = 1
modelX = linear_model.Ridge(alpha=alpha)
modelY = linear_model.Ridge(alpha=alpha)
# modelX = MyRidge()
# modelY = MyRidge()
# modelX = SVR(C=1.0, epsilon=0.2)
# modelY = SVR(C=1.0, epsilon=0.2)

modelX.fit(train_features, train_x)
modelY.fit(train_features, train_y)

predX = modelX.predict(test_features)
predY = modelY.predict(test_features)

errorX = np.sum(np.abs(predX - test_x)) / n_test
errorY = np.sum(np.abs(predY - test_y)) / n_test

print("Error", errorX, ",", errorY)








