import numpy as np
from sklearn import linear_model
from gaze_data import GazeData

# Prepare the data
test_file = "results/bahadır.json"
gaze_data = GazeData(test_file)

features = np.array(gaze_data.cal_features)
pointsX = np.array([x[0] for x in gaze_data.cal_points])
pointsY = np.array([x[1] for x in gaze_data.cal_points])


n_test = int(len(features) / 4)
n_train = len(features) - n_test

errorX = []
errorY = []

for i in range(500):
    idx = np.array(list(range(len(features))))
    np.random.shuffle(idx)
    train_features = features[idx[:n_train]]
    train_x = pointsX[idx[:n_train]]
    train_y = pointsY[idx[:n_train]]

    test_features = features[idx[n_train:]]
    test_x = pointsX[idx[n_train:]]
    test_y = pointsY[idx[n_train:]]

    # standard_deviation = sqrt( sum( (x - mean)^2 ) / count(x))
    # y = (x - mean) / standard_deviation
    # mean = np.mean(train_features, axis=0)
    # standard_deviation = np.sqrt(np.sum((train_features - mean) ** 2, axis=0) / train_features.shape[0])
    #
    # train_features = (train_features - mean) / standard_deviation
    # test_features = (test_features - mean) / standard_deviation

    # Test Model
    alpha = 10 ** 5
    modelX = linear_model.Ridge(alpha=alpha)
    modelY = linear_model.Ridge(alpha=alpha)

    modelX.fit(train_features,train_x)
    modelY.fit(train_features,train_y)

    predX = modelX.predict(test_features)
    predY = modelY.predict(test_features)

    errorX.append(np.sum(np.abs(predX - test_x)) / n_test)
    errorY.append(np.sum(np.abs(predY - test_y)) / n_test)

print("Avg Errors(X,Y)",sum(errorX) / len(errorX), ",", sum(errorY) / len(errorY))
print(errorX)



# 19, 20
# 16, 18 standatized data







