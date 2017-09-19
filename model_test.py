import numpy as np
from gaze_data import GazeData


def test_model(test_file, model_x, model_y):
    # Prepare the data
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

    # 20 point for each subject, between 1-10 example(eye feature vec) for each point
    n_train = 15
    n_test = len(data) - n_train

    train_features = []
    train_x = []
    train_y = []

    test_features = []
    test_x = []
    test_y = []

#seperate data into training and testing sets
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

    model_x.fit(train_features, train_x)
    model_y.fit(train_features, train_y)

    pred_x = model_x.predict(test_features)
    pred_y = model_y.predict(test_features)

    errorX = np.sum(np.abs(pred_x - test_x)) / n_test
    errorY = np.sum(np.abs(pred_y - test_y)) / n_test

    return errorX,errorY








