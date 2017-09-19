import json
from gaze_data import GazeData
import numpy as np
from sklearn import linear_model
from utils import get_files


def predict(result_path, output_path, model_x, model_y):
    files = get_files(result_path)
    for f in files:
        print("\n"+f)
        path = result_path + f
        data = GazeData(path)

        features = np.array(data.cal_features)
        pointsX = np.array([x[0] for x in data.cal_points])
        pointsY = np.array([x[1] for x in data.cal_points])

        model_x.fit(features, pointsX)
        model_y.fit(features, pointsY)

        errorX = np.sum(np.abs(model_x.predict(features) - pointsX)) / len(features)
        errorY = np.sum(np.abs(model_y.predict(features) - pointsY)) / len(features)
        print("Train ErrorX", errorX)
        print("Train ErrorY", errorY)

        predFeatures = np.array(data.pred_features)
        webgazeX = np.array([x[0] for x in data.pred_points])
        webgazeY = np.array([x[1] for x in data.pred_points])

        newPredX = model_x.predict(predFeatures)
        newPredY = model_y.predict(predFeatures)

        errorX = np.sum(np.abs(newPredX - webgazeX)) / len(predFeatures)
        errorY = np.sum(np.abs(newPredY - webgazeY)) / len(predFeatures)

        print("Webgaze Difference", errorX)
        print("Webgaze Difference", errorY)

        newPredPairs = [[x,y] for x,y in zip(newPredX, newPredY)]
        modifiedData = data.put_pred(newPredPairs)

        m_path = output_path + "modified_" + f
        with open(m_path, 'w') as outfile:
            json.dump(modifiedData, outfile)












