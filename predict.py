import json
from  PIL import Image,ImageOps
from gaze_data import GazeData
import numpy as np
from sklearn import linear_model

def get_files(dir):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(dir) if isfile(join(dir, f)) and not f.startswith(".")]

result_path = "test/"
files = get_files(result_path)
for f in files:
    print("file",f)
    path = result_path + f
    data = GazeData(path)

    features = np.array(data.cal_features)
    pointsX = np.array([x[0] for x in data.cal_points])
    pointsY = np.array([x[1] for x in data.cal_points])

    modelX = linear_model.Ridge(alpha=1.0)
    modelX.fit(features, pointsX)

    modelY = linear_model.Ridge(alpha=1.0)
    modelY.fit(features, pointsY)

    errorX = np.sum(np.abs(modelX.predict(features) - pointsX)) / len(features)
    errorY = np.sum(np.abs(modelY.predict(features) - pointsY)) / len(features)
    print("Train ErrorX", errorX)
    print("Train ErrorY", errorY)

    predFeatures = np.array(data.pred_features)
    webgazeX = np.array([x[0] for x in data.pred_points])
    webgazeY = np.array([x[1] for x in data.pred_points])

    newPredX = modelX.predict(predFeatures)
    newPredY = modelY.predict(predFeatures)

    errorX = np.sum(np.abs(newPredX - webgazeX)) / len(predFeatures)
    errorY = np.sum(np.abs(newPredY - webgazeY)) / len(predFeatures)

    print("Webgaze Difference", errorX)
    print("Webgaze Difference", errorY)

    newPredPairs = [[x,y] for x,y in zip(newPredX, newPredY)]
    modifiedData = data.put_pred(newPredPairs)

    m_path = "modified_results/modified_" + f
    with open(m_path, 'w') as outfile:
        json.dump(modifiedData, outfile)












