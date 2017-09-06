import json
from PIL import Image, ImageOps
RESIZE_WIDTH = 10
RESIZE_HEIGHT = 6

class GazeData:

    def eye_to_img(self,eye):
        eye_patch = eye["patch"]
        img_data = [x for x in zip(eye_patch[0::4], eye_patch[1::4], eye_patch[2::4], eye_patch[3::4])]
        eye_img = Image.new("RGBA", (eye["width"], eye["height"]))
        eye_img.putdata(img_data)
        return eye_img

    def eye_to_img2(self,eye):
        eye_patch = list(eye["patch"]["data"].values())
        img_data = [x for x in zip(eye_patch[0::4], eye_patch[1::4], eye_patch[2::4], eye_patch[3::4])]
        eye_img = Image.new("RGBA", (eye["width"], eye["height"]))
        eye_img.putdata(img_data)
        return eye_img

    def get_eyes_features(self,eyes):
        try:
            left_eye = self.eye_to_img(eyes["left"]).resize((RESIZE_WIDTH, RESIZE_HEIGHT)).convert("L")
            right_eye = self.eye_to_img(eyes["right"]).resize((RESIZE_WIDTH, RESIZE_HEIGHT)).convert("L")
        except:
            left_eye = self.eye_to_img2(eyes["left"]).resize((RESIZE_WIDTH, RESIZE_HEIGHT)).convert("L")
            right_eye = self.eye_to_img2(eyes["right"]).resize((RESIZE_WIDTH, RESIZE_HEIGHT)).convert("L")

        left_eye = ImageOps.equalize(left_eye)
        right_eye = ImageOps.equalize(right_eye)
        combined = Image.new("L", (RESIZE_WIDTH * 2, RESIZE_HEIGHT))
        combined.paste(left_eye)
        combined.paste(right_eye, (RESIZE_WIDTH, 0))

        return list(combined.getdata())
        pass

    def __init__(self,file_path):
        self.file_path = file_path
        with open(file_path) as json_file:
            data = json.load(json_file)

        self.data = data
        calibration = data["calibration"]
        self.windowSize = data["experiment"]["windowSize"]

        self.cal_features = []
        self.cal_points = []
        self.cal_eyes = []
        for i in range(len(calibration)):
            eyes = calibration[i][0]
            point = calibration[i][1]
            self.cal_features.append(self.get_eyes_features(eyes))
            self.cal_points.append(point)
            self.cal_eyes.append(eyes)

        self.pred_features = []
        self.pred_points = []
        self.pred_eyes = []
        for rec in self.data["experiment"]["recordings"]:
            for g in rec["gazeData"]:
                eyes = g[5]
                self.pred_features.append(self.get_eyes_features(eyes))
                self.pred_points.append([g[2], g[3]])
                self.pred_eyes.append(eyes)


    def normalize(self,point,imageScaledSize):
        windowWidth = self.windowSize["x"]
        windowHeight = self.windowSize["y"]
        leftX = (windowWidth - imageScaledSize[0]) / 2;
        upY = (windowHeight - imageScaledSize[1]) / 2;
        normalizedX = (point[0] - leftX) / imageScaledSize[0]
        normalizedY = (point[1] - upY) / imageScaledSize[1]
        return [normalizedX,normalizedY]

    def put_pred(self,predictions):
        pi = 0
        mdata = self.data.copy()
        for rec in mdata["experiment"]["recordings"]:
            imageScaledSize = rec["imageScaledSize"]
            for i in range(len(rec["gazeData"])):
                norm_pred = self.normalize(predictions[pi],imageScaledSize)
                rec["gazeData"][i][0] = norm_pred[0]
                rec["gazeData"][i][1] = norm_pred[1]
                rec["gazeData"][i][2] = predictions[pi][0]
                rec["gazeData"][i][3] = predictions[pi][1]
                pi += 1
        return mdata


