import json
from PIL import Image, ImageOps
RESIZE_WIDTH = 10
RESIZE_HEIGHT = 6
counter = 0
class GazeData:

    def get_eyes_features(self,eyes):
        from generate_eye_img import eye_to_img
        left_eye = eye_to_img(eyes["left"])
        right_eye = eye_to_img(eyes["right"])
        # left_eye.save("eyes/" + str(self.counter) + "r_left.png")
        # right_eye.save("eyes/" + str(self.counter) + "r_right.png")

        left_eye = left_eye.resize((RESIZE_WIDTH, RESIZE_HEIGHT), resample=Image.BILINEAR).convert("L")
        right_eye = right_eye.resize((RESIZE_WIDTH, RESIZE_HEIGHT), resample=Image.BILINEAR).convert("L")

        # left_eye.save("eyes/" + str(self.counter) + "left.png")
        # right_eye.save("eyes/" + str(self.counter) + "right.png")

        left_eye = ImageOps.equalize(left_eye)
        right_eye = ImageOps.equalize(right_eye)

        combined = Image.new("L", (RESIZE_WIDTH * 2, RESIZE_HEIGHT))
        combined.paste(left_eye)
        combined.paste(right_eye, (RESIZE_WIDTH, 0))
        # combined.save("eyes/" + str(self.counter) + "combined.png")
        self.counter += 1
        return list(combined.getdata())
        pass

    def __init__(self,file_path):
        self.counter = 0
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
            if not (eyes["left"]["blink"] or eyes["right"]["blink"]):
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
                ef = self.get_eyes_features(eyes)
                if ef is not None:
                    self.pred_features.append(ef)
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


