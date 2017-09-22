from  PIL import Image
from gaze_data import GazeData


def eye_to_img(eye):
    if "data" in eye["patch"]:
        data = eye["patch"]["data"]
        eye_patch = []
        n = max( [int(x) for x in data.keys()] ) + 1
        for i in range(n):
            eye_patch.append(int(data[str(i)]))
    else:
        eye_patch = eye["patch"]
    img_data = [x for x in zip(eye_patch[0::4], eye_patch[1::4], eye_patch[2::4], eye_patch[3::4])]
    eye_img = Image.new("RGBA", (eye["width"], eye["height"]))
    eye_img.putdata(img_data)
    return eye_img


def generate_eye_imgs(test_file,result_dir,number_of_eyes):
    gaze_data = GazeData(test_file)

    for i in range(number_of_eyes):
        eyes = gaze_data.cal_eyes[i]
        print(eyes)
        left_eye = eye_to_img(eyes["left"])
        right_eye = eye_to_img(eyes["right"])
        left_eye.save(result_dir + "left_" + str(i) + ".png")
        right_eye.save(result_dir + "right_" + str(i) + ".png")







