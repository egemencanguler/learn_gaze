from PIL import Image
import os

image_mappings = {
            0:"i05june05_static_street_boston_p1010764",
            1:"i2057541",
            2:"i64011654",
            3:"i102423191",
            4:"i109572486"
    }

class GenerationData:
    # [path, rec["imageSize"][0], rec["imageSize"][1], rec["gazeData"]]
    def __init__(self,path, img_width, img_heigh, gaze_data):
        self.path = path
        self.image_size = (img_width, img_heigh)
        self.gaze_data = [[p[0],p[1]] for p in gaze_data]


def put_gaze(img, x, y):
    radius = 0
    for xi in range(x - radius, x + radius + 1):
        for yi in range(y - radius, y + radius + 1):
            dis = ( (xi - x) ** 2 + (yi - y) ** 2 ) ** 0.5
            if (0 < xi < img.size[0] and 0 < yi < img.size[1]) and dis <= radius:
                img.putpixel((xi, yi), (256, 256, 256))


def generate_img(img_path, img_width, img_height, normalized_gaze_data,img_no = None):
    if img_no is None:
        img = Image.new("RGB", (int(img_width), int(img_height)))
    else:
        img = Image.open("./imgs/img" + str(img_no) + ".jpg")

    for d in normalized_gaze_data:
        if d[0] > 0 and d[1] > 0:
            x = int(d[0] * img_width)
            y = int(d[1] * img_height)
            put_gaze(img,x,y)
    img.save(img_path)


def web_gaze(file_name, output_dir):
    import json
    with open(file_name) as data_file:
        data = json.load(data_file)

    recordings = data["experiment"]["recordings"]

    generation_data = []
    for rec in recordings:
        img_name = image_mappings[rec["imageNumber"]]
        path = output_dir + img_name + ".png"

        gd = GenerationData(path, rec["imageSize"][0], rec["imageSize"][1], rec["gazeData"])
        generation_data.append(gd)
        tp = output_dir + "_" + img_name  + ".png"
        generate_img(tp, rec["imageSize"][0], rec["imageSize"][1], rec["gazeData"], rec["imageNumber"])
    return generation_data


def get_files(dir):
    from os import listdir
    from os.path import isfile, join
    return [f for f in listdir(dir) if isfile(join(dir, f))]


# Image name - gaze points dic
image_data_pairs = {}
results_dir = "./test/"
output_dir = "./fixations2/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


fnames = get_files(results_dir)
counter = 0

for fn in fnames:
    path = results_dir + fn
    for gd in web_gaze(path,output_dir):
        # Use partial data
        # gd.gaze_data = gd.gaze_data[:10]
        if gd.path not in image_data_pairs:
            image_data_pairs[gd.path] = gd
        else:
            image_data_pairs[gd.path].gaze_data += gd.gaze_data
        # Output individual fixation maps for each subject
        # generate_img(gd.path + str(counter) + ".png", gd.image_size[0], gd.image_size[1], gd.gaze_data)
        counter += 1

# Generate images from combined fixations
for item in image_data_pairs.items():
    combined = item[1]
    generate_img(combined.path, combined.image_size[0], combined.image_size[1], combined.gaze_data)

