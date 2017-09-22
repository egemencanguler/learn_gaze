from model_test import test_model
from sklearn import linear_model
from predict import predict
from generate_fixations import generate_fixation_maps
from generate_eye_img import generate_eye_imgs

# Model to predict fixation points wrt. eye features
# alpha = 1
# model_x = linear_model.Ridge(alpha=alpha)
# model_y = linear_model.Ridge(alpha=alpha)
#
# # Test the model
# errorX, errorY = test_model("webgaze_results/bekici.json",model_x, model_y)
# print("Error", errorX, ",", errorY)

# # Predict collected data with a different model
# predict("test_results/", "modified_test_results/",model_x,model_y)
#
# # Generate fixation maps
# # combines all fixation points in the given path and generates fixation maps
# generate_fixation_maps("./test_results/", "./test_fixations/")
#
# # Generate eyes' images
generate_eye_imgs("webgaze_results/bekici.json","eyes/",2)



