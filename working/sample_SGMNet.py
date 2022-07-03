import cv2
import os
import time

from exec_test_data import load_model_SGMNet, matching_SGM, extract_SGM
from tools.draw_result import draw_matching

input_dir = "../input/image-matching-challenge-2022"
weight_dir = "../input/imc2022-pretrained-weights"
output_dir = "../output"

image_dir = os.path.join(input_dir, "test_images", "d91db836")
image_1_id = "81dd07fb7b9a4e01996cee637f91ca1a"
image_2_id = "0006b1337a0347f49b4e651c035dfa0e"
ext = "png"

start = time.time()
SGM_param = {
    "model": load_model_SGMNet(os.path.join(weight_dir, "sgmnet", "sgm", "root")),
}
end = time.time()
print("load_model_SGMNet: ", end - start, " s")

start = time.time()
image_1 = cv2.imread(os.path.join(image_dir, f"{image_1_id}.{ext}"))
image_2 = cv2.imread(os.path.join(image_dir, f"{image_2_id}.{ext}"))
end = time.time()
print("cv2.imread: ", end - start, " s")

start = time.time()
test_data = extract_SGM(image_1, image_2, SGM_param)
end = time.time()
print("extract_SGM: ", end - start, " s")

start = time.time()
mkpts1, mkpts2 = matching_SGM(test_data, SGM_param)
end = time.time()
print("matching_SGM: ", end - start, " s")

start = time.time()
_, inliers = cv2.findFundamentalMat(mkpts1, mkpts2, cv2.USAC_MAGSAC, 0.15, 0.9999, 20000)
inliers = inliers > 0
end = time.time()
print("cv2.findFundamentalMat: ", end - start, " s")

draw_matching(
    image_dir,
    os.path.join(output_dir, "sample_SGMNet.png"),
    image_1_id, image_2_id, ext,
    mkpts1, mkpts2, inliers)