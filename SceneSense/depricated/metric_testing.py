import numpy as np
import cv2
import utils.utils as utils
house_path = "/hdd/sceneDiff_data/combined_image_data/"
#load the image data

#create numpy array that will be filled
gt_arr = np.empty((0,40,40), float)
running_occ_arr = np.empty((0,40,40), float)
pred_arr = np.empty((0,40,40), float)
for i in range(23):
    img_gt = cv2.imread(house_path + "gt/110_" + str(i) + ".png")
    img_pred = cv2.imread(house_path + "pred/110_" + str(i) + ".png")
    img_running_occ = cv2.imread(house_path + "running_occ/110_" + str(i) + ".png")
    
    #reduce dimensionality and norm to 0 to 1
    img_gt = img_gt[:,:,0]/255
    img_pred = img_pred[:,:,0]/255
    img_running_occ = img_running_occ[:,:,0]/255

    gt_arr = np.append(gt_arr, img_gt[None,:,:], axis = 0)
    running_occ_arr = np.append(running_occ_arr, img_gt[None,:,:], axis = 0)
    pred_arr = np.append(pred_arr, img_gt[None,:,:], axis = 0)

print(utils.get_IoU(img_gt, img_pred))
print(utils.get_IoU(img_gt, pred_arr))
