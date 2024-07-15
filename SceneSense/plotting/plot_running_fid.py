import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
import os

#load all the fid data
#loop through all the directories in house 1
house_path = "/hdd/sceneDiff_data/house_1/"
#get all the directories
house_dirs = natsorted(os.listdir(house_path))
# print(house_dirs)nning octomap
fid_arr = []
kid_arr = []
diff_fid_arr = []
diff_kid_arr = []
diff_fid_arr_10 = []
diff_kid_arr_10 = []
diff_fid_arr_30 = []
diff_kid_arr_30 = []
diff_fid_arr_50 = []
diff_kid_arr_50 = []
diff_fid_arr_30_guide_5 = []
diff_kid_arr_30_guide_5  = []
diff_fid_arr_30_guide_2 = []
diff_kid_arr_30_guide_2  = []
diff_fid_arr_30_guide_1 = []
diff_kid_arr_30_guide_1  = []
diff_fid_arr_30_guide_0 = []
diff_kid_arr_30_guide_0  = []

for idx, step in enumerate(house_dirs):
    fid = np.loadtxt(house_path + step + "/fid_val.txt")
    kid = np.loadtxt(house_path + step + "/kid_val.txt")
    # diff_fid = np.loadtxt(house_path + step + "/diffusion_fid_val.txt")
    # diff_kid = np.loadtxt(house_path + step + "/diffusion_kid_val.txt")
    diff_fid_50 = np.loadtxt(house_path + step + "/diffusion_fid_val_50.txt")
    diff_kid_50 = np.loadtxt(house_path + step + "/diffusion_kid_val_50.txt")
    diff_fid_30 = np.loadtxt(house_path + step + "/diffusion_fid_val_30.txt")
    diff_kid_30 = np.loadtxt(house_path + step + "/diffusion_kid_val_30.txt")
    # Lol I changed the script while it was running oops
    if idx < 10:
        diff_fid_30_guide_5 = np.loadtxt(house_path + step + "/diffusion_fid_val_305.txt")
        diff_kid_30_guide_5 = np.loadtxt(house_path + step + "/diffusion_kid_val_305.txt")
    else:
        diff_fid_30_guide_5 = np.loadtxt(house_path + step + "/diffusion_fid_val_30steps_guide5.txt")
        diff_kid_30_guide_5 = np.loadtxt(house_path + step + "/diffusion_kid_val_30steps_guide5.txt")
    diff_fid_30_guide_2 = np.loadtxt(house_path + step + "/diffusion_fid_val_30steps_guide2.txt")
    diff_kid_30_guide_2 = np.loadtxt(house_path + step + "/diffusion_kid_val_30steps_guide2.txt")
    #skipped 38 on accident 
    if idx != 38:
        diff_fid_30_guide_1 = np.loadtxt(house_path + step + "/diffusion_fid_val_30steps_guide1.txt")
        diff_kid_30_guide_1 = np.loadtxt(house_path + step + "/diffusion_kid_val_30steps_guide1.txt")
    # diff_fid_30_guide_0 = np.loadtxt(house_path + step + "/diffusion_fid_val_30steps_guide0.txt")
    # diff_kid_30_guide_0 = np.loadtxt(house_path + step + "/diffusion_kid_val_30steps_guide0.txt")
    # diff_fid_10 = np.loadtxt(house_path + step + "/diffusion_fid_val_10.txt")
    
    # diff_kid_10 = np.loadtxt(house_path + step + "/diffusion_kid_val_10.txt")


    fid_arr.append(fid)
    kid_arr.append(kid)
    # diff_fid_arr.append(diff_fid)
    # diff_kid_arr.append(diff_kid)
    diff_fid_arr_50.append(diff_fid_50)
    diff_kid_arr_50.append(diff_kid_50)
    diff_fid_arr_30.append(diff_fid_30)
    diff_kid_arr_30.append(diff_kid_30)
    diff_fid_arr_30_guide_5.append(diff_fid_30_guide_5)
    diff_kid_arr_30_guide_5.append(diff_kid_30_guide_5)
    diff_fid_arr_30_guide_2.append(diff_fid_30_guide_2)
    diff_kid_arr_30_guide_2.append(diff_kid_30_guide_2)
    diff_fid_arr_30_guide_1.append(diff_fid_30_guide_1)
    diff_kid_arr_30_guide_1.append(diff_kid_30_guide_1)
    # diff_fid_arr_30_guide_0.append(diff_fid_30_guide_0)
    # diff_kid_arr_30_guide_0.append(diff_kid_30_guide_0)
    # diff_fid_arr_10.append(diff_fid_10)
    # diff_kid_arr_10.append(diff_kid_10)

# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# ax1.plot(range(len(fid_arr)), fid_arr)
# ax1.plot(range(len(diff_fid_arr)), diff_fid_arr)
# ax1.set_title('FID')
# ax2.plot(range(len(kid_arr)), kid_arr)
# ax2.plot(range(len(diff_kid_arr)), diff_kid_arr)
# ax2.set_title('KID')
# plt.plot(range(len(fid_arr)), fid_arr)
# plt.plot(range(len(diff_fid_arr)), diff_fid_arr)
# plt.plot(range(len(diff_fid_arr_50)), diff_fid_arr_50)
# plt.plot(range(len(diff_fid_arr_30)), diff_fid_arr_30)
# plt.savefig("FID.jpg")
plt.plot(range(len(fid_arr)), fid_arr, label = "BL")
plt.plot(range(len(diff_fid_arr_30)), diff_fid_arr_30, label = "ds = 30, s = 3")
# plt.plot(range(len(diff_fid_arr_50)), diff_fid_arr_50, label = "ds = 50, s = 3")
# plt.plot(range(len(diff_fid_arr_30_guide_5)), diff_fid_arr_30_guide_5, label = "ds = 30, s = 5")
plt.legend()
plt.savefig("running_fid.jpg")
plt.clf()
# plt.plot(range(len(kid_arr)), kid_arr)
# plt.plot(range(len(diff_kid_arr_30)), diff_kid_arr_30)
# plt.show()
# plt.plot(range(len(diff_fid_arr_50)), diff_fid_arr_50)
# plt.savefig("FID.jpg")
# plt.clf()

# #concated data
fid_cat_arr = [fid_arr,diff_fid_arr_30, diff_fid_arr_50, diff_fid_arr_30_guide_5,diff_fid_arr_30_guide_2]
plt.boxplot(fid_cat_arr)
plt.savefig("fid_box.jpg")
plt.clf()

fid_cat_arr = [kid_arr,diff_kid_arr_30, diff_kid_arr_50, diff_kid_arr_30_guide_5, diff_kid_arr_30_guide_2]
plt.boxplot(fid_cat_arr)
plt.savefig("kid_box.jpg")

print("baseline fid: ", np.mean(kid_arr))
print("30 steps, g_s = 3 kid: ", np.mean(diff_kid_arr_30))
print("50 steps, g_s = 3: ", np.mean(diff_kid_arr_50))
print("30 steps, g_s = 5: ", np.mean(diff_kid_arr_30_guide_5))
print("30 steps, g_s = 2: ", np.mean(diff_kid_arr_30_guide_2))
print("30 steps, g_s = 1: ", np.mean(diff_kid_arr_30_guide_1))
# print("30 steps, g_s = 0: ", np.mean(diff_kid_arr_30_guide_0))

print("\nbaseline kid median: ", np.median(kid_arr))
print("30 steps, g_s = 3 kid median: ", np.median(diff_kid_arr_30))
print("50 steps, g_s = 3 median: ", np.median(diff_kid_arr_50))
print("30 steps, g_s = 5: ", np.median(diff_kid_arr_30_guide_5))
print("30 steps, g_s = 2: ", np.median(diff_kid_arr_30_guide_2))
print("30 steps, g_s = 1: ", np.median(diff_kid_arr_30_guide_1))
# print("30 steps, g_s = 0: ", np.median(diff_kid_arr_30_guide_0))

print("\nbaseline kid STD: ", np.std(kid_arr))
print("30 steps, g_s = 3 kid STD: ", np.std(diff_kid_arr_30))
print("50 steps, g_s = 3 STD: ", np.std(diff_kid_arr_50))
print("30 steps, g_s = 5: ", np.std(diff_kid_arr_30_guide_5))
print("30 steps, g_s = 2: ", np.std(diff_kid_arr_30_guide_2))
print("30 steps, g_s = 1: ", np.std(diff_kid_arr_30_guide_1))
# print("30 steps, g_s = 0: ", np.std(diff_kid_arr_30_guide_0))