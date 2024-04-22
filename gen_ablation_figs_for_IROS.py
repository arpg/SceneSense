import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
####################################3
#this is for the guidance scale
#######################################

# FID = [17.93 , 17.81, 18.50, 27.34]
# KID = [7.82, 7.91,  8.93, 18.20]
# x_axis = [1,3,5,10]
# plt.scatter(x_axis, FID, label = 'FID')
# plt.scatter(x_axis, KID, marker = "^", label='KID')
# plt.plot(x_axis, FID)
# plt.plot(x_axis, KID)
# # plt.legend()
# plt.xlabel("Guidance Scale")
# # plt.show()

# import tikzplotlib

# tikzplotlib.save("guidance_scale_plot.tex")

############################################
# # This is for Denoising Steps:
#############################################
# FID = [35.81, 17.81, 18.46, 17.59]
# KID = [27.7, 7.9, 8.2, 8.1]
# Steps = [10,30,60,100]
# plt.scatter(Steps, FID, label = 'FID')
# plt.scatter(Steps, KID, marker = "^", label='KID')
# plt.plot(Steps, FID)
# plt.plot(Steps, KID)
# # plt.legend()
# plt.xlabel("Denoising Steps")
# # plt.show()
# tikzplotlib.save("Denoising_steps_plot.tex")

#############################
# #This is for the conditioning
################################3

# c_0_i_1_FID = 17.98
# c_0_i_1_KID = 7.94

# c_1_i_0_FID = 66.54
# c_1_i_0_KID = 41.85

# c_1_i_1_FID = 17.81
# c_1_i_1_KID = 7.91
categories = ['Cond = 0\nInpaint = 1', 'Cond = 1\nInpaint = 0', 'Cond = 1\nInpaint = 1']
FID_values = [66.54, 17.98, 17.81]
KID_values = [41.85, 7.94,  7.91]

# Calculating positions for each bar
width = 0.35  # the width of the bars
x = [x - width/2 for x in range(len(categories))]  # x positions for the first group
x2 = [x + width for x in x]  # x positions for the second group

# Creating the bar chart
fig, ax = plt.subplots()
bars1 = ax.bar(x, FID_values, width, label='FID')
bars2 = ax.bar(x2, KID_values, width, label='KID')

# Adding labels, title, and custom x-axis tick labels
ax.set_ylabel('Values')
ax.set_title('Values by category and type')
ax.set_xticks([x for x in range(len(categories))])
ax.set_xticklabels(categories)
# ax.legend()

# plt.show()
tikzplotlib.save("bar_chart.tex")