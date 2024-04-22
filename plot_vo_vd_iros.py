import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

local_occ = np.loadtxt("local_oc_h2.txt")
local_diff = np.loadtxt("local_diff_h2.txt")
vd_vo = local_diff/local_occ

plt.plot(range(len(vd_vo)), vd_vo,  label = 'vd vo')
plt.xlabel("Exploration Step")
plt.ylabel("v_d/v_o")
# plt.show()
tikzplotlib.save("vd_vo.tex")