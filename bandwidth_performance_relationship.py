# 3D plot documents: 
#   https://matplotlib.org/stable/search.html?q=3d
#   https://matplotlib.org/stable/gallery/mplot3d/surface3d_radial.html?highlight=3d


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.ticker import LinearLocator

plt.style.use('ggplot')

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      # print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()

# 16 bytes / cycle, 140 MHz
def get_perf(PE_num_array, PQ_codes_per_cluster_exp_array):
    """
    Return the cluster per sec array given the PE_num_array & PQ_codes_per_cluster_array
    """
    
    assert PE_num_array.shape == PQ_codes_per_cluster_exp_array.shape
    dim_x = PE_num_array.shape[0]
    dim_y = PE_num_array.shape[1]
    CPS_array = np.zeros((dim_x, dim_y))

    for i in range(dim_x):
        for j in range(dim_y):
            PE_num = PE_num_array[i][j]
            PQ_codes_per_cluster = 10 ** PQ_codes_per_cluster_exp_array[i][j]
            PQ_codes_per_PE = np.ceil(PQ_codes_per_cluster / PE_num)

            # stage pipeline = 63 stages
            cycles = 256 + 63 + PQ_codes_per_PE
            CPS = 1 / (cycles / (140 * 1e6))
            CPS_array[i][j] = CPS

    return CPS_array

# x -> PQ codes per cluster
# y -> bandwidth
# z -> performance, cluster per search  
X = np.arange(0, 6 + 0.001, 0.01)  # number of PQ codes per cell, 1e8 / 65536 = 1525 = 1.5 * 1e3; 1e8 / 1024 = 1e5
Y = np.arange(1, 100 + 1)  # number of PEs (n x n)
# Y = np.arange(1, 100 + 1)  # number of PEs (n x n)
X, Y = np.meshgrid(X, Y)  # number of PEs (n x n)
Z = get_perf(Y, X)

print(X.shape, Y.shape, Z.shape)
print("===== X =====", X)
print("===== Y =====", Y)
print("===== Z =====", Z)

# fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 2))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

label_font = 12
tick_font = 10
tick_label_font = 9
legend_font = 10
title_font = 14

surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.YlGnBu_r)
# ax.zaxis.set_major_locator(LinearLocator(10))
fig.colorbar(surf, shrink=0.5, aspect=5)
# line, = ax.plot(x, y_recall, marker='X', markersize=10, color=default_colors[0])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('#PE', fontsize=label_font)
ax.set_xlabel('#PQ codes per cluster (10^n)', fontsize=label_font)
ax.set_zlabel('Throughput (#clusters per sec)', fontsize=label_font)
# ax.set_xticks(X)
# ax.set_xticklabels(x_labels, fontsize=tick_label_font) 
# plt.xticks(rotation=0)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)


# autolabel(rects, ax)

# ax.set(ylim=[0, np.amax(y_QPS) * 1.2])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/bandwidth_performance_relationship.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
