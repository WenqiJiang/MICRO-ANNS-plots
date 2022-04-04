import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('ggplot')

def get_default_colors():

  default_colors = []
  for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
      default_colors.append(color["color"])
      # print(color["color"], type(color["color"]))

  return default_colors

default_colors = get_default_colors()



# SIFT100M, K=100
max_bandwidth = 1.056 # evaluated using hdparm on d5005
y_throughput = np.array([0.8655295464940783, 0.8507665968552809, 0.7382743579114903, 0.7589342403561251, 0.9268145425899533, 0.9691437919700577]) # GB/s
y_QPS = np.array([17.310590929881567, 8.507665968552809, 3.6913717895574516, 1.8973356008903126, 1.1585181782374416, 0.6057148699812861])
y_recall = np.array([0.5200, 0.7034, 0.8450, 0.9312, 0.9782, 0.9954]) * 100

x_labels = ['nprobe=1', 'nprobe=2', 'nprobe=4','nprobe=8', 'nprobe=16', 'nprobe=32']

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars    

# fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 2))

fig, ax = plt.subplots(1, 1, figsize=(4, 1.5))
ax1 = ax.twinx()

label_font = 9
tick_font = 10
tick_label_font = 9
legend_font = 10
title_font = 14

rects = ax.bar(x, y_throughput, width, color=default_colors[0])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Equivalent Disk\n Bandwidth (GB/s)', fontsize=label_font)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=tick_label_font, rotation=30)

ax.text(0.5, max_bandwidth * 1.05, 'Max disk read performance', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
ax.hlines(y=max_bandwidth, xmin=-0.5, xmax=5.5, linestyles='dashed', color='#6C8EBF')

# ax.legend([rects, line], ["QPS", "Recall"], facecolor='white', framealpha=1, frameon=True, loc=(0.65, 0.42), fontsize=legend_font, ncol=1)


# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)

def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            notation = '{:.0f}'.format(height)
        else:   
            notation = 'N/A'
        ax.annotate(notation,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='center', rotation=90)


# autolabel(rects, ax)

ax.set(ylim=[0, max_bandwidth * 1.2])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/on_disk_bandwidth.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
