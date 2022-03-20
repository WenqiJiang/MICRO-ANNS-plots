import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('ggplot')

y_fpga_baseline = np.array([14228, 9539, 2801, 12628, 8464, 2593, 12630, 8464, 2513])
y_fpga_specialized = np.array([31033, 30965, 3519, 29574, 30908, 3519, 30911, 30909, 3818])

x_labels = ['SIFT100M\nR@1=25%', 'SIFT100M\nR@10=60%', 'SIFT100M\nR@100=95%',\
    'SIFT500M\nR@1=25%', 'SIFT500M\nR@10=60%', 'SIFT500M\nR@100=95%',\
    'SIFT1000M\nR@1=25%', 'SIFT1000M\nR@10=60%', 'SIFT1000M\nR@100=95%']

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars    

speedup_array = y_fpga_specialized / y_fpga_baseline
max_speedup = np.amax(speedup_array)
min_speedup = np.amin(speedup_array)
print("Speedup:\n{}\nmax: {:.2f}x\nmin: {:.2f}x".format(speedup_array, max_speedup, min_speedup))

fig, ax = plt.subplots(1, 1, figsize=(8, 2))
# 
rects1  = ax.bar(x - width / 2, y_fpga_baseline, width)#, label='Men')
rects2   = ax.bar(x + width / 2, y_fpga_specialized, width)#, label='Women')

label_font = 12
tick_font = 10
tick_label_font = 9
legend_font = 10
title_font = 14

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('QPS', fontsize=label_font)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=tick_label_font)
plt.xticks(rotation=45)

legend_list = ['FPGA baseline', 'FPGA specialized']
ax.legend([rects1, rects2], legend_list, facecolor='white', framealpha=1, frameon=False, loc=(0.2, 1.02), fontsize=legend_font, ncol=2)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='center', rotation=90)


autolabel(rects1)
autolabel(rects2)

# annotate speedup

def autolabel_speedup(rects, speedup_array):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate('{:.2f}x'.format(speedup_array[i]),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='top', fontsize=tick_font, horizontalalignment='center', rotation=90)

# autolabel_speedup(rects2, speedup_array)

ax.set(ylim=[0, np.amax(y_fpga_specialized) * 1.5])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/experiment_1_VLDB_comparison.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
