# Example usaage: 
#   python CPU_GPU_FPGA_latency_from_dict.py 
#   Deep100M are not measured on GPUs

# Run with python 3.9 if the following error occurs
# WenqideMacBook-Pro@~/Works/ANNS-FPGA/python_figures wenqi$python CPU_GPU_FPGA_throughput_from_dict.py 
# Traceback (most recent call last):
#   File "CPU_GPU_FPGA_throughput_from_dict.py", line 112, in <module>
#     fpga_cpu = pickle.load(f)
# ValueError: unsupported pickle protocol: 5

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import os

# plt.style.use('grayscale')
plt.style.use('ggplot')

import argparse 
parser = argparse.ArgumentParser()
# parser.add_argument('--dbname', type=str, default=0, help="dataset name, e.g., SIFT100M")

args = parser.parse_args()

dbname = 'SIFT100M'
index_key = 'IVF8192,PQ16'
topK = 10
recall_goal = 0.8
qbs = 1

num_device = 8

fpga_performance_parent_dir = './fpga_latency_results'
fpga_performance_file_names = ['Scalability_SIFT100M_K10_R80_util_50_{}_server'.format(i) for i in range(1, 1 + num_device)]

gpu_performance_parent_dir = './gpu_performance_result/gpu_scalability'
gpu_performance_dict_list = [
    f'gpu_response_time_SIFT{i}00M_IVF8192,PQ16_K_10_recall_80_qbs_1_gpu_{i}.pkl' for i in range(1,1 + num_device)] 


df = {"hardware" : [] , "num_device" : [], "latency" : []}

# add FPGAs
FPGA_median_latency = []
FPGA_P95_latency = []
for i in range(num_device):

    latency_array = np.fromfile(os.path.join(fpga_performance_parent_dir, fpga_performance_file_names[i]), dtype=np.float32)

    for j, latency in enumerate(latency_array): 
        if j <= 100: 
            continue
        df["hardware"].append('FPGA')
        df["num_device"].append(i)
        df["latency"].append(latency)

    sorted_latency_array = np.sort(latency_array)
    median_latency = sorted_latency_array[int(len(sorted_latency_array)/2)]
    P95_latency = sorted_latency_array[int(len(sorted_latency_array) * 0.95)]
    FPGA_median_latency.append(median_latency)
    FPGA_P95_latency.append(P95_latency)
    print(f'{i + 1} FPGAs: median latency: {median_latency} ms\tP95: {P95_latency} ms')

# add GPUs
GPU_median_latency = []
GPU_P95_latency = []
for i in range(num_device):

    with open(os.path.join(gpu_performance_parent_dir, gpu_performance_dict_list[i]), 'rb') as f:
        latency_array = pickle.load(f)

    for j, latency in enumerate(latency_array): 
        if j <= 100: 
            continue
        df["hardware"].append('GPU')
        df["num_device"].append(i)
        df["latency"].append(latency)

    sorted_latency_array = np.sort(latency_array)
    median_latency = sorted_latency_array[int(len(sorted_latency_array)/2)]
    P95_latency = sorted_latency_array[int(len(sorted_latency_array) * 0.95)]
    GPU_median_latency.append(median_latency)
    GPU_P95_latency.append(P95_latency)
    print(f'{i + 1} GPUs: median latency: {median_latency} ms\tP95: {P95_latency} ms')

print("Median latency speedup: {}".format(np.array(GPU_median_latency) / np.array(FPGA_median_latency)))
print("P95 latency speedup: {}".format(np.array(GPU_P95_latency) / np.array(FPGA_P95_latency)))

df = pd.DataFrame(data=df)

x_labels = [str(i) for i in range(1, 1 + num_device)]

# width = 0.15  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(7, 1.5))

ax = sns.violinplot(data=df, scale='width', inner='box', x="num_device", y="latency", hue="hardware")


label_font = 13
tick_font = 12
tick_label_font = 12
legend_font = 12
title_font = 13

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Latency (ms)', fontsize=label_font)
ax.set_xlabel('Number of accelerators', fontsize=label_font)
# x = np.array([-1] + list(x))
# ax.set_xticks(x[1:])
# plt.yscale('log')
ax.set_xticklabels(x_labels, fontsize=tick_label_font)
# plt.xticks(rotation=90)

ax.legend(facecolor='white', framealpha=1, frameon=False, loc=(0.3, 1.02), fontsize=legend_font, ncol=3)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)
# ax.set_title('{} R@{}={}'.format(dbname, topK, recall_goal), fontsize=title_font)


ax.set(ylim=[0, 15])
# ax.text(-2.5, -0.1 * best_qps, 'Index', fontsize=10, horizontalalignment='center', verticalalignment='top')

plt.rcParams.update({'figure.autolayout': True})


plt.savefig('./CPU_GPU_FPGA_throughput_comparison_fig/GPU_FPGA_latency_scalability.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
