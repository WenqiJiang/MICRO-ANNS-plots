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
# topK = args.topK

dbname_list = ['SIFT100M'] * 3
topK_list = [1, 10, 100] 
recall_goal_list = [0.3, 0.8, 0.95]
cpu_performance_parent_dir = './cpu_performance_result'
cpu_performance_dict_list = [
    'cpu_response_time_SIFT100M_qbs_1_m5.4xlarge.pkl'] * 3
fpga_performance_parent_dir = './fpga_performance_result'
fpga_performance_dict_list = [
    'FPGA_perf_dict_SIFT100M_K_1.pkl', 'FPGA_perf_dict_SIFT100M_K_10.pkl', 'FPGA_perf_dict_SIFT100M_K_100.pkl']
gpu_performance_parent_dir = './gpu_performance_result'
gpu_performance_dict_list = [
    'gpu_faiss_1.7.0_response_time_SIFT100M_qbs1.pkl'] * 3

assert len(dbname_list) == len(topK_list) and \
    len(dbname_list) == len(recall_goal_list) and \
    len(dbname_list) == len(cpu_performance_dict_list) and \
    len(dbname_list) == len(fpga_performance_dict_list) and \
    len(dbname_list) == len(gpu_performance_dict_list) 
num_scenario = len(dbname_list)


def get_cpu_best_latency(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = latency array (ms)
    return best index_key and it's 95% tail latency
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if index_key[:len("OPQ16,IMI")] == "OPQ16,IMI":
            continue
        if 'IMI' in index_key:
            continue
        if d[dbname][index_key][topK][recall_goal] is not None:
            performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal]))
        else:
            # performance_tuple.append((index_key.replace(",PQ16",""), 0))
            pass

    tail_latency_95_list = []
    best_latency = 100000000
    best_index = None
    for index_key, latency_array in performance_tuple:
        sorted_latency_array = np.sort(latency_array)
        tail_latency_95 = sorted_latency_array[int(0.95 * sorted_latency_array.shape[0])]
        if best_latency > tail_latency_95:
            best_latency = tail_latency_95
            best_index = index_key

    return index_key, latency_array

def get_fpga_best_latency(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return best index_key and it's 95% tail latency
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if topK in d[dbname][index_key] and \
            recall_goal in d[dbname][index_key][topK] and \
            d[dbname][index_key][topK][recall_goal] is not None and \
            index_key[:len("OPQ16,IMI")] != "OPQ16,IMI":
            performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal]))

    tail_latency_95_list = []
    best_latency = 100000000
    best_index = None
    for index_key, QPS in performance_tuple:
        tail_latency_95 = 1 / QPS * 6 * 1000 # 6 search stages 
        if best_latency > tail_latency_95:
            best_latency = tail_latency_95
            best_index = index_key

    return index_key, best_latency


def get_gpu_best_latency(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = latency (not sure, maybe average?)
        e.g., 0.8: {10000: array([122.15805], dtype=float32)},
    return best index_key and it's latency (not sure, maybe average?)
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if topK in d[dbname][index_key] and \
            recall_goal in d[dbname][index_key][topK] and \
            d[dbname][index_key][topK][recall_goal] is not None and \
            index_key[:len("OPQ16,IMI")] != "OPQ16,IMI":
                if 1 in d[dbname][index_key][topK][recall_goal]:
                    # print(d[dbname][index_key][topK][recall_goal])
                    performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal][1]))

    tail_latency_95_list = []
    best_latency = 100000000
    best_index = None
    for index_key, latency_array in performance_tuple:
        sorted_latency_array = np.sort(latency_array)
        tail_latency_95 = sorted_latency_array[int(0.95 * sorted_latency_array.shape[0])]
        if best_latency > tail_latency_95:
            best_latency = tail_latency_95
            best_index = index_key

    return index_key, latency_array

###### dataframe schema
# hardware   scenario (e.g., SIFT100M R@1=0.3) latency (in ms)

df = {"hardware" : [] , "scenario" : [], "latency" : []}


for i in range(num_scenario):
    d_cpu = None
    with open(os.path.join(cpu_performance_parent_dir, cpu_performance_dict_list[i]), 'rb') as f:
        d_cpu = pickle.load(f)
    index_key, latency_array = get_cpu_best_latency(d_cpu, dbname_list[i], topK_list[i], recall_goal_list[i])
    for latency in latency_array: 
        df["hardware"].append('CPU')
        df["scenario"].append('{}\nR@{}={}'.format(dbname_list[i], topK_list[i], recall_goal_list[i]))
        df["latency"].append(latency)


# for i in range(num_scenario):
#     d_fpga = None
#     with open(os.path.join(fpga_performance_parent_dir, fpga_performance_dict_list[i]), 'rb') as f:
#         d_fpga = pickle.load(f)
#     index_key, latency = get_fpga_best_latency(d_fpga, dbname_list[i], topK_list[i], recall_goal_list[i])
#     fpga_best_latency_list.append(latency)

for i in range(num_scenario):
    d_gpu = None
    with open(os.path.join(gpu_performance_parent_dir, gpu_performance_dict_list[i]), 'rb') as f:
        d_gpu = pickle.load(f)
    index_key, latency_array = get_gpu_best_latency(d_gpu, dbname_list[i], topK_list[i], recall_goal_list[i])
    for latency in latency_array: 
        df["hardware"].append('GPU')
        df["scenario"].append('{}\nR@{}={}'.format(dbname_list[i], topK_list[i], recall_goal_list[i]))
        df["latency"].append(latency)

df = pd.DataFrame(data=df)

x_labels = []
for i in range(num_scenario):
    x_labels.append('{}\nR@{}={}'.format(dbname_list[i], topK_list[i], recall_goal_list[i]))
print(x_labels)

x = np.arange(len(x_labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(7, 1.3))

ax = sns.violinplot(data=df, scale='width', inner='box', x="scenario", y="latency", hue="hardware")


label_font = 10
tick_font = 10
tick_label_font = 9
legend_font = 10
title_font = 11

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Latency (ms)', fontsize=label_font)
x = np.array([-1] + list(x))
ax.set_xticks(x[1:])
plt.yscale('log')
ax.set_xticklabels(x_labels, fontsize=tick_label_font)
# plt.xticks(rotation=90)


ax.legend(facecolor='white', framealpha=1, frameon=False, loc='upper center', fontsize=legend_font, ncol=3)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)
# ax.set_title('{} R@{}={}'.format(dbname, topK, recall_goal), fontsize=title_font)


# ax.set(ylim=[0, np.amax(cpu_best_latency_list) * 1.5])
# ax.text(-2.5, -0.1 * best_qps, 'Index', fontsize=10, horizontalalignment='center', verticalalignment='top')

plt.rcParams.update({'figure.autolayout': True})


plt.savefig('./CPU_GPU_FPGA_throughput_comparison_fig/CPU_GPU_FPGA_latency_comparison_violin.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
