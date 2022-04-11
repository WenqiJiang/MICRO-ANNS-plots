# Example usaage: 
#   Normalize by FLOPs
#   python CPU_GPU_FPGA_max_throughput_normalize.py --legend_loc_x 0.5

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
plt.style.use('grayscale')

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--legend_loc_x', type=float, default=0.5, help="the x position of legend, 0~1")


args = parser.parse_args()
cpu_performance_dict_dir = './cpu_performance_result/cpu_throughput_SIFT100M_qbs_10000_m4.4xlarge.pkl'
gpu_performance_dict_dir = './gpu_performance_result/gpu_throughput_SIFT100M_qbs_10000_V100_32GB.pkl'
dbname = 'SIFT100M'

recall_goal_R1 = 0.3
recall_goal_R10 = 0.8
recall_goal_R100 = 0.95

batch_size=10000


def get_cpu_performance_tuple(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return the performance tuple (index_key, QPS) of certain topK and recall_goal
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
            performance_tuple.append((index_key.replace(",PQ16",""), 0))

    return performance_tuple


def get_gpu_performance_tuple(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return the performance tuple (index_key, QPS) of certain topK and recall_goal
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if index_key[:len("OPQ16,IMI")] == "OPQ16,IMI":
            continue
        if topK in d[dbname][index_key] and d[dbname][index_key][topK] is not None and \
            recall_goal in d[dbname][index_key][topK] and d[dbname][index_key][topK][recall_goal] is not None and \
            d[dbname][index_key][topK][recall_goal][batch_size] is not None:
            performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal][batch_size]))
        else:
            performance_tuple.append((index_key.replace(",PQ16",""), 0))

    return performance_tuple

def get_fpga_performance_tuple(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return the performance tuple (index_key, QPS) of certain topK and recall_goal
    """
    performance_tuple = []
    for index_key in d[dbname]:
        if topK in d[dbname][index_key] and \
            recall_goal in d[dbname][index_key][topK] and \
            d[dbname][index_key][topK][recall_goal] is not None and \
            index_key[:len("OPQ16,IMI")] != "OPQ16,IMI":
            performance_tuple.append((index_key.replace(",PQ16",""), d[dbname][index_key][topK][recall_goal]))

    return performance_tuple

def normalize_performance_tuple(performance_tuple, factor):

    """
    The networked bitstream can be downgraded in terms of frequency,
        e.g., factor = 126/140 when downgrading to 126MHz from 140MHz
    """


    normalized_performance_tuple = []
    for idx_name, throughput in performance_tuple:
        normalized_performance_tuple.append([idx_name, throughput * factor])

    return normalized_performance_tuple

def sort_performance_tuple(performance_tuple):
    order = ['IVF1024', 'IVF2048', 'IVF4096', 'IVF8192', 'IVF16384', 'IVF32768', 'IVF65536', 'OPQ16,IVF1024', \
        'OPQ16,IVF2048', 'OPQ16,IVF4096', 'OPQ16,IVF8192', 'OPQ16,IVF16384', 'OPQ16,IVF32768', 'OPQ16,IVF65536']
    # order = ['IVF1024', 'IVF2048', 'IVF4096', 'IVF8192', 'IVF16384', 'IVF32768', 'IVF65536', 'IVF131072', 'IVF262144', 'OPQ16,IVF1024', \
    #     'OPQ16,IVF2048', 'OPQ16,IVF4096', 'OPQ16,IVF8192', 'OPQ16,IVF16384', 'OPQ16,IVF32768', 'OPQ16,IVF65536', 'OPQ16,IVF131072', 'OPQ16,IVF262144']

    sorted_performance_tuple = []

    for label in order:
        find = False 
        for (idx_name, throughput) in performance_tuple:
            if idx_name == label:
                sorted_performance_tuple.append((label, throughput))
                find = True
                break
        if not find:
            sorted_performance_tuple.append((label, 0))

    return sorted_performance_tuple


d_cpu = None
with open(cpu_performance_dict_dir, 'rb') as f:
    d_cpu = pickle.load(f)
d_gpu = None
with open(gpu_performance_dict_dir, 'rb') as f:
    d_gpu = pickle.load(f)
d_fpga = None
fpga_baseline_performance_dict_dir = './fpga_performance_result/FPGA_perf_dict_SIFT100M_K_{}.pkl'.format(1)
with open(fpga_baseline_performance_dict_dir, 'rb') as f:
    d_fpga = pickle.load(f)

performance_tuple_cpu_R1 = sort_performance_tuple(get_cpu_performance_tuple(d_cpu, dbname, topK=1, recall_goal=recall_goal_R1))
performance_tuple_gpu_R1 = sort_performance_tuple(get_gpu_performance_tuple(d_gpu, dbname, topK=1, recall_goal=recall_goal_R1))
performance_tuple_fpga_R1 = sort_performance_tuple(get_fpga_performance_tuple(d_fpga, dbname, topK=1, recall_goal=recall_goal_R1))

max_qps_cpu_R1 = np.max([p[1] for p in performance_tuple_cpu_R1])
max_qps_gpu_R1 = np.max([p[1] for p in performance_tuple_gpu_R1])
max_qps_fpga_R1 = np.max([p[1] for p in performance_tuple_fpga_R1])

d_fpga = None
fpga_baseline_performance_dict_dir = './fpga_performance_result/FPGA_perf_dict_SIFT100M_K_{}.pkl'.format(10)
with open(fpga_baseline_performance_dict_dir, 'rb') as f:
    d_fpga = pickle.load(f)
performance_tuple_cpu_R10 = sort_performance_tuple(get_cpu_performance_tuple(d_cpu, dbname, topK=10, recall_goal=recall_goal_R10))
performance_tuple_gpu_R10 = sort_performance_tuple(get_gpu_performance_tuple(d_gpu, dbname, topK=10, recall_goal=recall_goal_R10))
performance_tuple_fpga_R10 = sort_performance_tuple(get_fpga_performance_tuple(d_fpga, dbname, topK=10, recall_goal=recall_goal_R10))

max_qps_cpu_R10 = np.max([p[1] for p in performance_tuple_cpu_R10])
max_qps_gpu_R10 = np.max([p[1] for p in performance_tuple_gpu_R10])
max_qps_fpga_R10 = np.max([p[1] for p in performance_tuple_fpga_R10])


d_fpga = None
fpga_baseline_performance_dict_dir = './fpga_performance_result/FPGA_perf_dict_SIFT100M_K_{}.pkl'.format(100)
with open(fpga_baseline_performance_dict_dir, 'rb') as f:
    d_fpga = pickle.load(f)
performance_tuple_cpu_R100 = sort_performance_tuple(get_cpu_performance_tuple(d_cpu, dbname, topK=100, recall_goal=recall_goal_R100))
performance_tuple_gpu_R100 = sort_performance_tuple(get_gpu_performance_tuple(d_gpu, dbname, topK=100, recall_goal=recall_goal_R100))
performance_tuple_fpga_R100 = sort_performance_tuple(get_fpga_performance_tuple(d_fpga, dbname, topK=100, recall_goal=recall_goal_R100))
print("performance_tuple_gpu_R1: ", performance_tuple_gpu_R1)
print("performance_tuple_gpu_R10: ", performance_tuple_gpu_R10)
print("performance_tuple_gpu_R100: ", performance_tuple_gpu_R100)

max_qps_cpu_R100 = np.max([p[1] for p in performance_tuple_cpu_R100])
max_qps_gpu_R100 = np.max([p[1] for p in performance_tuple_gpu_R100])
max_qps_fpga_R100 = np.max([p[1] for p in performance_tuple_fpga_R100])


if recall_goal_R1 == 0.25:
    max_qps_fpga_optimized_R1 = 31033
elif recall_goal_R1 == 0.3:
    max_qps_fpga_optimized_R1 = 27709 # 126 MHz

if recall_goal_R10== 0.6:
    max_qps_fpga_optimized_R10 = 30965
elif recall_goal_R10 == 0.8:
    max_qps_fpga_optimized_R10 = 11035

if recall_goal_R100 == 0.95:
    max_qps_fpga_optimized_R100 = 3519

CPU_FLOPS = 32 * 16 * 2.3 # broadwell 32 op per cycle per core * 16 core * 2.3 GHz
GPU_FLOPS = 14 * 1000
FPGA_FLOPS = 9024 / 5 * 2 * (200 * 1e6) * 0.6 / 1e9 
# 9024 DSP 
# 5 DSP per MAC (2 op) 
# -> 4500 per cycle * 200 MHz * 60\%

y_cpu = np.array([max_qps_cpu_R1, max_qps_cpu_R10, max_qps_cpu_R100]) / CPU_FLOPS
y_fpga = np.array([max_qps_fpga_R1, max_qps_fpga_R10, max_qps_fpga_R100]) / FPGA_FLOPS
y_fpga_optimized = np.array([max_qps_fpga_optimized_R1, max_qps_fpga_optimized_R10, max_qps_fpga_optimized_R100]) / FPGA_FLOPS
y_gpu = np.array([max_qps_gpu_R1, max_qps_gpu_R10, max_qps_gpu_R100]) / GPU_FLOPS

print("FPGA opt / GPU: ", y_fpga_optimized / y_gpu)
print("FPGA opt / CPU: ", y_fpga_optimized / y_cpu)

x_labels = ['R@1={}%'.format(int(100 * recall_goal_R1)), \
    'R@10={}%'.format(int(100 * recall_goal_R10)), \
    'R@100={}%'.format(int(100 * recall_goal_R100))]

x = np.arange(3)  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))
# 
rects_cpu  = ax.bar(x - 1.5 * width, y_cpu, width)#, label='Men')
rects_gpu  = ax.bar(x - 0.5 * width, y_gpu, width)#, label='Men')
rects_fpga = ax.bar(x + 0.5 * width, y_fpga, width)#, label='Women')
rects_fpga_optimized = ax.bar(x + 1.5 * width, y_fpga_optimized, width)#, label='Women')


label_font = 8
tick_font = 10
tick_label_font = 10
legend_font = 8
title_font = 11

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('QPS Per Peak Gflop/s', fontsize=label_font)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
plt.xticks(rotation=0)


legend_list = ["CPU (16-core Xeon)", "GPU (V100)", "FPGA Data-independent (U280)", "FPGA Data-dependent (U280)"]
ax.legend([rects_cpu, rects_gpu, rects_fpga, rects_fpga_optimized], legend_list, facecolor='white', framealpha=1, frameon=False, loc=(args.legend_loc_x, 0.4), fontsize=legend_font, ncol=1)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)
# ax.set_title('{} R@{}={}'.format(dbname, topK, recall_goal), fontsize=label_font)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.1f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='left', rotation=90)


# autolabel(rects_cpu)
# autolabel(rects_gpu)
# autolabel(rects_fpga)
# autolabel(rects_fpga_optimized)

best_qps = np.amax([np.amax(y_gpu), np.amax(y_fpga), np.amax(y_fpga_optimized), np.amax(y_cpu)])
ax.set(ylim=[0, best_qps * 1.25])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./CPU_GPU_FPGA_throughput_comparison_fig/max_CPU_GPU_FPGA_normalized.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
