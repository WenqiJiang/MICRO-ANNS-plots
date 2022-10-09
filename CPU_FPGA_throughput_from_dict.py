# Example usaage: 
#   python CPU_GPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir './cpu_performance_result_m4.4xlarge/cpu_throughput_SIFT100M.pkl' --gpu_performance_dict_dir '../gpu_performance_result_V100_32GB/gpu_throughput_SIFT100M.pkl' --fpga_baseline_performance_dict_dir '../fpga_performance_result/FPGA_perf_dict_SIFT100M.pkl' --dbname SIFT100M --topK 100 --recall_goal 0.95

# Full command list
#   python CPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M_qbs_10000_m5.4xlarge.pkl' --fpga_baseline_performance_dict_dir './fpga_performance_result/FPGA_perf_dict_SIFT100M_K_1.pkl' --dbname SIFT100M --topK 1 --recall_goal 0.3 --legend_loc_x 0.05
#   python CPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M_qbs_10000_m5.4xlarge.pkl' --fpga_baseline_performance_dict_dir './fpga_performance_result/FPGA_perf_dict_SIFT100M_K_10.pkl' --dbname SIFT100M --topK 10 --recall_goal 0.8 --legend_loc_x 0.05
#   python CPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M_qbs_10000_m5.4xlarge.pkl' --fpga_baseline_performance_dict_dir './fpga_performance_result/FPGA_perf_dict_SIFT100M_K_100.pkl' --dbname SIFT100M --topK 100 --recall_goal 0.95 --legend_loc_x 0.05
#   python CPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M_qbs_10000_m5.4xlarge.pkl' --fpga_baseline_performance_dict_dir './fpga_performance_result/FPGA_perf_dict_Deep100M_K_1.pkl' --dbname Deep100M --topK 1 --recall_goal 0.3 --legend_loc_x 0.05
#   python CPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M_qbs_10000_m5.4xlarge.pkl' --fpga_baseline_performance_dict_dir './fpga_performance_result/FPGA_perf_dict_Deep100M_K_10.pkl' --dbname Deep100M --topK 10 --recall_goal 0.7 --legend_loc_x 0.05
#   python CPU_FPGA_throughput_from_dict.py --cpu_performance_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M_qbs_10000_m5.4xlarge.pkl' --fpga_baseline_performance_dict_dir './fpga_performance_result/FPGA_perf_dict_Deep100M_K_100.pkl' --dbname Deep100M --topK 100 --recall_goal 0.95 --legend_loc_x 0.05

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

# plt.style.use('grayscale')
plt.style.use('ggplot')

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--cpu_performance_dict_dir', type=str, default='../cpu_performance_result_m4.4xlarge/cpu_throughput_SIFT100M.pkl', 
    help="a dictionary of d[dbname][index_key][topK][recall_goal] -> throughput (QPS)")
parser.add_argument('--fpga_baseline_performance_dict_dir', type=str, default='./fpga_performance_result/FPGA_perf_dict_SIFT100M.pkl', 
    help="a dictionary of d[dbname][index_key][topK][recall_goal] -> throughput (QPS)")
parser.add_argument('--dbname', type=str, default=0, help="dataset name, e.g., SIFT100M")
parser.add_argument('--topK', type=int, default=10, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--recall_goal', type=float, default=0.5, help="target minimum recall, e.g., 50%=0.5")
parser.add_argument('--legend_loc_x', type=float, default=0.0, help="the x position of legend, 0~1")



args = parser.parse_args()
cpu_performance_dict_dir = args.cpu_performance_dict_dir
fpga_baseline_performance_dict_dir = args.fpga_baseline_performance_dict_dir
dbname = args.dbname
topK = args.topK
recall_goal = args.recall_goal

batch_size=10000


def get_cpu_performance_tuple(d, dbname, topK, recall_goal):
    """
    d: input dictionary
        d[dbname][index_key][topK][recall_goal] = throughput (QPS)
    return the performance tuple (index_key, QPS) of certain topK and recall_goal
    """
    print(d)
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
    # order = ['IVF1024', 'IVF2048', 'IVF4096', 'IVF8192', 'IVF16384', 'IVF32768', 'IVF65536', 'IVF131072', 'IVF262144', 'OPQ16,IVF1024', \
    #     'OPQ16,IVF2048', 'OPQ16,IVF4096', 'OPQ16,IVF8192', 'OPQ16,IVF16384', 'OPQ16,IVF32768', 'OPQ16,IVF65536', 'OPQ16,IVF131072', 'OPQ16,IVF262144']
    order = ['IVF1024', 'IVF2048', 'IVF4096', 'IVF8192', 'IVF16384', 'IVF32768', 'IVF65536', 'OPQ16,IVF1024', \
        'OPQ16,IVF2048', 'OPQ16,IVF4096', 'OPQ16,IVF8192', 'OPQ16,IVF16384', 'OPQ16,IVF32768', 'OPQ16,IVF65536']

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
d_fpga = None
with open(fpga_baseline_performance_dict_dir, 'rb') as f:
    d_fpga = pickle.load(f)

performance_tuple_cpu = sort_performance_tuple(get_cpu_performance_tuple(d_cpu, dbname, topK, recall_goal))
performance_tuple_fpga = sort_performance_tuple(get_fpga_performance_tuple(d_fpga, dbname, topK, recall_goal))

x_labels_cpu = []
y_cpu = []
for (idx_name, throughput) in performance_tuple_cpu:
    y_cpu.append(throughput)
    x_labels_cpu.append(idx_name)

x_labels_fpga = []
y_fpga = []
for (idx_name, throughput) in performance_tuple_fpga:
    y_fpga.append(throughput)
    x_labels_fpga.append(idx_name)

assert x_labels_cpu == x_labels_fpga

x_labels = []
for l in x_labels_cpu:
    if 'OPQ16,' in l:
        x_labels.append('OPQ+\n' + l[len('OPQ16,'):])
    else:
        x_labels.append(l)
print(x_labels)

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(6, 2))
# 
rects_cpu  = ax.bar(x - width / 2, y_cpu, width)#, label='Men')
rects_fpga = ax.bar(x + width / 2, y_fpga, width)#, label='Women')

optimized_FPGA_throughput = None
if args.dbname == 'SIFT100M' and args.topK == 1 and args.recall_goal == 0.25:
    optimized_FPGA_throughput = 31033
    x_labels.insert(0, 'IVF1024')
    # ax.text(0, optimized_FPGA_throughput * 1.05, 'FANNS-generated FPGA', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    # ax.text(0, optimized_FPGA_throughput * 0.95, 'QPS={}, Index={}'.format(optimized_FPGA_throughput, 'IVF1024'), fontsize=10, horizontalalignment='left', verticalalignment='top')
    # ax.hlines(y=optimized_FPGA_throughput, xmin=-0.5, xmax=13.5, linestyles='solid', color='black')
elif args.dbname == 'SIFT100M' and args.topK == 1 and args.recall_goal == 0.3:
    optimized_FPGA_throughput = 27709 # 126 MHz
    x_labels.insert(0, 'IVF4096')
    # ax.text(0, optimized_FPGA_throughput * 1.05, 'FANNS-generated FPGA', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    # ax.text(0, optimized_FPGA_throughput * 0.95, 'QPS={}, Index={}'.format(optimized_FPGA_throughput, 'IVF4096'), fontsize=10, horizontalalignment='left', verticalalignment='top')
    # ax.hlines(y=optimized_FPGA_throughput, xmin=-0.5, xmax=13.5, linestyles='solid', color='black')
elif args.dbname == 'SIFT100M' and args.topK == 10 and args.recall_goal == 0.6:
    optimized_FPGA_throughput = 30965
    x_labels.insert(0, 'IVF4096')
    # ax.text(0, optimized_FPGA_throughput * 1.05, 'FANNS-generated FPGA', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    # ax.text(0, optimized_FPGA_throughput * 0.95, 'QPS={}, Index={}'.format(optimized_FPGA_throughput, 'IVF4096'), fontsize=10, horizontalalignment='left', verticalalignment='top')
    # ax.hlines(y=optimized_FPGA_throughput, xmin=-0.5, xmax=13.5, linestyles='solid', color='black')
elif args.dbname == 'SIFT100M' and args.topK == 10 and args.recall_goal == 0.8:
    optimized_FPGA_throughput = 11035
    x_labels.insert(0, 'OPQ+\nIVF8192')
    # ax.text(0, optimized_FPGA_throughput * 1.05, 'FANNS-generated FPGA', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    # ax.text(0, optimized_FPGA_throughput * 0.95, 'QPS={}, Index={}'.format(optimized_FPGA_throughput, 'OPQ + IVF8192'), fontsize=10, horizontalalignment='left', verticalalignment='top')
    # ax.hlines(y=optimized_FPGA_throughput, xmin=-0.5, xmax=13.5, linestyles='solid', color='black')
elif args.dbname == 'SIFT100M' and args.topK == 100 and args.recall_goal == 0.95:
    optimized_FPGA_throughput = 3519
    x_labels.insert(0, 'OPQ+\nIVF16384')
    # ax.text(0, optimized_FPGA_throughput * 1.05, 'FANNS-generated FPGA', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    # ax.text(0, optimized_FPGA_throughput * 0.95, 'QPS={}, Index={}'.format(optimized_FPGA_throughput, 'OPQ+  IVF16384'), fontsize=10, horizontalalignment='left', verticalalignment='top')
    # ax.hlines(y=optimized_FPGA_throughput, xmin=-0.5, xmax=13.5, linestyles='solid', color='black')
elif args.dbname == 'Deep100M' and args.topK == 1 and args.recall_goal == 0.3:
    optimized_FPGA_throughput = 30658 
    x_labels.insert(0, 'OPQ+\nIVF4096')
elif args.dbname == 'Deep100M' and args.topK == 10 and args.recall_goal == 0.7:
    optimized_FPGA_throughput = 19680
    x_labels.insert(0, 'OPQ+\nIVF4096')
elif args.dbname == 'Deep100M' and args.topK == 100 and args.recall_goal == 0.95:
    optimized_FPGA_throughput = 3589
    x_labels.insert(0, 'OPQ+\nIVF16384')

rects_autogen = ax.bar(-1, optimized_FPGA_throughput, width)


best_qps_cpu = np.amax(y_cpu)
best_qps_fpga = np.amax(y_fpga)
worst_qps_cpu = np.amin([y for y in y_cpu if y != 0])
worst_qps_fpga = np.amin([y for y in y_fpga if y != 0])
print("Speedup over CPU best baseline: {:.2f} x".format(optimized_FPGA_throughput / best_qps_cpu))
print("Speedup over FPGA best baseline: {:.2f} x".format(optimized_FPGA_throughput / best_qps_fpga))
print("Speedup over CPU worst baseline: {:.2f} x".format(optimized_FPGA_throughput / worst_qps_cpu))
print("Speedup over FPGA worst baseline: {:.2f} x".format(optimized_FPGA_throughput / worst_qps_fpga))

label_font = 12
tick_font = 8
tick_label_font = 8
legend_font = 9
title_font = 11

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('QPS', fontsize=label_font)
x = np.array([-1] + list(x))
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=tick_label_font)
plt.xticks(rotation=90)



legend_list = ["CD-ANN FPGA", "CPU baseline", "FPGA baseline"]
# legend_list = ["CPU (16-core Xeon)", "FPGA baseline (U280)"]
ax.legend([rects_autogen, rects_cpu, rects_fpga], legend_list, facecolor='white', framealpha=1, frameon=False, loc=(args.legend_loc_x, 0.8), fontsize=legend_font, ncol=3)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)
ax.set_title('{} R@{}={}'.format(dbname, topK, recall_goal), fontsize=title_font)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.0f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='left', rotation=90)


# autolabel(rects_cpu)
# autolabel(rects_fpga)

best_qps = np.amax([best_qps_fpga, best_qps_cpu, optimized_FPGA_throughput])
if optimized_FPGA_throughput:
    best_qps = np.max([best_qps, optimized_FPGA_throughput])
ax.set(ylim=[0, best_qps * 1.3])
ax.text(-2.5, -0.1 * best_qps, 'Index', fontsize=10, horizontalalignment='center', verticalalignment='top')

plt.rcParams.update({'figure.autolayout': True})

for i, y in enumerate(y_cpu):
    if y == 0:
        ax.text(i + width/2, 0.1, 'cannot reach\nR@{}={}'.format(topK, recall_goal), fontsize=8, horizontalalignment='right', verticalalignment='bottom', rotation=90)


plt.savefig('./CPU_FPGA_throughput_comparison_fig/CPU_FPGA_throughput_comparison_{dbname}_R@{topK}={recall}.png'.format(dbname=dbname,topK=topK,recall=int(recall_goal*100)), transparent=False, dpi=200, bbox_inches="tight")
plt.show()
