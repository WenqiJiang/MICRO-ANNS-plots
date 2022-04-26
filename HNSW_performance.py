import matplotlib.pyplot as plt
import matplotlib

plt.style.use('grayscale')

import numpy as np


label_font = 12
tick_font = 8
tick_label_font = 8
legend_font = 10
title_font = 14

fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

x_FPGA_opt = np.array([0, 0.3, 0.6]) + 0.2 # no opt; opt 1 = iter-ovelap; opt 2 = iter-overlap + batching
x_FPGA_bank_num = 0.4 + float(x_FPGA_opt[-1]) + np.array([0, 0.2, 0.4])
x_FPGA_replication = float(0.4) + float(x_FPGA_bank_num[-1]) + np.array([0, 0.2, 0.4, 0.6])
x_CPU = 0.4 + float(x_FPGA_replication[-1]) + np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]) 


########### TODO: Performance of FPGA (2, 4, 8 banks) using ef=128 #############
########### TODO: Performance of CPU (1~32 threads) using m4.4xlarge #############
throughput_FPGA_opt = np.array([3650, 4760, 6410]) # 2 banks -> no opt; opt 1 = iter-ovelap; opt 2 = iter-overlap + batching
throughput_FPGA_bank_num = np.array([6410, 9900, 11100])  # opt 2: 2, 4, 8 banks
throughput_FPGA_replication = np.array([6250, 12600, 18900, 25300]) # rep opt 1 by 1, 2, 3, 4 times
throughput_CPU= np.array([0.45, 0.72, 1.25, 1.99, 2.32, 2.3]) * 1e4# 1, 2, 4, 8, 16, 32 threads

max_perf = np.maximum(np.max(throughput_FPGA_replication), np.max(throughput_CPU))

ax.bar(x_FPGA_opt, throughput_FPGA_opt, width=0.1) #, color='#92C6FF')
ax.bar(x_FPGA_bank_num, throughput_FPGA_bank_num, width=0.1) #, color='#92C6FF')
ax.bar(x_FPGA_replication, throughput_FPGA_replication, width=0.1) #, color='#92C6FF')
ax.bar(x_CPU, throughput_CPU, width=0.1) #, color=['#97F0AA', "#a1f1b2", "#abf3bb", "#b6f4c3", "#c0f6cc", "#cbf7d4"])

x_tick_list = list(x_FPGA_opt) + list(x_FPGA_bank_num) + list(x_FPGA_replication) + list(x_CPU)
ax.set_xticks(x_tick_list)

ax.set_xticklabels([
        # opt
        "Base", "Opt.1", "Opt.2", \
        # bank num
        "2", "4", "8", \
        # replicates
        "1", "2", "3", "4", \
        #  CPU
        "1", "2", "4", "8", "16", "32"],
        fontsize=tick_label_font, rotation=0)

ax.text(np.average(x_FPGA_opt), -0.2 * max_perf, 'FPGA designs',
        horizontalalignment='center',
        verticalalignment='top',
        rotation='horizontal',
        fontsize=tick_label_font)

ax.text(np.average(x_FPGA_bank_num), -0.2 * max_perf, 'FPGA #banks', # (Opt.2)
        horizontalalignment='center',
        verticalalignment='top',
        rotation='horizontal',
        fontsize=tick_label_font)

ax.text(np.average(x_FPGA_replication), -0.2 * max_perf, 'FPGA #replicas', # (Opt.1)
        horizontalalignment='center',
        verticalalignment='top',
        rotation='horizontal',
        fontsize=tick_label_font)

ax.text(np.average(x_CPU), -0.2 * max_perf, ' CPU #threads', # (32-core+16 banks)
        horizontalalignment='center',
        verticalalignment='top',
        rotation='horizontal',
        fontsize=tick_label_font)

ax.text(x_tick_list[0], 0.8 * max_perf, 'R@1=97%; L=128',
        horizontalalignment='left',
        fontsize=legend_font)
# ax.text(np.average(x_tick_list), 1.1 * max_perf, 'R@1=97%; L=128',
#         horizontalalignment='center',
#         fontsize=legend_font)

ax.set_ylabel('QPS', fontsize=label_font)
# ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

# plt.grid(True)
plt.xlim((0, x_CPU[-1] + 0.2))
# plt.ylim(0, 5.8)

plt.savefig('./out_img/HNSW_performance.png', dpi=200, bbox_inches = 'tight')
plt.show()