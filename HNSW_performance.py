import matplotlib.pyplot as plt
import matplotlib

plt.style.use('ggplot')

import numpy as np


label_font = 12
tick_font = 8
tick_label_font = 8
legend_font = 10
title_font = 14

fig, ax = plt.subplots(1, 1, figsize=(6, 1.5))

x_FPGA = np.array([0, 0.2, 0.4]) + 0.2
x_CPU = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0]) + 0.4 + x_FPGA[-1]


########### TODO: Performance of FPGA (2, 4, 8 banks) using ef=128 #############
########### TODO: Performance of CPU (1~32 threads) using m4.4xlarge #############
throughput_FPGA = np.array([0.6, 0.8, 1]) * 1e5 # 2, 4, 8 banks
throughput_CPU= np.array([0.45, 0.72, 1.25, 1.99, 2.32, 2.3]) * 1e5# 1, 2, 4, 8, 16, 32 threads

max_perf = np.maximum(np.max(throughput_FPGA), np.max(throughput_CPU))

ax.bar(x_FPGA, throughput_FPGA, width=0.1) #, color='#92C6FF')
ax.bar(x_CPU, throughput_CPU, width=0.1) #, color=['#97F0AA', "#a1f1b2", "#abf3bb", "#b6f4c3", "#c0f6cc", "#cbf7d4"])

ax.set_xticks(list(x_FPGA) + list(x_CPU))
ax.set_xticklabels(["2", "4", "8", "1", "2", "4", "8", "16", "32"],
           fontsize=tick_font, rotation=0)


ax.text(x_FPGA[0], 0.9 * max_perf, 'R@1=97%; ef=128',
        horizontalalignment='left',
        verticalalignment='top',
        rotation='horizontal',
        fontsize=legend_font)

ax.text(np.average(x_FPGA), -0.2 * max_perf, 'FPGA (different bank numbers)',
        horizontalalignment='center',
        verticalalignment='top',
        rotation='horizontal',
        fontsize=tick_label_font)

ax.text(np.average(x_CPU), -0.2 * max_perf, 'CPU (different thread numbers)',
        horizontalalignment='center',
        verticalalignment='top',
        rotation='horizontal',
        fontsize=tick_label_font)

ax.set_ylabel('QPS', fontsize=label_font)
ax.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

# plt.grid(True)
plt.xlim((0, x_CPU[-1] + 0.2))
# plt.ylim(0, 5.8)

plt.savefig('./out_img/HNSW_performance.png', dpi=200, bbox_inches = 'tight')
plt.show()