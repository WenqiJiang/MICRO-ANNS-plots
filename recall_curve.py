import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter

import seaborn as sns
sns.set_theme(style="whitegrid")

# plt.style.use('ggplot')

if __name__ == "__main__":

    nprobe=['1', '2', '4', '8', '16', '32', '64']

    recall_IVF1024_PQ8 = [0.3185, 0.3864, 0.4256, 0.4475, 0.4532, 0.4543, 0.4544]
    # recall_IVF204_PQ8 = [0.3228, 0.3915, 0.4380, 0.4602, 0.4698, 0.4714, 0.4720]
    # recall_IVF4096_PQ8 = [0.3084, 0.3862, 0.4374, 0.4696, 0.4831, 0.4875, 0.4883]
    # recall_IVF8192_PQ8 = [0.2966, 0.3748, 0.4387, 0.4774, 0.4944, 0.5004, 0.5027]
    # recall_IVF16384_PQ8 = [0.2831, 0.3706, 0.4387, 0.4849, 0.5081, 0.5191, 0.5237]
    # recall_IVF32768_PQ8 = [0.2684, 0.3574, 0.4358, 0.4874, 0.5173, 0.5295, 0.5347]
    # recall_IVF65536_PQ8 = [0.2454, 0.3355, 0.4189, 0.4782, 0.5223, 0.5441, 0.5520]
    # recall_IVF131072_PQ8 = [0.2360, 0.3250, 0.4148, 0.4775, 0.5303, 0.5613, 0.5751]
    recall_IVF262144_PQ8 = [0.2104, 0.2977, 0.3915, 0.4629, 0.5199, 0.5566, 0.5754]


    recall_OPQ8_IVF1024_PQ8 = [0.3249, 0.4057, 0.4445, 0.4646, 0.4721, 0.4741, 0.4741]
    # recall_OPQ8_IVF2048_PQ8 = [0.3320, 0.4004, 0.4520, 0.4783, 0.4885, 0.4908, 0.4914]
    # recall_OPQ8_IVF4096_PQ8 = [0.3193, 0.4016, 0.4559, 0.4921, 0.5072, 0.5116, 0.5126]
    # recall_OPQ8_IVF8192_PQ8 = [0.2999, 0.3830, 0.4510, 0.4918, 0.5103, 0.5170, 0.5206]
    # recall_OPQ8_IVF16384_PQ8 = [0.2902, 0.3761, 0.4493, 0.4955, 0.5225, 0.5329, 0.5360]
    # recall_OPQ8_IVF32768_PQ8 = [0.2684, 0.3644, 0.4480, 0.5021, 0.5354, 0.5507, 0.5562]
    # recall_OPQ8_IVF65536_PQ8 = [0.2511, 0.3484, 0.4303, 0.4943, 0.5393, 0.5602, 0.5695]
    # recall_OPQ8_IVF131072_PQ8 = [0.2375, 0.3281, 0.4158, 0.4849, 0.5343, 0.5642, 0.5798]
    recall_OPQ8_IVF262144_PQ8 = [0.2170, 0.3069, 0.4052, 0.4835, 0.5409, 0.5787, 0.6009]


    recall_IVF1024_PQ16 = [0.4674, 0.6107, 0.7105, 0.7630, 0.7857, 0.7926, 0.7943]
    # recall_IVF2048_PQ16 = [0.4452, 0.5771, 0.6881, 0.7567, 0.7904, 0.8016, 0.8037]
    # recall_IVF4096_PQ16 = [0.4128, 0.5526, 0.6606, 0.7420, 0.7860, 0.8029, 0.8075]
    # recall_IVF8192_PQ16 = [0.3788, 0.5128, 0.6381, 0.7282, 0.7862, 0.8132, 0.8228]
    # recall_IVF16384_PQ16 = [0.3502, 0.4844, 0.6077, 0.7067, 0.7686, 0.8052, 0.8208]
    # recall_IVF32768_PQ16 = [0.3194, 0.4526, 0.5866, 0.6901, 0.7645, 0.8069, 0.8296]
    # recall_IVF65536_PQ16 = [0.2814, 0.4102, 0.5427, 0.6541, 0.7444, 0.7999, 0.8296]
    # recall_IVF131072_PQ16 = [0.2607, 0.3766, 0.5104, 0.6237, 0.7179, 0.7870, 0.8282]
    recall_IVF262144_PQ16 = [0.2283, 0.3353, 0.4648, 0.5808, 0.6885, 0.7704, 0.8220]


    recall_OPQ16_IVF1024_PQ16 = [0.4693, 0.6181, 0.7230, 0.7808, 0.8070, 0.8147, 0.8156]
    # recall_OPQ16_IVF2048_PQ16 = [0.4490, 0.5885, 0.7041, 0.7754, 0.8108, 0.8217, 0.8237]
    # recall_OPQ16_IVF4096_PQ16 = [0.4127, 0.5586, 0.6677, 0.7520, 0.7992, 0.8191, 0.8250]
    # recall_OPQ16_IVF8192_PQ16 = [0.3759, 0.5139, 0.6444, 0.7402, 0.7990, 0.8288, 0.8405]
    # recall_OPQ16_IVF16384_PQ16 = [0.3529, 0.4889, 0.6156, 0.7147, 0.7836, 0.8224, 0.8379]
    # recall_OPQ16_IVF32768_PQ16 = [0.3163, 0.4513, 0.5838, 0.6926, 0.7695, 0.8179, 0.8411]
    # recall_OPQ16_IVF65536_PQ16 = [0.2805, 0.4106, 0.5392, 0.6535, 0.7459, 0.8049, 0.8380]
    # recall_OPQ16_IVF131072_PQ16 = [0.2613, 0.3802, 0.5105, 0.6247, 0.7174, 0.7873, 0.8357]
    recall_OPQ16_IVF262144_PQ16 = [0.2285, 0.3365, 0.4701, 0.5862, 0.6936, 0.7747, 0.8288]


    fig, ax = plt.subplots(1, 1, figsize=(8, 3))

    label_font = 18
    markersize = 8
    tick_font = 16
    legend_font = 14
    color_plot0 = "#008A00"
    color_plot1 = "#1BA1E2"

    plot_IVF1024_PQ8 = ax.plot(nprobe, 100 * np.array(recall_IVF1024_PQ8), marker='o', markersize=markersize, linestyle='--')
    # plot_IVF16384_PQ8 = ax.plot(nprobe, 100 * np.array(recall_IVF16384_PQ8), marker='v', markersize=markersize, linestyle='--')
    plot_IVF262144_PQ8 = ax.plot(nprobe, 100 * np.array(recall_IVF262144_PQ8), marker='^', markersize=markersize, linestyle='--')
    plot_OPQ8_IVF1024_PQ8 = ax.plot(nprobe, 100 * np.array(recall_OPQ8_IVF1024_PQ8), marker='P', markersize=markersize, linestyle='--')
    # plot_OPQ8_IVF16384_PQ8 = ax.plot(nprobe, 100 * np.array(recall_OPQ8_IVF16384_PQ8), marker='s', markersize=markersize, linestyle='--')
    plot_OPQ8_IVF262144_PQ8 = ax.plot(nprobe, 100 * np.array(recall_OPQ8_IVF262144_PQ8), marker='X', markersize=markersize, linestyle='--')

    plot_IVF1024_PQ16 = ax.plot(nprobe, 100 * np.array(recall_IVF1024_PQ16), marker='o', markersize=markersize)
    # plot_IVF16384_PQ16 = ax.plot(nprobe, 100 * np.array(recall_IVF16384_PQ16), marker='v', markersize=markersize)
    plot_IVF262144_PQ16 = ax.plot(nprobe, 100 * np.array(recall_IVF262144_PQ16), marker='^', markersize=markersize)
    plot_OPQ16_IVF1024_PQ16 = ax.plot(nprobe, 100 * np.array(recall_OPQ16_IVF1024_PQ16), marker='P', markersize=markersize)
    # plot_OPQ16_IVF16384_PQ16 = ax.plot(nprobe, 100 * np.array(recall_OPQ16_IVF16384_PQ16), marker='s', markersize=markersize)
    plot_OPQ16_IVF262144_PQ16 = ax.plot(nprobe, 100 * np.array(recall_OPQ16_IVF262144_PQ16), marker='X', markersize=markersize)


    ax.legend([plot_IVF1024_PQ8[0], 
        # plot_IVF16384_PQ8[0], 
        plot_IVF262144_PQ8[0],
        plot_OPQ8_IVF1024_PQ8[0], 
        # plot_OPQ8_IVF16384_PQ8[0], 
        plot_OPQ8_IVF262144_PQ8[0],
        plot_IVF1024_PQ16[0], 
        # plot_IVF16384_PQ16[0], 
        plot_IVF262144_PQ16[0],
        plot_OPQ16_IVF1024_PQ16[0], 
        # plot_OPQ16_IVF16384_PQ16[0], 
        plot_OPQ16_IVF262144_PQ16[0]], 
        ["IVF2^10,m=8", 
        # "IVF2^14,m=8", 
        "IVF2^18,m=8", 
        "OPQ,IVF2^10,m=8", 
        # "OPQ,IVF2^14,m=8", 
        "OPQ,IVF2^18,m=8", 
        "IVF2^10,m=16", 
        # "IVF2^14,m=16", 
        "IVF2^18,m=16", 
        "OPQ,IVF2^10,m=16", 
        # "OPQ,IVF2^14,m=16", 
        "OPQ,IVF2^18,m=16"], loc=(1.02, 0.0), fontsize=legend_font)

    ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    # ax.set_xlabel('nprobe', fontsize=label_font)
    ax.set_ylabel('Recall R@10 (%)', fontsize=label_font)
    ax.text(6.5, 10, 'nprobe', fontsize=label_font)
    plt.grid(b=False, axis='x')

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    plt.rcParams.update({'figure.autolayout': True})

    plt.savefig('./recall_curve.png', transparent=False, dpi=400, bbox_inches="tight")
    plt.show()
