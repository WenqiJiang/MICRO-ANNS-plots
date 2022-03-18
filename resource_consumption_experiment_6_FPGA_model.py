import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from resource_consumption import draw_resource_consumption_plot

x_labels = ['nlist=1024', 'nlist=2048', 'nlist=4096',\
    'nlist=8192', 'nlist=16384', 'nlist=32768', 'nlist=65536']

# Stage 1: OPQ
# Stage 2: vector quantizer
# Stage 3: select centroids
# Stage 4: construct distance LUT
# Stage 5: PQ code scan
# Stage 6: collect topK results

resource_utilization_cases = [
    # 1024
    [1.4448298890723796, 6.123070905362501, 1.9170650187549259, 8.45212057320269, 44.08073940378075, 37.982174209826766], \
    # 2048
    [1.4448298890723796, 6.123070905362501, 1.9170650187549259, 8.45212057320269, 44.08073940378075, 37.982174209826766], \
    # 4096
    [1.2884543295927244, 10.816550147965687, 2.225655173338718, 12.488187921303458, 39.30983153519513, 33.871320892604274], \
    # 8192
    [0.9968139473169931, 24.94350885828071, 1.7218802931133264, 15.72112703157005, 30.412089463310643, 26.20458040640827], \
    # 16384
    [0.6849655815935489, 39.91961585144039, 4.424225157353495, 16.066771667137157, 20.897816089733293, 18.00660565274212], \
    # 32768
    [0.824701243160479, 54.92004421808932, 6.07001575593258, 14.331100227337021, 12.580526213044704, 11.273612342435884], \
    # 65536
    [2.785625894239416, 46.54532337595768, 8.438038808911934, 26.361185565142055, 7.0822898742326155, 8.787536481516321]]


##### Draw the consumption breakdown plot

y_stage_1 = [r[0] for r in resource_utilization_cases]
y_stage_2 = [r[1] for r in resource_utilization_cases]
y_stage_3 = [r[2] for r in resource_utilization_cases]
y_stage_4 = [r[3] for r in resource_utilization_cases]
y_stage_5 = [r[4] for r in resource_utilization_cases]
y_stage_6 = [r[5] for r in resource_utilization_cases]

draw_resource_consumption_plot(x_labels, y_stage_1, y_stage_2, y_stage_3, 
    y_stage_4, y_stage_5, y_stage_6, 'resource_consumption_experiment_6_FPGA_model', x_tick_rotation=0)

##### Draw the total consumption plot

total_resource_consumption = [22.7, 22.7, 23.7, 26.3, 31.5, 26.4, 17.8]

plt.style.use('ggplot')


x = np.arange(len(x_labels))  # the label locations
width = 0.4  # the width of the bars

fig, ax = plt.subplots(1, 1, figsize=(8, 1.5))

rects = ax.bar(x, total_resource_consumption, width)

label_font = 10
tick_font = 10
tick_label_font = 9
legend_font = 8
title_font = 14

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Consumed Total\n FPGA Resource (%)', fontsize=9)
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=tick_label_font)


def number_single_bar(rects, annotate_threshold=0):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        if height > annotate_threshold:
            ax.annotate('{:.0f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=tick_font)


number_single_bar(rects, annotate_threshold=0)

ax.annotate('Severe waste of the available FPGA resoureces!',
    xy=(3, 60),
    xytext=(0, 5),  # 3 points vertical offset
    textcoords="offset points",
    ha='center', va='bottom', fontsize=tick_font)

ax.set(ylim=[0, 100])
plt.xticks(rotation=0)
plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/resource_consumption_experiment_6_FPGA_model_total_consumption.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()