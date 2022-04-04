import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from resource_consumption import draw_resource_consumption_plot

x_labels = ['K=1', 'K=10', 'K=20', 'K=50',\
    'K=100', 'K=200', 'K=500', 'K=1000']

# Stage 1: OPQ
# Stage 2: vector quantizer
# Stage 3: select centroids
# Stage 4: construct distance LUT
# Stage 5: PQ code scan
# Stage 6: collect topK results

resource_utilization_cases = [
    # 1
    [0.48273456750078353, 24.120128766305413, 2.537943340779828, 15.143553864967657, 51.54761741329463, 6.168022047151682], \
    # 10
    [0.39725708749009564, 18.197771228984028, 2.088551447626425, 12.462095129818247, 36.36009944644477, 30.49422565963645], \
    # 20
    [0.5243961202192401, 17.481982554260558, 0.9058333780551058, 10.28542557543397, 26.66492539429556, 44.13743697773556], \
    # 50
    [0.4988666200115105, 12.48325626748134, 0.8617341322339166, 7.8678127709993815, 15.220068217228503, 63.06826199204535], \
    # 100
    [0.3986947342078026, 8.319232502261695, 0.6886988365765292, 5.236203193733794, 9.122921051050472, 76.23424968216972], \
    # 200
    [0.38679852558105765, 4.85511234804625, 0.6681495181683239, 3.74899797703053, 4.917062421691048, 85.42387920948279], \
    # 500
    [0.37134564693487787, 1.573732481434055, 0.6414564654834709, 2.470518414110895, 1.8882488998727027, 93.05469809216399], \
    # 1000
    [0.3157408260600662, 1.3380837981727713, 0.5454055970900863, 2.100586155530654, 0.8027524657605214, 94.89743115738591]]

y_stage_1 = [r[0] for r in resource_utilization_cases]
y_stage_2 = [r[1] for r in resource_utilization_cases]
y_stage_3 = [r[2] for r in resource_utilization_cases]
y_stage_4 = [r[3] for r in resource_utilization_cases]
y_stage_5 = [r[4] for r in resource_utilization_cases]
y_stage_6 = [r[5] for r in resource_utilization_cases]

draw_resource_consumption_plot(x_labels, y_stage_1, y_stage_2, y_stage_3, y_stage_4, y_stage_5, y_stage_6, 
    'resource_consumption_experiment_5_topK', x_tick_rotation=0, title="FPGA,SIFT100M,OPQ+IVF8192")

