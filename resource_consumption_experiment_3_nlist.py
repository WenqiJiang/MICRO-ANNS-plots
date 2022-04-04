import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from resource_consumption import draw_resource_consumption_plot

x_labels = []

x_labels = ['nlist=1024', 'nlist=2048', 'nlist=4096', 'nlist=8192', 'nlist=16384', 'nlist=32768', 'nlist=65536']

# Stage 1: OPQ
# Stage 2: vector quantizer
# Stage 3: select centroids
# Stage 4: construct distance LUT
# Stage 5: PQ code scan
# Stage 6: collect topK results

resource_utilization_cases = [
    # 1024, OPQ
    [0.5409868696553979, 2.2926581093202736, 0.880320380870845, 5.243449873267792, 49.5153818505039, 41.527202916381775], \
    # 2048, OPQ
    [0.5196278556653465, 4.3622661898002555, 0.8455639452059186, 6.824461935255431, 47.560436558871736, 39.88764351520131], \
    # 4096, OPQ
    [0.49164799264830905, 8.214999312104041, 0.8000337391150346, 7.753965088514687, 44.99949898510786, 37.73985488251007], \
    # 8192, OPQ
    [0.39660218084757204, 19.816471235044624, 1.965967330964262, 11.076848270289252, 36.30015722917499, 30.44395375367931], \
    # 16384, OPQ
    [0.6861044069391512, 45.69034505504171, 3.401037399215397, 14.44071236426669, 19.188180776098434, 16.593619998438626], \
    # 32768, OPQ
    [0.9933217575037567, 66.14913618395138, 4.923921800467693, 13.045656520598827, 7.576386304677817, 7.311577432800538], \
    # 65536, OPQ
    [1.02847509164277, 68.49013261306942, 5.09817779256556, 9.968370567604508, 7.844512153450014, 7.570331781667738]]

y_stage_1 = [r[0] for r in resource_utilization_cases]
y_stage_2 = [r[1] for r in resource_utilization_cases]
y_stage_3 = [r[2] for r in resource_utilization_cases]
y_stage_4 = [r[3] for r in resource_utilization_cases]
y_stage_5 = [r[4] for r in resource_utilization_cases]
y_stage_6 = [r[5] for r in resource_utilization_cases]

draw_resource_consumption_plot(x_labels, y_stage_1, y_stage_2, y_stage_3, y_stage_4, y_stage_5, y_stage_6, 
    'resource_consumption_experiment_3_nlist', 
    x_tick_rotation=30, title="FPGA,SIFT100M,nprobe=16")

