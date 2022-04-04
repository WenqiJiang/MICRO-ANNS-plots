import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from resource_consumption import draw_resource_consumption_plot

x_labels = ['nprobe=1', 'nprobe=2', 'nprobe=4', 'nprobe=8', 'nprobe=16', 'nprobe=32', 'nprobe=64']

# Stage 1: OPQ
# Stage 2: vector quantizer
# Stage 3: select centroids
# Stage 4: construct distance LUT
# Stage 5: PQ code scan
# Stage 6: collect topK results

resource_utilization_cases = [
    # 1
    [1.156418685254055, 77.01039116345662, 0.5215007627688211, 3.9792195001423227, 8.820379321449842, 8.51209056692835], \
    # 2
    [1.0586572185992036, 70.50007713636239, 0.7954392207443292, 6.860610395253016, 10.7662946630033, 10.018921366037752], \
    # 4
    [0.8352745228824207, 55.6241597928809, 1.1294370950953057, 10.490613292038132, 16.989090481526702, 14.931424815576536], \
    # 8
    [0.529585773528025, 35.267164128392785, 1.3524518172511055, 13.090021591776352, 26.92881315578997, 22.83196353326175], \
    # 16
    [0.39660218084757204, 19.816471235044624, 1.965967330964262, 11.076848270289252, 36.30015722917499, 30.44395375367931], \
    # 32
    [0.44475823690394695, 11.129289524757223, 1.4363045649844899, 12.141362520165508, 40.70777899930141, 34.14050615388743], \
    # 64
    [0.45670719292181955, 5.732609059109753, 2.9383250467435147, 14.01318584973408, 41.8014416242681, 35.05773122722274]]

y_stage_1 = [r[0] for r in resource_utilization_cases]
y_stage_2 = [r[1] for r in resource_utilization_cases]
y_stage_3 = [r[2] for r in resource_utilization_cases]
y_stage_4 = [r[3] for r in resource_utilization_cases]
y_stage_5 = [r[4] for r in resource_utilization_cases]
y_stage_6 = [r[5] for r in resource_utilization_cases]

draw_resource_consumption_plot(x_labels, y_stage_1, y_stage_2, y_stage_3, y_stage_4, y_stage_5, y_stage_6, 
    'resource_consumption_experiment_4_nprobe', 
    x_tick_rotation=30, title="FPGA,SIFT100M,OPQ+IVF8192")

