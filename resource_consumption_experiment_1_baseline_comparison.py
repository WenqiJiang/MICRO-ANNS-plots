import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from resource_consumption import draw_resource_consumption_plot


x_labels = ['SIFT100M\nR@1=25%', 'SIFT100M\nR@10=60%', 'SIFT100M\nR@100=95%',\
    'SIFT500M\nR@1=25%', 'SIFT500M\nR@10=60%', 'SIFT500M\nR@100=95%',\
    'SIFT1000M\nR@1=25%', 'SIFT1000M\nR@10=60%', 'SIFT1000M\nR@100=95%']

# Stage 1: OPQ
# Stage 2: vector quantizer
# Stage 3: select centroids
# Stage 4: construct distance LUT
# Stage 5: PQ code scan
# Stage 6: collect topK results

resource_utilization_cases = [
    # 100M, 1
    [0, 8.606278140845747, 0.11607633274229297, 3.3378707089447355, 78.57136070072978, 9.368414116737446], \
    # 100M, 10
    [0, 32.7008185883583, 0.5164703077320218, 4.674772663594282, 33.70847203114799, 28.399466409167403], \
    # 100M, 100
    [0.36579157321941314, 12.19452557895313, 1.217917661098765, 7.9088829508950145, 8.370031899846813, 69.94285033598688], \
    # 500M, 1
    [0, 52.028652124146305, 0.5870295775983332, 5.0630910789431605, 37.74095230234924, 4.5802749169629715], \
    # 500M, 10
    [0, 28.76389841123366, 0.45429136349850247, 4.11196697195551, 36.239180194975106, 30.430663058337203], \
    # 500M, 100
    [0.3682691665191441, 24.524467590082967, 4.259371745807707, 8.701728089373587, 6.554118756926367, 55.59204465129023], \
    # 1000M, 1
    [0, 38.91245617834552, 0.6145756921263626, 5.562762471047133, 49.02518744557251, 5.885018212908454], \
    # 1000M, 10
    [0, 28.76389841123366, 0.45429136349850247, 4.11196697195551, 36.239180194975106, 30.430663058337203], \
    # 1000M, 100
    [0.372256650548711, 24.790009567906957, 3.9700081996215912, 8.048666214396363, 6.625084361020105, 56.19397500650628]]

y_stage_1 = [r[0] for r in resource_utilization_cases]
y_stage_2 = [r[1] for r in resource_utilization_cases]
y_stage_3 = [r[2] for r in resource_utilization_cases]
y_stage_4 = [r[3] for r in resource_utilization_cases]
y_stage_5 = [r[4] for r in resource_utilization_cases]
y_stage_6 = [r[5] for r in resource_utilization_cases]

draw_resource_consumption_plot(x_labels, y_stage_1, y_stage_2, y_stage_3, y_stage_4, y_stage_5, y_stage_6, 'resource_consumption_experiment_1_VLDB_comparison')