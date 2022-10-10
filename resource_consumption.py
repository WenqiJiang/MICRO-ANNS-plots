import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def draw_resource_consumption_plot(
    x_labels, y_stage_1, y_stage_2, y_stage_3, y_stage_4, y_stage_5, y_stage_6, filename,
    x_tick_rotation=45, title='Title'):

    """
    Example input:


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

    filename = resource_consumption_experiment_1_VLDB_comparison
    """

    # assert input values are correct -> sum to 100
    y_all = np.array(y_stage_1) + np.array(y_stage_2) + np.array(y_stage_3) + \
        np.array(y_stage_4) + np.array(y_stage_5) + np.array(y_stage_6)
    for y in y_all:
        assert np.isclose(y, 100)

    # style = 'grayscale'
    style = 'ggplot'
    plt.style.use(style)


    x = np.arange(len(x_labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(1, 1, figsize=(6, 1.6))


    bottom_stage_1 = np.zeros(len(y_stage_1))
    bottom_stage_2 = y_stage_1 + bottom_stage_1
    bottom_stage_3 = y_stage_2 + bottom_stage_2
    bottom_stage_4 = y_stage_3 + bottom_stage_3
    bottom_stage_5 = y_stage_4 + bottom_stage_4
    bottom_stage_6 = y_stage_5 + bottom_stage_5

    if style == 'grayscale':
        rects_stage1 = ax.bar(x, y_stage_1, width, bottom=bottom_stage_1, color='#000000')
        rects_stage2 = ax.bar(x, y_stage_2, width, bottom=bottom_stage_2, color='#222222')
        rects_stage3 = ax.bar(x, y_stage_3, width, bottom=bottom_stage_3, color='#555555')
        rects_stage4 = ax.bar(x, y_stage_4, width, bottom=bottom_stage_4, color='#888888')
        rects_stage5 = ax.bar(x, y_stage_5, width, bottom=bottom_stage_5, color='#AAAAAA')
        rects_stage6 = ax.bar(x, y_stage_6, width, bottom=bottom_stage_6, color='#CCCCCC')
    else:
        rects_stage1 = ax.bar(x, y_stage_1, width, bottom=bottom_stage_1)
        rects_stage2 = ax.bar(x, y_stage_2, width, bottom=bottom_stage_2)
        rects_stage3 = ax.bar(x, y_stage_3, width, bottom=bottom_stage_3)
        rects_stage4 = ax.bar(x, y_stage_4, width, bottom=bottom_stage_4)
        rects_stage5 = ax.bar(x, y_stage_5, width, bottom=bottom_stage_5)
        rects_stage6 = ax.bar(x, y_stage_6, width, bottom=bottom_stage_6)



    label_font = 10
    tick_font = 10
    tick_label_font = 9
    legend_font = 9
    title_font = 11

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Resource Consumption (%)', fontsize=label_font)
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=tick_label_font)


    ax.legend([rects_stage1, rects_stage2, rects_stage3, rects_stage4, rects_stage5, rects_stage6], 
        ["Stage OPQ", "Stage IVFDist", "Stage SelCells", \
        "Stage BuildLUT", "Stage PQDist", "Stage SelK"], loc=(0.0, 1.05), ncol=3, \
        facecolor='white', framealpha=1, frameon=False, fontsize=legend_font)


    def number_single_bar(rects, bottom, annotate_threshold=20, color='black'):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for i, rect in enumerate(rects):
            height = rect.get_height()
            if height > annotate_threshold:
                ax.annotate('{:.0f}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height + bottom[i]),
                            xytext=(0, -20),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=tick_font, color=color)


    if style == 'grayscale':
        number_single_bar(rects_stage1, bottom_stage_1, annotate_threshold=20, color='white')
        number_single_bar(rects_stage2, bottom_stage_2, annotate_threshold=20, color='white')
        number_single_bar(rects_stage3, bottom_stage_3, annotate_threshold=20, color='black')
        number_single_bar(rects_stage4, bottom_stage_4, annotate_threshold=20, color='black')
        number_single_bar(rects_stage5, bottom_stage_5, annotate_threshold=20, color='black')
        number_single_bar(rects_stage6, bottom_stage_6, annotate_threshold=20, color='black')
    else:
        color = 'black'
        number_single_bar(rects_stage1, bottom_stage_1, annotate_threshold=20, color=color)
        number_single_bar(rects_stage2, bottom_stage_2, annotate_threshold=20, color=color)
        number_single_bar(rects_stage3, bottom_stage_3, annotate_threshold=20, color=color)
        number_single_bar(rects_stage4, bottom_stage_4, annotate_threshold=20, color=color)
        number_single_bar(rects_stage5, bottom_stage_5, annotate_threshold=20, color=color)
        number_single_bar(rects_stage6, bottom_stage_6, annotate_threshold=20, color=color)

    ax.set_title(title, fontsize=title_font, y=1.35)

    ax.set(ylim=[0, 100])
    plt.xticks(rotation=x_tick_rotation)
    plt.rcParams.update({'figure.autolayout': True})

    plt.savefig('./out_img/{}.png'.format(filename), transparent=False, dpi=200, bbox_inches="tight")
    plt.show()