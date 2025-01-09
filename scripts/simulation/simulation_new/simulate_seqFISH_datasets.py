# -*- coding: utf-8 -*-
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_dir = Path("Z:/jxliu/project/ID-GAT/ID/simulation/rawdata/Dataset4_seqFISH+/Rawdata")
# anno
anno = pd.read_csv(data_dir / "Spatial_annotate.txt", index_col=0, sep='\t')
# Load FISH data
exp = pd.read_csv(data_dir / "Spatial_count.txt", index_col=0, sep='\t')
# Load location data
spa = pd.read_csv(data_dir / "Locations_seqfish.txt", sep='\t')
spa.columns = ['x', 'y']
# alignment
cell_list = ['cell_' + str(i) for i in range(1, anno.shape[0] + 1)]
anno.index = cell_list
exp.index = cell_list
spa.index = cell_list
anno = anno.iloc[:-1, :]
exp = exp.iloc[:-1, :]
spa = spa.iloc[:-1, :]

# obtain the boundary of spatial data
x_max, y_max = spa.max()
x_min, y_min = spa.min()
x_range = x_max - x_min
y_range = y_max - y_min

resolution_list = [200, 300, 400, 500, 600]
res_fig_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2\simulation_process"
save_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\simulated_datasets"
for resolution in resolution_list:
    spatial_meta = anno
    spatial_loc = spa
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = sns.color_palette('tab20', n_colors=len(np.unique(np.array(spatial_meta.celltype))))
    for i, c in enumerate(np.unique(spatial_meta.celltype)):
        ax.scatter(x=spatial_loc[spatial_meta['celltype'] == c]['x'][:-1],
                   y=spatial_loc[spatial_meta['celltype'] == c]['y'][:-1],
                   c=matplotlib.colors.to_hex(cmap[i]),
                   label=c,
                   s=50, marker='o', edgecolors='none')
    ax.set_title('seqFISH+_SVZ', fontsize=16)
    ax.legend(bbox_to_anchor=(1, 1.02), prop={'size': 10})
    plt.xticks(np.arange(x_min, x_max + resolution, resolution))
    plt.yticks(np.arange(y_min, y_max + resolution, resolution))
    # plt.xlim(-2000,1500)
    # plt.ylim(0,8000)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    # ax.invert_yaxis()
    # ax.invert_xaxis()
    plt.grid()
    ax.set_aspect('equal')
    filename = "seqFISH+_SVZ" + f"_{resolution}.png"
    plt.savefig(os.path.join(res_fig_dir, filename),
                dpi=300)
    filename = "seqFISH+_SVZ" + f"_{resolution}.pdf"
    plt.savefig(os.path.join(res_fig_dir, filename),
                dpi=300)
    plt.show()

    spot_1 = [np.floor(x_min), np.floor(y_min)]
    cur_x = x_min
    cur_y = y_min
    num_x = int(np.floor((x_max - x_min) / resolution))
    num_y = int(np.floor((y_max - y_min) / resolution))

    cell_ID = 1
    pseudo_exp = pd.DataFrame(columns=exp.columns)
    pseudo_anno = pd.DataFrame(columns=np.unique(anno.loc[:, 'celltype']))
    pseudo_spa = pd.DataFrame(columns=['x', 'y'])
    pseudo_cellNumber = pd.DataFrame(columns=['cell_number'])

    for ii in range(num_x - 1):
        for jj in range(num_y - 1):
            # select cells
            spot_1_t = [spot_1[0] + ii * resolution, spot_1[1] + jj * resolution]
            spot_2_t = [spot_1[0] + (ii + 1) * resolution, spot_1[1] + (jj + 1) * resolution]
            pseudo_name = "pseudo_" + str(cell_ID) + "_" + str(spot_1_t[0] + resolution / 2) + "_" + str(
                spot_1_t[1] + resolution / 2)
            cell_ID += 1
            appro_x_index = np.array((spa.iloc[:, 0] >= spot_1_t[0]).tolist()) & np.array(
                (spa.iloc[:, 0] < spot_2_t[0]).tolist())
            appro_y_index = np.array((spa.iloc[:, 1] >= spot_1_t[1]).tolist()) & np.array(
                (spa.iloc[:, 1] < spot_2_t[1]).tolist())
            choosed_cells_index = spa.index[appro_x_index & appro_y_index]

            # Extract data
            exp_temp = exp.loc[choosed_cells_index, :]
            anno_temp = anno.loc[choosed_cells_index, :]

            # Obtain cell type abundunce
            pseudo_anno.loc[pseudo_name, :] = np.zeros(pseudo_anno.shape[1])
            for kk in anno_temp.loc[:, 'celltype']:
                pseudo_anno.loc[pseudo_name, kk] += 1

            # Obtain the gene expression of pseudo_spots
            exp_simu = exp_temp.values.sum(axis=0)
            pseudo_exp.loc[pseudo_name, :] = exp_simu

            # Obtain the cell number for each pseudo-spot
            pseudo_cellNumber.loc[pseudo_name, :] = len(choosed_cells_index)

            # Obtain the spaital information for each pseudo-spot
            pseudo_spa.loc[pseudo_name, :] = [spot_1_t[0] + resolution / 2,
                                              spot_1_t[1] + resolution / 2]
            print(pseudo_name)

    # Normlization aimming to cell type
    pseudo_anno_norm = pseudo_anno.div((pseudo_anno.sum(axis=1) + 0.0000001),
                                       axis=0)

    # QC
    pre_index = pseudo_cellNumber.iloc[:, 0] > 2
    pseudo_anno_norm = pseudo_anno_norm.loc[pre_index, :]
    pseudo_exp = pseudo_exp.loc[pre_index, :]
    pseudo_cellNumber = pseudo_cellNumber.loc[pre_index, :]
    pseudo_spa = pseudo_spa.loc[pre_index, :]

    # Save results
    save_name = 'seqFISH+_'
    pseudo_anno_norm.to_csv(save_dir + "\\" + save_name + "_" + str(resolution) + "_simulation_annotation.csv")
    pseudo_exp.to_csv(save_dir + "\\" + save_name + "_" + str(resolution) + "_simulation_expression.csv")
    pseudo_cellNumber.to_csv(save_dir + "\\" + save_name + "_" + str(resolution) + "_simulation_cellNumber.csv")
    pseudo_spa.to_csv(save_dir + "\\" + save_name + "_" + str(resolution) + "_simulation_location.csv")
