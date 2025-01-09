import os

import numpy as np
import pandas as pd
import scanpy as sc
from plotnine import *

from staid import plot

# dataset_dir
dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
res_dir = "Z:/jxliu/project/ID-GAT/ID/results/Human_developing_heart"
sample_list = os.listdir(dataset_dir)

# *******************
# Part I
# *******************
for sample in sample_list:
    sample = sample
    adata = sc.read_h5ad(os.path.join(dataset_dir, sample))
    adata.var_names_make_unique()
    adata.obs[['x', 'y']] = np.around(adata.obs[['new_x', 'new_y']]).astype(int)

    # load deconvolution results
    result_list = os.listdir(res_dir)
    for res in result_list:
        if '_'.join(sample.split('_')[3:7]) in res:
            break
    res_df = pd.read_csv(os.path.join(res_dir, res),
                         index_col=0)
    adata.obsm['deconvolution'] = res_df

    # ************
    # Pie Plot
    # plot.scatter_pie(adata, pt_size=16, figsize=(8, 7))
    # ***********

    # ***********
    # Plot single cell types and multi
    # ***********
    # plot.scatter_cell_type(adata,
    #                        cell_type='Erythrocytes_1',
    #                        spatial_info=['x', 'y'],
    #                        size=80)
    # plot.scatter_cell_type(adata,
    #                        cell_type='Erythrocytes_2',
    #                        spatial_info=['x', 'y'],
    #                        size=80)
    plot.scatter_cell_type(adata,
                           cell_type=['Fibroblast_like_1',
                                      'Fibroblast_like_2',
                                      'Fibroblast_like_3'],
                           spatial_info=['x', 'y'],
                           vmax='auto',
                           size=100)
    plot.scatter_cell_type(adata,
                           cell_type=['Fibroblast_like_1',
                                      'Fibroblast_like_2',
                                      'Fibroblast_like_3'],
                           spatial_info=['x', 'y'],
                           n_col=3,
                           size=100)

# ******************************************************************
# component changes
# ******************************************************************
import plotnine

plotnine.options.figure_size = (8, 5)
sc_adata = sc.read(
    r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\scRNA-seq\scRNA-seq_development_heart_with_meta.h5ad")
sc_prop = sc_adata.obs['cell_type'].value_counts()
res_dir = r"Z:\jxliu\project\ID-GAT\ID\results\Human_developing_heart"
res_list = os.listdir(res_dir)
colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
          "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
          "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
proportion_df = pd.DataFrame()
for res in res_list:
    if 'att' in res:
        continue
    res_df = pd.read_csv(os.path.join(res_dir, res),
                         index_col=0)
    sample_name = res.split('_')[4] + "PCW" + '_' + \
                  res.split('_')[8]
    if 'FH5' in sample_name:
        sample_name = sample_name.replace('FH5', 'FH4.5')
    elif 'FH6' in sample_name:
        sample_name = sample_name.replace('FH6', 'FH6.5')
    sample_name = sample_name.replace('FH', '')
    proportion_df.loc[sample_name, res_df.columns] = res_df.mean(axis=0)
proportion_df['sample'] = proportion_df.index.tolist()
# plot
plot_prop_df = pd.melt(proportion_df, id_vars='sample')
plot_prop_df['sample'] = pd.Categorical(plot_prop_df['sample'],
                                        categories=['4.5PCW_1', '4.5PCW_2', '4.5PCW_3',
                                                    '4.5PCW_4', '6.5PCW_5', '6.5PCW_6',
                                                    '6.5PCW_7', '6.5PCW_8', '6.5PCW_9',
                                                    '6.5PCW_10', '6.5PCW_11', '6.5PCW_12',
                                                    '6.5PCW_13', '9PCW_14', '9PCW_15',
                                                    '9PCW_16', '9PCW_17', '9PCW_18',
                                                    '9PCW_19'],
                                        ordered=True)
plot_prop_df.columns = ["Sample", "Cell_type", 'Percentage']
base_plot = (ggplot(plot_prop_df, aes(x='Sample', y='Percentage',
                                      fill='Cell_type'))
             + geom_bar(stat='identity', color='black', position='fill',
                        width=0.9, size=0.02)
             + scale_fill_manual(colors)
             + scale_y_continuous(expand=[0, 0])
             + coord_flip()
             + theme(panel_background=element_blank(),
                     legend_text=element_text(size=12),
                     legend_key_size=16,
                     legend_key=element_blank(),
                     axis_text_x=element_text(size=10),
                     axis_text_y=element_text(size=10),
                     axis_title_x=element_text(size=12),
                     axis_title_y=element_text(size=12))
             )
print(base_plot)
res_fig_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig3"
save_file = os.path.join(res_fig_dir, "heart_proportion_changes.png")
base_plot.save(save_file, dpi=600)
save_file = os.path.join(res_fig_dir, "heart_proportion_changes.pdf")
base_plot.save(save_file, dpi=600)

# ******************************************************
# Other results
# ******************************************************
import plotnine

plotnine.options.figure_size = (15, 25)

method_list = ['GAT-ID', 'cell2location', 'SpatialDWLS', 'RCTD', 'Stereoscope',
               'DestVI', 'Tangram', 'SPOTlight']
dataset_other_dir = r"Z:\jxliu\project\ID-GAT\ID\results\Others\human_developing_heart"
res_other_list = os.listdir(dataset_other_dir)
all_classes = ['Atrial_cardiomyocytes',
               'Capillary_endothelium',
               'Cardiac_neural_crest_cells',
               'Endothelium_pericytes_adventitia',
               'Epicardial_cells',
               'Epicardium_derived_cells',
               'Erythrocytes_1',
               'Erythrocytes_2',
               'Fibroblast_like_1',
               'Fibroblast_like_2',
               'Fibroblast_like_3',
               'Immune_cells',
               'Myoz2_enriched_cardiomyocytes',
               'Smooth_muscle_cells',
               'Ventricular_cardiomyocytes']
all_colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
all_classes = np.sort(all_classes)
proportion_df = pd.DataFrame()
num = 0
for method in method_list:
    res_list = []
    for res in res_other_list:
        if method in res:
            res_list.append(res)
    for res in res_list:
        res_df = pd.read_csv(os.path.join(dataset_other_dir, res),
                             index_col=0)
        res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        res_df = res_df.loc[:, all_classes]
        res_df = res_df.dropna(0)
        sample_name = res.split('_')[4] + "PCW" + '_' + \
                      res.split('_')[8]
        if 'FH5' in sample_name:
            sample_name = sample_name.replace('FH5', 'FH4.5')
        elif 'FH6' in sample_name:
            sample_name = sample_name.replace('FH6', 'FH6.5')
        sample_name = sample_name.replace('FH', '')
        proportion_df.loc[str(num), all_classes] = res_df.mean(axis=0)
        proportion_df.loc[str(num), 'Method'] = method
        proportion_df.loc[str(num), 'Sample'] = sample_name
        num += 1
proportion_values = proportion_df.loc[:, all_classes].values
proportion_values[proportion_values < 0] = 0
proportion_df.loc[:, all_classes] = proportion_values
# plot
plot_prop_df = pd.melt(proportion_df, id_vars=['Sample', 'Method'])
plot_prop_df['Sample'] = pd.Categorical(plot_prop_df['Sample'],
                                        categories=['4.5PCW_1', '4.5PCW_2', '4.5PCW_3',
                                                    '4.5PCW_4', '6.5PCW_5', '6.5PCW_6',
                                                    '6.5PCW_7', '6.5PCW_8', '6.5PCW_9',
                                                    '6.5PCW_10', '6.5PCW_11', '6.5PCW_12',
                                                    '6.5PCW_13', '9PCW_14', '9PCW_15',
                                                    '9PCW_16', '9PCW_17', '9PCW_18',
                                                    '9PCW_19'],
                                        ordered=True)
plot_prop_df['Method'] = pd.Categorical(plot_prop_df['Method'],
                                        categories=method_list,
                                        ordered=True)
plot_prop_df.columns = ["Sample", "Method", "Cell_type", 'Percentage']
base_plot = (ggplot(plot_prop_df, aes(x='Sample', y='Percentage',
                                      fill='Cell_type'))
             + geom_bar(stat='identity', color='black', position='fill',
                        width=0.9, size=0.02)
             + scale_fill_manual(all_colors)
             + scale_y_continuous(expand=[0, 0])
             + coord_flip()
             + facet_grid('Method~.', margins=False)
             + theme(panel_background=element_blank(),
                     legend_text=element_text(size=15),
                     legend_key_size=20,
                     legend_key=element_blank(),
                     axis_text_x=element_text(size=10),
                     axis_text_y=element_text(size=10),
                     axis_title_x=element_text(size=12),
                     axis_title_y=element_text(size=12))
             )
print(base_plot)
base_plot.save(os.path.join(r"Z:\jxliu\project\ID-GAT\ID\figures\fig3",
                            "all_methods_percentages_barplot.pdf"),
               dpi=300)
base_plot.save(os.path.join(r"Z:\jxliu\project\ID-GAT\ID\figures\fig3",
                            "all_methods_percentages_barplot.png"),
               dpi=300)

# *************
# plot dominate cell types
# *************
import plotnine

plotnine.options.figure_size = (10, 10)
res_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig3\dominated"
method_list = ['GAT-ID', 'cell2location', 'RCTD', 'Stereoscope',
               'DestVI', 'Tangram', 'SPOTlight', 'SpatialDWLS']
res_other_dir = r"Z:\jxliu\project\ID-GAT\ID\results\Others\human_developing_heart"
res_other_list = os.listdir(res_other_dir)
all_dominated_cts = []
for sample in sample_list[9:10]:
    adata = sc.read_h5ad(os.path.join(dataset_dir, sample))
    adata.var_names_make_unique()
    adata.obs[['x', 'y']] = np.around(adata.obs[['new_x', 'new_y']]).astype(int)
    for method in method_list:
        for res in res_other_list:
            if method in res and sample.split('.h5')[0] in res:
                break
        res_df = pd.read_csv(os.path.join(res_other_dir, res),
                             index_col=0)
        max_cts_per_spot = res_df.idxmax(axis=1)
        adata.obs[method] = max_cts_per_spot
        all_dominated_cts.extend(max_cts_per_spot)
all_classes = ['Atrial_cardiomyocytes',
               'Capillary_endothelium',
               'Cardiac_neural_crest_cells',
               'Endothelium_pericytes_adventitia',
               'Epicardial_cells',
               'Epicardium_derived_cells',
               'Erythrocytes_1',
               'Erythrocytes_2',
               'Fibroblast_like_1',
               'Fibroblast_like_2',
               'Fibroblast_like_3',
               'Immune_cells',
               'Myoz2_enriched_cardiomyocytes',
               'Smooth_muscle_cells',
               'Ventricular_cardiomyocytes']
all_colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
# GAT-ID 
res_fig_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig3"
save_path = os.path.join(res_fig_dir, "GAT-ID_dominated_cell_types.png")
fig = plot.scatter_spots(adata, method_list[0],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors,
                         show_legend=True, return_fig=True, save_path=save_path)
# fig.get_figure().savefig(save_path, figsize=(10, 5))
save_path = os.path.join(res_fig_dir, "GAT-ID_dominated_cell_types.pdf")
fig = plot.scatter_spots(adata, method_list[0],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors,
                         show_legend=True, return_fig=True, save_path=save_path)
# cell2location
save_path = os.path.join(res_fig_dir, "cell2location_dominated_cell_types.png")
fig = plot.scatter_spots(adata, method_list[1],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors,
                         show_legend=False, return_fig=True, save_path=save_path)
save_path = os.path.join(res_fig_dir, "cell2location_dominated_cell_types.pdf")
fig = plot.scatter_spots(adata, method_list[1],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors,
                         show_legend=False, return_fig=True, save_path=save_path)
# others
save_path = os.path.join(res_fig_dir, "others_dominated_cell_types.png")
fig = plot.scatter_spots(adata, method_list[2:],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors, n_col=3,
                         show_legend=False, return_fig=True, save_path=save_path)
save_path = os.path.join(res_fig_dir, "others_dominated_cell_types.pdf")
fig = plot.scatter_spots(adata, method_list[2:],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors, n_col=3,
                         show_legend=False, return_fig=True, save_path=save_path)
# GAT-ID with cell2location
save_path = os.path.join(res_fig_dir,
                         "GAT-ID-cell2location_dominated_cell_types.pdf")
fig = plot.scatter_spots(adata, method_list[:2],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors,
                         show_legend=True, return_fig=True, save_path=save_path)
save_path = os.path.join(res_fig_dir,
                         "GAT-ID-cell2location_dominated_cell_types.png")
fig = plot.scatter_spots(adata, method_list[:2],
                         all_classes=all_classes, size=120,
                         cmap='tab20', all_colors=all_colors,
                         show_legend=True, return_fig=True, save_path=save_path)
