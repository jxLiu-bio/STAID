import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats
import sklearn.metrics as skm
from plotnine import *
from scipy.stats import pearsonr


def js_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


res_data_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\results"
method_list = ['GAT-ID', 'Cell2location', 'RCTD',
               'SpatialDWLS', 'Tangram', 'Stereoscope', 'DestVI', 'SPOTlight']
all_res_list = os.listdir(res_data_dir)
dataset_list = ['seqFISH+_OBcortex_cellType_100',
                'seqFISH+_OBcortex_cellType_150',
                'seqFISH+_OBcortex_cellType_200',
                'seqFISH+_SScortex_cellType_100',
                'seqFISH+_SScortex_cellType_150',
                'seqFISH+_SScortex_cellType_200']
fig_res_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2"
for dataset in dataset_list:
    # Load ground_truth
    gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
    sp_adata = sc.read(os.path.join(gd_dir, dataset + '.h5ad'))
    gd_df = sp_adata.obsm['cell_type_proprotion']
    # Define dataframes to store metrics
    rmse_spot_df = pd.DataFrame(columns=['Spot', 'Methods', 'RMSE (spot level)'])
    rmse_ct_df = pd.DataFrame(columns=['Cell type', 'Methods', 'RMSE (cell type level)'])
    mae_spot_df = pd.DataFrame(columns=['Spot', 'Methods', 'MAE (spot level)'])
    mae_ct_df = pd.DataFrame(columns=['Cell type', 'Methods', 'MAE (cell type level)'])
    pcc_spot_df = pd.DataFrame(columns=['Spot', 'Methods', 'PCC (spot level)'])
    pcc_ct_df = pd.DataFrame(columns=['Cell type', 'Methods', 'PCC (cell type level)'])
    js_spot_df = pd.DataFrame(columns=['Spot', 'Methods', 'JS (spot level)'])
    js_ct_df = pd.DataFrame(columns=['Cell type', 'Methods', 'JS (cell type level)'])
    # Calculate
    dataset_res_list = [i + "_" + dataset + "_results.csv" \
                        for i in method_list]
    num_spot = 0
    for index, res in enumerate(dataset_res_list):
        method = method_list[index]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        res_df = res_df.loc[gd_df.index, gd_df.columns]
        # spot level
        for spot in gd_df.index:
            print(method)
            # rmse
            rmse_spot = np.sqrt(skm.mean_squared_error(res_df.loc[spot, :],
                                                       gd_df.loc[spot, :]))
            rmse_spot_df.loc[str(num_spot), :] = [spot, method, rmse_spot]
            # mase
            mae_spot = skm.mean_absolute_error(res_df.loc[spot, :],
                                               gd_df.loc[spot, :])
            mae_spot_df.loc[str(num_spot), :] = [spot, method, mae_spot]
            # pcc
            pcc_spot = pearsonr(res_df.loc[spot, :],
                                gd_df.loc[spot, :])[0]
            pcc_spot_df.loc[str(num_spot), :] = [spot, method, pcc_spot]
            # js
            js_spot = js_divergence(res_df.loc[spot, :],
                                    gd_df.loc[spot, :])
            js_spot_df.loc[str(num_spot), :] = [spot, method, js_spot]
            num_spot += 1
    num_ct = 0
    for index, res in enumerate(dataset_res_list):
        method = res.split('_')[0]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        res_df = res_df.loc[gd_df.index, gd_df.columns]
        # spot level
        for ct in gd_df.columns:
            # rmse
            rmse_ct = np.sqrt(skm.mean_squared_error(res_df.loc[:, ct],
                                                     gd_df.loc[:, ct]))
            rmse_ct_df.loc[str(num_ct), :] = [ct, method, rmse_ct]
            # mase
            mae_ct = skm.mean_absolute_error(res_df.loc[:, ct],
                                             gd_df.loc[:, ct])
            mae_ct_df.loc[str(num_ct), :] = [ct, method, mae_ct]
            # pcc
            pcc_ct = pearsonr(res_df.loc[:, ct],
                              gd_df.loc[:, ct])[0]
            pcc_ct_df.loc[str(num_ct), :] = [ct, method, pcc_ct]
            # js
            js_ct = js_divergence(res_df.loc[:, ct],
                                  gd_df.loc[:, ct])
            js_ct_df.loc[str(num_ct), :] = [ct, method, js_ct]
            num_ct += 1

    # *************************************************************************
    # Plot spot level
    # *************************************************************************
    # rmse spot
    plot_df = rmse_spot_df.copy().loc[:, ['Methods', 'RMSE (spot level)']]
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    plot_df.loc[:, 'RMSE (spot level)'] = plot_df.loc[:, 'RMSE (spot level)'].astype(float)
    rmse_plot = (ggplot(plot_df, aes(x='Methods', y='RMSE (spot level)',
                                     fill='Methods'))
                 + geom_boxplot(show_legend=False)
                 + theme_classic()
                 + scale_fill_brewer(type='qualitative', palette='Paired')
                 + theme(axis_title=element_text(size=10),
                         axis_text=element_text(size=7.2)))
    print(rmse_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_RMSE.png")
    rmse_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_RMSE.pdf")
    rmse_plot.save(filename=file_name, dpi=300)

    # rmse ct
    plot_df = rmse_ct_df.copy().loc[:, ['Methods', 'RMSE (cell type level)']]
    plot_df.loc[plot_df['RMSE (cell type level)'] > 0.25, 'RMSE (cell type level)'] = 0.25
    plot_df.loc[:, 'RMSE (cell type level)'] = plot_df.loc[:, 'RMSE (cell type level)'].astype(float)
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    rmse_plot = (ggplot(plot_df, aes(x='Methods', y='RMSE (cell type level)',
                                     fill='Methods'))
                 + geom_boxplot(show_legend=False)
                 + theme_classic()
                 + scale_fill_brewer(type='qualitative', palette='Paired')
                 + theme(axis_title=element_text(size=10),
                         axis_text=element_text(size=7.2)))
    print(rmse_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_RMSE.png")
    rmse_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_RMSE.pdf")
    rmse_plot.save(filename=file_name, dpi=300)
    # *****************************************************************
    # mae spot
    plot_df = mae_spot_df.copy().loc[:, ['Methods', 'MAE (spot level)']]
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    plot_df.loc[:, 'MAE (spot level)'] = plot_df.loc[:, 'MAE (spot level)'].astype(float)
    mae_plot = (ggplot(plot_df, aes(x='Methods', y='MAE (spot level)',
                                    fill='Methods'))
                + geom_boxplot(show_legend=False)
                + theme_classic()
                + scale_fill_brewer(type='qualitative', palette='Paired')
                + theme(axis_title=element_text(size=10),
                        axis_text=element_text(size=7.2)))
    print(mae_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_MAE.png")
    mae_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_MAE.pdf")
    mae_plot.save(filename=file_name, dpi=300)

    # rmse ct
    plot_df = mae_ct_df.copy().loc[:, ['Methods', 'MAE (cell type level)']]
    plot_df.loc[plot_df['MAE (cell type level)'] > 0.2, 'MAE (cell type level)'] = 0.2
    plot_df.loc[:, 'MAE (cell type level)'] = plot_df.loc[:, 'MAE (cell type level)'].astype(float)
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    mae_plot = (ggplot(plot_df, aes(x='Methods', y='MAE (cell type level)',
                                    fill='Methods'))
                + geom_boxplot(show_legend=False)
                + theme_classic()
                + scale_fill_brewer(type='qualitative', palette='Paired')
                + theme(axis_title=element_text(size=10),
                        axis_text=element_text(size=7.2)))
    print(mae_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_MAE.png")
    mae_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_MAE.pdf")
    mae_plot.save(filename=file_name, dpi=300)

    # *****************************************************************
    # pcc spot
    plot_df = pcc_spot_df.copy().loc[:, ['Methods', 'PCC (spot level)']]
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    plot_df.loc[:, 'PCC (spot level)'] = plot_df.loc[:, 'PCC (spot level)'].astype(float)
    pcc_plot = (ggplot(plot_df, aes(x='Methods', y='PCC (spot level)',
                                    fill='Methods'))
                + geom_boxplot(show_legend=False)
                + theme_classic()
                + scale_fill_brewer(type='qualitative', palette='Paired')
                + theme(axis_title=element_text(size=10),
                        axis_text=element_text(size=7.2)))
    print(pcc_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_PCC.png")
    pcc_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_PCC.pdf")
    pcc_plot.save(filename=file_name, dpi=300)

    # PCC ct
    plot_df = pcc_ct_df.copy().loc[:, ['Methods', 'PCC (cell type level)']]
    # plot_df.loc[plot_df['MAE (cell type level)'] > 0.2, 'MAE (cell type level)'] = 0.2
    plot_df.loc[:, 'PCC (cell type level)'] = plot_df.loc[:, 'PCC (cell type level)'].astype(float)
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    pcc_plot = (ggplot(plot_df, aes(x='Methods', y='PCC (cell type level)',
                                    fill='Methods'))
                + geom_boxplot(show_legend=False)
                + theme_classic()
                + scale_fill_brewer(type='qualitative', palette='Paired')
                + theme(axis_title=element_text(size=10),
                        axis_text=element_text(size=7.2)))
    print(pcc_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_PCC.png")
    pcc_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_PCC.pdf")
    pcc_plot.save(filename=file_name, dpi=300)

    # *****************************************************************
    # JS spot
    plot_df = js_spot_df.copy().loc[:, ['Methods', 'JS (spot level)']]
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    plot_df.loc[:, 'JS (spot level)'] = plot_df.loc[:, 'JS (spot level)'].astype(float)
    plot_df.loc[plot_df['JS (spot level)'] > 5, 'JS (spot level)'] = 5
    js_plot = (ggplot(plot_df, aes(x='Methods', y='JS (spot level)',
                                   fill='Methods'))
               + geom_boxplot(show_legend=False)
               + theme_classic()
               + scale_fill_brewer(type='qualitative', palette='Paired')
               + theme(axis_title=element_text(size=10),
                       axis_text=element_text(size=7.2)))
    print(js_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_JS.png")
    js_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_spot_level_JS.pdf")
    js_plot.save(filename=file_name, dpi=300)

    # js ct
    plot_df = js_ct_df.copy().loc[:, ['Methods', 'JS (cell type level)']]
    # plot_df.loc[plot_df['JS (cell type level)'] > 10, 'JS (cell type level)'] = 10
    plot_df.loc[:, 'JS (cell type level)'] = plot_df.loc[:, 'JS (cell type level)'].astype(float)
    plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
                                        ordered=True)
    js_plot = (ggplot(plot_df, aes(x='Methods', y='JS (cell type level)',
                                   fill='Methods'))
               + geom_boxplot(show_legend=False)
               + theme_classic()
               + scale_fill_brewer(type='qualitative', palette='Paired')
               + theme(axis_title=element_text(size=10),
                       axis_text=element_text(size=7.2)))
    print(js_plot)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_JS.png")
    js_plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset + "_cellType_level_JS.pdf")
    js_plot.save(filename=file_name, dpi=300)

# &&&&&&&&&   
# Dataset 5
# &&&&&&&&&
dataset = dataset_list[5]
# Load ground_truth
gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
sp_adata = sc.read(os.path.join(gd_dir, dataset + '.h5ad'))
gd_df = sp_adata.obsm['cell_type_proprotion']
# Calculate
dataset_res_list = [i + "_" + dataset + "_results.csv" \
                    for i in method_list]
res_distri_fig = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2\distribution"
all_cell_types = gd_df.columns.tolist()
for cell_type in all_cell_types:
    pred_ct_df = pd.DataFrame(0, index=gd_df.index.tolist(),
                              columns=['ground_truth', 'GAT-ID',
                                       'Cell2location', 'RCTD',
                                       'SpatialDWLS', 'Tangram', 'Stereoscope',
                                       'DestVI', 'SPOTlight'])
    pred_ct_df.loc[gd_df.index, 'ground truth'] = gd_df[cell_type]
    old_method_list = ['ground truth']
    new_method_list = ['ground truth']
    for method_res in dataset_res_list:
        method_name = method_res.split('_')[0]
        old_method_list.append(method_name)
        tmp_df = pd.read_csv(os.path.join(res_data_dir, method_res),
                             index_col=0)
        pred_ct_df.loc[tmp_df.index, method_name] = tmp_df[cell_type]
        pcc = pearsonr(gd_df[cell_type], tmp_df[cell_type])[0]
        pcc = np.around(pcc * 1000) / 1000
        new_method_name = method_name + f": {pcc}"
        new_method_list.append(new_method_name)
    pred_ct_df = pred_ct_df.loc[:, old_method_list]
    pred_ct_df.columns = new_method_list
    sp_adata.obsm['gatid_deconvolution'] = pred_ct_df
    from staid import plot

    file_name = dataset + "_" + cell_type + ".png"
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=2400,
                           spatial_info=['x', 'y'],
                           vmax='auto',
                           prop_key='gatid_deconvolution',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
    file_name = dataset + "_" + cell_type + ".pdf"
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=2400,
                           spatial_info=['x', 'y'],
                           prop_key='gatid_deconvolution',
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
# &&&&&&&&&   
# Dataset 4
# &&&&&&&&&
dataset = dataset_list[4]
# Load ground_truth
gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
sp_adata = sc.read(os.path.join(gd_dir, dataset + '.h5ad'))
gd_df = sp_adata.obsm['cell_type_proprotion']
# Calculate
dataset_res_list = [i + "_" + dataset + "_results.csv" \
                    for i in method_list]
res_distri_fig = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2\distribution"
all_cell_types = gd_df.columns.tolist()
for cell_type in all_cell_types:
    pred_ct_df = pd.DataFrame(0, index=gd_df.index.tolist(),
                              columns=['ground_truth', 'GAT-ID',
                                       'Cell2location', 'RCTD',
                                       'SpatialDWLS', 'Tangram', 'Stereoscope',
                                       'DestVI', 'SPOTlight'])
    pred_ct_df.loc[gd_df.index, 'ground truth'] = gd_df[cell_type]
    old_method_list = ['ground truth']
    new_method_list = ['ground truth']
    for method_res in dataset_res_list:
        method_name = method_res.split('_')[0]
        old_method_list.append(method_name)
        tmp_df = pd.read_csv(os.path.join(res_data_dir, method_res),
                             index_col=0)
        pred_ct_df.loc[tmp_df.index, method_name] = tmp_df[cell_type]
        pcc = pearsonr(gd_df[cell_type], tmp_df[cell_type])[0]
        pcc = np.around(pcc * 1000) / 1000
        new_method_name = method_name + f": {pcc}"
        new_method_list.append(new_method_name)
    pred_ct_df = pred_ct_df.loc[:, old_method_list]
    pred_ct_df.columns = new_method_list
    sp_adata.obsm['gatid_deconvolution'] = pred_ct_df
    from staid import plot

    file_name = dataset + "_" + cell_type + ".png"
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=1050,
                           spatial_info=['x', 'y'],
                           vmax='auto',
                           n_col=3,
                           prop_key='gatid_deconvolution',
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
    file_name = dataset + "_" + cell_type + ".pdf"
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=1050,
                           spatial_info=['x', 'y'],
                           prop_key='gatid_deconvolution',
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
# &&&&&&&&&   
# Dataset 3
# &&&&&&&&&
dataset = dataset_list[3]
# Load ground_truth
gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
sp_adata = sc.read(os.path.join(gd_dir, dataset + '.h5ad'))
gd_df = sp_adata.obsm['cell_type_proprotion']
# Calculate
dataset_res_list = [i + "_" + dataset + "_results.csv" \
                    for i in method_list]
res_distri_fig = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2\distribution"
all_cell_types = gd_df.columns.tolist()
for cell_type in all_cell_types:
    pred_ct_df = pd.DataFrame(0, index=gd_df.index.tolist(),
                              columns=['ground_truth', 'GAT-ID',
                                       'Cell2location', 'RCTD',
                                       'SpatialDWLS', 'Tangram', 'Stereoscope',
                                       'DestVI', 'SPOTlight'])
    pred_ct_df.loc[gd_df.index, 'ground truth'] = gd_df[cell_type]
    old_method_list = ['ground truth']
    new_method_list = ['ground truth']
    for method_res in dataset_res_list:
        method_name = method_res.split('_')[0]
        old_method_list.append(method_name)
        tmp_df = pd.read_csv(os.path.join(res_data_dir, method_res),
                             index_col=0)
        pred_ct_df.loc[tmp_df.index, method_name] = tmp_df[cell_type]
        pcc = pearsonr(gd_df[cell_type], tmp_df[cell_type])[0]
        pcc = np.around(pcc * 1000) / 1000
        new_method_name = method_name + f": {pcc}"
        new_method_list.append(new_method_name)
    pred_ct_df = pred_ct_df.loc[:, old_method_list]
    pred_ct_df.columns = new_method_list
    sp_adata.obsm['gatid_deconvolution'] = pred_ct_df
    from staid import plot

    file_name = dataset + "_" + cell_type + ".png"
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=410,
                           spatial_info=['x', 'y'],
                           prop_key='gatid_deconvolution',
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
    file_name = dataset + "_" + cell_type + ".pdf"
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=410,
                           spatial_info=['x', 'y'],
                           prop_key='gatid_deconvolution',
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))

# &&&&&&&&&   
# Dataset 2
# &&&&&&&&&
dataset = dataset_list[2]
# Load ground_truth
gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
sp_adata = sc.read(os.path.join(gd_dir, dataset + '.h5ad'))
gd_df = sp_adata.obsm['cell_type_proprotion']
# Calculate
dataset_res_list = [i + "_" + dataset + "_results.csv" \
                    for i in method_list]
res_distri_fig = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2\distribution"
all_cell_types = gd_df.columns.tolist()
for cell_type in all_cell_types:
    pred_ct_df = pd.DataFrame(0, index=gd_df.index.tolist(),
                              columns=['ground_truth', 'GAT-ID',
                                       'Cell2location', 'RCTD',
                                       'SpatialDWLS', 'Tangram', 'Stereoscope',
                                       'DestVI', 'SPOTlight'])
    pred_ct_df.loc[gd_df.index, 'ground truth'] = gd_df[cell_type]
    old_method_list = ['ground truth']
    new_method_list = ['ground truth']
    for method_res in dataset_res_list:
        method_name = method_res.split('_')[0]
        old_method_list.append(method_name)
        tmp_df = pd.read_csv(os.path.join(res_data_dir, method_res),
                             index_col=0)
        pred_ct_df.loc[tmp_df.index, method_name] = tmp_df[cell_type]
        pcc = pearsonr(gd_df[cell_type], tmp_df[cell_type])[0]
        pcc = np.around(pcc * 1000) / 1000
        new_method_name = method_name + f": {pcc}"
        new_method_list.append(new_method_name)
    pred_ct_df = pred_ct_df.loc[:, old_method_list]
    pred_ct_df.columns = new_method_list
    sp_adata.obsm['gatid_deconvolution'] = pred_ct_df
    from staid import plot

    file_name = dataset + "_" + cell_type + ".png"
    file_name = file_name.replace('5/6', '5_or_6')
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=1730,
                           spatial_info=['x', 'y'],
                           vmax='auto',
                           n_col=3,
                           prop_key='gatid_deconvolution',
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
    file_name = dataset + "_" + cell_type + ".pdf"
    file_name = file_name.replace('5/6', '5_or_6')
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=1730,
                           spatial_info=['x', 'y'],
                           prop_key='gatid_deconvolution',
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
# &&&&&&&&&   
# Dataset 1
# &&&&&&&&&
dataset = dataset_list[1]
# Load ground_truth
gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
sp_adata = sc.read(os.path.join(gd_dir, dataset + '.h5ad'))
gd_df = sp_adata.obsm['cell_type_proprotion']
# Calculate
dataset_res_list = [i + "_" + dataset + "_results.csv" \
                    for i in method_list]
res_distri_fig = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2\distribution"
all_cell_types = gd_df.columns.tolist()
for cell_type in all_cell_types:
    pred_ct_df = pd.DataFrame(0, index=gd_df.index.tolist(),
                              columns=['ground_truth', 'GAT-ID',
                                       'Cell2location', 'RCTD',
                                       'SpatialDWLS', 'Tangram', 'Stereoscope',
                                       'DestVI', 'SPOTlight'])
    pred_ct_df.loc[gd_df.index, 'ground truth'] = gd_df[cell_type]
    old_method_list = ['ground truth']
    new_method_list = ['ground truth']
    for method_res in dataset_res_list:
        method_name = method_res.split('_')[0]
        old_method_list.append(method_name)
        tmp_df = pd.read_csv(os.path.join(res_data_dir, method_res),
                             index_col=0)
        pred_ct_df.loc[tmp_df.index, method_name] = tmp_df[cell_type]
        pcc = pearsonr(gd_df[cell_type], tmp_df[cell_type])[0]
        pcc = np.around(pcc * 1000) / 1000
        new_method_name = method_name + f": {pcc}"
        new_method_list.append(new_method_name)
    pred_ct_df = pred_ct_df.loc[:, old_method_list]
    pred_ct_df.columns = new_method_list
    sp_adata.obsm['gatid_deconvolution'] = pred_ct_df
    from staid import plot

    file_name = dataset + "_" + cell_type + ".png"
    file_name = file_name.replace('5/6', '5_or_6')
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=825,
                           spatial_info=['x', 'y'],
                           prop_key='gatid_deconvolution',
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
    file_name = dataset + "_" + cell_type + ".pdf"
    file_name = file_name.replace('5/6', '5_or_6')
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=825,
                           spatial_info=['x', 'y'],
                           prop_key='gatid_deconvolution',
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
# &&&&&&&&&   
# Dataset 0
# &&&&&&&&&
dataset = dataset_list[0]
# Load ground_truth
gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
sp_adata = sc.read(os.path.join(gd_dir, dataset + '.h5ad'))
gd_df = sp_adata.obsm['cell_type_proprotion']
# Calculate
dataset_res_list = [i + "_" + dataset + "_results.csv" \
                    for i in method_list]
res_distri_fig = r"Z:\jxliu\project\ID-GAT\ID\figures\fig2\distribution"
all_cell_types = gd_df.columns.tolist()
for cell_type in all_cell_types:
    pred_ct_df = pd.DataFrame(0, index=gd_df.index.tolist(),
                              columns=['ground_truth', 'GAT-ID',
                                       'Cell2location', 'RCTD',
                                       'SpatialDWLS', 'Tangram', 'Stereoscope',
                                       'DestVI', 'SPOTlight'])
    pred_ct_df.loc[gd_df.index, 'ground truth'] = gd_df[cell_type]
    old_method_list = ['ground truth']
    new_method_list = ['ground truth']
    for method_res in dataset_res_list:
        method_name = method_res.split('_')[0]
        old_method_list.append(method_name)
        tmp_df = pd.read_csv(os.path.join(res_data_dir, method_res),
                             index_col=0)
        pred_ct_df.loc[tmp_df.index, method_name] = tmp_df[cell_type]
        pcc = pearsonr(gd_df[cell_type], tmp_df[cell_type])[0]
        pcc = np.around(pcc * 1000) / 1000
        new_method_name = method_name + f": {pcc}"
        new_method_list.append(new_method_name)
    pred_ct_df = pred_ct_df.loc[:, old_method_list]
    pred_ct_df.columns = new_method_list
    sp_adata.obsm['gatid_deconvolution'] = pred_ct_df
    from staid import plot

    file_name = dataset + "_" + cell_type + ".png"
    file_name = file_name.replace('5/6', '5_or_6')
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=310,
                           prop_key='gatid_deconvolution',
                           spatial_info=['x', 'y'],
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
    file_name = dataset + "_" + cell_type + ".pdf"
    file_name = file_name.replace('5/6', '5_or_6')
    plot.scatter_cell_type(sp_adata, cell_type=new_method_list,
                           size=310,
                           prop_key='gatid_deconvolution',
                           spatial_info=['x', 'y'],
                           vmax='auto',
                           n_col=3,
                           save_path=os.path.join(res_distri_fig,
                                                  file_name))
# for dataset in all_res_list:
#     if 'RCTD' not in dataset:
#         continue
#     if 'OB' not in dataset:
#         continue
#     print(dataset)
#     tmp_df = pd.read_csv(os.path.join(res_data_dir, dataset), index_col=0)
#     colnames = tmp_df.columns.copy()
#     colnames = [i.replace('5.6', '5/6') for i in colnames]
#     colnames = [i.replace('.', ' ') for i in colnames]
#     tmp_df.columns = colnames
#     tmp_df.to_csv(os.path.join(res_data_dir, dataset))

# for dataset in all_res_list:
#     if 'RCTD' not in dataset:
#         continue
#     if 'SS' not in dataset:
#         continue
#     print(dataset)
#     tmp_df = pd.read_csv(os.path.join(res_data_dir, dataset), index_col=0)
#     colnames = tmp_df.columns.copy()
#     colnames = [i.replace('5.6', '5/6') for i in colnames]
#     colnames = [i.replace('.', ' ') for i in colnames]
#     tmp_df.columns = colnames
#     tmp_df.to_csv(os.path.join(res_data_dir, dataset))   

# for dataset in all_res_list:
#     if 'SPOTlight' not in dataset:
#         continue
#     if 'OB' not in dataset:
#         continue
#     print(dataset)
#     tmp_df = pd.read_csv(os.path.join(res_data_dir, dataset), index_col=0)
#     colnames = tmp_df.columns.copy()
#     colnames = [i.replace('5.6', '5/6') for i in colnames]
#     colnames = [i.replace('.', ' ') for i in colnames]
#     tmp_df.columns = colnames
#     tmp_df.to_csv(os.path.join(res_data_dir, dataset))

# for dataset in all_res_list:
#     if 'SPOTlight' not in dataset:
#         continue
#     if 'SS' not in dataset:
#         continue
#     print(dataset)
#     tmp_df = pd.read_csv(os.path.join(res_data_dir, dataset), index_col=0)
#     colnames = tmp_df.columns.copy()
#     colnames = [i.replace('5.6', '5/6') for i in colnames]
#     colnames = [i.replace('.', ' ') for i in colnames]
#     tmp_df.columns = colnames
#     tmp_df.to_csv(os.path.join(res_data_dir, dataset))   


# dataset_list = ['seqFISH+_OBcortex_cellType_100',
#                 'seqFISH+_OBcortex_cellType_150',
#                 'seqFISH+_OBcortex_cellType_200',
#                 'seqFISH+_SScortex_cellType_100',
#                 'seqFISH+_SScortex_cellType_150',
#                 'seqFISH+_SScortex_cellType_200']
# num_res = 0
# for dataset in dataset_list:
#     # Load ground_truth
#     gd_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\ST"
#     sp_adata = sc.read(os.path.join(gd_dir, dataset +'.h5ad'))
#     gd_df = sp_adata.obsm['cell_type_proprotion']
#     for res in all_res_list:
#         if dataset not in res:
#             continue
#         num_res += 1
#         tmp_df = pd.read_csv(os.path.join(res_data_dir, res),
#                              index_col=0)
#         diff_cts = np.setdiff1d(gd_df.columns, tmp_df.columns)
#         print(res)
#         print(f"Unpredicted cell types {diff_cts}")
#         if len(diff_cts) > 0:
#             tmp_df.loc[:, diff_cts] = 0
#         diff_cts = np.setdiff1d(tmp_df.columns, gd_df.columns, )
#         print(f"Additional cell types {diff_cts}")
#         tmp_df = tmp_df.loc[gd_df.index, gd_df.columns]
#         tmp_df.to_csv(os.path.join(res_data_dir, res))
