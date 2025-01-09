# %%%%%%% s1
import os
from pathlib import Path
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


def save_plot(plot, fig_res_dir, dataset, plot_name):
    file_name = os.path.join(fig_res_dir, dataset, dataset + plot_name + ".png")
    plot.save(filename=file_name, dpi=300)
    file_name = os.path.join(fig_res_dir, dataset, dataset + plot_name + ".pdf")
    plot.save(filename=file_name, dpi=300)


def plot_boxplot(df, x, y, fig_res_dir, dataset, plot_name):
    plot_df = df.copy().loc[:, [x, y]]
    plot_df[x] = pd.Categorical(plot_df[x], categories=method_list, ordered=True)
    plot_df[y] = plot_df[y].astype(float)
    plot = (ggplot(plot_df, aes(x=x, y=y, fill=x))
            + geom_boxplot(show_legend=False)
            + theme_classic()
            + scale_fill_brewer(type='qualitative', palette='Paired')
            + theme(axis_title=element_text(size=9),
                    axis_text=element_text(size=9),
                    figure_size=(10, 6), ))
    print(plot)
    save_plot(plot, fig_res_dir, dataset, plot_name)


# %%%%%%%% s1 %%%%%%%%%%%%%
res_data_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/results/simulation/s1")
method_list = ['STAID', 'SONAR', 'RCTD', 'DSTG',
               'Stereoscope', 'SpatialDWLS', 'Cell2location',
               'DestVI', 'SPOTlight', 'Tangram']
all_res_list = os.listdir(res_data_dir)
dataset_list = ['Puck-191204-01_55_simulated_s1',
                'Puck-191204-01_100_simulated_s1',
                'Puck-200115-08_55_simulated_s1',
                'Puck-200115-08_100_simulated_s1', ]
fig_res_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/figures/fig2")

for dataset in dataset_list:
    # *************************************************************************
    # Calculate performance
    # *************************************************************************
    # Load ground_truth
    gd_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/data/simulation")
    sp_adata = sc.read(os.path.join(gd_dir, dataset[:-3] + '.h5ad'))
    gd_df = sp_adata.obsm['cell_type_proportion']
    # Define dataframes to save metrics
    rmse_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                         'RMSE (spot level)'])
    rmse_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                       'RMSE (cell type level)'])
    mae_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                        'MAE (spot level)'])
    mae_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                      'MAE (cell type level)'])
    pcc_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                        'PCC (spot level)'])
    pcc_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                      'PCC (cell type level)'])
    js_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                       'JS (spot level)'])
    js_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                     'JS (cell type level)'])

    dataset_res_list = [i + "_" + dataset + "_results.csv" \
                        for i in method_list]
    num_spot = 0
    for index, res in enumerate(dataset_res_list):
        method = method_list[index]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        if method == 'SPOTlight':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        if method == 'RCTD':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        res_df.loc[:, np.setdiff1d(gd_df.columns, res_df.columns)] = 0
        res_df = res_df.loc[gd_df.index, gd_df.columns]
        print(method)
        # spot level
        for spot in gd_df.index:
            # rmse
            rmse_spot = np.sqrt(skm.mean_squared_error(res_df.loc[spot, :],
                                                       gd_df.loc[spot, :]))
            rmse_spot_df.loc[str(num_spot), :] = [spot, method, rmse_spot]
            # mae
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
        method = method_list[index]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        if method == 'SPOTlight':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        if method == 'RCTD':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        res_df.loc[:, np.setdiff1d(gd_df.columns, res_df.columns)] = 0
        res_df = res_df.loc[gd_df.index, gd_df.columns]
        print(method)
        # cell type level
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

    if not os.path.exists(fig_res_dir / dataset):
        os.makedirs(fig_res_dir / dataset)
    plot_boxplot(rmse_spot_df, 'Methods', 'RMSE (spot level)',
                 fig_res_dir, dataset, "spot_level_RMSE")
    plot_boxplot(rmse_ct_df, 'Methods', 'RMSE (cell type level)',
                 fig_res_dir, dataset, "cellType_level_RMSE")
    # *************************************************************************
    # mae spot
    plot_boxplot(mae_spot_df, 'Methods', 'MAE (spot level)',
                 fig_res_dir, dataset, "spot_level_MAE")
    plot_boxplot(mae_ct_df, 'Methods', 'MAE (cell type level)',
                 fig_res_dir, dataset, "cellType_level_MAE")
    # *************************************************************************
    # pcc spot
    plot_boxplot(pcc_spot_df, 'Methods', 'PCC (spot level)',
                 fig_res_dir, dataset, "spot_level_PCC")
    plot_boxplot(pcc_ct_df, 'Methods', 'PCC (cell type level)',
                 fig_res_dir, dataset, "cellType_level_PCC")
    # *************************************************************************
    # js spot
    plot_boxplot(js_spot_df, 'Methods', 'JS (spot level)',
                 fig_res_dir, dataset, "spot_level_JS")
    plot_boxplot(js_ct_df, 'Methods', 'JS (cell type level)',
                 fig_res_dir, dataset, "cellType_level_JS")

# %%%%%%% s2
res_data_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/results/simulation/s2")
method_list = ['STAID', 'SONAR', 'RCTD',
               'Stereoscope', 'SpatialDWLS', 'Cell2location',
               'DestVI', 'SPOTlight', 'Tangram']
all_res_list = os.listdir(res_data_dir)
dataset_list = ['Puck-191204-01_55_simulated_s2_Pol',
                'Puck-191204-01_55_simulated_s2_End']
fig_res_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/figures/fig2")

for dataset in dataset_list:
    # *************************************************************************
    # Calculate performance
    # *************************************************************************
    # Load ground_truth
    gd_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/data/simulation")
    sp_adata = sc.read(os.path.join(gd_dir, 'Puck-191204-01_55_simulated.h5ad'))
    gd_df = sp_adata.obsm['cell_type_proportion']
    # Define dataframes to save metrics
    rmse_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                         'RMSE (spot level)'])
    rmse_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                       'RMSE (cell type level)'])
    mae_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                        'MAE (spot level)'])
    mae_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                      'MAE (cell type level)'])
    pcc_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                        'PCC (spot level)'])
    pcc_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                      'PCC (cell type level)'])
    js_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                       'JS (spot level)'])
    js_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                     'JS (cell type level)'])

    dataset_res_list = [i + "_" + dataset + "_results.csv" \
                        for i in method_list]
    num_spot = 0
    for index, res in enumerate(dataset_res_list):
        method = method_list[index]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        if method == 'SPOTlight':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        if method == 'RCTD':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        common_cts = np.intersect1d(gd_df.columns, res_df.columns)
        res_df = res_df.loc[gd_df.index, common_cts]
        gd_df = gd_df.loc[:, common_cts]
        print(method)
        # spot level
        for spot in gd_df.index:
            # rmse
            rmse_spot = np.sqrt(skm.mean_squared_error(res_df.loc[spot, :],
                                                       gd_df.loc[spot, :]))
            rmse_spot_df.loc[str(num_spot), :] = [spot, method, rmse_spot]
            # mae
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
    pcc_spot_df = pcc_spot_df.dropna()

    num_ct = 0
    for index, res in enumerate(dataset_res_list):
        method = method_list[index]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        if method == 'SPOTlight':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        if method == 'RCTD':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        common_cts = np.intersect1d(gd_df.columns, res_df.columns)
        res_df = res_df.loc[gd_df.index, common_cts]
        gd_df = gd_df.loc[:, common_cts]
        print(method)
        # cell type  level
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
    pcc_ct_df = pcc_ct_df.dropna()

    # *************************************************************************
    # Plot
    # *************************************************************************
    # rmse spot
    if not os.path.exists(fig_res_dir / dataset):
        os.makedirs(fig_res_dir / dataset)
    plot_boxplot(rmse_spot_df, 'Methods', 'RMSE (spot level)',
                 fig_res_dir, dataset, "spot_level_RMSE")
    plot_boxplot(rmse_ct_df, 'Methods', 'RMSE (cell type level)',
                 fig_res_dir, dataset, "cellType_level_RMSE")
    # *************************************************************************
    # mae spot
    plot_boxplot(mae_spot_df, 'Methods', 'MAE (spot level)',
                 fig_res_dir, dataset, "spot_level_MAE")
    plot_boxplot(mae_ct_df, 'Methods', 'MAE (cell type level)',
                 fig_res_dir, dataset, "cellType_level_MAE")
    # *************************************************************************
    # pcc spot
    plot_boxplot(pcc_spot_df, 'Methods', 'PCC (spot level)',
                 fig_res_dir, dataset, "spot_level_PCC")
    plot_boxplot(pcc_ct_df, 'Methods', 'PCC (cell type level)',
                 fig_res_dir, dataset, "cellType_level_PCC")
    # *************************************************************************
    # js spot
    plot_boxplot(js_spot_df, 'Methods', 'JS (spot level)',
                 fig_res_dir, dataset, "spot_level_JS")
    plot_boxplot(js_ct_df, 'Methods', 'JS (cell type level)',
                 fig_res_dir, dataset, "cellType_level_JS")

    # %%%%%%% s3
res_data_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/results/simulation/s3")
method_list = ['STAID', 'SONAR', 'RCTD',
               'Stereoscope', 'SpatialDWLS', 'Cell2location',
               'DestVI', 'SPOTlight', 'Tangram']
all_res_list = os.listdir(res_data_dir)
dataset_list = ['Puck-191204-01_55_simulated_s3']
fig_res_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/figures/fig2")

for dataset in dataset_list:
    # *************************************************************************
    # Calculate performance
    # *************************************************************************
    # Load ground_truth
    gd_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/data/simulation")
    sp_adata = sc.read(os.path.join(gd_dir, 'Puck-191204-01_55_simulated.h5ad'))
    gd_df = sp_adata.obsm['cell_type_proportion']
    # Define dataframes to save metrics
    rmse_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                         'RMSE (spot level)'])
    rmse_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                       'RMSE (cell type level)'])
    mae_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                        'MAE (spot level)'])
    mae_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                      'MAE (cell type level)'])
    pcc_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                        'PCC (spot level)'])
    pcc_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                      'PCC (cell type level)'])
    js_spot_df = pd.DataFrame(columns=['Spot', 'Methods',
                                       'JS (spot level)'])
    js_ct_df = pd.DataFrame(columns=['Cell type', 'Methods',
                                     'JS (cell type level)'])

    dataset_res_list = [i + "_" + dataset + "_results.csv" \
                        for i in method_list]
    num_spot = 0
    for index, res in enumerate(dataset_res_list):
        method = method_list[index]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        if method == 'SPOTlight':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        if method == 'RCTD':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        common_cts = np.intersect1d(gd_df.columns, res_df.columns)
        res_df = res_df.loc[gd_df.index, common_cts]
        gd_df = gd_df.loc[:, common_cts]
        print(method)
        # spot level
        for spot in gd_df.index:
            # rmse
            rmse_spot = np.sqrt(skm.mean_squared_error(res_df.loc[spot, :],
                                                       gd_df.loc[spot, :]))
            rmse_spot_df.loc[str(num_spot), :] = [spot, method, rmse_spot]
            # mae
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
    pcc_spot_df = pcc_spot_df.dropna()

    num_ct = 0
    for index, res in enumerate(dataset_res_list):
        method = method_list[index]
        res_df = pd.read_csv(os.path.join(res_data_dir, res), index_col=0)
        if method == 'SPOTlight':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        if method == 'RCTD':
            res_df.columns = [i.replace('Dcn.Cbln1.Ptgfr.Fezf2',
                                        'Dcn-Cbln1-Ptgfr-Fezf2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Pvrl3.Rgs15Calb2',
                                        'Pvrl3-Rgs15Calb2') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('Fibroblast.Like',
                                        'Fibroblast-Like') \
                              for i in res_df.columns]
            res_df.columns = [i.replace('.', '_') for i in res_df.columns]
        common_cts = np.intersect1d(gd_df.columns, res_df.columns)
        res_df = res_df.loc[gd_df.index, common_cts]
        gd_df = gd_df.loc[:, common_cts]
        print(method)
        # cell type  level
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
    pcc_ct_df = pcc_ct_df.dropna()

    # *************************************************************************
    # Plot
    # *************************************************************************
    # rmse spot
    if not os.path.exists(fig_res_dir / dataset):
        os.makedirs(fig_res_dir / dataset)
    plot_boxplot(rmse_spot_df, 'Methods', 'RMSE (spot level)',
                 fig_res_dir, dataset, "spot_level_RMSE")
    plot_boxplot(rmse_ct_df, 'Methods', 'RMSE (cell type level)',
                 fig_res_dir, dataset, "cellType_level_RMSE")
    # *************************************************************************
    # mae spot
    plot_boxplot(mae_spot_df, 'Methods', 'MAE (spot level)',
                 fig_res_dir, dataset, "spot_level_MAE")
    plot_boxplot(mae_ct_df, 'Methods', 'MAE (cell type level)',
                 fig_res_dir, dataset, "cellType_level_MAE")
    # *************************************************************************
    # pcc spot
    plot_boxplot(pcc_spot_df, 'Methods', 'PCC (spot level)',
                 fig_res_dir, dataset, "spot_level_PCC")
    plot_boxplot(pcc_ct_df, 'Methods', 'PCC (cell type level)',
                 fig_res_dir, dataset, "cellType_level_PCC")
    # *************************************************************************
    # js spot
    plot_boxplot(js_spot_df, 'Methods', 'JS (spot level)',
                 fig_res_dir, dataset, "spot_level_JS")
    plot_boxplot(js_ct_df, 'Methods', 'JS (cell type level)',
                 fig_res_dir, dataset, "cellType_level_JS")

    # # *************************************************************************
    # # Plot
    # # *************************************************************************
    # # rmse spot
    # if not os.path.exists(fig_res_dir / dataset):
    #     os.makedirs(fig_res_dir / dataset)
    #
    # plot_df = rmse_spot_df.copy().loc[:, ['Methods', 'RMSE (spot level)']]
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # plot_df.loc[:, 'RMSE (spot level)'] = plot_df.loc[:, 'RMSE (spot level)'].astype(float)
    # rmse_plot = (ggplot(plot_df, aes(x='Methods', y='RMSE (spot level)',
    #                                  fill='Methods'))
    #              + geom_boxplot(show_legend=False)
    #              + theme_classic()
    #              + scale_fill_brewer(type='qualitative', palette='Paired')
    #              + theme(axis_title=element_text(size=10),
    #                      axis_text=element_text(size=7.2)))
    # print(rmse_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_RMSE.png")
    # rmse_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_RMSE.pdf")
    # rmse_plot.save(filename=file_name, dpi=300)
    #
    # # rmse ct
    # plot_df = rmse_ct_df.copy().loc[:, ['Methods', 'RMSE (cell type level)']]
    # plot_df.loc[plot_df['RMSE (cell type level)'] > 0.25, 'RMSE (cell type level)'] = 0.25
    # plot_df.loc[:, 'RMSE (cell type level)'] = plot_df.loc[:, 'RMSE (cell type level)'].astype(float)
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # rmse_plot = (ggplot(plot_df, aes(x='Methods', y='RMSE (cell type level)',
    #                                  fill='Methods'))
    #              + geom_boxplot(show_legend=False)
    #              + theme_classic()
    #              + scale_fill_brewer(type='qualitative', palette='Paired')
    #              + theme(axis_title=element_text(size=10),
    #                      axis_text=element_text(size=7.2)))
    # print(rmse_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_RMSE.png")
    # rmse_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_RMSE.pdf")
    # rmse_plot.save(filename=file_name, dpi=300)
    # # *****************************************************************
    # # mae spot
    # plot_df = mae_spot_df.copy().loc[:, ['Methods', 'MAE (spot level)']]
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # plot_df.loc[:, 'MAE (spot level)'] = plot_df.loc[:, 'MAE (spot level)'].astype(float)
    # mae_plot = (ggplot(plot_df, aes(x='Methods', y='MAE (spot level)',
    #                                 fill='Methods'))
    #             + geom_boxplot(show_legend=False)
    #             + theme_classic()
    #             + scale_fill_brewer(type='qualitative', palette='Paired')
    #             + theme(axis_title=element_text(size=10),
    #                     axis_text=element_text(size=7.2)))
    # print(mae_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_MAE.png")
    # mae_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_MAE.pdf")
    # mae_plot.save(filename=file_name, dpi=300)
    #
    # # rmse ct
    # plot_df = mae_ct_df.copy().loc[:, ['Methods', 'MAE (cell type level)']]
    # plot_df.loc[plot_df['MAE (cell type level)'] > 0.2, 'MAE (cell type level)'] = 0.2
    # plot_df.loc[:, 'MAE (cell type level)'] = plot_df.loc[:, 'MAE (cell type level)'].astype(float)
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # mae_plot = (ggplot(plot_df, aes(x='Methods', y='MAE (cell type level)',
    #                                 fill='Methods'))
    #             + geom_boxplot(show_legend=False)
    #             + theme_classic()
    #             + scale_fill_brewer(type='qualitative', palette='Paired')
    #             + theme(axis_title=element_text(size=10),
    #                     axis_text=element_text(size=7.2)))
    # print(mae_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_MAE.png")
    # mae_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_MAE.pdf")
    # mae_plot.save(filename=file_name, dpi=300)
    #
    # # *****************************************************************
    # # pcc spot
    # plot_df = pcc_spot_df.copy().loc[:, ['Methods', 'PCC (spot level)']]
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # plot_df.loc[:, 'PCC (spot level)'] = plot_df.loc[:, 'PCC (spot level)'].astype(float)
    # pcc_plot = (ggplot(plot_df, aes(x='Methods', y='PCC (spot level)',
    #                                 fill='Methods'))
    #             + geom_boxplot(show_legend=False)
    #             + theme_classic()
    #             + scale_fill_brewer(type='qualitative', palette='Paired')
    #             + theme(axis_title=element_text(size=10),
    #                     axis_text=element_text(size=7.2)))
    # print(pcc_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_PCC.png")
    # pcc_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_PCC.pdf")
    # pcc_plot.save(filename=file_name, dpi=300)
    #
    # # PCC ct
    # plot_df = pcc_ct_df.copy().loc[:, ['Methods', 'PCC (cell type level)']]
    # # plot_df.loc[plot_df['MAE (cell type level)'] > 0.2, 'MAE (cell type level)'] = 0.2
    # plot_df.loc[:, 'PCC (cell type level)'] = plot_df.loc[:, 'PCC (cell type level)'].astype(float)
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # pcc_plot = (ggplot(plot_df, aes(x='Methods', y='PCC (cell type level)',
    #                                 fill='Methods'))
    #             + geom_boxplot(show_legend=False)
    #             + theme_classic()
    #             + scale_fill_brewer(type='qualitative', palette='Paired')
    #             + theme(axis_title=element_text(size=10),
    #                     axis_text=element_text(size=7.2)))
    # print(pcc_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_PCC.png")
    # pcc_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_PCC.pdf")
    # pcc_plot.save(filename=file_name, dpi=300)
    #
    # # *****************************************************************
    # # JS spot
    # plot_df = js_spot_df.copy().loc[:, ['Methods', 'JS (spot level)']]
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # plot_df.loc[:, 'JS (spot level)'] = plot_df.loc[:, 'JS (spot level)'].astype(float)
    # plot_df.loc[plot_df['JS (spot level)'] > 5, 'JS (spot level)'] = 5
    # js_plot = (ggplot(plot_df, aes(x='Methods', y='JS (spot level)',
    #                                fill='Methods'))
    #            + geom_boxplot(show_legend=False)
    #            + theme_classic()
    #            + scale_fill_brewer(type='qualitative', palette='Paired')
    #            + theme(axis_title=element_text(size=10),
    #                    axis_text=element_text(size=7.2)))
    # print(js_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_JS.png")
    # js_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_spot_level_JS.pdf")
    # js_plot.save(filename=file_name, dpi=300)
    #
    # # js ct
    # plot_df = js_ct_df.copy().loc[:, ['Methods', 'JS (cell type level)']]
    # # plot_df.loc[plot_df['JS (cell type level)'] > 10, 'JS (cell type level)'] = 10
    # plot_df.loc[:, 'JS (cell type level)'] = plot_df.loc[:, 'JS (cell type level)'].astype(float)
    # plot_df['Methods'] = pd.Categorical(plot_df['Methods'], categories=method_list,
    #                                     ordered=True)
    # js_plot = (ggplot(plot_df, aes(x='Methods', y='JS (cell type level)',
    #                                fill='Methods'))
    #            + geom_boxplot(show_legend=False)
    #            + theme_classic()
    #            + scale_fill_brewer(type='qualitative', palette='Paired')
    #            + theme(axis_title=element_text(size=10),
    #                    axis_text=element_text(size=7.2)))
    # print(js_plot)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_JS.png")
    # js_plot.save(filename=file_name, dpi=300)
    # file_name = os.path.join(fig_res_dir / dataset, dataset + "_cellType_level_JS.pdf")
    # js_plot.save(filename=file_name, dpi=300)
