import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from plotnine import *

data_dir = "/home/jxliu/Desktop/projects/staid/data/simulation"
res_dir = "/home/jxliu/Desktop/projects/staid/results/simulation_iteration"
fig_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/figures/fig2/iteration_process")
dataset_name = "Puck-191204-01_55_simulated"
sp_adata = sc.read(os.path.join(data_dir, 'Puck-191204-01_55_simulated.h5ad'))
interation_list = os.listdir(res_dir)
gd_df = sp_adata.obsm['cell_type_proportion'].copy()

res_list = []
res_index = []
for i in interation_list:
    index_i = i.split("STAID_iteration_")[1].split('Puck')[0]
    res_index.append(index_i)
    tmp = pd.read_csv(os.path.join(res_dir, i),
                      index_col=0)
    tmp = tmp.loc[gd_df.index, gd_df.columns]
    res_list.append(tmp)

# %%%%
# ********************
# Plot RMSE
# ********************
# RMSE (spot level)
from sklearn.metrics import mean_squared_error

mse_df_spot = pd.DataFrame(0,
                           index=gd_df.index,
                           columns=res_index)
num = 0
for prediction in res_list:
    for spot in prediction.index:
        tmp = np.sqrt(mean_squared_error(gd_df.loc[spot, :],
                                         prediction.loc[spot, :]))
        mse_df_spot.loc[spot, res_index[num]] = tmp
    num += 1
mean = mse_df_spot.mean(axis=0)
sd = np.sqrt(mse_df_spot.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['RMSE (spot level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'RMSE (spot level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
plot_df.loc[:, 'group'] = '1'
base_plot = (ggplot(plot_df, aes(x='Iterations', y='RMSE (spot level)'))
             + geom_bar(stat="identity", fill='#f39c8e', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + geom_line(group='group', color='r')
             + theme_classic()
             + coord_cartesian(ylim=[0.04, 0.07]))
print(base_plot)
# save
file_name = dataset_name + "_spot_level_iteration_RMSE.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_spot_level_iteration_RMSE.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# RMSE (cell type level)
mse_df_ct = pd.DataFrame(0,
                         index=gd_df.columns,
                         columns=res_index)
num = 0
for prediction in res_list:
    for ct in prediction.columns:
        tmp = np.sqrt(mean_squared_error(gd_df.loc[:, ct],
                                         prediction.loc[:, ct]))
        mse_df_ct.loc[ct, res_index[num]] = tmp
    num += 1
mean = mse_df_ct.mean(axis=0)
sd = np.sqrt(mse_df_ct.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['RMSE (cell type level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'RMSE (cell type level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
plot_df.loc[:, 'group'] = '1'
base_plot = (ggplot(plot_df, aes(x='Iterations', y='RMSE (cell type level)'))
             + geom_bar(stat="identity", fill='#f39c8e', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + geom_line(group='group', color='r')
             + theme_classic()
             + coord_cartesian(ylim=[0.04, 0.06]))
print(base_plot)
# save
file_name = dataset_name + "_cellType_level_iteration_RMSE.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_cellType_level_iteration_RMSE.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# %%%%%%
# ********************
# Plot MAE
# ********************
from sklearn.metrics import mean_absolute_error

mae_df_spot = pd.DataFrame(0,
                           index=gd_df.index,
                           columns=res_index)
num = 0
for prediction in res_list:
    for spot in prediction.index:
        tmp = mean_absolute_error(gd_df.loc[spot, :],
                                  prediction.loc[spot, :])
        mae_df_spot.loc[spot, res_index[num]] = tmp
    num += 1
mean = mae_df_spot.mean(axis=0)
sd = np.sqrt(mae_df_spot.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['MAE (spot level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'MAE (spot level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
plot_df.loc[:, 'group'] = '1'
base_plot = (ggplot(plot_df, aes(x='Iterations', y='MAE (spot level)'))
             + geom_bar(stat="identity", fill='#9cdae7', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + geom_line(group='group', color='r')
             + theme_classic()
             + coord_cartesian(ylim=[0.015, 0.03]))
print(base_plot)
# save
file_name = dataset_name + "_spot_level_iteration_MAE.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_spot_level_iteration_MAE.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# MAE (cell type level)
mae_df_ct = pd.DataFrame(0,
                         index=gd_df.columns,
                         columns=res_index)
num = 0
for prediction in res_list:
    for ct in prediction.columns:
        tmp = mean_absolute_error(gd_df.loc[:, ct],
                                  prediction.loc[:, ct])
        mae_df_ct.loc[ct, res_index[num]] = tmp
    num += 1
mean = mae_df_ct.mean(axis=0)
sd = np.sqrt(mae_df_ct.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['MAE (cell type level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'MAE (cell type level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
plot_df.loc[:, 'group'] = '1'

base_plot = (ggplot(plot_df, aes(x='Iterations', y='MAE (cell type level)'))
             + geom_bar(stat="identity", fill='#9cdae7', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + geom_line(group='group', color='r')
             + theme_classic()
             + coord_cartesian(ylim=[0.015, 0.03]))
print(base_plot)
# save
file_name = dataset_name + "_cellType_level_iteration_MAE.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_cellType_level_iteration_MAE.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# %%%%%%
# ********************
# PlotPearson
# ********************   
from scipy.stats import pearsonr

per_df_spot = pd.DataFrame(0,
                           index=gd_df.index,
                           columns=res_index)
num = 0
for prediction in res_list:
    for spot in prediction.index:
        tmp = pearsonr(gd_df.loc[spot, :],
                       prediction.loc[spot, :])[0]
        per_df_spot.loc[spot, res_index[num]] = tmp
    num += 1
mean = per_df_spot.mean(axis=0)
sd = np.sqrt(per_df_spot.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['PCC (spot level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'PCC (spot level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
plot_df.loc[:, 'group'] = '1'
base_plot = (ggplot(plot_df, aes(x='Iterations', y='PCC (spot level)'))
             + geom_bar(stat="identity", fill='#92a3bb', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + coord_cartesian(ylim=[0.8, 0.95])
             + geom_line(group='group', color='r')
             + theme_classic())
print(base_plot)
# save
file_name = dataset_name + "_spot_level_iteration_Pearson.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_spot_level_iteration_Pearson.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# Pearson (cell type level)
per_df_ct = pd.DataFrame(0,
                         index=gd_df.columns,
                         columns=res_index)
num = 0
for prediction in res_list:
    for ct in prediction.columns:
        tmp = pearsonr(gd_df.loc[:, ct],
                       prediction.loc[:, ct])[0]
        per_df_ct.loc[ct, res_index[num]] = tmp
    num += 1
mean = per_df_ct.mean(axis=0)
sd = np.sqrt(per_df_ct.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['PCC (cell type level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'PCC (cell type level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
plot_df.loc[:, 'group'] = '1'
base_plot = (ggplot(plot_df, aes(x='Iterations', y='PCC (cell type level)'))
             + geom_bar(stat="identity", fill='#92a3bb', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + geom_line(group='group', color='red')
             + theme_classic()
             + coord_cartesian(ylim=[0.7, 0.9]))
print(base_plot)
# save
file_name = dataset_name + "_cellType_level_iteration_Pearson.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_cellType_level_iteration_Pearson.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# %%%%%%%%
# ********************
# Plot JS
# ********************   
import numpy as np
import scipy.stats


def JS_divergence(p, q):
    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


per_df_spot = pd.DataFrame(0,
                           index=gd_df.index,
                           columns=res_index)
num = 0
for prediction in res_list:
    for spot in prediction.index:
        tmp = JS_divergence(gd_df.loc[spot, :],
                            prediction.loc[spot, :])
        per_df_spot.loc[spot, res_index[num]] = tmp
    num += 1
mean = per_df_spot.mean(axis=0)
sd = np.sqrt(per_df_spot.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['JS (spot level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'JS (spot level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
plot_df.loc[:, 'group'] = '1'
base_plot = (ggplot(plot_df, aes(x='Iterations', y='JS (spot level)'))
             + geom_bar(stat="identity", fill='#c3e6df', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + geom_line(group='group', color='r')
             + theme_classic()
             + coord_cartesian(ylim=[0.03, 0.08]))
print(base_plot)
# save
file_name = dataset_name + "_spot_level_iteration_JS_divergence.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_spot_level_iteration_JS_divergence.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# JS (cell type level)
per_df_ct = pd.DataFrame(0,
                         index=gd_df.columns,
                         columns=res_index)
num = 0
for prediction in res_list:
    for ct in prediction.columns:
        tmp = JS_divergence(gd_df.loc[:, ct],
                            prediction.loc[:, ct])
        per_df_ct.loc[ct, res_index[num]] = tmp
    num += 1
mean = per_df_ct.mean(axis=0)
sd = np.sqrt(per_df_ct.var(axis=0))

plot_df = pd.DataFrame(mean, index=res_index, columns=['JS (cell type level)'])
plot_df.loc[:, 'Iterations'] = res_index
plot_df.loc[:, 'std'] = sd
plot_df = plot_df.loc[:, ['Iterations', 'JS (cell type level)', 'std']]
plot_df.loc[:, 'Iterations'] = pd.Categorical(plot_df.loc[:, 'Iterations'],
                                              categories=[str(i) for i in range(1,
                                                                                1 + len(res_list))],
                                              ordered=True)
base_plot = (ggplot(plot_df, aes(x='Iterations', y='JS (cell type level)'))
             + geom_bar(stat="identity", fill='#c3e6df', width=0.65)
             # + geom_errorbar(aes(ymin=mean - sd, ymax=mean + sd), color='grey',
             #                 size=0.5)
             + geom_line(group='group', color='r')
             + theme_classic()
             + coord_cartesian(ylim=[0.05, 0.13]))
print(base_plot)
# save
file_name = dataset_name + "_cellType_level_iteration_JS_divergence.png"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)
file_name = dataset_name + "_cellType_level_iteration_JS_divergence.pdf"
base_plot.save(filename=os.path.join(fig_dir, file_name), dpi=300)

# %%%%%%%%%%
# **************************************************
# Plot cell type distribution changes 
# **************************************************
from scipy.stats import pearsonr

sp_adata.obsm['spatial'] = sp_adata.obs.loc[:, ['x', 'y']].values
sc.set_figure_params(dpi=300)
res_list = []
res_index = []
for i in interation_list:
    index_i = i.split("STAID_iteration_")[1].split('Puck')[0]
    res_index.append(index_i)
    tmp = pd.read_csv(os.path.join(res_dir, i),
                      index_col=0)
    tmp = tmp.loc[gd_df.index, gd_df.columns]
    res_list.append(tmp)
gd_df = sp_adata.obsm['cell_type_proportion'].copy()

for ct in gd_df.columns:
    tmp_ct_df = gd_df.loc[:, [ct]]
    tmp_values = gd_df.loc[:, ct].values
    tmp_ct_df.columns = ['Ground truth']
    for step, iter in enumerate(res_list):
        if step in [0, 1, 2, 3, 4, 9, 19]:
            tmp = iter.loc[:, ct].values
            pcc = pearsonr(tmp_values, tmp)[0]
            tmp_ct_df.loc[:, f'Iteration {step + 1} (PCC={pcc.round(2)})'] = tmp
    plot_cts = tmp_ct_df.columns.tolist()
    sp_adata.obs.loc[:, plot_cts] = tmp_ct_df
    # plot
    sc.pl.spatial(adata=sp_adata,
                  color=plot_cts,
                  spot_size=75,
                  vmax=1,
                  save=f"_{ct}.pdf"
                  )
