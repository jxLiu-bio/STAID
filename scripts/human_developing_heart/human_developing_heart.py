import os

import pandas as pd
import scanpy as sc

from staid import plot

# Define dataset folder
res_dir = r"Z:\jxliu\project\ID-GAT\ID\results\Human_developing_heart"
all_res_list = os.listdir(res_dir)
fig_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig3\all_cts"
fig_pie_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig3\pie_plot"
for res in all_res_list:
    if 'att' in res:
        all_res_list.remove(res)

for res_file in all_res_list[11:13]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=110,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=110,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
# *********
# New datasets
# **********
for res_file in all_res_list[5:11]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=120,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=120,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
# *********
# New datasets
# **********    
for res_file in all_res_list[:4]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=400,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=400,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
# *********
# New datasets
# **********
for res_file in all_res_list[12:]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=85,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=85,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           vmax=1)
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
        # file_path = os.path.join(fig_pie_dir, 
        #                          sample + f"_with_lengend.{fig_format}")
        # fig = plot.scatter_pie(sp_adata,
        #                    pt_size=10,
        #                    colors=colors,
        #                    figsize=(20, 6),
        #                    show_legend=True, 
        #                    return_fig=True)
        # fig.get_figure().savefig(file_path)

# Define dataset folder
res_dir = r"Z:\jxliu\project\ID-GAT\ID\results\Human_developing_heart"
all_res_list = os.listdir(res_dir)
fig_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig3\all_cts_re"
fig_pie_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig3\pie_plot"
for res in all_res_list:
    if 'att' in res:
        all_res_list.remove(res)

for res_file in all_res_list[11:13]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=110,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=110,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
# *********
# New datasets
# **********
for res_file in all_res_list[5:11]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=120,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=120,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
# *********
# New datasets
# **********    
for res_file in all_res_list[:4]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=400,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=400,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
# *********
# New datasets
# **********
for res_file in all_res_list[12:]:
    res_df = pd.read_csv(os.path.join(res_dir, res_file), index_col=0)
    week = res_file.split('_')[4]
    sample = '_'.join(res_file.split('_')[5:9])
    # correspondings load spatial datasets
    dataset_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\ST"
    dataset_filename = '_'.join(res_file.split('_')[1:9]) + ".h5ad"
    sp_adata = sc.read_h5ad(os.path.join(dataset_dir, dataset_filename))
    sp_adata.var_names_make_unique()
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    sp_adata.obs['x'] = [eval(i.split('x')[1]) for i in sp_adata.obs.index]
    sp_adata.obs['y'] = [eval(i.split('x')[2]) for i in sp_adata.obs.index]
    # obtain all cell types
    all_cell_types = res_df.columns.tolist()
    # load results to adata object
    sp_adata.obsm['deconvolution'] = res_df
    # plot
    save_name = res_file.split('.csv')[0] + ".png"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=85,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    save_name = res_file.split('.csv')[0] + ".pdf"
    save_path = os.path.join(fig_dir, save_name)
    plot.scatter_cell_type(sp_adata,
                           cell_type=all_cell_types,
                           size=85,
                           spatial_info=['x', 'y'],
                           n_col=3,
                           save_path=save_path,
                           )
    colors = ["#05B9E2", "#EECA3B", "#F6CAE5", "#82B0D2", "#778AAE",
              "#9E9E9E", "#54B345", "#68855C", "#c82423", "#BEBADA",
              "#D9AF6B", "#C97937", "#8494FF", "#B883D4", "#FA7F6F"]
    for fig_format in ['png', 'pdf']:
        file_path = os.path.join(fig_pie_dir,
                                 sample + f"_no_lengend.{fig_format}")
        fig = plot.scatter_pie(sp_adata,
                               pt_size=10,
                               colors=colors,
                               figsize=(5, 6),
                               show_legend=False,
                               return_fig=True)
        fig.get_figure().savefig(file_path)
