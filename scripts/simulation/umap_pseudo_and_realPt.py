import pandas as pd
import os

import pandas as pd
import scanpy as sc
from plotnine import *

from staid.staid_pred_train import gat_predict_tmp

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=200, facecolor='white')

data_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation"
dataset_list = os.listdir(data_dir + "/ST")
output_file_path = r"Z:\jxliu\project\ID-GAT\ID\simulation\results\SpaGFT"
spatial_key = ['x', 'y']
celltype_key = 'cell_type'
dataset = dataset_list[0]

# find reference
dataset_name = dataset.split('.h5a')[0]
file_name = 'GAT-ID_' + \
            dataset_name + "_results.csv"
res_file_path = os.path.join(output_file_path, file_name)
if "OB" in dataset_name:
    sc_file_path = os.path.join(data_dir + r"\scRNA-seq",
                                "seqFISH+_OBcortex_single.h5ad")
else:
    sc_file_path = os.path.join(data_dir + r"\scRNA-seq",
                                "seqFISH+_SScortex_single.h5ad")
spatial_file_path = os.path.join(data_dir + r"\ST", dataset)

# Load data
sc_adata = sc.read(sc_file_path)
sp_adata = sc.read(spatial_file_path)
# correct two datasets
sp_adata.var_names_make_unique()
sc_adata.var_names_make_unique()

# run deconvolution
pseudo_df, pseudo_comp = gat_predict_tmp(spa_adata=sp_adata,
                                         sc_adata=sc_adata,
                                         num_iter=5,
                                         spatial_key=spatial_key,
                                         anno_key=celltype_key,
                                         num_pseudo=5000,
                                         remove_platform=False)

# sort the pseudo_df
abs_list = []
ral_list = []
ran_list = []
for spot_name in pseudo_df.columns:
    if 'rela' in spot_name:
        ral_list.append(spot_name)
    elif 'abs' in spot_name:
        abs_list.append(spot_name)
    elif 'rand' in spot_name:
        ran_list.append(spot_name)
    else:
        print(spot_name)
abs_df = pseudo_df.loc[:, abs_list].transpose()
ral_df = pseudo_df.loc[:, ral_list].transpose()
ran_df = pseudo_df.loc[:, ran_list].transpose()

gatid_df = pd.concat((abs_df, ral_df), axis=0)
gatid_df = abs_df
sp_df = pd.DataFrame(sp_adata.X, index=sp_adata.obs_names,
                     columns=sp_adata.var_names)
sp_df = sp_df.loc[:, gatid_df.columns]
meta = ['Real'] * sp_df.shape[0] + ['GAT-ID'] * gatid_df.shape[0] \
       + ['Random'] * ran_df.shape[0]
adata = sc.AnnData(pd.concat((sp_df, gatid_df, ran_df), axis=0))
adata.obs['Source'] = meta
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=30)
# UMAP
sc.tl.umap(adata)
sc.pl.umap(adata, color='Source', size=30)
sc.pl.umap(adata[sp_df.index, :], color='Source', size=30)
sc.pl.umap(adata[gatid_df.index, :], color='Source', size=30)
sc.pl.umap(adata[ran_df.index, :], color='Source', size=30)

# UMAP coordinates
res_dir = r"Z:\jxliu\project\ID-GAT\ID\figures\fig1"
umap_df = pd.DataFrame(adata.obsm['X_umap'],
                       index=adata.obs_names,
                       columns=['UMAP_1', 'UMAP_2'])
umap_df.to_csv(os.path.join(res_dir,
                            dataset_name + "_real_GAT-ID_random_umap.csv"))
plot_df = umap_df.copy()
plot_df['Source'] = meta
plot_df['Source'] = pd.Categorical(plot_df['Source'], categories=['GAT-ID',
                                                                  'Real',
                                                                  'Random'],
                                   ordered=True)
# + scale_color_hue(s=0.9, l=0.65, h=0.0417, color_space='husl')
base_plot_mix = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2',
                                     color='Source'))
                 + geom_point(size=0.5)
                 + scale_color_manual(values=("#F8766D", "#5e59da", "#00C19A"))
                 + coord_equal()
                 + theme_classic()
                 )
ind_p = (base_plot_mix + facet_grid('.~Source', margins=False)
         + theme(strip_text_x=element_text(size=9),
                 strip_background_x=element_blank(),
                 legend_position="none",
                 axis_title=element_text(size=9),
                 axis_text=element_text(size=6))
         + coord_equal(1.3))
print(ind_p)
ind_p_name = "three_sources_umap_spots.png"
ind_p.save(os.path.join(res_dir, ind_p_name), dpi=600)
ind_p_name = "three_sources_umap_spots.pdf"
ind_p.save(os.path.join(res_dir, ind_p_name), dpi=600)

# # + scale_color_hue(s=0.9, l=0.65, h=0.0417, color_space='husl')
base_plot_mix = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2',
                                     color='Source'))
                 + geom_point(size=4)
                 + scale_color_manual(values=("#F8766D", "#5e59da", "#00C19A"))
                 + coord_equal(1.3)
                 + theme_classic()
                 )
print(base_plot_mix)
ind_p = (base_plot_mix + facet_grid('Source~.', margins=False)
         + theme(strip_text_y=element_text(size=20),
                 strip_background_y=element_blank(),
                 legend_position="none",
                 axis_title=element_text(size=18),
                 axis_text=element_text(size=15))
         + coord_equal(1.3))
print(ind_p)
ind_p_name = "three_sources_umap_spots_col.png"
ind_p.save(os.path.join(res_dir, ind_p_name), dpi=600)
ind_p_name = "three_sources_umap_spots_col.pdf"
ind_p.save(os.path.join(res_dir, ind_p_name), dpi=600)

# mixture
base_plot_mix = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2',
                                     color='Source'))
                 + geom_point(size=2)
                 + scale_color_manual(values=("#F8766D", "#5e59da", "#00C19A"))
                 + coord_equal()
                 + theme_classic()
                 + theme(axis_line=element_blank(),
                         axis_text=element_blank(),
                         axis_ticks=element_blank(),
                         legend_position=(0.18, 0.15),
                         legend_background=element_blank(),
                         axis_title=element_text(size=12),
                         legend_text=element_text(size=11),
                         legend_key_size=element_rect(size=80),
                         legend_title=element_text(size=12)
                         )
                 + coord_equal(1.3)
                 )
print(base_plot_mix)
mix_p_name = "mix_sources_umap_spots.png"
base_plot_mix.save(os.path.join(res_dir, mix_p_name), dpi=600)
mix_p_name = "mix_sources_umap_spots.pdf"
base_plot_mix.save(os.path.join(res_dir, mix_p_name), dpi=600)
