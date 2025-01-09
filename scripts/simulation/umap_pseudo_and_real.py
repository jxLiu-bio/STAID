import os
from pathlib import Path

import pandas as pd
import scanpy as sc
from plotnine import *

data_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/results/simulation_pseudo_spots")
fig_dir = Path("/home/jxliu/Desktop/projects/staid/figures/fig2/pseudo_spots_distribution")
staid_df = pd.read_csv(data_dir / "STAID_Puck-191204-01_100_simulated.csv",
                       index_col=0)
sp_df = pd.read_csv(data_dir / "Original_Puck-191204-01_100_simulated.csv",
                    index_col=0)
ran_df = pd.read_csv(data_dir / "Random_Puck-191204-01_100_simulated.csv",
                     index_col=0)

meta = ['Real'] * sp_df.shape[0] + ['STAID'] * staid_df.shape[0] \
       + ['Random'] * ran_df.shape[0]
adata = sc.AnnData(pd.concat((sp_df, staid_df, ran_df), axis=0))
adata.obs['Source'] = meta
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=30)
# UMAP
sc.tl.umap(adata)
sc.pl.umap(adata, color='Source', size=10)
sc.pl.umap(adata[sp_df.index, :], color='Source', size=30)
sc.pl.umap(adata[staid_df.index, :], color='Source', size=30)
sc.pl.umap(adata[ran_df.index, :], color='Source', size=30)

# UMAP coordinates
dataset_name = "Puck-191204-01_100_simulated"
umap_df = pd.DataFrame(adata.obsm['X_umap'],
                       index=adata.obs_names,
                       columns=['UMAP_1', 'UMAP_2'])
umap_df.to_csv(os.path.join(fig_dir,
                            dataset_name + "_real_STAID_random_umap.csv"))
plot_df = umap_df.copy()
plot_df['Source'] = meta
plot_df['Source'] = pd.Categorical(plot_df['Source'], categories=['STAID',
                                                                  'Real',
                                                                  'Random'],
                                   ordered=True)

base_plot_mix = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2',
                                     color='Source'))
                 + geom_point(size=0.0001)
                 + scale_color_manual(values=("#F8766D", "#5e59da", "#00C19A"))
                 + coord_equal(1)
                 + theme_classic()
                 )
print(base_plot_mix)
ind_p = (base_plot_mix + facet_grid('.~Source', margins=False)
         + theme(strip_text_x=element_text(size=12),
                 strip_background_x=element_blank(),
                 legend_position="none",
                 axis_title=element_text(size=12),
                 axis_text=element_text(size=12),
                 figure_size=(18, 6)))
print(ind_p)
ind_p_name = "three_sources_umap_spots_row.png"
ind_p.save(os.path.join(fig_dir, ind_p_name), dpi=300, width=6, height=2)
ind_p_name = "three_sources_umap_spots_row.pdf"
ind_p.save(os.path.join(fig_dir, ind_p_name), dpi=300, width=6, height=2)

# # + scale_color_hue(s=0.9, l=0.65, h=0.0417, color_space='husl')
base_plot_mix = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2',
                                     color='Source'))
                 + geom_point(size=0.2)
                 + scale_color_manual(values=("#F8766D", "#5e59da", "#00C19A"))
                 + coord_equal(1)
                 + theme_classic()
                 )
print(base_plot_mix)
ind_p = (base_plot_mix + facet_grid('Source~.', margins=False)
         + theme(strip_text_y=element_text(size=20),
                 strip_background_y=element_blank(),
                 legend_position="none",
                 axis_title=element_text(size=18),
                 axis_text=element_text(size=15))
         + coord_equal(1))
print(ind_p)
ind_p_name = "three_sources_umap_spots_col.png"
ind_p.save(os.path.join(fig_dir, ind_p_name), dpi=300)
ind_p_name = "three_sources_umap_spots_col.pdf"
ind_p.save(os.path.join(fig_dir, ind_p_name), dpi=300)

# mixture
base_plot_mix = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2',
                                     color='Source'))
                 + geom_point(size=0.1)
                 + scale_color_manual(values=("#F8766D", "#5e59da", "#00C19A"))
                 + coord_equal()
                 + theme_classic()
                 + theme(axis_line=element_blank(),
                         axis_text=element_blank(),
                         axis_ticks=element_blank(),
                         legend_position=(0.65, 0.15),
                         legend_background=element_blank(),
                         axis_title=element_text(size=12),
                         legend_text=element_text(size=11),
                         legend_key_size=element_rect(size=80),
                         legend_title=element_text(size=12)
                         )
                 + coord_equal(1)
                 )
print(base_plot_mix)
mix_p_name = "mix_sources_umap_spots.png"
base_plot_mix.save(os.path.join(fig_dir, mix_p_name), dpi=300)
mix_p_name = "mix_sources_umap_spots.pdf"
base_plot_mix.save(os.path.join(fig_dir, mix_p_name), dpi=300)

# mixture
base_plot_mix = (ggplot(plot_df, aes(x='UMAP_1', y='UMAP_2',
                                     color='Source'))
                 + geom_point(size=0.1)
                 + scale_color_manual(values=("#F8766D", "#5e59da", "#00C19A"))
                 + coord_equal()
                 + theme_classic()
                 + theme(axis_line=element_blank(),
                         axis_text=element_blank(),
                         axis_ticks=element_blank(),
                         legend_position=(0.65, 0.15),
                         legend_background=element_blank(),
                         legend_text=element_text(size=11),
                         legend_key_size=element_rect(size=80),
                         legend_title=element_text(size=12)
                         )
                 + coord_equal(1)
                 )
print(base_plot_mix)
mix_p_name = "mix_sources_umap_spots_noAxis.png"
base_plot_mix.save(os.path.join(fig_dir, mix_p_name), dpi=300)
mix_p_name = "mix_sources_umap_spots_noAxis.pdf"
base_plot_mix.save(os.path.join(fig_dir, mix_p_name), dpi=300)
