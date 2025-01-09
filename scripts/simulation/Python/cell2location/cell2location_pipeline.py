import os
import sys

import cell2location
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  # enables correct plotting of text
from cell2location.utils.filtering import filter_genes

plt.ion()
spatial_file_path = sys.argv[1]
sc_file_path = sys.argv[2]
celltype_key = sys.argv[3]
output_file_path = sys.argv[4]
sample_name = sys.argv[5]
sample_name = sample_name.split('.h5a')[0]
sce = sys.argv[6]

file_name = 'cell2location_' + \
            sample_name + "_" + sce + "_results.csv"
res_file_path = os.path.join(output_file_path, file_name)
print(res_file_path)
print(sc_file_path)
print(celltype_key)
adata_snrna_raw = sc.read_h5ad(sc_file_path)
adata_vis = sc.read_h5ad(spatial_file_path)
adata_snrna_raw.var_names_make_unique()
adata_vis.var_names_make_unique()

sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
sc.pp.filter_cells(adata_snrna_raw, min_genes=1)

adata_snrna_raw.obs[celltype_key] = pd.Categorical(adata_snrna_raw.obs[celltype_key])
adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isna(), :]
selected = filter_genes(adata_snrna_raw)
# filter the object
adata_snrna_raw = adata_snrna_raw[:, selected].copy()
cell2location.models.RegressionModel.setup_anndata(adata=adata_snrna_raw,
                                                   labels_key=celltype_key)

# create and train the regression model
from cell2location.models import RegressionModel

mod = RegressionModel(adata_snrna_raw)

# Use all data for training (validation not implemented yet, train_size=1)

mod.train(use_gpu=True)
plt.close('all')

# plot ELBO loss history during training, removing first 20 epochs from the plot
# mod.plot_history(20)

# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_snrna_raw = mod.export_posterior(
    adata_snrna_raw, sample_kwargs={'use_gpu': True}
)

# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in adata_snrna_raw.varm.keys():
    inf_aver = adata_snrna_raw.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                                                for i in
                                                                adata_snrna_raw.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_snrna_raw.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_snrna_raw.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_snrna_raw.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]

intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
adata_vis = adata_vis[:, intersect].copy()
inf_aver = inf_aver.loc[intersect, :].copy()

# prepare anndata for cell2location model
cell2location.models.Cell2location.setup_anndata(adata=adata_vis)

# create and train the model
mod = cell2location.models.Cell2location(
    adata_vis,
    cell_state_df=inf_aver,
    N_cells_per_location=30
)
mod.train(use_gpu=True)
# plot ELBO loss history during training, removing first 100 epochs from the plot
# mod.plot_history(1000)
# plt.legend(labels=['full data training'])
adata_vis = mod.export_posterior(
    adata_vis, sample_kwargs={'use_gpu': True}
)
print(adata_vis)
results = adata_vis.obsm['q05_cell_abundance_w_sf'].copy()
results = results.div(results.sum(axis=1), axis=0)
results.columns = [i.replace('q05cell_abundance_w_sf_', '') for i in \
                   results.columns]
results.to_csv(res_file_path)
print(results.iloc[:5, :5])
plt.ioff()
