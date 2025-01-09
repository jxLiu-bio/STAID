import os
import sys

import cell2location
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  # enables correct plotting of text
from scipy.sparse import csr_matrix
from cell2location.utils.filtering import filter_genes

sc_file_path = sys.argv[1]
spatial_file_path = sys.argv[2]
celltype_key = sys.argv[3]
output_file_path = sys.argv[4]

# sc_file_path = "/home/frank/Documents/GAT-ID/simulation/scRNA-seq/scRNA-seq_development_heart_with_meta.h5ad"
# spatial_file_path = "/home/frank/Documents/GAT-ID/simulation/ST/ST_development_heart_FH5_1000L3_CN20_C1_1.h5ad"
# celltype_key = 'cell_type'
# output_file_path = "/home/frank/Documents/GAT-ID/"

file_name = 'cell2location_' + \
            spatial_file_path.split('/')[-1].split('.h')[0] + "_results.csv"
res_file_path = os.path.join(output_file_path, file_name)
print(res_file_path)
# sc_file_path = "/home/frank/Documents/GAT-ID/simulation/scRNA-seq/seqFISH+_OBcortex_single.h5ad"
# spatial_file_path = "/home/frank/Documents/GAT-ID/simulation/ST/seqFISH+_OBcortex_cellType_100_simulation.h5ad"
# celltype_key = 'cell_type'
# output_file_path = "/home/frank/Documents/GAT-ID/simulation/results"


adata_snrna_raw = sc.read_h5ad(sc_file_path)
adata_vis = sc.read_h5ad(spatial_file_path)
adata_snrna_raw.X = csr_matrix(adata_snrna_raw.X)
adata_vis.X = csr_matrix(adata_vis.X)
adata_snrna_raw.var_names_make_unique()
adata_vis.var_names_make_unique()

adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isin(np.array(
    adata_snrna_raw.obs[celltype_key].value_counts()[adata_snrna_raw.obs[celltype_key].value_counts() <= 1].index))]

# remove cells and genes with 0 counts everywhere
sc.pp.filter_genes(adata_snrna_raw, min_cells=1)
sc.pp.filter_cells(adata_snrna_raw, min_genes=1)

adata_snrna_raw.obs[celltype_key] = pd.Categorical(adata_snrna_raw.obs[celltype_key])
adata_snrna_raw = adata_snrna_raw[~adata_snrna_raw.obs[celltype_key].isna(), :]

selected = filter_genes(adata_snrna_raw, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)

# filter the object
adata_snrna_raw = adata_snrna_raw[:, selected].copy()
cell2location.models.RegressionModel.setup_anndata(adata=adata_snrna_raw,
                                                   labels_key=celltype_key)

# create and train the regression model
from cell2location.models import RegressionModel

mod = RegressionModel(adata_snrna_raw)

# Use all data for training (validation not implemented yet, train_size=1)
mod.train(max_epochs=250, batch_size=2500, train_size=1, lr=0.002,
          use_gpu=False)

# plot ELBO loss history during training, removing first 20 epochs from the plot
# mod.plot_history(20)

# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_snrna_raw = mod.export_posterior(
    adata_snrna_raw, sample_kwargs={'num_samples': 1000, 'batch_size': 2500,
                                    'use_gpu': False}
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
# scvi.data.view_anndata_setup(adata_vis)

# create and train the model
mod = cell2location.models.Cell2location(
    adata_vis, cell_state_df=inf_aver,
    # the expected average cell abundance: tissue-dependent
    # hyper-prior which can be estimated from paired histology:
    N_cells_per_location=30,
    # hyperparameter controlling normalisation of
    # within-experiment variation in RNA detection (using default here):
    detection_alpha=200
)

mod.train(max_epochs=30000,  # 30000
          # train using full data (batch_size=None)
          batch_size=None,
          # use all data points in training because
          # we need to estimate cell abundance at all locations
          train_size=1,
          use_gpu=False)

# plot ELBO loss history during training, removing first 100 epochs from the plot
# mod.plot_history(1000)
# plt.legend(labels=['full data training'])

adata_vis = mod.export_posterior(
    adata_vis, sample_kwargs={'num_samples': 1000,
                              'batch_size': mod.adata.n_obs, 'use_gpu': False}
)
print(adata_vis)
results = adata_vis.obsm['q05_cell_abundance_w_sf'].copy()
results = results.div(results.sum(axis=1), axis=0)
results.columns = [i.replace('q05cell_abundance_w_sf_', '') for i in \
                   results.columns]
results.to_csv(res_file_path)
print(results.iloc[:5, :5])
