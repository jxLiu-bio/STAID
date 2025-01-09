import os

import numpy as np
import scanpy as sc
from torch_geometric import seed_everything

from staid import run_deconvolution

# Data dir
# load spatial datasets
sc_data_path = "/home/jxliu/Desktop/projects/staid/data/lymphoid_developmenta/processed/scRNA-seq"
sp_data_path = "/home/jxliu/Desktop/projects/staid/data/lymphoid_developmenta/processed/SRT"
sample_sc_list = ['CK366', 'CK356']
sample_srt_list = ['10X_Visium_ACH005', '10X_Visium_ACH0022']

sample_sc = sample_sc_list[0]
sample_srt = sample_srt_list[0]
seed_everything(2023)

sp_adata = sc.read_h5ad(os.path.join(sp_data_path,
                                     f"h5ad/{sample_srt}.h5ad"))
sc_adata = sc.read_h5ad(os.path.join(sc_data_path,
                                     f"{sample_sc}/{sample_sc}.h5ad"))
sc_adata.var_names_make_unique()
sp_adata.var_names_make_unique()
celltype_key = 'cell_type'
device = np.random.choice(['cuda:1', 'cuda:2', 'cuda:3', 'cuda:0'])
prediction = run_deconvolution(spa_adata=sp_adata,
                               sc_adata=sc_adata,
                               anno_key=celltype_key,
                               lr=0.0005,
                               num_pseudo=5000,
                               num_iter=5,
                               min_cells=1,
                               max_cells=10,
                               remove_platform='auto',
                               error_cutoff=0.01,
                               device=device,
                               batch_size=128,
                               hidden_dims=[512, 256, 128],
                               library_size=1e1,
                               random_spot_rate=0.3,
                               dropout=0.15)

# plot
sp_adata.obs.loc[:, prediction.columns] = prediction
sc.pl.spatial(sp_adata, color=prediction.columns, img_key=None, spot_size=150)
