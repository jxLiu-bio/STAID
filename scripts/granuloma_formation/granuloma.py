from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from torch_geometric import seed_everything

from staid import run_deconvolution

data_dir = Path("/home/jxliu/Desktop/projects/staid/data/granuloma")
# match scRNA-seq with Visium
meta_sc = pd.read_csv(data_dir / "metadata-scRNA-seq.csv")
meta_sp = pd.read_csv(data_dir / "metadata-Visium.csv")
meta_sp.index = meta_sp.hca_sample_id
meta_sc.index = meta_sc.sample_id
# match_dir
match_df = pd.DataFrame(columns=['Visium', 'scRNA-seq'])
for sp_sample in meta_sp.index:
    region_id = meta_sp.loc[sp_sample, 'patient_region_id']
    sc_sample = meta_sc.index[meta_sc.patient_region_id == region_id]
    if len(sp_sample) > 0:
        match_df.loc[sp_sample, :] = [sp_sample, sc_sample.to_list()]
match_df = pd.concat((match_df, meta_sp), axis=1)
prediction_list = []
# match_df.to_csv(data_dir / "match-Visium-scRNA-seq.csv")
for sp_sample in match_df.index:
    sc_sample = match_df.loc[sp_sample, "scRNA-seq"][0]

    # load data
    sp_adata = sc.read_h5ad(data_dir / f"processed/SRT/h5ad/10X_Visium_{sp_sample}.h5ad")
    sc_adata = sc.read_h5ad(data_dir / f"processed/scRNA-seq/{sc_sample}/{sc_sample}.h5ad")
    anno_key = 'cell_type_original'
    cell_list = []
    for ct in np.unique(sc_adata.obs[anno_key]):
        ct_tmp_list = sc_adata.obs.loc[sc_adata.obs[anno_key] == ct,
                      :].index.tolist()
        if len(ct_tmp_list) >= 10:
            cell_list.extend(ct_tmp_list)
    sc_adata = sc_adata[cell_list, :]
    sc_adata.var_names_make_unique()
    sp_adata.var_names_make_unique()
    seed_everything(2023)

    # run deconvolution
    prediction = run_deconvolution(spa_adata=sp_adata,
                                   sc_adata=sc_adata,
                                   anno_key=anno_key,
                                   lr=0.0005,
                                   num_pseudo=5000,
                                   num_iter=1,
                                   min_cells=3,
                                   max_cells=10,
                                   remove_platform='auto',
                                   error_cutoff=0.01,
                                   device='cuda:1',
                                   batch_size=128,
                                   hidden_dims=[512, 256, 256],
                                   library_size=1e1,
                                   random_spot_rate=0.3,
                                   dropout=0.05)
    prediction_list.append(prediction)
    sp_adata.obs[prediction.columns] = prediction.values
    sc.pl.spatial(sp_adata, color=prediction.columns, img_key=None, spot_size=130, cmap='magma', vmax=1)
