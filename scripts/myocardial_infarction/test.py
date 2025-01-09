from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from staid.staid_pred_train import gat_predict

# sc.set_figure_params(dpi=200)
data_dir = Path("/home/jxliu/Desktop/projects/staid/data/human_myocardial_infraction")
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
# match_df.to_csv(data_dir / "match-Visium-scRNA-seq.csv")c
for sp_sample in match_df.index:
    print(sp_sample)
    sc_sample = match_df.loc[sp_sample, "scRNA-seq"][0]
    device = np.random.choice(['cuda:1', 'cuda:2', 'cuda:3'])
    # load data
    sp_adata = sc.read_h5ad(data_dir / f"processed/SRT/10X_Visium_{sp_sample}.h5ad")
    sc_adata = sc.read_h5ad(data_dir / f"processed/scRNA-seq/{sc_sample}.h5ad")
    anno_key = 'cell_type'
    cell_list = []
    for ct in np.unique(sc_adata.obs[anno_key]):
        ct_tmp_list = sc_adata.obs.loc[sc_adata.obs[anno_key] == ct,
                      :].index.tolist()
        if len(ct_tmp_list) >= 10:
            cell_list.extend(ct_tmp_list)
    sc_adata = sc_adata[cell_list, :]
    sc_adata.var_names_make_unique()
    sp_adata.var_names_make_unique()

    prediction = gat_predict(spa_adata=sp_adata,
                             sc_adata=sc_adata,
                             anno_key=anno_key,
                             lr=0.0005,
                             num_pseudo=5000,
                             num_iter=1,
                             min_cells=1,
                             max_cells=10,
                             remove_platform=False,
                             device=device,
                             batch_size=256,
                             library_size=1e1)

    # plot
    # add spatial information
    # sp_adata.obsm['spatial'] = sp_adata.obs.loc[:, ['x', 'y']].values
    sp_adata.obs[prediction.columns] = prediction.values
    sc.pl.spatial(sp_adata, color=prediction.columns, spot_size=120, img_key=None, cmap='magma')
