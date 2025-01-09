from pathlib import Path
import numpy as np
import scanpy as sc
from staid import run_deconvolution
from torch_geometric import seed_everything

sc.set_figure_params(dpi=300)
data_dir = Path("/home/jxliu/Desktop/projects/staid/data/granuloma")
sc_dataset_list = ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08']
sp_dataset_list = ['P01_V02', 'P02_V02', 'P03_V02', 'P04V02_1', 'P05_V02', 'P06V06_1', 'P07V06_1', 'P08_V02']
for ind in range(len(sc_dataset_list)):
    sc_sample = sc_dataset_list[ind] + ".h5ad"
    sc_data_path = data_dir / f"scRNA-seq/{sc_sample}"
    sp_sample = sp_dataset_list[ind] + ".h5ad"
    sp_data_path = data_dir / f"SRT/{sp_sample}"
    sc_adata = sc.read_h5ad(sc_data_path)
    sp_adata = sc.read_h5ad(sp_data_path)
    sp_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()
    sc.pp.filter_genes(sc_adata, min_cells=20)
    sc.pp.filter_genes(sp_adata, min_cells=20)
    sc_adata = sc.AnnData(sc_adata.X.todense(),
                          obs=sc_adata.obs,
                          var=sc_adata.var)
    sp_adata = sc.AnnData(sp_adata.X.todense(),
                          obs=sp_adata.obs,
                          var=sp_adata.var,
                          obsm=sp_adata.obsm)

    # sc_adata = sc_adata[sc_adata.obs['tissue'] == 'Lesional skin', :]
    sp_adata.obsm['spatial'] = sp_adata.obs.loc[:, ['pxl_col_in_hires', 'pxl_row_in_hires']].values
    # sc.pl.spatial(sp_adata, color=sp_adata.var_names[100], img_key=None, spot_size=10, use_raw=False)
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
    device = np.random.choice(['cuda:1', 'cuda:2', 'cuda:3', 'cuda:0'])

    # run deconvolution
    prediction = run_deconvolution(spa_adata=sp_adata,
                                   sc_adata=sc_adata,
                                   anno_key=anno_key,
                                   lr=0.0005,
                                   num_pseudo=5000,
                                   num_iter=2,
                                   min_cells=1,
                                   max_cells=15,
                                   remove_platform=False,
                                   error_cutoff=0.01,
                                   device=device,
                                   batch_size=128,
                                   hidden_dims=[512, 128, 64],
                                   library_size=3e4,
                                   random_spot_rate=0.3,
                                   dropout=0.05)
    sp_adata.obs[prediction.columns] = prediction.values
    sc.pl.spatial(sp_adata, color=prediction.columns, img_key=None, spot_size=25, cmap='magma', vmax=1)
