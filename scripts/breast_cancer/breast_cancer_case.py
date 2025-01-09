import os
import numpy as np
import scanpy as sc
from torch_geometric import seed_everything
from staid.staid_pred_train import gat_predict

# load spatial datasets
spa_data_path = "/home/jxliu/Desktop/projects/staid/data/Breast_cancer/merged_datasets"
sc_data_path = "/home/jxliu/Desktop/projects/staid/data/scRNA-seq/Breast_cancer"
sample_list = ['CID4290',
               'CID4465',
               'CID44971',
               'CID4535']

for sample in sample_list[3:]:
    seed_everything(2023)
    device = np.random.choice(['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    sp_adata = sc.read_h5ad(os.path.join(spa_data_path,
                                         f"{sample}_visium_breast_cancer.h5ad"))
    sc_adata = sc.read_h5ad(os.path.join(sc_data_path,
                                         f"{sample}_scRNA_seq_with_annotations.h5ad"))

    sp_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()
    sc.pp.filter_genes(sp_adata, min_cells=10)
    sc.pp.filter_genes(sc_adata, min_cells=10)

    # intersection of genes and remove useless cell types
    anno_key = 'celltype_major'
    cell_list = []
    for ct in np.unique(sc_adata.obs[anno_key]):
        ct_tmp_list = sc_adata.obs.loc[sc_adata.obs[anno_key] == ct,
                      :].index.tolist()
        if len(ct_tmp_list) >= 10:
            cell_list.extend(ct_tmp_list)
    sc_adata = sc_adata[cell_list, :]

    prediction = gat_predict(spa_adata=sp_adata,
                             sc_adata=sc_adata,
                             anno_key=anno_key,
                             lr=0.0005,
                             num_pseudo=5000,
                             num_iter=2,
                             min_cells=1,
                             max_cells=10,
                             remove_platform=False,
                             device=device,
                             batch_size=256,
                             library_size=1e2)

    # plot
    sp_adata.obs[prediction.columns] = prediction.values
    sc.pl.spatial(sp_adata, color=prediction.columns, spot_size=120, img_key=None, cmap='magma')
