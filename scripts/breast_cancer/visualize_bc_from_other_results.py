import os
import pandas as pd
import scanpy as sc
from staid import plot

sc.set_figure_params(dpi=300)

spa_data_path = r"/home/jxliu/Desktop/projects/staid/data/Breast_cancer/merged_datasets"
sample_list = ['CID4290',
               'CID4465',
               'CID44971',
               'CID4535']
res_dir = "/data/bqliu_data/jxliu_data/projects/staid/results/breast_cancer/others"
res_list = os.listdir(res_dir)
method_list = ['cell2location', 'SONAR', 'Tangram', 'DestVI',
               'DSTG', 'SPOTlight', 'RCTD',
               'Stereoscope', ]

for sample in sample_list[2:3]:
    sp_adata = sc.read_h5ad(os.path.join(spa_data_path,
                                         f"{sample}_visium_breast_cancer.h5ad"))
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    for method in method_list:
        res_file = f"{method}_{sample}_visium_breast_cancer_breast_cancer_results.csv"
        res = pd.read_csv(os.path.join(res_dir, res_file),
                          index_col=0)
        sp_adata.obs[res.columns] = res.values
        sc.pl.spatial(sp_adata, color=res.columns, spot_size=125, cmap='magma', img_key=None,
                      vmax=None)
