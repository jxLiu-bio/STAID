import os
import pandas as pd
import scanpy as sc
from staid import plot
from pathlib import Path

sc.set_figure_params(dpi=300)

# sc.set_figure_params(dpi=200)
spa_data_path = Path("/home/jxliu/Desktop/projects/staid/data/human_myocardial_infraction")
sample_list = ['10X0027']
res_dir = "/data/bqliu_data/jxliu_data/projects/staid/results/myocardial_infaration/others"
res_list = os.listdir(res_dir)
method_list = ['cell2location', 'SONAR', 'Tangram', 'DestVI',
               'DSTG', 'SPOTlight', 'RCTD', 'Stereoscope', ]

for sample in sample_list[-1:]:
    sp_adata = sc.read_h5ad(os.path.join(spa_data_path,
                                         f"processed/SRT/10X_Visium_{sample}.h5ad"))
    sc.pp.normalize_total(sp_adata)
    sc.pp.log1p(sp_adata)
    for method in method_list:
        try:
            res_file = f"{method}_10X_Visium_{sample}_myocardial-infarction_results.csv"
            res = pd.read_csv(os.path.join(res_dir, res_file),
                              index_col=0)
            sp_adata.obs[res.columns] = res.values
            sc.pl.spatial(sp_adata, color=res.columns, spot_size=125, cmap='magma', img_key=None, vmax=1)
        except Exception as error:
            print(method)
