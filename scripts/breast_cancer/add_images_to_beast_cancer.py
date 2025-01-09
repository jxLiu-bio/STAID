import os
from pathlib import Path

import pandas as pd
import scanpy as sc

from staid import plot

data_dir = r"Z:/jxliu/datasets/Spatial/Visium/Breast_cancer"
res_dir = r"Z:/jxliu/project/ID-GAT/ID/data/Breast_cancer/merged_datasets"
data_list = ["CID4465", "CID4535", "CID44971",
             "1142243F", "1160920F", "CID4290"]
for sample in data_list:
    adata = sc.read_h5ad(os.path.join(data_dir,
                                      sample + "_visium_breast_cancer.h5ad"))
    # meta
    meta = pd.read_csv(os.path.join(data_dir,
                                    sample + "/spatial/tissue_positions_list.csv"),
                       header=None, index_col=0)
    meta.columns = ['in_tissue', 'array_row', 'array_col',
                    'pixel_x', 'pixel_y']
    meta = meta.loc[adata.obs_names, :]
    adata.obs.loc[:, meta.columns] = meta
    adata.obsm['spatial'] = meta.loc[:, ['pixel_x', 'pixel_y']].values
    adata = plot.add_image(adata,
                           spatial_dir=Path(data_dir) / f'{sample}/spatial',
                           library_id=sample)
    sc.pl.spatial(adata, color=adata.var_names[1000])
    adata.write_h5ad(os.path.join(res_dir,
                                  sample + "_visium_breast_cancer.h5ad"))
