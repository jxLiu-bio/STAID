import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

"""
please the vefity that you have installed the Seurat,SpaOTsc,Tangram,
novoSpaRc please  make sure you are in SpatialBenmarking dir and have prepared
the data files
"""


def simulated_pseudo_spots(spatial_rna,
                           spatial_meta,
                           spatial_loc,
                           CoordinateXlable,
                           CoordinateYlable,
                           window,
                           outdir,
                           source):
    if os.path.exists(outdir):
        print(outdir)
    else:
        os.mkdir(outdir)
    combined_spot = []
    combined_spot_loc = []
    window = window
    c = 0
    for x in np.arange((spatial_loc[CoordinateXlable].min() // window),
                       spatial_loc[CoordinateXlable].max() // window + 1):
        for y in np.arange((spatial_loc[CoordinateYlable].min() // window),
                           spatial_loc[CoordinateYlable].max() // window + 1):
            tmp_loc = spatial_loc[(x * window < spatial_loc[CoordinateXlable]) &
                                  (spatial_loc[CoordinateXlable] <
                                   (x + 1) * window) &
                                  (y * window < spatial_loc[CoordinateYlable]) &
                                  (spatial_loc[CoordinateYlable] <
                                   (y + 1) * window)]
            if len(tmp_loc) > 0:
                c += 1
                combined_spot_loc.append([x, y])
                combined_spot.append(tmp_loc.index.to_list())

    combined_cell_counts = pd.DataFrame([len(s) for s in combined_spot],
                                        columns=['cell_count'])
    combined_cell_counts.to_csv(
        outdir / f'{source}_combined_cell_counts_{window}.txt',
        sep='\t')
    combined_cell_counts = pd.read_csv(
        outdir / f'{source}_combined_cell_counts_{window}.txt',
        sep='\t', index_col=0)
    print('The simulated spot has cells with ' +
          str(combined_cell_counts.min()[0]) +
          ' to ' + str(combined_cell_counts.max()[0]))
    combined_spot_loc = pd.DataFrame(combined_spot_loc, columns=['x', 'y'])
    combined_spot_loc.to_csv(outdir /
                             f'{source}_combined_Locations_{window}.txt',
                             sep='\t', index=False)

    combined_spot_exp = []
    for s in combined_spot:
        combined_spot_exp.append(spatial_rna.loc[s,
                                 :].sum(axis=0).values)
    combined_spot_exp = pd.DataFrame(combined_spot_exp,
                                     columns=spatial_rna.columns)
    combined_spot_exp.index = combined_cell_counts.index
    combined_spot_exp.to_csv(outdir /
                             f'{source}_combined_spatial_count_{window}.txt',
                             sep='\t',
                             index=False)

    combined_spot_clusters = \
        pd.DataFrame(np.zeros((len(combined_spot_loc.index),
                               len(np.unique(spatial_meta['celltype'])))),
                     columns=np.unique(spatial_meta['celltype']))
    for i, c in enumerate(combined_spot):
        for clt in spatial_meta.loc[c, 'celltype']:
            combined_spot_clusters.loc[i, clt] += 1
    combined_spot_clusters.to_csv(outdir /
                                  f'{source}_combined_spot_clusters_{window}.txt',
                                  sep='\t')
    print('The simulated spot has size ' +
          str(combined_spot_clusters.shape[0]))


data_dir = r"Z:\jxliu\project\datasets\matched\SpatialBenchmarking-main\SpatialBenchmarking-main\FigureData\Figure4\Dataset4_seqFISH+\Rawdata"
data_dir = Path(data_dir)
res_dir = Path(r"Z:\jxliu\project\ID-GAT\ID\simulation\seqFISH-plus")
exp_df = pd.read_csv(data_dir / "Spatial_count.txt",
                     sep='\t',
                     index_col=0)
loc_df = pd.read_csv(data_dir / "Locations_seqfish.txt",
                     sep='\t')
loc_df.columns = ['x', 'y']
ann_df = pd.read_csv(data_dir / "Spatial_annotate.txt",
                     sep='\t', index_col=0)
ann_df_major = ann_df.copy()
for i in range(ann_df_major.shape[0]):
    cell_name = ann_df_major.iloc[i, 1]
    if 'Excitatory' in cell_name:
        cell_name = 'Excitatory'
        ann_df_major.iloc[i, 1] = cell_name
cell_ids = ['cell-ID_' + str(i) for i in range(loc_df.shape[0])]
exp_df.index = cell_ids
loc_df.index = cell_ids
ann_df.index = cell_ids
ann_df_major.index = cell_ids
valid_cells = []
for i in ann_df.index:
    cell_name = ann_df.loc[i, 'celltype']
    if cell_name != 'Non-known':
        valid_cells.append(i)

adata = sc.AnnData(exp_df)
adata.obs.loc[:, ['x', 'y']] = loc_df.values
adata.obs.loc[:, 'major_cell_type'] = ann_df_major.celltype
adata.obs.loc[:, 'minor_cell_type'] = ann_df.celltype
adata = adata[valid_cells, :]
adata.write_h5ad(res_dir / "seqFISH+_reference_data.h5ad")
exp_df.to_csv(res_dir / "seqFISH+_reference_expresssion.csv")
meta_df = adata.obs.copy()
meta_df.to_csv(res_dir / "seqFISH+_reference_meta.csv")
# simulate
spatial_rna = exp_df
spatial_meta = adata.obs.loc[:, 'major_cell_type']
spatial_meta = pd.DataFrame(spatial_meta)
spatial_meta.columns = ['celltype']
spatial_loc = adata.obs.loc[:, ['x', 'y']]
spatial_loc.columns = ['x', 'y']
CoordinateXlable = 'x'
CoordinateYlable = 'y'
source = 'seqFISH-plus_major'
for window in [200, 300, 400, 500, 600, 700]:
    simulated_pseudo_spots(spatial_rna,
                           spatial_meta,
                           spatial_loc,
                           CoordinateXlable,
                           CoordinateYlable,
                           window,
                           outdir=res_dir,
                           source=source)
source = 'seqFISH-plus_minor'
for window in [200, 300, 400, 500, 600, 700]:
    simulated_pseudo_spots(spatial_rna,
                           spatial_meta,
                           spatial_loc,
                           CoordinateXlable,
                           CoordinateYlable,
                           window,
                           outdir=res_dir,
                           source=source)
# export h5ad
source_list = ['seqFISH-plus_major', 'seqFISH-plus_minor']
for source in source_list:
    for window in [200, 300, 400, 500, 600, 700]:
        combined_cell_counts = pd.read_csv(
            res_dir / f'{source}_combined_cell_counts_{window}.txt',
            sep='\t',
            index_col=0)
        cell_ids = [f'spot-{window}-{source[-5:]}-' + str(i) for i in range(1,
                                                                            1 + combined_cell_counts.shape[0])]
        combined_spot_loc = pd.read_csv(res_dir /
                                        f'{source}_combined_Locations_{window}.txt',
                                        sep='\t')
        combined_spot_exp = pd.read_csv(res_dir /
                                        f'{source}_combined_spatial_count_{window}.txt',
                                        sep='\t')
        combined_spot_clusters = pd.read_csv(res_dir /
                                             f'{source}_combined_spot_clusters_{window}.txt',
                                             sep='\t',
                                             index_col=0)
        combined_spot_clusters = combined_spot_clusters.div(combined_spot_clusters.sum(axis=1),
                                                            axis=0)
        combined_cell_counts.index = cell_ids
        combined_spot_exp.index = cell_ids
        combined_spot_loc.index = cell_ids
        combined_spot_clusters.index = cell_ids

        adata = sc.AnnData(combined_spot_exp)
        adata.obs.loc[:, 'cell_counts'] = combined_cell_counts.values
        adata.obs.loc[:, ['x', 'y']] = combined_spot_loc.values
        adata.obsm['ground_truth'] = combined_spot_clusters

        adata.write_h5ad(res_dir / f'{source}_combined_simulation_{window}.h5ad')

# STARmap
data_dir = r"Z:\jxliu\project\datasets\matched\SpatialBenchmarking-main\SpatialBenchmarking-main\FigureData\Figure4\Dataset10_STARmap\Rawdata"
data_dir = Path(data_dir)
res_dir = Path(r"Z:\jxliu\project\ID-GAT\ID\simulation\STARmap")
exp_df = pd.read_csv(data_dir / "Spatial_count.txt",
                     sep='\t',
                     index_col=0)
loc_df = pd.read_csv(data_dir / "Locations.txt",
                     sep='\t')
loc_df.columns = ['x', 'y']
ann_df = pd.read_csv(data_dir / "Spatial_annotate.txt",
                     sep='\t', index_col=0)
ann_df_major = ann_df.copy()
for i in range(ann_df_major.shape[0]):
    cell_name = ann_df_major.iloc[i, 1]
    if 'Excitatory' in cell_name:
        cell_name = 'Excitatory'
        ann_df_major.iloc[i, 1] = cell_name
cell_ids = ['cell-ID_' + str(i) for i in range(loc_df.shape[0])]
exp_df.index = cell_ids
loc_df.index = cell_ids
ann_df.index = cell_ids
ann_df_major.index = cell_ids
valid_cells = []
for i in ann_df.index:
    cell_name = ann_df.loc[i, 'celltype']
    if cell_name != 'Other':
        valid_cells.append(i)

adata = sc.AnnData(exp_df)
adata.obs.loc[:, ['x', 'y']] = loc_df.values
adata.obs.loc[:, 'major_cell_type'] = ann_df_major.celltype
adata.obs.loc[:, 'minor_cell_type'] = ann_df.celltype
adata = adata[valid_cells, :]
adata.write_h5ad(res_dir / "STARmap_reference_data.h5ad")
exp_df.to_csv(res_dir / "STARmap_reference_expresssion.csv")
meta_df = adata.obs.copy()
meta_df.to_csv(res_dir / "STARmap_reference_meta.csv")
# simulate
spatial_rna = exp_df
spatial_meta = adata.obs.loc[:, 'major_cell_type']
spatial_meta = pd.DataFrame(spatial_meta)
spatial_meta.columns = ['celltype']
spatial_loc = adata.obs.loc[:, ['x', 'y']]
spatial_loc.columns = ['x', 'y']
CoordinateXlable = 'x'
CoordinateYlable = 'y'
source = 'STARmap_major'
for window in [300, 400, 500, 600, 700, 800]:
    simulated_pseudo_spots(spatial_rna,
                           spatial_meta,
                           spatial_loc,
                           CoordinateXlable,
                           CoordinateYlable,
                           window,
                           outdir=res_dir,
                           source=source)
source = 'STARmap_minor'
for window in [300, 400, 500, 600, 700, 800]:
    simulated_pseudo_spots(spatial_rna,
                           spatial_meta,
                           spatial_loc,
                           CoordinateXlable,
                           CoordinateYlable,
                           window,
                           outdir=res_dir,
                           source=source)
# export h5ad
source_list = ['STARmap_major', 'STARmap_minor']
for source in source_list:
    for window in [300, 400, 500, 600, 700, 800]:
        combined_cell_counts = pd.read_csv(
            res_dir / f'{source}_combined_cell_counts_{window}.txt',
            sep='\t',
            index_col=0)
        cell_ids = [f'spot-{window}-{source[-5:]}-' + str(i) for i in range(1,
                                                                            1 + combined_cell_counts.shape[0])]
        combined_spot_loc = pd.read_csv(res_dir /
                                        f'{source}_combined_Locations_{window}.txt',
                                        sep='\t')
        combined_spot_exp = pd.read_csv(res_dir /
                                        f'{source}_combined_spatial_count_{window}.txt',
                                        sep='\t')
        combined_spot_clusters = pd.read_csv(res_dir /
                                             f'{source}_combined_spot_clusters_{window}.txt',
                                             sep='\t',
                                             index_col=0)
        combined_spot_clusters = combined_spot_clusters.div(combined_spot_clusters.sum(axis=1),
                                                            axis=0)
        combined_cell_counts.index = cell_ids
        combined_spot_exp.index = cell_ids
        combined_spot_loc.index = cell_ids
        combined_spot_clusters.index = cell_ids

        adata = sc.AnnData(combined_spot_exp)
        adata.obs.loc[:, 'cell_counts'] = combined_cell_counts.values
        adata.obs.loc[:, ['x', 'y']] = combined_spot_loc.values
        adata.obsm['ground_truth'] = combined_spot_clusters

        adata.write_h5ad(res_dir / f'{source}_combined_simulation_{window}.h5ad')

# # seqFISH+
# data_dir = r'Z:\jxliu\datasets\matched\spatial-datasets-master\spatial-datasets-master\data\2019_seqfish_plus_SScortex'
# data_dir = Path(data_dir)
# res_dir = Path(r"Z:\jxliu\project\ID-GAT\ID\simulation\simulations\tmp")
# # seqFISH plus
# exp_df = pd.read_csv(data_dir / "count_matrix/cortex_svz_expression.txt",
#                      sep=' ')
# exp_df = exp_df.transpose()
# loc_df = pd.read_csv(data_dir / "cell_locations/cortex_svz_centroids_coord.txt",
#                      sep='\t', index_col=0)
# ann_df = pd.read_csv(data_dir / "cell_locations/cortex_svz_centroids_annot.txt",
#                      sep='\t', index_col=0)
# # create anndata
# adata = sc.AnnData(exp_df)
# adata.obs.loc[:, ann_df.columns] = ann_df
# adata.obs.loc[:, loc_df.columns] = loc_df
# adata.obsm['spatial'] = loc_df.values
# # plot and simulation
# for fov in [0, 1, 2, 3, 4, 5, 6]:
#     sub_adata = adata[adata.obs.FOV == fov, ]
#     sc.pl.spatial(sub_adata, color='cell_types', spot_size=50)
#     print(sub_adata.shape[0])
#     spatial_rna = pd.DataFrame(sub_adata.X,
#                                index=sub_adata.obs_names,
#                                columns=sub_adata.var_names)
#     spatial_meta = sub_adata.obs.loc[:, 'cell_types']
#     spatial_meta = pd.DataFrame(spatial_meta)
#     spatial_meta.columns = ['celltype']
#     spatial_loc = sub_adata.obs.loc[:, ['X', 'Y']]
#     spatial_loc.columns = ['x', 'y']
#     CoordinateXlable = 'x'
#     CoordinateYlable = 'y'
#     source = f'seqFISH-plus_{fov}'
#     for window in [100, 150, 200, 250, 300]:
#         simulated_pseudo_spots(spatial_rna,
#                                spatial_meta,
#                                spatial_loc,
#                                CoordinateXlable,
#                                CoordinateYlable,
#                                window,
#                                outdir=res_dir,
#                                source=source)
#     spatial_rna.to_csv(res_dir /
#                        f"seqFISH-plus_{fov}_reference_expression.csv")
#     spatial_meta.to_csv(res_dir /
#                        f"seqFISH-plus_{fov}_reference_clusters.csv")
#     ref_adata = sc.AnnData(spatial_rna)
#     ref_adata.obs.loc[:, 'cell_type'] = spatial_meta.values
#     ref_adata.write_h5ad(res_dir / 
#                          f"seqFISH-plus_{fov}_reference.h5ad")  
# # seqFISH+
# data_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\rawdata\Dataset4_seqFISH+\Rawdata"
# data_dir = Path(data_dir)
# # generate simulation datasets with anndata format
# fov = 5
# for window in [100, 150, 200, 250, 300]:
#     source = f'seqFISH-plus_{fov}'
#     combined_cell_counts = pd.read_csv(
#         res_dir / f'{source}_combined_cell_counts_{window}.txt',
#         sep='\t',
#         index_col=0)
#     cell_ids = [f'spot_{window}_{fov}_' + str(i) for i in range(1, 
#                                         1 + combined_cell_counts.shape[0])]
#     combined_spot_loc = pd.read_csv(res_dir /
#                              f'{source}_combined_Locations_{window}.txt',
#                              sep='\t')
#     combined_spot_exp = pd.read_csv(res_dir /
#                              f'{source}_combined_spatial_count_{window}.txt',
#                              sep='\t')
#     combined_spot_clusters = pd.read_csv(res_dir /
#                                   f'{source}_combined_spot_clusters_{window}.txt',
#                                   sep='\t',
#                                   index_col=0)
#     combined_spot_clusters = combined_spot_clusters.div(combined_spot_clusters.sum(axis=1),
#                                                         axis=0)
#     combined_cell_counts.index = cell_ids
#     combined_spot_exp.index = cell_ids
#     combined_spot_loc.index = cell_ids
#     combined_spot_clusters.index = cell_ids

#     adata = sc.AnnData(combined_spot_exp)
#     adata.obs.loc[:, 'cell_counts'] = combined_cell_counts.values
#     adata.obs.loc[:, ['x', 'y']] = combined_spot_loc.values
#     adata.obsm['ground_truth'] = combined_spot_clusters

#     adata.write_h5ad(res_dir / f'{source}_combined_simulation_{window}.h5ad')


# # seqFISH VISp
# data_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation\rawdata\Dataset4_seqFISH+\Rawdata"
# data_dir = Path(data_dir)
# res_dir = Path(r"Z:\jxliu\project\ID-GAT\ID\simulation\simulations\tmp")
# exp_df = pd.read_csv(data_dir / "Spatial_count.txt",
#                      sep='\t',
#                      index_col=0)
# exp_df = exp_df.transpose()
# loc_df = pd.read_csv(data_dir / "cell_locations/cortex_svz_centroids_coord.txt",
#                      sep='\t', index_col=0)
# ann_df = pd.read_csv(data_dir / "cell_locations/cortex_svz_centroids_annot.txt",
#                      sep='\t', index_col=0)
