import numpy as np
import scanpy as sc

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
results_file = './write_paga/hdh.h5ad'
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(3, 3),
                              facecolor='white')

adata = sc.read_h5ad(
    r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\scRNA-seq\scRNA-seq_development_heart_with_meta.h5ad")

ctoi_list = ['Fibroblast_like_1', 'Fibroblast_like_2',
             'Fibroblast_like_3', 'Smooth_muscle_cells',
             'Epicardium_derived_cells']
cells_list = [barcode for barcode in adata.obs_names if adata.obs.loc[barcode,
'cell_type'] in ctoi_list]
adata = adata[cells_list, :]
sc.pp.recipe_zheng17(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=12)
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')
sc.tl.paga(adata, groups='cell_type', model='v1.2')
pos = np.array([[0, 0],
                [-0.02, 0.25],
                [-0.15, 0.1],
                [-0.12, -0.15],
                [0.08, 0.15]])
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.10, save='EPDC_fib_thres-0.1.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.11, save='EPDC_fib_thres-0.11.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.15, save='EPDC_fib_thres-0.15.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.09, save='EPDC_fib_thres-0.09.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.10, save='EPDC_fib_thres-0.1.png', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.11, save='EPDC_fib_thres-0.11.png', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.15, save='EPDC_fib_thres-0.15.png', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.09, save='EPDC_fib_thres-0.09.png', pos=pos)

# all cts
adata = sc.read_h5ad(
    r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\scRNA-seq\scRNA-seq_development_heart_with_meta.h5ad")

# ctoi_list = ['Fibroblast_like_1', 'Fibroblast_like_2',
#               'Fibroblast_like_3', 'Smooth_muscle_cells',
#               'Epicardium_derived_cells']
# cells_list = [barcode for barcode in adata.obs_names if adata.obs.loc[barcode,
#                                                     'cell_type'] in ctoi_list]
# adata = adata[cells_list, :]
sc.settings.set_figure_params(dpi=300, frameon=False, figsize=(4.2, 4.2),
                              facecolor='white')
sc.pp.recipe_zheng17(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=12)
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')
sc.tl.paga(adata, groups='cell_type', model='v1.2')
pos = np.array([[0.52273157, -0.48790005],
                [-2.08860338, -2.02175004],
                [2.94450859, -2.433588],  # Cardiac
                [-1.3059393, -2.94931962],
                [-2.62377116, -0.60949988],
                [1.98117686, -1.77949122],  # DPDC
                [0.04990147, 2.36668678],
                [-0.89897468, 1.74322234],
                [0.81836677, -1.52678992],
                [1.03740293, -2.31662446],
                [1.36881142, -3.03794537],
                [2.69052595, 0.60238018],
                [-2.50250295, 0.91654227],
                [1.90690366, -0.75840165],
                [-1.31359924, 0.43430851]])
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.09, save='all_cts_thres-0.09.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.10, save='all_cts_thres-0.10.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.11, save='all_cts_thres-0.11.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.12, save='all_cts_thres-0.12.pdf', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.09, save='all_cts_thres-0.09.png', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.10, save='all_cts_thres-0.10.png', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.11, save='all_cts_thres-0.11.png', pos=pos)
sc.pl.paga(adata, color=['cell_type'], fontsize=6, node_size_scale=1.5,
           threshold=0.12, save='all_cts_thres-0.12.png', pos=pos)
