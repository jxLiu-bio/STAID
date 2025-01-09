import os

import pandas as pd
import scanpy as sc

# %%%%%%% Cerebellum
data_dir = r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\exp"
adata = sc.read_h5ad(os.path.join(data_dir,
                                  "GSE116470_F_GRCm38.81.P60Cerebellum_ALT.h5ad"))
celebellum_dict = {1: "GranularNeuron_Gabra6",
                   10: "Endothelial_Flt1",
                   11: "Fibroblast_Like_Dcn",
                   2: "PurkinjeNeuron_Pcp2",
                   3: "Interneurons_Pvalb",
                   4: "Interneurons_and_Other_Nnat",
                   5: "Microglia_Macrophage_C1qb",
                   6: "Oligodendrocyte_Polydendrocyte_Tfr_Tnr",
                   7: "BergmannGlia_Gpr37l1",
                   8: "Astrocyte_Gja1",
                   9: "Choroid_Plexus_Ttr"}
meta_df = pd.read_csv(
    r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\meta\F_GRCm38.81.P60Cerebellum_ALT.cell_cluster_outcomes.csv",
    index_col=0)
meta_df = meta_df.loc[adata.obs_names, :]
meta_df = meta_df.loc[meta_df.reason != 'doublet', :]
meta_df = meta_df.loc[meta_df.reason != 'min_genes', :]
meta_df = meta_df.iloc[:, :-1]
meta_df.loc[:, 'cell_type'] = None
ct_list = [celebellum_dict.get(i) for i in meta_df['cluster']]
meta_df.loc[:, 'cell_type'] = ct_list
meta_df = meta_df.dropna()
adata = adata[meta_df.index, :]
adata.obs = meta_df
var_names = adata.var_names
adata = sc.AnnData(adata.X, obs=adata.obs)
adata.var.index = var_names

adata.write_h5ad(r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad\scRNA_P60Cerebellum.h5ad")
meta_df.to_csv(r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad\scRNA_P60Cerebellum_annotations.csv")
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
# adata = adata[:, adata.var.highly_variable]
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')
sc.pl.umap(adata, color='Dcn')
sc.pl.umap(adata, color='Flt1')
sc.pl.umap(adata, color='Nnat')
sc.pl.umap(adata, color='Gabra6')
sc.pl.umap(adata, color='Pcp2')
sc.pl.umap(adata, color='Gpr37l1')

# %%%%%%% H
data_dir = r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\exp"
adata = sc.read_h5ad(os.path.join(data_dir,
                                  "GSE116470_F_GRCm38.81.P60Hippocampus.h5ad"))
hippocampus_dict = {1: "Interneuron_Gad2",
                    10: "Microglia_Macrophage_C1qb",
                    11: "Ependyma",
                    12: "Choroid_Plexus_Ttr",
                    13: "Neurogenesis_Sox4",
                    14: "Neuron_CajalRetzius_Lhx1",
                    15: "Endothelial_Fit1",
                    16: "Mural_Rgs5Acta2",
                    17: "Fibroblast-Like_Dcn",
                    2: "Neuron_Subiculum_SIc17a6",
                    3: "Neuron_Subiculum_Entorhinal_Nxph3",
                    4: "Neuron_Dentate_C1gl2",
                    5: "Neuron_CA1_Subiculum_Postsubiculum_Dcn-Cbln1-Ptgfr-Fezf2",
                    6: "Neuron_CA2CA3_Pvrl3-Rgs15Calb2",
                    7: "Astrocyte_Gja1",
                    8: "Oligodendrocyte_Tfr",
                    9: "Polydendrocyte_Tnr"}
meta_df = pd.read_csv(
    r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\meta\F_GRCm38.81.P60Hippocampus.cell_cluster_outcomes.csv",
    index_col=0)
meta_df = meta_df.loc[adata.obs_names, :]
meta_df = meta_df.loc[meta_df.reason != 'doublet', :]
meta_df = meta_df.loc[meta_df.reason != 'min_genes', :]
meta_df = meta_df.iloc[:, :-1]
meta_df.loc[:, 'cell_type'] = None
ct_list = [hippocampus_dict.get(i) for i in meta_df['cluster']]
meta_df.loc[:, 'cell_type'] = ct_list
meta_df = meta_df.dropna()

adata = adata[meta_df.index, :]
adata.obs = meta_df
var_names = adata.var_names
adata = sc.AnnData(adata.X, obs=adata.obs)
adata.var.index = var_names
adata.write_h5ad(r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad\scRNA_P60Hippocampus.h5ad")
meta_df.to_csv(r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad\scRNA_P60Hippocampus_annotations.csv")

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
# adata = adata[:, adata.var.highly_variable]
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color='cell_type')
sc.pl.umap(adata, color='Dcn')
sc.pl.umap(adata, color='Flt1')
sc.pl.umap(adata, color='Nnat')
sc.pl.umap(adata, color='Gabra6')
sc.pl.umap(adata, color='Pcp2')
sc.pl.umap(adata, color='Gpr37l1')
sc.pl.umap(adata, color='Tnr')
sc.pl.umap(adata, color='Gad2')

# %%%%%%%%%slide-seq mouse cerebellum
data_dir = r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\exp"
adata = sc.read_h5ad(os.path.join(data_dir,
                                  "Puck-191204-01_Slideseq-v2_hippocampus_MouseBrain_expression.h5ad"))
adata.var.index = adata.var.features.values.tolist()
data_dir = r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\meta"
meta_df = pd.read_csv(os.path.join(data_dir,
                                   "Puck-191204-01_Slideseq-v2_MouseBrain_xy.csv"))
adata.obs.loc[:, 'x'] = meta_df.x.values
adata.obs.loc[:, 'y'] = meta_df.y.values
data_dir = "Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad"
var_names = adata.var_names
adata = sc.AnnData(adata.X, obs=adata.obs)
adata.var_names = var_names
adata.write_h5ad(os.path.join(data_dir,
                              "Puck-191204-01_Slideseq-v2_hippocampus_MouseBrain_expression.h5ad"))

data_dir = r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\exp"
adata = sc.read_h5ad(os.path.join(data_dir,
                                  "Puck-200115-08_Slideseq-v2_hippocampus_MouseBrain_expression.h5ad"))
adata.var.index = adata.var.features.values.tolist()
data_dir = r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\raw_data\meta"
meta_df = pd.read_csv(os.path.join(data_dir,
                                   "Puck-200115-08_Slideseq-v2_MouseBrain_xy.csv"))
adata.obs.loc[:, 'x'] = meta_df.x.values
adata.obs.loc[:, 'y'] = meta_df.y.values
data_dir = "Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad"
var_names = adata.var_names
adata = sc.AnnData(adata.X, obs=adata.obs)
adata.var_names = var_names
adata.write_h5ad(os.path.join(data_dir,
                              "Puck-200115-08_Slideseq-v2_hippocampus_MouseBrain_expression.h5ad"))

# %%%%%%% simulate datasets (calculate pearson)
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import scanpy as sc

sc_adata = sc.read_h5ad(r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad\scRNA_P60Hippocampus.h5ad")
sp_adata = sc.read_h5ad(
    r"Z:\jxliu\datasets\scRNA-seq\mouse_brain_atlas\sorted_data\h5ad\Puck-200115-08_Slideseq-v2_hippocampus_MouseBrain_expression.h5ad")
# create a new dataframe to save new mtx
new_mtx = pd.DataFrame(0, index=sp_adata.obs_names,
                       columns=sc_adata.var_names)
sc_adata_raw = sc_adata.copy()
sc.pp.filter_genes(sc_adata, min_cells=30)
sc.pp.normalize_total(sc_adata, target_sum=1e5)
sc.pp.log1p(sc_adata)
sc.pp.filter_genes(sp_adata, min_cells=30)
sc.pp.normalize_total(sp_adata, target_sum=1e5)
sc.pp.log1p(sp_adata)
common_genes = np.intersect1d(sc_adata.var_names,
                              sp_adata.var_names).tolist()
sc_adata = sc_adata[:, common_genes]
sp_adata = sp_adata[:, common_genes]

sc.pp.highly_variable_genes(sc_adata, n_top_genes=2000)
hvg_list = sc_adata.var_names[sc_adata.var.highly_variable].tolist()
sc_adata = sc_adata[:, hvg_list]
sp_adata = sp_adata[:, hvg_list]
selected_cells = []
pearson_list = []
cell_type_list = []
spot_num = 0

cells_sampled = np.random.choice(sc_adata.obs_names, 30000, replace=False)
sc_adata = sc_adata[:30000, :]

aliged_tup_list = []


def find_best_match(spot_list):
    for spot in spot_list:
        print(spot)
        tmp_exp1 = sp_adata[spot, :].X.toarray()[0]
        tmp_exp2 = sc_adata[0, :].X.toarray()[0]
        pear_max = pearsonr(tmp_exp1, tmp_exp2)[0]
        perfect_cell = sc_adata.obs_names[0]
        for cell in sc_adata.obs_names[1:]:
            tmp_exp2 = sc_adata[cell, :].X.toarray()[0]
            pear_tmp = pearsonr(tmp_exp1, tmp_exp2)[0]
            if pear_tmp > pear_max:
                perfect_cell = cell
                pear_max = pear_tmp
        best_ct = sc_adata.obs.loc[perfect_cell, 'cell_type']
        best_com = [spot, perfect_cell, best_ct, pear_max]
        aliged_tup_list.append(best_com)


find_best_match(np.random.choice(sp_adata.obs_names, 200))

import multiprocessing as mp

cpu_count = mp.cpu_count()
p_instance = mp.Pool(50)

# for i in range(int(np.ceil(sp_adata.shape[0] / 200))):
#     p_instance.map(find_best_match, sp_adata.obs_names[i*200: (i+1)*200])
#     print(i)
spot_lists = [list(range(50)), list(range(50, 100)), list(range(100, 150))]
p_instance.map(find_best_match, spot_lists)
p_instance.close()

# %%%%%%% simulate datasets (simulate directly)
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import scipy.sparse as ss

sc.set_figure_params(dpi=150)

# #### dataset 1
data_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
sp_adata = sc.read_h5ad(data_dir / "Puck-200115-08_Slideseq-v2_hippocampus_MouseBrain_expression.h5ad")
sc_adata = sc.read_h5ad(data_dir / "scRNA_P60Hippocampus.h5ad")
sc_adata.var_names_make_unique()
align_df = pd.read_csv(data_dir / "map/200115.csv", index_col=0)
align_df = align_df.dropna()

new_sp_adata = pd.DataFrame(0,
                            index=align_df.index + "_new",
                            columns=sc_adata.var_names)
new_sp_adata.index = new_sp_adata.index + [str(i) for i in range(1,
                                                                 1 + new_sp_adata.shape[0])]
cell_type_list = []
cell_list = []
coord_list = []
for spot_index in range(align_df.shape[0]):
    spot = align_df.index[spot_index]
    xy_coord = sp_adata.obs.loc[spot, ['x', 'y']].values.tolist()
    coord_list.append(xy_coord)
    cell = align_df.iloc[spot_index, 0]
    cell = cell.split('-')[0]
    cell_list.append(cell)
    cell_type = align_df.iloc[spot_index, 1]
    cell_type_list.append(cell_type)
    exp = sc_adata[cell, :].X.toarray()
    new_sp_adata.iloc[spot_index, :] = exp
    if spot_index % 1000 == 0:
        print(spot_index)

# transform format
new_X = ss.csr_matrix(new_sp_adata)
adata = sc.AnnData(new_X)
adata.obs_names = new_sp_adata.index.tolist()
adata.var_names = new_sp_adata.columns.tolist()
adata.obs['cell_type'] = cell_type_list
adata.obs.loc[:, ['x', 'y']] = np.array(coord_list)

adata_p = adata.copy()
adata_p.obsm['X_umap'] = adata_p.obs.loc[:, ['x', 'y']].values
sc.pl.umap(adata_p, color='cell_type', size=5)

# remove some cells
adata_sub = adata_p.copy()
adata_sub = adata_p[adata_p.obs['x'] > 800, :]
adata_sub = adata_sub[adata_sub.obs['y'] > 800, :]
adata_sub = adata_sub[adata_sub.obs['y'] < 5500, :]
sc.pl.umap(adata_sub, color='cell_type', size=5)

center_point = [(np.max(adata_sub.obs['x']) + np.min(adata_sub.obs['x'])) / 2,
                (np.max(adata_sub.obs['y']) + np.min(adata_sub.obs['y'])) / 2]
distances = np.sqrt(np.sum(np.asarray(adata_sub.obs.loc[:, ['x', 'y']] - np.array(center_point)) ** 2, axis=1))
distances = pd.DataFrame(distances)
distances.index = adata_sub.obs_names
distances.columns = ['distance']
distances = distances.sort_values(by='distance', ascending=False)
distances = distances[distances.distance < 2500]
adata_sub = adata_sub[distances.index, :]
adata = adata[adata_sub.obs_names, :]

# save
res_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
adata.write_h5ad(res_dir / "Puck-200115-08_reconstructed.h5ad")

# #######dataset 2 #########
data_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
sp_adata = sc.read_h5ad(data_dir / "Puck-191204-01_Slideseq-v2_hippocampus_MouseBrain_expression.h5ad")
sc_adata = sc.read_h5ad(data_dir / "scRNA_P60Hippocampus.h5ad")
sc_adata.var_names_make_unique()
align_df = pd.read_csv(data_dir / "map/191204.csv", index_col=0)
align_df = align_df.dropna()

new_sp_adata = pd.DataFrame(0,
                            index=align_df.index + "_new",
                            columns=sc_adata.var_names)
new_sp_adata.index = new_sp_adata.index + [str(i) for i in range(1,
                                                                 1 + new_sp_adata.shape[0])]
cell_type_list = []
cell_list = []
coord_list = []
for spot_index in range(align_df.shape[0]):
    spot = align_df.index[spot_index]
    xy_coord = sp_adata.obs.loc[spot, ['x', 'y']].values.tolist()
    coord_list.append(xy_coord)
    cell = align_df.iloc[spot_index, 0]
    cell = cell.split('-')[0]
    cell_list.append(cell)
    cell_type = align_df.iloc[spot_index, 1]
    cell_type_list.append(cell_type)
    exp = sc_adata[cell, :].X.toarray()
    new_sp_adata.iloc[spot_index, :] = exp
    if spot_index % 1000 == 0:
        print(spot_index)

# transform format
new_X = ss.csr_matrix(new_sp_adata)
adata = sc.AnnData(new_X)
adata.obs_names = new_sp_adata.index.tolist()
adata.var_names = new_sp_adata.columns.tolist()
adata.obs['cell_type'] = cell_type_list
adata.obs.loc[:, ['x', 'y']] = np.array(coord_list)

adata_p = adata.copy()
adata_p.obsm['X_umap'] = adata_p.obs.loc[:, ['x', 'y']].values
sc.pl.umap(adata_p, color='cell_type', size=1)
sc.pl.umap(adata_p, color='Astro', size=0.6)

# remove some cells
adata_sub = adata_p.copy()
adata_sub = adata_p[adata_p.obs['x'] > 700, :]
adata_sub = adata_sub[adata_sub.obs['y'] > 800, :]
adata_sub = adata_sub[adata_sub.obs['y'] < 5800, :]
sc.pl.umap(adata_sub, color='cell_type', size=5)

center_point = [(np.max(adata_sub.obs['x']) + np.min(adata_sub.obs['x'])) / 2,
                (np.max(adata_sub.obs['y']) + np.min(adata_sub.obs['y'])) / 2]
distances = np.sqrt(np.sum(np.asarray(adata_sub.obs.loc[:, ['x', 'y']] - np.array(center_point)) ** 2, axis=1))
distances = pd.DataFrame(distances)
distances.index = adata_sub.obs_names
distances.columns = ['distance']
distances = distances.sort_values(by='distance', ascending=False)
distances = distances[distances.distance < 2555]
adata_sub = adata_sub[distances.index, :]
adata = adata[adata_sub.obs_names, :]

# save
res_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
adata.write_h5ad(res_dir / "Puck-191204-01_reconstructed.h5ad")

# %%%%%%% simulate datasets (simulate bins)
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
import scipy.sparse as ss

# dataset_1
data_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
adata = sc.read_h5ad(data_dir / "Puck-191204-01_reconstructed.h5ad")
max_x, min_x = adata.obs['x'].max(), adata.obs['x'].min()
max_y, min_y = adata.obs['y'].max(), adata.obs['y'].min()
grid = 55
exp_mtx = pd.DataFrame(columns=adata.var_names)
pseudo_spot_list = []
pseudo_spot_coord_list = []
pseudo_spot_cells_list = []
pseudo_spot_celln_list = []
index_sum = 0
for i in range(0, int((max_x - min_x) // grid) + 1):
    for j in range(0, int((max_y - min_y) // grid) + 1):
        rb = [min_x + i * grid, min_y + j * grid]
        lt = [rb[0] + grid, rb[1] + grid]
        adata_sub = adata[adata.obs['x'] >= rb[0], :]
        adata_sub = adata_sub[adata_sub.obs['x'] < lt[0], :]
        adata_sub = adata_sub[adata_sub.obs['y'] >= rb[1], :]
        adata_sub = adata_sub[adata_sub.obs['y'] < lt[1], :]
        num_cells = adata_sub.shape[0]
        if num_cells == 0:
            continue
        else:
            cell_coord = [(rb[0] + lt[0]) / 2, (rb[1] + lt[1]) / 2]
            cell_name = "cell_" + "%.2f" % (cell_coord[0]) + "_" + \
                        "%.2f" % (cell_coord[1])
            pseudo_spot_list.append(cell_name)
            pseudo_spot_coord_list.append(cell_coord)
            pseudo_spot_celln_list.append(num_cells)
            pseudo_spot_cells_list.append(adata_sub.obs_names.tolist())
            exp = adata_sub.X.sum(axis=0).A[0]
            exp_mtx.loc[cell_name, :] = exp
        index_sum += 1
        if index_sum % 100 == 0:
            print(index_sum)

all_cts = adata.obs['cell_type'].cat.categories.tolist()
cell_count_df = pd.DataFrame(0, columns=all_cts,
                             index=pseudo_spot_list)
for spot_index, spot in enumerate(pseudo_spot_list):
    count_ct = adata.obs.loc[pseudo_spot_cells_list[spot_index], 'cell_type']
    count_ct = count_ct.value_counts()
    cell_count_df.loc[spot, count_ct.index] = count_ct.values
cell_density_df = cell_count_df.div(cell_count_df.sum(axis=1), axis=0)
simulated_adata = sc.AnnData(ss.csr_matrix(exp_mtx.values.astype(int)))
simulated_adata.obs_names = exp_mtx.index
simulated_adata.var_names = exp_mtx.columns
simulated_adata.obs.loc[:, ['x', 'y']] = np.array(pseudo_spot_coord_list)
simulated_adata.obs['cell_counts'] = pseudo_spot_celln_list
simulated_adata.obsm['cell_type_proportion'] = cell_density_df
simulated_adata.write_h5ad(data_dir / "Puck-191204-01_55_simulated.h5ad")

# dataset_1 100
data_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
adata = sc.read_h5ad(data_dir / "Puck-191204-01_reconstructed.h5ad")
max_x, min_x = adata.obs['x'].max(), adata.obs['x'].min()
max_y, min_y = adata.obs['y'].max(), adata.obs['y'].min()
grid = 100
exp_mtx = pd.DataFrame(columns=adata.var_names)
pseudo_spot_list = []
pseudo_spot_coord_list = []
pseudo_spot_cells_list = []
pseudo_spot_celln_list = []
index_sum = 0
for i in range(0, int((max_x - min_x) // grid) + 1):
    for j in range(0, int((max_y - min_y) // grid) + 1):
        rb = [min_x + i * grid, min_y + j * grid]
        lt = [rb[0] + grid, rb[1] + grid]
        adata_sub = adata[adata.obs['x'] >= rb[0], :]
        adata_sub = adata_sub[adata_sub.obs['x'] < lt[0], :]
        adata_sub = adata_sub[adata_sub.obs['y'] >= rb[1], :]
        adata_sub = adata_sub[adata_sub.obs['y'] < lt[1], :]
        num_cells = adata_sub.shape[0]
        if num_cells == 0:
            continue
        else:
            cell_coord = [(rb[0] + lt[0]) / 2, (rb[1] + lt[1]) / 2]
            cell_name = "cell_" + "%.2f" % (cell_coord[0]) + "_" + \
                        "%.2f" % (cell_coord[1])
            pseudo_spot_list.append(cell_name)
            pseudo_spot_coord_list.append(cell_coord)
            pseudo_spot_celln_list.append(num_cells)
            pseudo_spot_cells_list.append(adata_sub.obs_names.tolist())
            exp = adata_sub.X.sum(axis=0).A[0]
            exp_mtx.loc[cell_name, :] = exp
        index_sum += 1
        if index_sum % 100 == 0:
            print(index_sum)

all_cts = adata.obs['cell_type'].cat.categories.tolist()
cell_count_df = pd.DataFrame(0, columns=all_cts,
                             index=pseudo_spot_list)
for spot_index, spot in enumerate(pseudo_spot_list):
    count_ct = adata.obs.loc[pseudo_spot_cells_list[spot_index], 'cell_type']
    count_ct = count_ct.value_counts()
    cell_count_df.loc[spot, count_ct.index] = count_ct.values
cell_density_df = cell_count_df.div(cell_count_df.sum(axis=1), axis=0)
simulated_adata = sc.AnnData(ss.csr_matrix(exp_mtx.values.astype(int)))
simulated_adata.obs_names = exp_mtx.index
simulated_adata.var_names = exp_mtx.columns
simulated_adata.obs.loc[:, ['x', 'y']] = np.array(pseudo_spot_coord_list)
simulated_adata.obs['cell_counts'] = pseudo_spot_celln_list
simulated_adata.obsm['cell_type_proportion'] = cell_density_df
simulated_adata.write_h5ad(data_dir / "Puck-191204-01_100_simulated.h5ad")

# dataset_2 55
data_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
adata = sc.read_h5ad(data_dir / "Puck-200115-08_reconstructed.h5ad")
max_x, min_x = adata.obs['x'].max(), adata.obs['x'].min()
max_y, min_y = adata.obs['y'].max(), adata.obs['y'].min()
grid = 55
exp_mtx = pd.DataFrame(columns=adata.var_names)
pseudo_spot_list = []
pseudo_spot_coord_list = []
pseudo_spot_cells_list = []
pseudo_spot_celln_list = []
index_sum = 0
for i in range(0, int((max_x - min_x) // grid) + 1):
    for j in range(0, int((max_y - min_y) // grid) + 1):
        rb = [min_x + i * grid, min_y + j * grid]
        lt = [rb[0] + grid, rb[1] + grid]
        adata_sub = adata[adata.obs['x'] >= rb[0], :]
        adata_sub = adata_sub[adata_sub.obs['x'] < lt[0], :]
        adata_sub = adata_sub[adata_sub.obs['y'] >= rb[1], :]
        adata_sub = adata_sub[adata_sub.obs['y'] < lt[1], :]
        num_cells = adata_sub.shape[0]
        if num_cells == 0:
            continue
        else:
            cell_coord = [(rb[0] + lt[0]) / 2, (rb[1] + lt[1]) / 2]
            cell_name = "cell_" + "%.2f" % (cell_coord[0]) + "_" + \
                        "%.2f" % (cell_coord[1])
            pseudo_spot_list.append(cell_name)
            pseudo_spot_coord_list.append(cell_coord)
            pseudo_spot_celln_list.append(num_cells)
            pseudo_spot_cells_list.append(adata_sub.obs_names.tolist())
            exp = adata_sub.X.sum(axis=0).A[0]
            exp_mtx.loc[cell_name, :] = exp
        index_sum += 1
        if index_sum % 100 == 0:
            print(index_sum)

all_cts = adata.obs['cell_type'].cat.categories.tolist()
cell_count_df = pd.DataFrame(0, columns=all_cts,
                             index=pseudo_spot_list)
for spot_index, spot in enumerate(pseudo_spot_list):
    count_ct = adata.obs.loc[pseudo_spot_cells_list[spot_index], 'cell_type']
    count_ct = count_ct.value_counts()
    cell_count_df.loc[spot, count_ct.index] = count_ct.values
cell_density_df = cell_count_df.div(cell_count_df.sum(axis=1), axis=0)
simulated_adata = sc.AnnData(ss.csr_matrix(exp_mtx.values.astype(int)))
simulated_adata.obs_names = exp_mtx.index
simulated_adata.var_names = exp_mtx.columns
simulated_adata.obs.loc[:, ['x', 'y']] = np.array(pseudo_spot_coord_list)
simulated_adata.obs['cell_counts'] = pseudo_spot_celln_list
simulated_adata.obsm['cell_type_proportion'] = cell_density_df
simulated_adata.write_h5ad(data_dir / "Puck-200115-08_100_simulated.h5ad")

# dataset_2 100
data_dir = Path("Z:/jxliu/datasets/scRNA-seq/mouse_brain_atlas/sorted_data/h5ad")
adata = sc.read_h5ad(data_dir / "Puck-200115-08_reconstructed.h5ad")
max_x, min_x = adata.obs['x'].max(), adata.obs['x'].min()
max_y, min_y = adata.obs['y'].max(), adata.obs['y'].min()
grid = 100
exp_mtx = pd.DataFrame(columns=adata.var_names)
pseudo_spot_list = []
pseudo_spot_coord_list = []
pseudo_spot_cells_list = []
pseudo_spot_celln_list = []
index_sum = 0
for i in range(0, int((max_x - min_x) // grid) + 1):
    for j in range(0, int((max_y - min_y) // grid) + 1):
        rb = [min_x + i * grid, min_y + j * grid]
        lt = [rb[0] + grid, rb[1] + grid]
        adata_sub = adata[adata.obs['x'] >= rb[0], :]
        adata_sub = adata_sub[adata_sub.obs['x'] < lt[0], :]
        adata_sub = adata_sub[adata_sub.obs['y'] >= rb[1], :]
        adata_sub = adata_sub[adata_sub.obs['y'] < lt[1], :]
        num_cells = adata_sub.shape[0]
        if num_cells == 0:
            continue
        else:
            cell_coord = [(rb[0] + lt[0]) / 2, (rb[1] + lt[1]) / 2]
            cell_name = "cell_" + "%.2f" % (cell_coord[0]) + "_" + \
                        "%.2f" % (cell_coord[1])
            pseudo_spot_list.append(cell_name)
            pseudo_spot_coord_list.append(cell_coord)
            pseudo_spot_celln_list.append(num_cells)
            pseudo_spot_cells_list.append(adata_sub.obs_names.tolist())
            exp = adata_sub.X.sum(axis=0).A[0]
            exp_mtx.loc[cell_name, :] = exp
        index_sum += 1
        if index_sum % 100 == 0:
            print(index_sum)

all_cts = adata.obs['cell_type'].cat.categories.tolist()
cell_count_df = pd.DataFrame(0, columns=all_cts,
                             index=pseudo_spot_list)
for spot_index, spot in enumerate(pseudo_spot_list):
    count_ct = adata.obs.loc[pseudo_spot_cells_list[spot_index], 'cell_type']
    count_ct = count_ct.value_counts()
    cell_count_df.loc[spot, count_ct.index] = count_ct.values
cell_density_df = cell_count_df.div(cell_count_df.sum(axis=1), axis=0)
simulated_adata = sc.AnnData(ss.csr_matrix(exp_mtx.values.astype(int)))
simulated_adata.obs_names = exp_mtx.index
simulated_adata.var_names = exp_mtx.columns
simulated_adata.obs.loc[:, ['x', 'y']] = np.array(pseudo_spot_coord_list)
simulated_adata.obs['cell_counts'] = pseudo_spot_celln_list
simulated_adata.obsm['cell_type_proportion'] = cell_density_df
simulated_adata.write_h5ad(data_dir / "Puck-200115-08_55_simulated.h5ad")

# %%%%%%%%%  create new datasets  %%%%%%%%%%
import numpy as np
import pandas as pd
import scanpy as sc
import os
from pathlib import Path

# s2 (omit cells)
data_dir = Path("Z:/jxliu/project/ID-GAT/ID/data/simulation")
sc_adata_all = sc.read_h5ad(data_dir / "tmp/scRNA_P60Hippocampus.h5ad")
omit_cts = ['Endothelial_Fit1', 'Polydendrocyte_Tnr']
sc_adata_s1 = sc.read_h5ad(data_dir / "scRNA_Hippocampus_191204-01_s1.h5ad")

sc_adata_end = sc_adata_s1.copy()
valid_cells = sc_adata_s1.obs_names[sc_adata_s1.obs.cell_type != omit_cts[0]]
sc_adata_end = sc_adata_end[valid_cells, :]
sc_adata_end.write_h5ad(data_dir / "scRNA_Hippocampus_191204-01_s2_Endothelial-Fit1.h5ad")
sc_adata_pol = sc_adata_s1.copy()
valid_cells = sc_adata_s1.obs_names[sc_adata_s1.obs.cell_type != omit_cts[1]]
sc_adata_pol = sc_adata_pol[valid_cells, :]
sc_adata_pol.write_h5ad(data_dir / "scRNA_Hippocampus_191204-01_s2_Polydendrocyte-Tnr.h5ad")
sc_adata_s1.write_h5ad(data_dir / "scRNA_Hippocampus_191204-01_s1.h5ad")

# s3 (add cells)
data_dir = Path("Z:/jxliu/project/ID-GAT/ID/data/simulation")
sc_adata_all = sc.read_h5ad(data_dir / "scRNA_P60Hippocampus.h5ad")
sc_adata_s1 = sc.read_h5ad(data_dir / "scRNA_Hippocampus_191204-01_s1.h5ad")
add_ct = 'Choroid_Plexus_Ttr'
cpt_cells = sc_adata_all.obs_names[sc_adata_all.obs.cell_type == add_ct]
valid_cells = np.union1d(sc_adata_s1.obs_names, cpt_cells)
sc_adata_s2 = sc_adata_all[valid_cells, :]
sc_adata_s2.write_h5ad(data_dir / "scRNA_Hippocampus_191204-01_s3.h5ad")

# s1 (for Puck-200115-08)
sc_adata_s12 = sc.read_h5ad(data_dir / "scRNA_Hippocampus.h5ad")
sc_adata_s12.obs_names = [i.split('-')[0] for i in sc_adata_s12.obs_names]
sc_adata_s12.write_h5ad(data_dir / "scRNA_Hippocampus_200115-08_s1.h5ad")

# generate csv format
data_dir = Path("Z:/jxliu/project/ID-GAT/ID/data/simulation/tmp2")
dataset_list = os.listdir(data_dir)
for dataset in dataset_list:
    dataset_name = dataset.split('.h5ad')[0]
    adata = sc.read_h5ad(data_dir / dataset)
    exp = pd.DataFrame(adata.X.toarray(),
                       index=adata.obs_names,
                       columns=adata.var_names)
    meta = adata.obs.copy()
    exp.to_csv(data_dir / (dataset_name + '_expression.csv'))
    meta.to_csv(data_dir / (dataset_name + '_meta.csv'))
