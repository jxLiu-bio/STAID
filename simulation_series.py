# %%% *** Scenario 1 ****
from pathlib import Path
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_absolute_error
from staid import run_deconvolution

device = 'cuda:0'
print(device)
sample_list = ['Puck-191204-01_55_simulated.h5ad',
               'Puck-191204-01_100_simulated.h5ad',
               'Puck-200115-08_55_simulated.h5ad',
               'Puck-200115-08_100_simulated.h5ad']

res_dir = Path("/opt/data/private/jxliu/project/staid/results/simulation/s1")
data_dir = Path("/opt/data/private/jxliu/project/staid/data/simulation")
mae_list = []
for sample in sample_list[-1:]:
    # determine parameters
    if '55' in sample:
        ave_cells = 8
    else:
        ave_cells = 15
    if '191204' in sample:
        ref_dataset = "scRNA_Hippocampus_191204-01_s1.h5ad"
    else:
        ref_dataset = "scRNA_Hippocampus_200115-08_s1.h5ad"
    file_name = "STAID_" + sample.split('.h5ad')[0] + "_s1_results.csv"
    sp_adata = sc.read_h5ad(data_dir / sample)
    sc_adata = sc.read_h5ad(data_dir / ref_dataset)
    sp_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()

    celltype_key = 'cell_type'
    prediction = run_deconvolution(spa_adata=sp_adata,
                                   sc_adata=sc_adata,
                                   anno_key=celltype_key,
                                   lr=0.0005,
                                   num_pseudo=5000,
                                   num_iter=10,
                                   max_cells=ave_cells,
                                   remove_platform=False,
                                   device=device,
                                   batch_size=128,
                                   hidden_dims=[512, 256, 256],
                                   library_size=1e4,
                                   error_cutoff=0.005,
                                   dropout=0.01)
    # prediction.to_csv(res_dir / file_name)
    common_cts = np.intersect1d(prediction.columns,
                                sp_adata.obsm['cell_type_proportion'].columns)
    gd_df = sp_adata.obsm['cell_type_proportion'].loc[:, common_cts]
    prediction = prediction.loc[:, common_cts]
    sp_adata.obs.loc[:, prediction.columns] = prediction
    sp_adata.obsm['spatial'] = sp_adata.obs.loc[:, ['x', 'y']].values
    sc.pl.spatial(adata=sp_adata,
                  color=prediction.columns,
                  spot_size=100)
    mae = mean_absolute_error(gd_df, prediction)
    mae_list.append(mae)
    print(sample, mae)


# %%% *** Scenario 2 ****
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from staid import run_deconvolution
from torch_geometric import seed_everything

sample_list = ['Puck-191204-01_55_simulated.h5ad']
sample = sample_list[0]
ref_list = ['scRNA_Hippocampus_191204-01_s2_Endothelial-Fit1.h5ad',
            'scRNA_Hippocampus_191204-01_s2_Polydendrocyte-Tnr.h5ad']
res_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/results/simulation/s2")
data_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/data/simulation")

for ref_dataset in ref_list[:1]:
    device = np.random.choice(['cuda:1', 'cuda:2', 'cuda:3', 'cuda:0'])
    max_cells = 8
    if 'Endothelial' in ref_dataset:
        ref = 'End'
    else:
        ref = 'Pol'
    file_name = "STAID_" + sample.split('.h5a')[0] + f"_s2_{ref}_results.csv"
    seed_everything(2023)
    sp_adata = sc.read_h5ad(data_dir / sample)
    sc_adata = sc.read_h5ad(data_dir / ref_dataset)
    sp_adata.var_names_make_unique()
    sc_adata.var_names_make_unique()

    celltype_key = 'cell_type'
    prediction = run_deconvolution(spa_adata=sp_adata,
                                   sc_adata=sc_adata,
                                   anno_key=celltype_key,
                                   lr=0.0005,
                                   num_pseudo=5000,
                                   num_iter=20,
                                   min_cells=1,
                                   max_cells=max_cells,
                                   remove_platform=False,
                                   error_cutoff=0.005,
                                   device=device,
                                   batch_size=128,
                                   hidden_dims=[256, 64, 32],
                                   library_size=1e3,
                                   random_spot_rate=0.3,
                                   dropout=0.05)
    prediction.to_csv(res_dir / file_name)
    common_cts = np.intersect1d(prediction.columns,
                                sp_adata.obsm['cell_type_proportion'].columns)
    gd_df = sp_adata.obsm['cell_type_proportion'].loc[:, common_cts]
    prediction = prediction.loc[:, common_cts]
    mae = mean_absolute_error(gd_df, prediction)
    print(mae)

# %%%*** Scenario 3 ****
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from staid import run_deconvolution
from torch_geometric import seed_everything

device = np.random.choice(['cuda:1', 'cuda:2', 'cuda:3', 'cuda:0'])
sample_list = ['Puck-191204-01_55_simulated.h5ad']
sample = sample_list[0]
res_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/results/simulation/s3")
data_dir = Path("/data/bqliu_data/jxliu_data/projects/staid/data/simulation")
ref_dataset = 'scRNA_Hippocampus_191204-01_s3.h5ad'

max_cells = 8
file_name = "STAID_" + sample.split('.h5a')[0] + "_s3_results.csv"
seed_everything(2023)
sp_adata = sc.read_h5ad(data_dir / sample)
sc_adata = sc.read_h5ad(data_dir / ref_dataset)
sp_adata.var_names_make_unique()
sc_adata.var_names_make_unique()

celltype_key = 'cell_type'
prediction = run_deconvolution(spa_adata=sp_adata,
                               sc_adata=sc_adata,
                               anno_key=celltype_key,
                               lr=0.00005,
                               num_pseudo=5000,
                               num_iter=20,
                               min_cells=1,
                               max_cells=max_cells,
                               remove_platform=False,
                               error_cutoff=0.005,
                               device=device,
                               batch_size=128,
                               hidden_dims=[512, 256, 256],
                               library_size=1e3,
                               random_spot_rate=0.3,
                               dropout=0.05)
prediction.to_csv(res_dir / file_name)
common_cts = np.intersect1d(prediction.columns,
                            sp_adata.obsm['cell_type_proportion'].columns)
gd_df = sp_adata.obsm['cell_type_proportion'].loc[:, common_cts]
prediction = prediction.loc[:, common_cts]
mae = mean_absolute_error(gd_df, prediction)
print(mae)
