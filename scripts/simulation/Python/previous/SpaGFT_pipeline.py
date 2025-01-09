import os
import random

import numpy as np
import scanpy as sc
import torch
import torch_geometric
from sklearn.metrics import mean_absolute_error

from staid.staid_pred_train import gat_predict

# sc_file_path = sys.argv[1]
# spatial_file_path = sys.argv[2]
# celltype_key = sys.argv[3]
# output_file_path = sys.argv[4]
# spatial_key = [sys.argv[5], sys.argv[6]]
data_dir = r"Z:\jxliu\project\ID-GAT\ID\simulation"
dataset_list = os.listdir(data_dir + "/ST")
output_file_path = r"Z:\jxliu\project\ID-GAT\ID\results\simulation_tmp"
spatial_key = ['x', 'y']
celltype_key = 'cell_type'


def seed_all(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    torch_geometric.seed.seed_everything(2023)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


seed_all(2023)
for dup in range(1, 6):
    for dataset in dataset_list:
        dataset_name = dataset.split('.h5')[0]
        file_name = f'Dup-{str(dup)}_STAID_' + \
                    dataset_name + "_results.csv"
        res_file_path = os.path.join(output_file_path, file_name)
        if "OB" in dataset_name:
            sc_file_path = os.path.join(data_dir + r"\scRNA-seq",
                                        "seqFISH+_OBcortex_single.h5ad")
        else:
            sc_file_path = os.path.join(data_dir + r"\scRNA-seq",
                                        "seqFISH+_SScortex_single.h5ad")
        spatial_file_path = os.path.join(data_dir + r"\ST", dataset)
        # Load data
        sc_adata = sc.read(sc_file_path)
        sp_adata = sc.read(spatial_file_path)
        # correct two datasets
        sp_adata.var_names_make_unique()
        sc_adata.var_names_make_unique()
        # run deconvolution
        prediction = gat_predict(spa_adata=sp_adata,
                                 sc_adata=sc_adata,
                                 anno_key=celltype_key,
                                 lr=0.0005,
                                 num_pseudo=8000,
                                 num_epoch=200,
                                 num_iter=10,
                                 max_cells=20,
                                 remove_platform=False,
                                 device='cuda:0',
                                 error_cutoff=0.005,
                                 k=[4, 15],
                                 hidden_dims=[512, 128, 64],
                                 library_size=1e4,
                                 random_spot_rate=0.3)
        gd_df = sp_adata.obsm['cell_type_proprotion'].copy()
        gd_df = gd_df.loc[prediction.index, prediction.columns]
        mae = mean_absolute_error(prediction, gd_df)
        print(dataset)
        print("MAE: \t", mae)
        # save results
        res_file_path = os.path.join(output_file_path,
                                     file_name)
        prediction.to_csv(res_file_path)
