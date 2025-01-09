import os
from pathlib import Path

import pandas as pd

# %%% simulation
data_path = Path("Z:/jxliu/project/ID-GAT/ID/results/simulation_tmp")
result_path = Path("Z:/jxliu/project/ID-GAT/ID/results/simulation")
data_list = os.listdir(data_path)
sample_set = set('_'.join(i.split('_')[1:]) for i in data_list)
for sample in sample_set:
    tmp_data_list = [i for i in data_list if sample in i]
    tmp_res = pd.read_csv(data_path / tmp_data_list[0], index_col=0)
    for res in tmp_data_list[1:]:
        tmp_res = tmp_res + pd.read_csv(data_path / res, index_col=0)
    tmp_res = tmp_res / len(tmp_data_list)
    tmp_res.to_csv(result_path / sample)

# %%%
data_path = Path("Z:/jxliu/project/ID-GAT/ID/results/human_developing_heart_tmp")
result_path = Path("Z:/jxliu/project/ID-GAT/ID/results/human_developing_heart")
data_list = os.listdir(data_path)
sample_set = set('_'.join(i.split('_')[1:]) for i in data_list)
for sample in sample_set:
    tmp_data_list = [i for i in data_list if sample in i]
    tmp_res = pd.read_csv(data_path / tmp_data_list[0], index_col=0)
    for res in tmp_data_list[1:]:
        tmp_res = tmp_res + pd.read_csv(data_path / res, index_col=0)
    tmp_res = tmp_res / len(tmp_data_list)
    tmp_res.to_csv(result_path / sample)

# %%%
data_path = Path("Z:/jxliu/project/ID-GAT/ID/results/breast_cancer_tmp")
result_path = Path("Z:/jxliu/project/ID-GAT/ID/results/breast_cancer")
data_list = os.listdir(data_path)
sample_set = set('_'.join(i.split('_')[1:]) for i in data_list)
for sample in sample_set:
    tmp_data_list = [i for i in data_list if sample in i]
    tmp_res = pd.read_csv(data_path / tmp_data_list[0], index_col=0)
    for res in tmp_data_list[1:]:
        tmp_res = tmp_res + pd.read_csv(data_path / res, index_col=0)
    tmp_res = tmp_res / len(tmp_data_list)
    tmp_res.to_csv(result_path / sample)

# %%%
data_path = Path("Z:/jxliu/project/ID-GAT/ID/results/breast_cancer_minor_tmp")
result_path = Path("Z:/jxliu/project/ID-GAT/ID/results/breast_cancer_minor")
data_list = os.listdir(data_path)
sample_set = set('_'.join(i.split('_')[1:]) for i in data_list)
for sample in sample_set:
    tmp_data_list = [i for i in data_list if sample in i]
    tmp_res = pd.read_csv(data_path / tmp_data_list[0], index_col=0)
    for res in tmp_data_list[1:]:
        tmp_res = tmp_res + pd.read_csv(data_path / res, index_col=0)
    tmp_res = tmp_res / len(tmp_data_list)
    tmp_res.to_csv(result_path / sample)
