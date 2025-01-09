# process human developing datasets
import os

import pandas as pd

# Define data dir
data_dir = r"Z:\jxliu\project\ID-GAT\ID\results\human_developing_heart_tmp"
tmp_res_list = os.listdir(data_dir)
# Define res_dir
res_dir = r"Z:\jxliu\project\ID-GAT\ID\results\Human_developing_heart"
# Obtain dataset list
dataset_list = []
for i in tmp_res_list:
    current_dataset = '_'.join(i.split('_')[1:-5])
    if current_dataset not in dataset_list:
        dataset_list.append(current_dataset)

# Extract files and merge them
for dataset in dataset_list:
    current_dataset_list = []
    current_dataset_list.extend([i for i in tmp_res_list if dataset in i])
    merged_df = pd.read_csv(os.path.join(data_dir, current_dataset_list[0]),
                            index_col=0)
    for other_file in current_dataset_list[1:]:
        tmp_df = pd.read_csv(os.path.join(data_dir, other_file),
                             index_col=0)
        merged_df = merged_df + tmp_df
    merged_df = merged_df.div(merged_df.sum(axis=1), axis=0)
    merged_df.to_csv(os.path.join(res_dir,
                                  'GAT-ID_' + dataset + "_results.csv"))

## Adjust SPOTlight results
data_dir = r"Z:\jxliu\project\ID-GAT\ID\results\Others\human_developing_heart"
all_res_list = os.listdir(data_dir)
spotlight_list = []
spotlight_list.extend([i for i in all_res_list if "SPOTlight" in i])
# modify column names
for res in spotlight_list:
    tmp_df = pd.read_csv(os.path.join(data_dir, res),
                         index_col=0)
    old_columns = tmp_df.columns.tolist()
    new_columns = [i.replace('.', '_') for i in old_columns]
    tmp_df.columns = new_columns
    tmp_df.to_csv(os.path.join(data_dir, res))
