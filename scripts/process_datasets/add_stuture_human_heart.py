import os

import numpy as np
import pandas as pd

data_dir = "Z:/jxliu/project/ID-GAT/ID/data/heart_development/ST-meta/he"
he_df = pd.read_csv(os.path.join(data_dir,
                                 "meta_st.csv"))
data_dir = "Z:/jxliu/project/ID-GAT/ID/data/heart_development/ST-meta"
meta_list = os.listdir(data_dir)
meta_list.remove('he')

res_dir = r"Z:\jxliu\project\ID-GAT\ID\data\heart_development\Developmental_heart-master\data_processed"
# add
samples_list = np.unique(he_df['image'])
for sample in samples_list:
    print(sample)
    sample_name = '_'.join(sample.split('_')[1:3])
    for meta_file in meta_list:
        if sample_name in meta_file:
            break
    # if sample_name not in meta_file:
    #     continue
    # load original meta date
    tmp_meta = pd.read_csv(os.path.join(os.path.join(data_dir,
                                                     meta_file)),
                           index_col=0)
    tmp_df = he_df.loc[he_df.loc[:, 'image'] == sample, :]
    # add index 
    index_num = tmp_meta.index[0].split('x')[0]
    new_index = index_num + 'x' + tmp_df['spot.pos']
    tmp_df.index = new_index
    tmp_df = tmp_df.dropna(axis=0)
    tmp_df = tmp_df[~tmp_df.index.duplicated(keep='first')]
    # Add meta
    intersect_spot = np.intersect1d(tmp_df.index, tmp_meta.index)
    tmp_meta.loc[:, "Annotation"] = "Undefined"
    tmp_meta.loc[:, "Acronym"] = "Undefined"
    tmp_meta.loc[intersect_spot, "Annotation"] = t = tmp_df.loc[intersect_spot,
    'name']
    tmp_meta.loc[intersect_spot, "Acronym"] = tmp_df.loc[intersect_spot,
    'acronym']

    # Plot
