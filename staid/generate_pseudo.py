# import multiprocessing
import os
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

from staid.utils import merge_real_pseudo
from staid.utils import remove_batch_combat
from staid.utils import sc_cell_type_collect

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore")


def enrichment_binary_page(enrichment_score_df,
                           cutoff=2,
                           lower_bound=2,
                           cutoff2=1):
    """
    According to PAGE scores, determine which cell types will present in each 
    location

    Parameters
    ----------
    enrichment_score_df : dataframe
        The dataframe stores prime enchiment score, (n_cellTYpes, n_spots)
    cutoff : float, optional
        The cutoff used for preserving potential cell types . The default is 2.
    lower_bound : int, optional
        The minimal cell types a spot contains. The default is 3. If the 
        
    cutoff2 : float, optional
        The cutoff for deciding the existence of a cell type. The default is 1.

    Returns
    -------
    enrich_df : dataframe
        

    """

    # Create a dataframe for saving enrichment binary results. The column is
    # 'cell type_list', and the indexes are all locations' index (such as barcodes)
    # The elements in this dataframe is a cell type lists which are enriched.
    enrich_df = pd.DataFrame(columns=["cell_type_list"],
                             index=enrichment_score_df.columns)

    # Obtain enriched cell types for each location
    for i in enrichment_score_df.columns:
        tmp_series = enrichment_score_df.loc[:, i]
        # obtain cell types that have large scores
        tmp_series_cutoff = tmp_series[tmp_series > cutoff]
        # If the enriched cell types are extremely low, select cell types
        # ranking the top k position
        if tmp_series_cutoff.shape[0] < lower_bound:
            tmp_series_cutoff = \
                tmp_series.sort_values(ascending=False)[:lower_bound]
            tmp_series_cutoff = tmp_series_cutoff[tmp_series_cutoff > cutoff2]
        if tmp_series_cutoff.shape[0] == 0:
            tmp_series_cutoff = \
                tmp_series.sort_values(ascending=False)[:lower_bound]
        enrich_df.loc[i, :] = [tmp_series_cutoff.index.tolist()]

    return enrich_df


def enrichment_binary_MIA(enrichment_score_df):
    enrich_df = pd.DataFrame(columns=["cell_type_list"],
                             index=enrichment_score_df.columns)
    for i in enrichment_score_df.columns:
        tmp_index = enrichment_score_df.loc[enrichment_score_df[i] == 1,
                    :].index.tolist()
        enrich_df.loc[i, "cell_type_list"] = tmp_index

    return enrich_df


def spatial_clustering(tmp_spa_adata, clustering_method='leiden',
                       resolution=1):
    '''
    Before synthesizing pseudo spots, clustering may be useful to ensure the 
    label balance of training set, that is, pseudo spots.

    Parameters
    ----------
    tmp_spa_adata : AnnData
        AnnData
    clustering_method : str, optional
        The method used in clustering. The default is 'leiden'.
    resolution : float, optional
        resolution used in clustering. The default is 1.

    Returns
    -------
    DataFrame
        the clustering results by dataframe, whose index are barcodes and colu-
        mn is the label of cluster.

    '''
    # Preprocessing
    sc.pp.filter_genes(tmp_spa_adata, min_cells=3)
    sc.pp.log1p(tmp_spa_adata)
    sc.pp.highly_variable_genes(tmp_spa_adata, n_top_genes=3000)
    sc.tl.pca(tmp_spa_adata, svd_solver='arpack')
    sc.pp.neighbors(tmp_spa_adata)

    # clustering
    if clustering_method == 'leiden':
        sc.tl.leiden(tmp_spa_adata, resolution=resolution)
    elif clustering_method == 'louvain':
        sc.tl.louvain(tmp_spa_adata, resolution=resolution)

    return tmp_spa_adata.obs[clustering_method]


def generator_sampling_absolute(spa_adata,
                                sc_adata,
                                ct_cellList_df_absolute,
                                pseudo_num_absolute,
                                min_cells=1,
                                max_cells=20,
                                source='abs'):
    '''
    Generate pseudo spots according to each spot's enrichment of cell type. 
    Here, ignore the mount difference of spaital cluster.

    Parameters
    ----------
    spa_adata : AnnData
        AnnData contains spatial transcriptomics data.
    sc_adata : AnnData
        AnnData contains scRNA-seq data with annotations.
    ct_cellList_df_absolute : dataframe
        DataFrame, whose index indicates cell types, and column indicates the 
        single cell/barcode list under current cell type.
    pseudo_num_absolute : int
        the number of pseudo spots generated for each real spot.
    min_cells : int, optional
        the lower bound of cell mount appearing in each pseudo spot.
        The default is 1.
    max_cells : int, optional
        the upper bound of cell mount appearing in each pseudo spot. The 
        default is 20.
    method : str, optional
        Similarity metric used in generating pseudo spots. Here, the default is 
        'Pearson'.

    Returns
    -------
    pseudo_df : dataframe
        gene expression matrix, dataframe, whose index indicates pseudo and
        columns indicate gene(HVGs).
    pseudo_index_df : dataframe
        For each pseudo spot, the dataframe record the single cells which 
        constitute this pseudo spots.

    '''
    # Determine the number of pseudo spots
    pseudo_num_per_spot = int(np.ceil(pseudo_num_absolute /
                                      (spa_adata.shape[0] + 1)))
    # Create a dataframe to save pseudo_spot
    pseudo_df = pd.DataFrame(index=sc_adata.var.index)
    pseudo_index_df = pd.DataFrame(index=["cell_selected"])
    for i in spa_adata.enrich_df.index:
        tmp_ct_list = spa_adata.enrich_df.loc[i, :][0]  # current cell types
        tmp_cell_list = []  # store potential cells existing in current spot
        for j in tmp_ct_list:
            tmp_cell_list.extend(ct_cellList_df_absolute.loc[j, :][0].tolist())
        for k in range(pseudo_num_per_spot):
            tmp_cell_num = np.random.randint(min_cells, max_cells + 1,
                                             size=[1])[0]
            tmp_cell_selected = np.random.choice(tmp_cell_list, tmp_cell_num)
            pseudo_index_df[str(i) + "_" + str(k) + '_' + source] = \
                [tmp_cell_selected]
            pseudo_df[str(i) + "_" + str(k) + '_' + source] = \
                sc_adata[tmp_cell_selected, :].X.sum(axis=0).tolist()

    return pseudo_df, pseudo_index_df


def generator_sampling_random(sc_adata,
                              ct_cellList_df_relative,
                              pseudo_num_random,
                              min_cells=1,
                              max_cells=20):
    '''
    Generate pseudo spots randomly

    Parameters
    ----------
    sc_adata : AnnData
        AnnData contains scRNA-seq data with annotations.
    ct_cellList_df_relative : dataframe
        DataFrame, whose index indicates cell types, and column indicates the 
        single cell/barcode list under current cell type.
    pseudo_num_ramdom : int
        the number of pseudo spots generated.
    min_cells : int, optional
        the lower bound of cell mount appearing in each pseudo spot. The default is 1.
    max_cells : int, optional
        the upper bound of cell mount appearing in each pseudo spot. The 
        default is 20.
    Returns
    -------
    pseudo_df : dataframe
        gene expression matrix, dataframe, whose index indicates pseudo and 
        columns indicate gene(HVGs).
    pseudo_index_df : dataframe
        For each pseudo spot, the dataframe record the single cells which cons-
        titute this pseudo spots.

    '''
    # Create a dataframe to save pseudo_spot
    pseudo_df = pd.DataFrame(index=sc_adata.var.index)
    pseudo_index_df = pd.DataFrame(index=["cell_selected"])
    # Merge all cells
    all_cells = []
    for i in range(ct_cellList_df_relative.shape[0]):
        all_cells.extend(ct_cellList_df_relative.iloc[i, 0])

    # pool = multiprocessing.dummy.Pool()
    pseudo_df_dict = dict()
    pseudo_index_dict = dict()

    def _random_generate(i):
        if i % 10 == 0:
            tmp_cell_num = 2
            tmp_cell_selected = np.random.choice(all_cells, tmp_cell_num)
        else:
            tmp_cell_num = np.random.randint(min_cells, max_cells + 1,
                                             size=[1])[0]
            tmp_cell_selected = np.random.choice(all_cells, tmp_cell_num)
        pseudo_df_dict[str(i) + '_random'] = sc_adata[tmp_cell_selected, :].X.sum(axis=0)
        pseudo_index_dict[str(i) + '_random'] = [tmp_cell_selected]

    for i in range(int(pseudo_num_random)):
        _random_generate(i)
    pseudo_df = pd.DataFrame(pseudo_df_dict)
    pseudo_index_df = pd.DataFrame(pseudo_index_dict)
    pseudo_df.index = sc_adata.var.index
    pseudo_index_df.index = ["cell_selected"]
    pseudo_index_df = pseudo_index_df[pseudo_df.columns]

    return pseudo_df, pseudo_index_df


def generator_sampling_relative(spa_adata,
                                sc_adata,
                                ct_cellList_df_relative,
                                pseudo_num_relative,
                                num_candidate=100,
                                min_cells=1,
                                max_cells=20):
    '''
    Similar to generator_sampling_absolute(). There are only one difference.
    Label districbution, that is, clusters' proportion will be considered to
    balance the training set.

    Parameters
    ----------
    spa_adata : AnnData
        AnnData contains spatial transcriptomics data. Note: the clustering 
        should be implemented before this step. And clustering results should
        be found by spa_adata.obs['cluster'].
    sc_adata : AnnData
        AnnData contains scRNA-seq data with annotations.
    ct_cellList_df_relative : dataframe
        DataFrame, whose index indicates cell types, and column indicates the 
        single cell/barcode list under current cell type.
    pseudo_num_relative : int
        the number of pseudo spots generated for each real spot.
    num_candidate : int, optional
        the number of candidate pseudo spots. The default is 100.
    min_cells : int, optional
        the lower bound of cell mount appearing in each pseudo spot.
        The default is 1.
    max_cells : int, optional
        the upper bound of cell mount appearing in each pseudo spot. 
        The default is 20.
    method : str, optional
        Similarity metric used in generating pseudo spots. Here, the default is 
        'Pearson'.

    Returns
    -------
    pseudo_df : dataframe
        gene expression matrix, dataframe, whose index indicates pseudo and columns indicate gene(HVGs).
    pseudo_index_df : dataframe
        For each pseudo spot, the dataframe record the single cells which constitute this pseudo spots.
    '''

    # Counte cluster information
    total_cluster = np.unique(spa_adata.obs["id_cluster"])
    total_cluster = np.sort(total_cluster)

    # Get pseudo_num for each cluster
    tmp_num_per_cluster = pseudo_num_relative / total_cluster.shape[0]
    tmp_num_per_cluster = int(np.ceil(tmp_num_per_cluster))

    # Create dataframes for storing pseudo spots and corresponding cell lists
    pseudo_df = pd.DataFrame(index=sc_adata.var.index)
    pseudo_index_df = pd.DataFrame(index=["cell_selected"])

    # Generate pseudo spots for each cluster
    tmp_pseudo_list = []

    def _cluster_generate(i):
        tmp_spa_adata = spa_adata[spa_adata.obs["id_cluster"] == i, :]
        tmp_spa_adata.enrich_df = \
            spa_adata.enrich_df.loc[tmp_spa_adata.obs_names, :]
        tmp_pseudo_df, tmp_pseudo_index_df = generator_sampling_absolute(
            tmp_spa_adata,
            sc_adata,
            ct_cellList_df_relative,
            tmp_num_per_cluster,
            min_cells=min_cells,
            max_cells=max_cells,
            source='rela')
        tmp_pseudo_list.append([tmp_pseudo_df, tmp_pseudo_index_df])

    # pool.map(_cluster_generate, total_cluster)
    for c in total_cluster:
        _cluster_generate(c)
    for i in range(len(tmp_pseudo_list)):
        pseudo_df = pd.concat((pseudo_df, tmp_pseudo_list[i][0]), axis=1)
        pseudo_index_df = pd.concat((pseudo_index_df,
                                     tmp_pseudo_list[i][1]), axis=1)

    return pseudo_df, pseudo_index_df


def cell_type_composition(pseudo_index_df, sc_cellType_df,
                          source="absolute"):
    '''
    According to cell type annotations and barcodes of single cells constituted
    one pseudo spot, calculating cell type composition of the pseudo spot.

    Parameters
    ----------
    pseudo_index_df : dataframe
        A dataframe,pseudo_num_random whose index indicates the pseudo spot
        names and whose column indicates the lists that store present cells.
    sc_cellType_df : dataframe
        A dataframe, whose index indicates the barcodes of single cells, and 
        whose columns indicate the cell type annotations.
    source : str, optional
       Distinguish the source of pseudo spots to make names unique. The default
       is "absolute".

    Returns
    -------
    pseudo_composition_df : dataframe
        For each generated pseudo spot, the cell type composition could be got.

    '''
    # Obtain all cell types
    all_ct_array = np.unique(sc_cellType_df)
    all_ct_array = np.sort(all_ct_array)
    pseudo_composition_df = pd.DataFrame(index=all_ct_array)
    pseudo_composition_dict = dict()

    def _add_ct_comp(i):
        tmp_cell_list = pseudo_index_df.loc[:, i][0]
        tmp_ct_list = sc_cellType_df[tmp_cell_list]
        tmp_frequency = tmp_ct_list.value_counts(normalize=True)
        tmp_frequency = dict(tmp_frequency)
        pseudo_composition_dict[i + "_" + source] = tmp_frequency

    for i in pseudo_index_df.columns:
        _add_ct_comp(i)
    pseudo_composition_df = pd.DataFrame(pseudo_composition_dict,
                                         index=all_ct_array)

    # Replace Nan with 0
    pseudo_composition_df = pseudo_composition_df.fillna(0)
    # resort
    new_columns = [i + "_" + source for i in pseudo_index_df.columns]
    pseudo_composition_df = pseudo_composition_df[new_columns]

    return pseudo_composition_df


def generate_pseudo_spots(sc_adata_tmp,
                          spa_adata_tmp,
                          enrichment_score_df,
                          min_cells=1,
                          max_cells=20,
                          anno_name="cell_type",
                          enrich_method="MIA",
                          pseudo_num_rate=4,
                          abs_relative_rate=(0.4, 0.4, 0.2),
                          domains=None):
    '''
    Synthesize pseudo spots using enrichment information.

    Parameters
    ----------
    sc_adata_tmp : AnnData
        spatial data.
    spa_adata_tmp : AnnData
        scRNA-seq data with annotation.
    min_cells : int, optional
        the lower bound of cell mount appearing in each pseudo spot.
        The default is 1.
    max_cells : int, optional
        the upper bound of cell mount appearing in each pseudo spot.
        The default is 20.
    anno_name : str, optional
        the column name which indicates cell type annotation. In this way, the
        annotation information can be found by sc_adata.obs[anno_name].
        The default is "cell_type".
    enrich_method : str, optional
        Enrichment method used. The default is "MIA".
    pseudo_num_rate : tuple, optional
        For current real spots, pseudo_num_rate indicates the mount of pseudo
        spots (by rate). The default is 4. That is, Generate 4 times pseudo 
        spots for current real spots.
    abs_relative_rate : tuple, optional
        Indicate the proportions of absolute pseudo spots,
        relative pseudo spots, and random spots. 
        The default is (0.4, 0.4, 0.2).

    Returns
    -------
    pseudo_df : dataframe
        gene expression matrix, dataframe, whose index indicates pseudo and
        columns indicate gene(HVGs).
    pseudo_df_composition : dataframe
        For each generated pseudo spot, the cell type composition could be got.

    '''
    global pseudo_index_df_relative, pseudo_index_df_absolute, pseudo_index_df_random
    sc_adata = sc_adata_tmp.copy()
    spa_adata = spa_adata_tmp.copy()

    # Count cell list for each cell type
    ct_cellList_df_absolute, ct_cellList_df_relative = \
        sc_cell_type_collect(sc_adata, anno_name=anno_name)

    # Obtain binary cell type enrichment analysis
    if enrich_method == "PAGE":
        pass
    else:
        spa_adata.enrich_df = enrichment_binary_MIA(
            enrichment_score_df)

    # Determine the mount of pseudo-spots according to setting pseudo_num_rate
    # Determine the mount of pseudo-spots from absolute cell lists and mount
    # of pseudo-spot from relative cell lists.
    pseudo_num_total = pseudo_num_rate * spa_adata.shape[0]

    pseudo_num_relative = int(pseudo_num_total * abs_relative_rate[1])
    pseudo_num_absolute = int(pseudo_num_total * abs_relative_rate[0])
    pseudo_num_random = int(pseudo_num_total * abs_relative_rate[2])
    pseudo_num_random = pseudo_num_total
    spa_adata.obs['id_cluster'] = domains
    # control the number of scRNA-seq
    from tqdm import tqdm
    pbar = tqdm(range(3), desc="Generate pseudo spots")
    for i in pbar:
        if i == 0:
            pseudo_df_absolute, pseudo_index_df_absolute = generator_sampling_absolute(spa_adata,
                                                                                       sc_adata,
                                                                                       ct_cellList_df_relative,
                                                                                       pseudo_num_absolute,
                                                                                       min_cells=min_cells,
                                                                                       max_cells=max_cells)
        elif i == 1:
            pseudo_df_relative, pseudo_index_df_relative = generator_sampling_relative(spa_adata,
                                                                                       sc_adata,
                                                                                       ct_cellList_df_relative,
                                                                                       pseudo_num_relative,
                                                                                       min_cells=min_cells,
                                                                                       max_cells=max_cells)
        elif i == 2:
            pseudo_df_random, pseudo_index_df_random = generator_sampling_random(sc_adata,
                                                                                 ct_cellList_df_relative,
                                                                                 pseudo_num_random,
                                                                                 min_cells=min_cells,
                                                                                 max_cells=max_cells)

    # Obtain cell type composition from generated pseudo-spots
    pseudo_composition_df_absoulte = cell_type_composition(pseudo_index_df_absolute, sc_adata.obs[anno_name],
                                                           source='absolute')
    pseudo_composition_df_relative = cell_type_composition(pseudo_index_df_relative, sc_adata.obs[anno_name],
                                                           source='relative')
    pseudo_composition_df_random = cell_type_composition(pseudo_index_df_random, sc_adata.obs[anno_name],
                                                         source='random')
    # Merge absolute pseudo-spots and relative pseudo-spots
    pseudo_df_absolute.columns = [i + "_absolute" for i in pseudo_df_absolute.columns]
    pseudo_df_relative.columns = [i + "_relative" for i in pseudo_df_relative.columns]
    pseudo_df_random.columns = [i + "_random" for i in pseudo_df_random.columns]
    pseudo_df = pd.concat((pseudo_df_absolute,
                           pseudo_df_relative,
                           pseudo_df_random), axis=1)
    tmp_df = pd.DataFrame(index=np.unique(sc_adata.obs[anno_name]))
    pseudo_num_abs = pseudo_df_absolute.shape[1]
    pseudo_df_composition = pd.concat((tmp_df,
                                       pseudo_composition_df_absoulte,
                                       pseudo_composition_df_relative,
                                       pseudo_composition_df_random), axis=1)

    pseudo_df_composition = pseudo_df_composition.fillna(0)

    return pseudo_df, pseudo_df_composition, pseudo_num_abs


def generator_iter_absolute(spa_adata,
                            sc_adata,
                            pre_deconvo,
                            ct_cellList_df_absolute,
                            pseudo_num_absolute,
                            num_candidate,
                            min_cells,
                            max_cells,
                            source="abs",
                            perturbation=0.2,
                            ):
    '''
    Similar to the function generator_sampling_absolute(). This function is used
    to synthesize pseudo spots. Differently, here used pre-deconvolution results
    rather than cell type enrichment information.

    Parameters
    ----------
    spa_adata : AnnData
        spatial data
    sc_adata : AnnData
        scRNA-seq data with annotation.
    pre_deconvo : dataframe
        the deconvolution results obtained from previous step/iteration.
    ct_cellList_df_absolute : dataframe
        DataFrame, whose index indicates cell types, and column indicates the 
        single cell/barcode list under current cell type.
    pseudo_num_absolute : int
        the number of pseudo spots generated for each real spot.
    num_candidate : int
        In the process of generation, indicate the mount of candidate pseudo 
        spots.
    min_cells : int
        the lower bound of cell mount appearing in each pseudo spot.
    max_cells : int
        the upper bound of cell mount appearing in each pseudo spot.
    method : str, optional
        Similarity metric used in generating pseudo spots. The default is "Pearson".
    perturbation : float, optinal
        Control the cell type perturbation when generate pseudo spots.
        The default is 0.2.

    Returns
    -------
    pseudo_df : dataframe
        gene expression matrix, dataframe, whose index indicates pseudo and columns indicate gene(HVGs).
    pseudo_index_df : dataframe
        A dataframe, whose index indicates the pseudo spot names, and whose 
        column indicates the lists that store present cells.

    '''

    # determine generating 
    pseudo_num_per_spot = int(np.ceil(pseudo_num_absolute / (spa_adata.shape[0] + 1)))
    # Create a dataframe to save pseudo_spot
    pseudo_df = pd.DataFrame(index=sc_adata.var.index)
    pseudo_index_df = pd.DataFrame(index=["cell_selected"])

    for i in spa_adata.obs.index:
        tmp_pseudo_spot_num = np.random.randint(pseudo_num_per_spot - 2, pseudo_num_per_spot + 3)
        tmp_pseudo_spot_num = max(tmp_pseudo_spot_num, 1)
        tmp_ct_composition = pre_deconvo.loc[i, :]
        for j in range(tmp_pseudo_spot_num):
            tmp_cell_num = np.random.randint(min_cells, max_cells + 1, size=[1])[0]
            if j % 3 == 0:
                tmp_cell_num = int(1.5 * tmp_cell_num)
            tmp_cell_num = max(1, tmp_cell_num)
            tmp_cell_selected = []
            for k in tmp_ct_composition.index:
                percent = tmp_ct_composition[k]
                ratio = np.random.uniform(-0.5, 0.5, 1)[0]
                percent = percent * (1 + ratio)
                tmp_ct_num = percent * tmp_cell_num
                tmp_ct_num = np.around(tmp_ct_num)
                tmp_ct_num = int(tmp_ct_num)
                if tmp_ct_num == 0:
                    continue
                tmp_current_chosen = np.random.choice(
                    ct_cellList_df_absolute.loc[k, :][0].tolist(),
                    tmp_ct_num)
                tmp_cell_selected.extend(tmp_current_chosen.tolist())
            # Ensure that there are cells are selected
            if len(tmp_cell_selected) == 0:
                k = tmp_ct_composition.argmax()
                k = tmp_ct_composition.index[k]
                tmp_current_chosen = np.random.choice(ct_cellList_df_absolute.loc[k, :][0].tolist(), 1)
                tmp_cell_selected.extend(tmp_current_chosen)

            # tmp_cell_selected = np.random.choice(tmp_cell_selected, tmp_cell_num).tolist()
            pseudo_index_df[str(i) + "_" + str(j) + source] = [tmp_cell_selected]
            pseudo_df[str(i) + "_" + str(j) + source] = sc_adata[tmp_cell_selected, :].X.sum(axis=0).tolist()

    return pseudo_df, pseudo_index_df


def generator_iter_relative(spa_adata, sc_adata, pre_deconvo,
                            ct_cellList_df_relative,
                            pseudo_num_relative,
                            num_candidate,
                            min_cells,
                            max_cells,
                            perturbation=0.2):
    '''
    Similar to the function generator_sampling_relative(). This function is 
    used to synthesize pseudo spots. Differently, here used pre-deconvlution
    results rather than cell type enrichment information.

    Parameters
    ----------
    spa_adata : AnnData
        spatial data
    sc_adata : AnnData
        scRNA-seq data with annotation.
    pre_deconvo : dataframe
        the deconvolution results obtained from previous step/iteration.
    ct_cellList_df_relative : dataframe
        DataFrame, whose index indicates cell types, and column indicates the 
        single cell/barcode list under current cell type.
    pseudo_num_absolute : int
        the number of pseudo spots generated for each real spot.
    num_candidate : int
        In the process of generation, indicate the mount of candidate pseudo 
        spots.
    min_cells : int
        the lower bound of cell mount appearing in each pseudo spot.
    max_cells : int
        the upper bound of cell mount appearing in each pseudo spot.

    Returns
    -------
    pseudo_df : dataframe
        gene expression matrix, dataframe, whose index indicates pseudo and col-
        umns indicate gene(HVGs).
    pseudo_index_df : dataframe
        A dataframe, whose index indicates the pseudo spot names, and whose 
        column indicates the lists that store present cells.

    '''
    # Count cluster information
    total_cluster = np.unique(spa_adata.obs["id_cluster"])
    total_cluster = np.sort(total_cluster)

    # Get pseudo_num for each cluster
    tmp_num_per_cluster = pseudo_num_relative / total_cluster.shape[0]
    tmp_num_per_cluster = int(np.ceil(tmp_num_per_cluster))

    # Create dataframes for storing pseudo spots and corresponding cell lists
    pseudo_df = pd.DataFrame(index=sc_adata.var.index)
    pseudo_index_df = pd.DataFrame(index=["cell_selected"])

    # Generate pseudo spots for each cluster
    # pool = multiprocessing.dummy.Pool(len(total_cluster))
    tmp_pseudo_list = []

    def _cluster_generate(i):
        # print(f'cluster: {i}')
        tmp_spa_adata = spa_adata[spa_adata.obs["id_cluster"] == i, :]
        tmp_pseudo_df, tmp_pseudo_index_df = \
            generator_iter_absolute(tmp_spa_adata,
                                    sc_adata,
                                    pre_deconvo,
                                    ct_cellList_df_relative,
                                    tmp_num_per_cluster,
                                    num_candidate=num_candidate,
                                    min_cells=min_cells,
                                    max_cells=max_cells,
                                    source='rela',
                                    perturbation=perturbation)
        tmp_pseudo_list.append([tmp_pseudo_df, tmp_pseudo_index_df])

    # pool.map(_cluster_generate, total_cluster)
    for c in total_cluster:
        _cluster_generate(c)
    for i in range(len(tmp_pseudo_list)):
        pseudo_df = pd.concat((pseudo_df, tmp_pseudo_list[i][0]), axis=1)
        pseudo_index_df = pd.concat((pseudo_index_df,
                                     tmp_pseudo_list[i][1]), axis=1)

    return pseudo_df, pseudo_index_df


def generator_iteration(spa_adata,
                        sc_adata,
                        pre_deconvo,
                        min_cells=1,
                        max_cells=20,
                        pseudo_num_rate=4,
                        anno_name="cell_type",
                        abs_relative_rate=(0.3, 0.4, 0.3),
                        num_candidate=100,
                        method="Pearson",
                        perturbation=0.2):
    """
    Similar to generator_without_density(). This function is used
    to synthesize pseudo spots. Differently, here used pre-deconvlution results
    rather than cell type enrichment information.

    Parameters
    ----------
    spa_adata : AnnData
        spatial data.
    sc_adata : AnnData
        scRNA-seq data with cell type annotation.
    pre_deconvo : dataframe
        the deconvolution results obtained from previous step/iteration.
    min_cells : int, optional
        the lower bound of cell mount appearing in each pseudo spot. The 
        default is 1.
    max_cells : TYPE, optional
        the upper bound of cell mount appearing in each pseudo spot. The 
        default is 20.
    pseudo_num_rate : int, optional
        For current real spots, pseudo_num_rate indicates the mount of pseudo
        spots (by rate). The default is 4. That is, generate 4 times pseudo
        spots for current real spots.
    anno_name : str, optional
        the column name which indicates cell type annotation. In this way, the
        cell type annotation information could be found by 
        sc_adata.obs[anno_name]. The default is "cell_type".
        
    abs_relative_rate : tupple, optional
        Indicate the proportions of absolute psudo spot and relative pseudo sp-
        ot.  The default is (0.6, 0.3, 0.1).
    num_candidate : int, optional
        In the process of generation, indicate the mount of candidate pseudo 
        spots. The default is 100.
    method : str, optional
        Similarity metric used in generating pseudo spots.
        The default is "Pearson".
    perturbation : float, optinal
        Control the cell type perturbation when generate pseudo spots.
        The default is 0.2.

    Returns
    -------
    pseudo_df : dataframe
        gene expression matrix, dataframe, whose index indicates pseudo and 
        columns indicate gene(HVGs).
    pseudo_df_composition : dataframe
        For each generated pseudo spot, the cell type composition could be got.

    """
    # Check whether need to clustering
    global pseudo_index_df_relative, pseudo_index_df_absolute, pseudo_index_df_random
    if "id_cluster" not in spa_adata.obs.columns:
        spa_adata.obs['id_cluster'] = spatial_clustering(spa_adata.copy())
    # Count cell list for each cell type
    ct_cellList_df_absolute, ct_cellList_df_relative = sc_cell_type_collect(sc_adata, anno_name=anno_name)

    # Determine the mount of pseudo-spots according to setting pseudo_num_rate
    # Determine the mount of pseudo-spots from absolute cell lists and the
    # mount of pseudo-spot from relative cell lists.
    pseudo_num_total = pseudo_num_rate * spa_adata.shape[0]
    pseudo_num_relative = int(pseudo_num_total * abs_relative_rate[1])
    pseudo_num_absolute = int(pseudo_num_total * abs_relative_rate[0])
    pseudo_num_random = int(pseudo_num_total * abs_relative_rate[2])

    from tqdm import tqdm
    pbar = tqdm(range(3), desc="Generate pseudo spots")
    for i in pbar:
        if i == 0:
            # Get the synthesised synthetic pseudo spots absolutely
            pseudo_df_absolute, pseudo_index_df_absolute = generator_iter_absolute(spa_adata,
                                                                                   sc_adata,
                                                                                   pre_deconvo,
                                                                                   ct_cellList_df_relative,
                                                                                   pseudo_num_absolute,
                                                                                   num_candidate=num_candidate,
                                                                                   min_cells=min_cells,
                                                                                   max_cells=max_cells,
                                                                                   perturbation=perturbation,
                                                                                   )
        elif i == 1:
            # Get the synthesised synthetic pseudo spots relatively
            pseudo_df_relative, pseudo_index_df_relative = generator_iter_relative(spa_adata,
                                                                                   sc_adata,
                                                                                   pre_deconvo,
                                                                                   ct_cellList_df_relative,
                                                                                   pseudo_num_relative,
                                                                                   num_candidate=num_candidate,
                                                                                   min_cells=min_cells,
                                                                                   max_cells=max_cells,
                                                                                   perturbation=perturbation,
                                                                                   )
        else:
            # Get pseudo spots randomly
            pseudo_df_random, pseudo_index_df_random = generator_sampling_random(sc_adata,
                                                                                 ct_cellList_df_relative,
                                                                                 pseudo_num_random,
                                                                                 min_cells=min_cells,
                                                                                 max_cells=int(max_cells / 1.2))
    # Obtain cell type composition from generated pseudo-spots
    pseudo_composition_df_absoulte = cell_type_composition(pseudo_index_df_absolute,
                                                           sc_adata.obs[anno_name],
                                                           source='absolute')
    pseudo_composition_df_relative = cell_type_composition(pseudo_index_df_relative,
                                                           sc_adata.obs[anno_name],
                                                           source='relative')
    pseudo_composition_df_random = cell_type_composition(pseudo_index_df_random,
                                                         sc_adata.obs[anno_name],
                                                         source='random')
    # Merge absolute pseudo-spots and relative pseudo-spots
    pseudo_df_absolute.columns = [i + "_absolute" for i in pseudo_df_absolute.columns]
    pseudo_df_relative.columns = [i + "_relative" for i in pseudo_df_relative.columns]
    pseudo_df_random.columns = [i + "_random" for i in pseudo_df_random.columns]
    pseudo_df = pd.concat((pseudo_df_absolute,
                           pseudo_df_relative,
                           pseudo_df_random), axis=1)
    pseudo_df_composition = pd.concat((pseudo_composition_df_absoulte,
                                       pseudo_composition_df_relative,
                                       pseudo_composition_df_random), axis=1)

    return pseudo_df, pseudo_df_composition, pseudo_num_random


def generate_merge_initial(sc_adata,
                           spa_adata,
                           marker_genes,
                           enrichment_score_df,
                           anno_name='cell_type',
                           min_cells=1,
                           max_cells=20,
                           abs_relative_rate=(0.6, 0.25, 0.15),
                           pseudo_num_rate=3,
                           enrich_method="MIA",
                           remove_platform=False,
                           domains=None,
                           library_size=1e4):
    # Generate pseudo spots
    domains = domains.astype('str')
    domains_df = pd.DataFrame(domains, index=spa_adata.obs_names)
    if spa_adata.shape[0] > 5000:
        all_barcodes = np.random.choice(spa_adata.obs_names,
                                        5000,
                                        replace=False)
    else:
        all_barcodes = spa_adata.obs_names
    domains_df = domains_df.loc[all_barcodes, :]
    # Generate pseudo spots
    pseudo_df, pseudo_df_composition, rand_size = generate_pseudo_spots(sc_adata_tmp=sc_adata,
                                                                        spa_adata_tmp=spa_adata[all_barcodes, :],
                                                                        enrichment_score_df=enrichment_score_df.loc[:,
                                                                                            all_barcodes],
                                                                        anno_name=anno_name,
                                                                        min_cells=min_cells,
                                                                        max_cells=max_cells,
                                                                        abs_relative_rate=abs_relative_rate,
                                                                        pseudo_num_rate=pseudo_num_rate,
                                                                        enrich_method=enrich_method,
                                                                        domains=domains_df.values.flatten())

    # Merge them with real spots
    spa_X, pseudo_X = merge_real_pseudo(spa_adata,
                                        pseudo_df,
                                        marker_genes,
                                        library_size=library_size)
    # Whether you need to remove batch (platform actually) effects
    if remove_platform:
        spa_X, pseudo_X = remove_batch_combat(spa_X, pseudo_X, rand_size)

    return spa_X, pseudo_X, pseudo_df_composition, rand_size


def generate_merge_iter(sc_adata,
                        spa_adata,
                        pre_deconvo,
                        marker_genes,
                        anno_name='cell_type',
                        min_cells=1,
                        max_cells=15,
                        abs_relative_rate=(0.3, 0.4, 0.3),
                        pseudo_num_rate_iter=30,
                        remove_platform=False,
                        perturbation=0.2,
                        library_size=1e4):
    """
    Parameters
    ----------
    sc_adata : AnnData
        scRNA-seq data with cell type annotation.
    spa_adata : AnnData
        spatial data.
    pre_deconvo : dataframe
        The deconvolution results obtained from previous step/iteration.
    anno_name : str, optional
        cell_type key. The default is 'cell_type'.
    min_cells : int, optional
        the lower bound of cell mount appearing in each pseudo spot. The defau-
        lt is 1.
    max_cells : TYPE, optional
        the upper bound of cell mount appearing in each pseudo spot. The defau-
        lt is 20.
    abs_relative_rate : tupple, optional
        Indicate the proportions of absolute psudo spot and relative pseudo sp-
        ot. The default is (0.3, 0.4, 0.3).
    pseudo_num_rate_iter : int, optional
        The ratio between pseudo spots and real spots. The default is 30.
    pseudo_num_iter : int or None, optional
        The number of pseudo spots that need to be generated. If none, the 
        number of pseudo spots will be obtained by pseudo_num_rate_iter.
        The default is None.
    remove_platform : bool, optional
        Whether need to remove platform effect. This step will be implemented
        by sc.pp.combat(). The default is False.
    library_size : int, optional
        The library size in each spot.

    Returns
    -------
    spa_X : array
        Processed count matrix of real spots.
    pseudo_X : array.
        Processed count matrix of pseudo spots
    pseudo_df_composition : dataframe
        The cell type composition of pseudo spots.

    """
    # Generate new pseudo spots according to pre-deconvolution results.
    if spa_adata.shape[0] > 5000:
        all_barcodes = np.random.choice(spa_adata.obs_names, 5000,
                                        replace=False)
    else:
        all_barcodes = spa_adata.obs_names
    pseudo_df, pseudo_df_composition, rand_size = generator_iteration(
        spa_adata=spa_adata[all_barcodes, :],
        sc_adata=sc_adata,
        pre_deconvo=pre_deconvo.loc[all_barcodes, :],
        pseudo_num_rate=pseudo_num_rate_iter,
        min_cells=min_cells,
        max_cells=max_cells,
        anno_name=anno_name,
        abs_relative_rate=abs_relative_rate,
        perturbation=perturbation)

    spa_X, pseudo_X = merge_real_pseudo(spa_adata,
                                        pseudo_df,
                                        marker_genes,
                                        library_size=library_size)

    # Whether you need to remove batch (platform actually) effects
    if remove_platform:
        spa_X, pseudo_X = remove_batch_combat(spa_X, pseudo_X, rand_size)

    return spa_X, pseudo_X, pseudo_df_composition, rand_size


def generate_merge_iter_tmp(sc_adata,
                            spa_adata,
                            pre_deconvo,
                            marker_genes,
                            anno_name='cell_type',
                            min_cells=1,
                            max_cells=15,
                            abs_relative_rate=(0.3, 0.4, 0.3),
                            pseudo_num_rate_iter=30,
                            remove_platform=False,
                            perturbation=0.2,
                            library_size=1e3):
    """
    Parameters
    ----------
    sc_adata : AnnData
        scRNA-seq data with cell type annotation.
    spa_adata : AnnData
        spatial data.
    pre_deconvo : dataframe
        The deconvolution results obtained from previous step/iteration.
    anno_name : str, optional
        cell_type key. The default is 'cell_type'.
    min_cells : int, optional
        the lower bound of cell mount appearing in each pseudo spot. The default is 1.
    max_cells : TYPE, optional
        the upper bound of cell mount appearing in each pseudo spot. The default is 20.
    abs_relative_rate : tuple, optional
        Indicate the proportions of absolute pseudo spots and relative pseudo spots. The default is (0.3, 0.4, 0.3).
    pseudo_num_rate_iter : int, optional
        The ratio between pseudo spots and real spots. The default is 30.
    pseudo_num_iter : int or None, optional
        The number of pseudo spots that need to be generated. If none, the 
        number of pseudo spots will be obtained by pseudo_num_rate_iter.
        The default is None.
    remove_platform : bool, optional
        Whether you need to remove platform effect. This step will be implemented
        by sc.pp.combat(). The default is False.
    library_size : int, optional
        The library size in each spot.

    Returns
    -------
    pseudo_df: dataframe
        The reconstructed gene expression matrix.
    pseudo_df_composition : dataframe
        The cell type composition of pseudo spots.

    """
    # Generate new pseudo spots according to pre-deconvolution results.
    if spa_adata.shape[0] > 5000:
        all_barcodes = np.random.choice(spa_adata.obs_names,
                                        5000)
    else:
        all_barcodes = spa_adata.obs_names
    pseudo_df, pseudo_df_composition, rand_size = generator_iteration(
        spa_adata=spa_adata[all_barcodes, :],
        sc_adata=sc_adata,
        pre_deconvo=pre_deconvo.loc[all_barcodes, :],
        pseudo_num_rate=pseudo_num_rate_iter,
        min_cells=min_cells,
        max_cells=max_cells,
        anno_name=anno_name,
        abs_relative_rate=abs_relative_rate,
        perturbation=perturbation)
    spa_X, pseudo_X = merge_real_pseudo(spa_adata,
                                        pseudo_df,
                                        marker_genes,
                                        library_size=library_size,
                                        scale=False)
    spa_X = pd.DataFrame(spa_X, index=spa_adata.obs_names,
                         columns=spa_adata.var_names)
    pseudo_X = pd.DataFrame(pseudo_X, index=pseudo_df.columns,
                            columns=pseudo_df.index)

    return spa_X, pseudo_X
