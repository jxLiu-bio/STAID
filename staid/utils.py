import os
import random
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ss
import sklearn
import torch
import scipy

os.environ['OPENBLAS_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore")


def _load_data(sc_data_path,
               sc_meta_path,
               spa_data_path,
               spa_meta_path,
               sc_transpose=False,
               spa_transpose=False,
               sc_format='csv',
               spa_format='csv'):
    """
    Load scRNA-seq data with cell type annotation and spatial transcriptomics
    data with spatial coordinates.

    Parameters
    ----------
    sc_data_path : str, path object or file-like object
        The data path to scRNA-seq data.
    sc_meta_path : str, path object or file-like object
        The data path to annotation file for scRNA-seq cell type annotation. H-
        ere, the file format should be .csv. The frist column should be barcode
        or other cell_names and the second column should be cell type annotati-
        on for scRNA-seq data.
    spa_data_path : str, path object or file-like object
        The data path to spatial transcriptomics data.
    spa_meta_path : str, path object or file-like object
        The data path to location information and other meta information of cu-
        rrent spatial transcriptomics data. Note, the file format should be csv
        or txt and spatial coordinates are indispensable.
    sc_transpose : bool, optional
        Whether need to transpose the matrix. Here, AnnData object is used. He-
        nce, the count matrix should be cell * gene. The default is False.
    spa_transpose : bool, optional
        Whether need to transpose the matrix. Here, AnnData object is used. He-
        nce, the count matrix should be cell * gene. The default is False.
    sc_format : str, optional
        The file format of scRNA-seq count matrix. The default is 'csv'.
    spa_format : str, optional
        The file format of spatial transcriptomics count matrix. The default is
        'csv'.

    Returns
    -------
    sc_adata : AnnData
        AnnData object of scRNA-seq data, where the count matrix could be found 
        by sc_adata.X and the cell type annotation information could be found 
        by sc_adata.obs.
    spa_adata : AnnData
        AnnData object of spatial transcriptomics data, where the count matrix 
        could be found by sc_adata.X. The location information and other meta
        information could be found by sc_adata.obs.

    """

    # load scRNA-seq data and add meta information
    sc_adata = sc.read_csv(sc_data_path)
    if not sc_transpose:
        sc_adata = sc_adata.T
    sc_meta = pd.read_csv(sc_meta_path, index_col=0)
    ## Merge sc_meta with sx_adata
    sc_meta = sc_meta.loc[np.intersect1d(sc_meta.index, sc_adata.obs.index), :]
    sc_adata = sc_adata[sc_meta.index, :]
    ## Add meta information
    sc_adata.obs[sc_meta.columns] = sc_meta

    # load scRNA-seq data and add meta information
    spa_adata = sc.read_csv(spa_data_path)
    if not spa_transpose:
        spa_adata = spa_adata.T
    spa_meta = pd.read_csv(spa_meta_path, index_col=0)
    # Merge sc_meta with sx_adata
    spa_meta = spa_meta.loc[np.intersect1d(spa_meta.index,
                                           spa_adata.obs.index), :]
    spa_adata = spa_adata[spa_meta.index, :]
    # Add meta information
    spa_adata.obs[spa_meta.columns] = spa_meta

    return sc_adata, spa_adata


def correct_adata(sc_adata, spa_adata):
    common_genes = np.intersect1d(sc_adata.var_names, spa_adata.var_names)
    print(f"There are {len(common_genes)} common genes between two datasets")
    spa_adata = spa_adata[:, common_genes]
    sc_adata = sc_adata[:, common_genes]


def intergration_pp_adata(sc_adata,
                          spa_adata,
                          normalization=False):
    """
    Intergration scRNA-seq with cell type annotation and spatial data. Include
    1. Merge genes using intersection of sc's genes amd spatial's gene
    2. Normlized spa_adata and sc_adata

    Parameters
    ----------
    sc_adata : AnnData
        AnnData object. scRNA-seq data.
    spa_adata : AnnData
        AnnData object. spatial transcriptomics data.
    min_genes_sc : int, optional
        Minimum number of genes expressed required for a cell to pass filtering. 
        The default is 200.
    min_cells_sc : int, optional
        Minimum number of cells expressed required for a gene to pass filtering. 
        The default is 3.
    min_genes_spa : int, optional
        Minimum number of genes expressed required for a cell to pass filtering.
        The default is 200.
    min_spots_spa : int, optional
        Minimum number of cells expressed required for a gene to pass filtering.
        The default is 3.
    normlized_tol_sc : int, optional
        The target_sum when normalize data. If `None`, after normalization, 
        each observation (cell) has a total count equal to the median of total 
        counts for observations (cells) before normalization. The default is 
        1e4.
    normlized_tol_spa : int, optional
        The target_sum when normalize data. If `None`, after normalization, 
        each observation (cell) has a total count equal to the median of total 
        counts for observations (cell, target_sum=normlized_tol_scs) before 
        normalization. The default is 1e4.
    filter_spa_spot : bool, optional
        Whether need to filter spots/pixels in spatial transcriptomics data. 
        The default is False.
    Normlization : bool, optioal
        Whether need to normlizaiton transcrptomics data. The default of False.

    Returns
    -------
    sc_adata : AnnData
        AnnData object of scRNA-seq data.
    spa_adata : AnnData
        AnnData object of spatial transcriptomics data.

    """

    # Select the common genes in both scRNA-seq data and spatial data
    shared_genes = np.intersect1d(sc_adata.var.index, spa_adata.var.index)
    sc_adata = sc_adata[:, shared_genes]
    spa_adata = spa_adata[:, shared_genes]

    # Obtain the count of single cell or spot and save such the information
    sc_adata.obs['count'] = sc_adata.X.sum(axis=1)
    spa_adata.obs['count'] = spa_adata.X.sum(axis=1)

    if normalization:
        sc.pp.normalize_total(sc_adata, target_sum=1e4)
        sc.pp.normalize_total(spa_adata, target_sum=1e4)
    return sc_adata, spa_adata


def sc_cell_type_collect(sc_adata, anno_name='cell_type'):
    """
    Sort cell type annotation for scRNAs-seq data.

    Parameters
    ----------
    sc_adata : AnnData
        AnnData object of scRNA-seq data with cell type annotation. The cell 
        type annotation could be found by sc_adata.obs.
    anno_name : str, optional
        The columns name of cell type annotation in sc_adata.obs.columns. The 
        default is 'cell_type'.

    Returns
    -------
    ct_cellList_df_absolute : DataFrame
        DataFrame contain cell type annotaion list. The index denotes cell
        types. There is only on column in ct_cellList_df_absolute, whose items
        are cell lists thet belong to corresponding cell type.
    ct_cellList_df_relative : TYPE
        DataFrame contain cell type annotaion list. The index denotes cell
        types. There is only on column in ct_cellList_df_absolute, whose items
        are cell lists thet belong to corresponding cell type. Here, these cell
        lists have the same length to ensure each cell type will be treated 
        equally and avoid the domination by some enormous cell types.

    """
    # obtian all available cell types
    cell_type_array = np.unique(sc_adata.obs[anno_name])
    cell_type_array = np.sort(cell_type_array)

    # Create a dataframe to save cell list for each cell type. 
    # Two dataframes will be created, one for absolute cell list and one for 
    # relative cell type list to ensure cell numbers in each cell list are 
    # roughly equal
    ct_cellList_df_absolute = pd.DataFrame(index=cell_type_array,
                                           columns=['cell_list'])
    ct_cellList_df_relative = pd.DataFrame(index=cell_type_array,
                                           columns=['cell_list'])
    ct_cellNum_df_absolute = pd.DataFrame(index=cell_type_array,
                                          columns=['cell_number'])  # cellNum

    # Add cell list to ct_cellList_df_absolute
    for i in cell_type_array:
        tmp_list = sc_adata.obs.index[sc_adata.obs[anno_name] == i]
        tmp_num = tmp_list.shape[0]
        ct_cellList_df_absolute.loc[i, :] = [tmp_list]
        ct_cellNum_df_absolute.loc[i, :] = tmp_num

    # Add cell list to ct_cellList_df_relative
    # cell_num_mean = ct_cellNum_df_absolute.mean(axis=0).values[0]
    cell_num_mean = ct_cellNum_df_absolute.median(axis=0).values[0]
    cell_num_mean = int(np.round(cell_num_mean))

    for i in ct_cellList_df_absolute.index:
        tmp_list = ct_cellList_df_absolute.loc[i, :][0]
        if tmp_list.shape[0] > cell_num_mean:
            tmp_list = np.random.choice(tmp_list, cell_num_mean, replace=False)
        else:
            tmp_list = np.random.choice(tmp_list, cell_num_mean, replace=True)
        ct_cellList_df_relative.loc[i, :] = [tmp_list]

    return ct_cellList_df_absolute, ct_cellList_df_relative


def merge_real_pseudo(spa_adata,
                      pseudo_df,
                      marker_genes,
                      library_size=1e4):
    """
    For synthetic pseudo spots and real spots from spatial data, Merge them and 
    process both real spatial data and new synthetic pseudo  data.
    Include:
            1. Normalize spatial data and find highly variable genes.
            2. Convert pseudo data format from dataframe object to AnnData and
            preprocess it.

    Parameters
    ----------
    spa_adata : AnnData
        Spatial transcriptomics data which need to be processed.
    pseudo_df : dataframe
        Count matrix of synthetic pseudo spots.
    scale : bool, optional
        whether you need to scale count matrices. The default is True.
    library_size : int, optional
        library_size

    Returns
    -------
    spa_count_matrix: array
        Preprocessed spatial count matrix.
    pseudo_count_matrix: array
        Preprocessed pseudo count matrix.

    """
    spa_adata_norm = spa_adata.copy()
    spa_adata_norm = spa_adata_norm[:, marker_genes]

    # Convert pseudo spots dataframe into AnnData and normalization
    pseudo_adata = sc.AnnData(pseudo_df)
    pseudo_adata = pseudo_adata.T
    pseudo_adata = pseudo_adata[:, marker_genes]

    sc.pp.normalize_total(spa_adata_norm, target_sum=library_size)
    sc.pp.log1p(spa_adata_norm)
    sc.pp.normalize_total(pseudo_adata, target_sum=library_size)
    sc.pp.log1p(pseudo_adata)

    mtx_real = spa_adata_norm.X * 0.1
    mtx_pseudo = pseudo_adata.X * 0.1

    return mtx_real, mtx_pseudo


def remove_batch_mnn(exp_mtx1, exp_mtx2, random_size=0):
    exp_mtx1, exp_mtx2 = sc.external.pp.mnn_correct(exp_mtx1, exp_mtx2, )[0]
    return exp_mtx1, exp_mtx2


def remove_batch_combat(exp_mtx1, exp_mtx2, random_size=0):
    sources_list = ['0'] * exp_mtx1.shape[0] + ['1'] * exp_mtx2.shape[0]
    random_list = ['normal'] * int(len(sources_list) - random_size) + \
                  ['random'] * int(random_size)
    adata = sc.AnnData(np.concatenate((exp_mtx1, exp_mtx2), axis=0))
    adata.obs['batch'] = sources_list
    adata.obs['random'] = random_list
    sc.pp.combat(adata, key='batch')

    exp_mtx1 = adata[adata.obs['batch'] == '0', :].X
    exp_mtx2 = adata[adata.obs['batch'] == '1', :].X
    return exp_mtx1, exp_mtx2


def _remove_batch_tmp(df, source_index):
    """
    For real spatial data and synthetic psedo data. Remove "batch effect" for
    two sources.

    Parameters
    ----------
    df : dataframe
        Count matrix contains real spots and pseudo spots.
    source_index : list
        List indicates which spots are real spots and which spots are pseudo 
        spots.

    Returns
    -------
    count_matrix_array
        Count matrix array contains real spots and pseudo spots.

    """
    tmp_adata = sc.AnnData(df).T
    tmp_adata.obs['batch'] = source_index
    # obtain gene co-expresssion network
    net = construct_net(tmp_adata[tmp_adata.obs['batch'] == '0', :],
                        cutoff=False)
    freq = obtain_freq(net,
                       tmp_adata[tmp_adata.obs['batch'] == '0', :],
                       tmp_adata[tmp_adata.obs['batch'] == '1', :])

    freq = freq.transpose()

    return freq


def construct_net(adata, cutoff=None):
    if adata.shape[0] > 2000:
        all_barcodes = np.random.choice(adata.obs_names, 3000)
    else:
        all_barcodes = adata.obs_names
    adata = adata[all_barcodes, :]
    exp_df = pd.DataFrame(adata.X,
                          columns=adata.var_names)
    coor_df = exp_df.corr(method='pearson')
    coor_df[coor_df == 1] = 0
    coor_df = coor_df.fillna(0)

    return coor_df


def softmax_func(net):
    net_x = torch.nn.functional.softmax(net, axis=1)
    net_x = net_x.numpy()
    net_x = pd.DataFrame(net_x,
                         index=net.index,
                         columns=net.columns)
    return net_x


def construct_net_df(df, cutoff=0):
    if df.shape[0] > 2000:  # boost
        all_barcodes = np.random.choice(df.index.tolist(), 2000,
                                        replace=False)
    else:
        all_barcodes = df.index.tolist()

    df = df.loc[all_barcodes, :]
    df = df.corr(method='pearson')
    df[df < 0] = 0  # negative
    df = df - np.eye(df.shape[0])  # diagnal elements
    df = df.fillna(0)

    return df


def _get_lap_mtx(gene_net):
    diag = gene_net.sum(axis=1)
    deg_mtx = _create_degree_mtx(diag)
    adj_mtx = ss.coo_matrix(gene_net)
    lap_mtx = deg_mtx - adj_mtx
    return lap_mtx


def _create_degree_mtx(diag):
    diag = np.array(diag)
    diag = diag.flatten()
    row_index = list(range(diag.size))
    col_index = row_index
    sparse_mtx = ss.coo_matrix((diag, (row_index, col_index)),
                               shape=(diag.size, diag.size))
    return sparse_mtx


def apply_init_gft(exp_df, reduce_dim=100, rand_size=0):
    exp_df_sub = exp_df.iloc[:-rand_size, :]
    if exp_df_sub.shape[0] > 2000:  # boost
        all_barcodes = np.random.choice(exp_df_sub.index,
                                        size=2000,
                                        replace=False)
    else:
        all_barcodes = exp_df_sub.index.tolist()
    reduce_dim = min(exp_df.shape[1] - 1, reduce_dim)
    net = construct_net_df(exp_df_sub.loc[all_barcodes, :])
    lap_mtx = _get_lap_mtx(net)
    v0 = lap_mtx.shape[0] * [1 / np.sqrt(lap_mtx.shape[0])]
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx,
                                       k=reduce_dim,
                                       which='SM',
                                       v0=v0)
    exp_df = sklearn.preprocessing.scale(exp_df, axis=0)
    exp_df = exp_df.transpose()
    eigvecs_T = eigvecs.transpose()
    eigvecs_T = eigvecs_T[1:, :]
    exp_df = np.matmul(eigvecs_T, exp_df)
    exp_df = exp_df.transpose()
    exp_df = sklearn.preprocessing.normalize(exp_df, norm='l2')

    return exp_df, eigvecs_T


def apply_gft(exp_df, eigvecs_T, reduce_dim=100):
    exp_df = sklearn.preprocessing.scale(exp_df, axis=0)
    exp_df = exp_df.transpose()
    exp_df = np.matmul(eigvecs_T, exp_df)
    exp_df = exp_df.transpose()
    exp_df = sklearn.preprocessing.normalize(exp_df, norm='l2')
    return exp_df


def obtain_freq(net,
                adata_q,
                adata_r,
                fms=None,
                normalization='l2',
                cpm=False,
                n_gft=1500,
                weighted=False,
                start=0,
                c=0.0001):
    if cpm:
        sc.pp.normalize_total(adata_q, target_sum=1e3)
        sc.pp.normalize_total(adata_r, target_sum=1e3)
    common_genes = np.intersect1d(net.index.tolist(), adata_r.var_names)
    common_genes = np.intersect1d(common_genes, adata_q.var_names)
    if not fms:
        n_gft = min(n_gft, len(common_genes) - 1)
        net = net.loc[common_genes, common_genes].copy()
        x_exp_r = adata_r[:, common_genes].X.transpose()
        x_exp_q = adata_q[:, common_genes].X.transpose()
    else:
        x_exp_r = adata_r[:, common_genes].X.transpose()
        x_exp_q = adata_q[:, common_genes].X.transpose()
    lap_mtx = _get_lap_mtx(net)
    # Obtain Fourier modes and corresponding eigenvalues
    v0 = lap_mtx.shape[0] * [1 / np.sqrt(lap_mtx.shape[0])]
    eigvals, eigvecs = ss.linalg.eigsh(lap_mtx,
                                       k=n_gft,
                                       which='SM',
                                       v0=v0)
    if weighted:
        power = [1 / (1 + c * eigv) for eigv in eigvals]
        eigvecs = np.matmul(eigvecs, np.diag(power))
    eigvecs_T = eigvecs.transpose()
    if start != 0:
        eigvecs_T = eigvecs_T[start:, :]
        eigvecs_T = eigvecs_T[:-start, :]
    if ss.isspmatrix(x_exp_r):
        x_exp_r = x_exp_r.todense()
    x_exp_r = np.matmul(eigvecs_T, x_exp_r)
    x_exp_r = x_exp_r.transpose()
    if ss.isspmatrix(x_exp_q):
        x_exp_q = x_exp_q.todense()
    x_exp_q = np.matmul(eigvecs_T, x_exp_q)
    x_exp_q = x_exp_q.transpose()
    x_exp = np.concatenate((x_exp_q, x_exp_r), axis=0)
    return x_exp


def remove_low_values(prediction, cutoff=0.01):
    """
    For each predicted cell type composition, filter low values according to 
    the cutoff.

    Parameters
    ----------
    prediction : dataframe
        Dataframe stores predicted cell type composition information. The index
        indicates spots and the columns inticate cell types.
    cutoff : float, optional
        The threshold value. The cell type proportion under cutoff will be set
        to 0. The default is 0.02.

    Returns
    -------
    prediction : dataframe
        ataframe stores predicted cell type composition information.

    """
    prediction[prediction.values < cutoff] = 0
    prediction = prediction.div(prediction.sum(axis=1), axis=0)
    return prediction


def simplify_refer(sc_adata, anno_key, cell_per_ct=200):
    ct_cellList_df_absolute, ct_cellList_df_relative = sc_cell_type_collect(sc_adata, anno_name=anno_key)
    single_cells = []

    def _extract_ct(ct, cells_num):
        if len(np.unique(ct_cellList_df_absolute.loc[ct, 'cell_list'])) > cells_num:
            selected_cells_s = np.random.choice(ct_cellList_df_absolute.loc[ct, 'cell_list'],
                                                cells_num, replace=False)
        else:
            selected_cells_s = ct_cellList_df_absolute.loc[ct, 'cell_list']
        selected_cells_s = np.unique(selected_cells_s).tolist()
        single_cells.extend(selected_cells_s)

    ct_index = 0
    for ct in ct_cellList_df_relative.index:
        cells_num = cell_per_ct
        # cells_num = cell_per_ct
        cells_num = int(cells_num)
        _extract_ct(ct, cells_num)
        ct_index += 1

    # sort the results
    ## sort merged spots
    new_mtx = pd.DataFrame(index=sc_adata.var.index)

    ## merge single cell
    new_mtx[single_cells] = np.matrix(sc_adata[single_cells, :].X.transpose())
    new_mtx = new_mtx.transpose()
    cell_type_list = sc_adata.obs.loc[single_cells, anno_key].tolist()
    # obtain the new anndata object of scRNA-seq
    sc_adata = sc.AnnData(new_mtx)
    sc_adata.obs[anno_key] = cell_type_list

    return sc_adata


def _simplify_refer(sc_adata, anno_key, cell_per_ct=200, tmp_num=1):
    ct_cellList_df_absolute, ct_cellList_df_relative = \
        sc_cell_type_collect(sc_adata, anno_name=anno_key)
    ct_num_list = []
    new_mtx = pd.DataFrame(index=sc_adata.var.index)
    anno_df = pd.DataFrame(index=[anno_key])
    for ct in ct_cellList_df_absolute.index:
        ct_num_list.append(len(ct_cellList_df_absolute.loc[ct, 'cell_list']))
    median_ct_num = int(np.quantile(ct_num_list, q=0.5))
    if median_ct_num > cell_per_ct:
        median_ct_num = cell_per_ct
    median_ct_num = int(median_ct_num)
    selected_cells_list = []
    selected_cells_list2 = []
    for step, ct in enumerate(ct_cellList_df_absolute.index):
        if ct_num_list[step] > median_ct_num:
            selected_cells_s = np.random.choice(
                ct_cellList_df_absolute.loc[ct, 'cell_list'],
                median_ct_num, replace=False)
        else:
            selected_cells_s = ct_cellList_df_absolute.loc[ct, 'cell_list']
            diff_num = min(len(selected_cells_s),
                           int(median_ct_num - len(selected_cells_s)))
            selected_cells_d = np.random.choice(
                ct_cellList_df_absolute.loc[ct, 'cell_list'],
                diff_num, replace=False)
            selected_cells_list2.extend(selected_cells_d)
        selected_cells_list.extend(selected_cells_s)

    # merge
    if not ss.issparse(sc_adata.X):
        new_mtx.loc[:, selected_cells_list] = sc_adata[selected_cells_list,
                                              :].X.transpose()
        anno_df.loc[:, selected_cells_list] = sc_adata.obs.loc[ \
            selected_cells_list,
            anno_key].values
    else:
        new_mtx.loc[:, selected_cells_list] = sc_adata[selected_cells_list,
                                              :].X.toarray().transpose()
        anno_df.loc[:, selected_cells_list] = sc_adata.obs.loc[ \
            selected_cells_list,
            anno_key].values

    if len(selected_cells_list2) > 0:
        new_names_list = [i + "_diff" for i in selected_cells_list2]
        if not ss.issparse(sc_adata.X):

            new_mtx.loc[:, new_names_list] = sc_adata[
                                             selected_cells_list2,
                                             :].X.transpose()
            anno_df.loc[:, new_names_list] = sc_adata.obs.loc[ \
                selected_cells_list2,
                anno_key].values
        else:
            new_mtx.loc[:, new_names_list] = sc_adata[selected_cells_list2,
                                             :].X.toarray().transpose()
            anno_df.loc[:, new_names_list] = sc_adata.obs.loc[ \
                selected_cells_list2,
                anno_key].values

    new_mtx = new_mtx.transpose()
    anno_df = anno_df.transpose()

    # obtain the new anndata object of scRNA-seq
    sc_adata = sc.AnnData(new_mtx)
    sc_adata.obs = anno_df
    return sc_adata


def extract_hvgs(sc_adata, n_genes=3000):
    sc_adata_norm = sc_adata.copy()
    sc.pp.normalize_total(sc_adata_norm)
    sc.pp.log1p(sc_adata_norm)
    sc.pp.highly_variable_genes(sc_adata_norm,
                                n_top_genes=n_genes)
    hvg_list = sc_adata_norm.var.loc[sc_adata_norm.var.highly_variable,
               :].index.tolist()
    return hvg_list


def seed_all(seed=2023):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"


def low_pass_filter_init(data, c=0.001):
    # process data
    data = pd.DataFrame(data,
                        index=[f'cell_{i}' for i in range(data.shape[0])],
                        columns=[f'gene_{i}' for i in range(data.shape[1])])
    if data.shape[0] <= 3000:
        net = data.corr(method='pearson')
    else:
        random_cells = np.random.choice(data.index.tolist(), size=3000, replace=False)
        net = data.loc[random_cells, :].corr(method='pearson')
    net = net.fillna(0)
    net[net < 0.2] = 0
    data = data.values.transpose()
    lap_mtx = _get_lap_mtx(net)
    filter_mtx = np.identity(lap_mtx.shape[0]) - c * lap_mtx
    data = filter_mtx @ data
    data = data.transpose()

    return data, filter_mtx


def low_pass_filter(data, filter_mtx):
    # process data
    data = data.transpose()
    data = filter_mtx @ data
    data = data.transpose()

    return data


def construct_gene_co_expression_network(adata,
                                         cutoff=0.3,
                                         num_cells=5000,
                                         gene_list=None,
                                         ):
    # Determine the number of cells:
    if adata.shape[0] <= num_cells:
        selected_cells = adata.obs_names
    else:
        selected_cells = np.random.choice(adata.obs_names, num_cells, replace=False)
    if gene_list is None:
        gene_list = adata.var_names

    # Extract gene expression data
    if ss.issparse(adata.X):
        gene_expression_data = adata[selected_cells, gene_list].X.toarray()
    else:
        gene_expression_data = adata[selected_cells, gene_list].X

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(gene_expression_data.T)
    correlation_matrix[np.isnan(correlation_matrix)] = 0
    correlation_matrix = correlation_matrix - np.eye(correlation_matrix.shape[0])
    cutoff = correlation_matrix.max() * cutoff
    correlation_matrix[np.abs(correlation_matrix) < cutoff] = 0

    return correlation_matrix


def get_gene_lap_mtx(adata,
                     cutoff=0.5,
                     num_cells=5000,
                     gene_list=None):
    adj_mtx = construct_gene_co_expression_network(adata=adata,
                                                   cutoff=cutoff,
                                                   num_cells=num_cells,
                                                   gene_list=gene_list)
    deg_mtx = np.abs(adj_mtx).sum(axis=1)
    mean_deg = np.mean(deg_mtx)
    deg_mtx[deg_mtx > 0] = deg_mtx[deg_mtx > 0] ** (-0.5)
    deg_mtx = np.diag(deg_mtx)
    lap_mtx = np.eye(deg_mtx.shape[0]) - deg_mtx @ adj_mtx @ deg_mtx

    return lap_mtx, mean_deg


def fourier_modes_gene_network(sc_adata, gene_list, cutoff=0.3):
    lap_mtx_gene, mean_deg = get_gene_lap_mtx(sc_adata,
                                              gene_list=gene_list,
                                              cutoff=cutoff,
                                              )
    eigen_value_gene, eigen_vector_gene = scipy.linalg.eigh(lap_mtx_gene)

    return eigen_vector_gene, eigen_value_gene, mean_deg
