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
warnings.filterwarnings("ignore")


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
        DataFrame contain cell type annotation list. The index denotes cell
        types. There is only on column in ct_cellList_df_absolute, whose items
        are cell lists that belong to corresponding cell type.
    ct_cellList_df_relative : TYPE
        DataFrame contain cell type annotation list. The index denotes cell
        types. There is only on column in ct_cellList_df_absolute, whose items
        are cell lists that belong to corresponding cell type. Here, these cell
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
    from sklearn.preprocessing import normalize

    mtx_real = spa_adata_norm.X
    mtx_real = normalize(mtx_real, norm='l2', axis=1)
    mtx_pseudo = pseudo_adata.X
    mtx_pseudo = normalize(mtx_pseudo, norm='l2', axis=1)

    return mtx_real, mtx_pseudo


def remove_batch_mnn(exp_mtx1, exp_mtx2):
    exp_mtx1, exp_mtx2 = sc.external.pp.mnn_correct(exp_mtx1, exp_mtx2, )[0]
    return exp_mtx1, exp_mtx2


def remove_batch_scanorama(exp_mtx1, exp_mtx2, random_size):
    sources_list = ['real'] * exp_mtx1.shape[0] + ['syn'] * exp_mtx2.shape[0]
    random_list = ['real'] * int(len(sources_list) - random_size) + \
                  ['random'] * int(random_size)
    adata = sc.AnnData(np.concatenate((exp_mtx1, exp_mtx2), axis=0))
    adata.obs['batch'] = sources_list
    adata.obs['random'] = random_list
    adata.obsm['X_pca'] = adata.X
    sc.external.pp.scanorama_integrate(adata, key='batch', verbose=False)
    exp_mtx1 = adata[adata.obs['batch'] == 'real', :].obsm['X_scanorama']
    exp_mtx2 = adata[adata.obs['batch'] == 'syn', :].obsm['X_scanorama']

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


def remove_low_values(prediction, cutoff=0.01):
    """
    For each predicted cell type composition, filter low values according to the cutoff.

    Parameters
    ----------
    prediction : dataframe
        stores predicted cell type composition information. The index
        indicates spots and the columns indicate cell types.
    cutoff : float, optional
        The threshold value. The cell type proportion under cutoff will be set to 0. The default is 0.01.

    Returns
    -------
    prediction : dataframe
        stores predicted cell type composition information.

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
        cells_num = int(cells_num)
        _extract_ct(ct, cells_num)
        ct_index += 1

    # sort the results
    new_mtx = pd.DataFrame(index=sc_adata.var.index)

    ## merge single cell
    new_mtx[single_cells] = np.matrix(sc_adata[single_cells, :].X.transpose())
    new_mtx = new_mtx.transpose()
    cell_type_list = sc_adata.obs.loc[single_cells, anno_key].tolist()
    # obtain the new anndata object of scRNA-seq
    sc_adata = sc.AnnData(new_mtx)
    sc_adata.obs[anno_key] = cell_type_list

    return sc_adata


def _simplify_refer(sc_adata, anno_key, cell_per_ct=200):
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


def construct_gene_co_expression_network(adata,
                                         cutoff=0.3,
                                         num_cells=5000,
                                         gene_list=None,
                                         method='cosine',
                                         ):
    from scipy.stats import spearmanr
    # Determine the number of cells:
    if adata.shape[0] <= num_cells:
        selected_cells = adata.obs_names
    else:
        selected_cells = np.random.choice(adata.obs_names, num_cells, replace=False)
    if gene_list is None:
        gene_list = adata.var_names
    tmp_adata = adata[selected_cells, gene_list].copy()
    sc.pp.normalize_total(tmp_adata)
    sc.pp.log1p(tmp_adata)
    # sc.pp.scale(tmp_adata)

    # Extract gene expression data
    if ss.issparse(tmp_adata.X):
        gene_expression_data = tmp_adata.X.toarray()
    else:
        gene_expression_data = tmp_adata.X

    # Calculate the correlation matrix
    if method == 'pearson':
        correlation_matrix = np.corrcoef(gene_expression_data.T)
    elif method == 'spearman':
        correlation_matrix, _ = spearmanr(gene_expression_data, axis=0)
    elif method == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        correlation_matrix = cosine_similarity(gene_expression_data.T)
    else:
        raise ValueError('Invalid method')

    correlation_matrix[np.isnan(correlation_matrix)] = 0
    correlation_matrix = correlation_matrix - np.eye(correlation_matrix.shape[0])
    cutoff = np.abs(correlation_matrix.max()) * cutoff
    correlation_matrix[correlation_matrix < cutoff] = 0

    return correlation_matrix


def get_gene_lap_mtx(adata,
                     cutoff=0.5,
                     num_cells=5000,
                     gene_list=None,
                     method='cosine'):
    adj_mtx = construct_gene_co_expression_network(adata=adata,
                                                   cutoff=cutoff,
                                                   num_cells=num_cells,
                                                   gene_list=gene_list,
                                                   method=method)
    deg_mtx = np.abs(adj_mtx).sum(axis=1)
    mean_deg = np.mean(deg_mtx)
    deg_mtx[deg_mtx > 0] = deg_mtx[deg_mtx > 0] ** (-0.5)
    lap_mtx = np.eye(deg_mtx.shape[0]) - deg_mtx @ adj_mtx @ deg_mtx

    return lap_mtx, mean_deg


def fourier_modes_gene_network(sc_adata, gene_list, cutoff=0.3, method='cosine'):
    lap_mtx_gene, mean_deg = get_gene_lap_mtx(sc_adata,
                                              gene_list=gene_list,
                                              cutoff=cutoff,
                                              method=method
                                              )
    eigen_value_gene, eigen_vector_gene = scipy.linalg.eigh(lap_mtx_gene)
    eigen_vector_gene = eigen_vector_gene
    eigen_value_gene = eigen_value_gene

    return eigen_vector_gene, eigen_value_gene, mean_deg


def seed_everything(seed):
    import numpy as np
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
