import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats


def scRNA_processing(sc_data_path, sc_meta_path, transpose=True,
                     anno_name='cell_type', pval_cutoff=0.0001,
                     min_genes=200):
    # Load data
    sc_adata = sc.read_csv(sc_data_path)
    sc_meta = pd.read_csv(sc_meta_path, index_col=0)

    # Whether transpose
    if transpose:
        sc_adata = sc_adata.T
    all_sc_genes = sc_adata.var.index.tolist()

    # Add annotation
    sc_adata.obs[anno_name] = sc_meta[anno_name]

    # Preprocessing
    sc.pp.filter_cells(sc_adata, min_genes=200)
    sc.pp.filter_genes(sc_adata, min_cells=3)
    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)
    sc.pp.highly_variable_genes(sc_adata)
    sc_adata = sc_adata[:, sc_adata.var.highly_variable]

    # Find marker genes for each cell type
    sc.tl.rank_genes_groups(sc_adata, anno_name, method='t-test')
    sc.pl.rank_genes_groups(sc_adata, n_genes=25, sharey=False)

    # Extract marker gene list
    cell_type_array = np.unique(sc_adata.obs[anno_name].values)
    cell_type_array = np.sort(cell_type_array)

    ct_markerList_df = pd.DataFrame(index=cell_type_array,
                                    columns=['marker_genes'])

    for i in cell_type_array:
        tmp_markers = sc_adata.uns['rank_genes_groups']['names'][i]
        tmp_pvalue = sc_adata.uns['rank_genes_groups']['pvals'][i]
        tmp_markers_select = tmp_markers[tmp_pvalue < pval_cutoff]

        # If the number of genes according to p values for current cell type,
        # select the top min_genes genes
        if len(tmp_markers_select) < min_genes:
            tmp_markers_select = tmp_markers[:(min(min_genes,
                                                   len(tmp_markers)))]

        ct_markerList_df.loc[i,] = [tmp_markers_select]

    return all_sc_genes, ct_markerList_df


def spatial_processing(spa_data_path, spa_meta_path,
                       clustering_method='leiden',
                       position_name=['X', 'Y'],
                       transpose=True,
                       n_neighbors=10,
                       n_pcs=50,
                       resolution=0.5,
                       pval_cutoff=0.05,
                       min_genes=200):
    spa_adata = sc.read_csv(spa_data_path)
    spa_meta = pd.read_csv(spa_meta_path, index_col=0)
    if transpose:
        spa_adata = spa_adata.T
    all_spa_genes = spa_adata.var.index.tolist()

    # Add spatial information
    spa_adata.obs[['X', 'Y']] = spa_meta[position_name]

    # Preprocessing
    sc.pp.filter_cells(spa_adata, min_genes=200)
    sc.pp.filter_genes(spa_adata, min_cells=3)
    sc.pp.normalize_total(spa_adata, target_sum=1e4)
    sc.pp.log1p(spa_adata)
    sc.pp.highly_variable_genes(spa_adata)
    spa_adata = spa_adata[:, spa_adata.var.highly_variable]
    sc.tl.pca(spa_adata, svd_solver='arpack')
    sc.pp.neighbors(spa_adata)

    # Clustering
    if clustering_method == 'leiden':
        sc.tl.leiden(spa_adata, resolution=resolution)
    elif clustering_method == 'louvain':
        sc.tl.louvain(spa_adata, resolution=resolution)
    spot_region_df = spa_adata.obs[clustering_method]
    spot_region_df.rename({clustering_method: "region"})

    # Obtain marker genes
    sc.tl.rank_genes_groups(spa_adata, 'leiden', method='t-test', )

    # Extract marker gene dictionary
    region_array = np.unique(spa_adata.obs['leiden'].values)
    region_array = np.sort(region_array)
    region_array = region_array.astype(str)

    region_markerList_df = pd.DataFrame(index=region_array,
                                        columns=['marker_genes'])

    for i in region_array:
        tmp_markers = spa_adata.uns['rank_genes_groups']['names'][i]
        tmp_pvalue = spa_adata.uns['rank_genes_groups']['pvals'][i]
        logfolds = spa_adata.uns['rank_genes_groups']['logfoldchanges'][i]
        tmp_markers_select1 = tmp_markers[tmp_pvalue < pval_cutoff]
        tmp_markers_select2 = tmp_markers[logfolds > 1]
        tmp_markers_select = np.intersect1d(tmp_markers_select1,
                                            tmp_markers_select2)

        # If the number of genes according to p values for current cell type,
        # select the top min_genes genes
        if len(tmp_markers_select) < min_genes:
            tmp_markers_select = tmp_markers[:(min(min_genes,
                                                   len(tmp_markers)))]

        region_markerList_df.loc[i,] = [tmp_markers_select]

    return all_spa_genes, region_markerList_df, spot_region_df


def mia(all_sc_genes, ct_markerList_df, all_spa_genes, region_markerList_df):
    # Find the intersection between all scRNA-seq genes and all spatial genes.
    # The intersection will be viewd as background in the following test.
    intersect_genes = np.intersect1d(all_sc_genes, all_spa_genes)
    intersect_genes_num = len(intersect_genes)
    # Update gene lists using the intersected genes
    for i in ct_markerList_df.index:
        tmp_list = ct_markerList_df.loc[i, :].values[0]
        tmp_list = np.intersect1d(tmp_list, intersect_genes)
        ct_markerList_df.loc[i, :] = [tmp_list]
    for i in region_markerList_df.index:
        tmp_list = region_markerList_df.loc[i, :].values[0]
        tmp_list = np.intersect1d(tmp_list, intersect_genes)
        region_markerList_df.loc[i, :] = [tmp_list]

    # Create a dataframe to save rerichemnt score
    enrich_score_df = pd.DataFrame(0, index=ct_markerList_df.index,
                                   columns=region_markerList_df.index)

    # Calculate enrichment score
    for i in ct_markerList_df.index:
        for j in region_markerList_df.index:
            tmp_intersect_num = len(np.intersect1d(ct_markerList_df.loc[i,
                                                   :].values[0],
                                                   region_markerList_df.loc[j,
                                                   :].values[0]
                                                   ))
            tmp_pval = stats.hypergeom.sf(tmp_intersect_num,
                                          intersect_genes_num,
                                          len(ct_markerList_df.loc[i,
                                              :].values[0]) - tmp_intersect_num,
                                          len(region_markerList_df.loc[j,
                                              :].values[0]) - tmp_intersect_num
                                          )
            tmp_socre = -1 * (np.log(tmp_pval + 1e-50) / np.log(10))
            enrich_score_df.loc[i, j] = tmp_socre

    return enrich_score_df


def spatial_clustering_enrich(spa_adata,
                              clustering_method='leiden',
                              resolution=1.0):
    sc.pp.highly_variable_genes(spa_adata)
    spa_adata = spa_adata[:, spa_adata.var.highly_variable]
    sc.tl.pca(spa_adata, svd_solver='arpack', random_state=2023)
    sc.pp.neighbors(spa_adata)

    # clustering
    if clustering_method == 'leiden':
        sc.tl.leiden(spa_adata, resolution=resolution)
    elif clustering_method == 'louvain':
        sc.tl.louvain(spa_adata, resolution=resolution)

    return spa_adata.obs[clustering_method]


def enrichment_mia(spa_adata,
                   sc_adata,
                   anno_name='cell_type',
                   clustering_method='leiden',
                   resolution=1.0,
                   p=0.05):
    # *************** preprocessing for scRNA-seq adata **********************
    all_sc_genes = sc_adata.var.index.tolist()
    sc_adata_mia = sc_adata.copy()
    # Preprocessing
    sc.pp.filter_cells(sc_adata_mia, min_genes=100)
    sc.pp.filter_genes(sc_adata_mia, min_cells=10)
    sc.pp.normalize_total(sc_adata_mia)
    sc.pp.log1p(sc_adata_mia)
    # Find marker genes for each cell type
    sc.tl.rank_genes_groups(sc_adata_mia, anno_name,
                            method='wilcoxon')

    # Extract marker gene dictionary
    cell_type_array = np.unique(sc_adata_mia.obs[anno_name].values)
    cell_type_array = np.sort(cell_type_array)

    if len(cell_type_array) >= 30:
        min_genes = 300
        ave_genes = int(7000 / len(cell_type_array))
    elif len(cell_type_array) >= 20:
        min_genes = 350
        ave_genes = int(6000 / len(cell_type_array))
    elif len(cell_type_array) >= 10:
        min_genes = 400
        ave_genes = int(5000 / len(cell_type_array))
    else:
        ave_genes = int(4000 / len(cell_type_array))
        min_genes = 500

    ct_markerList_df = pd.DataFrame(index=cell_type_array,
                                    columns=['marker_genes'])
    all_markers = []
    for i in cell_type_array:
        tmp_markers = sc_adata_mia.uns['rank_genes_groups']['names'][i]
        tmp_markers_select = tmp_markers[:ave_genes]
        tmp_markers_select = tmp_markers[:max(min_genes,
                                              len(tmp_markers_select))]
        ct_markerList_df.loc[i,] = [tmp_markers[:200]]
        all_markers.extend(tmp_markers_select)

    # *************** preprocessing for spatial adata *************************
    # spa_adata_mia = spa_adata[:, np.intersect1d(spa_adata.var_names,
    #                                             all_markers)].copy()
    spa_adata_mia = spa_adata.copy()
    all_spa_genes = spa_adata_mia.var_names
    sc.pp.filter_genes(spa_adata_mia, min_cells=20)
    sc.pp.normalize_total(spa_adata_mia)
    sc.pp.log1p(spa_adata_mia)
    spa_adata_mia.obs[clustering_method] = spatial_clustering_enrich(
        spa_adata_mia,
        clustering_method=clustering_method,
        resolution=resolution)

    spot_region_df = spa_adata_mia.obs[clustering_method]
    spot_region_df.rename({clustering_method: "region"})

    # Obtain marker genes
    sc.tl.rank_genes_groups(spa_adata_mia,
                            groupby=clustering_method,
                            method='wilcoxon')

    # Extract marker gene dictionary
    region_array = np.unique(spa_adata_mia.obs[clustering_method].values)
    region_array = np.sort(region_array)
    region_array = region_array.astype(str)

    region_markerList_df = pd.DataFrame(index=region_array,
                                        columns=['marker_genes'])

    for i in region_array:
        tmp_markers = spa_adata_mia.uns['rank_genes_groups']['names'][i]
        tmp_markers_select = tmp_markers[:200]
        region_markerList_df.loc[i,] = [tmp_markers_select[:200]]

    # **********************************MIA*************************************
    enrichment_score = mia(all_sc_genes,
                           ct_markerList_df,
                           all_spa_genes,
                           region_markerList_df)

    enrichment_thres_max = 0.9999 * enrichment_score.max(axis=0)
    enrichment_thres_p = -1 * (np.log(p + 1e-50) / np.log(10))
    enrichment_thres = []
    for v in enrichment_thres_max:
        if enrichment_thres_p > v:
            enrichment_thres.append(v)
        else:
            enrichment_thres.append(enrichment_thres_p)
    enrichment_score = enrichment_score > enrichment_thres
    enrichment_score = enrichment_score.astype(np.int32)
    ct_spot_df = pd.DataFrame(index=enrichment_score.index)
    for i in spa_adata_mia.obs_names:
        ct_spot_df[i] = enrichment_score.loc[:, spa_adata_mia.obs.loc[i,
        clustering_method]]

    all_markers = np.unique(all_markers).tolist()

    return ct_spot_df, all_markers, \
        spa_adata_mia.obs[clustering_method].values
