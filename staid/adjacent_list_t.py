import warnings

import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA, IncrementalPCA

from staid.utils import apply_gft, apply_init_gft

warnings.filterwarnings("ignore")


class subgraph_sampling():
    def __init__(self,
                 method="pearson",
                 reduce_dim=None):
        self.fms_real = None
        self.pseudo_dense_adj = None
        self.real_dense_adj = None
        self.pca_model_real = None
        self.reduce_dim = reduce_dim
        self.fms = [1]
        self.pca_model = None

    def add_pseudoCross_edges(self,
                              pseudo_df,
                              k_pseudo: int = 10,
                              reduce_dim=None,
                              gft_init=True,
                              rand_size=0):
        """
        Add edges between pseudo spots and real spots according to pearson cor-
        relation or other similarity metrics.

        Parameters
        ----------
        pseudo_df : numpy.array
            The reconstructed gene expression matrix.
        k_pseudo : int, optional
            the number of edges between a pseudo spot to pseudo spots. The def-
            ault is 10.
        reduce_dim : int or None, optional
            the number of dimension after dimensionality reduction. If None, 
            PCA will not be implemented before calculating similarity. The def-
            ault is None.
        gft_init : bool, optional
            Reconstruct co-expression network and run GFT.
            The default is True.

        Returns
        -------
        None.

        """

        # Check whether currect pseudo spots or real_spots satisfying the requ-
        # ired mount when search the neighboors
        if pseudo_df.shape[0] < k_pseudo + 1:
            k_pseudo = pseudo_df.shape[0] - 1
        all_barcodes = pseudo_df.index.tolist()
        # if reduce_dim:
        #     if not gft_init:
        #         pseudo_df = self.gft(exp_df=pseudo_df,
        #                              reduce_dim=reduce_dim)
        #     else:
        #         pseudo_df = self.gft_init(exp_df=pseudo_df,
        #                                   reduce_dim=reduce_dim,
        #                                   rand_size=rand_size)
        if gft_init:
            pca_model = IncrementalPCA(n_components=reduce_dim)
            pca_model.fit(pseudo_df)
            pseudo_df = pca_model.transform(pseudo_df)
            self.pca_model = pca_model
        else:
            pseudo_df = self.pca_model.transform(pseudo_df)
        pseudo_df = sc.AnnData(pseudo_df)
        sc.pp.neighbors(pseudo_df, n_neighbors=k_pseudo, use_rep='X')
        corr_df = pd.DataFrame(pseudo_df.obsp['connectivities'].toarray(), index=all_barcodes, columns=all_barcodes)
        corr_df = corr_df.applymap(lambda x: 1 if x > 0 else 0)

        self.pseudo_dense_adj = corr_df.astype(int)

    def add_realCross_edges(self,
                            real_df,
                            k_real: int = 10,
                            reduce_dim=None,
                            gft_init=True):

        # Check whether currect pseudo spots or real_spots satisfying the requ-
        # ired mount when search the neighboors
        if real_df.shape[0] < k_real + 1:
            k_real = real_df.shape[0]
        all_barcodes = real_df.index.tolist()
        if gft_init:
            pca_model_real = PCA(n_components=reduce_dim)
            real_df = pca_model_real.fit_transform(real_df)
            self.pca_model_real = pca_model_real
        else:
            real_df = self.pca_model_real.transform(real_df)
        # if gft_init:
        #     real_df, fms = self.gft_init2(real_df, reduce_dim=reduce_dim)
        #     self.fms_real = fms
        # else:
        #     real_df = self.gft2(real_df, reduce_dim=reduce_dim, fms=self.fms_real)
        real_df = sc.AnnData(real_df)
        sc.pp.neighbors(real_df, n_neighbors=k_real, use_rep='X')
        corr_df = pd.DataFrame(real_df.obsp['connectivities'].toarray(), index=all_barcodes, columns=all_barcodes)
        corr_df = corr_df.applymap(lambda x: 1 if x > 0 else 0)

        self.real_dense_adj = corr_df.astype(int)

    def subgraph_by_pseudo(self, pseudo_nodes):
        adj = self.pseudo_dense_adj.loc[pseudo_nodes, pseudo_nodes].values
        return adj

    def subgraph_by_real(self, real_nodes):
        adj = self.real_dense_adj.loc[real_nodes, real_nodes].values
        return adj

    def gft_init(self, exp_df, reduce_dim=100, rand_size=0):
        exp_df, self.fms = apply_init_gft(exp_df, reduce_dim=reduce_dim, rand_size=rand_size)
        return exp_df

    def gft_init2(self, exp_df, reduce_dim=100):
        exp_df, fms = apply_init_gft(exp_df, reduce_dim=reduce_dim)
        return exp_df, fms

    def gft(self, exp_df, reduce_dim=100):
        exp_df = apply_gft(exp_df, self.fms, reduce_dim=reduce_dim)

        return exp_df

    def gft2(self, exp_df, fms, reduce_dim=100):
        exp_df = apply_gft(exp_df, fms, reduce_dim=reduce_dim)

        return exp_df

    def construct_subgraph(self,
                           real_df,
                           loc_df=None,
                           fms=None,
                           k_real=6,
                           reduce_dim=None,
                           gft_i=True):
        all_barcodes = real_df.index.tolist()
        # if reduce_dim and gft_i:
        #     # real_df, fms = self.gft_init2(real_df, reduce_dim=reduce_dim)
        #     real_df = IncrementalPCA(n_components=reduce_dim).fit_transform((real_df))
        adata = sc.AnnData(real_df)
        sc.pp.neighbors(adata, n_neighbors=k_real)
        corr_df = adata.obsp['connectivities']
        corr_df = corr_df.toarray()
        corr_df = pd.DataFrame(corr_df,
                               index=all_barcodes,
                               columns=all_barcodes)
        corr_df[corr_df > 0] = 1
        corr_df = corr_df.astype(int)

        if not (type(loc_df) == type(None)):
            # construct network based on locations
            # adata = sc.AnnData(loc_df)
            # sc.pp.neighbors(adata, n_neighbors=k_real * 3, use_rep='X')
            # corr_df2 = adata.obsp['connectivities']
            # corr_df2 = corr_df2.toarray()
            # corr_df2 = pd.DataFrame(corr_df2,
            #                         index=all_barcodes,
            #                         columns=all_barcodes)
            # corr_df2[corr_df2 > 0] = 1
            # corr_df2 = corr_df2.astype(int)
            # corr_df = corr_df + corr_df2
            # corr_df[corr_df > 1] = 1
            # corr_df = corr_df.astype(int)
            pass

        if gft_i:
            return corr_df.values, fms
        else:
            return corr_df.values
