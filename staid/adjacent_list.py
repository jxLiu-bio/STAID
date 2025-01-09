from multiprocessing.dummy import Pool

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


class subgraph_sampling():
    def __init__(self, real_df, pseudo_df, spa_coor_df, method="pearson",
                 reduce_dim=None):
        # Initialize 
        self.real_nodes = real_df.index.values.tolist()  # all real_nodes
        self.pseudo_nodes = pseudo_df.index.values.tolist()
        self.real_df = real_df
        self.pseudo_df = pseudo_df
        self.spa_coor_df = spa_coor_df
        self.method = method
        self.reduce_dim = reduce_dim
        pass

    def add_pseudoCross_edges(self,
                              k_pseudo: int = 10,
                              k_cross: int = 10,
                              reduce_dim=None):
        """
        Add edges between pseudo spots and real spots according to pearson cor-
        relation or other similarity metrics.

        Parameters
        ----------
        k_pseudo : int, optional
            the number of edges between a pseudo spot to pseudo spots. The def-
            ault is 10.
        k_cross : int, optional
            the number of edges between a real spot to pseudo spots. The defau-
            lt is 10.
        reduce_dim : int or None, optional
            the number of dimension after dimensionality reduction. If None, 
            PCA will not be implemented before calculating similarity. The def-
            ault is None.

        Returns
        -------
        None.

        """

        # Check whether currect pseudo spots or real_spots satisfying the requ-
        # ired mount when search the neighboors
        if self.pseudo_df.shape[0] < k_pseudo + 1:
            k_pseudo = self.pseudo_df.shape[0] - 1
        pseudo_df = self.pseudo_df
        # Whether reducing dimension is needed for the following calculation
        if self.reduce_dim and pseudo_df.shape[1] != reduce_dim:
            try:
                pseudo_df = pd.DataFrame(
                    normalize(self.pca.transform(pseudo_df.values)),
                    index=pseudo_df.index)
            except:
                reduce_dim = min(self.reduce_dim, pseudo_df.shape[1])
                self.reduce_dim = reduce_dim
                pca = PCA(n_components=reduce_dim, random_state=2023)
                tmp_spots = np.random.choice(pseudo_df.index.tolist(),
                                             min(3000,
                                                 pseudo_df.shape[0]))
                pca.fit(pseudo_df.loc[tmp_spots, :].values)
                pseudo_df = pd.DataFrame(
                    normalize(pca.transform(pseudo_df.values)),
                    index=pseudo_df.index)
                self.pca = pca

        # Calculate correlation
        # print("calculate similarities")
        # corr_df = pseudo_df.transpose().corr(method=self.method)
        corr_df = distance_matrix(pseudo_df, pseudo_df, p=2)
        corr_df = pd.DataFrame(corr_df)
        corr_df.index = pseudo_df.index
        corr_df.columns = pseudo_df.index

        # ******************Add edges***********************
        # Add edges between pseudo spots and pseudo spots
        # Note: shoule directed edges (Maybe)
        pseudo_edge_index = []
        pseudo_neighbor_df = pd.DataFrame(index=self.pseudo_df.index,
                                          columns=["Neighbors"])

        def _seek_neighbors_p(spot_name):
            tmp_series = corr_df.loc[spot_name, :]
            tmp_series = tmp_series.sort_values(ascending=True)
            tmp_thres = np.quantile(tmp_series, 0.25)
            tmp_series = tmp_series[tmp_series < tmp_thres]
            tmp_pseudo = min(k_pseudo - 1, tmp_series.size - 1)
            pseudo_edge_index.extend([spot_name, j] for j in \
                                     tmp_series.index[1:(tmp_pseudo + 1)])
            tmp_nodes = []
            tmp_nodes.extend(j for j in tmp_series.index[1:(tmp_pseudo + 1)])
            pseudo_neighbor_df.loc[spot_name, "Neighbors"] = tmp_nodes

        p = Pool(200)
        p.map(_seek_neighbors_p, corr_df.index)
        self.pseudo_edge_index = pseudo_edge_index
        self.pseudo_neighbor_df = pseudo_neighbor_df
        pass

    def add_spatial_edges(self, k_real: int = 6, spatial_names=['x', 'y'],
                          reduce_dim=None):
        '''
        Add spatial edges according to Euclidean distance. This step will enha-
        nce the spatial relationships. Here, to avoid the differences caused by
        the boundary. An edge between a pair of real spots exists if and only 
        if the two spots are neighoboring in space and similar in gene express-
        ion simultaneously.

        Parameters
        ----------
        k_real : int, optional
            the number of . The default is 6.
        reduce_dim : int or None, optional
            the number of dimension after dimensionality reduction. If None, 
            PCA will not be implemented before calculating similarity. The def-
            ault is None.

        Returns
        -------
        None.

        '''
        # Check the relationship between the mount of all real spots with the k
        # to ensure there are efficent spots to be considered
        if self.spa_coor_df.shape[0] < (k_real + 1):
            self.k_real = self.spa_coor_df.shape[0] - 1

        # Calculate the distance matrix using scipy.distance_matrix
        distance_df = pd.DataFrame(
            distance_matrix(self.spa_coor_df[spatial_names].values,
                            self.spa_coor_df[spatial_names].values),
            index=self.spa_coor_df.index,
            columns=self.spa_coor_df.index)

        # Calculate the correlation based on gene expression to avoid adding 
        # the edges whose ends are not similar
        if reduce_dim:
            real_df = self.pca.transform(self.real_df)
            real_df = normalize(real_df, norm='l2')
            real_df = pd.DataFrame(real_df,
                                   index=self.real_df.index)
        else:
            real_df = self.real_df

        real_corr_df = distance_matrix(real_df, real_df, p=2)
        real_corr_df = pd.DataFrame(real_corr_df,
                                    index=real_df.index,
                                    columns=real_df.index)
        # Record quartile information as the threshold values
        threshold_df = pd.DataFrame(0, index=self.real_df.index,
                                    columns=['threshold'])
        for i in real_corr_df.index:
            tmp_list = real_corr_df[i].values
            tmp_thres = np.quantile(tmp_list, 0.25)
            threshold_df.loc[i, 'threshold'] = tmp_thres
        # Define one list to store edge_list and a dataframe to store neighbors
        real_edge_index = []
        real_neighbor_df = pd.DataFrame(index=self.spa_coor_df.index,
                                        columns=['Neighbors'])

        # Find neighors as well as edges
        for i in distance_df.index:
            tmp_series = distance_df.loc[i, :]
            tmp_series = tmp_series.sort_values(ascending=True)
            real_edge_index.extend([i, j] for j in \
                                   tmp_series.index[1:(k_real + 1)] if \
                                   real_corr_df.loc[i, j] <= \
                                   threshold_df.loc[i, 'threshold'])
            tmp_nodes = []
            tmp_nodes.extend(j for j in tmp_series.index[1:(k_real + 1)])
            real_neighbor_df.loc[i, "Neighbors"] = tmp_nodes

        # Update the class object
        self.real_edge_index = real_edge_index
        self.real_neighbor_df = real_neighbor_df
        pass

    def bidirect_edges(self, edge_list):
        '''
        Extend the bidirectional edges for the following training.

        Parameters
        ----------
        edge_list : list
            edge list, consisted by nodes pair.

        Returns
        -------
        edge_list : list
            edge list which is bidirectional.

        '''
        edge_list_bidirect = edge_list.copy()
        edge_list_bidirect.extend([i, j] for [j, i] in edge_list)
        # Remove deplicate edges
        edge_list_tmp = [(i[0], i[1]) for i in edge_list_bidirect]
        del edge_list_bidirect
        edge_list_tmp = set(edge_list_tmp)
        edge_list_tmp2 = [[i[0], i[1]] for i in edge_list_tmp]

        return edge_list_tmp2

    def subgraph_by_real(self, real_nodes, rate=None):
        '''
        Divided a subgraph by choosed real nodes.

        Parameters
        ----------
        real_nodes : list
            choosed real spots/nodes
        rate : int|None, optional
            Whether need to control the mount of pseudo spots if necessary. The 
            default is None.

        Returns
        -------
        sampled_X : array
            the gene expression matrix, whose rows represent nodes/spots, colu-
            mns represent genes.
        nodes: list
            the rownames of sample_X, that are, obtained pseudo spots and 
            real spots. 
        pseudo_nodes : list
            the selected pseudo spots. They are selected due to similarity to
            current real spots.

        '''
        # ***************Obtain all sampled nodes******************
        # Create a list to store sampled nodes
        pseudo_nodes = []  # save adjacent pseudo spots to sampled
        # real spots

        # Add adjacent pseudo spots
        for i in real_nodes:
            tmp_list = self.pseuReal_neighbor_df.loc[i, :].values[0]
            pseudo_nodes.extend(j for j in tmp_list if j not in pseudo_nodes)

        # Add adjacent pseudo spots to the above adjacent spots
        pseudo_nodes_2 = []
        for i in pseudo_nodes:
            tmp_list = self.pseudo_neighbor_df.loc[i, :].values[0]
            pseudo_nodes_2.extend(j for j in tmp_list)
        pseudo_nodes = pseudo_nodes + pseudo_nodes_2

        # Remove deplicate pseudo spots
        pseudo_nodes = np.unique(pseudo_nodes).tolist()

        # Control the mount of pseudo spot if necessary
        if rate:
            if rate * len(real_nodes) < len(pseudo_nodes):
                pseudo_nodes = np.random.choice(pseudo_nodes,
                                                int(rate * len(real_nodes))).tolist()

        # *****Obtain node * feature matrix and corresponding edge_index *****
        all_nodes = pseudo_nodes + real_nodes
        all_nodes = pd.DataFrame(range(len(all_nodes)), index=all_nodes,
                                 columns=["num_index"])
        self.all_nodes_sub = all_nodes
        # node * feature matrix
        sampled_X = np.concatenate((self.pseudo_df.loc[pseudo_nodes, :].values,
                                    self.real_df.loc[real_nodes, :].values), axis=0)

        # The edge index contain three sources, that are, the edges between
        # pseudo spots and pseudo spots according to gene expression similarity,
        # the edges between real spots and pseudo spots according to gene expression,
        # the edges between pseudo real spots and real spots according to spatial
        # distance.

        # Firtly, obtain the edge index between inner pseudo spots
        edge_index_pseudo = []
        for i in pseudo_nodes:
            tmp_list = self.pseudo_neighbor_df.loc[i, :].values[0]
            # inital tmp_list may contains the pseudo list that don't belong
            # to pseudo_nodes, that are, the neighbors of current real spots
            tmp_list = np.intersect1d(tmp_list, pseudo_nodes).tolist()
            tmp_index = all_nodes.loc[i, "num_index"]  # index of currect spot
            edge_index_pseudo.extend([tmp_index, all_nodes.loc[j, "num_index"]]
                                     for j in tmp_list)
        edge_index_pseudo = self.bidirect_edges(edge_index_pseudo)

        # Secondly, obtain the edge index between pseudo spots and real spots
        edge_index_cross = []
        for i in real_nodes:
            tmp_list = self.pseuReal_neighbor_df.loc[i, :].values[0]
            tmp_list = np.intersect1d(tmp_list, real_nodes).tolist()
            tmp_index = all_nodes.loc[i, "num_index"]  # index of currect spot
            edge_index_cross.extend([tmp_index, all_nodes.loc[j, "num_index"]]
                                    for j in tmp_list)
        edge_index_cross = self.bidirect_edges(edge_index_cross)

        # Finally, obtain the edge index between inner real spots
        edge_index_real = []
        for i in real_nodes:
            tmp_list = self.real_neighbor_df.loc[i, :].values[0]
            tmp_list = np.intersect1d(tmp_list, real_nodes).tolist()
            tmp_index = all_nodes.loc[i, "num_index"]  # index of currect spot
            edge_index_real.extend([tmp_index, all_nodes.loc[j, "num_index"]]
                                   for j in tmp_list)
        edge_index_real = self.bidirect_edges(edge_index_real)

        # Merge all obtained edge index
        return sampled_X, edge_index_pseudo + edge_index_cross + \
                          edge_index_real, pseudo_nodes

    def subgraph_by_pseudo(self, pseudo_nodes, rate=None):
        '''
        Divided a subgraph by choosed pseudo nodes.

        Parameters
        ----------
        pseudo_nodes : list
            choosed pseudo spots/nodes
        rate : int|float|None, optional
            Whether need to downsample. The default is None.

        Returns
        -------
        sampled_X : array
            the gene expression matrix, whose rows represent nodes/spots, colu-
            mns represent genes
        nodes : list
            the rownames of sample_X, that are, obtained pseudo spots and 
            real spots.
        real_nodes : list
            the selected real spots. They are selected due to similarity to
            current pseudo spots.

        '''

        # *****Obtain node * feature matrix and corresponding edge_index *****
        all_nodes = pseudo_nodes
        all_nodes = pd.DataFrame(range(len(all_nodes)), index=all_nodes,
                                 columns=["num_index"])

        # node * feature matrix
        sampled_X = self.pseudo_df.loc[pseudo_nodes, :].values

        # The edge index contain three sources, that are, the edges between
        # pseudo spots and pseudo spots according to gene expression similarity,

        # Firtly, obtain the edge index between inner pseudo spots
        edge_index_pseudo = []

        def _obtain_index_pc(spot_name):
            tmp_list_p = self.pseudo_neighbor_df.loc[spot_name, :].values[0]
            tmp_list_p = np.intersect1d(tmp_list_p, pseudo_nodes).tolist()
            tmp_index = all_nodes.loc[spot_name, "num_index"]
            # number index of currect spot
            edge_index_pseudo.extend([tmp_index, all_nodes.loc[j, "num_index"]]
                                     for j in tmp_list_p)

        p = Pool(200)
        p.map(_obtain_index_pc, pseudo_nodes)
        # print("Add edges between inner pseudo spots as well as cross")
        edge_index_pseudo = self.bidirect_edges(edge_index_pseudo)
        return sampled_X, edge_index_pseudo, pseudo_nodes

    def convert_tensor(self):
        pass

    @property
    def whole_graph(self):
        # Obtaint the whole graph
        all_nodes = self.real_nodes
        all_nodes_df = pd.DataFrame(range(len(all_nodes)), index=all_nodes,
                                    columns=["num_index"])
        X = self.real_df.loc[all_nodes, :].values

        # Lastly, obtain the num_edge index between real spots
        edge_index_real = []
        for i in self.real_edge_index:
            edge_index_real.append([all_nodes_df.loc[i[0], "num_index"],
                                    all_nodes_df.loc[i[1], "num_index"]])
        edge_index_real = self.bidirect_edges(edge_index_real)

        return X, edge_index_real

    # def whole_graph(self):
    #     # Obtaint the whole graph
    #     all_real_nodes = self.real_nodes
    #     part_pseudo_nodes = []
    #     tmp_edge_index_list = []
    #     for i in all_real_nodes:
    #         tmp_list = self.pseuReal_neighbor_df.loc[i, "Neighbors"][:2]
    #         part_pseudo_nodes.extend(j for j in tmp_list)
    #         tmp_edge_index_list.extend([i, j] for j in tmp_list)
    #     part_pseudo_nodes = np.unique(part_pseudo_nodes).tolist()
    #     all_nodes =  part_pseudo_nodes + all_real_nodes
    #     all_nodes_df = pd.DataFrame(range(len(all_nodes)), index=all_nodes,
    #                              columns=["num_index"])
    #     X = np.concatenate((self.pseudo_df.loc[part_pseudo_nodes, :].values, 
    #                         self.real_df.loc[all_real_nodes, :].values), 
    #                         axis=0) # feature matirx

    #     # Lastly, obtain the num_edge index between real spots
    #     edge_index_real = []
    #     for i in self.real_edge_index:
    #         edge_index_real.append([all_nodes_df.loc[i[0], "num_index"], 
    #                                   all_nodes_df.loc[i[1], "num_index"]])
    #     edge_index_real = self.bidirect_edges(edge_index_real)

    #     # Secondly, obtain the num_edge index between real spots and pseudo spots,
    #     # that are, cross edges
    #     edge_index_cross = []
    #     for i in tmp_edge_index_list:
    #         edge_index_cross.append([all_nodes_df.loc[i[0], "num_index"], 
    #                                   all_nodes_df.loc[i[1], "num_index"]])
    #     edge_index_cross = self.bidirect_edges(edge_index_cross)

    #     # Return results
    #     # X_df = pd.DataFrame(X, index=all_nodes)
    #     # X_neigh_df = pd.DataFrame(edge_index_cross)
    #     # X_neigh_tmp_df = pd.DataFrame(tmp_edge_index_list)
    #     # X_df.to_csv("X_final.csv")
    #     # X_neigh_df.to_csv("neigh_final.csv")
    #     # X_neigh_tmp_df.to_csv("neigh_final_tmp.csv")
    #     return X, edge_index_real

    # def whole_graph(self):
    #     # Obtaint the whole graph

    #     all_nodes = self.pseudo_nodes + self.real_nodes
    #     all_nodes = pd.DataFrame(range(len(all_nodes)), index=all_nodes,
    #                              columns=["num_index"])
    #     X = np.concatenate((self.pseudo_df.values, 
    #                         self.real_df.values), axis=0) # feature matirx

    #     # Obtain edge index
    #     # Firstly, obtain the num_edge index of the edges between pseudo spots
    #     edge_index_pseudo = []
    #     for i in self.pseudo_edge_index:
    #         edge_index_pseudo.append([all_nodes.loc[i[0], "num_index"], 
    #                                   all_nodes.loc[i[1], "num_index"]])
    #     edge_index_pseudo = self.bidirect_edges(edge_index_pseudo)

    #     # Secondly, obtain the num_edge index between real spots and pseudo spots,
    #     # that are, cross edges
    #     edge_index_cross = []
    #     for i in self.pseuReal_edge_index:
    #         edge_index_cross.append([all_nodes.loc[i[0], "num_index"], 
    #                                   all_nodes.loc[i[1], "num_index"]])
    #     edge_index_cross = self.bidirect_edges(edge_index_cross)

    #     # Lastly, obtain the num_edge index between real spots
    #     edge_index_real = []
    #     for i in self.real_edge_index:
    #         edge_index_real.append([all_nodes.loc[i[0], "num_index"], 
    #                                   all_nodes.loc[i[1], "num_index"]])
    #     edge_index_real = self.bidirect_edges(edge_index_real)

    #     # Return results
    #     return X, edge_index_pseudo + edge_index_cross + edge_index_real
