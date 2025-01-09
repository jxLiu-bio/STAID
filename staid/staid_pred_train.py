import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ss
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from .generate_pseudo import generate_merge_iter, generate_merge_initial
from .network import MLP_autoencoder, MLP_pred
from .utils import remove_low_values
from .utils import fourier_modes_gene_network
from .enrichment_mia import enrichment_mia

warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.FloatTensor)


def sparsity_targets(targets):
    norm1_targets = torch.norm(targets, dim=1, p=1)
    norm2_targets = torch.norm(targets, dim=1, p=2)
    sp_targets = torch.mean(norm1_targets / norm2_targets)

    return sp_targets


def gat_pred_train_initial(real_df,
                           pseudo_df,
                           targets_df,
                           num_epoch=200,
                           batch_size=64,
                           lr=0.0005,
                           device='cpu',
                           hidden_dims=[512, 128, 64],
                           weight_decay=0,
                           dropout=0.1,
                           abs_size=None):
    # AE training
    total_df = pd.concat((pseudo_df, real_df), axis=0)
    total_df = total_df.values
    total_df = torch.from_numpy(total_df).float().to(device)

    real_tensor = torch.from_numpy(real_df.values).float().to(device)
    abs_tensor = torch.from_numpy(pseudo_df.values[:abs_size, :]).float().to(device)

    data_loader_ae = Data.DataLoader(total_df,
                                     batch_size=int(batch_size),
                                     shuffle=True)
    ae_dims = [1920, 1920]
    ae_model = MLP_autoencoder(dim_input=total_df.shape[1],
                               hidden_dims=ae_dims,
                               dropout=dropout).to(device=device)
    optimizer_ae = torch.optim.Adam(ae_model.parameters(),
                                    lr=lr,
                                    )
    ae_model.train()
    epoch_tqdm_ae = tqdm(range(100), desc="AE Epoch")
    for epoch in epoch_tqdm_ae:
        for step, sampled_X in enumerate(data_loader_ae):
            optimizer_ae.zero_grad()
            outputs = ae_model.forward(sampled_X)
            loss = F.mse_loss(sampled_X, outputs)
            epoch_tqdm_ae.set_postfix(loss=loss.item())
            loss.backward()
            optimizer_ae.step()
        real_emb = ae_model.encoder(real_tensor).mean(axis=0)
        abs_emb = ae_model.encoder(abs_tensor).mean(axis=0)
        loss = F.mse_loss(real_emb, abs_emb)
        loss.backward()
        optimizer_ae.step()

    # Select validation datasets
    validated_spots = np.random.choice(pseudo_df.index.tolist(),
                                       size=int(pseudo_df.shape[0] * 0.05),
                                       replace=False).tolist()
    training_spots = np.setdiff1d(pseudo_df.index.tolist(), validated_spots)

    # Obtain the validated datasets
    b_X_v = pseudo_df.loc[validated_spots, :].values
    b_X_v = torch.from_numpy(b_X_v)
    b_X_v = b_X_v.float().to(device)
    targets_v = targets_df.loc[validated_spots, :].values
    targets_v = torch.from_numpy(targets_v)
    targets_v = targets_v.float().to(device)

    # obtain the training datasets
    b_X_train = pseudo_df.loc[training_spots, :].values
    b_X_train = torch.from_numpy(b_X_train)
    b_X_train = b_X_train.float().to(device)
    targets_train = targets_df.loc[training_spots, :].values
    targets_train = torch.from_numpy(targets_train)
    targets_train = targets_train.float().to(device)

    # Define the training loader
    training_data = Data.TensorDataset(b_X_train, targets_train)
    pseudo_train_loader = Data.DataLoader(dataset=training_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)
    # Define the model and optimizer
    gat_model = MLP_pred(ae_dims[0],
                         targets_df.shape[1],
                         dim=hidden_dims,
                         dropout=dropout).to(device)
    optimizer = torch.optim.Adam(gat_model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    # Train the model
    epoch_tqdm = tqdm(range(num_epoch), desc="Epoch")
    valid_loss_list = []
    for epoch in epoch_tqdm:
        loss_list = []
        # if epoch == 50:
        #     optimizer = torch.optim.Adam(list(ae_model.parameters()) +
        #                                   list(gat_model.parameters()),
        #                                   lr=lr,
        #                                   weight_decay=weight_decay)
        for step, (b_x, b_y) in enumerate(pseudo_train_loader):
            optimizer.zero_grad()
            outputs = ae_model.encoder(b_x)
            outputs = gat_model.forward(outputs)
            sp_loss = sparsity_targets(outputs)
            loss = F.mse_loss(outputs, b_y) - 5e-5 * sp_loss
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        epoch_tqdm.set_postfix(loss=np.mean(loss_list))
        # validation
        with torch.no_grad():
            outputs = ae_model.encoder(b_X_v)
            outputs = gat_model.forward(outputs)
            loss = F.mse_loss(outputs, targets_v)
            valid_loss_list.append(loss.item())
        if len(valid_loss_list) > 70:
            if np.mean(valid_loss_list[-10:-5]) < 1.00 * np.mean(valid_loss_list[-5:]):
                break

    # Predict
    b_X_v = real_df.values
    b_X_v = torch.from_numpy(b_X_v)
    b_X_v = b_X_v.float().to(device)

    with torch.no_grad():
        outputs = ae_model.encoder(b_X_v)
        outputs = gat_model.forward(outputs)
        pred_df = pd.DataFrame(outputs.cpu().numpy(),
                               index=real_df.index,
                               columns=targets_df.columns)

    return pred_df, gat_model, ae_model


def gat_pred_train_iter(real_df,
                        pseudo_df,
                        gat_model,
                        ae_model,
                        targets_df,
                        num_epoch=100,
                        batch_size=32,
                        lr=0.0001,
                        device='cpu',
                        weight_decay=0):
    # Select validation datasets
    validated_spots = np.random.choice(pseudo_df.index.tolist(),
                                       size=int(pseudo_df.shape[0] * 0.05),
                                       replace=False).tolist()
    training_spots = np.setdiff1d(pseudo_df.index.tolist(), validated_spots)

    # Obtain the validated datasets
    b_X_v = pseudo_df.loc[validated_spots, :].values
    b_X_v = torch.from_numpy(b_X_v)
    b_X_v = b_X_v.float().to(device)
    targets_v = targets_df.loc[validated_spots, :].values
    targets_v = torch.from_numpy(targets_v)
    targets_v = targets_v.float().to(device)

    # obtain the training datasets
    b_X_train = pseudo_df.loc[training_spots, :].values
    b_X_train = torch.from_numpy(b_X_train)
    b_X_train = b_X_train.float().to(device)
    targets_train = targets_df.loc[training_spots, :].values
    targets_train = torch.from_numpy(targets_train)
    targets_train = targets_train.float().to(device)

    # Define the training loader
    training_data = Data.TensorDataset(b_X_train, targets_train)
    pseudo_train_loader = Data.DataLoader(dataset=training_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)
    optimizer = torch.optim.Adam(list(ae_model.parameters()) +
                                 list(gat_model.parameters()),
                                 lr=lr)

    # Train the model
    epoch_tqdm = tqdm(range(num_epoch), desc="Epoch")
    valid_loss_list = []
    for epoch in epoch_tqdm:
        loss_list = []
        for step, (b_x, b_y) in enumerate(pseudo_train_loader):
            optimizer.zero_grad()
            outputs = ae_model.encoder(b_x)
            outputs = gat_model.forward(outputs)
            loss = F.mse_loss(outputs, b_y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        epoch_tqdm.set_postfix(loss=np.mean(loss_list))
        # validation
        with torch.no_grad():
            outputs = ae_model.encoder(b_X_v)
            outputs = gat_model.forward(outputs)
            loss = F.mse_loss(outputs, targets_v) - 5e-5 * sparsity_targets(outputs)
            valid_loss_list.append(loss.item())
        if len(valid_loss_list) > 30:
            if np.mean(valid_loss_list[-10:-5]) < 1.00 * np.mean(valid_loss_list[-5:]):
                break

    # Predict
    b_X_v = real_df.values
    b_X_v = torch.from_numpy(b_X_v)
    b_X_v = b_X_v.float().to(device)

    with torch.no_grad():
        outputs = ae_model.encoder(b_X_v)
        outputs = gat_model.forward(outputs)
        pred_df = pd.DataFrame(outputs.cpu().numpy(),
                               index=real_df.index,
                               columns=targets_df.columns)

    return pred_df, gat_model, ae_model


def run_deconvolution(spa_adata,
                      sc_adata,
                      anno_key='cell_type',
                      device='auto',
                      error_cutoff=0.01,
                      lr=0.0005,
                      num_pseudo=5000,
                      num_epoch=200,
                      num_iter=5,
                      batch_size=512,
                      min_cells=1,
                      max_cells=20,
                      k=(8, 15),
                      remove_platform='auto',
                      random_spot_rate=0.3,
                      hidden_dims=[512, 128, 64],
                      library_size=1e4,
                      dropout=0.1):
    """
    Infer cell type composition by graph attention neural networks. This 
    function contains two parts, that are, initial prediction using pseudo
    spots from enrichment analysis and interation. that are, generating pseudo
    spots and predict cell type composition using GAT.
    reference: https://arxiv.org/abs/2105.14491.
    
    Parameters
    ----------
    spa_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are essential, that are, count matrix should be
        found by spa_adata.X and spatial coordinates which could be extract by spa_adata.obs[spatial_names].
    sc_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are essential, that are, count matrix should be
        found by sc_adata and cell type annotation.
    spatial_key : list or tuple or str
        List or tuple indicates the spatial columns in spa_adata.obs. The
        coordinate information can be found at spa_adata.obs[spatial_names]. If
        spatial key belongs to str, The spatial information can be found at
        spa_adata.obsm['spatial_key'].
        The default is ['tissue_in_row', 'tissue_in_col']. 
    anno_key : string
        The column name indicate cell type annotation information in sc_adata.
        obs. That is, the cell type annotation information could be found by
        sc_adata.obs[anno_name]. The default is 'cell_type'.
    device : str, optional
        The device for training, if 'auto', the model will select device 
        automatically. It could be 'cup', 'gpu' or 'cuda:0' etc. The default
        is 'auto'. 
    iter_cutoff : float, optional
        If error between two contiguous prediction is less the iter_cutoff, 
        the iteration will be stopped beforehand. The default is 0.01.
    lr : float, optional
        Learn rate in training process. The default is 0.001.
    num_pseudo : int, optional
        The number of pseudo spots to be generated. The default is 5000.
    num_epoch : int optional
        The epoch in initial training. The default is 100.
    num_whole_train : int, optional
        The number of epochs in training the whole graph. The default is None.
    num_iter :int, optional
        The number of iteration processes. The default is 5.
    batch_size : int, optional
        The size of batch in subgraph training. The default is 64.
    min_cells : int, optional
        The minimum cells that a pseudo spot contains. The default is 1.
    max_cells :int, optional
        The maximum cells that a pseudo spot contains. The default is 20.
    k : tuple or list, optional
        Define the numbers of genes in real-real, pseudo-pseudo
        The default is (8, 15).
    remove_platform : bool | 'auto', optional
        Whether you need to remove the platform effects. This step is implemented
        by scanpy.pp.combat(). More details could be found in Johnson, Li & 
        Rabinovic (2007), Adjusting batch effects in microarray expression data
        using empirical Bayes methods, Biostatistics.
        The default is 'auto'.
    random_spot_rate: tuple, optional
        To control the ratio of random pseudo spots. 
        The default is 0.2.
    dropout: float , optional
        dropout controls the dropout rate of ARMAConv layer. 
        The default is 0.1.
    Returns
    -------
    prediction : DateFrame
        The prediction of cell type composition of real spots.

    """
    # ************Generate pseudo spots by enrichment**************
    # Enrichment_analysis
    # Process scRNA-seq to boost
    # correct_adata(sc_adata, spa_adata)
    if ss.issparse(sc_adata.X):
        sc_adata_total = sc_adata.copy()
        sc_adata_total.X = sc_adata_total.X.toarray()
    else:
        sc_adata_total = sc_adata
    if ss.issparse(spa_adata.X):
        spa_adata.X = spa_adata.X.toarray()
    else:
        spa_adata = spa_adata
    shared_genes = np.intersect1d(sc_adata.var_names,
                                  spa_adata.var_names).tolist()
    sc_adata = sc_adata[:, shared_genes]
    spa_adata = spa_adata[:, shared_genes]

    from staid.utils import simplify_refer, extract_hvgs

    # select device for training
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    if device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            raise KeyError("Can not find gpu device or cuda is not available.")

    # remove platform
    if remove_platform:
        remove_p = True
    else:
        remove_p = False
    wd = 0
    # check parameters
    if random_spot_rate > 1 or random_spot_rate < 0:
        raise ValueError(f"random_spot_rate should in [0, 1), but got \
                         {random_spot_rate}")

    # simplify
    cell_per_ct = min(int(sc_adata_total.shape[0] / np.unique(sc_adata.obs[anno_key]).size), 500)
    sc_adata = simplify_refer(sc_adata=sc_adata_total,
                              anno_key=anno_key,
                              cell_per_ct=cell_per_ct)

    ct_spot_enrich_df, marker_genes, domains = enrichment_mia(spa_adata,
                                                              sc_adata,
                                                              anno_name=anno_key,
                                                              resolution=1)
    print(f'The number of marker genes: {len(marker_genes)}')
    if len(marker_genes) <= 2000:
        hvgs = extract_hvgs(sc_adata, n_genes=2000)
        marker_genes = np.union1d(marker_genes, hvgs)
        sc_adata = sc_adata[:, marker_genes]
        sc_adata_total = sc_adata_total[:, marker_genes]
        spa_adata = spa_adata[:, marker_genes]
    elif len(marker_genes) >= 5000:
        spa_adata = spa_adata[:, marker_genes]
        extract_genes = extract_hvgs(sc_adata, n_genes=5000)
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names
    else:
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names

    print("*************** Iteration: \t1 ***************")
    # Construct gene network and obtain the fourier modes
    eigen_vector_gene, eigen_value_gene, mean_deg = fourier_modes_gene_network(sc_adata,
                                                                               gene_list=marker_genes)
    c = 2 / (1e-4 + mean_deg)
    # Generate pseudo spots and their cell type compositions and merge them 
    # with real spots
    pseudo_num_rate = int(np.ceil(num_pseudo / spa_adata.shape[0]))
    spa_X, pseudo_X, pseudo_df_composition, abs_size = generate_merge_initial(sc_adata,
                                                                              spa_adata,
                                                                              marker_genes=marker_genes,
                                                                              enrichment_score_df=ct_spot_enrich_df,
                                                                              anno_name=anno_key,
                                                                              min_cells=min_cells,
                                                                              max_cells=max_cells,
                                                                              pseudo_num_rate=pseudo_num_rate,
                                                                              abs_relative_rate=(
                                                                                  (1 - random_spot_rate) * 0.65,
                                                                                  (1 - random_spot_rate) * 0.35,
                                                                                  random_spot_rate),
                                                                              remove_platform=remove_p,
                                                                              domains=domains,
                                                                              library_size=library_size)
    print(pseudo_X.shape)

    # ************* initial prediction *********************
    # Adjust data type
    # GFT
    spa_X = gft_using_gene_network(exp_mtx=spa_X,
                                   eigen_vectors=eigen_vector_gene,
                                   eigen_values=eigen_value_gene,
                                   c=c)
    pseudo_X = gft_using_gene_network(exp_mtx=pseudo_X,
                                      eigen_vectors=eigen_vector_gene,
                                      eigen_values=eigen_value_gene,
                                      c=c)
    spa_X = pd.DataFrame(spa_X,
                         index=spa_adata.obs_names)
    pseudo_X = pd.DataFrame(pseudo_X,
                            index=pseudo_df_composition.columns)
    # Prediction using pseudo_spots from enrichment_analysis.
    prediction, gat_model, ae_model = gat_pred_train_initial(spa_X,
                                                             pseudo_X,
                                                             pseudo_df_composition.transpose(),
                                                             num_epoch=num_epoch,
                                                             batch_size=batch_size,
                                                             lr=lr,
                                                             device=device,
                                                             hidden_dims=hidden_dims,
                                                             dropout=dropout,
                                                             weight_decay=wd,
                                                             abs_size=abs_size
                                                             )
    prediction = remove_low_values(prediction)
    prediction_pre = prediction.copy()

    gd_df = spa_adata.obsm['cell_type_proportion']
    common_cts = np.intersect1d(gd_df.columns, prediction.columns)
    gd_df = gd_df.loc[prediction.index, common_cts]
    mae = mean_absolute_error(gd_df, prediction.loc[:, common_cts])
    print(mae)
    # *****************iteration processes***********************
    # Ensure pseudo num rate.
    if num_iter < 0 or not isinstance(num_iter, int):
        raise ValueError('num_iter should be non-negative integer')
    # random_spot_rate = random_spot_rate / 2
    # Remove platform effects
    if not remove_platform:
        remove_p = False
    else:
        remove_p = True
    for i in range(num_iter - 1):
        # adjust learning rate in iterations
        if i % 5 == 1:
            lr = lr / 2
        perturbation = 0.5
        print(f"*************** Iteration:\t{i + 2} ***************")
        sc_adata = simplify_refer(sc_adata=sc_adata_total.copy(),
                                  anno_key=anno_key,
                                  cell_per_ct=cell_per_ct)

        spa_X, pseudo_X, pseudo_df_composition, abs_size = generate_merge_iter(sc_adata=sc_adata,
                                                                               spa_adata=spa_adata,
                                                                               pre_deconvo=prediction,
                                                                               marker_genes=marker_genes,
                                                                               anno_name=anno_key,
                                                                               min_cells=min_cells,
                                                                               max_cells=max_cells,
                                                                               abs_relative_rate=(
                                                                                   (1 - random_spot_rate) * 0.65,
                                                                                   (1 - random_spot_rate) * 0.35,
                                                                                   random_spot_rate),
                                                                               pseudo_num_rate_iter=pseudo_num_rate,
                                                                               remove_platform=remove_p,
                                                                               perturbation=perturbation,
                                                                               library_size=library_size)
        spa_X = gft_using_gene_network(exp_mtx=spa_X,
                                       eigen_vectors=eigen_vector_gene,
                                       eigen_values=eigen_value_gene,
                                       c=c)
        pseudo_X = gft_using_gene_network(exp_mtx=pseudo_X,
                                          eigen_vectors=eigen_vector_gene,
                                          eigen_values=eigen_value_gene,
                                          c=c)
        spa_X = pd.DataFrame(spa_X,
                             index=spa_adata.obs_names)
        pseudo_X = pd.DataFrame(pseudo_X,
                                index=pseudo_df_composition.columns)
        print(pseudo_X.shape)
        # Prediction using pseudo spots from previous deconvolution results
        prediction, gat_model, ae_model = gat_pred_train_iter(spa_X,
                                                              pseudo_X,
                                                              gat_model,
                                                              ae_model,
                                                              pseudo_df_composition.transpose(),
                                                              num_epoch=num_epoch,
                                                              batch_size=batch_size,
                                                              lr=lr,
                                                              device=device,
                                                              weight_decay=wd)

        # Check iteration error. If the MAE between two iterations is less than
        # setting threshold values, exit.
        prediction = remove_low_values(prediction, cutoff=0.005)
        iter_error = mean_absolute_error(prediction, prediction_pre)
        prediction_pre = prediction.copy()
        gd_df = spa_adata.obsm['cell_type_proportion']
        common_cts = np.intersect1d(gd_df.columns, prediction.columns)
        gd_df = gd_df.loc[prediction.index, common_cts]
        mae = mean_absolute_error(gd_df, prediction.loc[:, common_cts])
        print(mae)
        if iter_error < error_cutoff:
            print("Iteration Error ", iter_error, " < ", "Setting Error ",
                  error_cutoff)
            print("Exit!")
            break

    spa_adata.obsm['deconvolution'] = prediction

    return prediction


def gat_predict_t(spa_adata,
                  sc_adata,
                  anno_key='cell_type',
                  device='auto',
                  error_cutoff=0.02,
                  spatial_info='spatial',
                  lr=0.0005,
                  num_pseudo=5000,
                  num_epoch=200,
                  num_iter=5,
                  batch_size=512,
                  min_cells=1,
                  max_cells=20,
                  k=(8, 15),
                  remove_platform='auto',
                  random_spot_rate=0.5,
                  hidden_dims=[512, 128, 64],
                  library_size=1e3,
                  dropout=0.1):
    """
    Infer cell type composition by graph attention neural networks. This
    function contains two parts, that are, initial prediction using pseudo
    spots from enrichment analysis and interation. that are, generating pseudo
    spots and predict cell type composition using GAT.
    reference: https://arxiv.org/abs/2105.14491.

    Parameters
    ----------
    spa_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by spa_adata.X and spat-
        ial coordinates which could be extract by spa_adata.obs[spatial_names].
    sc_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by sc_adata and cell ty-
        pe annotation.
    spatial_key : list or tupple or str
        List or tupple indicates the spatial columns in spa_adata.obs. The
        coordinate information can be found at spa_adat.obs[spatial_names]. If
        spatial key belongs to str, The spatial information can be found at
        spa_adata.obsm['spatial_key'].
        The default is ['tissue_in_row', 'tissue_in_col'].
    anno_key : string
        The column name indicate cell type annotation information in sc_adata.
        obs. That is, the cell type annotation information could be found by
        sc_adata.obs[anno_name]. The default is 'cell_type'.
    device : str, optional
        The device for training, if 'auto', the model will select device
        automatically. It could be 'cup', 'gpu' or 'cuda:0' etc. The default
        is 'auto'.
    iter_cutoff : float, optional
        If error between two contiguous prediction is less the iter_cutoff,
        the iteration will be stoped beforehand. The default is 0.01.
    lr : float, optional
        Learn rate in training process. The default is 0.001.
    num_pseudo : int, optional
        The number of pseudo spots to be generated. The default is 5000.
    num_epoch : int optional
        The epoch in initial training. The default is 100.
    num_whole_train : int, optional
        The number of epochs in training the whole graph. The default is None.
    num_iter :int, optional
        The number of iteration processes. The default is 5.
    batch_size : int, optional
        The size of batch in subgraph training. The default is 64.
    min_cells : int, optional
        The minimum cells that a pseudo spot contains. The default is 1.
    max_cells :int, optional
        The maximum cells that a pseudo spot contains. The default is 20.
    k : tuple or list, optional
        Define the numbers of genes in real-real, pseudo-pseudo
        The default is (8, 15).
    remove_platform : bool | 'auto', optional
        Whether you need to remove the platform effects. This step is implemented
        by scanpy.pp.combat(). More details could be found in Johnson, Li &
        Rabinovic (2007), Adjusting batch effects in microarray expression data
        using empirical Bayes methods, Biostatistics.
        The default is 'auto'.
    random_spot_rate: tuple, optional
        To control the ratio of random pseudo spots.
        The default is 0.2.
    dropout: float , optional
        dropout controls the dropout rate of ARMAConv layer.
        The default is 0.1.
    Returns
    -------
    prediction : DateFrame
        The prediction of cell type composition of real spots.

    """
    # ************Generate pseudo spots by enrichment**************
    # Enrichment_analysis
    # Process scRNA-seq to boost
    # correct_adata(sc_adata, spa_adata)
    shared_genes = np.intersect1d(sc_adata.var_names,
                                  spa_adata.var_names).tolist()
    sc_adata = sc_adata[:, shared_genes]
    spa_adata = spa_adata[:, shared_genes]
    if ss.issparse(sc_adata.X):
        sc_adata_total = sc.AnnData(sc_adata.X.todense(),
                                    obs=sc_adata.obs,
                                    var=sc_adata.var)
    else:
        sc_adata_total = sc_adata
    if ss.issparse(spa_adata.X):
        spa_adata = sc.AnnData(spa_adata.X.todense(),
                               obs=spa_adata.obs,
                               var=spa_adata.var,
                               obsm=spa_adata.obsm)
    else:
        spa_adata = spa_adata
    from staid.utils import simplify_refer, extract_hvgs

    # select device for training
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    if device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            raise KeyError("Can not find gpu device or cuda is not available.")

    # remove platform
    if remove_platform == True:
        remove_p = True
    else:
        remove_p = False
    wd = 0
    # check parameters
    if random_spot_rate > 1 or random_spot_rate < 0:
        raise ValueError(f"random_spot_rate should in [0, 1), but got \
                         {random_spot_rate}")

    # simplify
    sc_adata = simplify_refer(sc_adata=sc_adata_total,
                              anno_key=anno_key,
                              cell_per_ct=500,
                              tmp_num=1)
    ct_spot_enrich_df, marker_genes, domains = enrichment_mia(spa_adata,
                                                              sc_adata,
                                                              anno_name=anno_key,
                                                              resolution=1)
    print(f'The number of marker genes: {len(marker_genes)}')
    if len(marker_genes) <= 2000:
        hvgs = extract_hvgs(sc_adata, n_genes=2000)
        marker_genes = np.union1d(marker_genes, hvgs)
        sc_adata = sc_adata[:, marker_genes]
        sc_adata_total = sc_adata_total[:, marker_genes]
        spa_adata = spa_adata[:, marker_genes]
    elif len(marker_genes) >= 5000:
        spa_adata = spa_adata[:, marker_genes]
        extract_genes = extract_hvgs(sc_adata, n_genes=5000)
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names
    else:
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names

    print("*************** Iteration: \t1 ***************")
    # Generate pseudo spots and their cell type compositions and merge them
    # with real spots
    pseudo_num_rate = int(np.ceil(num_pseudo / spa_adata.shape[0]))
    spa_X, pseudo_X, pseudo_df_composition, rand_size = \
        generate_merge_initial(sc_adata,
                               spa_adata,
                               marker_genes=marker_genes,
                               enrichment_score_df=ct_spot_enrich_df,
                               anno_name=anno_key,
                               min_cells=min_cells,
                               max_cells=max_cells,
                               pseudo_num_rate=pseudo_num_rate,
                               abs_relative_rate=((1 - random_spot_rate) * 0.65,
                                                  (1 - random_spot_rate) * 0.35,
                                                  random_spot_rate),
                               remove_platform=remove_p,
                               domains=domains,
                               library_size=library_size)
    print(pseudo_X.shape)
    # low pass filter
    # spa_X, filter_mtx = low_pass_filter_init(spa_X, c=0.001)
    # pseudo_X = low_pass_filter(pseudo_X, filter_mtx)

    # ************* initial prediction *********************
    # Adjust data type
    if spatial_info == None:
        loc_df = None
    elif isinstance(spatial_info, str):
        loc_df = pd.DataFrame(spa_adata.obsm[spatial_info],
                              index=spa_adata.obs_names)
    elif set(spatial_info) < set(spa_adata.obs.columns):
        loc_df = spa_adata.obs.loc[:, spatial_info].copy()

    else:
        raise KeyError(f"Can not find spatial information by {spatial_info}")
    spa_X = pd.DataFrame(spa_X,
                         index=spa_adata.obs_names,
                         columns=marker_genes)
    pseudo_X = pd.DataFrame(pseudo_X,
                            index=pseudo_df_composition.columns,
                            columns=marker_genes)
    # Prediction using pseudo_spots from enrichment_analysis.
    prediction, gat_model, ae_model = gat_pred_train_initial(spa_X,
                                                             pseudo_X,
                                                             pseudo_df_composition.transpose(),
                                                             num_epoch=num_epoch,
                                                             batch_size=batch_size,
                                                             lr=lr,
                                                             device=device,
                                                             hidden_dims=hidden_dims,
                                                             dropout=dropout,
                                                             weight_decay=wd)
    prediction = remove_low_values(prediction)
    prediction_pre = prediction.copy()

    # ***************** iteration processes ***********************
    # Ensure pseudo num rate.
    if num_iter < 0 or not isinstance(num_iter, int):
        raise ValueError('num_iter should be non-negative integer')
    if remove_platform == False:
        remove_p = False
    else:
        remove_p = True
    for i in range(num_iter - 1):
        if i % 5 == 1:
            lr = lr / 2
        perturbation = 0.5 * (num_iter - i) / num_iter
        print(f"*************** Iteration:\t{i + 2} ***************")
        sc_adata = simplify_refer(sc_adata=sc_adata_total.copy(),
                                  anno_key=anno_key,
                                  cell_per_ct=500,
                                  tmp_num=1)
        spa_X, pseudo_X, pseudo_df_composition, rand_size = \
            generate_merge_iter(sc_adata=sc_adata, spa_adata=spa_adata, pre_deconvolution=prediction,
                                marker_genes=marker_genes, anno_name=anno_key, min_cells=min_cells, max_cells=max_cells,
                                abs_relative_rate=((1 - random_spot_rate) * 0.65,
                                                   (1 - random_spot_rate) * 0.35,
                                                   random_spot_rate), pseudo_num_rate_iter=pseudo_num_rate,
                                remove_platform=remove_p, perturbation=perturbation, library_size=library_size)
        # pseudo_X = pseudo_X = low_pass_filter(pseudo_X, filter_mtx)
        spa_X = pd.DataFrame(spa_X, index=spa_adata.obs_names,
                             columns=marker_genes)
        pseudo_X = pd.DataFrame(pseudo_X,
                                index=pseudo_df_composition.columns,
                                columns=marker_genes)
        print(pseudo_X.shape)
        # Prediction using pseudo spots from previous deconvolution results
        prediction, gat_model, ae_model = gat_pred_train_iter(spa_X,
                                                              pseudo_X,
                                                              gat_model,
                                                              ae_model,
                                                              pseudo_df_composition.transpose(),
                                                              num_epoch=num_epoch,
                                                              k_real=k[0],
                                                              k_pseudo=k[1],
                                                              batch_size=batch_size,
                                                              lr=lr,
                                                              device=device,
                                                              hidden_dims=hidden_dims,
                                                              weight_decay=wd,
                                                              rand_size=rand_size)

        # Check iteration error. If the MAE between two iterations is less than
        # setting threshold values, exit.
        prediction = remove_low_values(prediction, cutoff=0.005)
        iter_error = mean_absolute_error(prediction, prediction_pre)
        prediction_pre = prediction.copy()
        if iter_error < error_cutoff:
            print("Iteration Error ", iter_error, " < ", "Setting Error ",
                  error_cutoff)
            print("Exit!")
            break

    spa_adata.obsm['deconvolution'] = prediction

    return prediction


def obtain_pseudo_spots(spa_adata,
                        sc_adata,
                        anno_key='cell_type',
                        device='auto',
                        error_cutoff=0.02,
                        spatial_info='spatial',
                        lr=0.0005,
                        num_pseudo=5000,
                        num_epoch=200,
                        num_iter=5,
                        batch_size=512,
                        min_cells=1,
                        max_cells=20,
                        k=(8, 15),
                        remove_platform='auto',
                        random_spot_rate=0.5,
                        hidden_dims=[512, 128, 64],
                        library_size=1e3,
                        dropout=0.1):
    """
    Infer cell type composition by graph attention neural networks. This
    function contains two parts, that are, initial prediction using pseudo
    spots from enrichment analysis and interation. that are, generating pseudo
    spots and predict cell type composition using GAT.
    reference: https://arxiv.org/abs/2105.14491.

    Parameters
    ----------
    spa_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by spa_adata.X and spat-
        ial coordinates which could be extract by spa_adata.obs[spatial_names].
    sc_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by sc_adata and cell ty-
        pe annotation.
    spatial_key : list or tupple or str
        List or tupple indicates the spatial columns in spa_adata.obs. The
        coordinate information can be found at spa_adat.obs[spatial_names]. If
        spatial key belongs to str, The spatial information can be found at
        spa_adata.obsm['spatial_key'].
        The default is ['tissue_in_row', 'tissue_in_col'].
    anno_key : string
        The column name indicate cell type annotation information in sc_adata.
        obs. That is, the cell type annotation information could be found by
        sc_adata.obs[anno_name]. The default is 'cell_type'.
    device : str, optional
        The device for training, if 'auto', the model will select device
        automatically. It could be 'cup', 'gpu' or 'cuda:0' etc. The default
        is 'auto'.
    iter_cutoff : float, optional
        If error between two contiguous prediction is less the iter_cutoff,
        the iteration will be stoped beforehand. The default is 0.01.
    lr : float, optional
        Learn rate in training process. The default is 0.001.
    num_pseudo : int, optional
        The number of pseudo spots to be generated. The default is 5000.
    num_epoch : int optional
        The epoch in initial training. The default is 100.
    num_whole_train : int, optional
        The number of epochs in training the whole graph. The default is None.
    num_iter :int, optional
        The number of iteration processes. The default is 5.
    batch_size : int, optional
        The size of batch in subgraph training. The default is 64.
    min_cells : int, optional
        The minimum cells that a pseudo spot contains. The default is 1.
    max_cells :int, optional
        The maximum cells that a pseudo spot contains. The default is 20.
    k : tuple or list, optional
        Define the numbers of genes in real-real, pseudo-pseudo
        The default is (8, 15).
    remove_platform : bool | 'auto', optional
        Whether you need to remove the platform effects. This step is implemented
        by scanpy.pp.combat(). More details could be found in Johnson, Li &
        Rabinovic (2007), Adjusting batch effects in microarray expression data
        using empirical Bayes methods, Biostatistics.
        The default is 'auto'.
    random_spot_rate: tuple, optional
        To control the ratio of random pseudo spots.
        The default is 0.2.
    dropout: float , optional
        dropout controls the dropout rate of ARMAConv layer.
        The default is 0.1.
    Returns
    -------
    prediction : DateFrame
        The prediction of cell type composition of real spots.

    """
    # ************Generate pseudo spots by enrichment**************
    # Enrichment_analysis
    # Process scRNA-seq to boost
    # correct_adata(sc_adata, spa_adata)
    shared_genes = np.intersect1d(sc_adata.var_names,
                                  spa_adata.var_names).tolist()
    sc_adata = sc_adata[:, shared_genes]
    spa_adata = spa_adata[:, shared_genes]
    if ss.issparse(sc_adata.X):
        sc_adata_total = sc.AnnData(sc_adata.X.todense(),
                                    obs=sc_adata.obs,
                                    var=sc_adata.var)
    else:
        sc_adata_total = sc_adata
    if ss.issparse(spa_adata.X):
        spa_adata = sc.AnnData(spa_adata.X.todense(),
                               obs=spa_adata.obs,
                               var=spa_adata.var,
                               obsm=spa_adata.obsm)
    else:
        spa_adata = spa_adata
    from staid.utils import simplify_refer, extract_hvgs

    # select device for training
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    if device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            raise KeyError("Can not find gpu device or cuda is not available.")

    # remove platform
    if remove_platform == True:
        remove_p = True
    else:
        remove_p = False
    wd = 0
    # check parameters
    if random_spot_rate > 1 or random_spot_rate < 0:
        raise ValueError(f"random_spot_rate should in [0, 1), but got \
                         {random_spot_rate}")

    # simplify
    cell_per_ct = min(int(sc_adata_total.shape[0] / np.unique(sc_adata.obs[anno_key]).size),
                      500)
    sc_adata = simplify_refer(sc_adata=sc_adata_total,
                              anno_key=anno_key,
                              cell_per_ct=cell_per_ct,
                              tmp_num=1)
    ct_spot_enrich_df, marker_genes, domains = enrichment_mia(spa_adata,
                                                              sc_adata,
                                                              anno_name=anno_key,
                                                              resolution=1)
    print(f'The number of marker genes: {len(marker_genes)}')
    if len(marker_genes) <= 2000:
        hvgs = extract_hvgs(sc_adata, n_genes=2000)
        marker_genes = np.union1d(marker_genes, hvgs)
        sc_adata = sc_adata[:, marker_genes]
        sc_adata_total = sc_adata_total[:, marker_genes]
        spa_adata = spa_adata[:, marker_genes]
    elif len(marker_genes) >= 5000:
        spa_adata = spa_adata[:, marker_genes]
        extract_genes = extract_hvgs(sc_adata, n_genes=5000)
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names
    else:
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names

    print("*************** Iteration: \t1 ***************")
    # Generate pseudo spots and their cell type compositions and merge them
    # with real spots
    pseudo_num_rate = int(np.ceil(num_pseudo / spa_adata.shape[0]))
    spa_X, pseudo_X, pseudo_df_composition, rand_size = \
        generate_merge_initial(sc_adata,
                               spa_adata,
                               marker_genes=marker_genes,
                               enrichment_score_df=ct_spot_enrich_df,
                               anno_name=anno_key,
                               min_cells=min_cells,
                               max_cells=max_cells,
                               pseudo_num_rate=pseudo_num_rate,
                               abs_relative_rate=((1 - random_spot_rate) * 0.65,
                                                  (1 - random_spot_rate) * 0.35,
                                                  random_spot_rate),
                               remove_platform=remove_p,
                               domains=domains,
                               library_size=library_size)
    print(pseudo_X.shape)
    # low pass filter

    # ************* initial prediction *********************
    # Adjust data type
    spa_X = pd.DataFrame(spa_X,
                         index=spa_adata.obs_names,
                         columns=marker_genes)
    pseudo_X = pd.DataFrame(pseudo_X,
                            index=pseudo_df_composition.columns,
                            columns=marker_genes)
    # Prediction using pseudo_spots from enrichment_analysis.
    prediction, gat_model, ae_model = gat_pred_train_initial(spa_X,
                                                             pseudo_X,
                                                             pseudo_df_composition.transpose(),
                                                             num_epoch=num_epoch,
                                                             batch_size=batch_size,
                                                             lr=lr,
                                                             device=device,
                                                             hidden_dims=hidden_dims,
                                                             dropout=dropout,
                                                             weight_decay=wd)
    prediction = remove_low_values(prediction)
    prediction_pre = prediction.copy()

    # ***************** iteration processes ***********************
    # Ensure pseudo num rate.
    if num_iter < 0 or not isinstance(num_iter, int):
        raise ValueError('num_iter should be non-negative integer')
    if remove_platform == False:
        remove_p = False
    else:
        remove_p = True
    for i in range(num_iter - 1):
        if i % 5 == 1:
            lr = lr / 2
        perturbation = max(0.5 * (num_iter - i) / num_iter, 0.25)
        print(f"*************** Iteration:\t{i + 2} ***************")
        sc_adata = simplify_refer(sc_adata=sc_adata_total.copy(),
                                  anno_key=anno_key,
                                  cell_per_ct=cell_per_ct,
                                  tmp_num=1)
        spa_X, pseudo_X, pseudo_df_composition, rand_size = \
            generate_merge_iter(sc_adata=sc_adata, spa_adata=spa_adata, pre_deconvolution=prediction,
                                marker_genes=marker_genes, anno_name=anno_key, min_cells=min_cells, max_cells=max_cells,
                                abs_relative_rate=((1 - random_spot_rate) * 0.65,
                                                   (1 - random_spot_rate) * 0.35,
                                                   random_spot_rate), pseudo_num_rate_iter=pseudo_num_rate,
                                remove_platform=remove_p, perturbation=perturbation, library_size=library_size)
        # pseudo_X = pseudo_X = low_pass_filter(pseudo_X, filter_mtx)
        spa_X = pd.DataFrame(spa_X, index=spa_adata.obs_names,
                             columns=marker_genes)
        pseudo_X = pd.DataFrame(pseudo_X,
                                index=pseudo_df_composition.columns,
                                columns=marker_genes)
        print(pseudo_X.shape)
        # Prediction using pseudo spots from previous deconvolution results
        prediction, gat_model, ae_model = gat_pred_train_iter(spa_X,
                                                              pseudo_X,
                                                              gat_model,
                                                              ae_model,
                                                              pseudo_df_composition.transpose(),
                                                              num_epoch=num_epoch,
                                                              k_real=k[0],
                                                              k_pseudo=k[1],
                                                              batch_size=batch_size,
                                                              lr=lr,
                                                              device=device,
                                                              hidden_dims=hidden_dims,
                                                              weight_decay=wd,
                                                              rand_size=rand_size)

        # Check iteration error. If the MAE between two iterations is less than
        # setting threshold values, exit.
        prediction = remove_low_values(prediction, cutoff=0.005)
        iter_error = mean_absolute_error(prediction, prediction_pre)
        prediction_pre = prediction.copy()

        if i == num_iter - 2:
            return spa_X, pseudo_X


def staid_deconv(spa_adata,
                 sc_adata,
                 anno_key='cell_type',
                 device='auto',
                 error_cutoff=0.02,
                 lr=0.0005,
                 num_pseudo=5000,
                 num_epoch=200,
                 num_iter=5,
                 batch_size=512,
                 min_cells=1,
                 max_cells=20,
                 k=(8, 15),
                 remove_platform='auto',
                 random_spot_rate=0.2,
                 hidden_dims=[512, 128, 64],
                 library_size=1e3,
                 dropout=0.1):
    """
    Infer cell type composition by graph attention neural networks. This 
    function contains two parts, that are, initial prediction using pseudo
    spots from enrichment analysis and interation. that are, generating pseudo
    spots and predict cell type composition using GAT.
    reference: https://arxiv.org/abs/2105.14491.
    
    Parameters
    ----------
    spa_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by spa_adata.X and spat-
        ial coordinates which could be extract by spa_adata.obs[spatial_names].
    sc_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by sc_adata and cell ty-
        pe annotaion.
    spatial_key : list or tupple or str
        List or tupple indicates the spatial columns in spa_adata.obs. The
        coordinate information can be found at spa_adat.obs[spatial_names]. If
        spatial key belongs to str, The spatial information can be found at
        spa_adata.obsm['spatial_key'].
        The default is ['tissue_in_row', 'tissue_in_col']. 
    anno_key : string
        The column name indicate cell type annotation information in sc_adata.
        obs. That is, the cell type annotaion information could be found by
        sc_adata.obs[anno_name]. The default is 'cell_type'.
    device : str, optional
        The device for training, if 'auto', the model will select device 
        automatically. It could be 'cup', 'gpu' or 'cuda:0' etc. The default
        is 'auto'. 
    iter_cutoff : float, optional
        If error between two contiguous prediction is less the iter_cutoff, 
        the iteration will be stoped beforehand. The default is 0.01.
    lr : float, optional
        Learn rate in training process. The default is 0.001.
    num_pseudo : int, optional
        The number of pseudo spots to be generated. The default is 5000.
    num_epoch : int optional
        The epoch in initial training. The default is 100.
    num_whole_train : int, optional
        The number of epochs in training the whole graph. The default is None.
    num_iter :int, optional
        The number of iteration processes. The default is 5.
    batch_size : int, optional
        The size of batch in subgraph training. The default is 64.
    min_cells : int, optional
        The minimum cells that a pseudo spot contains. The default is 1.
    max_cells :int, optional
        The maximum cells that a pseudo spot contains. The default is 20.
    k : tupple or list, optional
        Define the numbers of genes in real-real, pseudo-pseudo
        The default is (8, 15).
    remove_platform : bool | 'auto', optional
        Whether need to remove the platform effects. This step is implemented
        by scanpy.pp.combat(). More details could be found in Johnson, Li & 
        Rabinovic (2007), Adjusting batch effects in microarray expression data
        using empirical Bayes methods, Biostatistics.
        The default is 'auto'.
    random_spot_rate: tupple, optional
        To control the ratio of random pseudo spots. 
        The default is 0.2.
    dropout: float , optional
        dropout controls the dropout rate of ARMAConv layer. 
        The default is 0.1.
    Returns
    -------
    prediction : DateFrame
        The prediction of cell type composition of real spots.

    """
    res_list = []
    # ************Generate pseudo spots by enrichment**************
    # Enrichment_analysis
    shared_genes = np.intersect1d(sc_adata.var_names,
                                  spa_adata.var_names).tolist()
    sc_adata = sc_adata[:, shared_genes]
    spa_adata = spa_adata[:, shared_genes]
    if ss.issparse(sc_adata.X):
        sc_adata_total = sc.AnnData(sc_adata.X.todense(),
                                    obs=sc_adata.obs,
                                    var=sc_adata.var)
    else:
        sc_adata_total = sc_adata
    if ss.issparse(spa_adata.X):
        spa_adata = sc.AnnData(spa_adata.X.todense(),
                               obs=spa_adata.obs,
                               var=spa_adata.var,
                               obsm=spa_adata.obsm)
    else:
        spa_adata = spa_adata
    from staid.utils import simplify_refer, extract_hvgs

    # select device for training
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    if device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            raise KeyError("Can not find gpu device or cuda is not availabel.")

    # remove platform
    if remove_platform == True:
        remove_p = True
    else:
        remove_p = False

    # check parameters
    if not 0 <= random_spot_rate < 1:
        raise ValueError(f"random_spot_rate should in [0, 1), but got \
                         {random_spot_rate}")

    # simplify
    sc_adata = simplify_refer(sc_adata=sc_adata_total,
                              anno_key=anno_key)
    ct_spot_enrich_df, marker_genes, domains = enrichment_mia(spa_adata,
                                                              sc_adata,
                                                              anno_name=anno_key,
                                                              resolution=1)
    print(f"The number of marker genes: {len(marker_genes)}")
    if len(marker_genes) <= 2000:
        hvgs = extract_hvgs(sc_adata, n_genes=2000)
        marker_genes = np.union1d(marker_genes, hvgs)
        sc_adata = sc_adata[:, marker_genes]
        sc_adata_total = sc_adata_total[:, marker_genes]
        spa_adata = spa_adata[:, marker_genes]
    elif len(marker_genes) >= 5000:
        spa_adata = spa_adata[:, marker_genes]
        extract_genes = extract_hvgs(sc_adata, n_genes=5000)
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names
    else:
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names

    print("*************** Iteration: \t1 ***************")
    # Generate pseudo spots and their cell type compositions and Merge them 
    # with real spots.
    pseudo_num_rate = int(np.ceil(num_pseudo / spa_adata.shape[0]))
    spa_X, pseudo_X, pseudo_df_composition = \
        generate_merge_initial(sc_adata,
                               spa_adata,
                               marker_genes=marker_genes,
                               enrichment_score_df=ct_spot_enrich_df,
                               anno_name=anno_key,
                               min_cells=min_cells,
                               max_cells=max_cells,
                               pseudo_num_rate=pseudo_num_rate,
                               abs_relative_rate=((1 - random_spot_rate) * 0.65,
                                                  (1 - random_spot_rate) * 0.35,
                                                  random_spot_rate),
                               remove_platform=remove_p,
                               domains=domains,
                               library_size=library_size)
    print(pseudo_X.shape)

    # ************* initial prediction *********************
    # Adjust data type
    spa_X = pd.DataFrame(spa_X, index=spa_adata.obs_names,
                         columns=marker_genes)
    pseudo_X = pd.DataFrame(pseudo_X, index=pseudo_df_composition.columns,
                            columns=marker_genes)
    # Prediction using pseudo_spots from enrichment_analysis.
    prediction, gat_model = gat_pred_train_initial(spa_X,
                                                   pseudo_X,
                                                   pseudo_df_composition.transpose(),
                                                   num_epoch=num_epoch,
                                                   batch_size=batch_size,
                                                   k_real=k[0],
                                                   k_pseudo=k[1],
                                                   lr=lr,
                                                   device=device,
                                                   hidden_dims=hidden_dims,
                                                   dropout=dropout)
    prediction = remove_low_values(prediction)
    prediction_pre = prediction.copy()
    res_list.append(prediction)
    # *****************iteration processes***********************
    # Ensure pseudo num rate.
    if num_iter < 0 or not isinstance(num_iter, int):
        raise ValueError('num_iter should be non-negative integer')
    pert = [0.3, 0.1]
    # Remove platform effects
    if remove_platform == False:
        remove_p = False
    else:
        remove_p = True
    for i in range(num_iter - 1):
        if i == 0:
            perturbation = pert[0]
            gft_init = True
        else:
            perturbation = pert[1]
            gft_init = False
        if remove_platform == False:
            remove_p = False
        else:
            remove_p = True

        print(f"*************** Iteration:\t{i + 2} ***************")
        sc_adata = simplify_refer(sc_adata=sc_adata_total.copy(),
                                  anno_key=anno_key)
        spa_X, pseudo_X, pseudo_df_composition = \
            generate_merge_iter(sc_adata=sc_adata, spa_adata=spa_adata, pre_deconvolution=prediction,
                                marker_genes=marker_genes, anno_name=anno_key, min_cells=min_cells, max_cells=max_cells,
                                abs_relative_rate=((1 - random_spot_rate) * 0.65,
                                                   (1 - random_spot_rate) * 0.35,
                                                   random_spot_rate), pseudo_num_rate_iter=pseudo_num_rate,
                                remove_platform=remove_p, perturbation=perturbation, library_size=library_size)
        # Adjust data format        
        spa_X = pd.DataFrame(spa_X, index=spa_adata.obs_names,
                             columns=marker_genes)
        pseudo_X = pd.DataFrame(pseudo_X,
                                index=pseudo_df_composition.columns,
                                columns=marker_genes)
        # Prediction using pseudo spots from previous deconvolution results
        prediction, gat_model = gat_pred_train_iter(spa_X,
                                                    pseudo_X,
                                                    gat_model,
                                                    pseudo_df_composition.transpose(),
                                                    num_epoch=num_epoch,
                                                    k_real=k[0],
                                                    k_pseudo=k[1],
                                                    batch_size=batch_size,
                                                    lr=lr,
                                                    device=device,
                                                    hidden_dims=hidden_dims,
                                                    gft_init=gft_init)

        # Check iteration error. If the MAE between two iterations is less than
        # setting threshold values, exit.
        prediction = remove_low_values(prediction)
        iter_error = mean_absolute_error(prediction, prediction_pre)
        prediction_pre = prediction.copy()
        res_list.append(prediction)
    spa_adata.obsm['deconvolution'] = prediction

    return res_list


def gat_predict_iteration(spa_adata,
                          sc_adata,
                          anno_key='cell_type',
                          device='auto',
                          error_cutoff=0.02,
                          spatial_info='spatial',
                          lr=0.0005,
                          num_pseudo=5000,
                          num_epoch=200,
                          num_iter=5,
                          batch_size=512,
                          min_cells=1,
                          max_cells=20,
                          k=(8, 15),
                          remove_platform='auto',
                          random_spot_rate=0.5,
                          hidden_dims=[512, 128, 64],
                          library_size=1e3,
                          dropout=0.1):
    """
    Infer cell type composition by graph attention neural networks. This
    function contains two parts, that are, initial prediction using pseudo
    spots from enrichment analysis and interation. that are, generating pseudo
    spots and predict cell type composition using GAT.
    reference: https://arxiv.org/abs/2105.14491.

    Parameters
    ----------
    spa_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by spa_adata.X and spat-
        ial coordinates which could be extract by spa_adata.obs[spatial_names].
    sc_adata : AnnData object
        The spatial data with anndata format. Note that, the two parts are ess-
        ential, that are, count matrix should be found by sc_adata and cell ty-
        pe annotation.
    spatial_key : list or tupple or str
        List or tupple indicates the spatial columns in spa_adata.obs. The
        coordinate information can be found at spa_adat.obs[spatial_names]. If
        spatial key belongs to str, The spatial information can be found at
        spa_adata.obsm['spatial_key'].
        The default is ['tissue_in_row', 'tissue_in_col'].
    anno_key : string
        The column name indicate cell type annotation information in sc_adata.
        obs. That is, the cell type annotation information could be found by
        sc_adata.obs[anno_name]. The default is 'cell_type'.
    device : str, optional
        The device for training, if 'auto', the model will select device
        automatically. It could be 'cup', 'gpu' or 'cuda:0' etc. The default
        is 'auto'.
    iter_cutoff : float, optional
        If error between two contiguous prediction is less the iter_cutoff,
        the iteration will be stoped beforehand. The default is 0.01.
    lr : float, optional
        Learn rate in training process. The default is 0.001.
    num_pseudo : int, optional
        The number of pseudo spots to be generated. The default is 5000.
    num_epoch : int optional
        The epoch in initial training. The default is 100.
    num_whole_train : int, optional
        The number of epochs in training the whole graph. The default is None.
    num_iter :int, optional
        The number of iteration processes. The default is 5.
    batch_size : int, optional
        The size of batch in subgraph training. The default is 64.
    min_cells : int, optional
        The minimum cells that a pseudo spot contains. The default is 1.
    max_cells :int, optional
        The maximum cells that a pseudo spot contains. The default is 20.
    k : tuple or list, optional
        Define the numbers of genes in real-real, pseudo-pseudo
        The default is (8, 15).
    remove_platform : bool | 'auto', optional
        Whether you need to remove the platform effects. This step is implemented
        by scanpy.pp.combat(). More details could be found in Johnson, Li &
        Rabinovic (2007), Adjusting batch effects in microarray expression data
        using empirical Bayes methods, Biostatistics.
        The default is 'auto'.
    random_spot_rate: tuple, optional
        To control the ratio of random pseudo spots.
        The default is 0.2.
    dropout: float , optional
        dropout controls the dropout rate of ARMAConv layer.
        The default is 0.1.
    Returns
    -------
    prediction : DateFrame
        The prediction of cell type composition of real spots.

    """
    prediction_list = []
    # ************Generate pseudo spots by enrichment**************
    # Enrichment_analysis
    # Process scRNA-seq to boost
    # correct_adata(sc_adata, spa_adata)
    shared_genes = np.intersect1d(sc_adata.var_names,
                                  spa_adata.var_names).tolist()
    sc_adata = sc_adata[:, shared_genes]
    spa_adata = spa_adata[:, shared_genes]
    if ss.issparse(sc_adata.X):
        sc_adata_total = sc.AnnData(sc_adata.X.todense(),
                                    obs=sc_adata.obs,
                                    var=sc_adata.var)
    else:
        sc_adata_total = sc_adata
    if ss.issparse(spa_adata.X):
        spa_adata = sc.AnnData(spa_adata.X.todense(),
                               obs=spa_adata.obs,
                               var=spa_adata.var,
                               obsm=spa_adata.obsm)
    else:
        spa_adata = spa_adata
    from staid.utils import simplify_refer, extract_hvgs

    # select device for training
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    if device == 'gpu':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            raise KeyError("Can not find gpu device or cuda is not available.")

    # remove platform
    if remove_platform == True:
        remove_p = True
    else:
        remove_p = False
    wd = 0
    # check parameters
    if random_spot_rate > 1 or random_spot_rate < 0:
        raise ValueError(f"random_spot_rate should in [0, 1), but got \
                         {random_spot_rate}")

    # simplify
    cell_per_ct = min(int(sc_adata_total.shape[0] / np.unique(sc_adata.obs[anno_key]).size),
                      500)
    sc_adata = simplify_refer(sc_adata=sc_adata_total,
                              anno_key=anno_key,
                              cell_per_ct=cell_per_ct,
                              tmp_num=1)
    ct_spot_enrich_df, marker_genes, domains = enrichment_mia(spa_adata,
                                                              sc_adata,
                                                              anno_name=anno_key,
                                                              resolution=1)
    print(f'The number of marker genes: {len(marker_genes)}')
    if len(marker_genes) <= 2000:
        hvgs = extract_hvgs(sc_adata, n_genes=2000)
        marker_genes = np.union1d(marker_genes, hvgs)
        sc_adata = sc_adata[:, marker_genes]
        sc_adata_total = sc_adata_total[:, marker_genes]
        spa_adata = spa_adata[:, marker_genes]
    elif len(marker_genes) >= 5000:
        spa_adata = spa_adata[:, marker_genes]
        extract_genes = extract_hvgs(sc_adata, n_genes=5000)
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names
    else:
        extract_genes = marker_genes
        spa_adata = spa_adata[:, extract_genes]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names

    print("*************** Iteration: \t1 ***************")
    # Generate pseudo spots and their cell type compositions and merge them
    # with real spots
    pseudo_num_rate = int(np.ceil(num_pseudo / spa_adata.shape[0]))
    spa_X, pseudo_X, pseudo_df_composition, rand_size = \
        generate_merge_initial(sc_adata,
                               spa_adata,
                               marker_genes=marker_genes,
                               enrichment_score_df=ct_spot_enrich_df,
                               anno_name=anno_key,
                               min_cells=min_cells,
                               max_cells=max_cells,
                               pseudo_num_rate=pseudo_num_rate,
                               abs_relative_rate=((1 - random_spot_rate) * 0.65,
                                                  (1 - random_spot_rate) * 0.35,
                                                  random_spot_rate),
                               remove_platform=remove_p,
                               domains=domains,
                               library_size=library_size)
    print(pseudo_X.shape)
    # low pass filter
    # spa_X, filter_mtx = low_pass_filter_init(spa_X, c=0.001)
    # pseudo_X = low_pass_filter(pseudo_X, filter_mtx)

    # ************* initial prediction *********************
    # Adjust data type
    spa_X = pd.DataFrame(spa_X,
                         index=spa_adata.obs_names,
                         columns=marker_genes)
    pseudo_X = pd.DataFrame(pseudo_X,
                            index=pseudo_df_composition.columns,
                            columns=marker_genes)
    # Prediction using pseudo_spots from enrichment_analysis.
    prediction, gat_model, ae_model = gat_pred_train_initial(spa_X,
                                                             pseudo_X,
                                                             pseudo_df_composition.transpose(),
                                                             num_epoch=num_epoch,
                                                             batch_size=batch_size,
                                                             lr=lr,
                                                             device=device,
                                                             hidden_dims=hidden_dims,
                                                             dropout=dropout,
                                                             weight_decay=wd)
    prediction = remove_low_values(prediction)
    prediction_pre = prediction.copy()
    prediction_list.append(prediction)

    # *****************iteration processes***********************
    # Ensure pseudo num rate.
    if num_iter < 0 or not isinstance(num_iter, int):
        raise ValueError('num_iter should be non-negative integer')
    # random_spot_rate = random_spot_rate / 2
    # Remove platform effects
    if remove_platform == False:
        remove_p = False
    else:
        remove_p = True
    for i in range(num_iter - 1):
        # adjust learning rate in iterations
        if i % 5 == 1:
            lr = lr / 2
        perturbation = max(0.5 * (num_iter - i) / num_iter, 0.25)
        print(f"*************** Iteration:\t{i + 2} ***************")
        sc_adata = simplify_refer(sc_adata=sc_adata_total.copy(),
                                  anno_key=anno_key,
                                  cell_per_ct=cell_per_ct,
                                  tmp_num=1)
        spa_X, pseudo_X, pseudo_df_composition, rand_size = generate_merge_iter(sc_adata=sc_adata,
                                                                                spa_adata=spa_adata,
                                                                                pre_deconvolution=prediction,
                                                                                marker_genes=marker_genes,
                                                                                anno_name=anno_key,
                                                                                min_cells=min_cells,
                                                                                max_cells=max_cells,
                                                                                abs_relative_rate=(
                                                                                    (1 - random_spot_rate) * 0.65,
                                                                                    (1 - random_spot_rate) * 0.35,
                                                                                    random_spot_rate),
                                                                                pseudo_num_rate_iter=pseudo_num_rate,
                                                                                remove_platform=remove_p,
                                                                                perturbation=perturbation,
                                                                                library_size=library_size)
        # pseudo_X = pseudo_X = low_pass_filter(pseudo_X, filter_mtx)
        spa_X = pd.DataFrame(spa_X, index=spa_adata.obs_names,
                             columns=marker_genes)
        pseudo_X = pd.DataFrame(pseudo_X,
                                index=pseudo_df_composition.columns,
                                columns=marker_genes)
        print(pseudo_X.shape)
        # Prediction using pseudo spots from previous deconvolution results
        prediction, gat_model, ae_model = gat_pred_train_iter(spa_X,
                                                              pseudo_X,
                                                              gat_model,
                                                              ae_model,
                                                              pseudo_df_composition.transpose(),
                                                              num_epoch=num_epoch,
                                                              k_real=k[0],
                                                              k_pseudo=k[1],
                                                              batch_size=batch_size,
                                                              lr=lr,
                                                              device=device,
                                                              hidden_dims=hidden_dims,
                                                              weight_decay=wd,
                                                              rand_size=rand_size)

        # Check iteration error. If the MAE between two iterations is less than
        # setting threshold values, exit.
        prediction = remove_low_values(prediction, cutoff=0.005)
        prediction_list.append(prediction)
        iter_error = mean_absolute_error(prediction, prediction_pre)
        prediction_pre = prediction.copy()

        if iter_error < error_cutoff:
            print("Iteration Error ", iter_error, " < ", "Setting Error ",
                  error_cutoff)
            print("Exit!")
            break

    spa_adata.obsm['deconvolution'] = prediction

    return prediction_list


def gft_using_gene_network(exp_mtx, eigen_vectors, eigen_values, c=0.2):
    freq_weight = np.diag([1 / (1 + c * eigv) for eigv in eigen_values])
    # freq_weight = np.diag([(1 + c * eigv) for eigv in eigen_values])
    freq_mtx = eigen_vectors.transpose() @ exp_mtx.transpose()
    # freq_mtx = eigen_vectors @ freq_weight @ freq_mtx
    freq_mtx = freq_weight @ freq_mtx
    freq_mtx = freq_mtx.transpose()

    # freq_mtx2 = exp_mtx
    # # freq_weight2 = np.diag([(1 + c * eigv) for eigv in eigen_values])
    # # freq_mtx2 = eigen_vectors.transpose() @ exp_mtx.transpose()
    # # freq_mtx2 = freq_weight2 @ freq_mtx2
    # # freq_mtx2 = freq_mtx2.transpose()
    # #
    # freq_mtx = np.concatenate((freq_mtx, freq_mtx2), axis=1)

    return freq_mtx
