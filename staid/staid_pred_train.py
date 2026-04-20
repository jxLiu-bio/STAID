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
from sklearn.preprocessing import Normalizer
from typing import Callable, Union


warnings.filterwarnings("ignore")
torch.set_default_tensor_type(torch.FloatTensor)


def sparsity_targets(targets):
    norm1_targets = torch.norm(targets, dim=1, p=1)
    norm2_targets = torch.norm(targets, dim=1, p=2)
    sp_targets = torch.mean(norm1_targets / norm2_targets)

    return sp_targets


def hoyer_sparsity(targets):
    n = targets.shape[1]
    l1 = torch.norm(targets, p=1, dim=1)
    l2 = torch.norm(targets, p=2, dim=1)
    sparsity = (torch.sqrt(torch.tensor(float(n))) - (l1 / l2)) / (torch.sqrt(torch.tensor(float(n))) - 1)
    return torch.mean(sparsity)


def staid_pred_train_initial(real_df,
                             pseudo_df,
                             targets_df,
                             num_epoch=200,
                             batch_size=128,
                             lr=0.0005,
                             device='cpu',
                             hidden_dims=[512, 128, 128],
                             dropout=0.1,
                             abs_size=None,
                             weight=1e-8):
    # AE training
    total_df = pd.concat((pseudo_df, real_df), axis=0)
    total_df = total_df.values
    total_df = torch.from_numpy(total_df).float().to(device)

    real_tensor = torch.from_numpy(real_df.values).float().to(device)
    abs_tensor = torch.from_numpy(pseudo_df.values[:abs_size, :]).float().to(device)
    data_loader_ae = Data.DataLoader(total_df,
                                     batch_size=int(batch_size),
                                     shuffle=True)
    ae_dims = [512, 512]
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
            loss = F.mse_loss(outputs, sampled_X)
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
                                       size=300,
                                       replace=False).tolist()
    training_spots = np.setdiff1d(pseudo_df.index.tolist(), validated_spots)

    # Obtain the validated datasets
    b_X_v = pseudo_df.loc[validated_spots, :].values
    b_X_v = torch.from_numpy(b_X_v)
    b_X_v = b_X_v.float().to(device)
    targets_v = targets_df.loc[validated_spots, :].values
    targets_v = torch.from_numpy(targets_v)
    targets_v = targets_v.float().to(device)

    # Obtain the training datasets
    b_X_train = pseudo_df.loc[training_spots, :].values
    b_X_train = torch.from_numpy(b_X_train)
    b_X_train = b_X_train.float().to(device)
    targets_train = targets_df.loc[training_spots, :].values
    targets_train = torch.from_numpy(targets_train)
    targets_train = targets_train.float().to(device)

    # Define the training loader
    training_data = Data.TensorDataset(b_X_train, targets_train)
    pseudo_train_loader = Data.DataLoader(dataset=training_data,
                                          batch_size=int(batch_size),
                                          shuffle=True,
                                          num_workers=0)
    # Define the model and optimizer
    pred_model = MLP_pred(ae_dims[0],
                          targets_df.shape[1],
                          dim=hidden_dims,
                          dropout=dropout).to(device)
    optimizer = torch.optim.Adam(pred_model.parameters(),
                                 lr=lr)

    # Train the model
    epoch_tqdm = tqdm(range(num_epoch), desc="Epoch")
    valid_loss_list = []
    for epoch in epoch_tqdm:
        loss_list = []
        if epoch == 120:
            optimizer = torch.optim.Adam(list(ae_model.parameters()) +
                                         list(pred_model.parameters()),
                                         lr=lr)
        for step, (b_x, b_y) in enumerate(pseudo_train_loader):
            optimizer.zero_grad()
            outputs = ae_model.encoder(b_x)
            outputs = pred_model.forward(outputs)
            sparsity_term = hoyer_sparsity(outputs)
            loss = F.mse_loss(outputs, b_y) - weight * sparsity_term
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        epoch_tqdm.set_postfix(loss=np.mean(loss_list))
        # validation
        with torch.no_grad():
            outputs = ae_model.encoder(b_X_v)
            outputs = pred_model.forward(outputs)
            loss = F.mse_loss(outputs, targets_v)
            valid_loss_list.append(loss.item())
        if len(valid_loss_list) > 150:
            if np.mean(valid_loss_list[-10:-5]) < 1.00 * np.mean(valid_loss_list[-5:]):
                break

    # Predict
    b_X_v = real_df.values
    b_X_v = torch.from_numpy(b_X_v)
    b_X_v = b_X_v.float().to(device)

    with torch.no_grad():
        outputs = ae_model.encoder(b_X_v)
        outputs = pred_model.forward(outputs)
        pred_df = pd.DataFrame(outputs.cpu().numpy(),
                               index=real_df.index,
                               columns=targets_df.columns)

    return pred_df, pred_model, ae_model


def staid_pred_train_iter(real_df,
                          pseudo_df,
                          mlp_model,
                          ae_model,
                          targets_df,
                          num_epoch=100,
                          batch_size=32,
                          lr=0.0001,
                          device='cpu',
                          weight_decay=0,
                          weight=1e-5):
    # Select validation datasets
    validated_spots = np.random.choice(pseudo_df.index.tolist(),
                                       size=300,
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
                                 list(mlp_model.parameters()),
                                 lr=lr,
                                 weight_decay=weight_decay)

    # Train the model
    epoch_tqdm = tqdm(range(num_epoch), desc="Epoch")
    valid_loss_list = []
    for epoch in epoch_tqdm:
        loss_list = []
        for step, (b_x, b_y) in enumerate(pseudo_train_loader):
            optimizer.zero_grad()
            outputs = ae_model.encoder(b_x)
            outputs = mlp_model.forward(outputs)
            sparsity_term = hoyer_sparsity(outputs)
            loss = F.mse_loss(outputs, b_y) - weight * sparsity_term
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        epoch_tqdm.set_postfix(loss=np.mean(loss_list))
        # validation
        with torch.no_grad():
            outputs = ae_model.encoder(b_X_v)
            outputs = mlp_model.forward(outputs)
            loss = F.mse_loss(outputs, targets_v)
            valid_loss_list.append(loss.item())
        if len(valid_loss_list) > 20:
            if np.mean(valid_loss_list[-10:-5]) < 1.00 * np.mean(valid_loss_list[-5:]):
                break

    # Predict
    b_X_v = real_df.values
    b_X_v = torch.from_numpy(b_X_v)
    b_X_v = b_X_v.float().to(device)

    with torch.no_grad():
        outputs = ae_model.encoder(b_X_v)
        outputs = mlp_model.forward(outputs)
        pred_df = pd.DataFrame(outputs.cpu().numpy(),
                               index=real_df.index,
                               columns=targets_df.columns)

    return pred_df, mlp_model, ae_model


def run_deconvolution(sp_adata,
                      sc_adata,
                      anno_key='cell_type',
                      device='auto',
                      error_cutoff=0.01,
                      lr=0.0005,
                      num_pseudo=5000,
                      num_epoch=200,
                      num_iter=3,
                      batch_size=512,
                      min_cells=1,
                      max_cells=8,
                      cutoff=0.5,
                      batch_correction: Union[str, bool, Callable] = False,
                      random_spot_rate=0.3,
                      hidden_dims=[512, 128, 64],
                      library_size=1e4,
                      dropout=0.05,
                      c=0.1,
                      weight=0,
                      method='cosine'):
    """
    Deconvolve spatial transcriptomics data to infer cell type compositions
    using deep learning integrated with  graph Fourier transform (GFT).
    The procedure combines pseudo-spot generation, initial training, and iterative refinement.

    This function contains three main stages:
      1. **Preprocessing**: Filter genes, remove mitochondrial genes,
         select shared markers, and construct the gene network.
      2. **Pseudo-spot generation**: Generate synthetic spots with known
         cell type proportions for model training.
      3. **Iterative deconvolution**: Train deep learning models with
         pseudo-spots and refine predictions across multiple iterations
         until convergence.

    Parameters
    ----------
    sp_adata : AnnData
        Spatial transcriptomics data. Requires:
          - Counts matrix: `sp_adata.X`
    sc_adata : AnnData
        Single-cell RNA-seq reference data. Requires:
          - Counts matrix: `sc_adata.X`
          - Cell type annotations: `sc_adata.obs[anno_key]`
    anno_key : str, optional
        Column name in `sc_adata.obs` containing cell type labels.
        Default is `'cell_type'`.
    device : str, optional
        Computational device. Options:
          - `'auto'`: automatically choose GPU if available, otherwise CPU
          - `'cpu'`, `'cuda:0'`, etc.
        Default is `'auto'`.
    error_cutoff : float, optional
        Convergence threshold for mean absolute error (MAE) between
        consecutive iterations. If MAE < `error_cutoff`, iteration stops
        early. Default is `0.01`.
    lr : float, optional
        Learning rate for model training. Default is `0.0005`.
    num_pseudo : int, optional
        Number of pseudo-spots to generate. Default is `5000`.
    num_epoch : int, optional
        Training epochs for each iteration. Default is `200`.
    num_iter : int, optional
        Maximum number of deconvolution iterations. Default is `5`.
    batch_size : int, optional
        Mini-batch size during model training. Default is `512`.
    min_cells : int, optional
        Minimum number of cells per pseudo-spot. Default is `1`.
    max_cells : int, optional
        Maximum number of cells per pseudo-spot. Default is `20`.
    cutoff : float, optional
        Graph construction threshold for gene network. Default is `0.5`.
    batch_correction : str, callable or bool, optional
        Method for platform correction (e.g., `'combat'`, `'scanorama'`).
        Default is False.
    random_spot_rate : float, optional
        Proportion of pseudo-spots generated randomly. Must be between 0
        and 1. Default is `0.3`.
    hidden_dims : list of int, optional
        Hidden layer dimensions for the neural network. Default is `[512, 128, 64]`.
    library_size : float, optional
        Target library size for normalization. Default is `1e4`.
    dropout : float, optional
        Dropout rate applied to GAT layers. Default is `0.05`.
    c : float, optional
        Regularization coefficient for GFT. Default is `0.1`.
    weight : float, optional
        Loss weighting factor for reconstruction loss. Default is `0`.
    method : str, optional
        Similarity measure for graph construction. Default is `'cosine'`.

    Returns
    -------
    prediction : pandas.DataFrame
        A dataframe containing predicted cell type proportions for each
        spatial spot. The results are also stored in
        `sp_adata.obsm['deconvolution']`.

    Notes
    -----
    - Pseudo-spots are generated by combining single-cell profiles, with
      enrichment scores guiding their compositions.
    - Graph Fourier transform (GFT) is applied to both real and pseudo
      expression matrices to incorporate network-based smoothness.
    - Iterative refinement continues until convergence or until the
      maximum number of iterations is reached.
    """

    # ************Generate pseudo spots by enrichment**************
    # Enrichment_analysis
    # Process scRNA-seq to boost
    # correct_adata(sc_adata, spa_adata)
    if ss.issparse(sc_adata.X):
        adata_sc = sc_adata.X.toarray()
    else:
        adata_sc = sc_adata.X
    adata_sc = pd.DataFrame(adata_sc,
                            index=sc_adata.obs_names,
                            columns=sc_adata.var_names)
    adata_sc = sc.AnnData(adata_sc)
    adata_sc.obs[anno_key] = sc_adata.obs[anno_key]
    if ss.issparse(sp_adata.X):
        adata_sp = sp_adata.X.toarray()
    else:
        adata_sp = sp_adata.X
    adata_sp = pd.DataFrame(adata_sp,
                            index=sp_adata.obs_names,
                            columns=sp_adata.var_names)
    adata_sp = sc.AnnData(adata_sp)
    # filter cells and gene
    sc.pp.filter_genes(adata_sc, min_cells=int(0.02 * adata_sc.shape[1]))
    sc.pp.filter_genes(adata_sp, min_cells=int(0.02 * adata_sp.shape[1]))
    adata_sc.var['mt'] = adata_sc.var_names.str.upper().str.startswith('MT-')
    adata_sp.var['mt'] = adata_sp.var_names.str.upper().str.startswith('MT-')
    adata_sc = adata_sc[:, ~adata_sc.var['mt']]
    adata_sp = adata_sp[:, ~adata_sp.var['mt']]

    shared_genes = np.intersect1d(adata_sc.var_names,
                                  adata_sp.var_names).tolist()
    adata_sp = adata_sp[:, shared_genes]
    adata_sc = adata_sc[:, shared_genes]

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

    # check parameters
    if random_spot_rate > 1 or random_spot_rate < 0:
        raise ValueError(f"random_spot_rate should in [0, 1), but got {random_spot_rate}")
    ct_spot_enrich_df, marker_genes, domains = enrichment_mia(adata_sp,
                                                              adata_sc,
                                                              anno_name=anno_key,
                                                              resolution=1)
    if len(marker_genes) <= 3000:
        hvgs = extract_hvgs(adata_sc, n_genes=3000)
        marker_genes = np.union1d(marker_genes, hvgs)
        adata_sp = adata_sp[:, marker_genes]
        adata_sc = adata_sc[:, marker_genes]
    elif len(marker_genes) >= 5000:
        adata_sc = adata_sc[:, marker_genes]
        extract_genes = extract_hvgs(adata_sc, n_genes=5000)
        marker_genes = extract_genes
        adata_sp = adata_sp[:, marker_genes]
        adata_sc = adata_sc[:, marker_genes]
    else:
        adata_sp = adata_sp[:, marker_genes]
        adata_sc = adata_sc[:, marker_genes]
    print(f'The number of marker genes: {len(marker_genes)}', )

    # simplify
    cell_per_ct = min(int(sc_adata.shape[0] / np.unique(sc_adata.obs[anno_key]).size), 500)
    adata_sc_sim = simplify_refer(sc_adata=adata_sc,
                                  anno_key=anno_key,
                                  cell_per_ct=cell_per_ct)

    print("*************** Iteration: \t1 ***************")
    # Construct gene network and obtain the fourier modes
    eigen_vector_gene, eigen_value_gene, mean_deg = fourier_modes_gene_network(adata_sc,
                                                                               gene_list=marker_genes,
                                                                               cutoff=cutoff,
                                                                               method=method)
    c = c * 1 / (1e-5 + mean_deg)
    # Generate pseudo spots and their cell type compositions and merge them 
    # with real spots
    pseudo_num_rate = int(np.ceil(num_pseudo / sp_adata.shape[0]))
    spa_X, pseudo_X, pseudo_df_composition, abs_size = generate_merge_initial(adata_sc_sim,
                                                                              adata_sp,
                                                                              marker_genes=marker_genes,
                                                                              enrichment_score_df=ct_spot_enrich_df,
                                                                              anno_name=anno_key,
                                                                              min_cells=min_cells,
                                                                              max_cells=max_cells,
                                                                              pseudo_num_rate=pseudo_num_rate,
                                                                              abs_relative_rate=(
                                                                                  (1 - random_spot_rate) * 0.5,
                                                                                  (1 - random_spot_rate) * 0.5,
                                                                                  random_spot_rate),
                                                                              remove_platform=batch_correction,
                                                                              domains=domains,
                                                                              library_size=library_size)

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
                         index=sp_adata.obs_names)
    pseudo_X = pd.DataFrame(pseudo_X,
                            index=pseudo_df_composition.columns)
    # Prediction using pseudo_spots from enrichment_analysis.
    prediction, gat_model, ae_model = staid_pred_train_initial(spa_X, pseudo_X, pseudo_df_composition.transpose(),
                                                               num_epoch=num_epoch, batch_size=batch_size, lr=lr,
                                                               device=device, hidden_dims=hidden_dims,
                                                               dropout=dropout, abs_size=abs_size, weight=weight)
    prediction = remove_low_values(prediction, cutoff=0.005)
    prediction_pre = prediction.copy()

    try:
        gd_df = sp_adata.obsm['cell_type_proportion']
        common_cts = np.intersect1d(gd_df.columns, prediction.columns)
        gd_df = gd_df.loc[prediction.index, common_cts]
        mae = mean_absolute_error(gd_df, prediction.loc[:, common_cts])
        print(mae)
    except:
        mae = 0

    # *****************iteration processes***********************
    # Ensure pseudo num rate.
    if num_iter < 0 or not isinstance(num_iter, int):
        raise ValueError('num_iter should be non-negative integer')
    # random_spot_rate = random_spot_rate / 2
    # Remove platform effects
    perturbation = 0.5
    for i in range(num_iter - 1):
        im = min(i, 9)
        # adjust learning rate in iterations
        lr_decrease = lr * (1 - im / 10)
        perturbation_decrease = perturbation * (1 - im / 10)
        print(f"*************** Iteration:\t{i + 2} ***************")
        adata_sc_sim = simplify_refer(sc_adata=adata_sc,
                                      anno_key=anno_key,
                                      cell_per_ct=cell_per_ct)
        spa_X, pseudo_X, pseudo_df_composition, abs_size = generate_merge_iter(sc_adata=adata_sc_sim,
                                                                               spa_adata=adata_sp,
                                                                               pre_deconvo=prediction,
                                                                               marker_genes=marker_genes,
                                                                               anno_name=anno_key,
                                                                               min_cells=min_cells,
                                                                               max_cells=max_cells,
                                                                               abs_relative_rate=(
                                                                                   (1 - random_spot_rate) * 0.5,
                                                                                   (1 - random_spot_rate) * 0.5,
                                                                                   random_spot_rate),
                                                                               pseudo_num_rate_iter=pseudo_num_rate,
                                                                               remove_platform=batch_correction,
                                                                               perturbation=perturbation_decrease,
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
                             index=sp_adata.obs_names)
        pseudo_X = pd.DataFrame(pseudo_X,
                                index=pseudo_df_composition.columns)
        # Prediction using pseudo spots from previous deconvolution results
        prediction, gat_model, ae_model = staid_pred_train_iter(spa_X, pseudo_X, gat_model, ae_model,
                                                                pseudo_df_composition.transpose(), num_epoch=num_epoch,
                                                                batch_size=batch_size, lr=lr_decrease, device=device,
                                                                weight=weight)

        # Check iteration error. If the MAE between two iterations is less than
        # setting threshold values, exit.
        prediction = remove_low_values(prediction, cutoff=0.005)
        iter_error = mean_absolute_error(prediction, prediction_pre)
        prediction_pre = prediction.copy()
        try:
            gd_df = sp_adata.obsm['cell_type_proportion']
            common_cts = np.intersect1d(gd_df.columns, prediction.columns)
            gd_df = gd_df.loc[prediction.index, common_cts]
            mae = mean_absolute_error(gd_df, prediction.loc[:, common_cts])
            print(mae)
        except:
            mae = 0

        if iter_error < error_cutoff:
            print("Iteration Error ", iter_error, " < ", "Setting Error ",
                  error_cutoff)
            print("Exit!")
            break

    sp_adata.obsm['deconvolution'] = prediction

    return prediction


def gft_using_gene_network(exp_mtx, eigen_vectors, eigen_values, c=0.1):
    freq_weight = np.diag([1 / (1 + c * eigv) for eigv in eigen_values])
    freq_mtx = eigen_vectors.transpose() @ exp_mtx.transpose()
    freq_mtx = freq_weight @ freq_mtx
    normalizer = Normalizer(norm='l2')
    freq_mtx = freq_mtx.transpose()
    freq_mtx = normalizer.fit_transform(freq_mtx)

    return freq_mtx
