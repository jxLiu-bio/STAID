import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as ss
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

from staid.enrichment_mia import enrichment_mia
from staid.generate_pseudo import generate_merge_initial, generate_merge_iter
from staid.mlp_pred_nn import MLP_pre


def mlp_pred_train(X, targets,
                   num_epochs=100,
                   lr=0.0001,
                   batch_size=64,
                   model_name="MLP_0",
                   device='cpu',
                   ):
    """
    Train the model by mini batch from generated pseudo spots.

    Parameters
    ----------
    X : array
        The count matrix of pseudo spots, which is spot * gene.
    targets : array
        The cell type composition of pseudo spots, which is spot * cellType.
    num_epochs : int, optional
        The number of traning epochs. The default is 100.
    lr : float, optional
        The learning rate in optimizer. The default is 0.0001.
    batch_size : int, optional
        The mini batch size in traning process. The default is 64.
    model_name : str, optional
        Model name if need to save. The default is "MLP_0".
    device : str, optional
        device for training.

    Returns
    -------
    MLP : neural network model
        The trained model.

    """
    # Convert data
    if device == 'cpu':
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(targets.astype(np.float32))
        num_workers = 2
    else:
        X = torch.from_numpy(X.astype(np.float32)).cuda(device)
        y = torch.from_numpy(targets.astype(np.float32)).cuda(device)
        num_workers = 0

    # Create Dataset
    valid_spots = np.random.choice(list(range(X.shape[0])),
                                   int(0.1 * X.shape[0]),
                                   replace=False)
    train_spots = np.setdiff1d(list(range(X.shape[0])), valid_spots)
    b_X_v = X[valid_spots, :]
    y_v = y[valid_spots, :]
    valid_loss_list = []
    Dataset = Data.TensorDataset(X[train_spots, :], y[train_spots, :])
    train_loader = Data.DataLoader(
        dataset=Dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)
    train_loss = []
    MLP = MLP_pre(X.shape[1], y.shape[1], lr=lr)
    if device != 'cpu':
        MLP.cuda(device)
        print("Load to cuda")

    # Training
    epoch_tqdm = tqdm(range(num_epochs), desc='Epoch')
    for epoch in epoch_tqdm:
        MLP.train()
        tmp_train_loss = []
        for step, (b_X, b_y) in enumerate(train_loader):
            # tmp_loss_item = MLP.train(b_X, b_y)
            outputs = MLP.forward(b_X)
            loss = MLP.loss_function(outputs, b_y)

            # Update the parameters
            MLP.optimizer.zero_grad()
            loss.backward()
            MLP.optimizer.step()
            tmp_train_loss.append(loss.item())
        # MLP.scheduler.step()
        # print("lr:", MLP.optimizer.param_groups[0]['lr'])
        train_loss.append(np.mean(tmp_train_loss))
        epoch_tqdm.set_postfix(loss=train_loss[-1])
        # print("Epoch: ", epoch, "; \t Loss:", train_loss[-1])
        MLP.eval()
        outputs = MLP.forward(b_X_v)
        valid_loss_list.append(F.mse_loss(outputs, y_v).item())
        if len(valid_loss_list) >= 30 and np.mean(valid_loss_list[-20:-10]) < \
                np.mean(valid_loss_list[-10:]):
            loss_cutoff = np.mean(valid_loss_list[-20:-10]).copy()
            for new_epoch in range(1, 11):
                MLP.train()
                tmp_train_loss = []
                for step, (b_X, b_y) in enumerate(train_loader):
                    outputs = MLP.forward(b_X)
                    loss = MLP.loss_function(outputs, b_y)
                    # Update the parameters
                    MLP.optimizer.zero_grad()
                    loss.backward()
                    MLP.optimizer.step()
                    tmp_train_loss.append(loss.item())
                train_loss.append(np.mean(tmp_train_loss))
                MLP.eval()
                outputs = MLP.forward(b_X_v)
                valid_loss_list.append(F.mse_loss(outputs, y_v).item())
                if np.mean(valid_loss_list[-10:]) <= loss_cutoff:
                    break
            print("Early stopping")
            epoch_tqdm.clost()
            break
    return MLP


def mlp_pred_final(MLP, X_test, device='cpu'):
    """
    Predict cell type composition by trained MLP model.

    Parameters
    ----------
    MLP : neural network model
        The retrained neural network models.
    X_test : array
        The count matrix of real spots, which is spot * gene.
    device : str, optional
        device for training.

    Returns
    -------
    prediction : array
        The predicted cell type compositions of real spots, which is spot * ce-
        llType.

    """
    if device == 'cpu':
        X_test = torch.from_numpy(X_test.astype(np.float32))
    else:
        X_test = torch.from_numpy(X_test.astype(np.float32)).cuda(device)
    MLP.eval()
    prediction = MLP.forward(X_test)
    prediction = prediction.cpu().detach().numpy()

    return prediction


def mlp_train_iter(X, targets, MLP, batch_size=64,
                   num_epochs=100, model_name="MLP_0", device='cpu'):
    """
    For new generated pseudo spots, retrain the model to achieve a better perf-
    ormance. 

    Parameters
    ----------
    X : array
        The count matrix of pseudo spots, which is spot * gene.
    targets : array
        The cell type composition of pseudo spots, which is spot * cellType.
    MLP : neural network model
        The previousMLP  model.
    batch_size : int, optional
        The mini batch size in traning process. The default is 64.
    epoch_num : int, optional
        The number of traning epochs. The default is 100.
    model_name : str, optional
        model name. The default is "MLP_0".
    device : str, optional
        device for training.

    Returns
    -------
    MLP : neural network model
        The retrained neural network models.

    """
    # Convert data
    if device == 'cpu':
        X = torch.from_numpy(X.astype(np.float32))
        y = torch.from_numpy(targets.astype(np.float32))
        num_workers = 2
    else:
        X = torch.from_numpy(X.astype(np.float32)).cuda(device)
        y = torch.from_numpy(targets.astype(np.float32)).cuda(device)
        num_workers = 0

    # Create Dataset
    valid_spots = np.random.choice(list(range(X.shape[0])),
                                   int(0.1 * X.shape[0]),
                                   replace=False)
    train_spots = np.setdiff1d(list(range(X.shape[0])), valid_spots)
    b_X_v = X[valid_spots, :]
    y_v = y[valid_spots, :]
    valid_loss_list = []
    Dataset = Data.TensorDataset(X[train_spots, :], y[train_spots, :])

    train_loader = Data.DataLoader(
        dataset=Dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    train_loss = []
    epoch_tqdm = tqdm(range(num_epochs), desc="Epoch")
    for epoch in epoch_tqdm:
        MLP.train()
        tmp_train_loss = []
        for step, (b_X, b_y) in enumerate(train_loader):
            outputs = MLP.forward(b_X)
            loss = MLP.loss_function(outputs, b_y)

            # Update the parameters
            MLP.optimizer.zero_grad()
            loss.backward()
            MLP.optimizer.step()
            tmp_train_loss.append(loss.item())
        train_loss.append(np.mean(tmp_train_loss))
        # MLP.scheduler.step()
        # print("lr:", MLP.optimizer.param_groups[0]['lr'])
        epoch_tqdm.set_postfix(loss=train_loss[-1])
        # print("Epoch: ", epoch, "; \t Loss:", train_loss[-1])
        MLP.eval()
        outputs = MLP.forward(b_X_v)
        valid_loss_list.append(F.mse_loss(outputs, y_v).item())
        if len(valid_loss_list) >= 30 and np.mean(valid_loss_list[-20:-10]) < \
                np.mean(valid_loss_list[-10:]):
            loss_cutoff = np.mean(valid_loss_list[-20:-10]).copy()
            for new_epoch in range(1, 11):
                MLP.train()
                tmp_train_loss = []
                for step, (b_X, b_y) in enumerate(train_loader):
                    outputs = MLP.forward(b_X)
                    loss = MLP.loss_function(outputs, b_y)

                    # Update the parameters
                    MLP.optimizer.zero_grad()
                    loss.backward()
                    MLP.optimizer.step()
                    tmp_train_loss.append(loss.item())
                train_loss.append(np.mean(tmp_train_loss))
                # MLP.scheduler.step()
                # print("lr:", MLP.optimizer.param_groups[0]['lr'])
                # print("Epoch: ", epoch + new_epoch,
                #       "; \t Loss:", train_loss[-1])
                MLP.eval()
                outputs = MLP.forward(b_X_v)
                valid_loss_list.append(F.mse_loss(outputs, y_v).item())
                if np.mean(valid_loss_list[-10:]) <= loss_cutoff:
                    break
            epoch_tqdm.close()
            print("Early stopping")
            break
    # plt.plot(range(num_epochs), train_loss)
    # plt.show()
    # torch.save(MLP, "./" + model_name + ".pkl")
    return MLP


def mlp_predict(spa_adata,
                sc_adata,
                anno_key='cell_type',
                device='auto',
                num_iter=5,
                iter_cutoff=0.01,
                num_epochs=100,
                num_pseudo=5000,
                lr=0.0001,
                batch_size=64,
                min_cells=1,
                max_cells=15,
                remove_platform=False,
                abs_relative_rate=(0.3, 0.4, 0.3),
                q=0.05):
    """
    

    Parameters
    ----------
    spa_adata : AnnData
        The spatial data with anndata object, whose count matrix will be used 
        via spa_adata.X.
    sc_adata : AnnData
        The scRNA-seq with anndata object, which count matrix will be used by 
        sc_adata.X. Note: the cell type composition should be contained in
        sc_adta.obs.
    anno_key : str, optional
        The cell type annotation key in sc_adata.obs. The cell type annotaion
        could be found by sc_adata.obs[anno_name]. The default is 'cell_type'.
    device : str, optional
        The device for training, if 'auto', the model will select device 
        automatically. It could be 'cup', 'gpu' or 'cuda:0' etc. The default
        is 'auto'. 
    num_iter : int, optional
        The number of iterative deconvolution. The default is 10.
    iter_cutoff : float, optional
        The error cutoff for stopping interation. The default is 0.01.
    num_epochs: int, optional
        The number of epochs in each interaction. The default is 100.
    pseudo_num : int, optional
        The pseudo_num pseudo spots will be generated for following
        deconvolution step. The default is 5000.
    lr : float, optional
        Learn rate parameter in deep learning. The default is 0.0001.
    batch_size : int, optional
        Batch size paramter in deep learning. The default is 64.
    min_cells : int, optional
        The minimum cells that a pseudo spot contains. The default is 1.
    max_cells :int, optional
        The maximum cells that a pseudo spot contains. The default is 20.
    abs_relative_rate :tupple or list, optional
        It should contain 3 elements, representing the proportions of diverse
        types of pseudo spots.
    q : float
        To determine the sensitivity of cell type compositions in a spot in 
        enrichment step. It indicates a threshold for statistical 
        significance when determine the enriched cell types in a spot.
        The default is 0.05.

    Returns
    -------
    prediction : dataframe
        The prediction results.

    """
    # ************Generate pseudo spots by enrichment**************
    # Enrichment_analysis
    # Process scRNA-seq to boost
    if ss.issparse(sc_adata.X):
        sc_adata_total = sc.AnnData(sc_adata.X.todense(),
                                    obs=sc_adata.obs,
                                    var=sc_adata.var)
    else:
        sc_adata_total = sc_adata.copy()
    if ss.issparse(spa_adata.X):
        spa_adata = sc.AnnData(spa_adata.X.todense(),
                               obs=spa_adata.obs,
                               var=spa_adata.var)
    else:
        spa_adata = spa_adata.copy()
    # Ensure device for training
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    if device == 'gpu':
        device = 'cuda:0'
    from staid.utils import simplify_refer, extract_hvgs
    # samplyfy
    sc_adata = simplify_refer(sc_adata=sc_adata_total.copy(),
                              anno_key=anno_key,
                              cell_per_ct=200)

    ct_spot_enrich_df, marker_genes = enrichment_mia(spa_adata,
                                                     sc_adata,
                                                     anno_name=anno_key,
                                                     q=q)
    # print(f"The number of marker genes: {len(marker_genes)}")
    if len(marker_genes) >= 3000:
        spa_adata = spa_adata[:, marker_genes]
        spa_adata = spa_adata[:, extract_hvgs(spa_adata.copy(),
                                              n_genes=3000)]
        sc_adata_total = sc_adata_total[:, spa_adata.var_names]
        sc_adata = sc_adata[:, spa_adata.var_names]
        marker_genes = spa_adata.var_names

    # Intergration and pp
    sc_adata, spa_adata = integration_pp_adata(sc_adata, spa_adata)

    # Add enrichment information
    spa_adata.enrichment_score_df = ct_spot_enrich_df

    # Generate pseudo spots and their cell type compositions and Merge them 
    # with real spots.
    pseudo_num_rate = int(np.ceil(num_pseudo / spa_adata.shape[0]))
    spa_X, pseudo_X, pseudo_df_composition = \
        generate_merge_initial(sc_adata,
                               spa_adata,
                               marker_genes,
                               anno_name=anno_key,
                               pseudo_num_rate=pseudo_num_rate,
                               abs_relative_rate=abs_relative_rate,
                               remove_platform=remove_platform)

    # ***************Initial training process**********************
    # Predict by MLP
    MLP = mlp_pred_train(pseudo_X,
                         pseudo_df_composition.values.transpose(),
                         num_epochs=num_epochs,
                         lr=lr,
                         batch_size=batch_size,
                         device=device)
    prediction = mlp_pred_final(MLP,
                                spa_X,
                                device=device)
    prediction = pd.DataFrame(prediction,
                              index=spa_adata.obs_names,
                              columns=pseudo_df_composition.index)

    # **************Iterative training process**********************
    # Ensure pseudo num rate.
    prediction_pre = prediction.copy()
    # Training
    for i in range(num_iter - 1):
        # resample scRNA-seq
        print("\n")
        print(f"*************** Iteration: \t{i + 2} ***************")
        sc_adata = simplify_refer(sc_adata=sc_adata_total.copy(),
                                  anno_key=anno_key,
                                  cell_per_ct=200)
        spa_X, pseudo_X, pseudo_df_composition = \
            generate_merge_iter(sc_adata=sc_adata, spa_adata=spa_adata, pre_deconvolution=prediction,
                                marker_genes=marker_genes, anno_name=anno_key, min_cells=min_cells, max_cells=max_cells,
                                abs_relative_rate=abs_relative_rate, pseudo_num_rate_iter=pseudo_num_rate,
                                remove_platform=remove_platform)
        # Iterative traning
        MLP = mlp_train_iter(pseudo_X,
                             pseudo_df_composition.values.transpose(),
                             MLP,
                             batch_size=batch_size,
                             num_epochs=num_epochs,
                             device=device)
        prediction = mlp_pred_final(MLP, spa_X, device=device)
        prediction = pd.DataFrame(prediction,
                                  index=spa_adata.obs_names,
                                  columns=pseudo_df_composition.index)
        # Check iteration error. If the MAE between two iterations is less than
        # setting threshold values, exit.
        error = mean_absolute_error(prediction, prediction_pre)
        prediction_pre = prediction.copy()
        if error < iter_cutoff:
            print("Iteration Error ", error, " < ", "Setting Error ",
                  iter_cutoff)
            print("Exit!")
            break
    spa_adata.obsm['deconvolution_mlp'] = prediction
    return prediction
