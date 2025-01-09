import numpy as np
import torch
import torch.utils.data as Data

from staid.GAN_modi_nn import *
from plot.plot_GNA_training import *


def GAN_train(X, labels, epoch=50):
    # Convert data
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(labels.astype(np.float32))
    y = y.reshape(len(y), 1)

    # Define datasets
    Dataset_D1 = Data.TensorDataset(X[y[:, 0] == 1, :],
                                    y[y[:, 0] == 1, :])
    Dataset_D0 = Data.TensorDataset(X[y[:, 0] == 0, :],
                                    y[y[:, 0] == 0, :])
    Dataset_G = Data.TensorDataset(X[y[:, 0] == 0, :],
                                   torch.ones_like(y[y[:, 0] == 0, :]))
    train_loader_D1 = Data.DataLoader(
        dataset=Dataset_D1,
        batch_size=64,
        shuffle=True,
        num_workers=2)
    train_loader_D0 = Data.DataLoader(
        dataset=Dataset_D0,
        batch_size=64,
        shuffle=True,
        num_workers=2)
    train_loader_G = Data.DataLoader(
        dataset=Dataset_G,
        batch_size=64,
        shuffle=True,
        num_workers=2)

    # Define generator and discriminator
    G = Generator(X.shape[1])
    D = Discriminator(X.shape[1])

    # Define train loss lists of Generator and Discriminator
    train_loss_D1 = []
    train_loss_D0 = []
    train_loss_G = []
    train_loss_D = []

    # Training process
    for epoch in range(50):
        # Train Discrimetor with label 1(pseudo spots)
        for step_D1, (b_X_D1, b_y_D1) in enumerate(train_loader_D1):
            D.train(b_X_D1, b_y_D1)

        # Train Discrimetor with label 0(real spots)
        for step_D0, (b_X_D0, b_y_D0) in enumerate(train_loader_D0):
            D.train(G.forward(b_X_D0).detach(), b_y_D0)
            # Train Discrimetor with label 0(real spots)
        for step_D0, (b_X_D0, b_y_D0) in enumerate(train_loader_D0):
            D.train(G.forward(b_X_D0).detach(), b_y_D0)

        # Train Generator
        for step_G, (b_X_G, b_y_G) in enumerate(train_loader_G):
            G.train(D, b_X_G, b_y_G)

        train_loss_D1.append(D.loss_function(D.forward(b_X_D1),
                                             b_y_D1).detach().numpy())
        train_loss_D0.append(D.loss_function(D.forward(G.forward(b_X_D0).detach()),
                                             b_y_D0).detach().numpy())
        train_loss_G.append(D.loss_function(D.forward(G.forward(b_X_G)),
                                            b_y_G).detach().numpy())
        train_loss_D.append(np.mean([train_loss_D0[-1:], train_loss_D1[-1:]]))
        print("GAN epoch:  ", epoch, "\t; Loss_D:  ", train_loss_D[-1:][0],
              "\t; Loss_G:  ", train_loss_G[-1:][0])

    # Visualize training process
    plot_GNA_training(train_loss_D, train_loss_G)

    new_spa_X = G.forward(X[y[:, 0] == 0, :]).detach().numpy()

    plot_difference_tsne(new_spa_X, pseudo_X)
