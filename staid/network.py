import torch
import torch.nn as nn
from torch.nn import Linear

torch.set_default_tensor_type(torch.FloatTensor)


class GAT_pred(nn.Module):
    def __init__(self, dim_input, dim_out, dim=None, dropout=0.05):
        super().__init__()
        if dim is None:
            dim = [512, 256, 128]
        self.con1 = Linear(dim_input,
                           dim_input)
        self.fl1 = nn.Linear(dim_input, dim[0])
        self.fl2 = nn.Linear(dim[0], dim[1])
        self.relu1 = nn.Tanh()
        self.drop1 = nn.Dropout(0.05)
        self.fl3 = Linear(dim[1], dim[2])
        self.fl4 = Linear(dim[2], dim_out)
        self.soft = nn.Softmax(dim=1)

    def forward(self, inputs, edge_index):
        x = self.con1(inputs, edge_index)
        x = self.relu1(x)
        x = self.fl1(x)
        x = self.relu1(x)
        x = self.fl2(x)
        x = self.relu1(x)
        x = self.fl4(x)
        x = self.soft(x)

        return x


# class MLP_pred(nn.Module):
#     def __init__(self, dim_input, dim_out, dim=None, dropout=0.05):
#         super().__init__()
#         if dim is None:
#             dim = [512, 256, 128]
#         self.norm = nn.BatchNorm1d(dim_input)
#         self.fl1 = nn.Linear(dim_input, dim[0])
#         self.fl2 = nn.Linear(dim[0], dim[1])
#         self.act = nn.LeakyReLU()
#         self.fl3 = Linear(dim[1], dim_out)
#         self.soft = nn.Softmax(dim=1)
#
#     def forward(self, inputs):
#         x = inputs
#         x = self.fl1(x)
#         x = self.act(x)
#         x = self.fl2(x)
#         x = self.act(x)
#         x = self.fl3(x)
#         x = self.soft(x)
#
#         return x


class MLP_pred(nn.Module):
    def __init__(self, dim_input, dim_out, dim=None, dropout=0.02):
        super(MLP_pred, self).__init__()
        if dim is None:
            dim = [512, 256, 256]

        self.layer1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_input, dim[0]),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(dim[0], dim[1]),
            nn.LeakyReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(dim[1], dim[2]),
            nn.LeakyReLU(),
        )

        self.out_layer = nn.Linear(dim[2], dim_out)
        self.softmax = nn.Softmax(dim=1)

        self.shortcut1 = nn.Linear(dim_input, dim[0]) if dim_input != dim[0] else nn.Identity()
        self.shortcut2 = nn.Linear(dim[0], dim[1]) if dim[0] != dim[1] else nn.Identity()

    def forward(self, inputs):
        x = inputs
        x = self.layer1(x)
        residual1 = self.shortcut1(inputs)
        x = self.layer2(x + residual1)
        residual2 = self.shortcut2(residual1)  # 使用 residual1
        x = self.layer3(x + residual2)
        x = self.out_layer(x)
        x = self.softmax(x)
        return x


class MLP_autoencoder(nn.Module):
    def __init__(self, dim_input, hidden_dims=None, dropout=0.05):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024]
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=dim_input,
                                                                    nhead=1,
                                                                    dim_feedforward=512,
                                                                    dropout=0)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        self.fl1 = nn.Linear(dim_input, hidden_dims[0])
        self.fl2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fl3 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fl4 = nn.Linear(hidden_dims[0], dim_input)
        self.act = nn.LeakyReLU()
        self.act2 = nn.Softmax(dim=1)
        self.batch = nn.BatchNorm1d(dim_input)

    def encoder(self, inputs):
        x = inputs
        x = self.fl1(x)
        x = self.act(x)
        return x

    def decoder(self, emb):
        x = emb
        x = self.fl4(x)

        return x

    def forward(self, inputs):
        emb = self.encoder(inputs)
        x = self.decoder(emb)
        return x
