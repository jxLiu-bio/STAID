import torch
import torch.nn as nn
torch.set_default_tensor_type(torch.FloatTensor)


class MLP_pred(nn.Module):
    def __init__(self, dim_input, dim_out, dim=[512, 512, 512], dropout=0.01):
        super(MLP_pred, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(dim_input, dim[2]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.05),
        )
        self.out_layer = nn.Sequential(nn.Linear(dim[2],
                                                 dim_out),
                                       nn.Softmax(dim=1))

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = self.out_layer(x)

        return x


class MLP_autoencoder(nn.Module):
    def __init__(self, dim_input, hidden_dims=None, dropout=0.05):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 1024]
        self.fl1 = nn.Linear(dim_input, hidden_dims[0])
        self.fl2 = nn.Linear(hidden_dims[0], dim_input)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(dropout)

    def encoder(self, inputs):
        x = self.fl1(inputs)
        x = self.act(x)

        return x

    def decoder(self, emb):
        x = emb
        x = self.drop(x)
        x = self.fl2(x)
        x = self.act(x)

        return x

    def forward(self, inputs):
        emb = self.encoder(inputs)
        x = self.decoder(emb)
        return x