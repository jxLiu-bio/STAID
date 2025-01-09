import torch
import torch.nn as nn


class MLP_pre(nn.Module):
    def __init__(self, dim_input, dim_out, lr):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_input, 512),
            # nn.Dropout(p=0.2),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            # nn.LeakyReLU(),
            nn.Linear(128, 64),
            # nn.LeakyReLU(),
            nn.Linear(64, dim_out),
            nn.Softmax())
        # Define loss function
        self.loss_function = nn.MSELoss()
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.scheduler = ExponentialLR(self.optimizer, 
        #                                gamma=0.1)
        # Define counter and progress when train
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    # def train(self, inputs, targets):
    #     outputs = self.forward(inputs)
    #     loss = self.loss_function(outputs, targets)

    #     # Record training log
    #     self.counter += 1
    #     if self.counter % 100 == 0:
    #         self.progress.append(loss.item())

    #     # Update the parameters
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #     return loss.item()
