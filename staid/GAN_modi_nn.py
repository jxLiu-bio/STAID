import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, dim_output):
        # Initialize
        super().__init__()
        # Define model
        self.model = nn.Sequential(
            nn.Linear(dim_output, dim_output),
            # nn.T(),
            # nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(dim_output, dim_output),
            # nn.T(),
            # nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(dim_output, dim_output),
            # nn.T(),
            # nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(dim_output, 1024),
            nn.LeakyReLU(),
            # nn.Linear(64, 16),
            # nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid())
        # Define loss function
        self.loss_function = nn.BCELoss()
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # Define counter and progress when train
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, inputs, targets):
        outputs = self.forward(inputs)
        loss = self.loss_function(outputs, targets)

        # Record training log
        self.counter += 1
        if self.counter % 100 == 0:
            self.progress.append(loss.item())

        # Update the parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Generator(nn.Module):
    def __init__(self, dim_input):
        # Initialize
        super().__init__()
        # Define neural networks to modificate pseudo_spot
        self.model = nn.Sequential(
            nn.Linear(dim_input, 500),
            nn.LeakyReLU(),
            nn.Linear(500, dim_input)
        )
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        # Counter
        self.counter = 0
        self.progress = []

    def forward(self, inputs):
        return self.model(inputs)

    def train(self, D, inputs, targets):
        # The out put of neural networks
        g_output = self.forward(inputs)
        # The outputs of Discriminator
        d_output = D.forward(g_output)
        # Calculate loss
        loss = D.loss_function(d_output, targets)

        # Record
        self.counter += 1
        if self.counter % 100 == 0:
            self.progress.append(loss.item())

        # Update the parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
