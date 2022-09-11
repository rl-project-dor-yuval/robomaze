import torch
import torch.nn as nn


class ImitationNN(nn.Module):
    def __init__(self, input_size, output_size, layers=(400, 300)):
        super(ImitationNN, self).__init__()

        ff_layers = [nn.Linear(input_size, layers[0]),
                     nn.Tanh()]

        for i in range(len(layers) - 1):
            ff_layers.append(nn.Linear(layers[i], layers[i + 1]))
            ff_layers.append(nn.Tanh())

        ff_layers.append(nn.Linear(layers[-1], output_size))

        self.ff_layers = nn.Sequential(*ff_layers)

    def forward(self, x):
        return self.ff_layers(x)
