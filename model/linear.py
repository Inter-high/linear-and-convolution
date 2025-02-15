import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.layers = self.get_layer()

    def get_layer(self):
        layers = nn.ModuleList()

        if self.hidden_sizes:
            layers.append(nn.Linear(self.input_size, self.hidden_sizes[0]))
            layers.append(nn.ReLU())
            if len(self.hidden_sizes) >= 2:
                for idx in range(1, len(self.hidden_sizes)):
                    layers.append(nn.Linear(self.hidden_sizes[idx-1], self.hidden_sizes[idx]))
                    layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_sizes[-1], self.num_classes))
        else:
            layers.append(nn.Linear(self.input_size, self.num_classes))

        return layers
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    