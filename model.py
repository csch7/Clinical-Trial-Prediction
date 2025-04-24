from torch import nn

class TrialPred(nn.Module):

    def __init__(self, input_dim, hidden_dim, hidden_layers, dropout_rate = 0.3, output_dim = 1, layer_act = nn.ReLU, 
                 output_act = nn.Sigmoid(), include_last = False):
        super(TrialPred, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = hidden_layers
        self.output_dim = output_dim
        self.act = layer_act
        self.output_act = output_act

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(layer_act())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(1, self.num_hidden-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(layer_act())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(self.hidden_dim, output_dim))
        if include_last:
            layers.append(output_act)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
