import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, dropout, batch_norm):
        super(DenseBlock, self).__init__()

        layers = nn.ModuleList()
        layers.append(nn.Linear(num_inputs, num_outputs))
        if batch_norm:
            layers.append(nn.BatchNorm1d(num_outputs))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, dense_layers, output_size, num_inputs, dropout, skip_conn, batch_norm):
        super(DenseNet, self).__init__()
        self.skip_conn = skip_conn
        self.linear_in = DenseBlock(num_inputs, dense_layers[0], dropout, batch_norm)

        self.dense_list = nn.ModuleList([DenseBlock(dense_layers[i], dense_layers[i+1], dropout, batch_norm) for i in range(len(dense_layers)-1)])
        self.linear = nn.Sequential(*self.dense_list)
        
        self.linear_out = nn.Linear(dense_layers[-1], output_size)

    def forward(self, x):
        o = self.linear_in(x)
        o = self.linear(o)
        o = self.linear_out(o)
        # Todo skip connection
        return o