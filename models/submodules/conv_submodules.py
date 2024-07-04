import torch
import torch.nn as nn

class DilatedConv(nn.Module): #temporal layer
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, batch_norm):
        super(DilatedConv, self).__init__()
        layers = nn.ModuleList()
        layers.append(nn.ConstantPad1d(padding, 0)) # Conv1D gives erorr with non symmetric padding
        layers.append(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=0, dilation=dilation))
        if batch_norm:
            layers.append(nn.BatchNorm1d(n_outputs))
        layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalConv(nn.Module): #fully interconnected tcn
    def __init__(self, num_channels, kernel_size, skip_conn, batch_norm, vital_signs, max_pooling=[2, 2, 2, 2]):
        super(TemporalConv, self).__init__()
        self.skip_conn = skip_conn
        self.layers = nn.ModuleList()
        num_levels = len(num_channels)

        assert len(max_pooling) == num_levels, "max_pooling must have the same length as num_channels"

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = 1 if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            self.layers.append(DilatedConv(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(int((kernel_size-1)*dilation_size), 0), batch_norm=batch_norm))
            if max_pooling[i] != 0:
                self.layers.append(nn.MaxPool1d(max_pooling[i]))
            
        self.network = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.skip_conn:
            o = self.layers[0](x)
            s = o
            for i, layer in enumerate(self.layers[1:]):
                # We apply residuals on every other layer as in Imagenet
                # Skipping first layer
                o = layer(o)
                if (i % 2):
                    o = o + s
                    s = o
            return o
        else:
            return self.network(x)


class TemporalConvNet(nn.Module):
    def __init__(self, time_samples, vital_signs, num_channels, kernel_size, skip_conn, batch_norm, max_pooling):
        super(TemporalConvNet, self).__init__()
        self.seq_lenght = time_samples
        self.vital_signs = vital_signs
        
        self.tcn = nn.ModuleList([TemporalConv(num_channels, kernel_size, skip_conn, batch_norm, vital_signs=vital_signs, max_pooling=max_pooling) for i in range(vital_signs)])

    def forward(self, x):
        tcn_out = []
        for i, tcn_i in enumerate(self.tcn):
            o = tcn_i(x[:,i,:].reshape(-1,1,self.seq_lenght))
            tcn_out.append(o)
        o = torch.cat(tcn_out, dim=1)
        return o