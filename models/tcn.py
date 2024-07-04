import torch.nn as nn
from models.submodules import conv_submodules, dense_submodules


class MultiTCN(nn.Module):
    def __init__(self, time_samples, output_size, vital_signs, num_channels, device, dtype, dense_layers=[64], kernel_size=3, dropout=0.2, skip_conn=False, batch_norm=False, max_pooling=[2, 2, 2, 2]):
        super(MultiTCN, self).__init__()
        self.device = device
        self.dtype = dtype

        self.TemporalConvNet = conv_submodules.TemporalConvNet(time_samples, vital_signs, num_channels, kernel_size, skip_conn, batch_norm, max_pooling)
        self.seq_length = time_samples

        # Calculate the sequence length after max pooling 
        # https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html using default values (stride=None, padding=0, dilation=1))
        for i in range(len(max_pooling)):
            if max_pooling[i] != 0:
                self.seq_length = int((self.seq_length - max_pooling[i]) / max_pooling[i]) + 1

        dense_inputs = self.seq_length * num_channels[-1] * vital_signs
        self.DenseNet = dense_submodules.DenseNet(dense_layers, output_size, dense_inputs, dropout, skip_conn, batch_norm)

    def forward(self, data, ids=None):
        o = self.TemporalConvNet(data).flatten(1)

        o = self.DenseNet(o)
        return o