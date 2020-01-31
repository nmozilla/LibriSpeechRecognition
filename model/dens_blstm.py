import torch

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from util.functions import TimeDistributed, CreateOnehotVariable
import numpy as np


class _maxpool2d(nn.MaxPool2d):
    def __init___(self, kernel_size, stride=None, padding=0, **kwargs):
        super().__init__(kernel_size, stride=stride, padding=padding, **kwargs)

    def forward(self, x):
        return super().forward(x), None


class _lstm(nn.LSTM):
    def __init___(self, input_size, hidden_size, num_layes, **kwargs):
        super().__init__(input_size, hidden_size, num_layes, **kwargs)

    def forward(self, x):
        return super().forward(x)[0]


class BLSTM(nn.Module):
    def __init__(self, input_feature_dim, lstm_hidden_dim, blstm_layer, lstm_output_dim, use_gpu, dropout_rate=0.7, rnn_unit='LSTM', bidirectional=True, **kwargs):
        super(BLSTM, self).__init__()
        
        feature_sizes = ([input_feature_dim] * blstm_layer)[:blstm_layer] + [lstm_output_dim]
        print('feature_sizes',feature_sizes)
        print('blstm_layer',blstm_layer)
        self.conv = nn.Conv1d(input_feature_dim, input_feature_dim, 30)
        self.maxpool = nn.MaxPool2d((2,1), stride=(2, 1), padding=0)
        self.fc = nn.Linear(lstm_output_dim, lstm_output_dim)
        self.lstm_layers = []
        for i in range(blstm_layer):
            l = _lstm(feature_sizes[i], feature_sizes[i + 1],1, batch_first=True)#,dropout=0.5)
            for param in l.parameters():
                if len(param.shape) >= 2:
                   torch.nn.init.kaiming_uniform_(param.data)
                else:
                    torch.nn.init.uniform_(param.data)
            self.lstm_layers.append(l)
            #if i%8==0:
            self.lstm_layers.append(self.maxpool)
        self.lstm_layers.pop(-1)
        self.lstm_layers.append(nn.ReLU())
        self.lstm_layers.append(self.fc)
        print('lstm_layers', self.lstm_layers)
        self.rnn = nn.Sequential(*self.lstm_layers)
        print('sequentioal', self.rnn)
        if use_gpu:
            self.rnn = self.rnn.cuda()
            self.conv = self.conv.cuda()

    def forward(self, x):
        inputs = x.contiguous().permute((0, 2, 1))
        conv1_out = self.conv(inputs)
        conv2_out = self.conv(conv1_out)
        #conv3_out = self.conv(conv2_out)
        rnn_in = conv2_out.contiguous().permute((0, 2, 1))
        blstm_output = self.rnn.forward(rnn_in)
        return blstm_output

    def ___forward(self, input_x, **kwargs):
        x = input_x.contiguous()#.permute((1, 0, 2))
        for l in self.lstm_layers:
            x, _ = l(x)
        return x
