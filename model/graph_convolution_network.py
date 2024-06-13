import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


class graph_convolution(nn.Module):
    def __init__(self, in_features, out_features, node_n = 21, seq_len = 40, bias=True):
        super(graph_convolution, self).__init__()
        
        self.temporal_graph_weights = Parameter(torch.FloatTensor(seq_len, seq_len))
        self.feature_weights = Parameter(torch.FloatTensor(in_features, out_features))
        self.spatial_graph_weights = Parameter(torch.FloatTensor(node_n, node_n))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(seq_len))
            
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.spatial_graph_weights.size(1))
        self.feature_weights.data.uniform_(-stdv, stdv)
        self.temporal_graph_weights.data.uniform_(-stdv, stdv)
        self.spatial_graph_weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input):
        y = torch.matmul(input, self.temporal_graph_weights)
        y = torch.matmul(y.permute(0, 3, 2, 1), self.feature_weights)
        y = torch.matmul(self.spatial_graph_weights, y).permute(0, 3, 2, 1).contiguous()
        
        if self.bias is not None:
            return (y + self.bias)
        else:
            return y

            
class residual_graph_convolution(nn.Module):
    def __init__(self, features, node_n=21, seq_len = 40, bias=True, p_dropout=0.3):
        super(residual_graph_convolution, self).__init__()
        
        self.gcn = graph_convolution(features, features, node_n=node_n, seq_len=seq_len, bias=bias)        
        self.ln = nn.LayerNorm([features, node_n, seq_len], elementwise_affine=True)                      
        self.act_f = nn.Tanh()
        self.dropout = nn.Dropout(p_dropout)        
        
    def forward(self, x):

        y = self.gcn(x)
        y = self.ln(y)
        y = self.act_f(y)
        y = self.dropout(y)
        
        return y + x

        
class graph_convolution_network(nn.Module):
    def __init__(self, in_features, latent_features, node_n=21, seq_len=40, p_dropout=0.3, residual_gcns_num=1):
        super(graph_convolution_network, self).__init__()
        self.residual_gcns_num = residual_gcns_num
        self.seq_len = seq_len
        
        self.start_gcn = graph_convolution(in_features=in_features, out_features=latent_features, node_n=node_n, seq_len=seq_len)
        
        self.residual_gcns = []
        for i in range(residual_gcns_num):
            self.residual_gcns.append(residual_graph_convolution(features=latent_features, node_n=node_n, seq_len=seq_len*2, p_dropout=p_dropout))        
        self.residual_gcns = nn.ModuleList(self.residual_gcns)
                
    def forward(self, x):
        y = self.start_gcn(x)
        
        y = torch.cat((y, y), dim=3)
        for i in range(self.residual_gcns_num):
            y = self.residual_gcns[i](y)
        y = y[:, :, :, :self.seq_len]
                
        return y