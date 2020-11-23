# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:59:38 2020

@author: msplsh
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride = L//2, bias=False)
        
    def forward(self, x):

        mixture_w = F.relu(self.conv1d_U(x)) #[M, N, K]
        return mixture_w
    
class DepthwiseConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel, padding=0, dilation=1, skip=128):
        super(DepthwiseConv, self).__init__()
        
        self.skip = skip
        
        self.conv1d = nn.Conv1d(in_channel, hidden_channel, 1) 
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel_size= kernel, dilation= dilation, groups=hidden_channel, padding=padding)

        self.pconv1d_res = nn.Conv1d(hidden_channel, in_channel, 1)
        self.pconv1d_skip = nn.Conv1d(hidden_channel, skip, 1) 
        
        self.nonlinear1 = nn.PReLU()
        self.nonlinear2 = nn.PReLU()

        self.norm1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        self.norm2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
        
    def forward(self, x):
        
        out = self.norm1(self.nonlinear1(self.conv1d(x))) # [M, 128, K] -> [M, 512, K]
        out = self.norm2(self.nonlinear2(self.dconv1d(out)))  # [M, 512, K] -> [M, 512, K]

        residual = self.pconv1d_res(out)  # [M, 512, K] -> [M, 128, K]

        if self.skip:
            skip = self.pconv1d_skip(out) # [M, 512, K] -> [M, 128, K]
            return residual, skip
        else:
            return residual
        
        
class Separator(nn.Module):
    def __init__(self, in_channel, out_channel, bn_channel, hidden_channel, kernel, layer, stack, skip):
        super(Separator, self).__init__()
        
        self.skip = skip
        
        self.LN = nn.GroupNorm(1, in_channel, eps=1e-08) # global normalization [M, N, num_mic, K] 
        self.BN = nn.Conv1d(in_channel, bn_channel, 1) # bottleneck channel [M, B, num_mic, K] 
        
        self.TCN = nn.ModuleList([])
        for i in range(stack):
            for j in range(layer):
                self.TCN.append(DepthwiseConv(bn_channel, hidden_channel, kernel, dilation=2**j, padding=2**j, skip=skip))
                
        if self.skip:
            self.out = nn.Sequential(nn.PReLU(), nn.Conv1d(skip, out_channel, 1))
        else:
            self.out = nn.Sequential(nn.PReLU(), nn.Conv1d(bn_channel, out_channel, 1))               

    def forward(self, x):
        
        out = self.BN(self.LN(x))
        
        if self.skip:
            
            skip_connection = 0. 
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](out)
                out = out + residual
                skip_connection = skip_connection + skip
                
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](out)
                out = residual + out
        
        if self.skip:
            out = self.out(skip_connection)
        else:
            out = self.out(out)
        return out
    
  
    
class ConvTasNet(nn.Module):
    
    """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            Sc: Number of channels in skip blocks
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            #norm_type: BN, gLN, cLN
            #causal: causal or non-causal
            #mask_nonlinear: use which non-linear function to generate mask
    """

    def __init__(self, N=512, L=32, B=128, Sc=128, H=512, X=8, R=3, P=3, C=2):
        super(ConvTasNet, self).__init__()
        
        self.C = C
        
        self.N = N
        self.B = B
        self.Sc = Sc
        self.H = H
        
        self.L = L
        
        self.P = P
        self.X = X
        self.R = R
        
        self.encoder = Encoder(L, N)
        self.separator = Separator(in_channel=N, out_channel=N*C, bn_channel=B, hidden_channel=H, kernel=P, layer=X, stack=R, skip=Sc)
        self.decoder = nn.ConvTranspose1d(N, 1, L, bias=False, stride=L//2)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.L - (self.L//2 + nsample % self.L) % self.L
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.L//2)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest        
        
    def forward(self, x):
    
        # padding
        M = x.shape[0]
        out, rest = self.pad_signal(x)
        enc_out = self.encoder(out) # [M, N, K]
        
        out = self.separator(enc_out)

        masks = torch.sigmoid(out.view(M, self.C, self.N, -1)) #[M, 2, N, K]
        out = enc_out.unsqueeze(1) * masks
        
        out = self.decoder(out.view(M*self.C, self.N, -1))
        out = out[:,:,self.L//2:-(rest+self.L//2)].contiguous()  # B*C, 1, L
        out = out.view(M, self.C, -1)

        return out
        


if __name__ == "__main__":
    
    input = torch.rand(3,64000) #[batch, T]
    
    model = ConvTasNet()
    out = model(input) 
    print(out.shape)
    
    k = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of parameters:', k)