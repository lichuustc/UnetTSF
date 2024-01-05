import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class block_model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, input_channels, input_len, out_len, individual):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len
        self.individual = individual

        if self.individual:
            self.Linear_channel = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_channel.append(nn.Linear(self.input_len, self.out_len))
        else:
            self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),x.size(1),self.out_len],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,i,:] = self.Linear_channel[i](x[:,i,:])
        else:
            output = self.Linear_channel(x)
        #output = self.ln(output)
        #output = self.relu(output)
        return output # [Batch, Channel, Output length]


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.input_channels = configs.enc_in
        self.input_len = configs.seq_len
        self.out_len = configs.pred_len
        self.individual = configs.individual
        # 下采样设定
        n1 = 1
        filters = [n1, n1 * 2, n1 * 4, n1 * 8]

        # 最大池化层
        self.Maxpool1 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool2 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool3 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.Maxpool4 = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)

        # 左边特征提取层
        self.down_block1 = block_model(self.input_channels, int(self.input_len / filters[0]), int(self.out_len / filters[0]), self.individual)
        self.down_block2 = block_model(self.input_channels, int(self.input_len / filters[1]), int(self.out_len / filters[1]), self.individual)
        self.down_block3 = block_model(self.input_channels, int(self.input_len / filters[2]), int(self.out_len / filters[2]), self.individual)
        self.down_block4 = block_model(self.input_channels, int(self.input_len / filters[3]), int(self.out_len / filters[3]), self.individual)


        # 右边特征融合层
        self.Up4 = nn.Upsample(scale_factor=(2), mode='nearest')
        self.up_block3 = block_model(self.input_channels, int(self.out_len / filters[2]), int(self.out_len / filters[2]), self.individual)

        self.Up3 = nn.Upsample(scale_factor=(2), mode='nearest')
        self.up_block2 = block_model(self.input_channels, int(self.out_len / filters[1]), int(self.out_len / filters[1]), self.individual)

        self.Up2 = nn.Upsample(scale_factor=(2), mode='nearest')
        self.up_block1 = block_model(self.input_channels, int(self.out_len / filters[0]), int(self.out_len / filters[0]), self.individual)

        #self.linear_out = nn.Linear(self.out_len * 2, self.out_len)

    def forward(self, x):
        x = x.permute(0,2,1)
        e1 = self.down_block1(x)

        e2 = self.Maxpool1(e1)#48
        e2 = self.down_block2(e2)

        e3 = self.Maxpool2(e2)#24
        e3 = self.down_block3(e3)

        e4 = self.Maxpool3(e3)#12
        e4 = self.down_block4(e4)


        d4 = self.Up4(e4)#24
        #d4 = torch.cat((e3, d4), dim=2)  # 将e3特征图与d4特征图横向拼接
        d4 = d4 + e3
        d4 = self.up_block3(d4)#24

        d3 = self.Up3(d4)#48
        #d3 = torch.cat((e2, d3), dim=2)  # 将e2特征图与d3特征图横向拼接
        d3=d3 + e2
        d3 = self.up_block2(d3)#48

        d2 = self.Up2(d3)#96
        #d2 = torch.cat((e1, d2), dim=2)  # 将e1特征图与d1特征图横向拼接
        d2 = d2 + e1
        out = self.up_block1(d2)#96

        #out = self.linear_out(d2)

        return out.permute(0,2,1)