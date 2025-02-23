import torch
import torch.nn as nn
import math

class ImCA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ImCA_Block, self).__init__()
        
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu   = nn.ReLU()
        self.bn     = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
        # batch_size, c, h, w
        _, _, h, w = x.size()
        x_h = torch.mean(x, dim = 3, keepdim = True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim = 2, keepdim = True)
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
    
        x_h = torch.max(out, dim = 3, keepdim = True).values.permute(0, 1, 3, 2)
        x_w = torch.max(out, dim = 2, keepdim = True).values
        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
        out = out * s_h.expand_as(out) * s_w.expand_as(out)
        return out
    
if __name__ == '__main__':
    input = torch.randn(1, 64, 56, 56)
    net = ImCA_Block(channel=64)
    output = net(input)
    print(output.shape)