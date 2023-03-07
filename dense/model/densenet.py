from torch import nn
import torch


class ResBlock(nn.Module):
    def __init__(self, n_chan=64, out_chan=64, padding=1, dilation=(1,1), groups=64):
        super(ResBlock, self).__init__()
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3, padding=padding, dilation=dilation, groups = groups),
            # nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=padding,dilation=dilation, groups = groups),
            # nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, out_chan, kernel_size=1,padding=0),
        )
    def forward(self, inputs):
        # return self.rs1(inputs) + inputs
        return torch.cat([self.rs1(inputs),inputs],dim=1)

class UpsResBlock(nn.Module):
    def __init__(self, n_chan=64, out_chan=64, padding=(2,3), dilation=(2,3), groups=64):
        super(UpsResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chan, out_chan, kernel_size=1,padding=0)
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3, padding=padding, dilation=dilation, groups = groups),
            nn.Conv2d(n_chan, n_chan*2, kernel_size=1,padding=0),
            # nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=padding,dilation=dilation, groups = groups*2),
            # nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, out_chan, kernel_size=1,padding=0),
        )
    def forward(self, inputs):
        return self.rs1(inputs) + self.conv(inputs)

class DspResBlock(nn.Module):
    def __init__(self, n_chan=64, out_chan=64, padding=(2,3), dilation=(2,3), groups=64):
        super(DspResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chan*2, out_chan, kernel_size=1,padding=0)
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3, padding=padding, dilation=dilation, groups = groups),
            nn.Conv2d(n_chan*2, n_chan, kernel_size=1,padding=0),
            # nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=padding,dilation=dilation, groups = groups),
            # nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, out_chan, kernel_size=1,padding=0),
        )
    def forward(self, inputs):
        return self.rs1(inputs) + self.conv(inputs)


class DenseNet(nn.Module):
    def __init__(self, in_planes, planes, stride=1, n_chan=64):
        super(DenseNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, n_chan, kernel_size=3, dilation = (1,1),padding=1, bias=False)
        self.rs1=ResBlock(n_chan, out_chan=n_chan, padding=1, dilation=(1,1), groups=n_chan)
        self.rs2=ResBlock(n_chan*2, out_chan=n_chan*2, padding=(2,3), dilation=(2,3), groups=n_chan*2)
        self.rs3=UpsResBlock(n_chan*4, out_chan=n_chan, padding=(3,6), dilation=(3,6), groups=n_chan*4)

        self.rs4=ResBlock(n_chan, out_chan=n_chan, padding=1, dilation=(1,1), groups=n_chan)
        self.rs5=ResBlock(n_chan*2, out_chan=n_chan*2, padding=(2,3), dilation=(2,3), groups=n_chan*2)
        self.rs6=UpsResBlock(n_chan*4, out_chan=n_chan*2, padding=(3,6), dilation=(3,6), groups=n_chan*4)

        self.rs7=ResBlock(n_chan*2, out_chan=n_chan*2,padding=(3,6), dilation=(3,6), groups=n_chan*2)
        self.rs8=ResBlock(n_chan*4, out_chan=n_chan*4,padding=(2,3), dilation=(2,3), groups=n_chan*4)
        self.rs9=DspResBlock(n_chan*4,out_chan=n_chan*2, padding=1, dilation=1, groups=n_chan*4)

        self.rs10=ResBlock(n_chan*2, out_chan=n_chan*2,padding=(3,6), dilation=(3,6), groups=n_chan*2)
        self.rs11=ResBlock(n_chan*4, out_chan=n_chan*4,padding=(2,3), dilation=(2,3), groups=n_chan*4)
        self.rs12=DspResBlock(n_chan*4,out_chan=n_chan*2, padding=1, dilation=1, groups=n_chan*4)
        self.conv_end = nn.Conv2d(n_chan*2, planes, kernel_size=1, dilation=(1,1),stride=1, padding=0, bias=False)

    def forward(self, inputs):
        out1 = self.conv1(inputs)
        out_res1 = self.rs1(out1)
        out_res2 = self.rs2(out_res1)
        out_res3 = self.rs3(out_res2)
        out_res4 = self.rs4(out_res3)
        out_res5 = self.rs5(out_res4)
        out_res6 = self.rs6(out_res5)
        out_res7 = self.rs7(out_res6)
        out_res8 = self.rs8(out_res7)
        out_res9 = self.rs9(out_res8)
        out_res10 = self.rs10(out_res9)
        out_res11 = self.rs11(out_res10)
        out_res12 = self.rs12(out_res11)
        output = torch.sigmoid(self.conv_end(out_res12))
        return output

