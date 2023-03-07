import torch
from torch import nn
import numpy as np
import time
from scipy.linalg import toeplitz
import math
import scipy.special as scipy
from datetime import datetime

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.sin(1*input)

def ber(Recover_X, Ideal_X):
    batchsize, S, F = Recover_X.shape[0], Recover_X.shape[1], Recover_X.shape[2]
    Ideal_X = Ideal_X.float()
    Recover_X_id = (torch.sign(
        torch.stack([-Recover_X.real, -Recover_X.imag], dim=-1)) + 1) / 2
    Recover_X_id[:, 0, ::2, :] = Ideal_X[:, 0, ::2, :]
    ber = (Ideal_X != Recover_X_id).sum() / (batchsize * (S - 0.5) * F * 2)
    return ber


def calc_ber( outputs, ideal_x):
    RecX=torch.stack([outputs.real, outputs.imag],dim=-1)
    # RecX = RecX.cpu().detach().numpy()
    RecX = (torch.sign(RecX-0.5)+1)/2
    RecX[:,0,::2,:] = ideal_x[:,0,::2,:]
    ber = (RecX != ideal_x).sum()/(RecX.shape[0]*(12-0.5)*24*2)
    # ber =  cnt / ((len(RecX)*(12-0.5)*24*2))
    return ber


    
def bler(H_full, Recover_X, Ideal_X, sigma2):
    batchsize, S, F = H_full.shape[0], H_full.shape[1], H_full.shape[2]
    tmp=torch.linalg.norm(H_full,dim=(3,4))**2
    # print('tmp.shape=',tmp.shape,'sig.shape=',sigma2.shape)
    assert len(tmp.shape)==len(sigma2.shape)
    H_vecnorm = torch.div(1,torch.linalg.norm(H_full,dim=(3,4))**2 + sigma2)
    G = torch.mul(H_vecnorm, torch.linalg.norm(H_full,dim=(3,4))**2)
    eps2 = torch.mul(G, 1-G)
    Recover_Xc = torch.div(torch.mul(-2*torch.sqrt(torch.tensor([2.0]).cuda())*Recover_X,G),eps2)
    llr = torch.clamp(torch.stack([Recover_Xc.real,Recover_Xc.imag],dim=-1),min=-15,max=15)
    Ideal_X = Ideal_X.float()
    ce = -torch.mul(Ideal_X,torch.log(torch.div(pow(2,llr),1+pow(2,llr))))-torch.mul(1-Ideal_X,torch.log(torch.div(1,1+pow(2,llr))))
    ce = torch.mean(ce,dim=(1,2,3))
    ce = torch.reshape(ce,(int(batchsize/25),25))
    ce = torch.mean(ce,1)
    xp = [0,0.415,0.42,0.425,0.43,0.435,0.44,1]
    fp = [0,0,0.0454545454545455,0.21875,0.576923076923077,0.84,1,1]
    bler = np.mean(np.interp(ce.cpu(),xp,fp))
    
    return bler

class FCN(nn.Module):
    def __init__(self, depth=5, in_chan=8, out_chan=2, inter_chan=64, init = False):
        super(FCN,self).__init__()
        layer = []
        # inter_chan_list=[3600,4800,4800,2400]
        bias=True
        if depth==1:
            layer.append(nn.Linear(in_chan, out_chan,bias=False))
        else:
            layer.append(nn.Linear(in_chan, inter_chan,bias=True))
            # layer.append(nn.BatchNorm1d(in_chan, affine=False)) # Without Learnable Parameters
            # layer.append(nn.BatchNorm1d(inter_chan, affine=True)) # With Learnable Parameters
            # layer.append(nn.Tanh())
            layer.append(nn.ReLU())
            for i in range(depth-1):
                layer.append(nn.Linear(inter_chan,inter_chan,bias=bias))
                # layer.append(nn.BatchNorm1d(inter_chan, affine=False)) # Without Learnable Parameters
                # layer.append(nn.BatchNorm1d(inter_chan, affine=True)) # With Learnable Parameters
                # layer.append(nn.Tanh())
                layer.append(nn.ReLU())
            layer.append(nn.Linear(inter_chan,out_chan,bias=bias))
            # layer.append(nn.BatchNorm1d(out_chan, affine=False)) # Without Learnable Parameters
            # layer.append(nn.Tanh())
        self.fcn = nn.Sequential(*layer)

        if init == 0:
            self._init_weights()

    def forward(self, x):
        out = self.fcn(x)
        return out

    def _init_weights(self):
        import torch.nn.init as init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # init.constant_(m.weight, 0)
                init.constant_(m.weight, 1.0/(75*12*24*8*1024))
                if m.bias is not None:
                    init.constant_(m.bias,0)



# import torch.nn as nn


class FCN6(nn.Module):
    def __init__(self, depth=5, in_chan=8, out_chan=2, inter_chan=64, init = False, dpr=0.5):
        super(FCN6,self).__init__()
        layer = []
        # inter_chan_list=[3600,4800,4800,2400]
        # inter_chan_list=[4800,2400]
        #inter_chan_list=[1024]
        #inter_chan_list=[2048]
        inter_chan_list=[512]
        # bias=True

        layer.append(nn.Linear(in_chan, inter_chan_list[0],bias=True))
        # layer.append(nn.Linear(in_chan, inter_chan_list[0],bias=False))
        layer.append(nn.Dropout(p=dpr))
        layer.append(nn.ReLU())
        # layer.append(nn.Tanh())
        # layer.append(nn.Linear(inter_chan_list[0],inter_chan_list[1],bias=True))
        # layer.append(nn.ReLU())
        layer.append(nn.Linear(inter_chan_list[0],out_chan,bias=True))
        # layer.append(nn.Dropout(p=0.5))
        # layer.append(nn.Linear(inter_chan_list[0],out_chan,bias=False))
        #layer.append(nn.ReLU())
        # layer.append(nn.BatchNorm1d(out_chan, affine=False)) # Without Learnable Parameters
        # layer.append(nn.Tanh())
        self.fcn = nn.Sequential(*layer)

    def forward(self, x):
        out = self.fcn(x)
        return out

class FCN5(nn.Module):
    def __init__(self, inter_chan_list, in_chan=8, out_chan=2, inter_chan=64, init = False, dpr=0.5):
        super(FCN5,self).__init__()
        layer = []
        # inter_chan_list=[3600,4800,4800,2400]
        # inter_chan_list=[4800,2400]
        #inter_chan_list=[2048]
        # inter_chan_list=[3096]
        bias=True
        print('inter_chan_list=',inter_chan_list)

        layer.append(nn.Linear(in_chan, inter_chan_list[0],bias=True))
        layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.Tanh())
        # layer.append(nn.ReLU())
        layer.append(nn.PReLU())
        # layer.append(Sine())
        for ii in range(len(inter_chan_list)-1):
            layer.append(nn.Linear(inter_chan_list[ii], inter_chan_list[ii+1],bias=True))
            layer.append(nn.Dropout(p=dpr))
            # layer.append(nn.ReLU())
            # layer.append(nn.Tanh())
            layer.append(nn.PReLU())
            # layer.append(Sine())

        ii=len(inter_chan_list)-1
        # layer.append(nn.Linear(inter_chan_list[0],inter_chan_list[1],bias=True))
        # layer.append(nn.ReLU())
        layer.append(nn.Linear(inter_chan_list[ii],out_chan,bias=True))
        layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.PReLU())
        # layer.append(nn.Tanh())
        # layer.append(nn.BatchNorm1d(out_chan, affine=False)) # Without Learnable Parameters
        # layer.append(nn.Tanh())
        self.fcn = nn.Sequential(*layer)

    def forward(self, x):
        out = self.fcn(x)
        return out


class FCN7(nn.Module):
    def __init__(self, inter_chan_list, in_chan=8, out_chan=2, inter_chan=64, init = False, dpr=0.5):
        super(FCN7,self).__init__()
        layer = []
        # inter_chan_list=[3600,4800,4800,2400]
        # inter_chan_list=[4800,2400]
        #inter_chan_list=[1024]
        #inter_chan_list=[2048]
        # inter_chan_list=[512]
        # inter_chan_list=[256,256]
        # bias=True

        layer.append(nn.Linear(in_chan, inter_chan_list[0],bias=True))
        layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.ReLU())
        # layer.append(nn.Tanh())
        layer.append(nn.PReLU())
        # layer.append(Sine())
        for ii in range(len(inter_chan_list)-1):
            layer.append(nn.Linear(inter_chan_list[ii], inter_chan_list[ii+1],bias=True))
            layer.append(nn.Dropout(p=dpr))
            # layer.append(nn.ReLU())
            # layer.append(nn.Tanh())
            layer.append(nn.PReLU())
            # layer.append(Sine())
        # layer.append(nn.Tanh())
        # layer.append(nn.Linear(inter_chan_list[0],inter_chan_list[1],bias=True))
        # layer.append(nn.ReLU())
        ii=len(inter_chan_list)-1
        layer.append(nn.Linear(inter_chan_list[ii],out_chan,bias=True))
        layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.Linear(inter_chan_list[0],out_chan,bias=False))
        #layer.append(nn.ReLU())
        # layer.append(nn.BatchNorm1d(out_chan, affine=False)) # Without Learnable Parameters
        #layer.append(nn.Tanh())
        self.fcn = nn.Sequential(*layer)

    def forward(self, x):
        out = self.fcn(x)
        return out



class CNN(nn.Module):
    def __init__(self, inter_chan_list, in_chan=8, out_chan=2, inter_chan=64, init = False, dpr=0.5):
        super(CNN,self).__init__()
        layer = []
        # inter_chan_list=[3600,4800,4800,2400]
        # inter_chan_list=[4800,2400]
        #inter_chan_list=[1024]
        #inter_chan_list=[2048]
        # inter_chan_list=[512]
        # inter_chan_list=[256,256]
        # bias=True

        # layer.append(nn.Linear(in_chan, inter_chan_list[0],bias=True))
        layer.append(nn.Conv2d(in_channels=in_chan, out_channels=inter_chan_list[0], kernel_size=[3,3], padding=1, bias=True))
        layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.ReLU())
        # layer.append(nn.Tanh())
        layer.append(nn.PReLU())
        # layer.append(Sine())
        for ii in range(len(inter_chan_list)-1):
            # layer.append(nn.Linear(inter_chan_list[ii], inter_chan_list[ii+1],bias=True))
            layer.append(nn.Conv2d(in_channels=inter_chan_list[ii], out_channels=inter_chan_list[ii+1], kernel_size=[3,3], padding=1, bias=True))
            layer.append(nn.Dropout(p=dpr))
            # layer.append(nn.ReLU())
            # layer.append(nn.Tanh())
            layer.append(nn.PReLU())
            # layer.append(Sine())
        # layer.append(nn.Tanh())
        # layer.append(nn.Linear(inter_chan_list[0],inter_chan_list[1],bias=True))
        # layer.append(nn.ReLU())
        ii=len(inter_chan_list)-1
        # layer.append(nn.Linear(inter_chan_list[ii],out_chan,bias=True))
        layer.append(nn.Conv2d(in_channels=inter_chan_list[ii], out_channels=out_chan, kernel_size=[3,3], padding=1, bias=True))
        # layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.Linear(inter_chan_list[0],out_chan,bias=False))
        #layer.append(nn.ReLU())
        # layer.append(nn.BatchNorm1d(out_chan, affine=False)) # Without Learnable Parameters
        #layer.append(nn.Tanh())
        self.fcn = nn.Sequential(*layer)

    def forward(self, x):
        out = self.fcn(x)
        return out



class cnn(nn.Module):
    def __init__(self, depth=3, n_channels=32, in_chan=2, out_chan=2,add_bias=True):#bias false
        super(cnn, self).__init__()
        layers = []
        layers.append(nn.Conv1d(in_channels=in_chan, out_channels=n_channels, kernel_size=3, padding=1, bias=True))
        layers.append(nn.PReLU(n_channels,init=0.025))
        for _ in range(depth-2):
            layers.append(nn.Conv1d(in_channels=n_channels, out_channels=n_channels, 
                kernel_size=3, padding=1, bias=True))
            layers.append(nn.PReLU(n_channels,init=0.025))
        layers.append(nn.Conv1d(in_channels=n_channels, out_channels=out_chan, 
            kernel_size=3, padding=1, bias=True))
        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        return out



class mmsenet():
    def __init__(self, args):
        self.layers = args.layers
        self.noise = args.noise
        # self.net = MIMONet(args,CNN=FCN)
        # self.args = args
        # if args.parallel == True:
        #     self.net = nn.DataParallel(self.net)
        # self.net = self.net.cuda()

        # self.net = MIMONet(args,CNN=FCN)
        # self.net = MIMONet(args,MLP1=FCN3,MLP2=FCN4)
        # self.net = MIMONet(args,MLP1=FCN5,MLP2=FCN5,MLP3=FCN6,MLP4=FCN6)
        # self.net = MIMONet(args,MLP1=FCN5,MLP2=FCN5,MLP3=FCN7,MLP4=FCN6)
        # self.net = MIMONet(args,MLP1=FCN5,MLP2=FCN7)
        # self.net = MIMONet(args,MLP1=FCN5,MLP2=CNN) #dec15
        self.net = MIMONet(args,MLP1=CNN,MLP2=CNN) #dec15
        # self.net = MIMONet(args,MLP1=CNN1D,MLP2=FCN7)
        # CNN1D
        # Jan08-v3,v4,v5
        # self.net = MetaCTmodelJan08v3(args, InvNet=CG_rec_layerJan08v3)
        self.args = args
        self.net = self.net.cuda()
        if args.parallel == True:
            self.net = nn.DataParallel(self.net)

    def tr_model(self, Received_Y, Hls, X0, sigma, dop):
        self.net.train()
        # H1, H2, X1, Recover_X = self.net(Received_Y, Hls, X0, sigma)
        H1, Recover_X = self.net(Received_Y, Hls, X0, sigma, dop)
        return H1, Recover_X

    def test_model(self, Received_Y,  Hls, X0, sigma, dop):
        with torch.no_grad():
            self.net.eval()
            # H1, H2, X1,Recover_X = self.net(Received_Y.cuda(), Hls.cuda(), X0.cuda(), sigma.cuda())
            H1, Recover_X = self.net(Received_Y.cuda(), Hls.cuda(), X0.cuda(), sigma.cuda(), dop)
        # return H1, H2,X1, Recover_X
        return H1, Recover_X

    def vl_model(self, Received_Y,  Hls, X0, Ideal_H, Ideal_X, sigma2, dop):
        with torch.no_grad():
            self.net.eval()
            # id_valid = torch.arange(300000,375000,5).long()
            data_len=Received_Y.shape[0]
            # print('validation data len=',data_len)
            id_valid = torch.arange(0,data_len,1).long()
            Received_Y_valid = Received_Y[id_valid,:,:,:,:].cuda()
            Hls_valid = Hls[id_valid,:,:,:,:].cuda()
            Ideal_H_valid = Ideal_H[id_valid,:,:,:,:].cuda()
            Ideal_X_valid = Ideal_X[id_valid,:,:,:].cuda()
            X0_valid = X0[id_valid,:,:,:].cuda()
            sigma2 = sigma2[id_valid,:,:].cuda()

            batchsize = self.args.tr_batch
            # sigma2 = np.power(10,-self.noise/10)
            ber_vl = 0.0
            bler_vl = 0.0
            batch_num=data_len//batchsize
            print('data length={}, valid batch size={}, #batch={}'.format(data_len,batchsize,batch_num))
            for i in range(batch_num):
                batch_idx = torch.arange(batchsize*i,batchsize*(i+1),1)
                Received_Y_batch = Received_Y_valid[batch_idx,:,:,:,:].cuda()
                # TranX_batch = Hls_valid[batch_idx,:,:,:].cuda()
                Hls_batch = Hls_valid[batch_idx,:,:,:,:].cuda()
                Ideal_H_batch = Ideal_H_valid[batch_idx,:,:,:,:].cuda()
                Ideal_X_batch = Ideal_X_valid[batch_idx,:,:,:].cuda()
                X0_batch = X0_valid[batch_idx,:,:,:].cuda()
                sigma_batch = sigma2[batch_idx,:,:].cuda()

                # H1, H2,_,Recover_X_batch = self.net(Received_Y_batch,  Hls_batch, X0_batch, sigma2)
                # H_full_batch,Recover_X_batch = self.net(Received_Y_batch,  TranX_batch, sigma)
                H1, Recover_X_batch = self.net(Received_Y_batch,  Hls_batch, X0_batch, sigma_batch, dop)
                # H1, Recover_X_batch = self.net(Received_Y_batch,  Ideal_H_batch, X0_batch, sigma_batch, dop)
                # ber_vl = ber_vl + ber(Recover_X_batch,Ideal_X_batch)
                ber_vl = ber_vl + calc_ber(Recover_X_batch,Ideal_X_batch)
                # bler_vl = bler_vl + bler(H1, Recover_X_batch,Ideal_X_batch,sigma_batch)
                bler_vl = bler_vl + bler(Ideal_H_batch, Recover_X_batch,Ideal_X_batch,sigma_batch)
            ber_vl = ber_vl/float(batch_num)
            bler_vl = bler_vl/float(batch_num)
            # print('Valid: BER = {:.6f}, BLER = {:.6f}'.format(ber_vl,bler_vl.item()))
                                    
            now_time=datetime.now().strftime("%H:%M:%S")
            f=open('result/'+self.args.info+'/training_log.txt',"a")
            msg='Valid: Doppler={}, SNR={}, BER = {:.6f} dB, BLER = {:.6f}, Now={}'.format(self.args.doppler,-round(10.0*torch.log10(sigma2[0,0,0]).cpu().numpy()),ber_vl, bler_vl.item(),now_time)
            # msg='Valid: BER = {:.6f}, BLER = {:.6f}, Now={}'.format(ber_vl, bler_vl.item(),now_time)
            print(msg)
            f.write('\n'+msg)

        return ber_vl

    def load_model(self, args):
        if self.args.phase=='train':
            load_model_dir=self.args.resume_ckp_dir
        elif self.args.phase=='test':
            load_model_dir=self.args.test_ckp_dir
            print('test epoch=',load_model_dir)
        ckp = torch.load(load_model_dir, map_location=lambda storage, loc: storage.cuda(args.gpu_idx))
        self.net.load_state_dict(ckp['model'])
        return ckp


class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BasicBlock,self).__init__()
        if in_channels!=out_channels:
           self.flag = 1
           self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=(1,1), padding=(0,0), bias=True)
        else:
           self.flag = 0
        kernel_size = (3,1)
        padding = (1,0)
        #self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.PReLU(in_channels,init=0.025)
        self.relu2 = nn.PReLU(out_channels,init=0.025)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, padding=padding, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                kernel_size=(1,1), padding=(0,0), bias=True)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                kernel_size=kernel_size, padding=padding, bias=True)
        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                kernel_size=(1,1), padding=(0,0), bias=True)


    def forward(self, x):
        #out = self.bn1(x)
        out = self.relu1(x)
        out = self.conv1(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        if self.flag == 1:
           x = self.conv5(x)
        out = out + x
        return out



import torch.nn.functional as F
# class DeepRX(nn.Module):
#     def __init__(self, in_planes, planes, stride=1, n_chan=64):
#         super(DeepRX, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=3, dilation = (1,1),
# #                               stride=stride,
#                                padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         # dilation是什么
        
#         self.rs1 = nn.Sequential(
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64,kernel_size=3,padding=1, dilation=(1,1), groups = 64),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64,kernel_size=3,padding=1,dilation=(1,1), groups = 64),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#         )
        
#         self.rs2 = nn.Sequential(
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             nn.Conv2d(64, 64,kernel_size=3,padding=1, dilation=(1,1), groups = 64),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             nn.Conv2d(64, 64,kernel_size=3,padding=1,dilation=(1,1), groups = 64),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#         )
        
#         self.downsample_rs3 = nn.Conv2d(64, 128, kernel_size=1,padding=0)
#         self.rs3 = nn.Sequential(
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64,kernel_size=3,padding=(2,3), dilation=(2,3), groups = 64),
#             nn.Conv2d(64, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
            
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 128,kernel_size=3,padding=(2,3),dilation=(2,3), groups = 128),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#         )
        
#         self.rs4 = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.Conv2d(128, 128,kernel_size=3,padding=(2,3), dilation=(2,3), groups = 128),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.Conv2d(128, 128,kernel_size=3,padding=(2,3),dilation=(2,3), groups = 128),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#         )
        
#         self.downsample_rs5 = nn.Conv2d(128, 256, kernel_size=1,padding=0)
#         self.rs5 = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.Conv2d(128, 128,kernel_size=3,padding=(2,3), dilation=(2,3), groups = 128),
#             nn.Conv2d(128, 256, kernel_size=1,padding=0),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             nn.Conv2d(256, 256,kernel_size=3,padding=(2,3),dilation=(2,3), groups = 256),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#         )
        
#         self.rs6 = nn.Sequential(
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             nn.Conv2d(256, 256,kernel_size=3,padding=(3,6), dilation=(3,6), groups = 256),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             nn.Conv2d(256, 256,kernel_size=3,padding=(3,6),dilation=(3,6), groups = 256),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#         )
        
#         self.rs7 = nn.Sequential(
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             nn.Conv2d(256, 256,kernel_size=3,padding=(2,3), dilation=(2,3), groups = 256),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             nn.Conv2d(256, 256,kernel_size=3,padding=(2,3),dilation=(2,3), groups = 256),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#             nn.Conv2d(256, 256, kernel_size=1,padding=0),
#         )
        
#         self.downsample_rs8 = nn.Conv2d(256, 128, kernel_size=1,padding=0)
#         self.rs8 = nn.Sequential(
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
            
#             nn.Conv2d(256, 256,kernel_size=3,padding=(2,3), dilation=(2,3), groups = 128),
#             nn.Conv2d(256, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.Conv2d(128, 128,kernel_size=3,padding=(2,3),dilation=(2,3), groups = 128),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#         )
        
#         self.rs9 = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.Conv2d(128, 128,kernel_size=3,padding=(2,3), dilation=(2,3), groups = 128),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.Conv2d(128, 128,kernel_size=3,padding=(2,3),dilation=(2,3), groups = 128),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#             nn.Conv2d(128, 128, kernel_size=1,padding=0),
#         )
        
#         self.downsample_rs10 = nn.Conv2d(128, 64, kernel_size=1,padding=0)
#         self.rs10 = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
            
#             nn.Conv2d(128, 128,kernel_size=3,padding=1, dilation=(1,1), groups = 64),
#             nn.Conv2d(128, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             nn.Conv2d(64, 64,kernel_size=3,padding=1,dilation=(1,1), groups = 64),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#         )
        
#         self.rs11 = nn.Sequential(
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             nn.Conv2d(64, 64,kernel_size=3,padding=1, dilation=(1,1), groups = 64),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
            
#             nn.Conv2d(64, 64,kernel_size=3,padding=1,dilation=(1,1), groups = 64),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#             nn.Conv2d(64, 64, kernel_size=1,padding=0),
#         )
        
#         self.conv_end = nn.Conv2d(64, planes, kernel_size=1, dilation=(1,1),
#                                stride=1, padding=0, bias=False)

#     def forward(self, x):
#         self.y_val = x[:, :8, :, :]
#         self.x_val = x[:, :18, :, :]
#         self.Hls = x[:, 10:, :, :]
#         debug=0

#         out1 = self.conv1(self.x_val)
#         if debug:
#             print('out.shape=', out1.shape)
#         out_res1 = self.rs1(out1) + out1
#         if debug:
#             print('out_res1.shape=',out_res1.shape)
#         out_res2 = self.rs2(out_res1) + out_res1
#         if debug:
#             print('out_res2.shape=',out_res2.shape)
        
#         out_downsample_rs3 = self.downsample_rs3(out_res2)
#         if debug:
#             print('oooooout_downsample_rs3.shape=',out_downsample_rs3.shape)
#         out_res3 = self.rs3(out_res2)# + out_downsample_rs3

        
#         if debug:
#             print('out_res3.shape=',out_res3.shape)
#         out_res3 = out_res3 + out_downsample_rs3
#         if debug:
#             print('ooooooo out_res3.shape=',out_res3.shape)

#         out_res4 = self.rs4(out_res3) + out_res3
#         if debug:
#             print('out_res4.shape=',out_res4.shape)
#         out_res5 = self.rs5(out_res4) + self.downsample_rs5(out_res4)
#         if debug:
#             print('out_res5.shape=',out_res5.shape)
#         out_res6 = self.rs6(out_res5) + out_res5
#         if debug:
#             print('out_res6.shape=',out_res6.shape)
#         out_res7 = self.rs7(out_res6) + out_res6
#         if debug:
#             print('out_res7.shape=',out_res7.shape)
#         out_res8 = self.rs8(out_res7) + self.downsample_rs8(out_res7)
#         if debug:
#             print('out_res8.shape=',out_res8.shape)
#         out_res9 = self.rs9(out_res8) + out_res8
#         if debug:
#             print('out_res9.shape=',out_res9.shape)
#         out_res10 = self.rs10(out_res9) + self.downsample_rs10(out_res9)
#         if debug:
#             print('out_res10.shape=',out_res10.shape)
#         out_res11 = self.rs11(out_res10) + out_res10
#         if debug:
#             print('out_res11.shape=',out_res11.shape)
#         out_conv_end = F.sigmoid(self.conv_end(out_res11))
#         if debug:
#             print('out_conv_end.shape=', out_conv_end.shape)
#         return out_conv_end




class ResBlock(nn.Module):
    def __init__(self, n_chan=64, padding=1, dilation=(1,1), groups=64):
        super(ResBlock, self).__init__()
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3, padding=padding, dilation=dilation, groups = groups),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=padding,dilation=dilation, groups = groups),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
    def forward(self, inputs):
        return self.rs1(inputs) + inputs

class UpsResBlock(nn.Module):
    def __init__(self, n_chan=64, padding=(2,3), dilation=(2,3), groups=64):
        super(UpsResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chan, n_chan*2, kernel_size=1,padding=0)
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3, padding=padding, dilation=dilation, groups = groups),
            nn.Conv2d(n_chan, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=padding,dilation=dilation, groups = groups*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
    def forward(self, inputs):
        return self.rs1(inputs) + self.conv(inputs)

class DspResBlock(nn.Module):
    def __init__(self, n_chan=64, padding=(2,3), dilation=(2,3), groups=64):
        super(DspResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chan*2, n_chan, kernel_size=1,padding=0)
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3, padding=padding, dilation=dilation, groups = groups),
            nn.Conv2d(n_chan*2, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=padding,dilation=dilation, groups = groups),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
    def forward(self, inputs):
        return self.rs1(inputs) + self.conv(inputs)



class DeepRxNew(nn.Module):
    def __init__(self, in_planes, planes, stride=1, n_chan=64):
        super(DeepRxNew, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, n_chan, kernel_size=3, dilation = (1,1),padding=1, bias=False)
        self.rs1=ResBlock(n_chan, padding=1, dilation=(1,1), groups=n_chan)
        self.rs2=ResBlock(n_chan, padding=1, dilation=(1,1), groups=n_chan)
        self.rs3=UpsResBlock(n_chan, padding=(2,3), dilation=(2,3), groups=n_chan)
        self.rs4=ResBlock(n_chan*2, padding=(2,3), dilation=(2,3), groups=n_chan*2)
        self.rs5=UpsResBlock(n_chan*2, padding=(2,3), dilation=(2,3), groups=n_chan*2)
        self.rs6=ResBlock(n_chan*4, padding=(3,6), dilation=(3,6), groups=n_chan*4)
        self.rs7=ResBlock(n_chan*4, padding=(2,3), dilation=(2,3), groups=n_chan*4)
        self.rs8=DspResBlock(n_chan*2, padding=(2,3), dilation=(2,3), groups=n_chan*2)
        self.rs9=ResBlock(n_chan*2, padding=(2,3), dilation=(2,3), groups=n_chan*2)
        self.rs10=DspResBlock(n_chan, padding=1, dilation=(1,1), groups=n_chan)
        self.rs11=ResBlock(n_chan, padding=1, dilation=(1,1), groups=n_chan)
        self.conv_end = nn.Conv2d(n_chan, planes, kernel_size=1, dilation=(1,1),stride=1, padding=0, bias=False)

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
        output = torch.sigmoid(self.conv_end(out_res11))
        return output


class DeepRX(nn.Module):
    def __init__(self, in_planes, planes, stride=1, n_chan=64):
        super(DeepRX, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, n_chan, kernel_size=3, dilation = (1,1),
#                               stride=stride,
                               padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(n_chan)
        # dilation是什么
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.rs2 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.downsample_rs3 = nn.Conv2d(n_chan, n_chan*2, kernel_size=1,padding=0)
        self.rs3 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan),
            nn.Conv2d(n_chan, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.rs4 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.downsample_rs5 = nn.Conv2d(n_chan*2, n_chan*4, kernel_size=1,padding=0)
        self.rs5 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
        )
        
        self.rs6 = nn.Sequential(
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(3,6), dilation=(3,6), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(3,6),dilation=(3,6), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
        )
        
        self.rs7 = nn.Sequential(
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
        )
        
        self.downsample_rs8 = nn.Conv2d(n_chan*4, n_chan*2, kernel_size=1,padding=0)
        self.rs8 = nn.Sequential(
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*4, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.rs9 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.downsample_rs10 = nn.Conv2d(n_chan*2, n_chan, kernel_size=1,padding=0)
        self.rs10 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan*2, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.rs11 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.conv_end = nn.Conv2d(n_chan, planes, kernel_size=1, dilation=(1,1),
                               stride=1, padding=0, bias=False)

    def forward(self, x):
        self.y_val = x[:, :8, :, :]
        self.x_val = x[:, :18, :, :]
        self.Hls = x[:, 10:, :, :]
        debug=0

        out1 = self.conv1(self.x_val)
        if debug:
            print('out.shape=', out1.shape)
        out_res1 = self.rs1(out1) + out1
        if debug:
            print('out_res1.shape=',out_res1.shape)
        out_res2 = self.rs2(out_res1) + out_res1
        if debug:
            print('out_res2.shape=',out_res2.shape)
        
        out_downsample_rs3 = self.downsample_rs3(out_res2)
        if debug:
            print('oooooout_downsample_rs3.shape=',out_downsample_rs3.shape)
        out_res3 = self.rs3(out_res2)# + out_downsample_rs3

        
        if debug:
            print('out_res3.shape=',out_res3.shape)
        out_res3 = out_res3 + out_downsample_rs3
        if debug:
            print('ooooooo out_res3.shape=',out_res3.shape)

        out_res4 = self.rs4(out_res3) + out_res3
        # debug=1
        if debug:
            print('out_res4.shape=',out_res4.shape)
        out_res5 = self.rs5(out_res4) + self.downsample_rs5(out_res4)
        if debug:
            print('out_res5.shape=',out_res5.shape)
        out_res6 = self.rs6(out_res5) + out_res5
        if debug:
            print('out_res6.shape=',out_res6.shape)
        out_res7 = self.rs7(out_res6) + out_res6
        if debug:
            print('out_res7.shape=',out_res7.shape)
        out_res8 = self.rs8(out_res7) + self.downsample_rs8(out_res7)
        if debug:
            print('out_res8.shape=',out_res8.shape)
        out_res9 = self.rs9(out_res8) + out_res8
        if debug:
            print('out_res9.shape=',out_res9.shape)
        out_res10 = self.rs10(out_res9) + self.downsample_rs10(out_res9)
        if debug:
            print('out_res10.shape=',out_res10.shape)
        out_res11 = self.rs11(out_res10) + out_res10
        if debug:
            print('out_res11.shape=',out_res11.shape)
        out_conv_end = F.sigmoid(self.conv_end(out_res11))
        if debug:
            print('out_conv_end.shape=', out_conv_end.shape)
        return out_conv_end




class DeepRXH(nn.Module):
    def __init__(self, in_planes, planes, stride=1, n_chan=64):
        super(DeepRXH, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, n_chan, kernel_size=3, dilation = (1,1),
#                               stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_chan)
        # dilation是什么
        
        self.rs1 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.rs2 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.downsample_rs3 = nn.Conv2d(n_chan, n_chan*2, kernel_size=1,padding=0)
        self.rs3 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan),
            nn.Conv2d(n_chan, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.rs4 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.downsample_rs5 = nn.Conv2d(n_chan*2, n_chan*4, kernel_size=1,padding=0)
        self.rs5 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
        )
        
        self.rs6 = nn.Sequential(
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(3,6), dilation=(3,6), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(3,6),dilation=(3,6), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
        )
        
        self.rs7 = nn.Sequential(
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*4),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*4, n_chan*4, kernel_size=1,padding=0),
        )
        
        self.downsample_rs8 = nn.Conv2d(n_chan*4, n_chan*2, kernel_size=1,padding=0)
        self.rs8 = nn.Sequential(
            nn.BatchNorm2d(n_chan*4),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*4, n_chan*4,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*4, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.rs9 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3), dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=(2,3),dilation=(2,3), groups = n_chan*2),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
            nn.Conv2d(n_chan*2, n_chan*2, kernel_size=1,padding=0),
        )
        
        self.downsample_rs10 = nn.Conv2d(n_chan*2, n_chan, kernel_size=1,padding=0)
        self.rs10 = nn.Sequential(
            nn.BatchNorm2d(n_chan*2),
            nn.ReLU(),
            
            nn.Conv2d(n_chan*2, n_chan*2,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan*2, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.rs11 = nn.Sequential(
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1, dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.BatchNorm2d(n_chan),
            nn.ReLU(),
            
            nn.Conv2d(n_chan, n_chan,kernel_size=3,padding=1,dilation=(1,1), groups = n_chan),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
            nn.Conv2d(n_chan, n_chan, kernel_size=1,padding=0),
        )
        
        self.conv_end = nn.Conv2d(n_chan, planes, kernel_size=1, dilation=(1,1),
                               stride=1, padding=0, bias=False)

    def forward(self, x):
        self.y_val = x[:, :8, :, :]
        self.x_val = x[:, :18, :, :]
        self.Hls = x[:, 10:, :, :]
        debug=0

        out1 = self.conv1(self.x_val)
        if debug:
            print('out.shape=', out1.shape)
        out_res1 = self.rs1(out1) + out1
        if debug:
            print('out_res1.shape=',out_res1.shape)
        out_res2 = self.rs2(out_res1) + out_res1
        if debug:
            print('out_res2.shape=',out_res2.shape)
        
        out_downsample_rs3 = self.downsample_rs3(out_res2)
        if debug:
            print('oooooout_downsample_rs3.shape=',out_downsample_rs3.shape)
        out_res3 = self.rs3(out_res2)# + out_downsample_rs3

        
        if debug:
            print('out_res3.shape=',out_res3.shape)
        out_res3 = out_res3 + out_downsample_rs3
        if debug:
            print('ooooooo out_res3.shape=',out_res3.shape)

        out_res4 = self.rs4(out_res3) + out_res3
        # debug=1
        if debug:
            print('out_res4.shape=',out_res4.shape)
        out_res5 = self.rs5(out_res4) + self.downsample_rs5(out_res4)
        if debug:
            print('out_res5.shape=',out_res5.shape)
        out_res6 = self.rs6(out_res5) + out_res5
        if debug:
            print('out_res6.shape=',out_res6.shape)
        out_res7 = self.rs7(out_res6) + out_res6
        if debug:
            print('out_res7.shape=',out_res7.shape)
        out_res8 = self.rs8(out_res7) + self.downsample_rs8(out_res7)
        if debug:
            print('out_res8.shape=',out_res8.shape)
        out_res9 = self.rs9(out_res8) + out_res8
        if debug:
            print('out_res9.shape=',out_res9.shape)
        out_res10 = self.rs10(out_res9) + self.downsample_rs10(out_res9)
        if debug:
            print('out_res10.shape=',out_res10.shape)
        out_res11 = self.rs11(out_res10) + out_res10
        if debug:
            print('out_res11.shape=',out_res11.shape)
        # out_conv_end = F.sigmoid(self.conv_end(out_res11))
        out_conv_end = self.conv_end(out_res11)
        if debug:
            print('out_conv_end.shape=', out_conv_end.shape)
        return out_conv_end



def cf_div(x, y):
    dim = len(x.size())-1
    # y_abs = y[..., 0]**2 + y[..., 1]**2
    y_abs = y[..., 0]**2 + y[..., 1]**2+1e-8
    real = (x[...,0] * y[...,0] + x[...,1] * y[...,1]) / y_abs
    image = (x[...,1] * y[...,0] - x[...,0] * y[...,1]) /y_abs
    div = torch.stack((real, image), dim=dim)
    return div

def cf_divR(x, y):
    # dim = len(x.size())-1
    # y_abs = y[..., 0]**2 + y[..., 1]**2
    # y_abs = y[..., 0]**2 + y[..., 1]**2+1e-8
    # print('y.shape=',y.shape)
    real = x[...,0]/ y
    # print('real.shape=',real.shape)
    image = x[...,1]/ y
    # print('image.shape=',image.shape)
    # image = (x[...,1] * y[...,0] - x[...,0] * y[...,1]) /y_abs
    div = torch.stack((real, image), dim=-1)
    return div

def th_div(H,Y):
    X=cf_div(Y[...,0:2],H[...,0:2])+cf_div(Y[...,2:4],H[...,2:4])+cf_div(Y[...,4:6],H[...,4:6])+cf_div(Y[...,6:8],H[...,6:8])
    return X

def cf_mul(x, y):
    # dim = len(x.size())-1
    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    mul = torch.stack((real, image), dim=-1)
    return mul

# def th_mul(Y, H, sigma):
#     H_norm=cf_mul(H[...,0:2],H[...,0:2])+cf_mul(H[...,2:4],H[...,2:4])+cf_mul(H[...,4:6],H[...,4:6])+cf_mul(H[...,6:8],H[...,6:8])
#     X_zf=cf_mul(Y[...,0:2],H[...,0:2])+cf_mul(Y[...,2:4],H[...,2:4])+cf_mul(Y[...,4:6],H[...,4:6])+cf_mul(Y[...,6:8],H[...,6:8])
#     X_zf=cf_div(X_zf,H_norm+sigma**2)
#     return X_zf

def cf_conj(x):
    dim = len(x.size())-1
    x_conj = torch.stack((x[..., 0], -x[..., 1]), dim=dim)
    return x_conj

def MMSE_SD(Y, H, sigma2):
    H_norm=cf_mul(H[...,0,:],H[...,0,:])+cf_mul(H[...,1,:],H[...,1,:])+cf_mul(H[...,2,:],H[...,2,:])+cf_mul(H[...,3,:],H[...,3,:])
    H_conj=cf_conj(H)
    # print('H.shape=',H.shape)
    HtY=cf_mul(Y[...,0,:],H_conj[...,0,:])+cf_mul(Y[...,1,:],H_conj[...,1,:])+cf_mul(Y[...,2,:],H_conj[...,2,:])+cf_mul(Y[...,3,:],H_conj[...,3,:])
    # print('X_zf.shape=',X_zf.shape)
    # X_zf=cf_div(HtY,H_norm+sigma)
    X_zf=cf_div(HtY,H_norm+sigma2)
    return X_zf


def cf_vect_norm(X):
    X_norm=(X[:,0,:,0]*X[:,0,:,0]+X[:,1,:,0]*X[:,1,:,0]) + (X[:,0,:,1]*X[:,0,:,1]+X[:,1,:,1]*X[:,1,:,1]) + (X[:,0,:,2]*X[:,0,:,2]+X[:,1,:,2]*X[:,1,:,2])+ (X[:,0,:,3]*X[:,0,:,3]+X[:,1,:,3]*X[:,1,:,3])
    return X_norm
 
def cnorm(x):
    # print('x.shape=',x.shape)
    return x[...,0]**2+x[...,1]**2

def H_cnorm(H):
    # print('H[...,0,:].shape=',H[...,0,:].shape)
    H_norm=cnorm(H[...,0,:])+cnorm(H[...,1,:])+cnorm(H[...,2,:])+cnorm(H[...,3,:])
    H_norm=H_norm.unsqueeze(-1)
    # print('H_norm.shape=',H_norm.shape)
    return H_norm

def c_mul(X,Y):
    XtY_real=X[...,0]*Y[...,0]-X[...,1]*Y[...,1]
    XtY_imag=X[...,0]*Y[...,1]+X[...,1]*Y[...,0]
    XtY_cpx=torch.cat([XtY_real.unsqueeze(-1),XtY_imag.unsqueeze(-1)],dim=-1)
    # print('XtY_cpx.shape=',XtY_cpx.shape)
    return XtY_cpx

def HtY(H,Y):
    HtY=c_mul(H[...,0,:],Y[...,0,:])+c_mul(H[...,1,:],Y[...,1,:])+c_mul(H[...,2,:],Y[...,2,:])+c_mul(H[...,3,:],Y[...,3,:])
    # print('HtY.shape=',HtY.shape)
    return HtY

def HtH(X):
    X_tmp=torch.zeros_like(X)
    X_tmp[...,0]=X[...,0]**2-X[...,1]**2
    X_tmp[...,1]=X[...,0]*X[...,1]
    return X_tmp
 
def cf_div(x, y):
    dim = len(x.size())-1
    # y_abs = y[..., 0]**2 + y[..., 1]**2
    y_abs = y[..., 0]**2 + y[..., 1]**2+1e-8
    real = (x[...,0] * y[...,0] + x[...,1] * y[...,1]) / y_abs
    image = (x[...,1] * y[...,0] - x[...,0] * y[...,1]) /y_abs
    div = torch.stack((real, image), dim=dim)
    return div

def th_div(H,Y):
    X=cf_div(Y[...,0:2],H[...,0:2])+cf_div(Y[...,2:4],H[...,2:4])+cf_div(Y[...,4:6],H[...,4:6])+cf_div(Y[...,6:8],H[...,6:8])
    return X


# v11 model
# class MIMONet(nn.Module):
#     def __init__(self, args, MLP1=FCN5, MLP2=FCN5,MLP3=FCN6,MLP4=FCN6):
#     # def __init__(self, args, CNN):
#         super(MIMONet,self).__init__()
#         self.args = args

#         inter_chan1=self.args.fcn1_chan
#         inter_chan2=self.args.fcn2_chan
#         NN_depth=self.args.fcn_depth
#         # self.net1 = MLP1(depth=NN_depth, in_chan=12*24*2*9, out_chan=12*24*4*2, inter_chan=inter_chan1).cuda()
#         self.net1 = MLP1(depth=NN_depth, in_chan=12*24*2*5+12*4*2, out_chan=12*24*4*2, inter_chan=inter_chan1).cuda()

#         self.net3 = MLP3(depth=NN_depth, in_chan=12*24*2*3, out_chan=12*24*2, inter_chan=inter_chan2).cuda()

#     def forward(self, Received_Y, Hls, X0, sigma2):
#         Received_Y = Received_Y.detach()
#         Hls = Hls.detach()

#         Y_vec=Received_Y.reshape(self.args.tr_batch,12*24*4*2)
#         # h0_vec= Hls.reshape(self.args.tr_batch,12*24*4*2) # 75,2,24,5
#         h0_vec= Hls[:,0,::2,...].reshape(self.args.tr_batch,1*12*4*2) # 75,2,24,5
#         X0_vec=X0.reshape(self.args.tr_batch,12*24*2)
#         h1_input=torch.cat([h0_vec,Y_vec,X0_vec],dim=1)
#         h1_hat=Hls + self.net1(h1_input).reshape(self.args.tr_batch, 12,24,4,2)
#         # print('h1_hat.shape=',h1_hat.shape)
#         # HtH_norm=HtH(h1_hat)
#         HtH_norm=HtY(h1_hat, h1_hat)
#         H_conj_Y=HtY(h1_hat, Received_Y)
#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         Le=torch.mul(HtH_norm, H_conj_Y)
#         Le_vec=Le.reshape(self.args.tr_batch,12*24*2)
#         Le2=torch.mul(HtH_norm, Le)
#         Le2_vec=Le2.reshape(self.args.tr_batch,12*24*2)
#         # rec_X=H_conj_Y+self.net3(Le).reshape(self.args.tr_batch, 12,24,2)
#         HtY_vec=H_conj_Y.reshape(self.args.tr_batch,12*24*2)
#         XNet_input=torch.cat([HtY_vec, Le_vec, Le2_vec], dim=1)
#         rec_X=H_conj_Y/sigma2+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)
#         X1 = torch.complex(rec_X[...,0], rec_X[...,1])

#         # return h1_hat, h2_hat, X1, Recover_X
#         return h1_hat, X1




# class MIMONet(nn.Module):
#     def __init__(self, args, MLP1=FCN5, MLP2=FCN5,MLP3=FCN6,MLP4=FCN6):
#         super(MIMONet,self).__init__()
#         self.args = args

#         inter_chan1=self.args.fcn1_chan
#         inter_chan2=self.args.fcn2_chan
#         NN_depth=self.args.fcn_depth
#         # self.net1 = MLP1(depth=NN_depth, in_chan=12*24*2*9, out_chan=12*24*4*2, inter_chan=inter_chan1).cuda()
#         self.net1 = MLP1(depth=NN_depth, in_chan=12*24*2*5+12*4*2, out_chan=12*24*4*2, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()

#         #self.net3 = MLP3(depth=NN_depth, in_chan=12*24*2*3, out_chan=12*24*2, inter_chan=inter_chan2).cuda()
#         self.net3 = MLP3(depth=NN_depth, in_chan=12*24*2*2, out_chan=12*24*2, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()

#     def forward(self, Received_Y, Hls, X0, sigma2):
#         Received_Y = Received_Y.detach()
#         Hls = Hls.detach()

#         Y_vec=Received_Y.reshape(self.args.tr_batch,12*24*4*2)
#         # h0_vec= Hls.reshape(self.args.tr_batch,12*24*4*2) # 75,2,24,5
#         h0_vec= Hls[:,0,::2,...].reshape(self.args.tr_batch,1*12*4*2) # 75,2,24,5
#         X0_vec=X0.reshape(self.args.tr_batch,12*24*2)
#         h1_input=torch.cat([h0_vec,Y_vec,X0_vec],dim=1)
#         h1_hat=Hls + self.net1(h1_input).reshape(self.args.tr_batch, 12,24,4,2)
#         # print('h1_hat.shape=',h1_hat.shape)
#         # HtH_norm=HtH(h1_hat)
#         HtH_norm=HtY(h1_hat, h1_hat)
#         H_conj_Y=HtY(h1_hat, Received_Y)
#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         Le=torch.mul(HtH_norm, H_conj_Y)
#         Le_vec=Le.reshape(self.args.tr_batch,12*24*2)
#         #Le2=torch.mul(HtH_norm, Le)
#         #Le2_vec=Le2.reshape(self.args.tr_batch,12*24*2)
#         # rec_X=H_conj_Y+self.net3(Le).reshape(self.args.tr_batch, 12,24,2)
#         HtY_vec=H_conj_Y.reshape(self.args.tr_batch,12*24*2)

#         # design-1
#         #XNet_input=torch.cat([HtY_vec, Le_vec, Le2_vec], dim=1)
#         #rec_X=H_conj_Y/sigma2+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)
        
#         # Sep04 v-1221
#         # XNet_input=torch.cat([HtY_vec/sigma2, Le_vec], dim=1)
#         XNet_input=torch.cat([HtY_vec, Le_vec], dim=1)
#         #rec_X=self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)
#         #rec_X=H_conj_Y+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)
#         # print('sigma2.shape=',sigma2.shape,'H_conj_Y.shape=',H_conj_Y.shape)
#         #rec_X=H_conj_Y/sigma2.unsqueeze(-1)+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)
#         rec_X=H_conj_Y/sigma2.unsqueeze(-1)-Le/(sigma2.unsqueeze(-1)**2)+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)

#         X1 = torch.complex(rec_X[...,0], rec_X[...,1])

#         # return h1_hat, h2_hat, X1, Recover_X
#         return h1_hat, X1



def cf_XtX(x):
    # dim = len(x.size())-1
    x_sqr = x[..., 0]**2 + x[..., 1]**2
    # x_sqr_sum=torch.sum(x_sqr, dim=-1, keepdim=True)
    # print('x_sqr_sum.shape=',x_sqr_sum.shape)
    # real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    # image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    # mul = torch.stack((real, image), dim=dim)
    return x_sqr


def cf_HtH(x):
    # dim = len(x.size())-1
    x_sqr = x[..., 0]**2 + x[..., 1]**2
    x_sqr_sum=torch.sum(x_sqr, dim=-1, keepdim=True)
    # print('x_sqr_sum.shape=',x_sqr_sum.shape)
    # real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    # image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    # mul = torch.stack((real, image), dim=dim)
    return x_sqr_sum


def cf_conj(x):
    dim = len(x.size())-1
    x_conj = torch.stack((x[..., 0], -x[..., 1]), dim=dim)
    return x_conj

def cf_HtY(x,y):
    x_conj=cf_conj(x)
    x_conj_y=cf_mul(x_conj, y)
    x_conj_y_sum=torch.sum(x_conj_y, dim=-2, keepdim=False)
    # print('x_conj_y_sum.shape=',x_conj_y_sum.shape)
    # real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    # image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    # mul = torch.stack((real, image), dim=dim)
    return x_conj_y_sum

def cf_XtY(x,y):
    x_conj=cf_conj(x)
    x_conj_y=cf_mul(x_conj, y)
    return x_conj_y


# # v15 model
# class MIMONet(nn.Module):
#     def __init__(self, args, MLP1=FCN5, MLP2=FCN7):
#         super(MIMONet,self).__init__()
#         self.args = args
#         inter_chan1=self.args.fcn1_chan
#         inter_chan2=self.args.fcn2_chan
#         NN_depth=self.args.fcn_depth
#         self.netH = MLP1(depth=NN_depth, in_chan=12*24*2*5+12*4*2, out_chan=12*24*4*2, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()
#         self.netX = MLP2(depth=NN_depth, in_chan=12*24*2*3, out_chan=12*24*2, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()

#     def forward(self, Received_Y, Hls, X0, sigma2):
#         Received_Y = Received_Y.detach()
#         Hls = Hls.detach()

#         Y_vec=Received_Y.reshape(self.args.tr_batch,12*24*4*2)
#         # h0_vec= Hls.reshape(self.args.tr_batch,12*24*4*2) # 75,2,24,5
#         h0_vec= Hls[:,0,::2,...].reshape(self.args.tr_batch,1*12*4*2) # 75,2,24,5
#         X0_vec=X0.reshape(self.args.tr_batch,12*24*2)
#         h1_input=torch.cat([h0_vec,Y_vec,X0_vec],dim=1)
#         h1_hat=Hls + self.netH(h1_input).reshape(self.args.tr_batch, 12,24,4,2)


#         HtH_norm= cf_XtX(h1_hat)
#         # print('HtH_norm.shape=',HtH_norm.shape) # BS x 12 x 24 x 1
#         H_conj_Y=cf_XtY(h1_hat, Received_Y)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         HtY_vec=H_conj_Y.reshape(self.args.tr_batch,12*24*2)

#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         Le=torch.mul(HtH_norm, H_conj_Y)
#         Le_vec=Le.reshape(self.args.tr_batch,12*24*2)

#         # rec_X=H_conj_Y/sigma2+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)
#         # rec_X=H_conj_Y/sigma2 - Le/(sigma2**2)+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)

#         H_vecnorm = torch.div(1.0, HtH_norm + sigma2.unsqueeze(-1))
#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_vecnorm.shape=',H_vecnorm.shape)
#         # err=torch.mean(H_vecnorm-HtH_norm)
#         # print('H_err=',err)
#         MMSE_X= torch.mul(H_vecnorm, H_conj_Y)
#         MMSE_X_vec= MMSE_X.reshape(self.args.tr_batch,12*24*2)
#         # print('MMSE_X.shape=',MMSE_X.shape)

#         XNet_input=torch.cat([HtY_vec, Le_vec, MMSE_X_vec], dim=1)
#         rec_X = MMSE_X + self.netX(XNet_input).reshape(self.args.tr_batch, 12,24,2)

#         X1 = torch.complex(rec_X[...,0], rec_X[...,1])

#         # return h1_hat, h2_hat, X1, Recover_X
#         return h1_hat, X1


class CNN1D(nn.Module):
    def __init__(self, inter_chan_list, in_chan=2, out_chan=12):
        super(CNN1D,self).__init__()
        layer = []
        print('inter_chan_list=',inter_chan_list)

        layer.append(nn.Conv1d(in_channels=in_chan, out_channels=inter_chan_list[0], kernel_size=3, padding=1))
        # layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.Tanh())
        # layer.append(nn.ReLU())
        layer.append(nn.PReLU())
        # layer.append(Sine())
        for ii in range(len(inter_chan_list)-1):
            layer.append(nn.Conv1d(inter_chan_list[ii], inter_chan_list[ii+1],kernel_size=3, padding=1, bias=True))
            # layer.append(nn.Dropout(p=dpr))
            # layer.append(nn.ReLU())
            # layer.append(nn.Tanh())
            layer.append(nn.PReLU())
            # layer.append(Sine())

        ii=len(inter_chan_list)-1
        # layer.append(nn.Linear(inter_chan_list[0],inter_chan_list[1],bias=True))
        # layer.append(nn.ReLU())
        layer.append(nn.Conv1d(inter_chan_list[ii],out_chan,kernel_size=3, padding=1, bias=True))
        # layer.append(nn.Dropout(p=dpr))
        # layer.append(nn.PReLU())
        # layer.append(nn.Tanh())
        # layer.append(nn.BatchNorm1d(out_chan, affine=False)) # Without Learnable Parameters
        # layer.append(nn.Tanh())
        self.fcn = nn.Sequential(*layer)

    def forward(self, x):
        out = self.fcn(x)
        return out




import torch.nn.functional as F

# # MIMO-Dec12-v1631-v1
# class MIMONet(nn.Module):
#     def __init__(self, args, MLP1=FCN5, MLP2=FCN7):
#         super(MIMONet,self).__init__()
#         self.args = args
#         inter_chan1=self.args.fcn1_chan
#         inter_chan2=self.args.fcn2_chan
#         NN_depth=self.args.fcn_depth
#         inter_chan_H=np.array(self.args.inter_H).astype(int)
#         inter_chan_X=np.array(self.args.inter_X).astype(int)
#         inter_chan_X2=np.array(self.args.inter_X2).astype(int)
#         # detectors=np.array(self.args.sino_size[1]).astype(int)

#         # print('inter_chan_H=',np.array(inter_chan_H).astype(int))
#         # self.netH = MLP1(depth=NN_depth, in_chan=12*24*4*2+12*4*2+12*2, out_chan=12*24*4*2, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()
#         # self.netH = MLP1(depth=NN_depth, in_chan=12*24*2*5+12*4*2+1+1, out_chan=12*24*4*2, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()
#         # self.netX = MLP2(inter_chan_X, depth=NN_depth, in_chan=12*24*2, out_chan=12*24*2, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()
        
#         # self.netH = MLP1(inter_chan_H,depth=NN_depth, in_chan=2*24*4, out_chan=12*24*4, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()
#         self.netX = MLP2(inter_chan_X, depth=NN_depth, in_chan=12*24*2*3, out_chan=12*24*2, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()
        
#         # self.netConv1D = nn.Conv1d(in_channels=2, out_channels=12, kernel_size=3,padding=1)
#         # self.netH = CNN1D(inter_chan_H, in_chan=2, out_chan=12).cuda() #v5
#         # self.netX2 = CNN1D(inter_chan_X2, in_chan=12, out_chan=12).cuda() #v6
#         # self.netX = CNN1D(inter_chan_X2, in_chan=36, out_chan=12).cuda() #V7

#     def forward(self, Received_Y, Ideal_H, X0, sigma2, dop):
#         Received_Y = Received_Y.detach()
#         hat_H = Ideal_H.detach()

#         HtH_norm= cf_HtH(hat_H)
#         # print('HtH_norm.shape=',HtH_norm.shape) # BS x 12 x 24 x 1
#         # H_conj_Y=cf_XtY(h1_hat, Received_Y)
#         H_conj_Y=cf_HtY(hat_H, Received_Y)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         HtY_vec=H_conj_Y.reshape(self.args.tr_batch,12*24*2)

#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         Le=torch.mul(HtH_norm, H_conj_Y)
#         Le_vec=Le.reshape(self.args.tr_batch,12*24*2)

#         # rec_X=H_conj_Y/sigma2+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)
#         # rec_X=H_conj_Y/sigma2 - Le/(sigma2**2)+self.net3(XNet_input).reshape(self.args.tr_batch, 12,24,2)

#         H_vecnorm = torch.div(1.0, HtH_norm + sigma2.unsqueeze(-1))
#         # print('HtH_norm.shape=',HtH_norm.shape)
#         MMSE_X= torch.mul(H_vecnorm, H_conj_Y)
#         MMSE_X_vec= MMSE_X.reshape(self.args.tr_batch,12*24*2)
#         # print('MMSE_X.shape=',MMSE_X.shape)

#         XNet_input=torch.cat([HtY_vec, Le_vec, MMSE_X_vec], dim=1)
#         # XNet_input=torch.cat([ MMSE_X_vec], dim=1)
#         # rec_X = MMSE_X + self.netX(XNet_input).reshape(self.args.tr_batch, 12,24,2)
#         # rec_X = MMSE_X + self.netX(XNet_input.reshape(self.args.tr_batch,36,-1)).reshape(self.args.tr_batch, 12,24,2)  #v7
#         rec_X = MMSE_X + self.netX(XNet_input.reshape(self.args.tr_batch,-1)).reshape(self.args.tr_batch, 12,24,2)  #v7
#         #rec_X = MMSE_X 

#         X1 = torch.complex(rec_X[...,0], rec_X[...,1])
#         return hat_H, X1


# # learn detector, dec 12
# class MIMONet(nn.Module):
#     def __init__(self, args, MLP1=FCN5, MLP2=FCN7):
#         super(MIMONet,self).__init__()
#         self.args = args
#         inter_chan1=self.args.fcn1_chan
#         inter_chan2=self.args.fcn2_chan
#         NN_depth=self.args.fcn_depth
#         inter_chan_H=np.array(self.args.inter_H).astype(int)
#         inter_chan_X=np.array(self.args.inter_X).astype(int)
#         inter_chan_X2=np.array(self.args.inter_X2).astype(int)

#         # self.netH = MLP1(inter_chan_H,depth=NN_depth, in_chan=2*24*4, out_chan=12*24*4, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()
#         self.netX = MLP2(inter_chan_X, depth=NN_depth, in_chan=12*24*2*3, out_chan=12*24*2, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()

#     def forward(self, Received_Y, Ideal_H, X0, sigma2, dop):
#         Received_Y = Received_Y.detach()
#         hat_H = Ideal_H.detach()

#         HtH_norm= cf_HtH(hat_H)
#         # print('HtH_norm.shape=',HtH_norm.shape) # BS x 12 x 24 x 1
#         # H_conj_Y=cf_XtY(h1_hat, Received_Y)
#         H_conj_Y=cf_HtY(hat_H, Received_Y)/sigma2.unsqueeze(-1)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         HtY_vec=H_conj_Y.reshape(self.args.tr_batch,12*24*2)

#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         Le=torch.mul(HtH_norm, H_conj_Y)/ sigma2.unsqueeze(-1)**2
#         Le_vec=Le.reshape(self.args.tr_batch,12*24*2)

#         H_vecnorm = torch.div(1.0, HtH_norm + sigma2.unsqueeze(-1))
#         # print('HtH_norm.shape=',HtH_norm.shape)
#         MMSE_X= torch.mul(H_vecnorm, H_conj_Y)
#         MMSE_X_vec= MMSE_X.reshape(self.args.tr_batch,12*24*2)
#         # print('MMSE_X.shape=',MMSE_X.shape)

#         XNet_input=torch.cat([HtY_vec, Le_vec, MMSE_X_vec], dim=1)
#         rec_X = MMSE_X + self.netX(XNet_input.reshape(self.args.tr_batch,-1)).reshape(self.args.tr_batch, 12,24,2)  #v7
#         # rec_X = MMSE_X 

#         X1 = torch.complex(rec_X[...,0], rec_X[...,1])
#         return hat_H, X1


# # learn channel estimator, dec 13
# class MIMONet(nn.Module):
#     def __init__(self, args, MLP1=FCN5, MLP2=FCN7):
#         super(MIMONet,self).__init__()
#         self.args = args
#         inter_chan1=self.args.fcn1_chan
#         inter_chan2=self.args.fcn2_chan
#         # NN_depth=self.args.fcn_depth
#         inter_chan_H=np.array(self.args.inter_H).astype(int)
#         inter_chan_X=np.array(self.args.inter_X).astype(int)
#         inter_chan_X2=np.array(self.args.inter_X2).astype(int)

#         # self.netH = MLP1(inter_chan_H, in_chan=2*24*4, out_chan=12*24*4, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()
#         # self.netH = MLP1(inter_chan_H, in_chan=14*24*4*2, out_chan=12*24*4*2, inter_chan=inter_chan1, dpr=self.args.dpr).cuda()
#         # self.netX = MLP2(inter_chan_X, in_chan=12*24*2*3, out_chan=12*24*2, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()
        
#         self.netH = MLP2(inter_chan_H, in_chan=7, out_chan=6, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()
#         # self.netX = MLP2(inter_chan_X, in_chan=6, out_chan=2, inter_chan=inter_chan2, dpr=self.args.dpr).cuda()

#     def forward(self, Received_Y, Hls, X0, sigma2, dop):
#         Received_Y = Received_Y.detach()
#         Hls = Hls.detach()

#         Y0 = Received_Y[:,0:1,::2,...]
#         X0 = X0[:,0:1,::2,...]
#         YX0= cf_XtY(X0.unsqueeze(-2), Y0)
#         # print('X0.shape=',X0.shape)
#         XtX= cf_XtX(X0).unsqueeze(-1)
#         # print('YX0.shape=',YX0.shape)
#         # print('XtX.shape=',XtX.shape)
#         # print('sigma2.shape=',sigma2.shape)
#         H0 = cf_divR(YX0, XtX + sigma2.unsqueeze(-1))
#         # print('H0.shape=',H0.shape)
#         # H00=torch.nn.functional.interpolate(H0, size=[24,4,2])
#         H00=torch.nn.functional.interpolate(H0, size=[24,4,2],mode='nearest') # default:nearest, 'linear' | 'bilinear' | 'bicubic' |'trilinear' | 'area'
#         # print('H00.shape=',H00.shape)
#         Hls_sr=torch.nn.functional.interpolate(Hls[:,0:1,::2,...], size=[24,4,2])
#         # print('Hls_sr.shape=',Hls_sr.shape)

#         # h1_input=torch.cat([H00, Hls_sr], dim=1)
#         h1_input=torch.cat([H00, Hls_sr, Received_Y/torch.sqrt(torch.tensor(2.0, device=torch.device('cuda:0')))], dim=1)
#         # print('h1_input.shape=',h1_input.shape)
#         # print('h1_input.shape=',h1_input.shape)
#         # h1_input=self.netConv1D(h1_input.permute(0,4,2,3,1))  # bs, 12, 24, 4, 2
#         # hat_H=H00+self.netH(h1_input.permute(0,4,2,3,1).reshape(self.args.tr_batch,2,-1)).reshape(self.args.tr_batch,2,12,24,4).permute(0,2,3,4,1)
#         # print('hat_H.shape=',hat_H.shape)
#         # h1_input=h1_input.permute(0,4,2,3,1).reshape(self.args.tr_batch,2,-1)
#         # h1_input=h1_input.permute(0,4,2,3,1).reshape(self.args.tr_batch,-1)
#         h1_input=h1_input.reshape(self.args.tr_batch,7,16,24)
#         # h1_input=h1_input.reshape(self.args.tr_batch,2,-1)
#         # hat_H = self.netConv1D(h1_input).reshape(self.args.tr_batch,12,24,4,2)
#         hat_H = H00 + self.netH(h1_input).reshape(self.args.tr_batch,12,24,4,2)
#         # hat_H = self.netH(h1_input).reshape(self.args.tr_batch,12,24,4,2)


#         HtH_norm= cf_HtH(hat_H)
#         # print('HtH_norm.shape=',HtH_norm.shape) # BS x 12 x 24 x 1
#         # H_conj_Y=cf_XtY(h1_hat, Received_Y)
#         # H_conj_Y=cf_HtY(hat_H, Received_Y)/sigma2.unsqueeze(-1)
#         H_conj_Y=cf_HtY(hat_H, Received_Y) #/sigma2.unsqueeze(-1)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         # HtY_vec=H_conj_Y.reshape(self.args.tr_batch,12*24*2)

#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         # Le=torch.mul(HtH_norm, H_conj_Y)/ sigma2.unsqueeze(-1)**2
#         # Le_vec=Le.reshape(self.args.tr_batch,12*24*2)

#         # H_vecnorm = torch.div(1.0, HtH_norm + 4.0*sigma2.unsqueeze(-1))
#         H_vecnorm = torch.div(1.0, HtH_norm/4.0 + sigma2.unsqueeze(-1))
#         # print('HtH_norm.shape=',HtH_norm.shape)
#         MMSE_X= torch.mul(H_vecnorm, H_conj_Y/4.0)
#         # MMSE_X_vec= MMSE_X.reshape(self.args.tr_batch,12*24*2)
#         # print('MMSE_X.shape=',MMSE_X.shape)

#         # XNet_input=torch.cat([HtY_vec, Le_vec, MMSE_X_vec], dim=1)
#         # rec_X = MMSE_X + self.netX(XNet_input.reshape(self.args.tr_batch, 6, 12, 24)).reshape(self.args.tr_batch, 12,24,2)  #v7
#         rec_X = MMSE_X

#         X1 = torch.complex(rec_X[...,0], rec_X[...,1])
#         return hat_H, X1


# # learn channel estimator, dec 13
# class MIMONet(nn.Module):
#     def __init__(self, args, MLP1=FCN5, MLP2=FCN7):
#         super(MIMONet,self).__init__()
#         self.args = args
#         inter_chan1=self.args.fcn1_chan
#         inter_chan2=self.args.fcn2_chan
#         # NN_depth=self.args.fcn_depth
#         inter_chan_H=np.array(self.args.inter_H).astype(int)
#         inter_chan_X=np.array(self.args.inter_X).astype(int)
#         inter_chan_X2=np.array(self.args.inter_X2).astype(int)

#         self.netH=DeepRX(18,8,1)

#     def forward(self, Received_Y, Hls, Transmit_X, sigma2, dop):
#         Received_Y = Received_Y.detach()
#         Hls = Hls.detach()

#         Y0 = Received_Y[:,0:1,::2,...]
#         X0 = Transmit_X[:,0:1,::2,...]
#         YX0= cf_XtY(X0.unsqueeze(-2), Y0)
#         XtX= cf_XtX(X0).unsqueeze(-1)
#         H0 = cf_divR(YX0, XtX + sigma2.unsqueeze(-1))
#         H00=torch.nn.functional.interpolate(H0, size=[24,4,2],mode='nearest') # default:nearest, 'linear' | 'bilinear' | 'bicubic' |'trilinear' | 'area'

#         # h1_input=torch.cat([Received_Y, Hls, X0], dim=1)
#         input_x = torch.cat([Received_Y.reshape(self.args.tr_batch,8,12,24), Transmit_X.reshape(self.args.tr_batch,2,12,24), Hls.reshape(self.args.tr_batch,8,12,24)], axis=1) # channel = (8+2) + 8 = 10 +8 = 18 (first 10 mainly used)
#         # print('h1_input.shape=',h1_input.shape)
#         # h1_input=h1_input.reshape(self.args.tr_batch,7,16,24)
#         # input_x=input_x.reshape(self.args.tr_batch,18,12,24)
#         hat_H = H00 + self.netH(input_x).reshape(self.args.tr_batch,12,24,4,2)
#         # hat_H = self.netH(h1_input).reshape(self.args.tr_batch,12,24,4,2)
#         # print('hat_H.shape=',hat_H.shape) # BS x 12 x 24 x 4 x 2


#         HtH_norm= cf_HtH(hat_H)
#         # print('HtH_norm.shape=',HtH_norm.shape) # BS x 12 x 24 x 1
#         # H_conj_Y=cf_XtY(h1_hat, Received_Y)
#         # H_conj_Y=cf_HtY(hat_H, Received_Y)/sigma2.unsqueeze(-1)
#         H_conj_Y=cf_HtY(hat_H, Received_Y) #/sigma2.unsqueeze(-1)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         # HtY_vec=H_conj_Y.reshape(self.args.tr_batch,12*24*2)

#         # print('HtH_norm.shape=',HtH_norm.shape)
#         # print('H_conj_Y.shape=',H_conj_Y.shape)
#         # Le=torch.mul(HtH_norm, H_conj_Y)/ sigma2.unsqueeze(-1)**2
#         # Le_vec=Le.reshape(self.args.tr_batch,12*24*2)

#         # H_vecnorm = torch.div(1.0, HtH_norm + 4.0*sigma2.unsqueeze(-1))
#         H_vecnorm = torch.div(1.0, HtH_norm/4.0 + sigma2.unsqueeze(-1))
#         # print('HtH_norm.shape=',HtH_norm.shape)
#         MMSE_X= torch.mul(H_vecnorm, H_conj_Y/4.0)
#         # MMSE_X_vec= MMSE_X.reshape(self.args.tr_batch,12*24*2)
#         # print('MMSE_X.shape=',MMSE_X.shape)

#         # XNet_input=torch.cat([HtY_vec, Le_vec, MMSE_X_vec], dim=1)
#         # rec_X = MMSE_X + self.netX(XNet_input.reshape(self.args.tr_batch, 6, 12, 24)).reshape(self.args.tr_batch, 12,24,2)  #v7
#         rec_X = MMSE_X

#         X1 = torch.complex(rec_X[...,0], rec_X[...,1])
#         return hat_H, X1






# learn channel estimator, dec 13
class MIMONet(nn.Module):
    def __init__(self, args, MLP1=FCN5, MLP2=FCN7):
        super(MIMONet,self).__init__()
        self.args = args
        inter_chan1=self.args.fcn1_chan
        inter_chan2=self.args.fcn2_chan
        # NN_depth=self.args.fcn_depth
        inter_chan_H=np.array(self.args.inter_H).astype(int)
        inter_chan_X=np.array(self.args.inter_X).astype(int)
        inter_chan_X2=np.array(self.args.inter_X2).astype(int)

        # self.netH=DeepRXH(18,8,1, n_chan=32)
        # self.netX=DeepRX(18,2,1, n_chan=64)
        self.netX=DeepRX(18,2,1, n_chan=self.args.deepcnn)

    def forward(self, Received_Y, Hls, Transmit_X, sigma2, dop):
        Received_Y = Received_Y.detach()
        Hls = Hls.detach()

        # Y0 = Received_Y[:,0:1,::2,...]
        # X0 = Transmit_X[:,0:1,::2,...]
        # YX0= cf_XtY(X0.unsqueeze(-2), Y0)
        # XtX= cf_XtX(X0).unsqueeze(-1)
        # H0 = cf_divR(YX0, XtX + sigma2.unsqueeze(-1))
        # H00=torch.nn.functional.interpolate(H0, size=[24,4,2],mode='nearest') # default:nearest, 'linear' | 'bilinear' | 'bicubic' |'trilinear' | 'area'

        # input_h = torch.cat([Received_Y.reshape(self.args.tr_batch,8,12,24), Transmit_X.reshape(self.args.tr_batch,2,12,24), Hls.reshape(self.args.tr_batch,8,12,24)], axis=1) # channel = (8+2) + 8 = 10 +8 = 18 (first 10 mainly used)

        # # hat_H = H00 + self.netH(input_h).reshape(self.args.tr_batch,12,24,4,2)
        # hat_H = torch.zeros_like(Hls)
        # hat_H[:,:,...]= H00

        # HtH_norm= cf_HtH(hat_H)
        # # print('HtH_norm.shape=',HtH_norm.shape) # BS x 12 x 24 x 1
        # # H_conj_Y=cf_XtY(h1_hat, Received_Y)
        # # H_conj_Y=cf_HtY(hat_H, Received_Y)/sigma2.unsqueeze(-1)
        # H_conj_Y=cf_HtY(hat_H, Received_Y) #/sigma2.unsqueeze(-1)

        # H_vecnorm = torch.div(1.0, HtH_norm/4.0 + sigma2.unsqueeze(-1))
        # # print('HtH_norm.shape=',HtH_norm.shape)
        # MMSE_X= torch.mul(H_vecnorm, H_conj_Y/4.0)
        # # MMSE_X_vec= MMSE_X.reshape(self.args.tr_batch,12*24*2)
        # # print('MMSE_X.shape=',MMSE_X.shape)

        # # XNet_input=torch.cat([HtY_vec, Le_vec, MMSE_X_vec], dim=1)
        # # rec_X = MMSE_X + self.netX(XNet_input.reshape(self.args.tr_batch, 6, 12, 24)).reshape(self.args.tr_batch, 12,24,2)  #v7
        
        # input_x = torch.cat([Received_Y.reshape(self.args.tr_batch,8,12,24),MMSE_X.reshape(self.args.tr_batch,2,12,24)], axis=1) # channel = (8+2) + 8 = 10 +8 = 18 (first 10 mainly used)
        Received_Y_4 = Received_Y.permute(3, 0, 4, 1, 2) #before:25000, 12, 24, 4, 2
        Received_Y = torch.cat([Received_Y_4[0,:,:,:,:], Received_Y_4[1,:,:,:,:], Received_Y_4[2,:,:,:,:], Received_Y_4[3,:,:,:,:]], axis=1)

        Hls_4 = Hls.permute(3, 0, 4, 1, 2) #before:25000, 12, 24, 4, 2
        Hls = torch.cat([Hls_4[0,:,:,:,:], Hls_4[1,:,:,:,:], Hls_4[2,:,:,:,:], Hls_4[3,:,:,:,:]], axis=1)

        input_x = torch.cat([Received_Y, Hls, Transmit_X.permute(0,3,1,2)], axis=1) # channel = (8+2) + 8 = 10 +8 = 18 (first 10 mainly used)
        # rec_X = MMSE_X + self.netX(input_x).reshape(self.args.tr_batch,12,24,2)
        rec_X = self.netX(input_x).permute(0,2,3,1) # before bs, 2, 12, 24

        X1 = torch.complex(rec_X[...,0], rec_X[...,1])
        # return hat_H, X1
        return X1, X1





