import math
import time

import numpy as np
import scipy.special as scipy
import torch
from torch import nn


def shift(A, f, tau):
    '''
    -1000<=f<=1000
     -72<=tau<=72
    '''
    NumSample,S,F,Nr,_ = A.shape
    data = A[...,0] + 1j*A[...,1]
    consts = (2048+144)/(2048*15e3)
    shift_s = torch.exp(1j*2*math.pi*f*consts*torch.arange(S)).cuda()
    shift_f = torch.exp(-1j*2*math.pi*tau*torch.arange(F)/2048).cuda()
    data_1 = (data.permute(0,3,2,1)*shift_s).permute(0,3,2,1)
    data_1 = (data_1.permute(0,1,3,2)*shift_f).permute(0,1,3,2)
    data_1 = torch.stack([data_1.real,data_1.imag],dim=4)
    return data_1



def cdiv(x, y):
    div = torch.zeros_like(x)
    y_abs = y[:,0,::2,:,0]**2 + y[:,0,::2,:,1]**2
    div[:,0,::2,:,0] = (x[:,0,::2,:,0] * y[:,0,::2,:,0] + x[:,0,::2,:,1] * y[:,0,::2,:,1]) / y_abs
    div[:,0,::2,:,1] = (x[:,0,::2,:,1] * y[:,0,::2,:,0] - x[:,0,::2,:,0] * y[:,0,::2,:,1]) /y_abs
    return div



def toeplitz(c, r):
    n = c.size(0)
    a = torch.arange(n)+1
    A = torch.tile(a,(n,1))
    I = A-A.T
    vecI = torch.reshape(I,(-1,))+n-1
    h = torch.cat((torch.flip(c,[0]),r[1:n]),0)
    R = torch.reshape(h[vecI],(n,n))
    return R

def batch_toeplitz(c, r):
    batchsize, n = c.shape[0], c.shape[1]
    a = torch.arange(n)+1
    A = torch.tile(a,(n,1))
    I = A-A.T
    vecI = torch.reshape(I,(-1,))+n-1
    vecIflat = torch.tile(vecI,(1,batchsize))[0,:]
    h = torch.cat((torch.flip(c,[1]),r[:,1:n]),1)
    hflat = torch.flatten(h)
    b = torch.arange(batchsize).long()
    d = torch.ones(n**2).long()
    id = torch.flatten(torch.matmul(b[:,None],d[None,:]))
    vecIflat = (vecIflat + (2*n-1)*id).long()
    R = torch.reshape(hflat[vecIflat],(batchsize,n,n))
    return R


def ber(Recover_X, Ideal_X):
    batchsize, S, F = Recover_X.shape[0], Recover_X.shape[1], Recover_X.shape[2]
    Ideal_X = Ideal_X.float()
    Recover_X_id = (torch.sign(
        torch.stack([-Recover_X.real, -Recover_X.imag], dim=-1)) + 1) / 2
    Recover_X_id[:, 0, ::2, :] = Ideal_X[:, 0, ::2, :]
    ber = (Ideal_X != Recover_X_id).sum() / (batchsize * (S - 0.5) * F * 2)
    return ber
    
    
    
def bler(H_full, Recover_X, Ideal_X, sigma2):
    batchsize, S, F = H_full.shape[0], H_full.shape[1], H_full.shape[2]
    H_vecnorm = torch.div(1,torch.linalg.norm(H_full,dim=(3,4))**2 + sigma2)
    G = torch.mul(H_vecnorm, torch.linalg.norm(H_full,dim=(3,4))**2)
    eps2 = torch.mul(G, 1-G)
    Recover_Xc = torch.div(torch.mul(-2*torch.sqrt(torch.tensor([2.0]).cuda())*Recover_X,G),eps2)
    llr = torch.clamp(torch.stack([Recover_Xc.real,Recover_Xc.imag],dim=-1),min=-10,max=10)
    Ideal_X = Ideal_X.float()
    ce = -torch.mul(Ideal_X,torch.log(torch.div(torch.exp(llr),1+torch.exp(llr))))-torch.mul(1-Ideal_X,torch.log(torch.div(1,1+torch.exp(llr))))
    ce = torch.mean(ce,dim=(1,2,3))
    ce = torch.reshape(ce,(int(batchsize/25),25))
    ce = torch.mean(ce,1)
    xp = [0,0.415,0.42,0.425,0.43,0.435,0.44,1]
    fp = [0,0,0.0454545454545455,0.21875,0.576923076923077,0.84,1,1]
    bler = np.mean(np.interp(ce.cpu(),xp,fp))
    
    return bler


class mmsenet():
    def __init__(self, args):
        self.layers = args.layers
        self.noise = args.noise
        self.net = MmseNet(args)
        self.args = args
        if args.parallel == True:
            self.net = nn.DataParallel(self.net)
        self.net = self.net.cuda()

    def tr_model(self, Received_Y, Hls, Transmit_X):
        self.net.train()
        H_full,Recover_X, sigma2 = self.net(Received_Y, Hls, Transmit_X)
        return H_full, Recover_X, sigma2

    def test_model(self, Received_Y,  Hls, Transmit_X):
        with torch.no_grad():
            self.net.eval()
            H_full,Recover_X, sigma2 = self.net(Received_Y.cuda(), Hls.cuda(),Transmit_X.cuda())
        return H_full, Recover_X, sigma2

    def vl_model(self, Received_Y,  Hls, Ideal_H, Ideal_X, Transmit_X, n_iter):
        with torch.no_grad():
            self.net.eval()
            #id_valid = torch.arange(60000,75000).long()
            ff = 0
            for j in range(6):
                id_valid = torch.arange(270000+j,360000,6).long()
                Received_Y_valid = Received_Y[id_valid,:,:,:,:].cuda()
                Hls_valid = Hls[id_valid,:,:,:,:].cuda()
                Ideal_H_valid = Ideal_H[id_valid,:,:,:,:].cuda()
                Ideal_X_valid = Ideal_X[id_valid,:,:,:].cuda()
                Transmit_X_valid = Transmit_X[id_valid,:,:,:].cuda()

                #Received_Y_valid = shift(Received_Y_valid,ff,0)
                #Hls_valid = cdiv(Received_Y_valid,Transmit_X_valid[:,:,:,None,:].repeat(1,1,1,4,1))

                batchsize = 100
                ber_vl = 0.0
                bler_vl = 0.0
                for i in range(150):
                    batch_idx = torch.arange(100*i,100*(i+1),1)
                    Received_Y_batch = Received_Y_valid[batch_idx,:,:,:,:].cuda()
                    Hls_batch = Hls_valid[batch_idx,:,:,:,:].cuda()
                    Ideal_H_batch = Ideal_H_valid[batch_idx,:,:,:,:].cuda()
                    Ideal_X_batch = Ideal_X_valid[batch_idx,:,:,:].cuda()
                    Transmit_X_batch = Transmit_X_valid[batch_idx,:,:,:].cuda()

                    H_full_batch,Recover_X_batch,sigma2 = self.net(Received_Y_batch,  Hls_batch, Transmit_X_batch)
                    ber_vl = ber_vl + ber(Recover_X_batch,Ideal_X_batch)
                    bler_vl = bler_vl + bler(H_full_batch,Recover_X_batch,Ideal_X_batch,sigma2)
                ber_vl = ber_vl/(150)
                bler_vl = bler_vl/(150)
                print('{:d} BER = {:.6f}, BLER = {:.6f}'.format(j,ber_vl,bler_vl.item()))

    def load_model(self, args):
        ckp = torch.load(args.test_ckp_dir, map_location=lambda storage, loc: storage.cuda(args.gpu_idx))
        self.net.load_state_dict(ckp['model'])
        return ckp


class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dilation):
        super(BasicBlock,self).__init__()
        if in_channels!=out_channels:
           self.flag = 1
           self.conv5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=(1,1), padding=(0,0), bias=True)
        else:
           self.flag = 0
        kernel_size = (3,3)
        padding = (1,1)
        #self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, padding=dilation, dilation=dilation, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                kernel_size=(1,1), padding=(0,0), bias=True)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                kernel_size=kernel_size, padding=dilation, dilation=dilation, bias=True)
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




class hypernetf(nn.Module):
    def __init__(self, depth=6):
        super(hypernetf, self).__init__()
        #in_chan = [64,64,64,128,128,256,256,256,128,128,64]
        #out_chan = [64,64,128,128,256,256,256,128,128,64,64]
        in_chan = [64,64,128,256,128,64]
        out_chan = [64,128,256,128,64,64]
        dlt = [1,1,2,3,3,6,2,3,1,1,1,1]
        kernel_size = (3,3)
        padding = (1,1)
        layers = []
        layers.append(nn.Conv2d(in_channels=18, out_channels=64, 
            kernel_size=kernel_size, padding=(1,1),bias=True))
        for i in range(depth):
            layers.append(BasicBlock(in_channels=in_chan[i],out_channels=out_chan[i],dilation=(dlt[2*i],dlt[2*i+1])))
        layers.append(nn.Conv2d(in_channels=64, out_channels=2, 
            kernel_size=(1,1), padding=(0,0), bias=True))
        layers.append(nn.Sigmoid())
        self.hypernetf = nn.Sequential(*layers)

    def forward(self, x):
        out = self.hypernetf(x)
        out = torch.mean(out,dim=2)
        return out

class hypernets(nn.Module):
    def __init__(self, depth=6):
        super(hypernets, self).__init__()
        #in_chan = [64,64,64,128,128,256,256,256,128,128,64]
        #out_chan = [64,64,128,128,256,256,256,128,128,64,64]
        in_chan = [64,64,128,256,128,64]
        out_chan = [64,128,256,128,64,64]
        dlt = [1,1,2,3,3,6,2,3,1,1,1,1]
        kernel_size = (3,3)
        padding = (1,1)
        layers = []
        layers.append(nn.Conv2d(in_channels=18, out_channels=64, 
            kernel_size=kernel_size, padding=(1,1),bias=True))
        for i in range(depth):
            layers.append(BasicBlock(in_channels=in_chan[i],out_channels=out_chan[i],dilation=(dlt[2*i],dlt[2*i+1])))
        layers.append(nn.Conv2d(in_channels=64, out_channels=2, 
            kernel_size=(1,1), padding=(0,0), bias=True))
        layers.append(nn.Sigmoid())
        self.hypernets = nn.Sequential(*layers)

    def forward(self, x):
        out = self.hypernets(x)
        out = torch.mean(out,dim=3)
        return out

class hypernet(nn.Module):
    def __init__(self,depth=2,in_chan=18,n_channels=32):
        super().__init__()
        kernel_size = (3,3)
        padding = 1
        layers = []
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.act = nn.PReLU()
        self.fc = nn.Linear(288*n_channels,17)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act(out)
        out = self.fc(out.view(out.shape[0],288*32))
        return out



class MmseNet(nn.Module):
    def __init__(self, args):
        super(MmseNet,self).__init__()
        self.args = args
        self.net = nn.ModuleList()
        '''
        for i in range(24):
            self.net = self.net.append(dncnn().cuda())'''
        #self.hynet = cnn().cuda()
        self.net = self.net.append(hypernet().cuda())
        self.net = self.net.append(hypernetf().cuda())
        self.net = self.net.append(hypernets().cuda())
        self.noise = args.noise
        #self.sigma2 = np.power(10,-self.noise/10)
        #self.gamma = nn.Parameter(torch.ones(6))
        #self.rho = nn.Parameter(torch.ones(5))
        #self.creal = nn.Parameter(torch.ones(24))
        #self.cimag = nn.Parameter(torch.zeros(23))
        S = 12
        F = 24
        #c = [1.000 + 0.000j,0.9991 - 0.02380j ,0.9965 - 0.04740j  ,0.9922 - 0.07050j,0.9862 - 0.09310j ,0.9788 - 0.1149j  ,0.9700 - 0.1357j   ,0.9599 - 0.1554j   ,0.9488 - 0.1740j   ,0.9369 - 0.1914j ,0.9243 - 0.2075j ,0.9111 - 0.2223j   ,0.8976 - 0.2358j   ,0.8839 - 0.2482j   ,0.8701 - 0.2594j ,0.8564 - 0.2695j ,0.8429 - 0.2787j   ,0.8296 - 0.2870j   ,0.8165 - 0.2945j   ,0.8037 - 0.3013j ,0.7913 - 0.3076j ,0.7791 - 0.3134j   ,0.7672 - 0.3188j   ,0.7556 - 0.3238j]
        #c = torch.from_numpy(np.array(c))
        #self.Rf = torch.complex(toeplitz(c.real,c.real),toeplitz(c.imag,-c.imag)).cuda()
        #doppler = args.doppler
        #s = np.arange(0,12,1)
        #dt = 1e-3/12
        #j0 = scipy.jv(0,2*math.pi*doppler*s*dt)
        #j0 = torch.from_numpy(j0)
        #self.Rs = toeplitz(j0,j0).cuda()


    def forward(self, Received_Y, Hls, Transmit_X):
    
        #net = hypernet()
        #print(sum(param.numel() for param in net.parameters() if param.requires_grad))
        
        S = 12
        F = 24
        batchsize = Hls.shape[0]
        Received_Y = Received_Y.detach()
        Hls = Hls.detach()
        Received_Yc = torch.complex(Received_Y[:,:,:,:,0],Received_Y[:,:,:,:,1])
        Received_Yflat = Received_Y.view(Received_Y.shape[0],S,F,8).permute(0,3,1,2).float()
        Hlsflat = Hls.view(Hls.shape[0],S,F,8).permute(0,3,1,2).float()
        Transmit_Xflat = Transmit_X.permute(0,3,1,2).float()
        output = self.net[0](torch.cat([Received_Yflat, Transmit_Xflat, Hlsflat], dim=1))
        c = self.net[1](torch.cat([Received_Yflat, Transmit_Xflat, Hlsflat], dim=1))
        j0 = self.net[2](torch.cat([Received_Yflat, Transmit_Xflat, Hlsflat], dim=1))
        Rf = torch.complex(batch_toeplitz(c[:,0,:],c[:,0,:]),batch_toeplitz(c[:,1,:],-c[:,1,:]))
        Rs = torch.complex(batch_toeplitz(j0[:,0,:],j0[:,0,:]),batch_toeplitz(j0[:,1,:],-j0[:,1,:]))


        Rf1 = Rf[:,::2,::2]
        Rf12 = Rf[:,:,::2]

        Rfeye1 = Rf1 + torch.einsum('ij,k->kij', torch.eye(int(F/2)).cuda(), output[:,0]**2)

        Wf_par = torch.matmul(Rf12,torch.linalg.inv(Rfeye1)).to(torch.complex128)
        Hlsc0 = torch.complex(Hls[:,0,::2,:,0],Hls[:,0,::2,:,1]).cuda()
        hc_symb0 = torch.matmul(Wf_par,Hlsc0).cuda()
        h_symb0 = torch.stack([hc_symb0.real,hc_symb0.imag],dim = 1).float().permute(0,2,3,1)[:,None,:,:,:]
        H = h_symb0.repeat(1,S,1,1,1)
        Hc = torch.complex(H[:,:,:,:,0],H[:,:,:,:,1])


        for i in range(5):
            sigmas = output[:,3*i+1]**2
            H_norm = torch.div(1,torch.linalg.norm(H,dim=(3,4))**2 + sigmas[:,None,None].repeat(1,S,F)).cuda()
            Recover_Xc = torch.mul(H_norm,torch.sum(torch.mul(torch.conj(Hc),Received_Yc),3))
            k = 10
            Recover_X_qpsk = torch.complex(torch.tanh(k*Recover_Xc.real),torch.tanh(k*Recover_Xc.imag))/math.sqrt(2)

            Hlsc = Received_Yc/Recover_X_qpsk[:,:,:,None].repeat(1,1,1,4)

            Rfeye = Rf + torch.einsum('ij,k->kij', torch.eye(int(F)).cuda(), output[:,3*i+2]**2)
            Wf = torch.matmul(Rf,torch.linalg.inv(Rfeye)).to(torch.complex128)
            Hlsc = torch.matmul(Wf[:,None,:,:].repeat(1,S,1,1),Hlsc).cuda()

            Rseye = Rs + torch.einsum('ij,k->kij', torch.eye(int(S)).cuda(), output[:,3*i+3]**2)
            Ws = torch.matmul(Rs,torch.linalg.inv(Rseye)).to(torch.complex128)
            Hc = torch.matmul(Ws[:,None,:,:].repeat(1,F,1,1),Hlsc.permute(0,2,1,3)).permute(0,2,1,3)
            H = torch.stack([Hc.real,Hc.imag],dim=4)

        sigmas = output[:,16]**2
        H_norm = torch.div(1,torch.linalg.norm(H,dim=(3,4))**2 + sigmas[:,None,None].repeat(1,S,F)).cuda()
        Recover_X = torch.mul(H_norm,torch.sum(torch.mul(torch.conj(Hc),Received_Yc),3))
        #print(output)




        return H,Recover_X,sigmas[:,None,None].repeat(1,S,F)
