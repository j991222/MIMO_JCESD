import re
import time

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (CosineAnnealingLR, ExponentialLR,
                                      MultiStepLR)


def perturb(A, ratio):
    NumSample,S,F,Nr,_ = A.shape
    sigma1 = ratio*(10**0)
    sigma2 = ratio*(10**(-1/4))
    mid = int(F/2)
    A[:,:,0:mid,:,:] = A[:,:,0:mid,:,:] + sigma1*torch.randn(NumSample,S,mid,Nr,2)
    A[:,:,mid:F,:,:] = A[:,:,mid:F,:,:] + sigma2*torch.randn(NumSample,S,mid,Nr,2)
    return A


def cdiv(x, y):
    div = torch.zeros_like(x)
    y_abs = y[:,0,::2,:,0]**2 + y[:,0,::2,:,1]**2
    div[:,0,::2,:,0] = (x[:,0,::2,:,0] * y[:,0,::2,:,0] + x[:,0,::2,:,1] * y[:,0,::2,:,1]) / y_abs
    div[:,0,::2,:,1] = (x[:,0,::2,:,1] * y[:,0,::2,:,0] - x[:,0,::2,:,0] * y[:,0,::2,:,1]) /y_abs
    return div

class Trainer():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.lr = args.lr
        self.epoch = args.epoch
        self.shuffle = args.shuffle
        self.batch_size = args.tr_batch
        self.ckp_dir = args.ckp_dir
        self.noise = args.noise
        

    def _set_optim(self):
        for p in list(self.model.net.parameters()):
            p.requires_grad = True

        para_lst=[]    
        for param in self.model.net.parameters():
            para_lst.append(param)
        

        optimizer = optim.AdamW(para_lst, lr=1e-7)
        #optimizer = optim.SGD(para_lst, lr=1e-4)
        #scheduler = MultiStepLR(optimizer, milestones=[40], gamma=0.5)  # learning rates CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer,T_max=15)

        return optimizer, scheduler


    def weights_init(self):
        for ll in range(self.args.layers):
            for name, param in self.model.net.named_parameters():
                if 'hypernet' in name:
                    torch.nn.init.zeros_(param)

    def gradnorm(self):
        total_norm = 0.0
        for param in self.model.net.parameters():
            total_norm = total_norm + param.grad.data.norm(2).item()**2
        total_norm = total_norm**(1/2)
        
        return total_norm


    def tr(self):
        self.optimizer, self.scheduler = self._set_optim()
        #self.weights_init()
        start = 0

        # Resume Training
        if self.args.resume == True:
            start = self.resume_tr()
            start += 1

        NumSample = 750000
        S = 12
        F = 24
        Received_Y = np.empty([NumSample,S,F,4,2])
        Hls = np.empty([NumSample,S,F,4,2])
        Ideal_H = np.empty([NumSample,S,F,4,2])
        Ideal_X = np.empty([NumSample,S,F,2])
        Transmit_X = np.empty([NumSample,S,F,2])


        db = [-10,-5,0,10,20,30]
        #db = -5

        for i in range(6):
            for Nr in range(4):
                data = np.load('/home/hcju/{}Hz/EVA_{}dB_{}Hz_R{}.npz'.format(self.args.doppler,db[i],self.args.doppler,Nr))
                Received_Y[i:NumSample:6,:,:,Nr,:] = data['Recived_Y']
                Hls[i:NumSample:6,:,:,Nr,:] = data['Hls']
                Ideal_H[i:NumSample:6,:,:,Nr,:] = data['Ideal_H']
                Ideal_X[i:NumSample:6,:,:,:]= data['Ideal_X']
                Transmit_X[i:NumSample:6,:,:,:]= data['Transmit_X']
            
        
        id = np.arange(0,NumSample,1)
        id = np.argwhere(id % 750 < 300)
        Received_Y = torch.from_numpy(np.delete(Received_Y,id,0))
        Hls = torch.from_numpy(np.delete(Hls,id,0))
        Ideal_H = torch.from_numpy(np.delete(Ideal_H,id,0))
        Ideal_X = torch.from_numpy(np.delete(Ideal_X,id,0))
        Transmit_X = torch.from_numpy(np.delete(Transmit_X,id,0))
        train_size = 270000
        id_train = torch.arange(train_size).float()

        #Received_Y = perturb(Received_Y,1.9925)
        #Hls = cdiv(Received_Y,Transmit_X[:,:,:,None,:].repeat(1,1,1,4,1))



        n_iter = 0


        self.model.vl_model(Received_Y, Hls, Ideal_H, Ideal_X, Transmit_X, n_iter)
        
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(start, self.epoch):
                t_start = time.time()
                idx = torch.randperm(train_size)
                #idx = torch.arange(0,train_size,1)
                epoch_loss = 0.0
                # One epoch training
                for i in range(train_size//self.batch_size):
                    self.optimizer.zero_grad()
                    #batch_idx = torch.multinomial(id_train,self.batch_size,False)
                    batch_idx = torch.arange(self.batch_size*i,self.batch_size*(i+1),1)
                    Received_Y_batch = Received_Y[idx[batch_idx],:,:,:,:].cuda()
                    Hls_batch = Hls[idx[batch_idx],:,:,:,:].cuda()
                    Ideal_H_batch = Ideal_H[idx[batch_idx],:,:,:,:].cuda()
                    Ideal_X_batch = Ideal_X[idx[batch_idx],:,:,:].cuda()
                    Transmit_X_batch = Transmit_X[idx[batch_idx],:,:,:].cuda()


                    H_full_batch,Recover_X_batch,sigma2 = self.model.tr_model(Received_Y_batch,  Hls_batch, Transmit_X_batch)
                    loss = self.mix_loss(H_full_batch.cuda(),Recover_X_batch.cuda(),Ideal_H_batch,Ideal_X_batch,sigma2)
                    epoch_loss = epoch_loss + loss
                    loss.backward()
                    #grad_norm = self.gradnorm()
                    #print(grad_norm)
                    self.optimizer.step()
                    n_iter += 1
                    '''
                    if n_iter % self.args.disp_freq == 0:
                        print('Epoch = {:d}, loss = {:.6f}'.format(epoch,loss.item()))
                        self.model.vl_model(Received_Y, Hls, Ideal_H, Ideal_X, n_iter)
'''


                epoch_loss = epoch_loss/(train_size//self.batch_size)
                t_end = time.time()
                print('Epoch {:d} time = {:.2f} loss = {:,.6f}'.format(epoch,t_end-t_start,epoch_loss.item()))
                self.model.vl_model(Received_Y, Hls, Ideal_H, Ideal_X, Transmit_X, n_iter)
                #grad_norm = self.gradnorm()
                #print(grad_norm)

                # Save models
                if (epoch+1) % 5  == 0 or epoch == self.epoch-1:
                    print('\n'+'--->'*10+'Save CKP')
                    self.save_ckp(epoch)
        return 0

    def mix_loss(self, H_full, Recover_X, Ideal_H, Ideal_X, sigma2):
        #alpha = 1.0

        #loss = alpha * (torch.linalg.norm(H_full-Ideal_H)**2/torch.numel(H_full))
        #loss = alpha * torch.linalg.norm(torch.flatten(H_full-Ideal_H),1)/torch.numel(H_full)
        #print('loss_h = %.6f' % loss.item())
        
        batchsize, S, F = H_full.shape[0], H_full.shape[1], H_full.shape[2]
        H_vecnorm = torch.div(1,torch.linalg.norm(H_full,dim=(3,4))**2 + sigma2)
        G = torch.mul(H_vecnorm, torch.linalg.norm(H_full,dim=(3,4))**2)
        eps2 = torch.mul(G, 1-G)
        Recover_Xc = torch.div(torch.mul(-2*torch.sqrt(torch.tensor([2.0]).cuda())*Recover_X,G),eps2)
        llr = torch.clamp(torch.stack([Recover_Xc.real,Recover_Xc.imag],dim=-1),min=-10,max=10)
        Ideal_X = Ideal_X.float()
        ce = -torch.mul(Ideal_X,torch.log(torch.div(torch.exp(llr),1+torch.exp(llr))))-torch.mul(1-Ideal_X,torch.log(torch.div(1,1+torch.exp(llr))))
        ce = torch.mean(ce)
        loss = ce
    
        #print('loss_ce = %.6f' % ce.item())'''
        return loss

    def save_ckp(self, epoch):
        filename = self.ckp_dir + 'epoch%d' % epoch
        state = {'model': self.model.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(state, filename)

    def resume_tr(self):
        ckp = self.model.load_model(self.args)
        self.optimizer.load_state_dict(ckp['optimizer'])
        return int(re.findall('\d+',self.args.test_ckp_dir)[0])