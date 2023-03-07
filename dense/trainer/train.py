from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR,ExponentialLR
import torch.optim as optim
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch.nn.functional as F
import re
from datetime import datetime
from torch import nn
# import torch.nn.functional as F
import os, sys
# from .load_data import load_valid_data, load_train_data
from .load_data import load_train_data, simulate_mimo_training_data, sim_H, load_sim_H, MyDataset, load_data
import random
from glob import glob
from model import DeepRX, DeepRxNew, DenseNet

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.__stdout__
        self.log = open(fileN, "a+")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
#        self.close()
    def flush(self):
        self.log.flush()

class Trainer():
    def __init__(self, args, model):
        self.args = args
        # self.model = model
        self.lr = args.lr
        self.epoch = args.epoch
        self.shuffle = args.shuffle
        self.batch_size = args.tr_batch
        self.ckp_dir = args.ckp_dir
        self.noise = args.noise
        self.criterion=nn.BCELoss()
        # self.net = DeepRX(18,2,1,n_chan=self.args.deepcnn).cuda()  #原来的DeepRX
        # self.net = DeepRxNew(18,2,1,n_chan=self.args.deepcnn).cuda() #改写为module后的DeepRX
        self.net = DenseNet(18,2,1,n_chan=self.args.deepcnn).cuda() #DenseNet
        sys.stdout = Logger('result/'+self.args.info+'/training_logger.txt')

    def _set_optim(self):
        for p in list(self.net.parameters()):
            p.requires_grad = True

        num_para=len(list(self.net.parameters()))
        print('[***train.py***]-'+'--'*20+'-->Num. of Param Block =',num_para)
        i=0
        train_para_num=0.0
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                i=i+1
                # print('---->idx={},name={},size={}'.format(i,name,param.size()))
            if len(param.shape)==4:
                # print('len=4, param.shape=',param.shape)
                outch,inch,ksx,ksy=param.shape
                train_para_num_tmp=torch.prod(torch.as_tensor([outch,inch,ksx,ksy]))
                train_para_num+=train_para_num_tmp
            elif len(param.shape)==3:
                # print('len=4, param.shape=',param.shape)
                inch,ksx,ksy=param.shape
                train_para_num_tmp=torch.prod(torch.as_tensor([inch,ksx,ksy]))
                train_para_num+=train_para_num_tmp
            elif len(param.shape)==2:
                # print('len=2, param.shape=',param.shape)
                outch,inch=param.shape
                train_para_num_tmp=torch.prod(torch.as_tensor([outch,inch]))
                train_para_num+=train_para_num_tmp
            elif len(param.shape)==1:
                # print('len=1, param.shape=',param.shape)
                num_bias=param.shape
                num_bias=torch.prod(torch.as_tensor(num_bias))
                train_para_num+=num_bias
            else:
                print('param.shape=',param.shape)
                raise Exception("check param.shape")
        print('****'*20)
        print('\n******# Trainable param.=',train_para_num)

        # from thop import profile
        # input=torch.randn(1,1,18,12,24).cuda()
        # flops, params_list=profile(self.net, (input))
        # print('DenseNet flops=',flops)

        # para_lst=[]    
        # for name, param in self.model.net.named_parameters():
        #     # print('name={}, param={}'.format(name,param))
        #     if 'dncnn' in name or 'cnn' in name or 'fcn' in name or 'net' in name:
        #         para_lst.append(param)
        if self.args.print_net:
            print(self.net)
        num_params = sum(p.numel() for p in self.net.parameters())
        print("Total parameters: ", num_params)        
        
        # optimizer = optim.Adam(para_lst, lr=1e-4,weight_decay=0.1)

        # optimizer = optim.Adam(para_lst, lr=1e-4, weight_decay=0.)
        print('training phase: lr=',self.lr)
        if self.args.optimizer=='Adam':
            # optimizer = optim.Adam(para_lst, lr=self.lr, weight_decay=self.args.wdk)
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.args.wdk)
        elif self.args.optimizer=='SGD':
            optimizer = optim.SGD(para_lst, lr=self.lr,weight_decay=self.args.wdk)
        # optimizer = optim.Adadelta(para_lst, lr=self.lr, rho=0.9, eps=1e-06, weight_decay=self.args.wdk)
        #optimizer = optim.Adagrad(para_lst, lr=self.lr, lr_decay=0.1, weight_decay=self.args.wdk, initial_accumulator_value=0, eps=1e-10)
        elif self.args.optimizer=='AdamW':
            optimizer = optim.AdamW(para_lst, lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.args.wdk, amsgrad=False)

        # optimizer = optim.ASGD(para_lst, lr=self.lr, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=self.args.wdk)
        # optimizer = optim.LBFGS(para_lst, lr=self.lr, max_iter=20, max_eval=25, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn='strong_wolfe')

        # optimizer = optim.SGD(para_lst, lr=self.lr,weight_decay=0.)
        # scheduler = MultiStepLR(optimizer, milestones=[200,400,600], gamma=0.1)  # learning rates CosineAnnealingLR
        if self.args.lr_scheduler=='StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
        elif self.args.lr_scheduler=='CosineAnnealingLR':
        # scheduler = MultiStepLR(optimizer, milestones=[400,600,800], gamma=0.1)  # learning rates CosineAnnealingLR
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epoch)
        #scheduler = ExponentialLR(optimizer, gamma=0.9)
        return optimizer, scheduler

    def weights_init(self):
        for ll in range(self.args.layers):
            for name, param in self.net.named_parameters():
                if 'dncnn' in name or 'cnn' in name:
                    torch.nn.init.zeros_(param)

    def gradnorm(self):
        total_norm = 0.0
        for param in self.net.parameters():
            total_norm = total_norm + param.grad.data.norm(2).item()**2
        total_norm = total_norm**(1/2)
        
        return total_norm

    def tr(self):
        self.optimizer, self.scheduler = self._set_optim()
        #self.weights_init()
        # sys.stdout = Logger('result/'+self.args.info+'/training_logger.txt')

        # print('Args=',self.args)
        for k in self.args.__dict__:
            print(k + ": " + str(self.args.__dict__[k]))
        print('\n')

        start = 0
        # Resume Training
        if self.args.resume == True:
            print('\nResume training: start from epoch=',self.args.resume_ckp_dir)
            start = self.resume_tr()
            start += 1
            print('Resume train: start epoch=',start)

        dop=self.args.doppler
        my_batch_size=self.batch_size
        valid_snr=int(self.args.ts_snr)

        print('Phase: load valid data')
        vl_input, vl_snr, vl_dop, vl_label, vl_Hid = load_data(dop,snr=[valid_snr], data_per=0.2, phase='valid', dataset_name='new_EVA')
        # testset_new = MyDataset(x = valid_x_new_eva, y = valid_label_new_eva) #
        testset_new = MyDataset(x = vl_input, snr=vl_snr, dop=vl_dop, label = vl_label, ideal_H=vl_Hid)
        testloader_new = torch.utils.data.DataLoader(testset_new, batch_size=my_batch_size, shuffle=False, num_workers=8)
        print('New EVA testset len=',len(testset_new))

        print('\nPhase: load train data')
        train_input, tr_snr, tr_dop, train_label, tr_Hid = load_data(dop,snr=[30,20,10,0,-5,-10], data_per=1.0, phase='train', dataset_name='EVA')
        trainset = MyDataset(x = train_input, snr=tr_snr, dop=tr_dop, label = train_label, ideal_H=tr_Hid)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=my_batch_size, shuffle=True, num_workers=8)
        print('Training set len=',len(trainset))

        print('\nPhase: load test data')
        ts_input, ts_snr, ts_dop, ts_label, ts_Hid = load_data(dop, snr=[valid_snr], data_per=0.01, phase='test', dataset_name='old_EVA')
        testset = MyDataset(x = ts_input,snr=ts_snr, dop=ts_dop, label = ts_label, ideal_H=ts_Hid)
        testloader = torch.utils.data.DataLoader(testset, batch_size=my_batch_size, shuffle=False, num_workers=8)
        print('Old EVA testset len=',len(testset))

        # device='cuda'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_ber_cnt = 0
        test_loss_cnt=0
        print('Phase: valid *******************')
        # self.net=self.net.to(device)
        self.net.eval()
        for idx, data in enumerate(testloader): #enumerate(trainloader, 0):
            inputs,ts_snr,ts_dop,labels, _=data
            inputs, labels = inputs.to(device),labels.to(device)
            with torch.no_grad():
                outputs=self.net(inputs)
                labels=labels.permute(0,2,3,1)
                outputs=outputs.permute(0,2,3,1)
                loss_test = self.criterion(outputs, labels)
            test_loss_cnt+=loss_test.item()
            test_ber_cnt += self.calc_ber(outputs, labels)
        valid_ber=test_ber_cnt/(len(testset)*(12-0.5)*24*2)
        print('Valid old EVA, BER={:.6f}, snr={}, dop={}'.format(valid_ber.item(), -round(10.0*torch.log10(ts_snr.mean()).cpu().numpy()) , ts_dop.mean()))

        print('Start training','****'*20)
        n_iter = 0
        with torch.autograd.set_detect_anomaly(True):
            ber_list=[]
            ber_new_list=[]
            tr_H_loss=[]
            tr_X_loss=[]
            tr_CE_loss=[]
            tr_T_loss=[]
            epoch_list=[]
            val_ber_tmp0=1.
            val_ber_tmp=valid_ber
            
            self.net.train()
            for epoch in range(start, self.epoch):
                print('Epoch: start {} --> current/total={}/{}'.format(start, epoch, self.epoch))
                # if epoch==start:

                if epoch>1 and epoch % int(self.args.shuffle)==0:
                    print('\nPhase: load new train data')
                    del trainloader, data, trainset
                    print('\n','*****'*10+'\nload new training dataset, epoch=',epoch,'\n','*****'*10)
                    dop_list=[5, 30, 60, 90, 120, 150]
                    random.shuffle(dop_list)
                    dop = dop_list[0]
                    print('dop=',dop)
                    train_input, tr_snr, tr_dop, train_label, tr_Hid = load_data(dop,snr=[30,20,10,0,-5,-10], data_per=1.0, phase='train', dataset_name='EVA')
                    trainset = MyDataset(x = train_input, snr=tr_snr, dop=tr_dop, label = train_label, ideal_H=tr_Hid)
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=my_batch_size, shuffle=True, num_workers=8)
                    print('Training set len=',len(trainset))

                t_start = time.time()
                tr_loss_cnt=0
                tr_ber_cnt=0
                for idx, data in enumerate(trainloader):
                    inputs, tr_snr_batch,_,labels,_=data
                    inputs, labels = inputs.to(device),labels.to(device)

                    self.optimizer.zero_grad()
                    outputs=self.net(inputs)
                    loss_tr = self.criterion(outputs, labels)
                    labels=labels.permute(0,2,3,1)
                    outputs=outputs.permute(0,2,3,1)
                    loss_tr.backward()
                    self.optimizer.step()
                    # self.scheduler.step()

                    tr_loss_cnt+=loss_tr.item()
                    tr_ber_cnt += self.calc_ber(outputs, labels)
                    n_iter+=1
                
                    if (idx+1)% self.args.disp==0:
                        print('Train: step={}, BER={:.6f}, CE loss={:.4f}'.format(n_iter, tr_ber_cnt/(len(trainset)*(12-0.5)*24*2), tr_loss_cnt/(idx+1)))
                        now_time=datetime.now().strftime("%H:%M:%S")
                        f=open('result/'+self.args.info+'/training_log.txt',"a")
                        msg='Epoch {} --> {}/{}, i={:4d}/{}, {:.2f}dB/{}Hz, Tr_BER={:.6f}, L_CE={:.6f}, Now={}'.format(start, epoch, self.epoch, idx+1, int(len(trainset)/my_batch_size),-round(10.0*torch.log10(tr_snr_batch.mean()).cpu().numpy()), dop,(tr_ber_cnt/((idx+1)*my_batch_size*(12-0.5)*24*2)).item(), tr_loss_cnt/(idx+1), now_time)
                        print(msg)
                        f.write('\n'+msg)
                train_loss=tr_loss_cnt/len(trainset)
                t_end = time.time()
                print('Epoch {:d}, time/epoch = {:.2f}, epoch_loss = {:,.6f}'.format(epoch,t_end-t_start, train_loss))
                self.scheduler.step()

                test_ber_cnt = 0
                test_loss_cnt=0
                print('Phase: valid Old EVA *******************')
                self.net.eval()
                for idx, data in enumerate(testloader): #enumerate(trainloader, 0):
                    inputs,_,_,labels,_=data
                    inputs, labels = inputs.to(device),labels.to(device)
                    with torch.no_grad():
                        outputs=self.net(inputs)
                        labels=labels.permute(0,2,3,1)
                        outputs=outputs.permute(0,2,3,1)
                        loss_test = self.criterion(outputs, labels)
                    test_loss_cnt+=loss_test.item()
                    test_ber_cnt += self.calc_ber(outputs, labels)
                old_eva_ber=(test_ber_cnt/(len(testset)*(12-0.5)*24*2)).item()
                old_eva_loss=test_loss_cnt/(idx+1)
                print('Epoch {:d}, old EVA, BER={:.6f}, epoch_loss = {:,.6f}'.format(epoch, old_eva_ber, old_eva_loss))
                
                test_ber_cnt = 0
                test_loss_cnt=0
                print('Phase: valid new EVA *******************')
                self.net.eval()
                for idx, data in enumerate(testloader_new): #enumerate(trainloader, 0):
                    inputs,_,_,labels,_=data
                    inputs, labels = inputs.to(device),labels.to(device)
                    with torch.no_grad():
                        outputs=self.net(inputs)
                        labels=labels.permute(0,2,3,1)
                        outputs=outputs.permute(0,2,3,1)
                        loss_test = self.criterion(outputs, labels)
                    test_loss_cnt+=loss_test.item()
                    test_ber_cnt += self.calc_ber(outputs, labels)
                new_eva_ber=(test_ber_cnt/(len(testset_new)*(12-0.5)*24*2)).item()
                new_eva_loss=test_loss_cnt/(idx+1)
                print('Epoch {:d}, New EVA, BER={:.6f}, epoch_loss = {:,.6f}'.format(epoch, new_eva_ber, new_eva_loss))
   
                print('\tOld EVA (BER) \t New EVA (BER) \t Old EVA (Loss) \t New EVA (Loss)')
                print('\t{:.6f}, \t{:.6f}, \t{:.6f}, \t{:.6f}'.format(old_eva_ber, new_eva_ber, old_eva_loss, new_eva_loss))
                ber_list.append(old_eva_ber)
                ber_new_list.append(new_eva_ber)
                tr_CE_loss.append(train_loss)
                epoch_list.append(int(epoch+1))
 
                plt.plot(epoch_list, ber_list,'b',label='Validation BER, old EVA')
                plt.plot(epoch_list, ber_new_list,'r',label='Validation BER, new EVA')
                plt.ylabel('BER')
                plt.xlabel('Epoch')
                plt.legend()
                plt.savefig('result/' + self.args.info +'/valid_BER_vs_epoch_curve_start{}.png'.format(start))
                plt.close('all')

                plt.plot(epoch_list, tr_CE_loss,'cyan',label='Training CE loss')
                plt.ylabel('CE')
                plt.xlabel('Epoch')
                plt.legend()
                plt.savefig('result/' + self.args.info +'/Training_CE_loss_vs_epoch_start{}.png'.format(start))
                plt.close('all')

                plt.plot(epoch_list, np.log(tr_CE_loss),'cyan',label='Training log(CE)')
                plt.ylabel('CE')
                plt.xlabel('Epoch')
                plt.legend()
                plt.savefig('result/' + self.args.info +'/Training_log_CE_loss_vs_epoch_start{}.png'.format(start))
                plt.close('all')

                # grad_norm = self.gradnorm()
                # print('grad_norm=',grad_norm)

                # Save models
                if epoch>1:
                    if val_ber_tmp>old_eva_ber:
                        val_ber_tmp0=val_ber_tmp
                        val_ber_tmp=old_eva_ber
                        best_epoch=epoch
                        print('*****'*20+'Current best epoch=',best_epoch)
                        self.save_ckp(epoch,remove=True,flag=1)
                    elif val_ber_tmp<old_eva_ber:
                        val_ber_tmp0=val_ber_tmp
                        self.save_ckp(epoch,remove=True,flag=0)
                else:
                    self.save_ckp(epoch,remove=False,flag=0)

                # if (epoch+1) % 20  == 0 or epoch == self.epoch-1:
                if (epoch+1) % self.args.save_epoch  == 0 or epoch == self.epoch-1:
                    print('\n'+'--->'*10+'Save CKP'+'{}'.format(epoch))
                    self.save_ckp(epoch)
                    plt.plot(epoch_list, ber_list,'b',label='Validation BER, old EVA')
                    plt.plot(epoch_list, ber_new_list,'r',label='Validation BER, new EVA')
                    plt.ylabel('BER')
                    plt.xlabel('Epoch')
                    plt.legend()
                    plt.savefig('result/' + self.args.info +'/valid_BER_vs_epoch_curve_StartEpoch{}_EndEpoch{}.png'.format(start,epoch+1))
                    plt.close('all')
        return 0

    def ber(self, Recover_X,Ideal_X):
        batchsize, S, F = Recover_X.shape[0], Recover_X.shape[1], Recover_X.shape[2]
        Ideal_X = Ideal_X.float()
        Recover_X_id=(torch.sign(torch.stack([-Recover_X.real,-Recover_X.imag],dim = -1)) + 1)/2
        Recover_X_id[:,0,::2,:] = Ideal_X[:,0,::2,:]
        ber = (Ideal_X != Recover_X_id).sum()/(batchsize*(S-0.5)*F*2)
        return ber


    def calc_ber(self, outputs, ideal_x):
        # RecX=torch.stack([outputs.real, outputs.imag],dim=-1)
        RecX = (torch.sign(outputs-0.5)+1)/2
        RecX[:,0,::2,:] = ideal_x[:,0,::2,:]
        # ber = (RecX != ideal_x).sum()/(RecX.shape[0]*(12-0.5)*24*2)
        ber = (RecX != ideal_x).sum()
        # ber =  cnt / ((len(RecX)*(12-0.5)*24*2))
        return ber

    # def criterion(self):
    #     bce_fun = nn.BCELoss()
    #     return bce_fun

    # def mix_loss(self, H_full0, H_full, Recover_X, Ideal_H, Ideal_X):
    def mix_loss(self,Y, H1, Ideal_H, Recover_X, Ideal_X, sigma2):
        # sigma2 = np.power(10,-self.noise/10)
        X2_loss=0
        if X2_loss:
            # H_vecnorm = torch.div(1,torch.linalg.norm(H_full,dim=(3,4))**2 + sigma2)
            # H2_vecnorm = torch.div(1.0,torch.linalg.norm(Ideal_H,dim=(3,4))**2 + sigma2**2)
            # H2_vecnorm = torch.div(1.0,torch.linalg.norm(Ideal_H,dim=(3,4))**2 + sigma2)
            # H2_vecnorm = torch.div(1.0,torch.linalg.norm(H1,dim=(3,4))**2 + sigma2**2)
            H2_vecnorm = torch.div(1.0,torch.linalg.norm(H1,dim=(3,4))**2 + sigma2)
            # G = torch.mul(H2_vecnorm, torch.linalg.norm(Ideal_H,dim=(3,4))**2)
            G = torch.mul(H2_vecnorm, torch.linalg.norm(H1,dim=(3,4))**2)
            eps2 = torch.mul(G, 1-G)
            eps2[torch.where(eps2 == 0)] = 1.0
            if torch.isnan(G).int().sum()>0:
                print('G has #nan=',torch.isnan(G).int().sum())
                G[torch.isnan(G)]=0.
            Recover_Xc = torch.div(torch.mul(-2*torch.sqrt(torch.tensor([2.0]).cuda())*Recover_X,G),eps2)
            llr = torch.clamp(torch.stack([Recover_Xc.real, Recover_Xc.imag],dim=-1),min=-10,max=10)
            # print('#nan=',torch.isnan(llr).int().sum())
            # print('llr.mean={}, llr.min={}, llr.max={}'.format(llr.mean(),llr.min(),llr.max()))
            if torch.isnan(llr).int().sum()>0:
                print('X2,#Nan=',torch.isnan(llr).int().sum())
                llr[torch.isnan(llr)]=0.
            Ideal_X = Ideal_X.float()
            ce = -torch.mul(Ideal_X, torch.log(torch.div(pow(2.,llr),1.+pow(2.,llr))))-torch.mul(1.-Ideal_X,torch.log(torch.div(1.,1.+pow(2,llr))))
            # # ce = -torch.mul(Ideal_X, torch.log(torch.div(pow(2,llr),1.+pow(2,llr)))+1e-16)-torch.mul(1.-Ideal_X,torch.log(torch.div(1.,1.+pow(2,llr)))+1e-16)
            CE_loss = torch.mean(ce)

        Xl2_Hl2=0
        if Xl2_Hl2:
            H_loss = torch.mean((H1-Ideal_H)**2)
            # RecX1=torch.stack([X1.real, X1.imag],dim=-1)
            # RecX1 = torch.clamp(RecX1,min=-1,max=1).float()

            RecX=torch.stack([Recover_X.real, Recover_X.imag],dim=-1)
            # RecX = torch.clamp(RecX,min=0,max=1).float()
            RecX = torch.clamp(RecX,min=-1,max=2).float()
            # RecX = RecX.float()
            # X_loss = torch.mean((llr-Ideal_X)**2)
            # X1_loss = torch.mean((RecX1-Ideal_X)**2)
            X_loss = torch.mean((RecX-Ideal_X)**2)

        X_MSE_Loss=1
        if X_MSE_Loss:
            X_label=torch.zeros_like(Ideal_X)
            # X_label[Ideal_X==0]= torch.sqrt(torch.tensor(2.0, device=torch.device('cuda:0')))/2.0
            # X_label[Ideal_X==1]= -torch.sqrt(torch.tensor(2.0, device=torch.device('cuda:0')))/2.0

            X_label[Ideal_X==0]= torch.sqrt(torch.tensor(2.0, device=torch.device('cuda:0')))/2.0 + torch.normal(0,1,size=())*self.args.label_noise
            X_label[Ideal_X==1]= -torch.sqrt(torch.tensor(2.0, device=torch.device('cuda:0')))/2.0 + torch.normal(0,1,size=())*self.args.label_noise

            RecX=torch.stack([Recover_X.real, Recover_X.imag],dim=-1)
            RecX = torch.clamp(RecX,min=-1,max=1).float()
            X_MSE=torch.mean((RecX-X_label)**2)

        ber_loss=0
        if ber_loss:
            # ber_val=self.ber(Recover_X, Ideal_X)
            ber_val=self.calc_ber(Recover_X, Ideal_X)
            # H_loss = torch.mean((H1-Ideal_H)**2)

            RecX=torch.stack([Recover_X.real, Recover_X.imag],dim=-1)
            HxY=torch.mean((self.cf_mul(H1,RecX.unsqueeze(-2))-Y)**2)
        bce_loss=0
        if bce_loss:
            # m = nn.Sigmoid()
            # assert RecX.shape==Ideal_X.shape
            # bce = bce_fun(m(-RecX), Ideal_X)
            bce = bce_fun(RecX, Ideal_X)

        #loss=  X_loss + CE_loss*self.args.cew
        # loss=  X_loss*0.01 + CE_loss
        # loss=  X_loss*0.1 + CE_loss
        #loss=  H_loss*0.01 + X_loss*0.01 + CE_loss
        #loss=  CE_loss # from epoch=156
        # loss=  X_MSE*self.args.wXloss + CE_loss
        # loss=  X_MSE*self.args.wXloss +H_loss +HxY *5.0
        # loss=  X_MSE*self.args.wXloss + H_loss + HxY * 10.0 + CE_loss 
        # loss=  X_MSE*self.args.wXloss + H_loss + HxY * 10.0 + CE_loss 
        # loss=  X_MSE*self.args.wXloss + H_loss + HxY * 10.0 + CE_loss  + bce
        # loss=  X_MSE*self.args.wXloss + H_loss*self.args.wH  + HxY * self.args.wHX + CE_loss *self.args.wCE  + bce*self.args.wBCE 
        # loss=  X_MSE*self.args.wXloss + bce*self.args.wBCE 
        RecX=torch.stack([Recover_X.real, Recover_X.imag],dim=-1)
        loss=  self.criterion(RecX, Ideal_X)
        #loss=  X_MSE*self.args.wXloss + H_loss + HxY * 5.0 + CE_loss 
        # loss= CE_loss
        ber_val=self.calc_ber(Recover_X, Ideal_X)
        X_L2=torch.mean(((torch.sign(RecX-0.5)+1)/2- Ideal_X)**2)

        # return ber_val, H_loss, X_MSE, CE_loss, loss, HxY, bce
        return ber_val, X_MSE, X_L2, loss, loss, loss, loss
        # return  CE_loss, loss

    def cf_mul(self, x, y):
    # dim = len(x.size())-1
        real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
        image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
        mul = torch.stack((real, image), dim=-1)
        return mul

    def save_ckp(self, epoch,remove=False,flag=1):
        filename = self.ckp_dir + 'epoch%d' % epoch
        state = {'model': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        if flag==1:
            torch.save(state, filename)
            if remove:
                # idx=epoch-2
                print('Remove the last best performance model CKP!!!\n')
                if epoch>9:
                    for idx in range(epoch-7,epoch-2):
                        filename = self.ckp_dir + 'epoch%d' % idx
                        if os.path.exists(filename):
                            os.remove(filename)
                            print('!!!!!! Remove file path and name:',filename)
                    print('Remove epoch:',idx,'\n')
        if not remove:
            torch.save(state, filename)
            # print('Do not save model')

    def resume_tr(self):
        # ckp = self.model.load_model(self.args)

        if self.args.phase=='train':
            load_model_dir=self.args.resume_ckp_dir
        elif self.args.phase=='test':
            load_model_dir=self.args.test_ckp_dir
            print('test epoch=',load_model_dir)
        ckp = torch.load(load_model_dir, map_location=lambda storage, loc: storage.cuda(self.args.gpu_idx))
        self.net.load_state_dict(ckp['model'])

        self.optimizer.load_state_dict(ckp['optimizer'])
        # return int(re.findall('\d+',self.args.resume_ckp_dir)[0])
        index_list =[i.end() for i in re.finditer('epoch',self.args.resume_ckp_dir)][-1]
        resum_epoch=int(self.args.resume_ckp_dir[index_list:])
        return resum_epoch
