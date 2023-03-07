import torch
import numpy as np
import time
from .load_data import  load_train_data, load_data, MyDataset
from datetime import datetime

def ber(Recover_X,Ideal_X):
    batchsize, S, F = Recover_X.shape[0], Recover_X.shape[1], Recover_X.shape[2]
    Ideal_X = Ideal_X.float()
    Recover_X_id=(torch.sign(torch.stack([-Recover_X.real,-Recover_X.imag],dim = -1)) + 1)/2
    Recover_X_id[:,0,::2,:] = Ideal_X[:,0,::2,:]
    ber = (Ideal_X != Recover_X_id).sum()/(batchsize*(S-0.5)*F*2)
    return ber

    
def bler(H_full, Recover_X, Ideal_X, sigma2):
    batchsize, S, F = H_full.shape[0], H_full.shape[1], H_full.shape[2]
    # data=torch.linalg.norm(H_full,dim=(3,4))**2
    # print('data.shape=',data.shape)
    H_vecnorm = torch.div(1,torch.linalg.norm(H_full,dim=(3,4))**2 + sigma2)
    G = torch.mul(H_vecnorm, torch.linalg.norm(H_full,dim=(3,4))**2)
    eps2 = torch.mul(G, 1-G)
    # print('Recover_X.shape=',Recover_X.shape)

    Recover_Xc = torch.div(torch.mul(-2*torch.sqrt(torch.tensor([2.0]).cuda())*Recover_X,G),eps2)
    llr = torch.clamp(torch.stack([Recover_Xc.real, Recover_Xc.imag],dim=-1),min=-15,max=15)
    Ideal_X = Ideal_X.float()
    ce = -torch.mul(Ideal_X,torch.log(torch.div(pow(2,llr),1+pow(2,llr))))-torch.mul(1-Ideal_X,torch.log(torch.div(1,1+pow(2,llr))))
    ce = torch.mean(ce,dim=(1,2,3))
    ce = torch.reshape(ce,(int(batchsize/25),25))
    ce = torch.mean(ce,1)
    xp = [0,0.415,0.42,0.425,0.43,0.435,0.44,1]
    fp = [0,0,0.0454545454545455,0.21875,0.576923076923077,0.84,1,1]
    bler = np.mean(np.interp(ce.cpu(),xp,fp))
    
    return bler

def add_noise(X,SNR):
    noise=torch.randn(X.shape)
    noise=noise-torch.mean(noise) #  %均值为0，方差接近1
    # print('noise: mean={}, std={}'.format(torch.mean(noise),torch.std(noise)))
    signal_power=torch.sum((X-torch.mean(X))**2)/torch.prod(torch.tensor(X.shape))  # s1/(m*n);%信号功率
    # print('signal power=',signal_power)
    noise_variance=signal_power*(torch.pow(torch.tensor(10.0),-SNR/10.0))
    # print('noise_variance=',torch.sqrt(noise_variance))
    noise=(torch.sqrt(noise_variance)/torch.std(noise))*noise #  %期望的噪声
    return X+noise


def calc_bler(H_full, Recover_X, Ideal_X, sigma2):
    batchsize, S, F = H_full.shape[0], H_full.shape[1], H_full.shape[2]
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

from model import DeepRX, DenseNet

class Tester():
    def __init__(self, args, model):
        self.args = args
        self.layers = args.layers
        # self.model = model
        self.ckp_dir = args.ckp_dir
        # self.net = DeepRX(18,2,1,n_chan=self.args.deepcnn).cuda() #dec15
        self.net = DenseNet(18,2,1,n_chan=self.args.deepcnn).cuda() #DenseNet

    def calc_ber(self, outputs, ideal_x):
        # RecX=torch.stack([outputs.real, outputs.imag],dim=-1)
        # RecX = (torch.sign(outputs-0.5)+1)/2
        RecX = (torch.sign(outputs-self.args.thd)+1)/2
        RecX[:,0,::2,:] = ideal_x[:,0,::2,:]
        # ber = (RecX != ideal_x).sum()/(RecX.shape[0]*(12-0.5)*24*2)
        ber = (RecX != ideal_x).sum()
        # ber =  cnt / ((len(RecX)*(12-0.5)*24*2))
        return ber

    def load_model(self):
        if self.args.phase=='test':
            load_model_dir=self.args.test_ckp_dir
            print('test epoch=',load_model_dir)
        ckp = torch.load(load_model_dir, map_location=lambda storage, loc: storage.cuda(self.args.gpu_idx))
        self.net.load_state_dict(ckp['model'])

    def test(self):
        self.load_model()

        if self.args.data_mode=='SimH':
            doppler=120
            base_dir='/public/share/hcju/MIMO/Ideal_H/'
            # data0 = sio.loadmat(base_dir+'Ideal_H_CDLC_{}Hz.mat'.format(doppler))
            file_path=base_dir+'Ideal_H_CDLC_{}Hz.mat'.format(doppler)
            import h5py
            hc_H=h5py.File(file_path)['Ideal_H']
            hc_H=torch.from_numpy(np.array(hc_H))
            Hid_ts=hc_H.reshape(125000,12,24,4,2)
            print('Haocheng Generated H.shape=',Hid_ts.shape)

            data_mode='EVA'
            noise_list=self.args.ts_dB
            # base_dir='/public/share/hmzhang/MIMO-testset-Sep11-2021/'
            # data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(data_mode,data_mode, sigma,doppler))
            data0 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R0.npz'.format(noise_list,doppler),encoding='latin1')
            # data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            # data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            # data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            Ideal_X = np.array(data0['Ideal_X'], dtype=np.float32)
            Ideal_X = torch.from_numpy(Ideal_X)
            Xid_ts = Ideal_X.to(torch.float32)
            print('Ideal_X.shape=',Xid_ts.shape)
            X0_ts = torch.from_numpy(np.array(data0['Transmit_X'], dtype=np.float32))

            Ideal_X_new=torch.zeros_like(Xid_ts)
            Ideal_X_new[Xid_ts==0]=  torch.sqrt(torch.tensor(2.0, device=torch.device('cpu')))/2.0 
            Ideal_X_new[Xid_ts==1]= -torch.sqrt(torch.tensor(2.0, device=torch.device('cpu')))/2.0

            noise=torch.randn(Hid_ts.shape)
            noise=noise-torch.mean(noise) #  %均值为0，方差接近1

            Yms_ts=add_noise(Hid_ts, SNR=noise_list)*Ideal_X_new.unsqueeze(-2) + noise*0.0001
            print('Received_Y.shape=',Yms_ts.shape)
            Yms_ts[:,0:1,0::2,...]=add_noise(Hid_ts[:,0:1,0::2,...], SNR=noise_list)*X0_ts[:,0:1,0::2,...].unsqueeze(-2)

            Hls_ts=torch.zeros_like(Yms_ts)
            Hls_ts[:,0:1,0::2,...]=Yms_ts[:,0:1,0::2,...]/X0_ts[:,0:1,0::2,...].unsqueeze(-2)
            test_size=Yms_ts.shape[0]
            sigma=torch.zeros(test_size,1,1)
            sigma[...,0] = torch.tensor(np.power(10,-noise_list/10))

            batch_size=1
            H1=Hid_ts[0:batch_size][0,0,0:10,0,0]
            print('idH=',H1)
            Y1=Yms_ts[0:batch_size][0,0,0:10,0,0]
            # Y1=Yms_ts1[0:batch_size][0,0,0::2,0,0]
            print('Y1=',Y1)
            # X0=X0_ts[0:batch_size][0,0,0::2,0]
            X0=X0_ts[0:batch_size][0,0,0:10,0]
            print('X0=',X0)
            # Ideal_X=Xid_ts1[0:batch_size][0,0,0::2,0]
            Ideal_X=Xid_ts[0:batch_size][0,0,0:10,0]
            print('Ideal_X=',Ideal_X)
            Hls=Hls_ts[0:batch_size][0,0,0::2,0,0]
            # Hls=Hls_ts[0:batch_size][0,0,0:10,0,0]
            print('Hls=',Hls)
            # YdX=Yms_ts1[0:batch_size][0,0,0::2,0,0]/X0_ts[0:batch_size].unsqueeze(-2)[0,0,0::2,0,0]
            # print('YdX=',YdX)

        else:
            # Yms_ts, Hls_ts, Hid_ts, Xid_ts,X0_ts, sigma, test_size = load_train_data(noise_list=self.args.ts_dB, doppler=self.args.ts_doppler, 
                # tr_percent=1.0, phase=self.args.phase,data_mode=self.args.data_mode)

            print('Phase: load test data')
            my_batch_size=self.args.tr_batch
            test_dop=self.args.ts_doppler
            ts_mode=self.args.data_mode
            ts_snr=self.args.ts_snr
            test_input, test_snr, test_dop, test_label, test_Hid = load_data(test_dop,snr=[ts_snr], data_per=1.0, phase='test', dataset_name=ts_mode)
            testset = MyDataset(x = test_input, snr=test_snr, dop=test_dop, label = test_label, ideal_H=test_Hid)
            testloader = torch.utils.data.DataLoader(testset, batch_size=my_batch_size, shuffle=False, num_workers=8)

            print('test set len=',len(testset))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_ber_cnt = 0
        test_bler = 0
        # test_loss_cnt=0
        print('Phase: test *******************')
        # self.net=self.net.to(device)
        t_start=time.time()
        # with torch.no_grad():
        self.net.eval()
        for idx, data in enumerate(testloader): #enumerate(trainloader, 0):
            if idx %50==49:
                print('idx={}/{}'.format(idx,len(testset)/my_batch_size))
            inputs, snr, dop, labels, Ideal_H=data
            inputs, sigma_batch, dop, labels, Ideal_H = inputs.to(device), snr.to(device), dop.to(device), labels.to(device), Ideal_H.to(device)
            with torch.no_grad():
                outputs=self.net(inputs)
                # print('outputs.shape=',outputs.shape)
                # print('labels.shape=',labels.shape)
                labels=labels.permute(0,2,3,1)
                outputs=outputs.permute(0,2,3,1)
                # loss_test = self.criterion(outputs, labels)
            # test_loss_cnt+=loss_test.item()
                test_ber_cnt += self.calc_ber(outputs, labels)

                RecX = torch.complex(outputs[...,0], outputs[...,1])
                test_bler += bler(Ideal_H, RecX, labels, sigma_batch)

        test_ber=test_ber_cnt/(len(testset)*(12-0.5)*24*2)
        test_bler=test_bler/(idx+1)
        print('Test results: data mode={}, BER={:.6f}, BLER={:.6f}'.format(self.args.data_mode, test_ber,test_bler))

        cost_time=time.time()-t_start
        print('Test time={:.4f}'.format(cost_time))
        now_time=datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

        # f=open('result/test_ber_record_doppler_{}.txt'.format(self.args.data_mode),"a")
        # f=open('result/test_ber_MMSE_doppler_{}.txt'.format(self.args.data_mode),"a")
        f=open('result/test_ber_{}.txt'.format(self.args.data_mode),"a")
        # msg='\n Doppler={}, noise={}dB, BER ={:.6f}, test_epoch={}, model_suffix={}, now={}'.format(self.args.ts_doppler, self.args.ts_dB, ber_vl.item(), 
        msg='\n Doppler={}, noise={:3d}dB, BER = &{:.6f} , BLER= &{:.6f} , H={}, test_epoch={}, model={}, now={}'.format(self.args.ts_doppler, ts_snr, test_ber, test_bler, self.args.data_mode,
            self.args.test_epoch, self.args.suffix, now_time)
        print(msg)
        f.write(msg)
        print('\n')

