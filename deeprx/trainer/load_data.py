import numpy as np
import torch

import time
def load_train_data(noise_list=30, doppler=90, tr_percent=0.8, phase='test',data_mode='EVA'):
    t_start=time.time()
    if phase=='test':
        print('\nphase=',phase)
        print('\n')
        # base_dir='/public/share/hmzhang/MIMO-testset-Sep11-2021/'
        # data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(data_mode,data_mode, noise_list,doppler))
        # data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(data_mode,data_mode, noise_list,doppler))
        # data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(data_mode,data_mode, noise_list,doppler))
        # data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(data_mode,data_mode, noise_list,doppler))
        if data_mode=='AWGN':
            print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/'
            data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(data_mode,data_mode, noise_list,doppler))
        elif data_mode=='CDL':
            print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/CDL/'
            if noise_list==-8:
                data_mode='CDLA'
            elif noise_list==-3:
                data_mode='CDLE'
            data0 = np.load(base_dir+'{}_{}dB_{}Hz_R0_test.npz'.format(data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}_{}dB_{}Hz_R1_test.npz'.format(data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}_{}dB_{}Hz_R2_test.npz'.format(data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}_{}dB_{}Hz_R3_test.npz'.format(data_mode, noise_list,doppler))
        elif data_mode=='EVA':
            # print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/'
            data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(data_mode,data_mode, noise_list,doppler))

        elif data_mode=='TDL':
            print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/TDL/'
            if noise_list==-8:
                data_mode='TDLA'
            elif noise_list==-3:
                data_mode='TDLE'
            data0 = np.load(base_dir+'{}_{}dB_{}Hz_R0_test.npz'.format(data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}_{}dB_{}Hz_R1_test.npz'.format(data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}_{}dB_{}Hz_R2_test.npz'.format(data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}_{}dB_{}Hz_R3_test.npz'.format(data_mode, noise_list,doppler))
        elif data_mode=='old_EVA':
            print('\nphase=',phase)
            data0 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R0.npz'.format(noise_list,doppler),encoding='latin1')
            data1 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R1.npz'.format(noise_list,doppler))
            data2 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R2.npz'.format(noise_list,doppler))
            data3 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R3.npz'.format(noise_list,doppler))
            
    elif phase=='train':
        data0 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R0.npz'.format(noise_list,doppler),encoding='latin1')
        data1 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R1.npz'.format(noise_list,doppler))
        data2 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R2.npz'.format(noise_list,doppler))
        data3 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R3.npz'.format(noise_list,doppler))
    elif phase=='valid':
        if data_mode=='old_EVA':
            print('\nphase=',phase)
            data0 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R0.npz'.format(noise_list,doppler),encoding='latin1')
            data1 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R1.npz'.format(noise_list,doppler))
            data2 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R2.npz'.format(noise_list,doppler))
            data3 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R3.npz'.format(noise_list,doppler))
        elif data_mode=='new_EVA':
            data_mode='EVA'
            print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-testset-Sep11-2021/'
            data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(data_mode,data_mode, noise_list,doppler))

    if phase=='train':
        if doppler==5 or doppler==30 or doppler==60:
            NumSample=125000-175
        else:
            (NumSample,S,F,_) = data0['Recived_Y'].shape

    else:
        (NumSample,S,F,_) = data0['Recived_Y'].shape

    print('Load {} data: db={:3d}, Doppler={}, NumSample={}, tr_percent={}'.format(phase, noise_list,doppler,NumSample,tr_percent))

    def drop_trs(input_x):
        return input_x #np.delete(input_x, id, 0)

    Received_Y = np.array((drop_trs(data0['Recived_Y']), drop_trs(data1['Recived_Y']), drop_trs(data2['Recived_Y']), drop_trs(data3['Recived_Y'])))
    Hls = np.array((drop_trs(data0['Hls']), drop_trs(data1['Hls']), drop_trs(data2['Hls']), drop_trs(data3['Hls'])))
    Ideal_H = np.array((drop_trs(data0['Ideal_H']), drop_trs(data1['Ideal_H']), drop_trs(data2['Ideal_H']), drop_trs(data3['Ideal_H'])))
    Trans_X = np.array(drop_trs(data0['Transmit_X']), dtype=np.float32)
    Ideal_X = np.array(drop_trs(data0['Ideal_X']), dtype=np.float32)

    # Received_Y=Received_Y
    # Hls=Hls.permute(1, 2, 3,0,4)
    # Ideal_H=Ideal_H.permute(1, 2, 3,0,4)

    # print('Received_Y.shape=',Received_Y.shape)
    # print('Hls.shape=',Hls.shape)
    # print('Ideal_H.shape=',Ideal_H.shape)
    # print('Ideal_X.shape=',Ideal_X.shape)
    # print('Trans_X.shape=',Trans_X.shape)
    if phase=='train':
        if doppler==5 or doppler==30 or doppler==60:
            Received_Y=Received_Y[:,0:124950,...]
            Hls=Hls[:,0:124950,...]
            Ideal_H=Ideal_H[:,0:124950,...]
            Trans_X=Trans_X[0:124950,...]
            Ideal_X=Ideal_X[0:124950,...]


    # print('After: Received_Y.shape=',Received_Y.shape)
    # print('After: Hls.shape=',Hls.shape)
    # print('After: Ideal_H.shape=',Ideal_H.shape)
    # print('After: Ideal_X.shape=',Ideal_X.shape)
    # print('After: Trans_X.shape=',Trans_X.shape)

    remove_data=0
    if remove_data:
        id = np.arange(0,NumSample,1)
        # print('id=',id)
        id = np.argwhere(id % 125 < 50)
        Received_Y = torch.from_numpy(np.delete(Received_Y,id,1)).permute(1, 2, 3,0,4)
        Hls = torch.from_numpy(np.delete(Hls,id,1)).permute(1, 2, 3,0,4)
        Ideal_H = torch.from_numpy(np.delete(Ideal_H,id,1)).permute(1, 2, 3,0,4)
        Ideal_X = torch.from_numpy(np.delete(Ideal_X,id,0))
        Trans_X = torch.from_numpy(np.delete(Trans_X,id,0))
    
    Received_Y = torch.from_numpy(Received_Y) #.permute(1, 2, 3,0,4)
    # Hls = torch.from_numpy(Hls).permute(1, 2, 3,0,4)
    # Ideal_H = torch.from_numpy(Ideal_H).permute(1, 2, 3,0,4)
    # Ideal_X = torch.from_numpy(Ideal_X)
    # Trans_X = torch.from_numpy(Trans_X)

    Received_Y = Received_Y.permute(0, 1, 4, 2, 3)
    Received_Y = torch.cat([Received_Y[0,:,:,:,:], Received_Y[1,:,:,:,:], Received_Y[2,:,:,:,:], Received_Y[3,:,:,:,:]], axis=1)
    Received_Y = Received_Y.type(torch.FloatTensor)

    # Ideal_H = torch.from_numpy(Ideal_H)
    # Ideal_H = Ideal_H.permute(0, 1, 4, 2, 3)
    # Ideal_H = torch.cat([Ideal_H[0,:,:,:,:], Ideal_H[1,:,:,:,:], Ideal_H[2,:,:,:,:], Ideal_H[3,:,:,:,:]], axis=1)
    # Ideal_H = Ideal_H.type(torch.FloatTensor)
    Ideal_H = torch.from_numpy(Ideal_H).permute(1, 2, 3,0,4)

    Hls = torch.from_numpy(Hls)
    Hls = Hls.permute(0, 1, 4, 2, 3)
    Hls = torch.cat([Hls[0,:,:,:,:], Hls[1,:,:,:,:], Hls[2,:,:,:,:], Hls[3,:,:,:,:]], axis=1)
    Hls = Hls.type(torch.FloatTensor)

    Ideal_X = torch.from_numpy(Ideal_X)
    Ideal_X = Ideal_X.permute(0, 3, 1, 2)

    Trans_X = torch.from_numpy(Trans_X)
    Trans_X = Trans_X.permute(0, 3, 1, 2)

    # Received_Y = Received_Y.to(torch.float32)
    # Hls = Hls.to(torch.float32)
    # # Ideal_H = Ideal_H.to(torch.float32)
    # Ideal_X = Ideal_X.to(torch.float32)
    # Trans_X = Trans_X.to(torch.float32)

    idx_tr=int(Received_Y.shape[0]*tr_percent)
    # print('Training per=',idx_tr)
    # idx_per_db=125000*tr_percent

    # print('Received_Y.shape=',Received_Y.shape)
    # print('Hls.shape=',Hls.shape)
    # print('Ideal_X.shape=',Ideal_X.shape)
    # print('Trans_X.shape=',Trans_X.shape)

    Yms_tr=Received_Y[:idx_tr,...]
    # Yms_ts=Received_Y[idx_tr:,...]

    Hls_tr=Hls[:idx_tr,...]
    # Hls_ts=Hls[idx_tr:,...]

    Hid_tr=Ideal_H[:idx_tr,...]
    # Hid_ts=Ideal_H[idx_tr:,...]

    Xid_tr=Ideal_X[:idx_tr,...]
    # Xid_ts=Ideal_X[idx_tr:,...]

    X0_tr=Trans_X[:idx_tr,...]
    # X0_ts=Trans_X[idx_tr:,...]

    train_size = Yms_tr.shape[0]

    sigma=torch.zeros(idx_tr,1,1)

    sigma[...,0] = torch.tensor(np.power(10,-noise_list/10))
    end_time=time.time()-t_start
    print('Load data time cost={:.4f}, train dataset len={}'.format(end_time, len(Xid_tr)))
    # print('\n')
    return Yms_tr, Hls_tr, Hid_tr, Xid_tr,X0_tr, sigma, train_size

from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, x, label, dop, snr):               
        super(MyDataset,self).__init__()
        self.x = x
        self.snr = snr
        self.dop = dop
        self.label = label
        # self.ideal_H = ideal_H

    def __getitem__(self, index):
        return self.x[index],  self.label[index],  self.dop[index], self.snr[index]
    
    def __len__(self):
        return len(self.x)


def load_data(dop, snr=[0], data_per=1.0, phase='train', dataset_name='EVA'):
    if phase=='train':
        Yms_tr_30,  Hls_tr_30,  Hid_tr_30, Xid_tr_30, X0_tr_30, sigma_30, _ = load_train_data(noise_list=30, doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Yms_tr_20,  Hls_tr_20,  Hid_tr_20, Xid_tr_20, X0_tr_20, sigma_20, _ = load_train_data(noise_list=20, doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Yms_tr_10,  Hls_tr_10,  Hid_tr_10, Xid_tr_10, X0_tr_10, sigma_10, _ = load_train_data(noise_list=10, doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Yms_tr_0,  Hls_tr_0,  Hid_tr_0, Xid_tr_0, X0_tr_0, sigma_0, _ = load_train_data(noise_list=0, doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Yms_tr__5,  Hls_tr__5,  Hid_tr__5, Xid_tr__5, X0_tr__5, sigma__5, _ = load_train_data(noise_list=-5, doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Yms_tr__10,  Hls_tr__10,  Hid_tr__10, Xid_tr__10, X0_tr__10, sigma__10, _ = load_train_data(noise_list=-10, doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        
        Received_Y = torch.cat([Yms_tr_30, Yms_tr_20, Yms_tr_10, Yms_tr_0, Yms_tr__5, Yms_tr__10], axis=0)
        Hls = torch.cat([Hls_tr_30, Hls_tr_20, Hls_tr_10, Hls_tr_0, Hls_tr__5, Hls_tr__10], axis=0)
        Transmit_X = torch.cat([X0_tr_30, X0_tr_20, X0_tr_10, X0_tr_0, X0_tr__5, X0_tr__10], axis=0)
        Ideal_X = torch.cat([Xid_tr_30, Xid_tr_20, Xid_tr_10, Xid_tr_0, Xid_tr__5, Xid_tr__10], axis=0)
        sig_list = torch.cat([sigma_30, sigma_20, sigma_10, sigma_0, sigma__5, sigma__10], axis=0)
        dop_list=torch.zeros_like(sig_list)
        dop_list[:,0,0]=dop
        Ideal_H=torch.cat([Hid_tr_30, Hid_tr_20, Hid_tr_10, Hid_tr_0, Hid_tr__5, Hid_tr__10], axis=0)
        
    elif phase=='valid':
        Yms_tr_30,  Hls_tr_30,  Hid_vl, Xid_tr_30, X0_tr_30, sigma_vl, _ = load_train_data(noise_list=snr[0], doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Received_Y=Yms_tr_30
        Hls=Hls_tr_30
        Transmit_X=X0_tr_30
        Ideal_X=Xid_tr_30
        sig_list=sigma_vl
        dop_list=torch.zeros_like(sig_list)
        dop_list[:,0,0]=dop
        Ideal_H=Hid_vl

    elif phase=='test':
        Yms_ts,  Hls_ts,  Hid_ts, Xid_ts, X0_ts, sigma_ts, _ = load_train_data(noise_list=snr[0], doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Received_Y=Yms_ts
        Transmit_X=X0_ts
        Hls=Hls_ts
        Ideal_X=Xid_ts
        Ideal_H=Hid_ts
        sig_list=sigma_ts
        dop_list=torch.zeros_like(sig_list)
        dop_list[:,0,0]=dop
        print('Received_Y.shape=',Received_Y.shape)
        print('Transmit_X.shape=',Transmit_X.shape)
        print('Hls.shape=',Hls.shape)
        print('Ideal_X.shape=',Ideal_X.shape)
        print('Ideal_H.shape=',Ideal_H.shape)
        print('sig_list.shape=',sig_list.shape)
        print('dop_list.shape=',dop_list.shape)
    input_x = torch.cat([Received_Y, Transmit_X, Hls], axis=1) # channel = (8+2) + 8 = 10 +8 = 18 (first 10 mainly used)
    # testset = MyDataset(x = input_x, y = Ideal_X)
    # dataset_loader = torch.utils.data.DataLoader(testset, batch_size=my_batch_size, shuffle=shuffle, num_workers=num_workers)
    # return dataset_loader, len(testset)
    return input_x, sig_list, dop_list, Ideal_X, Ideal_H



def shift(A, f, tau):
    '''
    -1000<=f<=1000
     -72<=tau<=72
    '''
    NumSample,S,F,Nr,_ = np.shape(A)
    data = A[...,0] + 1j*A[...,1]
    consts = (2048+144)/(2048*15e3);
    shift_s = np.exp(1j*2*np.pi*f*consts*np.arange(S))
    shift_f = np.exp(-1j*2*np.pi*tau*np.arange(F)/2048)
    data_1 = np.swapaxes(np.swapaxes(data, 1, -1) * shift_s, -1, 1)
    data_1 = np.swapaxes(np.swapaxes(data_1, 2, -1) * shift_f, -1, 2)
    data_1 = np.stack((data_1.real,data_1.imag), axis=-1) #NumSample*S*F*Nr*2 real
    return data_1


def data_with_phase_shift_new(doppler=90, noise_list=30, f=100, tau=72, tr_percent=0.8, phase='test',data_mode='EVA'):
    t_start=time.time()
    if phase=='test':
        print('\nphase=',phase)
        print('\n')
        if data_mode=='AWGN':
            print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/'
            data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(data_mode,data_mode, noise_list,doppler))
        elif data_mode=='CDL':
            print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/CDL/'
            if noise_list==-8:
                data_mode='CDLA'
            elif noise_list==-3:
                data_mode='CDLE'
            data0 = np.load(base_dir+'{}_{}dB_{}Hz_R0_test.npz'.format(data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}_{}dB_{}Hz_R1_test.npz'.format(data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}_{}dB_{}Hz_R2_test.npz'.format(data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}_{}dB_{}Hz_R3_test.npz'.format(data_mode, noise_list,doppler))
        elif data_mode=='EVA':
            # print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/'
            data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(data_mode,data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(data_mode,data_mode, noise_list,doppler))

        elif data_mode=='TDL':
            print('\nphase=',phase)
            # print('\n')
            base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/TDL/'
            if noise_list==-8:
                data_mode='TDLA'
            elif noise_list==-3:
                data_mode='TDLE'
            data0 = np.load(base_dir+'{}_{}dB_{}Hz_R0_test.npz'.format(data_mode, noise_list,doppler))
            data1 = np.load(base_dir+'{}_{}dB_{}Hz_R1_test.npz'.format(data_mode, noise_list,doppler))
            data2 = np.load(base_dir+'{}_{}dB_{}Hz_R2_test.npz'.format(data_mode, noise_list,doppler))
            data3 = np.load(base_dir+'{}_{}dB_{}Hz_R3_test.npz'.format(data_mode, noise_list,doppler))
        elif data_mode=='old_EVA':
            print('\nphase=',phase)
            data0 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R0.npz'.format(noise_list,doppler),encoding='latin1')
            data1 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R1.npz'.format(noise_list,doppler))
            data2 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R2.npz'.format(noise_list,doppler))
            data3 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R3.npz'.format(noise_list,doppler))
            
    else:
        print('load data, phase error')

    # if phase=='train':
    #     if doppler==5 or doppler==30 or doppler==60:
    #         NumSample=125000-175
    #     else:
    #         (NumSample,S,F,_) = data0['Recived_Y'].shape
    # else:
    #     (NumSample,S,F,_) = data0['Recived_Y'].shape

    Received_Y = np.array((data0['Recived_Y'], data1['Recived_Y'], data2['Recived_Y'], data3['Recived_Y']))
    Ideal_H = np.array((data0['Ideal_H'], data1['Ideal_H'], data2['Ideal_H'], data3['Ideal_H']))
    Ideal_X = np.array(data0['Ideal_X'], dtype=np.float32)
    Trans_X = np.array(data0['Transmit_X'], dtype=np.float32)
    Hls = np.array((data0['Hls'], data1['Hls'], data2['Hls'], data3['Hls']))

    _, NumSample, S, F, _ = Received_Y.shape

    print('Load {} data: db={:3d}, Doppler={}, NumSample={}, tr_percent={}'.format(phase, noise_list, doppler,NumSample,tr_percent))
    print('Init: Received_Y.shape=',Received_Y.shape)

    if data_mode=='old_EVA':
        if doppler == 5:
            if noise_list == 10:
                id = np.arange(62125,62225)
                Received_Y = np.delete(Received_Y,id,1)
                Hls = np.delete(Hls,id,1)
                # Ideal_H = np.delete(Ideal_H,id,1)
                Ideal_X = np.delete(Ideal_X,id,0)
                Trans_X = np.delete(Trans_X,id,0)
            if noise_list == 20:
                id = np.arange(57250,57350)
                Received_Y = np.delete(Received_Y,id,1)
                Hls = np.delete(Hls,id,1)
                # Ideal_H = np.delete(Ideal_H,id,1)
                Ideal_X = np.delete(Ideal_X,id,0)
                Trans_X = np.delete(Trans_X,id,0)
        if doppler == 30:
            if noise_list == 10:
                id = np.arange(58750,58825)
                Received_Y = np.delete(Received_Y,id,1)
                Hls = np.delete(Hls,id,1)
                # Ideal_H = np.delete(Ideal_H,id,1)
                Ideal_X = np.delete(Ideal_X,id,0)
                Trans_X = np.delete(Trans_X,id,0)
            if noise_list == 20:
                id = np.arange(58000,58100)
                Received_Y = np.delete(Received_Y,id,1)
                Hls = np.delete(Hls,id,1)
                # Ideal_H = np.delete(Ideal_H,id,1)
                Ideal_X = np.delete(Ideal_X,id,0)
                Trans_X = np.delete(Trans_X,id,0)
        if doppler == 60:
            if noise_list == 10:
                id = np.arange(56875,56975)
                Received_Y = np.delete(Received_Y,id,1)
                Hls = np.delete(Hls,id,1)
                # Ideal_H = np.delete(Ideal_H,id,1)
                Ideal_X = np.delete(Ideal_X,id,0)
                Trans_X = np.delete(Trans_X,id,0)
            if noise_list == 20:
                id = np.arange(62500,62575)
                Received_Y = np.delete(Received_Y,id,1)
                Hls = np.delete(Hls,id,1)
                # Ideal_H = np.delete(Ideal_H,id,1)
                Ideal_X = np.delete(Ideal_X,id,0)
                Trans_X = np.delete(Trans_X,id,0)
            if noise_list == 30:
                id = np.arange(57000,57100)
                Received_Y = np.delete(Received_Y,id,1)
                Hls = np.delete(Hls,id,1)
                # Ideal_H = np.delete(Ideal_H,id,1)
                Ideal_X = np.delete(Ideal_X,id,0)
                Trans_X = np.delete(Trans_X,id,0)

        _, NumSample, S, F, _ = Received_Y.shape
        id = np.arange(0,NumSample,1)
        id = np.argwhere(id % 125 < 50)
        Received_Y = np.delete(Received_Y,id,1)
        Hls = np.delete(Hls,id,1)
        # Ideal_H = np.delete(Ideal_H,id,1)
        Ideal_X = np.delete(Ideal_X,id,0)
        Trans_X = np.delete(Trans_X,id,0)

    print('input: Received_Y.shape=',Received_Y.shape)
    # Received_Y = Received_Y.transpose((1, 2, 3, 0, 4)) # 25000, 12, 24, 4, 2
    Received_Y = torch.from_numpy(Received_Y)
    # Received_Y = Received_Y.permute(3, 0, 4, 1, 2) # 25000, 12, 24, 4, 2
    Received_Y = Received_Y.permute(0,1,4,2,3) # 25000, 12, 24, 4, 2
    Received_Y = torch.cat([Received_Y[0,:,:,:,:], Received_Y[1,:,:,:,:], Received_Y[2,:,:,:,:], Received_Y[3,:,:,:,:]], axis=1)
    # Received_Y = Received_Y.type(torch.FloatTensor)

    Hls = torch.from_numpy(Hls)
    # Hls = Hls.permute(3, 0, 4, 1, 2)
    Hls = Hls.permute(0,1,4,2,3)
    Hls = torch.cat([Hls[0,:,:,:,:], Hls[1,:,:,:,:], Hls[2,:,:,:,:], Hls[3,:,:,:,:]], axis=1)
    # Hls = Hls.type(torch.FloatTensor)

    Trans_X = torch.from_numpy(Trans_X) # 25000, 12, 24, 2
    print('input X0.shape=',Trans_X.shape)

    Ideal_X = torch.from_numpy(Ideal_X)
    Ideal_X = Ideal_X.permute(0, 3, 1, 2)

    idx_tr=int(Received_Y.shape[0]*tr_percent)
    Yms_tr=Received_Y[:idx_tr,...]
    Hls_tr=Hls[:idx_tr,...]
    # Hid_tr=Ideal_H[:idx_tr,...]
    Xid_tr=Ideal_X[:idx_tr,...]
    Trans_X = Trans_X.permute(0, 3, 1, 2)
    X0_tr=Trans_X[:idx_tr,...]

    train_size = Yms_tr.shape[0]
    sigma=torch.zeros(idx_tr,1,1)
    sigma[...,0] = torch.tensor(np.power(10,-noise_list/10))
    end_time=time.time()-t_start
    print('Load data time cost={:.4f}, train dataset len={}'.format(end_time, len(Xid_tr)))
    print('\n')

    print('Load Y.shape=',Yms_tr.shape)
    print('Load Hls.shape=',Hls_tr.shape)
    print('Load Xideal.shape=',Xid_tr.shape)
    print('Load X0.shape=',X0_tr.shape)
    return Yms_tr, Hls_tr, Xid_tr, X0_tr, sigma


def load_phase_shift_data(dop, snr=[0], f=100, tau=72, data_per=1.0, phase='train', dataset_name='EVA'):
    if phase=='test':
        # Yms_ts,  Hls_ts,  Hid_ts, Xid_ts, X0_ts, sigma_ts, _ = load_train_data(noise_list=snr[0], doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        # Yms_ts,  Hls_ts,  Hid_ts, Xid_ts, X0_ts, sigma_ts, _ = data_with_phase_shift(noise_list=snr[0],phase_param=phase_param, doppler=dop, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        # May 21,2022, test phase shift data, 新的相位平移添加方法
        Yms_ts,  Hls_ts, Xid_ts, X0_ts, sigma_ts = data_with_phase_shift_new(doppler=dop, noise_list=snr[0], f=f, tau=tau, tr_percent=data_per, phase=phase,data_mode=dataset_name)
        Received_Y=Yms_ts
        Transmit_X=X0_ts
        Hls=Hls_ts
        Ideal_X=Xid_ts
        # Ideal_H=Hid_ts
        sig_list=sigma_ts
        dop_list=torch.zeros_like(sig_list)
        dop_list[:,0,0]=dop
        print('Received_Y.shape=',Received_Y.shape)
        print('Transmit_X.shape=',Transmit_X.shape)
        print('Hls.shape=',Hls.shape)
        print('Ideal_X.shape=',Ideal_X.shape)
        print('sig_list.shape=',sig_list.shape)
        print('dop_list.shape=',dop_list.shape)
    input_x = torch.cat([Received_Y, Transmit_X, Hls], axis=1) # channel = (8+2) + 8 = 10 +8 = 18 (first 10 mainly used)
    # testset = MyDataset(x = input_x, y = Ideal_X)
    # dataset_loader = torch.utils.data.DataLoader(testset, batch_size=my_batch_size, shuffle=shuffle, num_workers=num_workers)
    # return dataset_loader, len(testset)
    return input_x, Ideal_X, dop_list, sig_list





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

def sim_H(doppler):
    base_dir='/public/share/hcju/MIMO/Ideal_H/'
    # data0 = sio.loadmat(base_dir+'Ideal_H_CDLC_{}Hz.mat'.format(doppler))
    file_path=base_dir+'Ideal_H_CDLC_{}Hz.mat'.format(doppler)
    hc_H=h5py.File(file_path)['Ideal_H']
    hc_H=torch.from_numpy(np.array(hc_H))
    # print('hc_H.shape=',hc_H.shape)
    # Hid_ts=hc_H.reshape(125000,12,24,4,2) #.permute(1,2)
    Hid_ts=hc_H.permute(-1, -2, -3, -4, -5) #.permute(1,2)
    # print('Hid_ts.shape=',Hid_ts.shape)
    return Hid_ts


import scipy.io as sio 


def load_sim_H(file_name):
    # for file_name in file_list:
    # print('Load SimH, File name=',file_name)
    # if 'CDLC' in file_name:
    #     hc_H=h5py.File(file_name)['Ideal_H']
    #     hc_H=torch.from_numpy(np.array(hc_H))
    #     Hid_ts=hc_H.permute(-1, -2, -3, -4, -5) #.permute(1,2)
    # else:
    hc_H = sio.loadmat(file_name)['Ideal_H']
    hc_H=torch.from_numpy(np.array(hc_H))
    Hid_ts=hc_H
        # print('H.shape=',Hid_ts.shape)
    return Hid_ts


import h5py
import random

def cf_mul(x, y):
    dim = len(x.size())-1
    real = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    image = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    mul = torch.stack((real, image), dim=dim)
    return mul

def cf_div(x, y):
    dim = len(x.size())-1
    # y_abs = y[..., 0]**2 + y[..., 1]**2
    y_abs = y[..., 0]**2 + y[..., 1]**2+1e-8
    real = (x[...,0] * y[...,0] + x[...,1] * y[...,1]) / y_abs
    image = (x[...,1] * y[...,0] - x[...,0] * y[...,1]) /y_abs
    div = torch.stack((real, image), dim=dim)
    return div

def simulate_mimo_training_data(Ideal_H, doppler, noise_list, snr, phase='train'):
    # doppler=120
    # print('Haocheng Generated H.shape=',Hid_ts.shape)
    # print('Load simulation data: org_doppler={}, org_snr={}'.format(doppler, noise_list))

    # data_mode='EVA'
    channel_mode_list=['EVA_tr','EVA','TDL','CDL','AWGN']
    random.shuffle(channel_mode_list)
    anntena_list=[0,1,2,3]
    random.shuffle(anntena_list)
    dop_list=[5, 30, 60, 90, 120, 150]
    random.shuffle(dop_list)
    dB_list = [30, 20, 10, 0, -5, -10]
    random.shuffle(dB_list)

    # noise_list=self.args.ts_dB
    if channel_mode_list[0] == 'EVA_tr':
        data0 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R0.npz'.format(dB_list[0], dop_list[0]),encoding='latin1')
        data1 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R1.npz'.format(dB_list[0], dop_list[0]),encoding='latin1')
        data2 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R2.npz'.format(dB_list[0], dop_list[0]),encoding='latin1')
        data3 = np.load('/public/share/yzhao/data/EVA_{}dB_{}Hz_R3.npz'.format(dB_list[0], dop_list[0]),encoding='latin1')
    elif channel_mode_list[0] == 'EVA':
        noise_list=[-8,-4]
        random.shuffle(noise_list)
        noise_list=noise_list[0]
        base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/'
        data0 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R0_test.npz'.format(channel_mode_list[0],channel_mode_list[0], noise_list, dop_list[0]))
        data1 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R1_test.npz'.format(channel_mode_list[0],channel_mode_list[0], noise_list, dop_list[0]))
        data2 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R2_test.npz'.format(channel_mode_list[0],channel_mode_list[0], noise_list, dop_list[0]))
        data3 = np.load(base_dir+'{}/{}_{}dB_{}Hz_R3_test.npz'.format(channel_mode_list[0],channel_mode_list[0], noise_list, dop_list[0]))
    elif channel_mode_list[0] == 'CDL':
        noise_list=[-8,-3]
        random.shuffle(noise_list)
        noise_list=noise_list[0]
        if noise_list==-8:
            data_mode='CDLA'
        elif noise_list==-3:
            data_mode='CDLE'
        dop_list=[15,45,75,105,135,165]
        random.shuffle(dop_list)
        base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/CDL/'
        data0 = np.load(base_dir+'{}_{}dB_{}Hz_R0_test.npz'.format(data_mode, noise_list, dop_list[0]))
        data1 = np.load(base_dir+'{}_{}dB_{}Hz_R1_test.npz'.format(data_mode, noise_list, dop_list[0]))
        data2 = np.load(base_dir+'{}_{}dB_{}Hz_R2_test.npz'.format(data_mode, noise_list, dop_list[0]))
        data3 = np.load(base_dir+'{}_{}dB_{}Hz_R3_test.npz'.format(data_mode, noise_list, dop_list[0]))
    elif channel_mode_list[0] == 'TDL':
        noise_list=[-8,-3]
        random.shuffle(noise_list)
        noise_list=noise_list[0]
        if noise_list==-8:
            data_mode='TDLA'
        elif noise_list==-3:
            data_mode='TDLE'
        dop_list=[15,45,75,105,135,165]
        random.shuffle(dop_list)
        base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/TDL/'
        data0 = np.load(base_dir+'{}_{}dB_{}Hz_R0_test.npz'.format(data_mode, noise_list, dop_list[0]))
        data1 = np.load(base_dir+'{}_{}dB_{}Hz_R1_test.npz'.format(data_mode, noise_list, dop_list[0]))
        data2 = np.load(base_dir+'{}_{}dB_{}Hz_R2_test.npz'.format(data_mode, noise_list, dop_list[0]))
        data3 = np.load(base_dir+'{}_{}dB_{}Hz_R3_test.npz'.format(data_mode, noise_list, dop_list[0]))
    elif channel_mode_list[0] == 'AWGN':
        data_mode=channel_mode_list[0]
        base_dir='/public/share/hmzhang/MIMO-1008-testsets/1008/'
        data0 = np.load(base_dir+'{}/{}_{}dB_0Hz_R0_test.npz'.format(data_mode, data_mode, dB_list[0]))
        data1 = np.load(base_dir+'{}/{}_{}dB_0Hz_R1_test.npz'.format(data_mode, data_mode, dB_list[0]))
        data2 = np.load(base_dir+'{}/{}_{}dB_0Hz_R2_test.npz'.format(data_mode, data_mode, dB_list[0]))
        data3 = np.load(base_dir+'{}/{}_{}dB_0Hz_R3_test.npz'.format(data_mode, data_mode, dB_list[0]))

    # print('channel={}, SNR={}, Dop={}, Ideal_H.shape={}'.format(channel_mode_list[0], dB_list[0], dop_list[0], Ideal_H.shape[0]))

    # data0 = np.load('/public/share/yzhao/data/{}_{}dB_{}Hz_R0.npz'.format(noise_list,doppler),encoding='latin1')
    Ideal_X = torch.from_numpy(np.array(data0['Ideal_X'], dtype=np.float32)).to(torch.float32)
    # print('Ideal_X.shape=',Xid_ts.shape)
    # random.shuffle(Ideal_X)
    # random.shuffle(Ideal_H)
    Trans_X = torch.from_numpy(np.array(data0['Transmit_X'], dtype=np.float32)).to(torch.float32)

    Ideal_H_old = np.array((data0['Ideal_H'], data1['Ideal_H'], data2['Ideal_H'], data3['Ideal_H']))
    Ideal_H_old = torch.from_numpy(Ideal_H_old).permute(1, 2, 3,0,4)
    # Ideal_H = torch.from_numpy(Ideal_H)

    if Ideal_H_old.shape[0]>Ideal_H.shape[0]:
        data_len=Ideal_H.shape[0]
        max_H_idx=torch.randint(0,data_len-2000,[1])
        Ideal_H1=Ideal_H_old[max_H_idx:(max_H_idx+2000),...]
        max_H_idx=torch.randint(0,data_len-2000,[1])
        Ideal_H2=Ideal_H[max_H_idx:(max_H_idx+2000),...]
        lam=torch.randn(1)[0]
        Ideal_H=lam*Ideal_H1+(1.0-lam)*Ideal_H2

    # if phase=='train':
    #     # print('phase=',phase)
    #     if Ideal_H.shape[0]>100000 and Ideal_H.shape[0]<125000:

    #         max_H_idx=torch.randint(0,125000-3000)
    #         Ideal_H=Ideal_H[max_H_idx:(max_H_idx+2000),...]

    #         if doppler==5 or doppler==30 or doppler==60:
    #             Ideal_X=Ideal_X[0:124950,...]
    #             Trans_X=Trans_X[0:124950,...]
    #             # Received_Y=Received_Y[:,0:124950,...]
    #             # Hls=Hls[:,0:124950,...]
    #             Ideal_H=Ideal_H[0:124950,...]
    #     else:
    #         data_len=Ideal_H.shape[0]
        max_H_idx=torch.randint(0,data_len-2000,[1])
        Ideal_X=Ideal_X[max_H_idx:(max_H_idx+2000),...]
        max_H_idx=torch.randint(0,data_len-2000,[1])
        Trans_X=Trans_X[max_H_idx:(max_H_idx+2000),...]
            # Ideal_X=Ideal_X[0:data_len,...]

    Ideal_X_new=torch.zeros_like(Ideal_X)
    Ideal_X_new[Ideal_X==0]=  torch.sqrt(torch.tensor(2.0, device=torch.device('cpu')))/2.0 
    Ideal_X_new[Ideal_X==1]= -torch.sqrt(torch.tensor(2.0, device=torch.device('cpu')))/2.0

    noisy_X=1
    if noisy_X:
        # X_noisy=add_noise(Ideal_X_new, SNR=30)
        noise=torch.randn(Ideal_X_new.shape)
        noise=noise-torch.mean(noise) #  %均值为0，方差接近1
        X_noisy=Ideal_X_new + noise *0.01
        # Received_Y=cf_mul(add_noise(Ideal_H, SNR=snr), X_noisy.unsqueeze(-2))  #直接点乘不正确
        Received_Y=add_noise(cf_mul(Ideal_H, X_noisy.unsqueeze(-2)), SNR=snr)  #直接点乘不正确

        noise2=torch.randn(Trans_X[:,0:1,0::2,...].shape)
        noise2=noise2-torch.mean(noise2) #  %均值为0，方差接近1
        Trans_X_data=Trans_X[:,0:1,0::2,...] + noise2*0.01
        # Received_Y[:,0:1,0::2,...]=cf_mul(add_noise(Ideal_H[:,0:1,0::2,...], SNR=snr), Trans_X_data.unsqueeze(-2))
        Received_Y[:,0:1,0::2,...]=add_noise(cf_mul(Ideal_H[:,0:1,0::2,...], Trans_X_data.unsqueeze(-2)), SNR=snr)
    else:
    #Received_Y=add_noise(Ideal_H, SNR=snr)*Ideal_X_new.unsqueeze(-2)+noise *0.0001
    # Received_Y=add_noise(Ideal_H, SNR=snr)*Ideal_X_new.unsqueeze(-2)  #直接点乘不正确
    # Received_Y=cf_mul(add_noise(Ideal_H, SNR=snr), Ideal_X_new.unsqueeze(-2))  #直接点乘不正确
        Noisy_H=add_noise(Ideal_H, SNR=snr)
        Received_Y=cf_mul(Noisy_H, Ideal_X_new.unsqueeze(-2))  #直接点乘不正确
    # print('Received_Y.shape=',Yms_ts.shape)
    # Received_Y[:,0:1,0::2,...]=add_noise(Ideal_H[:,0:1,0::2,...], SNR=snr)*X0[:,0:1,0::2,...].unsqueeze(-2) #直接点乘不正确
        Received_Y[:,0:1,0::2,...]=cf_mul(Noisy_H[:,0:1,0::2,...], Trans_X[:,0:1,0::2,...].unsqueeze(-2))
        # Received_Y[:,0:1,0::2,...]=cf_mul(add_noise(Ideal_H[:,0:1,0::2,...], SNR=snr), Trans_X[:,0:1,0::2,...].unsqueeze(-2))
    
    Hls=torch.zeros_like(Received_Y)
    # Hls[:,0:1,0::2,...]=Received_Y[:,0:1,0::2,...]/X0[:,0:1,0::2,...].unsqueeze(-2) #直接点乘不正确
    Hls[:,0:1,0::2,...]=cf_div(Received_Y[:,0:1,0::2,...],Trans_X[:,0:1,0::2,...].unsqueeze(-2))
    # Hls[:,0:1,0::2,...]=cf_div(Received_Y[:,0:1,0::2,...],Trans_X_data.unsqueeze(-2))
    dataset_size=Received_Y.shape[0]
    sigma=torch.zeros(dataset_size,1,1)
    sigma[...,0] = torch.tensor(np.power(10,-snr/10))

    Received_Y=Received_Y.to(torch.float32)
    Hls=Hls.to(torch.float32)
    Ideal_H=Ideal_H.to(torch.float32)
    Ideal_X=Ideal_X.to(torch.float32)
    Trans_X=Trans_X.to(torch.float32)
    sigma=sigma.to(torch.float32)

    return Received_Y, Hls, Ideal_H, Ideal_X, Trans_X, sigma, dataset_size

    # Ideal_X = np.array(data0['Ideal_X'], dtype=np.float32)
    # Ideal_X = torch.from_numpy(Ideal_X)
    # Xid_ts = Ideal_X.to(torch.float32)
    # print('Ideal_X.shape=',Xid_ts.shape)
    # X0_ts = torch.from_numpy(np.array(data0['Transmit_X'], dtype=np.float32))

    # Ideal_X_new=torch.zeros_like(Xid_ts)
    # Ideal_X_new[Xid_ts==0]=  torch.sqrt(torch.tensor(2.0, device=torch.device('cpu')))/2.0 
    # Ideal_X_new[Xid_ts==1]= -torch.sqrt(torch.tensor(2.0, device=torch.device('cpu')))/2.0

    # noise=torch.randn(Hid_ts.shape)
    # noise=noise-torch.mean(noise) #  %均值为0，方差接近1

    # Yms_ts=add_noise(Hid_ts, SNR=noise_list)*Ideal_X_new.unsqueeze(-2) + noise*0.0001
    # print('Received_Y.shape=',Yms_ts.shape)
    # Yms_ts[:,0:1,0::2,...]=add_noise(Hid_ts[:,0:1,0::2,...], SNR=noise_list)*X0_ts[:,0:1,0::2,...].unsqueeze(-2)

    # Hls_ts=torch.zeros_like(Yms_ts)
    # Hls_ts[:,0:1,0::2,...]=Yms_ts[:,0:1,0::2,...]/X0_ts[:,0:1,0::2,...].unsqueeze(-2)
    # test_size=Yms_ts.shape[0]
    # sigma=torch.zeros(test_size,1,1)
    # sigma[...,0] = torch.tensor(np.power(10,-noise_list/10))
