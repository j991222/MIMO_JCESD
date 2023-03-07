import argparse
import torch
import numpy as np
import time
from scipy.linalg import toeplitz
import math
import scipy.special as scipy

def str2bool(str):
    return True if str.lower() == 'true' else False

class get_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(description='MIMO')
        self.parser.add_argument('--log', default=False, help='write output to file rather than print into the screen')
        self.parser.add_argument('--phase', type=str, default='train', help='train, test')
        self.parser.add_argument('--resume_epoch', type=str, help='resume training from epoch ?')
        self.parser.add_argument('--test_epoch', type=str, help='test epoch')
        self.parser.add_argument('--resume',"--preprocess", type=str2bool, default='True', help='run prepare_data')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--parallel', default=False, help='idx of gpu used')
        # self.parser.add_argument('--resume', default=False, help='resume training')
        self.parser.add_argument('--suffix', type=str, default='MIMO', help='suffix_of_model name')
        self.parser.add_argument('--data_mode', type=str, default='EVA', help='data_mode: EVA, ETU, EPA')
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer: SGD, Adam, AdamW')

        # Training Parameters
        self.parser.add_argument('--epoch', type=int, default=50, help='# of fine_epoch ')
        self.parser.add_argument('--fcn_chan', type=int, default=32, help='# FCN inter channel ')
        self.parser.add_argument('--tr_batch', type=int, default=100, help='batch size')
        self.parser.add_argument('--ts_batch', type=int, default=100, help='batch size')

        self.parser.add_argument('--layers', type=int, default=24, help='net layers')
        self.parser.add_argument('--deep', type=int, default=17, help='one module deep')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', help='loss function criterion')
        
        self.parser.add_argument('--save_freq', type=int, default=5, help='the frequency of saving epoch')
        self.parser.add_argument('--shuffle', default=True, help='shuffle when training')

        self.parser.add_argument('--disp', type=int, default=10, help='display the result')
        self.parser.add_argument('--deepcnn', type=int, default=64, help='deep RX width')
        
        self.parser.add_argument('--noise', type=float, default=0.0, help='noise level')
        
        self.parser.add_argument('--doppler', type=int, default=90, help='doppler value')
        self.parser.add_argument('--ts_doppler', type=int, default=90, help='test doppler value')
        self.parser.add_argument('--dB', type=int, default=30, help='SNR')
        self.parser.add_argument('--ts_snr', type=int, default=30, help='test SNR')
        self.parser.add_argument('--save_epoch', type=int, default=20, help='save ckp frequency')
        self.parser.add_argument('--fcn1_chan', type=int, default=3, help='NN Depth')
        self.parser.add_argument('--fcn2_chan', type=int, default=3, help='NN Depth')
        self.parser.add_argument('--fcn_depth', type=int, default=3, help='NN Depth')
        self.parser.add_argument('--shuff', type=int, default=100, help='Shuffle the training data')
        self.parser.add_argument('--cew', type=float, default=1.0, help='CE loss weight')
        self.parser.add_argument('--wdk', type=float, default=1e-3, help='Weight decay')
        self.parser.add_argument('--wXloss', type=float, default=1e-3, help='Xloss coeff')
        self.parser.add_argument('--wBCE', type=float, default=1e-3, help='wBCE coeff')
        self.parser.add_argument('--wCE', type=float, default=1e-3, help='wCE coeff')
        self.parser.add_argument('--wH', type=float, default=1e-3, help='wH coeff')
        self.parser.add_argument('--wHX', type=float, default=1, help='wH coeff')

        self.parser.add_argument('--dpr', type=float, default=1e-3, help='Dropout ratio')
        self.parser.add_argument('--label_noise', type=float, default=0.1, help='Label noise')
        self.parser.add_argument('--data_noise', type=float, default=0.01, help='Label noise')
        self.parser.add_argument('--inter_H', nargs='*',default=[256,256], help='image size')
        self.parser.add_argument('--inter_X', nargs='*',default=[256,256], help='image size')
        self.parser.add_argument('--inter_X2', nargs='*',default=[256,256], help='image size')
        self.parser.add_argument('--laam', type=float, default=0.1, help='learning rate')
        self.parser.add_argument('--data_len', type=int, default=30, help='test SNR')
        self.parser.add_argument('--print_net', type=str2bool, default='True', help='print_net')
        self.parser.add_argument('--symbol', type=int, default=0, help='symbol shift, |f|<=1000')
        self.parser.add_argument('--subcarrier', type=int, default=0, help='subcarrier shift, |tau|<=72')

        self.parser.parse_args(namespace=self)

        self.info = self.suffix

        # Result saving locations
        self.img_dir = 'result/' + self.info + '/img/'
        self.ckp_dir = 'result/' + self.info + '/ckp/'

        if self.phase == 'train':
            self.tr_dir = 'train_data_path'  # training data directory

            self.vl_dir='validation_data_path'

            if self.resume == True:
                # resume_info = self.info
                self.resume_ckp_dir = self.ckp_dir+self.resume_epoch
                # self.resume_ckp_dir = 'result/tmp_ckp/MIMO_Nov25_v1013_epoch998'
                print('resume ckp path=',self.resume_ckp_dir)

        elif self.phase == 'test':
            self.test_info = self.info
            self.test_ckp_dir =self.ckp_dir+self.test_epoch
            self.test_verbose = True
            
