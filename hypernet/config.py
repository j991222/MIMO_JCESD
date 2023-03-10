import argparse
import torch
import numpy as np
import time
from scipy.linalg import toeplitz
import math
import scipy.special as scipy


class get_config():
    def __init__(self):
        # Parse from command line
        self.parser = argparse.ArgumentParser(description='MIMO')
        self.parser.add_argument('--log', default=False, help='write output to file rather than print into the screen')
        self.parser.add_argument('--phase', type=str, default='train', help='train, test')
        self.parser.add_argument('--gpu_idx', type=int, default=0, help='idx of gpu')
        self.parser.add_argument('--parallel', default=False, help='idx of gpu used')
        self.parser.add_argument('--resume', default=False, help='resume training')
        self.parser.add_argument('--suffix', type=str, default='MIMO', help='suffix_of_model name')


        # Training Parameters
        self.parser.add_argument('--epoch', type=int, default=50, help='# of fine_epoch ')
        self.parser.add_argument('--tr_batch', type=int, default=100, help='batch size')
        self.parser.add_argument('--ts_batch', type=int, default=100, help='batch size')

        self.parser.add_argument('--layers', type=int, default=24, help='net layers')
        self.parser.add_argument('--deep', type=int, default=17, help='one module deep')
        self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
        self.parser.add_argument('--save_freq', type=int, default=5, help='the frequency of saving epoch')
        self.parser.add_argument('--shuffle', default=True, help='shuffle when training')


        self.parser.add_argument('--disp_freq', type=int, default=10, help='display the result')
        
        self.parser.add_argument('--noise', type=float, default=0.0, help='noise level')
        
        self.parser.add_argument('--doppler', type=int, default=90, help='doppler value')
        

        self.parser.parse_args(namespace=self)









        self.info = self.suffix

        # Result saving locations
        self.img_dir = 'result/' + self.info + '/img/'
        self.ckp_dir = 'result/' + self.info + '/ckp/'

        if self.phase == 'train':
            self.tr_dir = 'train_data_path'  # training data directory

            self.vl_dir='validation_data_path'

            if self.resume == True:
                resume_info = 'MIMO'
                self.test_ckp_dir = 'result/' + resume_info + '/ckp/epoch454'

        elif self.phase == 'test':
            self.test_info = self.info
            self.test_ckp_dir =self.ckp_dir+self.test_epoch
            self.test_verbose = True
            
