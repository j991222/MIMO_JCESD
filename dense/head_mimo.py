import os
import sys
import time

from model import *
from trainer import *
from config import *


import numpy as np
import scipy.io as sc 
from scipy.linalg import toeplitz
import math
import scipy.special as scipy

import pickle

def log(args):
    ''' Folder settings when saving training results'''
    if not os.path.exists('result') and args.phase == 'train':
        os.makedirs('result')
    if not os.path.exists('log'):
        os.makedirs('log')
    if not os.path.exists('result/' + args.info) and args.phase == 'train':
        os.mkdir('result/' + args.info)
    if not os.path.exists('result/' + args.info + '/ckp') and args.phase == 'train':
        os.mkdir('result/' + args.info + '/ckp')

    if args.phase == 'train' and args.log == True:
        sys.stdout = open('result/' + args.info + '/' + 'file.txt', 'a')
        sys.stderr = open('result/' + args.info + '/' + 'file.txt', 'a')
    os.makedirs('log', exist_ok=True)


    print('[*] Info:', time.ctime())
    print('[*] Info:', os.path.basename(__file__))

    if args.phase == 'train' and args.log == True and args.resume == False:
        from shutil import copyfile
        copyfile(os.path.basename(__file__), 'result/' + args.info + '/scripts/' + os.path.basename(__file__))
        copyfile('config.py', 'result/' + args.info + '/scripts/config.py')
        copyfile('head.py', 'result/' + args.info + '/scripts/head.py')
        copyfile('main.py', 'result/' + args.info + '/scripts/main.py')
        copytree('./model/', 'result/' + args.info + '/scripts/model')
        copytree('./utils/', 'result/' + args.info + '/scripts/utils')
        copytree('./trainer/', 'result/' + args.info + '/scripts/trainer')

    #sys.stdout = Unbuffered(sys.stdout)
    #torch.cuda.set_device(args.gpu_idx)

    from torch import multiprocessing
    multiprocessing.set_sharing_strategy('file_system')

