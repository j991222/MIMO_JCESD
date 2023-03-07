#!/bin/bash
#SBATCH -J train_model            # Job 名
#SBATCH -o ./log/train_DeepRX_May26-v1-1750.out # 输出, 目录./out必须存在，否则无法成功提交job. 也可删除此行由系统自动指定.
#SBATCH --qos=short       # qos(quality of service): normal or debug, 对应不同优先级及最大可用时长.
#SBATCH -p V100       # 指定 partition: geforce,V100,etc. RTX3090
#SBATCH --nodelist=gpu00   # 指定属于上述partition的特定节点.也可删除这一行,由系统自动分配.
#SBATCH --cpus-per-task=8 # 申请 cpu processor 数; 可用内存与申请 cpu processor 数成正比.
#SBATCH --mem=60G          # 申请10G内存
#SBATCH --gres=gpu:1       # 申请 gpu 数
#SBATCH -N 1               # 申请节点数,一般为1
#SBATCH -t 2-00:00:00        # 申请 Job 运行时长0小时5分钟0秒,若要申请一天时间以上,如申请1天,书写格式为#SBATCH -t 1-00:00:00

module add anaconda; source activate mimo_cuda11

# python -u main.py --phase train --epoch 100 --lr 1e-3 --tr_batch 300 --doppler 90 --save_epoch 100 --resume False --resume_epoch epoch19 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler StepLR --ts_doppler 90 --ts_snr 0 --test_epoch epoch2 --wH 0.0 --label_noise 0.01 --deepcnn 64 --log True --suffix May25-v1-2100 --print_net True
python -u main.py --phase train --epoch 100 --lr 1e-3 --tr_batch 300 --doppler 90 --save_epoch 100 --resume False --resume_epoch epoch19 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler StepLR --ts_doppler 90 --ts_snr 0 --test_epoch epoch2 --wH 0.0 --label_noise 0.01 --deepcnn 64 --log True --suffix May26-v1-1750 --print_net True


