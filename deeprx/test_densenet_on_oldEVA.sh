# test model DeepRX-HM
ts_epoch=28

# module add anaconda; source activate mimo_cuda11


for ts_epoch in 28
do
	for dp in 5 30 60 90 120 150
	# for dp in 5
	do 
		for snr in 20 30
		# for snr in -5 0 10 20 30
		do
			for mode in old_EVA
			do
				for subcarrier in 0
				do 
					for symbol in 0
					do
						echo 'test_epoch='$ts_epoch', doppler='$dp', snr='$snr', channel mode='$mode', symbol='$symbol', subcarrier='$subcarrier
						python -u main.py --phase test --epoch 100 --lr 1e-3 --tr_batch 200 --doppler 90 --save_epoch 100 --resume False --resume_epoch epoch19 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler StepLR --wH 0.0 --label_noise 0.01 --deepcnn 64 --log True --suffix May26-v1-1750 --print_net True --ts_doppler $dp --ts_snr $snr --test_epoch epoch$ts_epoch  --data_mode $mode --symbol $symbol --subcarrier $subcarrier
					done
				done
			done
		done
	done
done 


