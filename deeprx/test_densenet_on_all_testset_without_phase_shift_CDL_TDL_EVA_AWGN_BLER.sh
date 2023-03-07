# test model DenseNet-v5
ts_epoch=28
# module add anaconda; source activate mimo_cuda11

for symbol in 0
do 
	for subcarrier in 0
	do
		for mode in CDL TDL
		# for mode in CDL
		do
			# for snr in -8
			for snr in -8 -3
			do 
				for dp in  15 45 75 105 135 165
				# for dp in  15
				do
				echo 'doppler='$dp', snr='$snr', channel mode='$mode', symbol='$symbol', subcarrier='$subcarrier
				python -u main.py --phase test --epoch 100 --lr 1e-3 --tr_batch 300 --doppler 90 --save_epoch 100 --resume False --resume_epoch epoch19 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler StepLR --wH 0.0 --label_noise 0.01 --deepcnn 64 --log True --suffix May26-v1-1750 --print_net True --ts_doppler $dp --ts_snr $snr --test_epoch epoch$ts_epoch  --data_mode $mode --symbol $symbol --subcarrier $subcarrier

				done
			done
		done
	done
done



for symbol in 0
do 
	for subcarrier in 0
	do
		for mode in EVA
		do
			for snr in -8 -4
			do 
				for dp in  5 30 60 90 120 150
				do
				echo 'doppler='$dp', snr='$snr', channel mode='$mode', symbol='$symbol', subcarrier='$subcarrier
				python -u main.py --phase test --epoch 100 --lr 1e-3 --tr_batch 300 --doppler 90 --save_epoch 100 --resume False --resume_epoch epoch19 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler StepLR --wH 0.0 --label_noise 0.01 --deepcnn 64 --log True --suffix May26-v1-1750 --print_net True --ts_doppler $dp --ts_snr $snr --test_epoch epoch$ts_epoch  --data_mode $mode --symbol $symbol --subcarrier $subcarrier
				done
			done
		done
	done
done


for symbol in 0
do 
	for subcarrier in 0
	do
		for mode in AWGN
		do
			for snr in -10 -5 0 10 20 30
			# for snr in -8 -4
			do 
				for dp in  0
				do
				echo 'doppler='$dp', snr='$snr', channel mode='$mode', symbol='$symbol', subcarrier='$subcarrier
				python -u main.py --phase test --epoch 100 --lr 1e-3 --tr_batch 300 --doppler 90 --save_epoch 100 --resume False --resume_epoch epoch19 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler StepLR --wH 0.0 --label_noise 0.01 --deepcnn 64 --log True --suffix May26-v1-1750 --print_net True --ts_doppler $dp --ts_snr $snr --test_epoch epoch$ts_epoch  --data_mode $mode --symbol $symbol --subcarrier $subcarrier
				done
			done
		done
	done
done

