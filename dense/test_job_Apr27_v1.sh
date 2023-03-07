# test model Apr 24-v2
ts_epoch=131

for mode in CDL TDL
do
	for snr in -8 -3
	do 
		for dp in  15 45 75 105 135 165
		do
		echo 'doppler'$dp', snr='$snr', channel mode='$mode
		python -u main.py --phase test --epoch 200 --lr 1e-3 --tr_batch 300 --doppler 120 --save_epoch 100 --resume True --resume_epoch epoch80 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler CosineAnnealingLR --ts_doppler $dp --ts_snr $snr --test_epoch epoch$ts_epoch  --data_mode $mode --thd 0.5 --deepcnn 24 --log True --suffix Apr27-v1-1107 --print_net False --shuffle 20

		done
	done
done


for mode in EVA
do
	# for dp in  15 45 75 105 135 165
	for snr in -8 -4
	do 
		for dp in  5 30 60 90 120 150
		do
		echo 'doppler'$dp', snr='$snr', channel mode='$mode		

		python -u main.py --phase test --epoch 200 --lr 1e-3 --tr_batch 300 --doppler 120 --save_epoch 100 --resume True --resume_epoch epoch80 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler CosineAnnealingLR --ts_doppler $dp --ts_snr $snr --test_epoch epoch$ts_epoch  --data_mode $mode --thd 0.5 --deepcnn 24 --log True --suffix Apr27-v1-1107 --print_net False --shuffle 20
		done
	done
done


for mode in AWGN
do
	for dp in  0
	do 
		for snr in -10 -5 0 10 20 30
		do
		echo 'doppler'$dp', snr='$snr', channel mode='$mode		
		python -u main.py --phase test --epoch 200 --lr 1e-3 --tr_batch 300 --doppler 120 --save_epoch 100 --resume True --resume_epoch epoch80 --disp 500 --dpr 0.1 --wdk 0.0 --optimizer Adam --lr_scheduler CosineAnnealingLR --ts_doppler $dp --ts_snr $snr --test_epoch epoch$ts_epoch  --data_mode $mode --thd 0.5 --deepcnn 24 --log True --suffix Apr27-v1-1107 --print_net False --shuffle 20

		done
	done
done


