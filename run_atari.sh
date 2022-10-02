#!/bin/bash
echo "Training Atari"
for S in 111 222 333 444 555
do
	for L in contrastive vic 
	do
		python train.py --dataset data/pong_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 250 --name pong --seed $S --ssl-loss $L     
	done
done

echo "Evaluation Atari"

for S in 111 222 333 444 555
do
	for L in contrastive vic 
	do
		for T in 1 5 10
		do
			python eval.py --dataset data/pong_eval.h5 --save-folder checkpoints/$S/pong_$L --num-steps $T
		done
	done
done

